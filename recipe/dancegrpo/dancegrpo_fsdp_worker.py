# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
import time
import warnings
from typing import Any, Dict, List, Union

import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import ProfilerConfig, WorkerProfiler, WorkerProfilerExtension, log_gpu_memory_usage, simple_timer
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs, copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker, create_device_mesh, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from PIL import Image
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

            
class DiffusionActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, model_deployment=None):
        super().__init__(config,role)
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .dp_actor import DiffusionDataParallelPPOActor as DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        # fhd compile vae
        self.rollout.vae_module.model.decoder = torch.compile(
            self.rollout.vae_module.model.decoder, 
            mode="max-autotune-no-cudagraphs", 
            # fullgraph=True, 
            # dynamic=True if self.is_hip() else None
        )
        
        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())
        timing_generate = {}
        with self.rollout_ulysses_sharding_manager:
            with self.rollout_sharding_manager:
                log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                prompts = self.rollout_ulysses_sharding_manager.preprocess_data(prompts)
                with simple_timer("generate_sequences", timing_generate):
                    output = self.rollout.generate_sequences(prompts=prompts)
                    
                prompts = self.rollout_sharding_manager.postprocess_data(prompts)
                log_gpu_memory_usage("After rollout generation", logger=logger)
        return output
 
class QwenRewardModelWorker(RewardModelWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    Use vllm based Qwen model as the reward model.
    """
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        initialize the reward model and sampling_params
        """
        # This is used to import external_lib into the huggingface systems
        # import_external_libs(self.config.model.get("external_lib", None))
        self.reward_rollout, self.reward_rollout_sharding_manager = self._build_reward_rollout()
        from vllm import SamplingParams
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.90, max_tokens = 128)
        print(f"成功初始化Qwen模型...")
        print("="*40)
        # exit(0)
        # TODO
        # self.image_processor = VaeImageProcessor(16)
        return

    def _build_reward_rollout(self,trust_remote_code=False):
        device_name = get_device_name()

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        
        from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
        from verl.workers.sharding_manager.reward_qwen import RewardVLLMManager
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
       
        # lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
        lora_kwargs = {}
        if vllm_mode == "customized":
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            
        elif vllm_mode == "spmd":
            # from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
            lora_kwargs={}
            from transformers import AutoConfig
            actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
            )
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            vllm_rollout_cls = vLLMRollout
            rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=tokenizer, model_hf_config=actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
        else:
            raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
        
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        full_params = torch.distributed.get_world_size() == 1
        #TODO
        rollout_sharding_manager = RewardVLLMManager(
            # module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            # model_config=self.actor_model_config,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            # offload_param=self._is_offload_param,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.get("layered_summon", False),
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)
        
        return rollout, rollout_sharding_manager
        
    
    def _create_simple_prompt(self) -> str:
        """创建结构化视频质量评估提示词（界面风格）"""
        return """请你作为一个专业视频质量评估助手，参考以下评分标准和格式，对给定的视频进行多维度质量评估。请严格按照输出格式，以客观、公正、结构化的方式打分。

                评估维度（每项满分100分）：
                1. 视觉审美（Aesthetics）：
                - 参考项：构图是否合理、光影运用是否自然、色彩搭配是否和谐、整体画面是否具有美感。
                - 高分标准：画面构图精妙、光影自然、色彩生动，具备艺术性。
                - 扣分项：画面凌乱、光照极端或失衡、颜色搭配不当或灰暗。

                2. 局部变形（Distortion）：
                - 参考项：人物或物体是否出现异常形态、肢体是否扭曲、是否有结构性突变或失真、是否突然消失。
                - 高分标准：视频中不存在明显变形，物体结构自然、稳定。
                - 扣分项：出现严重扭曲、肢体不合理、局部区域断裂或消失。

                3. 视觉伪影与不一致（Artifacts/Inconsistency）：
                - 参考项：是否存在突变区域、马赛克、色块、条纹、边缘断裂、纹理模糊等问题。
                - 高分标准：无明显视觉瑕疵，画面一致性强。
                - 扣分项：出现视觉伪影或明显瑕疵，视觉体验受到影响。

                4. 清晰度（Sharpness）：
                - 参考项：细节呈现的清晰度，边缘锐利程度，物体是否具备较高的辨识度。
                - 高分标准：画面细节丰富、边缘清晰锐利。
                - 扣分项：整体模糊、边缘不清晰、细节缺失。

                5. 视觉一致性（Consistency）：
                - 参考项：视频内容在时间上的连贯性，是否存在跳帧、镜头突变或画面不稳定等问题。
                - 高分标准：过渡自然，时间逻辑连贯，画面稳定。
                - 扣分项：镜头跳跃明显、物体突然改变状态、画面抖动。

                评分规则：
                - 每个维度评分在 0 ~ 100 范围内，越好越高分。
                - 合计为五项得分的算术平均，保留整数。
                - 对于某项严重失真或效果极差（如严重模糊、强伪影等），请大胆给出低分（例如低于30分）。
                - 每个视频的打分应充分拉开差距，避免视频之间出现“同分”或“几乎同分”情况。
                - 请确保不同维度之间的评分不互相矛盾，确保评分具有可比性与区分度。

                输出格式（严格遵守）：
                dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

                风格要求：
                - 禁止输出解释性文字或分析过程。
                - 禁止使用“我认为”、“可能”、“大致”等模糊词语。
                - 输出必须严格按照上述格式，一次性返回评估结果。

                请严格按照输出格式要求，输出且只输出输出格式的内容。请依照以上标准、逻辑和格式，对视频进行结构化质量评估。
                """
        
    def _generate_chat_batch_prompts(self,batch_path, max_pixels = 360*420, fps = 1.0) -> List:
        """
        generate prompts for a batch of paths.
        Args:
            - batch_path: a List[str] item that consists all the paths of videos in the batch.
            - max_pixels: default to 360*420, int
            - fps: default to 1.0, float
        Returns:
            - A List[List[Dict[str,Any]]] item that each element is a List satisfying the conversation format of llm.chat method.
        """
        messages=[]
        prompt = self._create_simple_prompt()
        total_time=0
        logger.info(f"Starting generating batch of prompts ...")
        VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        for file_path in batch_path:
            # if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
                start_time=time.time()
                video_path = str(file_path).replace("./","/gemini/space/ljm/Dancegrpo/")
                # print(f"视频地址是file://{video_path}")
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url":f"file://{video_path}"},
                                "max_pixels": max_pixels,
                                "fps": fps,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages.append(message)
                processing_time=time.time()-start_time
                total_time+=processing_time
        return messages
    
    def _generate_batch_prompts(self,batch_id) -> List:
        """
        generate prompts for a batch of paths.
        Args:
            - batch_id: a List[str] item that consists all the paths of videos in the batch.
            - max_pixels: default to 360*420, int
            - fps: default to 1.0, float
        Returns:
            - A List[List[Dict[str,Any]]] item that each element is a List satisfying the conversation format of llm.chat method.
        """
        messages=[]
        simple_prompt = self._create_simple_prompt()
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{simple_prompt}<|im_end|>\n<|im_start|>assistant\n"
        logger.info(f"Starting generating batch of prompts ...")
        for video_id in batch_id:
                message = [
                    {
                        "prompt": prompt,
                        'multi_modal_data': {'video':[video_id]},
                    }
                ]
                messages.append(message)   
        return messages
                
    def _parse_simple_evaluation(self, output_text: str) -> Dict[str, Any]:
        """
        extract the score in the output_text. Aligning with new prompts format.
        Args:
            - output_text: the output text in "str" form from llm rollout
        Returns:
            - A dict that includes keys:
                - overall_score: the score extract from the output text, float
                - dimensionn_scores: dimension scores, Dict[str,float]
                - summery: output_text[:500], str
                - raw_output: output_text, str
        """
        # 针对新格式的分数提取模式
        score_patterns = [
                r'合计[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 合计：80分
                r'综合得分[：:]\s*(\d+(?:\.\d+)?)\s*分',      # 综合得分：80分
                r'总分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 总分：80分
                r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分.*?合计[：:]\s*(\d+(?:\.\d+)?)\s*分',  # 提取合计分数
                r'最终[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 最终：80分
                r'评分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 评分：80分
                r'质量评分[：:]\s*(\d+(?:\.\d+)?)',          # 质量评分：80
                r'分数[：:]\s*(\d+(?:\.\d+)?)',              # 分数：80
                r'(\d+(?:\.\d+)?)\s*分',                     # 80分
                r'(\d+(?:\.\d+)?)/100',                      # 80/100
                r'(\d+(?:\.\d+)?)%',                         # 80%
            ]
            
        score = 50.0  # 默认分数
        for pattern in score_patterns:
            match = re.search(pattern, output_text)
            if match:
                # 对于有多个捕获组的模式，取最后一个（合计分数）
                if len(match.groups()) > 1:
                    found_score = float(match.group(2))  # 取合计分数
                else:
                    found_score = float(match.group(1))
                    
                if found_score > 100:
                    found_score = min(found_score, 100)
                score = found_score
                logger.info(f"Found score: {score} using pattern: {pattern}")
                break
            
        # 如果没找到分数，记录日志
        if score == 50.0:
            logger.warning(f"No score found in output, using default 50.0. Output: {output_text[:200]}...")
            
        # 尝试提取各个维度的分数
        dimension_scores = {}
        dim_patterns = [
                (r'dim1[：:]\s*(\d+(?:\.\d+)?)\s*分', 'visual_artifacts'),
                (r'dim2[：:]\s*(\d+(?:\.\d+)?)\s*分', 'local_deformation'),
                (r'dim3[：:]\s*(\d+(?:\.\d+)?)\s*分', 'noise_quality'),
                (r'dim4[：:]\s*(\d+(?:\.\d+)?)\s*分', 'clarity_sharpness'),
                (r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分', 'color_accuracy'),
            ]
            
        for pattern, dim_name in dim_patterns:
            match = re.search(pattern, output_text)
            if match:
                dimension_scores[dim_name] = float(match.group(1))
            
        result = {
                "overall_score": score,
                "summary": output_text[:500] + "..." if len(output_text) > 500 else output_text,
                "raw_output": output_text
            }
            
        # 如果提取到了维度分数，也加入结果
        if dimension_scores:
            result["dimension_scores"] = dimension_scores
            logger.info(f"Extracted dimension scores: {dimension_scores}")
            
        return result 
    
    def _get_batch_reward(self,batch_output: List) -> List:
        """
        extract rewards from a batch of output. Using self._parse_simple_evaluation method
        Args:
            - batch_output: the batch of output in the form of list and need to be extract rewards.
        Returns:
            - A list that includes all the rewards of the batch_output.
        """
        results = []
        for single_prompt_output in batch_output:
            for response in single_prompt_output:
                output = response.outputs[0]
                output_text = output.text
                logger.info(f"Generated text length: {len(output_text)}")
                logger.info(f"Generated text preview: {output_text[:200]}...")
                
            
                # 解析结果
                result = self._parse_simple_evaluation(output_text)
                overall_score = result.get("overall_score","N/A")
                # processing_time = result.get("processing_time", 0)
        
                print(f"🎯 综合质量分数: {overall_score}/100")
                # print(f"⏱️ 处理时间: {processing_time:.2f} 秒")
                results.append(overall_score)  #仅保留overallscore
        return results


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
        datas = data.pop(
                batch_keys=['video_frames'],
                non_tensor_batch_keys=["caption","video_ids"],
            )
            
        datas = datas.to(get_device_id())
        #TODO
        with self.reward_rollout_sharding_manager:
            # Support all hardwares
            
            datas= self.reward_rollout_sharding_manager.preprocess_data(datas)
            
            
            print("开始用llm计算得分：")
            # video_paths = datas.non_tensor_batch['video_paths']
            video_ids = datas.non_tensor_batch['video_ids']  # ids_batch
            
            import numpy as np
            # batch_paths = np.array_split(video_paths, datas.batch.batch_size[0])
            batch_ids = np.array_split(video_ids, datas.batch.batch_size[0])
            # batch_paths = [list(x) for x in  batch_paths]
            video_ids = [list(x) for x in video_ids]
            
            all_rewards = []  
            # 每个batch_path是一批video_paths的列表
            # batch_paths是很多批路径列表构成的列表
            # log_gpu_memory_usage("After entering reward rollout sharding manager", logger=logger)
            for batch_id in batch_ids:
                batch_message = self._generate_batch_prompts(batch_id)
                batch_output = []
                for message in batch_message:
                    output = self.reward_rollout.inference_engine.generate(message,
                                                                           sampling_params = self.sampling_params)
                    # with open('/gemini/space/ljm/Dancegrpo/videos/output.txt', 'a', encoding='utf-8') as f:
                    #     f.write(f"rank {dist.get_rank()} \n output内容是{output[0].outputs}\n\n")
               
            # for batch_path in batch_paths:
            #     batch_message=self._generate_chat_batch_prompts(batch_path)
            #     batch_output = []
            #     for message in batch_message:
            #         output = self.reward_rollout.inference_engine.chat(message, 
            #                                                            sampling_params = self.sampling_params)
                    
                    batch_output.append(output)
                    
                batch_reward = self._get_batch_reward(batch_output)
                
                all_rewards = all_rewards + batch_reward
                # print(type(all_rewards[0]),all_rewards[0])
                # print("\n")
        
            all_rewards = torch.tensor(all_rewards)
            batch = TensorDict(
                {
                    "rewards": all_rewards,
                },
                batch_size = len(batch_ids)
            )
            
            non_tensor_batch = data.non_tensor_batch
            batch_reward = DataProto(batch=batch, non_tensor_batch = non_tensor_batch)
            batch_reward = self.reward_rollout_sharding_manager.postprocess_data(batch_reward)
            # with open('/gemini/space/ljm/Dancegrpo/videos/my_file.txt', 'a', encoding='utf-8') as f:
            #     f.write(f"rank {dist.get_rank()} \n batch['rewards'],内容是{batch_reward.batch['rewards']}\n\n")
                
        return batch_reward
        
    
# TODO(sgm): we may need to extract it to dp_reward_model.py
class DiffusionRewardModelWorker(RewardModelWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module, self.preprocess_val,self.tokenizer = self._build_model(config=self.config)
        
        #TODO
        self.image_processor = VaeImageProcessor(16)

    def _build_model(self, config):
        # the following line is necessary

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from typing import Union

        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map, root_path

        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                self.config.model.path,
                precision='amp',
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        reward_module = model_dict['model']
        preprocess_val = model_dict['preprocess_val']

        checkpoint = torch.load(self.config.model.path)
        reward_module.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        
        return reward_module, preprocess_val,processor
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        
        # Support all hardwares
        data=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        decoded_image=data.batch['video_frames']
        decoded_images = decoded_image.chunk(data.batch.batch_size[0], dim=0)
        decoded_images = [x.squeeze(0) for x in decoded_images]
        caption=data.non_tensor_batch['caption']
        import numpy as np
        batch_caption = np.array_split(caption, data.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        all_rewards = []  
        self.reward_module.to(device=get_device_id())
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                frame = decoded_images[index][:, 0, :, :]  # (C, H, W)
                # 转换为PIL图像
                frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                frame_np = (frame_np * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                    
                image = self.preprocess_val(frame_pil).unsqueeze(0).to(device=get_device_id(), non_blocking=True)
                # Process the prompt
                text = self.tokenizer([batch_caption[index]]).to(device=get_device_id(), non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    outputs = self.reward_module(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards.append(hps_score.unsqueeze(0))

        all_rewards = torch.cat(all_rewards, dim=0)

        # all_rewards=all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.reward_module.to(torch.device('cpu'))
        
        return DataProto(batch=batch)


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.rollout.sleep()
        # return something to block the caller
        return True
