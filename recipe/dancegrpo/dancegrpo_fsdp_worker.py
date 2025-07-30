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
import warnings
import re
import numpy as np

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.distributed.device_mesh import init_device_mesh
from diffusers.image_processor import VaeImageProcessor

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_init_weight_context_manager,
    layered_summon_lora_params
)
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy, ActorRolloutRefWorker,RewardModelWorker
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.debug import ProfilerConfig, WorkerProfiler, WorkerProfilerExtension, log_gpu_memory_usage, simple_timer

from omegaconf import DictConfig, open_dict
from typing import Union
from verl.utils.fs import copy_to_local
from tensordict import TensorDict

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

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
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)
        return output

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
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification, GPT2Config

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

        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map

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

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
        # Support all hardwares
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        decoded_image=datas.batch['video_frames']
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        decoded_images = [x.squeeze(0) for x in decoded_images]
        caption=datas.non_tensor_batch['caption']
        import numpy as np
        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        all_rewards = []  
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                image_path = self.image_processor.postprocess(decoded_images[index])
                image = self.preprocess_val(image_path[0]).unsqueeze(0).to(device=get_device_id(), non_blocking=True)
                # Process the prompt
                text = self.tokenizer([batch_caption[index]]).to(device=get_device_id(), non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    self.reward_module.to(device=get_device_id())
                    outputs = self.reward_module(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards.append(hps_score)

        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards=all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.reward_module.to(torch.device('cpu'))

        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

# Helper function to compute position ids with attention mask
def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC, antialias=False),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    
# AestheticRewardModelWorker is a worker that computes aesthetic scores for images.
class AestheticRewardModelWorker(RewardModelWorker):
    """
    RewardModelWorker for aesthetic score evaluation using CLIP + linear regression.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.clip_model_path = "/nvfile-heatstorage/liangyzh/code/arena_models/ViT-L-14.pt"
        self.aes_model_path = "/nvfile-heatstorage/liangyzh/code/arena_models/sa_0_4_vit_l_14_linear.pth"
        import_external_libs(self.config.model.get("external_lib", None))
        
        self._build_model(config=self.config) 
    
    def _load_aesthetic_model(self, cache_folder):
        path_to_model = cache_folder
        m = nn.Linear(768, 1)
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m
    
    def _build_model(self, config):
        
        from verl.models.offline_clip import create_offline_clip_model
        self.clip_mode = create_offline_clip_model(self.clip_model_path, "cpu")
        self.aesthetic_model = self._load_aesthetic_model(self.aes_model_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # 读取数据，但是不删除
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        # # (B, C, H, W)
        # decoded_image = datas.batch['video_frames']
        # # List of [(1, C, H, W),...,...]
        # decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # # List of [(C, H, W),...,...]
        # decoded_images = [x.squeeze(0) for x in decoded_images]
        
        # (B, C, Frame, H, W)
        decoded_image = datas.batch['video_frames']
        # List of [(1, C, Frame, H, W),...,...]
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # List of [(C, Frame, H, W),...,...]
        decoded_images = [x.squeeze(0) for x in decoded_images]
        # List of [(Frame, C, H, W),...,...]
        decoded_images = [x.permute(1, 0, 2, 3) for x in decoded_images]
        caption = datas.non_tensor_batch['caption']       

        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        
        transform = clip_transform(224)
        
        all_rewards = []
        print(f"aes batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                transformed = torch.stack([transform(image) for image in decoded_images[index]])
                transformed = transformed.to(device=get_device_id()).to(device=get_device_id())
                self.clip_mode.to(device=get_device_id())
                self.aesthetic_model.to(device=get_device_id())
                # Compute the aesthetic score
                features = self.clip_mode.encode_image(transformed).float()
                features = F.normalize(features, dim=-1)
                score = self.aesthetic_model(features).squeeze(-1)
                
            mean_score = (score / 10).mean().item()
            print(f"aes_score value: {mean_score}")
            all_rewards.append(torch.tensor(mean_score, device=get_device_id()).unsqueeze(0))
            
        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "aes_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.clip_mode.to(torch.device('cpu'))
        self.aesthetic_model.to(torch.device('cpu'))
        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)        
   
import argparse
def dict_to_namespace(d):
    return argparse.Namespace(**d)
     
# RAFTRewardModelWorker is a worker that computes RAFT scores for images.    
class RAFTRewardModelWorker(RewardModelWorker):
    """
    RAFTRewardModelWorker is a worker that computes RAFT scores for images.
    It uses a pre-trained model to evaluate the RAFT quality of images.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, stride: int = 1):
        self.raft_model_path = "/nvfile-heatstorage/liangyzh/code/evalcrafter/EvalCrafter/checkpoints/RAFT/models/raft-things.pth"
        # self.raft_model = self._load_raft_model(self.raft_model_path)
        self.stride = stride
        self._build_model(config=self.config)
    
    def _build_model(self, config):

        args_dict = {
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
        args = dict_to_namespace(args_dict)
        
        from verl.models.raft.raft import RAFT
        model = RAFT(args)
        
        from collections import OrderedDict
        # 去除 "module." 前缀
        state_dict = torch.load(self.raft_model_path, map_location="cpu")
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        self.raft_model = model
        self.raft_model.eval()
        self.raft_model.args.mixed_precision = False

    def calculate_flow_score_from_tensor(self, frames):
        if len(frames) < 2:
            print("Not enough frames to compute optical flow.")
            return 0.0

        from verl.models.utils.utils import InputPadder

        optical_flows = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i].float().unsqueeze(0)
                image2 = frames[i + 1].float().unsqueeze(0)
                
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = self.raft_model(image1, image2, iters=20, test_mode=True)
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                mean_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_flow)
        res = float(np.mean(optical_flows))
        return res        
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # 读取数据，但是不删除
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        # # (B, C, H, W)
        # decoded_image = datas.batch['video_frames']
        # # List of [(1, C, H, W),...,...]
        # decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # # List of [(C, H, W),...,...]
        # # decoded_images = [x.squeeze(0) for x in decoded_images]

        # (B, C, Frame, H, W)
        decoded_image = datas.batch['video_frames']
        # List of [(1, C, Frame, H, W),...,...]
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # List of [(C, Frame, H, W),...,...]
        decoded_images = [x.squeeze(0) for x in decoded_images]
        # List of [(Frame, C, H, W),...,...]
        decoded_images = [x.permute(1, 0, 2, 3) for x in decoded_images]
        
        caption = datas.non_tensor_batch['caption']       
        
        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        
        self.raft_model.to(get_device_id())
        print(f"raft batch size: {len(batch_caption)}")
        all_rewards = []
        for index, batch_idx in enumerate(batch_indices):
            video = decoded_images[index][::self.stride]
            print(f"video shape: {video.shape[0]}")
            video = video.to(get_device_id())    
            flow_score = self.calculate_flow_score_from_tensor(video)
    
            print(f"flow_score value: {flow_score}")
            all_rewards.append(torch.tensor(flow_score, device=get_device_id()).unsqueeze(0))
            
        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "raft_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.raft_model.to(torch.device('cpu'))
        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

# VideoclipRewardModelWorker is a worker that computes video-clip scores for images. 
class VideoclipRewardModelWorker(RewardModelWorker):
    """
    VideoclipRewardModelWorker is a worker that computes video-clip scores for images.
    It uses a pre-trained model to evaluate the video-clip quality of images.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.videoclip_model_path = "/nvfile-heatstorage/liangyzh/code/arena_models/VideoCLIP-XL.bin"
        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        import_external_libs(self.config.model.get("external_lib", None))
        
        self._build_model(config=self.config) 
        
    def _build_model(self, config):
        from verl.models.VideoCLIP_XL.modeling import VideoCLIP_XL
        # 需要适配videoCLIP_XL中vision_model frame 配置
        self.videoclip_model = VideoCLIP_XL()
        state_dict = torch.load(self.videoclip_model_path, map_location="cpu")
        self.videoclip_model.load_state_dict(state_dict)
        # self.videoclip_model.to(get_device_id).eval()

    def _video_preprocessing(self, frames: torch.Tensor, fnum=8):
        """
        frames: torch.Tensor, shape [C, T, H, W], e.g. [3, 13, 720, 720]
        输出: torch.Tensor, shape [1, T, C, 224, 224]
        """
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [C, T, H, W] -> [T, H, W, C]

        total_frames = frames.shape[0]
        step = max(1, total_frames // fnum)
        sampled_frames = frames[::step][:fnum]  # [fnum, H, W, C]
        
        import cv2
        vid_tube = []
        
        for fr in sampled_frames:
            fr = fr[:, :, ::-1]  # BGR to RGB
            fr = cv2.resize(fr, (224, 224))
            fr = self._normalize(fr)
            fr = np.expand_dims(fr, axis=(0, 1)) # (1,1,H,W,C)
            vid_tube.append(fr)
        
        vid_tube = np.concatenate(vid_tube, axis=1)  # (1,T,H,W,C)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (1,T,C,H,W)
        print(f"vid_tube.shape: {vid_tube.shape}")
        return torch.from_numpy(vid_tube).float()
    
    def _normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std    


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # 读取数据，但是不删除
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        # (B, C, Frame, H, W)
        decoded_image = datas.batch['video_frames']
        # List of [(1, C, Frame, H, W),...,...]
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # List of [(C, Frame, H, W),...,...]
        decoded_images = [x.squeeze(0) for x in decoded_images]

        caption = datas.non_tensor_batch['caption']       

        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))

        from verl.models.VideoCLIP_XL.utils.text_encoder import text_encoder
        all_rewards = []
        self.videoclip_model.to(get_device_id()).eval()
        print(f"videoclip batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                video_inputs = self._video_preprocessing(decoded_images[index]).to(get_device_id())
                print(f"video_inputs.shape: {video_inputs.shape}")
                video_features = self.videoclip_model.vision_model.get_vid_features(video_inputs).float()
                video_features = F.normalize(video_features, dim=-1)
                
                text_inputs = text_encoder.tokenize([batch_caption[index]], truncate=True).to(get_device_id())
                text_features = self.videoclip_model.text_model.encode_text(text_inputs).float()
                text_features = F.normalize(text_features, dim=-1)

                similarity = (video_features @ text_features.T) * 100
                similarity = similarity.view(-1)  # 得到 torch.Size([1])
                print(f"similarity_score value: {similarity}")
            all_rewards.append(similarity)
        
        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "videoclip_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.videoclip_model.to(torch.device('cpu'))
        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

# Videophy CAPTION
CAPTION = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video follow the physical laws?\nAI: "

# VideophyRewardModelWorker is a worker that computes video-phy scores for images.
class VideophyRewardModelWorker(RewardModelWorker):
    """
    VideophyRewardModelWorker is a worker that computes video-phy scores for images.
    It uses a pre-trained model to evaluate the video-phy quality of images.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, media_tokens = ["<image>", "<|video|>"]):
        # self.checkpoint = "/nvfile-heatstorage/liangyzh/interns/zhangxin/videophy/arena-example/videocon_physics"
        self.checkpoint = "/root/videocon_physics"
        self.max_length = 256
        self._build_model(config=self.config)
        self.media_tokens = {k: -int(i + 1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {"<image>": 1 + 64, "<|video|>": 1 + 64}        
        
    def _build_model(self, config):

        from transformers.models.llama.tokenization_llama import LlamaTokenizer
        from verl.models.Videophy.mplug_owl_video import MplugOwlForConditionalGeneration
        from verl.models.Videophy.mplug_owl_video import (
            MplugOwlImageProcessor,
            MplugOwlProcessor,
        )
        from verl.models.Videophy.mplug_owl_video import MplugOwlConfig        
        self.tokenizer = LlamaTokenizer.from_pretrained(self.checkpoint)
        print("Model Loading")
        self.videophy_model = MplugOwlForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            config=MplugOwlConfig.from_pretrained(self.checkpoint)
        )
        print("Model Loaded")
        self.videophy_model.eval()
        image_processor = MplugOwlImageProcessor.from_pretrained(self.checkpoint)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)

    def _extract_text_token_from_conversation(self, max_length, index):  # index
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        # conversation = data["completion"]
        conversation = CAPTION

        # For Text only data
        if all(
            [
                media_token not in conversation
                for media_token in self.media_tokens.keys()
            ]
        ):
            pattern = "|".join(map(re.escape, ["AI: ", "\nHuman: "]))
            chunk_strs = re.split(f"({pattern})", conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = (
                        prompt_chunk
                        + self.tokenizer(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx - 1] == "AI: ":
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False
                        )["input_ids"]
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[: max_length - enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False
                        )["input_ids"]
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = "|".join(
                map(re.escape, list(self.media_tokens.keys()) + ["AI: ", "\nHuman: "])
            )
            chunk_strs = re.split(f"({pattern})", conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = (
                        prompt_chunk
                        + self.tokenizer(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [
                                self.media_tokens[chunk_str]
                            ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:
                        if chunk_strs[idx - 1] == "AI: ":
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False
                            )["input_ids"]
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[: max_length - enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False
                            )["input_ids"]
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * (
                max_length + 1 - enc_length
            )
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + 1, (
            index,
            prompt_length,
            enc_length,
            padding_length,
            max_length + 1,
        )
        assert len(label_chunk) == max_length + 1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length - 1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all(
            [
                media_token not in conversation
                for media_token in self.media_tokens.keys()
            ]
        ):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        return {
            "input_ids": enc_chunk,
            "prompt_length": prompt_length,
            "seq_length": enc_length,
            "non_padding_mask": non_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
        }

    def _get_input(self, batch):
        # TODO: batch_size > 1

        video = [data["video"] if data["video"] is not None else None for data in batch]
        if all([img is None for img in video]):
            video = None
        else:
            video = torch.cat([img for img in video if img is not None], dim=0)
        num_videos_per_sample = torch.LongTensor(
            [
                data["video"].size(0) if data["video"] is not None else 0
                for data in batch
            ]
        )
        num_images_per_sample = torch.LongTensor([0 for data in batch])

        text = torch.stack(
            [torch.LongTensor(data["text"]["input_ids"]) for data in batch], dim=0
        )
        non_padding_mask = torch.stack(
            [torch.LongTensor(data["text"]["non_padding_mask"]) for data in batch],
            dim=0,
        )
        non_media_mask = torch.stack(
            [torch.LongTensor(data["text"]["non_media_mask"]) for data in batch], dim=0
        )
        prompt_mask = torch.stack(
            [torch.LongTensor(data["text"]["prompt_mask"]) for data in batch], dim=0
        )
        # videopaths = [data["videopath"] for data in batch]
        captions = [data["caption"] for data in batch]
        output_batch = {
            "pixel_values": None,
            "video_pixel_values": video,
            "input_ids": text.long(),
            "labels": text.long().clone(),
            "num_images": num_images_per_sample.long(),
            "num_videos": num_videos_per_sample.long(),
            "non_padding_mask": non_padding_mask.long(),
            "non_media_mask": non_media_mask.long(),
            "prompt_mask": prompt_mask.long(),
            # "videopaths": videopaths,
            "captions": captions,
        }

        return output_batch

    def get_entail(self, logits, input_ids):
        softmax = nn.Softmax(dim=2)
        logits = softmax(logits)
        token_id_yes = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode("No", add_special_tokens=False)[0]
        entailment = []
        for j in range(len(logits)):
            for i in range(len(input_ids[j])):
                if (
                    input_ids[j][i] == self.tokenizer.pad_token_id
                ):  # pad token if the answer is not present
                    i = i - 1
                    break
                elif i == len(input_ids[j]) - 1:
                    break
            score = logits[j][i][token_id_yes] / (
                logits[j][i][token_id_yes] + logits[j][i][token_id_no]
            )
            entailment.append(score)
        entailment = torch.stack(entailment)
        return entailment

    def get_scores(self, inputs):
        with torch.no_grad():
            # for index, inputs in tqdm(enumerate(dataloader)):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(get_device_id())
                    # print(f'{k}: {v.shape}')
            # inputs["videophy_score"] = []
            print("compute videophy")
            outputs = self.videophy_model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )
            print("compute videophy done")
            logits = outputs["logits"]
            entail_scores = self.get_entail(logits, inputs["input_ids"])
            # print(len(entail_scores))
            # for m in range(len(entail_scores)):
            #     inputs["videophy_score"].append(entail_scores[m].item())
            # print(f"Batch {index} Done")
            assert len(entail_scores) == 1
        return entail_scores[0].item()
  
    def resize_video_frames(self, video_tensor, target_size=(224, 224)):
        """
        video_tensor: torch.Tensor, shape [B, C, T, H, W]
        return: torch.Tensor, shape [B, C, T, 224, 224]
        """
        B, C, T, H, W = video_tensor.shape
        # 把时间帧展平成 batch 维度
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        video_tensor = video_tensor.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        
        # 进行 resize
        resized = F.interpolate(video_tensor, size=target_size, mode='bilinear', align_corners=False)  # [B*T, C, 224, 224]
        
        # reshape 回视频形式
        resized = resized.reshape(B, T, C, *target_size)  # [B, T, C, 224, 224]
        resized = resized.permute(0, 2, 1, 3, 4)  # [B, C, T, 224, 224]
        return resized
  
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # 读取数据，但是不删除
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        # (B, C, Frame, H, W)
        decoded_image = datas.batch['video_frames']
        # [B, C, 224, 224]
        # Plan A
        print("start resize video frame")
        decoded_image = self.resize_video_frames(decoded_image, target_size=(224, 224)) 
        print("resize video frame done")
        # List of [(1, C, Frame, 224, 224),...,...]
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        print("decoded_image.chunk done")
        # List of [(C, Frame, 224, 224),...,...]
        # decoded_images = [x.squeeze(0) for x in decoded_images]

        caption = datas.non_tensor_batch['caption']       

        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        print("str(x.squeeze(0) done")
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        print("torch.chunk(torch.arange(len(batch_caption)) done")
        
        import time
        start_time = time.time()
        self.videophy_model.to(get_device_id())
        load_time = time.time() - start_time
        print(f"load model time: {load_time}s")
        
        all_rewards = []
        print(f"videophy batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                print("start _extract_text_token_from_conversation")
                text_input = self._extract_text_token_from_conversation(
                    self.max_length, index
                )
                print("done _extract_text_token_from_conversation")
                inputs = {
                    "video": decoded_images[index],
                    "text": text_input,
                    "caption": CAPTION,
                    # "video_path": "/nvfile-heatstorage/liangyzh/interns/zhangxin/videophy/arena-example/A_wooden_spoon_stirs_the_hot_soup_in_the_pot._1.mp4"
                }
                
                inputs = self._get_input([inputs])
                # print(f"inputs: {inputs}")
                score = self.get_scores(inputs)
                print(f"videophy_value: {score}")
            all_rewards.append(torch.tensor([score]))
        
        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "videophy_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        
        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
                
        
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
