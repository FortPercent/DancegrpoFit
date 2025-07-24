# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from .base import BaseRollout

from transformers import GenerationConfig
from wan.modules.vae import WanVAE
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
import math
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
from tensordict import TensorDict
import os

__all__ = ['DiffusionRollout']

class DiffusionRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        vae_dtype=torch.float16
        vae=WanVAE(
            vae_pth=os.path.join(self.config.model.vae_model_path),
            dtype=vae_dtype
        )
        self.vae_module=vae

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        num_frames = self.config.num_frames
        size = (self.config.w, self.config.h)
        sample_steps = self.config.sampling_steps

        sigma_schedule = torch.linspace(1, 0, self.config.sampling_steps + 1)

        def sd3_time_shift(shift, num_frames):
            return (shift * num_frames) / (1 + (shift - 1) * num_frames)

        sigma_schedule = sd3_time_shift(self.config.shift, sigma_schedule)
        
        def assert_eq(x, y, msg=None):
            assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

        context=prompts.batch['context']
        context_orig_lengths = prompts.batch['context_orig_lengths']
        caption=prompts.non_tensor_batch['caption']
        neg_context = prompts.batch['null_context']

        B = len(caption)
        
        # B = caption.shape[0]
        assert_eq(
            len(sigma_schedule),
            sample_steps + 1,
            "sigma_schedule must have length sample_steps + 1",
        )

        all_latents = []
        all_log_probs = []
        all_video_frames = []
        all_sigma_schedule = []
        # VAE参数
        vae_stride = [4, 8, 8]
        patch_size = [1, 2, 2]
        # 根据是否使用FP16选择数据类型

        batch_size=1
        batch_indices = torch.chunk(torch.arange(B), B // batch_size)
        latent_dtype = torch.float16 #TODO
        if self.config.init_same_noise:
            latent_shape = (
                16,
                (num_frames - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2]
            )
            input_latents = torch.randn(latent_shape, device=get_device_id(), dtype=latent_dtype)

        for index, batch_idx in enumerate(batch_indices):
            batch_captions = [caption[i] for i in batch_idx]
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            batch_neg_context = [neg_context[i].to(get_device_id()) for i in batch_idx]
            batch_context_orig_lengths = [context_orig_lengths[i] for i in batch_idx]

            for i in range(len(batch_contexts)):
                batch_contexts[i] = batch_contexts[i][:batch_context_orig_lengths[i]]
            
            if not self.config.init_same_noise:
                latent_shape = (
                    16,
                    (num_frames - 1) // vae_stride[0] + 1,
                    size[1] // vae_stride[1],
                    size[0] // vae_stride[2]
                )
                input_latents = torch.randn(latent_shape, device=get_device_id(), dtype=latent_dtype)

            seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
            )

            grpo_sample = True
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            
            with torch.no_grad():      
                _, final_latents, batch_latents, batch_log_probs = self.run_wan_sample_step(
                    [input_latents],
                    progress_bar,
                    sigma_schedule,
                    self.module,
                    batch_contexts,
                    batch_neg_context,
                    seq_len,
                    grpo_sample,
                )

            batch_latents = batch_latents.unsqueeze(0)
            batch_log_probs = batch_log_probs.unsqueeze(0)

            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)
            all_sigma_schedule.append(sigma_schedule.unsqueeze(0))    

            autocast_dtype = torch.float16 #TODO
            with torch.autocast("cuda", dtype=autocast_dtype):
                # 确保final_latents的数据类型正确
                final_latents_vae = final_latents.to(dtype=autocast_dtype)
                self.vae_module.model.to(get_device_id())
                decoded_videos = self.vae_module.decode([final_latents_vae])
                video_frames = decoded_videos[0]
                
                # 后处理
                video_frames = (video_frames + 1.0) / 2.0
                video_frames = torch.clamp(video_frames, 0, 1)
                video_frames = video_frames.unsqueeze(0)
            all_video_frames.append(video_frames)

        if len(all_latents) > 1:
            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_video_frames = torch.cat(all_video_frames, dim=0)
            all_sigma_schedule = torch.cat(all_sigma_schedule, dim=0)
        else:
            all_latents = all_latents[0]
            all_log_probs = all_log_probs[0]
            all_video_frames = all_video_frames[0]
            all_sigma_schedule = all_sigma_schedule[0]


        timestep_value = [int(sigma * 1000) for sigma in all_sigma_schedule[0].squeeze()][:self.config.sampling_steps]
        
        timestep_values = [timestep_value[:] for _ in range(B)]

        timesteps =  torch.tensor(timestep_values, device=get_device_id(), dtype=torch.long)
        
        timesteps = timesteps.unsqueeze(0).repeat(B, 1,1)

        latents=all_latents[:, :-1]
        next_latents=all_latents[:, 1:]
        batch = TensorDict(
            {
                "context_orig_lengths":context_orig_lengths,
                "contexts": context,
                "latents": latents,
                "next_latents": next_latents,
                "log_probs": all_log_probs,  # we will recompute old log prob with actor
                "video_frames": all_video_frames,
                "sigma_schedule": all_sigma_schedule,
                'timesteps':timesteps
            },
            batch_size=B
        )

        non_tensor_batch = prompts.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get('temperature', self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    # renormalize_logits=True,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)

    def run_wan_sample_step(
        self,
        latents,  # [(16, 7, 64, 64)]
        progress_bar, 
        sigma_schedule,  # 添加sigma_schedule
        transformer,
        context,
        neg_context,
        seq_len,
        grpo_sample,
    ):
        """WAN采样步骤，支持(C,T,H,W)格式输入"""
        if grpo_sample:
            all_latents = [latents[0]]  # 存储初始latent (16, 7, 64, 64)
            all_log_probs = []
            
            for i in progress_bar:
                B = len(context) if isinstance(context, list) else context.shape[0]
                
                # 确保设备一致
                device = latents[0].device
                
                # 使用sigma值计算timestep
                sigma = sigma_schedule[i]
                timestep_value = int(sigma * 1000)
                timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
                
                timestep_cond = timestep
                timestep_uncond = timestep
                transformer.eval()
                with torch.autocast("cuda", torch.float32):
                    # WAN模型输入：x是(C,T,H,W)格式的列表

                    #TODO!!!!
                    #管理!!!有hook!!!

                    transformer.to(device)
                    pred_cond = transformer(
                        x=latents,  # [(16, 7, 64, 64)]
                        t=timestep_cond,
                        context=context,
                        seq_len=seq_len
                    )
                    
                    # 处理模型输出
                    if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                        model_output_cond = pred_cond['rgb'][0]
                    elif isinstance(pred_cond, list):
                        model_output_cond = pred_cond[0]
                    else:
                        model_output_cond = pred_cond

                    # 为无条件预测准备输入
                    transformer.to(device)
                    pred_uncond = transformer(
                        x=latents,  # [(16, 7, 64, 64)]
                        t=timestep_uncond,
                        context=neg_context,
                        seq_len=seq_len
                    )

                    if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                        model_output_uncond = pred_uncond['rgb'][0]
                    elif isinstance(pred_uncond, list):
                        model_output_uncond = pred_uncond[0]
                    else:
                        model_output_uncond = pred_uncond
                        
                    del pred_cond, pred_uncond

                    # CFG组合
                    model_output = model_output_uncond + self.config.guide_scale * (model_output_cond - model_output_uncond)
                    del model_output_cond, model_output_uncond
                    torch.cuda.empty_cache()

                # WAN的SDE采样步骤
                next_latents, pred_original, log_prob = self.wan_step(
                    model_output, 
                    latents[0].to(torch.float32),  # (16, 7, 64, 64)
                    self.config.actor.eta, 
                    sigma_schedule,  # 传入sigma_schedule
                    i, 
                    prev_sample=None, 
                    grpo=True, 
                    sde_solver=True  # 启用SDE求解器
                )
                
                latents = [next_latents.to(torch.float32)]  # [(16, 7, 64, 64)]
                all_latents.append(latents[0])  # 存储 (16, 7, 64, 64)
                all_log_probs.append(log_prob)  # 存储 log概率
            
            final_latents = pred_original
                    # all the tp ranks should contain the same data here. data in all ranks are valid

            # 修正：WAN的all_latents维度是 (num_steps+1, 16, 7, 64, 64)
            all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
            all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
            
            # print(f"WAN after stack: all_latents={all_latents.shape}, all_log_probs={all_log_probs.shape}")
            # (9, 16, 7, 64, 64), (8,)
            
            return latents, final_latents, all_latents, all_log_probs

    def wan_step(
        self,
        model_output: torch.Tensor,  # 模型预测的flow
        latents: torch.Tensor,       # 当前时间步的潜在表示 (16, 7, 64, 64)
        eta: float,                  # 控制随机性强度
        sigmas: torch.Tensor,        # sigma调度序列 (类似FLUX)
        index: int,                  # 当前时间步索引  
        prev_sample: torch.Tensor,   # 前一步的样本（用于GRPO重计算）
        grpo: bool,                  # True时会得到logprob
        sde_solver: bool,            # 使用SDE求解器
    ):
        """WAN的Flow Matching采样步骤，转换为SDE求解器支持GRPO"""
        
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma  # sigma差分
        
        # 确定性更新部分
        prev_sample_mean = latents + dsigma * model_output
        
        # 预测的原始样本
        pred_original_sample = latents - sigma * model_output
        
        delta_t = sigma - sigmas[index + 1]  # 时间差分
        std_dev_t = eta * math.sqrt(abs(delta_t))  # 随机噪声的std
        
        if sde_solver:  # 使用SDE求解器（和FLUX相同）
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)  # 估计的得分
            log_term = -0.5 * eta**2 * score_estimate  # 对数项修正
            prev_sample_mean = prev_sample_mean + log_term * dsigma  # 修正的均值
        
        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            # 计算log概率
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
            )
            - math.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # 在除batch维度外的所有维度上求平均
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

