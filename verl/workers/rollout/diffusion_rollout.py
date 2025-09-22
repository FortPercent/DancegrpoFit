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
import math
import os

import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm.auto import tqdm
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.torch_functional import get_response_mask
from wan.modules.vae import WanVAE

from .base import BaseRollout
import logging

__all__ = ['DiffusionRollout']
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
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
        self._step_cache = {}

    def generate_sequences(self, prompts: DataProto) -> DataProto:

        context=prompts.batch['context']
        context_orig_lengths = prompts.batch['context_orig_lengths']
        # caption=prompts.non_tensor_batch['caption']
        neg_context = prompts.batch['null_context']
        sigma_schedule = prompts.batch["sigma_schedule"]
        input_latents = prompts.batch["input_latents"]
        latent_shape=input_latents[0].shape
        patch_size = [1, 2, 2]
        seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
            )
        
        B = prompts.batch.batch_size[0]
        

        all_latents = []
        all_log_probs = []
        all_video_frames = []
        all_video_ids = []

        batch_indices = torch.chunk(torch.arange(B), B // self.config.rollout.ulysses_sequence_parallel_size)
        
        grpo_sample = True
        # self.module.eval()
        
        self.vae_module.model.to(get_device_id())
        for index, batch_idx in enumerate(batch_indices):
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            # batch_captions = [caption[i] for i in batch_idx]
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            batch_neg_context = [neg_context[i].to(get_device_id()) for i in batch_idx]
            batch_context_orig_lengths = [context_orig_lengths[i] for i in batch_idx]
            batch_input_latents = [input_latents[i] for i in batch_idx]
            
            for i in range(len(batch_contexts)):
                batch_contexts[i] = batch_contexts[i][:batch_context_orig_lengths[i]]
            
            # ---- 打印信息 ----
            print(
                f"\n[Batch {index}] "
                f"indices={batch_idx.tolist()} "
            )
            print("-"*50)
            # with torch.no_grad(): 
            _, final_latents, batch_latents, batch_log_probs = self.run_wan_sample_step(
                batch_input_latents,
                progress_bar,
                sigma_schedule[0],
                self.module,
                batch_contexts,
                batch_neg_context,
                seq_len,
                grpo_sample,
            )

            all_latents.append(batch_latents.unsqueeze(0))
            all_log_probs.append(batch_log_probs.unsqueeze(0))
           
            
            # autocast_dtype = torch.float32 #TODO
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 确保final_latents的数据类型正确
                final_latents_vae = final_latents.to(dtype=torch.float32)

                # 记录开始时间
                decoded_videos = self.vae_module.decode([final_latents_vae])

                # 计算耗时
                video_frames = decoded_videos[0]
                # print(f"video_frames的形状是{video_frames.shape}")

                
                # 后处理
                video_frames = (video_frames + 1.0) / 2.0
                video_frames = torch.clamp(video_frames, 0, 1)
                
                
                # 确保video_frames是正确的格式 (C, T, H, W)
                if video_frames.dim() == 4:
                    # 调整帧率
                    fps=15
                    video_id = video_frames[:, ::fps, :, :]
                    C, T, H, W = video_frames.shape
                    # print(video_frames.shape)
                        
                    # 转换为numpy格式 (T, H, W, C)
                    video_id = video_id.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
                    # video_id = video_frames.cpu().numpy()
                    # print(f"video_id形状为{video_id.shape}")
                    import numpy as np
                    video_id = (video_id * 255).astype(np.uint8)
                        
                        # 如果是单通道，扩展为3通道
                    if C == 1:
                        video_id = id.repeat(video_id, 3, axis=-1)
                        
                all_video_ids.append(video_id)
                
                video_frames = video_frames.unsqueeze(0)
                
            all_video_frames.append(video_frames)
            torch.cuda.empty_cache()
        
        # 除了all_video_paths返回的都是一个张量
        if len(all_latents) > 1:
            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_video_frames = torch.cat(all_video_frames, dim=0)
        else:
            all_latents = all_latents[0]
            all_log_probs = all_log_probs[0]
            all_video_frames = all_video_frames[0]

        timestep_value = [int(sigma * 1000) for sigma in sigma_schedule[0].squeeze()][:self.config.sampling_steps]
        
        timestep_values = [timestep_value[:] for _ in range(B)]

        timesteps =  torch.tensor(timestep_values, device=get_device_id(), dtype=torch.long)
       
        latents=all_latents[:, :-1]
        next_latents=all_latents[:, 1:]

        batch = TensorDict(
            {
                "context_orig_lengths":context_orig_lengths,
                "contexts": context,
                "null_context":neg_context,
                "latents": latents,
                "next_latents": next_latents,
                "log_probs": all_log_probs,
                "video_frames": all_video_frames,
                "sigma_schedule": sigma_schedule,
                'timesteps':timesteps[:, :-1]
            },
            batch_size=B
        )

        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch['video_ids'] = np.array(all_video_ids)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    # 类里加一个缓存（也可以在 __init__ 里初始化）
# self._step_cache = {}

    def run_wan_sample_step(
        self,
        latents,          # [(16, 7, 64, 64)]  单个样本的起始 latent 放在 list 里
        progress_bar, 
        sigma_schedule,
        transformer,
        context,
        neg_context,
        seq_len,
        index,
    ):
        """WAN 采样步骤：index==0 先算并缓存指定步的 model_output；index>0 复用。"""
        if not hasattr(self, "_step_cache"):
            self._step_cache = {}

        # 这里定义你希望“复用”的步集合；如果只复用 i==0，就写 {0}
        reuse_steps = {0}

        # 第一个样本开始前，清空缓存，防止跨请求/跨 prompt 复用
        if index == 0:
            self._step_cache.clear()
            
        # 把 latents 统一成 list[tensor]
        if isinstance(latents, torch.Tensor):
            latents = [latents]
            
        all_latents = latents
        all_log_probs = []

        B = len(context) if isinstance(context, list) else context.shape[0]
        device = latents[0].device if isinstance(latents, list) else latents.device

        for i in progress_bar:
            # timestep
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)

            # ---- 1) 取/算 model_output ----
            use_cached = (index > 0) and (i in reuse_steps) and (i in self._step_cache)
            if use_cached:
                # 直接复用缓存的 CFG 后的 model_output
                model_output = self._step_cache[i]
            else:
                # 计算本步的 cond/uncond，然后做 CFG 组合
                # transformer.eval()
                with torch.autocast("cuda", torch.bfloat16):#, torch.inference_mode():
                    out_c = [transformer(
                        x=latents,              # [(C,T,H,W)]
                        t=timestep,
                        context=context,
                        seq_len=seq_len
                    )[0].detach()]
                    out_u = [transformer(
                        x=latents,
                        t=timestep,
                        context=neg_context,
                        seq_len=seq_len
                    )[0].detach()]
                    
                    # print(
                    #     f"out_c: {out_c}"
                    #     f"out_c: requires_grad={out_c[0].requires_grad}, "
                    #     f"grad_fn={out_c[0].grad_fn}, is_leaf={out_c[0].is_leaf}, "
                    #     f"dtype={out_c[0].dtype}, device={out_c[0].device}"
                    # )
                    # print(
                    #     f"out_u: {out_u}"
                    #     f"out_u: requires_grad={out_u[0].requires_grad}, "
                    #     f"grad_fn={out_u[0].grad_fn}, is_leaf={out_u[0].is_leaf}"
                    # )
                    # 规整输出为 tensor
                    def get_tensor(x):
                        if isinstance(x, dict) and 'rgb' in x:
                            return x['rgb'][0]
                        if isinstance(x, list) or isinstance(x, tuple):
                            return x[0]
                        return x
                    
                    model_output_cond = get_tensor(out_c)
                    model_output_uncond = get_tensor(out_u)
                    model_output = model_output_uncond + self.config.guide_scale * (
                        model_output_cond - model_output_uncond
                    )
                # 仅在首个样本且该步需要复用时，缓存 CFG 后的输出
                if (index == 0) and (i in reuse_steps):
                    # detach 避免把整个计算图塞进缓存；clone 避免后续就地改动
                    self._step_cache[i] = model_output.detach().clone()

            # ---- 2) 用（复用或现算的）model_output 走一步 SDE ----
            next_latents, pred_original, log_prob = self.wan_step(
                model_output, 
                latents[0].to(torch.float32),   # 当前样本的当前 latent
                self.config.actor.eta, 
                sigma_schedule,
                i,
                prev_sample=None,
                grpo=True,
                sde_solver=True
            )

            # 更新当前样本的 latent
            latents = [next_latents.to(torch.float32)]
            all_latents.append(latents[0])
            all_log_probs.append(log_prob)

        final_latents = pred_original
        all_latents = torch.stack(all_latents, dim=0)
        all_log_probs = torch.stack(all_log_probs, dim=0)
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

