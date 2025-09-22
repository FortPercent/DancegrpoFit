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

    def generate_sequences(self, prompts: DataProto) -> DataProto:

        context=prompts.batch['context']
        context_orig_lengths = prompts.batch['context_orig_lengths']
        caption=prompts.non_tensor_batch['caption']
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
        
        self.vae_module.model.to(get_device_id())
        for index, batch_idx in enumerate(batch_indices):
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            batch_captions = [caption[i] for i in batch_idx]
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            batch_neg_context = [neg_context[i].to(get_device_id()) for i in batch_idx]
            batch_context_orig_lengths = [context_orig_lengths[i] for i in batch_idx]
            batch_input_latents = [input_latents[i] for i in batch_idx]
            
            for i in range(len(batch_contexts)):
                batch_contexts[i] = batch_contexts[i][:batch_context_orig_lengths[i]]


            with torch.no_grad(): # fhd
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

            # _, final_latents, batch_latents, batch_log_probs = self.run_wan_sample_step(
            #     batch_input_latents,
            #     progress_bar,
            #     sigma_schedule[0],
            #     self.module,
            #     batch_contexts,
            #     batch_neg_context,
            #     seq_len,
            #     grpo_sample,
            # )

            all_latents.append(batch_latents.unsqueeze(0))
            all_log_probs.append(batch_log_probs.unsqueeze(0))
           
            
            autocast_dtype = torch.float16 #TODO
            with torch.autocast("cuda", dtype=autocast_dtype):
                # 确保final_latents的数据类型正确
                final_latents_vae = final_latents.to(dtype=autocast_dtype)
                
                import time

                # 记录开始时间
                start = time.time()
                decoded_videos = self.vae_module.decode([final_latents_vae])
                # 记录结束时间
                end = time.time()

                # 计算耗时
                logger.warning(f"vae decode {batch_idx}, {final_latents_vae.shape} 执行完毕，耗时: {end - start:.2f} 秒")
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
                    print(f"video_id形状为{video_id.shape}")
                    import numpy as np
                    video_id = (video_id * 255).astype(np.uint8)
                        
                        # 如果是单通道，扩展为3通道
                    if C == 1:
                        video_id = id.repeat(video_id, 3, axis=-1)
                # with open('/gemini/space/ljm/Dancegrpo/np.txt', 'a', encoding='utf-8') as f:
                #     f.write(f"video_np形状{video_id.shape}\n")
                #     f.write(f"{video_id}\n\n")
                all_video_ids.append(video_id)
                
                video_frames = video_frames.unsqueeze(0)
                
            all_video_frames.append(video_frames)
            # all_video_paths.append(video_path)  # 添加path
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
            all_latents = latents
            
            all_log_probs = []
            # i = 0
            import json

            hook_results2 = []
            hooks2 = []

            def get_shape(x):
                """获取 tensor 或嵌套结构的 shape，列表/元组递归处理"""
                if isinstance(x, torch.Tensor):
                    return list(x.shape)
                elif isinstance(x, (list, tuple)):
                    return [get_shape(i) for i in x]
                else:
                    return str(type(x))
                
            def compute_norm_tensor(x):
                """
                递归计算 tensor norm
                - 只对浮点 tensor 计算
                - meta tensor / int tensor 返回 None
                - list/tuple 递归处理
                """
                if isinstance(x, torch.Tensor):
                    if not x.is_meta:
                        return x.float().norm()  # 返回 tensor，延迟转 float
                    else:
                        return None
                elif isinstance(x, (list, tuple)):
                    return [compute_norm_tensor(i) for i in x]
                else:
                    return None

            def type_of(x):
                """递归获取输入输出的类型信息"""
                if isinstance(x, torch.Tensor):
                    return str(x.dtype)
                elif isinstance(x, (list, tuple)):
                    return [type_of(i) for i in x]
                else:
                    return str(type(x))
                
            def register_hooks2(module, prefix=""):
                for name, child in module.named_children():
                    layer_name = f"{prefix}.{name}" if prefix else name

                    def hook_fn(module, input, output, layer_name=layer_name):
                        # if layer_name == '_fsdp_wrapped_module.text_embedding.0':
                        #     logger.warning("---------------------------")
                        #     logger.warning(input)
                        #     logger.warning("---------------------------")
                        #     exit(0)   
                        hook_results2.append({
                            "layer": layer_name,
                            "input_type": type_of(input),
                            "output_type": type_of(output),
                            "input_norm": compute_norm_tensor(input),
                            "output_norm": compute_norm_tensor(output)
                        })

                    h = child.register_forward_hook(hook_fn)
                    hooks2.append(h)

                    register_hooks2(child, prefix=layer_name)
                    
            
            # torch.manual_seed(42)
            # torch.cuda.manual_seed_all(42)
            # if isinstance(latents, list):
            #     latents = [torch.rand_like(x) for x in latents]
            # else:
            #     latents = torch.rand_like(latents)

            # if isinstance(context, list):
            #     context = [torch.rand_like(x) for x in context]
            # else:
            #     context = torch.rand_like(context)
            # with torch.autocast("cuda", torch.bfloat16):
            #     for i in range(3):
            #         print(f"warmup {i}")
            #         B = len(context) if isinstance(context, list) else context.shape[0]
            #         device = latents[0].device
                
            #         # 使用sigma值计算timestep
            #         sigma = sigma_schedule[i]
                    
            #         timestep_value = int(sigma * 1000)
            #         timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
                    
            #         timestep_cond = timestep
            #         pred_cond = transformer(
            #             x=latents,  # [(16, 7, 64, 64)]
            #             t=timestep_cond,
            #             context=context,
            #             seq_len=seq_len
            #         )
                    
            print(" --------------- warmup finished. ------------------")
            # register_hooks2(transformer) 
            for i in progress_bar:
                hooks2.clear()
                hook_results2.clear()
                if i == 2: 
                    break
                B = len(context) if isinstance(context, list) else context.shape[0]
                # 确保设备一致
                device = latents[0].device
                
                # 使用sigma值计算timestep
                sigma = sigma_schedule[i]
                
                timestep_value = int(sigma * 1000)
                timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
                
                timestep_cond = timestep
                timestep_uncond = timestep
                

                        
                # transformer.eval() # fhd
                # with torch.autocast("cuda", torch.bfloat16):
                torch.cuda.synchronize()
                print(f"rank {torch.distributed.get_rank()}. begin rollout {i} !!!!!!!!!!!!!")
                with torch.autocast("cuda", torch.bfloat16):
                    # WAN模型输入：x是(C,T,H,W)格式的列表
                    # transformer.to(device)
                    print(f"rank {torch.distributed.get_rank()}. in rollout latents norm: i", i, latents[0].norm().item())
                    # torch.manual_seed(42)
                    # latents = torch.rand_like(latents)
                    # with torch.no_grad():

                    pred_cond = transformer(
                        x=latents,  # [(16, 7, 64, 64)]
                        t=timestep_cond,
                        context=context,
                        seq_len=seq_len
                    )
                    print(
                        f"[Rollout warmup finished] rank {torch.distributed.get_rank()} "
                        f"step {i}/{self.config.sampling_steps} "
                        f"shape={tuple(latents[0].shape)} "
                        f"norm={latents[0].norm().item():.4f} "
                        f"timestep_cond norm={timestep} "
                        f"context norm={context[0].norm().item():.4f} "
                        f"seq_len={seq_len} "
                        f"pred_cond1={pred_cond[0].norm()}"
                        f"pred_cond2={pred_cond[0].float().norm()}"
                    )
                    # hook_results_json = []
                    # def tensor_to_json_safe(x):
                    #     if isinstance(x, torch.Tensor):
                    #         return x.detach().cpu().item()  # 标量 norm
                    #     elif isinstance(x, (list, tuple)):
                    #         return [tensor_to_json_safe(i) for i in x]
                    #     else:
                    #         return None
                    # hook_results_json2 = []
                    # for entry in hook_results2:
                    #     hook_results_json2.append({
                    #         "layer": entry["layer"],
                    #         "input_type": entry["input_type"],
                    #         "output_type": entry["output_type"],
                    #         "input_norm": tensor_to_json_safe(entry["input_norm"]),
                    #         "output_norm": tensor_to_json_safe(entry["output_norm"])
                    #     })
                    # with open(f"14-diffusion-rollout-{i}-result-compile-true-grad-false-rank-{torch.distributed.get_rank()}.json", "w") as f:
                    #     json.dump(hook_results_json2, f, indent=2)
                    
                    # exit(0)
                    # 处理模型输出
                    if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                        model_output_cond = pred_cond['rgb'][0]
                    elif isinstance(pred_cond, list):
                        model_output_cond = pred_cond[0]
                    else:
                        model_output_cond = pred_cond
                        
                    model_output=model_output_cond
                    # 为无条件预测准备输入
                    # transformer.to(device)
                    # pred_uncond = transformer(
                    #     x=latents,  # [(16, 7, 64, 64)]
                    #     t=timestep_uncond,
                    #     context=neg_context,
                    #     seq_len=seq_len
                    # )

                    # if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                    #     model_output_uncond = pred_uncond['rgb'][0]
                    # elif isinstance(pred_uncond, list):
                    #     model_output_uncond = pred_uncond[0]
                    # else:
                    #     model_output_uncond = pred_uncond
                        
                    # del pred_cond, pred_uncond

                    # # CFG组合
                    # model_output = model_output_uncond + self.config.guide_scale * (model_output_cond - model_output_uncond)
                    # del model_output_cond, model_output_uncond
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
                
                latents=[next_latents.to(torch.float32)]
                all_latents.append(latents[0])  # 存储 (16, 7, 64, 64)
                all_log_probs.append(log_prob)  # 存储 log概率
                
                hooks2.clear()
                hook_results2.clear()
            
            final_latents = pred_original

            # 修正：WAN的all_latents维度是 (num_steps+1, 16, 7, 64, 64)
            all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
            all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
            
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
            # prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            prev_sample = prev_sample_mean
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

