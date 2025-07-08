import itertools
import math
import logging
from collections import defaultdict
import os
import numpy as np
import torch

from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend

__all__ = ["DiffusionDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DiffusionDataParallelPPOActor(DataParallelPPOActor):

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        data = data.to(get_device_id())
        # make sure we are in training mode
        self.actor_module.train()

        # 训练前 shuffle step-level 顺序（不同 batch 内 permute）
        timesteps = data.batch["timesteps"]
        perms = torch.stack([
            torch.randperm(len(timesteps[0])) 
            for _ in range(data.batch.batch_size[0])
        ]).to(get_device_id())

        # 对关键 tensor 字段进行 step-level 打乱
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            data.batch[key] =  data.batch[key][
                torch.arange(data.batch.batch_size[0]).to(get_device_id())[:, None],
                perms,
            ]

        train_timesteps = int(len(data.batch["timesteps"][0]) * self.config.timestep_fraction)
        grad_norm = None
        
        # num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size

        select_keys=["timesteps", "latents", "next_latents", "log_probs","contexts","sigma_schedule","advantages"]
        non_tensor_select_keys = ["caption"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        
        # from tensordict import TensorDict

        # batch = TensorDict()
        # for k, v in data.batch.items():
        #     if isinstance(v, torch.Tensor):
        #         batch[k] = v.unsqueeze(1)  # 添加一维，保持 [B, 1, ...] 结构
        # batch.batch_size=data.batch.batch_size


        # 合并到 data 中（假设 DataProto 接收一个包含 batch 的 dict）
        # data.batch=batch
        
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(data.batch.batch_size[0])

        for batch_idx, data in enumerate(dataloader):
            mini_batch = data

            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
            # split batch into micro_batches
            micro_batches = mini_batch.chunk(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
            context=[data.batch["contexts"].squeeze(0)]
            for step_idx in range(train_timesteps):
                clip_range = self.config.clip_range
                adv_clip_max = self.config.adv_clip_max
                
                latent_shape = data.batch["latents"][:, step_idx].shape
                seq_len = math.ceil(
                    (latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2]
                )
                
                new_log_probs = self.grpo_wan_one_step(
                    data.batch["latents"][:, step_idx],
                    data.batch["next_latents"][:, step_idx],
                    context,  # List[Tensor]格式
                    seq_len,
                    self.actor_module,
                    data.batch["timesteps"][:, step_idx],
                    perms[batch_idx][step_idx],
                    data.batch["sigma_schedule"][0], 
                )

                # 其余训练逻辑保持不变...
                advantages = torch.clamp(
                    data.batch["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                ratio = torch.exp(new_log_probs - data.batch["log_probs"][:, step_idx])

                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (self.gradient_accumulation * train_timesteps)

                loss.backward()
                avg_loss = loss.detach().clone()

                torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
                # total_loss += avg_loss.item()
                
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()

    def grpo_wan_one_step(
        self,
        latents,
        pre_latents,
        context,
        seq_len,
        transformer,
        timesteps,
        i,
        sigma_schedule,
    ):
        """GRPO的单步训练，支持FP16优化"""
        B = len(context) if isinstance(context, list) else context.shape[0]
        transformer.train()
        
        # 确保latents维度正确：(16, 7, 64, 64)
        if latents.dim() == 5:
            latents = latents.squeeze(0)
        
        if pre_latents.dim() == 5:
            pre_latents = pre_latents.squeeze(0)
        
        if latents.shape[0] != 16:
            raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")
        
        # 使用适当的数据类型进行autocast
        autocast_dtype = torch.float16
        # print("[DEBUG],157",context[0].shape)
        with torch.autocast("cuda", dtype=autocast_dtype):
            pred = transformer(
                x=[latents],
                t=timesteps.squeeze(),
                context=context,
                seq_len=seq_len
            )
            
            if isinstance(pred, dict) and 'rgb' in pred:
                model_output = pred['rgb'][0]
            elif isinstance(pred, list):
                model_output = pred[0]
            else:
                model_output = pred

        # 确保数据类型一致性
        # computation_dtype = torch.float32  # 关键计算使用FP32以保持精度 这里也暂时改成fp16吧
        computation_dtype = torch.float16
        _, _, log_prob = self.wan_step(
            model_output.to(computation_dtype), 
            latents.to(computation_dtype), 
            self.config.eta, 
            sigma_schedule,
            i, 
            prev_sample=pre_latents.to(computation_dtype), 
            grpo=True, 
            sde_solver=True
        )
        
        return log_prob

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
                    