import itertools
import logging
import math
import os
from collections import defaultdict

import numpy as np
import torch

from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outpus_and_unpad
from verl.workers.actor import DataParallelPPOActor

# from tensorwatch import TensorWatch,watch_module_forward_backward
__all__ = ["DiffusionDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def fprint(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    with open("output.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")

# hook 函数：保存输入数据和模块名
def save_input_hook(name):
    def hook(module, input, output):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        file_name = f"output/{rank}_{name}_input.pt"
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  Param: {name}, Shape: {param.shape},{param.float().norm().item()}")

        # print(f"[HOOK] Saving {name} input to {file_name}")
        torch.save({
            "name": name,
            "input": input,
            "output": output
        }, file_name)
    return hook

def register_all_hooks(model):
    for name, module in model.named_modules():
        print(f"[REGISTER] Hooking module: {name}")
        module.register_forward_hook(save_input_hook(name))

import torch
import json

hook_results = []
hooks = []

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
        if x.is_floating_point() and not x.is_meta:
            return x.norm(dim=None)  # 返回 tensor，延迟转 float
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
    
def register_hooks(module, prefix=""):
    for name, child in module.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name

        def hook_fn(module, input, output, layer_name=layer_name):
            # if layer_name == '_fsdp_wrapped_module.text_embedding.0':
            #     logger.warning("---------------------------")
            #     logger.warning(input)
            #     logger.warning("---------------------------")
            #     exit(0)   
            hook_results.append({
                "layer": layer_name,
                "input_type": type_of(input),
                "output_type": type_of(output),
                "input_norm": compute_norm_tensor(input),
                "output_norm": compute_norm_tensor(output)
            })

        h = child.register_forward_hook(hook_fn)
        hooks.append(h)

        register_hooks(child, prefix=layer_name)
        
class DiffusionDataParallelPPOActor(DataParallelPPOActor):

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        #这里显存就开始增加了
        # free, total = torch.cuda.mem_get_info()
        # print("begin forward")
        # print(f"剩余显存: {free / (1024 ** 3):.2f} GB")
        # print(f"总显存:   {total / (1024 ** 3):.2f} GB")
        # print(f"已用显存: {(total - free) / (1024 ** 3):.2f} GB")
        # data = data.to(get_device_id())
        # make sure we are in training mode
        # print(data.batch['timesteps'].device)

        self.actor_module.train()
 
        # torch.cuda.memory._record_memory_history(max_entries=100000)
        
        perms = torch.stack([
            torch.randperm(len(data.batch["timesteps"][0])) 
            for _ in range(data.batch.batch_size[0])
        ])

        from verl.utils.ulysses import get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size
        if get_ulysses_sequence_parallel_world_size() > 1:
            src_rank = (torch.distributed.get_rank() // get_ulysses_sequence_parallel_world_size()) * get_ulysses_sequence_parallel_world_size()
            torch.distributed.broadcast(perms, src=src_rank,group=get_ulysses_sequence_parallel_group())
            torch.distributed.barrier()

        # exit(0)
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            data.batch[key] = data.batch[key][
                torch.arange(data.batch.batch_size[0])[:, None],
                perms,
            ]
            
        # B = data.batch.batch_size[0]  # = 12
        # # 定义你想要的输出顺序：[1,2,3,1,2,3,1,2,3,1,2,3] 对应原始索引
        # # 原始数据： [1,1,1,1,  2,2,2,2,  3,3,3,3]
        # # 索引位置：[0,1,2,3,  4,5,6,7,  8,9,10,11]
        # # 目标顺序：取第0个1, 第0个2, 第0个3 → 然后第1个1,2,3 → ...
        # # 所以目标索引是：[0,4,8, 1,5,9, 2,6,10, 3,7,11]

        # # 自动生成这个 permutation
        # K = 3  # 有3种不同的值
        # T = 8  # 每个值重复4次
        # assert B == K * T

        # # 构造 permutation: [0,4,8,1,5,9,2,6,10,3,7,11]
        # perm = torch.tensor([i * T + j for j in range(T) for i in range(K)])  # 注意：i 是组号，j 是组内位置

        # # 应用到所有 key
        # for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        #     if key in data.batch:
        #         data.batch[key] = data.batch[key][perm]  # 直接按 batch 维度重排

        # print(data.batch["timesteps"].shape,type(data.batch["timesteps"]),data.batch["timesteps"][0].shape,type(data.batch["timesteps"][0]),len(data.batch["timesteps"][0].shape[1]))
        train_timesteps = int(len(data.batch["timesteps"][0]) * self.config.timestep_fraction)
        grad_norm = None
        
        # num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size

        select_keys=["timesteps", "latents", "next_latents", "log_probs","contexts","sigma_schedule","advantages","context_orig_lengths","null_context"]
        non_tensor_select_keys = ["caption"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        
        # from tensordict import TensorDict

        # batch = TensorDict()
        # for k, v in data.batch.items():
        #     if isinstance(v, torch.Tensor):
        #         batch[k] = v.unsqueeze(1)  # 添加一维，保持 [B, 1, ...] 结构
        # batch.batch_size=data.batch.batch_size

        self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )

        # data=data.to("cpu")
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(data.batch.batch_size[0])
        
        device = torch.device(f"cuda:{get_device_id()}")

        move_keys = ["latents", "next_latents", "timesteps", "log_probs", "advantages", "sigma_schedule"]
        perms=perms.to(device)
        # log_file = f"/gemini/space/wuxuaner/Dancegrpo/recipe/dancegrpo/log/rank{get_device_id()}"
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            td = data.batch
            ctx_lens = td["context_orig_lengths"].tolist() if torch.is_tensor(td["context_orig_lengths"]) else td["context_orig_lengths"]
            # 裁剪后的 CPU 列表（短生命周期）
            ctxs_cpu  = [td["contexts"][i][:int(ctx_lens[i])]   for i in range(len(data))]
            nctx_cpu  = [td["null_context"][i]                  for i in range(len(data))]
            
            td = data.pop(batch_keys=move_keys).to(device)
            # for k in move_keys:
                # print(k,device)
                # td[k] = td[k].to(device)
                # print("after",k,td[k].device)
                
            ctxs  = [c.to(device) for c in ctxs_cpu]
            nctxs = [c.to(device) for c in nctx_cpu]
            del ctxs_cpu, nctx_cpu
            # mini_batch = data
            # # split batch into micro_batches
            # micro_batches = mini_batch.chunk(self.config.ppo_micro_batch_size_per_gpu)
            self.actor_optimizer.zero_grad()
            # batch_contexts = [data.batch["contexts"][i] for i in range(len(data))]
            # batch_null_contexts = [data.batch["null_context"][i] for i in range(len(data))]
            # for i in range(len(data)):
            #     orig_lengths =int(data.batch['context_orig_lengths'][i])
            #     assert batch_contexts[i].shape[0] >= orig_lengths, \
            #         f"Context length mismatch: expected at least {orig_lengths}, but got {data.batch['contexts'][i].shape[0]}. Caption: {data.non_tensor_batch['caption']}"
            #     batch_contexts[i] = batch_contexts[i][:orig_lengths]
                
            for step_idx in range(train_timesteps):
                clip_range = self.config.clip_range
                adv_clip_max = self.config.adv_clip_max
                latent_t  = td.batch["latents"][:, step_idx]
                nlatent_t = td.batch["next_latents"][:, step_idx]
                t_t       = td.batch["timesteps"][:, step_idx]
                sigma_0   = td.batch["sigma_schedule"][0]  # 若每样本不同 schedule，这里改成按样本取
        
                latent_shape = td.batch["latents"][:, step_idx].shape
                seq_len = math.ceil(
                    (latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2]
                )
    
    
                new_log_probs = self.grpo_wan_one_step(
                    latent_t,
                    nlatent_t,
                    ctxs, 
                    nctxs,
                    seq_len,
                    self.actor_module,
                    t_t,
                    perms[batch_idx][step_idx],
                    sigma_0, 
                )
                # if step_idx==2:
                #     from datetime import datetime
                #     timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                #     file_name = f"visual_mem_{timestamp}_{torch.distributed.get_rank()}_{batch_idx}_step_idx_{step_idx}.pickle"
                #     # save record:
                #     torch.cuda.memory._dump_snapshot(file_name)
                #     # Stop recording memory snapshot history:
                #     torch.cuda.memory._record_memory_history(enabled=None)
                # print(new_log_probs)
                # exit(0)
                # 其余训练逻辑保持不变...
                advantages = torch.clamp(
                    td.batch["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                ratio = torch.exp(new_log_probs - td.batch["log_probs"][:, step_idx])
                clipped_mask = (ratio < (1.0 - clip_range)) | (ratio > (1.0 + clip_range))
                clip_count = clipped_mask.sum().detach().item()
                clip_fraction = clipped_mask.float().mean().detach().item()                     
                
                print("ratio",ratio)
                print("="*100)

                # ratio = torch.exp(new_log_probs - current_log_probs)
                
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                # temp = torch.maximum(unclipped_loss, clipped_loss)
                
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (self.gradient_accumulation * train_timesteps)
                data = {
                    "actor/clip_count": clip_count,
                    "actor/clip_fraction": clip_fraction,
                    "actor/loss": loss.detach().item(),
                    # "actor/log_ratio_mean": log_ratios.mean().detach().item(),
                    # "actor/preference_mean": preference.mean().detach().item(),
                }
                from verl.utils.py_functional import append_to_dict
                append_to_dict(metrics, data)
                loss.backward()

                avg_loss = loss.detach()

                torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
                
                # # 查看显存占用
                # import gc
                # # 遍历当前存活的所有 Tensor
                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(f"Tensor shape: {tuple(obj.shape)}, dtype: {obj.dtype}, device: {obj.device}, size: {obj.numel() * obj.element_size() / 1024**2:.2f} MB")
                #     except Exception as e:
                #         pass

                # print("batch_idx",batch_idx)
                # print("+"*100)
                # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                # print(f"Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                # print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
                # with open
                # if batch_idx==3:
                #     exit(0)
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                data = {"actor/grad_norm": grad_norm.detach().item(),}
                append_to_dict(metrics, data)
                
            del ctxs, nctxs
            for k in move_keys:
                if k in td: 
                    del td[k]
            del td, data
            torch.cuda.empty_cache()
                
            
        return metrics


            
    def grpo_wan_one_step(
        self,
        latents,
        pre_latents,
        context,
        context_null,
        seq_len,
        transformer,
        timesteps,
        i,
        sigma_schedule,
        guide_scale=5.0,
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
        
        register_hooks(transformer) 
        # exit(0)
        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype):
            # import torch
            # import hashlib

            # def tensor_md5(t) -> str:
            #     if isinstance(t, (list, tuple)):
            #         # 把多个 Tensor 的 md5 串起来再做一次 md5
            #         md5_all = hashlib.md5()
            #         for item in t:
            #             md5_all.update(tensor_md5(item).encode())
            #         return md5_all.hexdigest()
            #     elif isinstance(t, torch.Tensor):
            #         t_bytes = t.detach().cpu().numpy().tobytes()
            #         return hashlib.md5(t_bytes).hexdigest()
            #     else:
            #         # 其他类型转成字符串
            #         return hashlib.md5(str(t).encode()).hexdigest()
            # latents.to(pre_latents.device)
            # print("------------------------------------------------")
            # print(tensor_md5(a))
            # print("------------------------------------------------")
            # with torch.no_grad(): 
            
            # if torch.distributed.get_rank() == 0:
            #     print(latents)
            #     print(latents.float().norm())
            # exit(0)
            pred_cond = transformer(
                x=[latents],
                t=timesteps,
                context=context,
                seq_len=seq_len
            )
            print(
                f"[Update] rank {torch.distributed.get_rank()} "
                f"shape={tuple(latents.shape)} "
                f"norm={latents.norm().item():.4f} "
                f"timestep_cond norm={timesteps} "
                f"context norm={context[0].norm().item():.4f} "
                f"seq_len={seq_len} "
                f"pred_cond={pred_cond[0].norm()}"
            )
            
            # 处理条件预测输出
            hook_results_json = []
            def tensor_to_json_safe(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().item()  # 标量 norm
                elif isinstance(x, (list, tuple)):
                    return [tensor_to_json_safe(i) for i in x]
                else:
                    return None
            hook_results_json = []
            for entry in hook_results:
                hook_results_json.append({
                    "layer": entry["layer"],
                    "input_type": entry["input_type"],
                    "output_type": entry["output_type"],
                    "input_norm": tensor_to_json_safe(entry["input_norm"]),
                    "output_norm": tensor_to_json_safe(entry["output_norm"])
                })
            with open(f"14-dp-actor-result-compile-true-grad-true-rank-{torch.distributed.get_rank()}.json", "w") as f:
                json.dump(hook_results_json, f, indent=2)
                
            exit(0)
            
            if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                model_output_cond = pred_cond['rgb'][0]
            elif isinstance(pred_cond, list):
                model_output_cond = pred_cond[0]
            else:
                model_output_cond = pred_cond
                
            # 立即清理
            # del pred_cond
            # torch.cuda.empty_cache()
                
            #再计算无条件预测
            with torch.no_grad(): 
                pred_uncond = transformer(
                    x=[latents],  # 保持List[Tensor]格式
                    t=timesteps,
                    context=context_null,  # List[Tensor]
                    seq_len=seq_len
                )

                
                #处理无条件预测输出
                if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                    model_output_uncond = pred_uncond['rgb'][0]
                elif isinstance(pred_uncond, list):
                    model_output_uncond = pred_uncond[0]
                else:
                    model_output_uncond = pred_uncond
                    
                # del pred_uncond
                # torch.cuda.empty_cache()
            
            # CFG组合
            model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
            # del model_output_cond, model_output_uncond
            # model_output = model_output_cond

        # 确保数据类型一致性

        # computation_dtype = torch.float32
        _, _, log_prob = self.wan_step(
            model_output, 
            latents.to(torch.float32), 
            self.config.eta, 
            sigma_schedule,
            i, 
            prev_sample=pre_latents, 
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
        prev_sample_mean = latents + dsigma * model_output
        # 预测的原始样本
        pred_original_sample = latents - sigma * model_output
        
        delta_t = sigma - sigmas[index + 1]  # 时间差分
        # std_dev_t = eta * math.sqrt(abs(delta_t))  # 随机噪声的std
        std_dev_t = eta * torch.sqrt(delta_t)  # 根据hunyuan改的
        
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