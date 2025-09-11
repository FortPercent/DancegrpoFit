import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import time

# from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend


def fprint(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    with open("output.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    datas=data.pop(
        batch_keys=['rewards'],
    )
    advantages=torch.zeros_like(datas.batch['rewards'])
    #TODO when batchsize not equal to 1
    group_mean = datas.batch['rewards'].mean()
    group_std = datas.batch['rewards'].std() + 1e-8
    advantages = (datas.batch['rewards'] - group_mean) / group_std
    data.batch["advantages"] = advantages
    return data


class RayDanceGRPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    #TODO dataset!!!! DataProto
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop those keys for generation TODO!!!
                # trainer的类型是diffusion
                if self.config.trainer.type=="diffusion":
                    # print("non-tensor keys:", new_batch.non_tensor_batch_keys.keys())
                    gen_batch = new_batch.pop(
                        batch_keys=["context","context_orig_lengths","null_context"],
                        non_tensor_batch_keys=["caption"],
                    )
                    
                    # 形状与配置
                    B = new_batch.batch.batch_size[0]
                    S = self.config.actor_rollout_ref.sampling_steps
                    num_frames = self.config.actor_rollout_ref.num_frames
                    size = (self.config.actor_rollout_ref.w, self.config.actor_rollout_ref.h)
                    vae_stride = [4, 8, 8]
                    latent_dtype = torch.float16
                    latent_shape = (
                        16,
                        (num_frames - 1) // vae_stride[0] + 1,
                        size[1] // vae_stride[1],
                        size[0] // vae_stride[2],
                    )

                    # 预分配批量张量
                    input_latents = torch.empty((B, *latent_shape), dtype=latent_dtype)
                    sigma_schedule_B = torch.empty((B, S+1), dtype=torch.float32)

                    def sd3_time_shift(shift, x):
                        return (shift * x) / (1 + (shift - 1) * x)

                    for i in range(new_batch.batch.batch_size[0]):  # 注意：这里用 gen_batch，不是 new_batch
                        # 这两个其实对所有样本都一样，放在循环外算一次也行
                        
                        sigma_schedule = torch.linspace(1, 0, S + 1)
                        
                        sigma_schedule = sd3_time_shift(self.config.actor_rollout_ref.shift, sigma_schedule)   # [S+1]
                        
                        sigma_schedule_B[i] = sigma_schedule

                        input_latents[i] = torch.randn(latent_shape, dtype=latent_dtype)

                    # 塞回 gen_batch
                    gen_batch.batch["input_latents"]  = input_latents
                    gen_batch.batch["sigma_schedule"] = sigma_schedule_B
                    
                    gen_batch = gen_batch.repeat(self.config.actor_rollout_ref.rollout.n)
                elif "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        # gen_batch_output的数据类型是DataProto
                        # 具体见DiffusionActorRolloutWorker.generate_sequences方法
                        # 得到的gen_batch_output是聚合所有gpu的结果
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # 目前用的是gae，TODO:修改reward计算方法
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw):
                            from verl.utils.checkpoint.checkpoint_manager import save_video_and_prompt
                            video_frames = gen_batch_output.batch["video_frames"] 
                            for i in range(video_frames.shape[0]):
                                save_video_and_prompt(video_frames[i] , 0, i)
                    # repeat to align with repeated responses in rollout
                    # new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # Calculate the HPS 自动混合精度计算
                            with torch.amp.autocast('cuda'):
                                reward_tensor = self.rm_wg.compute_rm_score(gen_batch_output)
                                gen_batch_output.pop(batch_keys=['video_frames'])
                                
                                gen_batch_output = gen_batch_output.union(reward_tensor)
                                metrics["train/rewards"] = gen_batch_output.batch['rewards'].mean()
                                metrics["train/log_probs"] = gen_batch_output.batch["log_probs"].mean()                                

                        else:
                            reward_tensor = self.reward_fn(gen_batch_output, return_dict=True)
                            gen_batch_output = gen_batch_output.union(reward_tensor)
                            gen_batch_output.pop(batch_keys=['video_frames'])
                    
                    # === Updating ===
                    # batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.

                    
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(new_batch, metrics=metrics)

                    # compute global_valid tokens
                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # if self.use_reference_policy:
                    #     # compute reference log_prob
                    #     with marked_timer("ref", timing_raw):
                    #         ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    #         batch = batch.union(ref_log_prob)

                    # compute values
                    # if self.use_critic:
                    #     with marked_timer("values", timing_raw):
                    #         values = self.critic_wg.compute_values(batch)
                    #         batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        gen_batch_output = compute_advantage(
                            gen_batch_output,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        metrics["train/advantage"] = gen_batch_output.batch['advantages'].mean()

                    # # update critic
                    # if self.use_critic:
                    #     with marked_timer("update_critic", timing_raw):
                    #         critic_output = self.critic_wg.update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    #     metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.update_actor(gen_batch_output)
                        actor_output_metrics = reduce_metrics(gen_batch_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with marked_timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=new_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # timing_raw = defaultdict(float)  # clear timing

                # metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1