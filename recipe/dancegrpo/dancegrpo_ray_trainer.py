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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.debug import marked_timer
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_response_mask
# from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from omegaconf import OmegaConf, open_dict
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    datas=data.pop(
        batch_keys=['rewards'],
    )
    advantages=torch.zeros_like(datas.batch['rewards'])
    print("become to compute advantages")
    print(datas.batch['rewards'])
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
                print(new_batch.__len__)
                print("-"*100)
                # pop those keys for generation TODO!!!
                if self.config.trainer.type=="diffusion":
                    # print("new_batch keys:", new_batch.batch_keys.keys())
                    # print("non-tensor keys:", new_batch.non_tensor_batch_keys.keys())
                    gen_batch = new_batch.pop(
                        batch_keys=["context"],
                        non_tensor_batch_keys=["caption"],
                    )
                    gen_batch = gen_batch.repeat(self.config.actor_rollout_ref.sampling_steps)
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
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

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
                    # repeat to align with repeated responses in rollout
                    # new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # Calculate the HPS
                            with torch.amp.autocast('cuda'):
                                reward_tensor = self.rm_wg.compute_rm_score(gen_batch_output)
                                new_batch = gen_batch_output.union(reward_tensor)
                                del gen_batch_output

                    # === Updating ===
                    # batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.

                    
                    if self.config.trainer.balance_batch:
                        self._balance_batch(new_batch, metrics=metrics)

                    # compute global_valid tokens
                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    # with marked_timer("old_log_prob", timing_raw):
                    #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    #     entropys = old_log_prob.batch["entropys"]
                    #     response_masks = batch.batch["response_mask"]
                    #     loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    #     entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    #     old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                    #     metrics.update(old_log_prob_metrics)
                    #     old_log_prob.batch.pop("entropys")
                    #     batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        new_batch = compute_advantage(
                            new_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(new_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    # def init_workers(self):
    #     """Initialize distributed training workers using Ray backend.

    #     Creates:
    #     1. Ray resource pools from configuration
    #     2. Worker groups for each role (actor, critic, etc.)
    #     """
    #     self.resource_pool_manager.create_resource_pool()

    #     self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    #     # create actor and rollout
    #     if self.hybrid_engine:
    #         resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
    #         actor_rollout_cls = RayClassWithInitArgs(
    #             cls=self.role_worker_mapping[Role.ActorRollout],
    #             config=self.config.actor_rollout_ref,
    #             role="actor",
    #         )
    #         self.resource_pool_to_cls[resource_pool]["actor"] = actor_rollout_cls
    #     else:
    #         raise NotImplementedError

    #     # create critic
    #     if self.use_critic:
    #         resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
    #         critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
    #         self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

    #     # create reference policy if needed
    #     if self.use_reference_policy:
    #         resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
    #         ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
    #         self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

    #     # create a reward model if reward_fn is None
    #     if self.use_rm:
    #         # we create a RM here
    #         resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
    #         rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
    #         self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

    #     # initialize WorkerGroup
    #     # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
    #     # you should not use `create_colocated_worker_cls`.
    #     # Instead, directly pass different resource pool to different worker groups.
    #     # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
    #     all_wg = {}
    #     wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
    #     if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
    #         wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
    #     if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
    #         wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
    #         assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
    #         wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

    #     for resource_pool, class_dict in self.resource_pool_to_cls.items():
    #         worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    #         wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
    #         spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    #         all_wg.update(spawn_wg)

    #     if self.use_critic:
    #         self.critic_wg = all_wg["critic"]
    #         self.critic_wg.init_model()

    #     if self.use_reference_policy and not self.ref_in_actor:
    #         self.ref_policy_wg = all_wg["ref"]
    #         self.ref_policy_wg.init_model()

    #     if self.use_rm:
    #         self.rm_wg = all_wg["rm"]
    #         self.rm_wg.init_model()

    #     # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
    #     self.actor_rollout_wg = all_wg["actor"]
    #     self.actor_rollout_wg.init_model()

    #     # create async rollout manager and request scheduler
    #     self.async_rollout_mode = False
    #     if self.config.actor_rollout_ref.rollout.mode == "async":
    #         from verl.workers.rollout.async_server import AsyncLLMServerManager

    #         self.async_rollout_mode = True
    #         self.async_rollout_manager = AsyncLLMServerManager(
    #             config=self.config,
    #             worker_group=self.actor_rollout_wg,
    #         )

