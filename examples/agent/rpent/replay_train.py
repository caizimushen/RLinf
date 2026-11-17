# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify one synchronous PPO update using saved policy turns and original images."""

import json
import os
import socket
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent.replay import RPentReplayAgentLoopWorker, SavedRPentBatch
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.agent_ppo_runner import AgentPPORunner
from rlinf.scheduler import Cluster
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic
from rlinf.workers.rollout.utils import get_rollout_backend_worker


class _MemoryAudit:
    """Observe learner allocator peaks during the production update."""

    def reset_memory_peaks(self) -> None:
        """Start measuring the next update after completing pending device work."""
        self.torch_platform.synchronize()
        self.torch_platform.reset_peak_memory_stats()

    def training_memory(self) -> dict:
        """Return this rank's allocator peaks and allocation failure counters."""
        self.torch_platform.synchronize()
        stats = self.torch_platform.memory_stats()
        return {
            "rank": self._rank,
            "hostname": socket.gethostname(),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "peak_allocated_bytes": self.torch_platform.max_memory_allocated(),
            "peak_reserved_bytes": self.torch_platform.max_memory_reserved(),
            "allocated_bytes": self.torch_platform.memory_allocated(),
            "reserved_bytes": self.torch_platform.memory_reserved(),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
            "num_ooms": stats.get("num_ooms"),
            "num_alloc_retries": stats.get("num_alloc_retries"),
        }


class ReplayActor(_MemoryAudit, MAMegatronActor):
    """The production actor with peak-memory observations for replay validation."""


class ReplayCritic(_MemoryAudit, MAMegatronCritic):
    """The production critic with peak-memory observations for replay validation."""


class ReplayPPORunner(AgentPPORunner):
    """Run the production update and audit memory before worker cleanup."""

    def _run_training(self, rollout_handle: Handle):
        self.actor.reset_memory_peaks().wait()
        self.critic.reset_memory_peaks().wait()
        result = super()._run_training(rollout_handle)
        self.replay_memory = {
            "actor": self.actor.training_memory().wait(),
            "critic": self.critic.training_memory().wait(),
        }
        self.replay_ids = self.agent_loop.replayed_episode_ids().wait()
        self.replay_generation_states = self.rollout.get_async_rollout_state().wait()
        if any(state["generated_tokens"] for state in self.replay_generation_states):
            raise RuntimeError("Saved-episode replay unexpectedly generated new tokens")
        return result


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    """Validate a complete archived batch with fresh original-policy weights."""
    cfg = validate_cfg(cfg)
    if (
        not cfg.runner.synchronous
        or cfg.runner.max_steps != 1
        or cfg.runner.resume_dir is not None
        or cfg.replay.expected_version != 0
    ):
        raise ValueError(
            "Replay validation requires one fresh synchronous version-0 update"
        )
    batch = SavedRPentBatch(
        Path(cfg.replay.source_dir),
        Path(cfg.replay.manifest),
        cfg.replay.expected_version,
    )
    if len(batch) != cfg.data.rollout_batch_size:
        raise ValueError("Replay manifest does not contain the configured full batch")
    if cfg.data.get("sampling_batch_size", len(batch)) != len(batch):
        raise ValueError(
            "Training replay must use every saved episode without oversampling"
        )
    batch.verify_policy(Path(cfg.rollout.model.model_path))
    get_logger().info("Verified original policy and %s saved episodes", len(batch))
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
            env_vars=OmegaConf.to_container(
                cfg.rollout.get("env_vars", {}), resolve=True
            )
            if cfg.rollout.get("env_vars")
            else None,
        )
    )
    agent_loop = RPentReplayAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
    )
    actor = ReplayActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
        env_vars=OmegaConf.to_container(cfg.actor.get("env_vars", {}), resolve=True)
        if cfg.actor.get("env_vars")
        else None,
    )
    critic = ReplayCritic.create_group(cfg, placement).launch(
        cluster,
        name=cfg.critic.group_name,
        placement_strategy=placement.get_strategy("critic"),
        env_vars=OmegaConf.to_container(cfg.critic.get("env_vars", {}), resolve=True)
        if cfg.critic.get("env_vars")
        else None,
    )
    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)
    runner = ReplayPPORunner(
        cfg=cfg,
        placement=placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent_loop,
        critic=critic,
    )
    runner.init_workers()
    runner.run()
    ids = [episode for worker_ids in runner.replay_ids for episode in worker_ids]
    if len(ids) != len(batch) or len(set(ids)) != len(batch):
        raise RuntimeError(
            "Replay did not update on exactly one copy of every saved episode"
        )
    if any(
        record["num_ooms"] != 0
        for records in runner.replay_memory.values()
        for record in records
    ):
        raise RuntimeError("A learner reported an OOM during the saved-batch update")
    output = Path(cfg.runner.output_dir) / cfg.runner.experiment_name
    report = {
        "passed": True,
        "scope": "training-only saved-batch validation, excluded from formal training",
        "source_dir": str(Path(cfg.replay.source_dir).resolve()),
        "source_manifest": str(batch.manifest_path),
        "source_manifest_sha256": batch.manifest_sha256,
        "episodes": len(ids),
        "episode_ids": ids,
        "new_generated_tokens": sum(
            state["generated_tokens"] for state in runner.replay_generation_states
        ),
        "memory": runner.replay_memory,
        "checkpoint": str(output / "checkpoints/global_step_1"),
        "formal_initialization_must_remain_original_sft": True,
    }
    (output / "replay-training-memory.json").write_text(json.dumps(report, indent=2))
    get_logger().info("Saved-batch actor and critic update completed: %s", report)


if __name__ == "__main__":
    main()
