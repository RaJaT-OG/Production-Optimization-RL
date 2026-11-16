
"""
Masked Deep Q-Network for stable-baselines3 2.9.2.

sb3-contrib provides MaskablePPO but not MaskableDQN. This module adds
masking to the standard DQN in two places:

1. The Q-network sets invalid actions to a large negative value, so both
   action selection and the Bellman target ignore them.
2. Epsilon-greedy exploration (including the warm-up phase) samples only
   valid actions.

The mask is stored in the last ``n_actions`` entries of the observation,
so it travels with every transition through the replay buffer. The
environment therefore has to append its action mask to the observation.
"""

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from environment.job_shop_env import (
    GLOBAL_FEATURES, JOB_FEATURES, MACHINE_FEATURES, MAX_JOBS,
    MAX_MACHINES
)


NEGATIVE_INFINITY = -1e9


def masks_from_observation(observation, n_actions):
    """Read the boolean action masks stored at the end of the observation."""

    array = np.asarray(observation)

    return array[..., -n_actions:] >= 0.5


# ==========================================
# MASKED Q-NETWORK
# ==========================================

class MaskedQNetwork(QNetwork):
    """
    Q-network with a shared per-job encoder.

    A flat MLP sees each job in a fixed slot, which makes it hard to
    compare jobs across slots. This network instead encodes every job
    with the same small MLP, together with the state of the machine
    that its next operation needs. The output is one Q-value per job,
    so the network is permutation-equivariant and learns dispatch
    comparisons (remaining work, queue length, processing time)
    directly.
    """

    MACHINE_BLOCK = MAX_MACHINES * MACHINE_FEATURES
    JOB_BLOCK = MAX_JOBS * JOB_FEATURES
    GLOBAL_BLOCK = GLOBAL_FEATURES
    GLOBAL_START = MACHINE_BLOCK + JOB_BLOCK

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        job_input = (
            JOB_FEATURES + MACHINE_FEATURES + GLOBAL_FEATURES + 1
        )

        self.job_encoder = th.nn.Sequential(
            th.nn.Linear(job_input, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 1)
        )

    def forward(self, observation):

        batch = observation.shape[0]

        machines = observation[:, :self.MACHINE_BLOCK].reshape(
            batch, MAX_MACHINES, MACHINE_FEATURES
        )
        jobs = observation[
            :, self.MACHINE_BLOCK:self.GLOBAL_START
        ].reshape(batch, MAX_JOBS, JOB_FEATURES)
        global_features = observation[
            :, self.GLOBAL_START:self.GLOBAL_START + self.GLOBAL_BLOCK
        ]

        n_actions = self.action_space.n
        mask = observation[:, -n_actions:] >= 0.5

        # Machine used by each job's next operation (one-hot position).
        machine_ids = jobs[:, :, 2:2 + MAX_MACHINES].argmax(dim=-1)

        machine_context = machines.gather(
            1,
            machine_ids.unsqueeze(-1).expand(-1, -1, MACHINE_FEATURES)
        )

        globals_per_job = global_features.unsqueeze(1).expand(
            -1, MAX_JOBS, -1
        )

        eligible = mask.to(observation.dtype).unsqueeze(-1)

        per_job = th.cat(
            [jobs, machine_context, globals_per_job, eligible],
            dim=-1
        )

        q_values = self.job_encoder(per_job).squeeze(-1)

        return q_values.masked_fill(~mask, NEGATIVE_INFINITY)


# ==========================================
# MASKED POLICY
# ==========================================

class MaskedDQNPolicy(DQNPolicy):
    """DQN policy that builds masked Q-networks."""

    def make_q_net(self):

        net_args = self._update_features_extractor(
            self.net_args,
            features_extractor=None
        )

        return MaskedQNetwork(**net_args).to(self.device)


# ==========================================
# MASKED DQN ALGORITHM
# ==========================================

class MaskedDQN(DQN):
    """DQN with action masking for discrete environments."""

    policy_aliases = dict(DQN.policy_aliases)
    policy_aliases["MlpPolicy"] = MaskedDQNPolicy

    def _random_valid_actions(self, observations):

        n_actions = self.action_space.n
        masks = masks_from_observation(observations, n_actions)

        if masks.ndim == 1:
            masks = masks.reshape(1, -1)

        actions = []

        for mask in masks:

            valid = np.flatnonzero(mask)

            if len(valid) == 0:
                actions.append(0)
            else:
                actions.append(
                    int(self.action_space.np_random.choice(valid))
                )

        return np.array(actions)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False
    ):

        if (
            not deterministic
            and self.action_space.np_random.random() < self.exploration_rate
        ):

            actions = self._random_valid_actions(observation)

            if np.asarray(observation).ndim == 1:
                return np.array(actions[0]), state

            return actions, state

        # Exploration is handled above. Passing deterministic=True here
        # prevents DQN.predict from running a second epsilon-greedy draw
        # that would sample uniformly over invalid actions.
        return super().predict(
            observation, state, episode_start, deterministic=True
        )

    def _sample_action(
        self,
        learning_starts,
        action_noise=None,
        n_envs=1
    ):

        if (
            self.num_timesteps < learning_starts
            and not (self.use_sde and self.use_sde_at_warmup)
        ):

            assert self._last_obs is not None, "self._last_obs was not set"

            unscaled_action = self._random_valid_actions(self._last_obs)

            return unscaled_action, unscaled_action

        return super()._sample_action(
            learning_starts, action_noise, n_envs
        )


# ==========================================
# DEMONSTRATION-REGULARISED DQN
# ==========================================

class DemonstrationMaskedDQN(MaskedDQN):
    """
    Masked DQN with an auxiliary behaviour-cloning loss on
    demonstration actions.

    A warm start gives the Q-network a strong policy, but pure TD
    updates can pull it back towards short-sighted actions when the
    reward is sparse. The auxiliary cross-entropy loss anchors the
    policy to the demonstrations while the TD loss still improves it.
    """

    def set_demonstrations(self, observations, actions):

        self.demo_observations = th.as_tensor(
            np.asarray(observations), dtype=th.float32
        )
        self.demo_actions = th.as_tensor(
            np.asarray(actions), dtype=th.long
        )
        self.bc_coef = 1.0

    def train(self, gradient_steps, batch_size=100):

        super().train(gradient_steps, batch_size)

        observations = getattr(self, "demo_observations", None)

        if observations is None:
            return

        self.policy.set_training_mode(True)

        count = len(observations)
        indices = th.randint(0, count, (batch_size,), device=self.device)

        batch_observations = observations[indices].to(self.device)
        batch_actions = self.demo_actions[indices].to(self.device)

        q_values = self.q_net(batch_observations)
        loss = F.cross_entropy(q_values, batch_actions)

        self.policy.optimizer.zero_grad()
        (self.bc_coef * loss).backward()
        th.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.policy.optimizer.step()

        self.logger.record("train/bc_loss", float(loss.detach()))
