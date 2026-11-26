
"""
Train the primary DQN scheduling agent on random job-shop instances.

The agent sees a fixed-size state vector and chooses one eligible job at
every decision point. Action masking is implemented in agents/masked_dqn.py
because sb3-contrib does not ship a MaskableDQN.

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe agents\\train_dqn_jobshop.py
"""

import sys

from pathlib import Path

import numpy as np

from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents.masked_dqn import DemonstrationMaskedDQN
from environment.job_shop_env import (
    JobShopEnv, generate_instance, generate_simple_instance,
    run_dispatch, run_agent
)


TOTAL_TIMESTEPS = 400_000
N_PARALLEL_ENVS = 4
VALIDATION_INSTANCES = 10

# Set by the simple-mode wrapper scripts.
SIMPLE = False
MODEL_NAME = "jobshop_dqn"
LOG_NAME = "jobshop_dqn"

REWARD_MODE = "terminal"

WARM_START = True
WARM_START_INSTANCES = 400
WARM_START_EPOCHS = 60
WARM_START_LR = 3e-3
WARM_START_BC_COEF = 3.0


def problemset():
    """Instance generator used for validation and demonstrations."""

    return generate_simple_instance if SIMPLE else generate_instance


def make_env():
    return Monitor(JobShopEnv(reward_mode=REWARD_MODE, simple=SIMPLE))


class ProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps):
        super().__init__(total_timesteps)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc=f"{'Simple' if SIMPLE else 'Job-shop'} DQN training",
            unit="steps"
        )

    def _on_step(self):
        self.progress_bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()


def _remaining_work(env, job_index):
    """Processing time left on a job, including its next operation."""

    operations = env.instance["jobs"][job_index]["operations"]
    start = int(env.next_op[job_index])

    return sum(op["time"] for op in operations[start:])


def collect_demonstrations(n_instances):
    """
    Run the most-work-remaining (MWKR) rule and record its decisions.

    MWKR is a strong dispatching rule for makespan: it starts jobs with
    a long tail early. The DQN is warmed up from these decisions and
    then fine-tuned with its own reward, which is far more sample
    efficient than learning the rule from scratch.
    """

    observations = []
    actions = []

    for index in range(n_instances):

        instance = problemset()(seed=2_000_000 + index)

        env = JobShopEnv(simple=SIMPLE)
        observation, _ = env.reset(
            seed=0, options={"instance": instance}
        )

        done = False

        while not done:

            valid = np.flatnonzero(env.action_masks())

            action = int(max(
                valid,
                key=lambda job: (_remaining_work(env, job), -job)
            ))

            observations.append(observation.copy())
            actions.append(action)

            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64)
    )


def warm_start(model, observations, actions):
    """Fit the Q-network to imitate the MWKR rule."""

    import torch as th
    import torch.nn.functional as F

    device = model.device

    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    action_tensor = th.as_tensor(actions, dtype=th.long, device=device)

    optimizer = th.optim.Adam(
        model.q_net.parameters(), lr=WARM_START_LR
    )
    model.policy.set_training_mode(True)

    count = len(obs_tensor)

    for epoch in range(WARM_START_EPOCHS):

        permutation = th.randperm(count, device=device)
        total_loss = 0.0

        for start in range(0, count, 256):

            indices = permutation[start:start + 256]

            q_values = model.q_net(obs_tensor[indices])
            loss = F.cross_entropy(q_values, action_tensor[indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(indices)

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(
                f"  warm-start epoch {epoch + 1}/{WARM_START_EPOCHS}"
                f" loss={total_loss / count:.4f}"
            )

    model.q_net_target.load_state_dict(model.q_net.state_dict())


def validate(model):
    """Quick paired comparison on held-out validation instances."""

    algorithms = {
        "DQN": None,
        "Random": "random",
        "FCFS": "fcfs",
        "SPT": "spt"
    }

    totals = {
        name: {"makespan": 0.0, "waiting": 0.0, "utilization": 0.0}
        for name in algorithms
    }

    for index in range(VALIDATION_INSTANCES):

        instance = problemset()(seed=9000 + index)

        for name, rule in algorithms.items():

            if rule is None:
                env = run_agent(model, instance)
            else:
                env = run_dispatch(instance, strategy=rule, seed=index)

            metrics = env.metrics()

            totals[name]["makespan"] += metrics["makespan"]
            totals[name]["waiting"] += metrics["mean_waiting_time"]
            totals[name]["utilization"] += metrics["mean_utilization"]

    print("\n========== VALIDATION (10 unseen instances) ==========")
    print(
        f"{'Algorithm':10s} {'Makespan':>10s} {'Waiting':>10s} {'Util %':>10s}"
    )

    for name in algorithms:

        count = VALIDATION_INSTANCES

        print(
            f"{name:10s}"
            f" {totals[name]['makespan'] / count:10.2f}"
            f" {totals[name]['waiting'] / count:10.2f}"
            f" {totals[name]['utilization'] / count:10.2f}"
        )

    print("======================================================")


def train():

    models_dir = BASE_DIR / "models"
    logs_dir = BASE_DIR / "logs" / LOG_NAME

    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env] * N_PARALLEL_ENVS)

    model = DemonstrationMaskedDQN(
        "MlpPolicy",
        env,
        # A lower learning rate protects the warm-started policy.
        learning_rate=1e-4 if WARM_START else 3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=128,
        tau=1.0,
        # gamma=1 keeps the makespan reward perfectly aligned.
        gamma=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.10 if WARM_START else 0.25,
        exploration_initial_eps=0.30 if WARM_START else 1.0,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(logs_dir),
        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting DQN training...")
    print(f"Mode: {'simple' if SIMPLE else 'job-shop'}")
    print(f"Target timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Parallel environments: {N_PARALLEL_ENVS}")
    print(f"Reward mode: {REWARD_MODE}")
    print(f"Warm start: {WARM_START}")
    print("Network: per-job encoder [128, 128]")
    print("TensorBoard logs:", logs_dir)
    print()

    if WARM_START:

        print(
            f"Collecting MWKR demonstrations on "
            f"{WARM_START_INSTANCES} instances..."
        )
        demo_obs, demo_actions = collect_demonstrations(
            WARM_START_INSTANCES
        )
        print(f"  collected {len(demo_obs):,} decisions")

        warm_start(model, demo_obs, demo_actions)
        model.set_demonstrations(demo_obs, demo_actions)
        model.bc_coef = WARM_START_BC_COEF

        print("\nWarm-started network:")
        validate(model)
        print()

    try:

        checkpoint_dir = models_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = CallbackList([
            ProgressBarCallback(TOTAL_TIMESTEPS),
            CheckpointCallback(
                save_freq=max(50_000 // N_PARALLEL_ENVS, 1),
                save_path=str(checkpoint_dir),
                name_prefix=MODEL_NAME
            )
        ])

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False,
            tb_log_name=f"MaskedDQN_{LOG_NAME}"
        )

        model.save(str(models_dir / MODEL_NAME))

        print("\nDQN training completed.")
        print("Model saved:", models_dir / f"{MODEL_NAME}.zip")

        validate(model)

    finally:
        env.close()


if __name__ == "__main__":
    train()
