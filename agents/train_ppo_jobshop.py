
"""
Train the alternative PPO scheduling agent on random job-shop instances.

PPO uses sb3-contrib's MaskablePPO with the environment mask wrapper, so
only eligible jobs are sampled during training and evaluation.

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe agents\\train_ppo_jobshop.py
"""

import sys

from pathlib import Path

from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_shop_env import (
    JobShopEnv, generate_instance, generate_simple_instance,
    run_dispatch, run_agent
)


TOTAL_TIMESTEPS = 200_000
N_PARALLEL_ENVS = 4
VALIDATION_INSTANCES = 10

# Set by the simple-mode wrapper scripts.
SIMPLE = False
MODEL_NAME = "jobshop_ppo"
LOG_NAME = "jobshop_ppo"


def problemset():
    return generate_simple_instance if SIMPLE else generate_instance


def mask_fn(env):
    return env.action_masks()


def make_env():
    env = ActionMasker(JobShopEnv(simple=SIMPLE), mask_fn)
    return Monitor(env)


class ProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps):
        super().__init__(total_timesteps)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc=f"{'Simple' if SIMPLE else 'Job-shop'} PPO training",
            unit="steps"
        )

    def _on_step(self):
        self.progress_bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()


def validate(model):
    """Quick paired comparison on held-out validation instances."""

    algorithms = {
        "PPO": None,
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

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        # Match the DQN objective: no discounting, so the dense
        # makespan increments sum exactly to the final makespan.
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        tensorboard_log=str(logs_dir),
        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting PPO training...")
    print(f"Mode: {'simple' if SIMPLE else 'job-shop'}")
    print(f"Target timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Parallel environments: {N_PARALLEL_ENVS}")
    print("Network: [128, 128]")
    print("TensorBoard logs:", logs_dir)
    print()

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=ProgressBarCallback(TOTAL_TIMESTEPS),
            progress_bar=False,
            tb_log_name=f"MaskablePPO_{LOG_NAME}"
        )

        model.save(str(models_dir / MODEL_NAME))

        print("\nPPO training completed.")
        print("Model saved:", models_dir / f"{MODEL_NAME}.zip")

        validate(model)

    finally:
        env.close()


if __name__ == "__main__":
    train()
