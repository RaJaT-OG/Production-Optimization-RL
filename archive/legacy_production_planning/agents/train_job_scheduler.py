
import sys

from pathlib import Path

from tqdm import tqdm

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_scheduling_env import JobSchedulingEnv


TOTAL_TIMESTEPS = 300_000
N_PARALLEL_ENVS = 4


def mask_fn(env):
    return env.action_masks()


def make_env():
    env = JobSchedulingEnv(
        data_path=str(
            BASE_DIR / "data" / "raw" / "production_data.csv"
        )
    )

    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    return env


class ProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps):
        super().__init__(total_timesteps)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc="Job Scheduler Training",
            unit="steps"
        )

    def _on_step(self):
        self.progress_bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()


def train():

    models_dir = BASE_DIR / "models"
    reports_dir = BASE_DIR / "reports"
    logs_dir = BASE_DIR / "logs" / "job_scheduler"

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env] * N_PARALLEL_ENVS)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        ),
        tensorboard_log=str(logs_dir),
        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting job scheduler training (MaskablePPO)...")
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
            tb_log_name="MaskablePPO_JobScheduler"
        )

        model.save(
            str(models_dir / "job_scheduler_maskableppo")
        )

        print("\nJob scheduler training completed.")
        print(
            "Model saved:",
            models_dir / "job_scheduler_maskableppo.zip"
        )

    finally:
        env.close()


if __name__ == "__main__":
    train()
