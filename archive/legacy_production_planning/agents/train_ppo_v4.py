
import os
from tqdm import tqdm

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from environment.factory_env_ppo_v4 import FactoryPPOV4Env


TOTAL_TIMESTEPS = 100_000


def mask_fn(env):
    return env.action_masks()


def make_env():
    env = FactoryPPOV4Env(
        data_path="data/raw/production_data.csv",
        episode_length=30
    )

    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    return env


class ProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc="PPO V4 Training",
            unit="steps"
        )

    def _on_step(self):
        self.progress_bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self.progress_bar.close()


def train():

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    env = DummyVecEnv([make_env])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        ),
        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting PPO V4 training...")
    print(f"Target timesteps: {TOTAL_TIMESTEPS:,}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressBarCallback(TOTAL_TIMESTEPS),
        progress_bar=False
    )

    model.save("models/ppo_production_optimizer_v4")

    print("\nPPO V4 training completed.")
    print("Model saved: models/ppo_production_optimizer_v4.zip")

    env.close()


if __name__ == "__main__":
    train()
