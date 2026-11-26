
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from environment.factory_env_ppo_v3 import FactoryPPOV3Env


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ppo_production_optimizer_v3"


def main():

    print("\n========== PPO V3 TRAINING ==========")

    # Create environment
    env = FactoryPPOV3Env(episode_length=30)

    # Validate environment
    print("\nChecking environment...")
    check_env(env, warn=True)
    print("Environment check passed.")

    # Monitor environment
    env = Monitor(env)

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=0.0001,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,

        policy_kwargs={
            "net_arch": [128, 128]
        },

        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting PPO V3 training...")
    print("Total timesteps: 100,000")
    print("Device: CPU")

    # Train for exactly 100,000 timesteps
    model.learn(
        total_timesteps=100_000,
        progress_bar=True
    )

    # Save model separately
    model.save(str(MODEL_PATH))

    env.close()

    print("\n========== TRAINING COMPLETED ==========")
    print("Total timesteps: 100,000")
    print(f"Model saved at: {MODEL_PATH}.zip")
    print("========================================")


if __name__ == "__main__":
    main()
