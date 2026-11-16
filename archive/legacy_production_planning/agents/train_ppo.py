
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from environment.factory_env_ppo import FactoryPPOEnv


def main():

    print("\n========== PPO TRAINING ==========")

    # Create environment
    env = FactoryPPOEnv(episode_length=30)

    # Validate environment compatibility
    print("\nChecking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")

    # Wrap environment for monitoring
    env = Monitor(env)

    # PPO configuration
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[128, 128]
        ),
        verbose=1,
        seed=42,
        device="cpu"
    )

    print("\nStarting PPO training...")
    print("Total timesteps: 200,000")

    # Train model
    model.learn(
        total_timesteps=200_000,
        progress_bar=True
    )

    # Save trained model
    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "ppo_production_optimizer_v1"

    model.save(str(model_path))

    print("\n========== TRAINING COMPLETED ==========")
    print("Model saved at:", model_path)
    print("Total timesteps:", model.num_timesteps)

    env.close()


if __name__ == "__main__":
    main()
