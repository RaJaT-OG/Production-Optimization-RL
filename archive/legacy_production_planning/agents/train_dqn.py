
import sys
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env import FactoryEnv


# CONFIGURATION
TOTAL_TIMESTEPS = 200_000

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save separately so the old model is preserved
MODEL_PATH = MODEL_DIR / "dqn_production_optimizer_v2"


# CREATE ENVIRONMENT
env = Monitor(FactoryEnv())

print("\n========== DQN TRAINING V2 ==========")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# CREATE DQN MODEL
model = DQN(
    policy="MlpPolicy",
    env=env,

    learning_rate=0.0001,

    buffer_size=100_000,

    learning_starts=5_000,

    batch_size=128,

    tau=1.0,

    gamma=0.99,

    train_freq=4,

    gradient_steps=1,

    target_update_interval=1000,

    exploration_fraction=0.50,

    exploration_initial_eps=1.0,

    exploration_final_eps=0.05,

    policy_kwargs={
        "net_arch": [256, 128]
    },

    verbose=1,

    seed=42,

    device="cpu"
)


# TRAIN MODEL
print("\nStarting DQN V2 training...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    log_interval=10
)


# SAVE MODEL
model.save(str(MODEL_PATH))

print("\n========== TRAINING COMPLETED ==========")

print("Total timesteps:", TOTAL_TIMESTEPS)

print("Model saved at:")
print(MODEL_PATH)

env.close()
