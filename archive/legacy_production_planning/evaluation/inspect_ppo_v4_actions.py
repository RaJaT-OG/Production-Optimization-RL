
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sb3_contrib import MaskablePPO
from environment.factory_env_ppo_v4 import FactoryPPOV4Env
from sb3_contrib.common.wrappers import ActionMasker


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v4.zip"


def mask_fn(env):
    return env.action_masks()


def main():

    print("\n========== PPO V4 ACTION INSPECTION ==========")

    base_env = FactoryPPOV4Env(
        data_path=str(BASE_DIR / "data" / "raw" / "production_data.csv"),
        episode_length=30
    )

    env = ActionMasker(base_env, mask_fn)

    model = MaskablePPO.load(str(MODEL_PATH))

    action_counts = Counter()
    executed_production = np.zeros(3)

    for seed in range(100, 130):

        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action_mask = env.action_masks()

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_mask
            )

            action_counts[int(action)] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            executed_production += info["production"]

    print("\nRequested action distribution:")

    for action, count in action_counts.most_common(15):

        production = base_env.decode_action(action)

        print(
            f"Action {action:3d} | "
            f"Production {production} | "
            f"Selected {count} times"
        )

    total_days = 30 * 30

    print("\nAverage executed production per day:")

    print("A:", round(executed_production[0] / total_days, 2))
    print("B:", round(executed_production[1] / total_days, 2))
    print("C:", round(executed_production[2] / total_days, 2))

    env.close()

    print("\n==============================================")


if __name__ == "__main__":
    main()
