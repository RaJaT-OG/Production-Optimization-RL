
import sys
from pathlib import Path

# Add project root to Python's import path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from collections import Counter

from sb3_contrib import MaskablePPO
from environment.factory_env_ppo_v4 import FactoryPPOV4Env

DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v4.zip"


def diagnose_policy(model, deterministic):
    env = FactoryPPOV4Env(
        data_path=str(DATA_PATH),
        episode_length=30
    )

    total_days = 0
    days_c_feasible = 0
    days_c_selected = 0
    feasible_actions_with_c = 0
    total_feasible_actions = 0
    total_c_production = 0

    action_counts = Counter()
    production_counts = Counter()

    for seed in range(100, 130):
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Get valid actions in the current PPO state
            mask = env.action_masks()
            combinations = env.production_combinations
            valid_actions = combinations[mask]

            c_actions = valid_actions[valid_actions[:, 2] > 0]

            total_days += 1
            total_feasible_actions += len(valid_actions)
            feasible_actions_with_c += len(c_actions)

            if len(c_actions) > 0:
                days_c_feasible += 1

            # Ask PPO to choose an action
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask
            )

            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            production = tuple(int(x) for x in info["production"])
            production_counts[production] += 1

            c_quantity = production[2]
            total_c_production += c_quantity

            if c_quantity > 0:
                days_c_selected += 1

    env.close()

    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"

    print("\n==========", mode, "POLICY DIAGNOSIS ==========")
    print("Total decisions:", total_days)

    print(
        "Days where Product C was feasible:",
        days_c_feasible,
        f"({days_c_feasible / total_days * 100:.2f}%)"
    )

    print(
        "Days where PPO selected Product C:",
        days_c_selected,
        f"({days_c_selected / total_days * 100:.2f}%)"
    )

    print("Total feasible actions:", total_feasible_actions)
    print("Feasible actions including C:", feasible_actions_with_c)

    print(
        "Percentage of feasible actions including C:",
        f"{feasible_actions_with_c / total_feasible_actions * 100:.2f}%"
    )

    print("Total Product C units produced:", total_c_production)
    print("Average Product C production per day:", total_c_production / total_days)

    print("\nTop 10 selected production combinations:")
    for production, count in production_counts.most_common(10):
        print(f"{production}: {count} times")

    print("\nTop 10 selected action IDs:")
    for action, count in action_counts.most_common(10):
        print(f"Action {action}: {count} times")


def main():
    print("Loading PPO V4 model...")

    model = MaskablePPO.load(str(MODEL_PATH))

    diagnose_policy(model, deterministic=True)
    diagnose_policy(model, deterministic=False)


if __name__ == "__main__":
    main()
