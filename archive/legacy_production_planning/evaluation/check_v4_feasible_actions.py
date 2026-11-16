
import numpy as np
from pathlib import Path

from environment.factory_env_ppo_v4 import FactoryPPOV4Env

BASE_DIR = Path(__file__).resolve().parent.parent

def main():

    env = FactoryPPOV4Env(
        data_path=str(BASE_DIR / "data" / "raw" / "production_data.csv"),
        episode_length=30
    )

    total_days = 0
    days_with_c_feasible = 0
    total_feasible_actions = 0
    feasible_actions_with_c = 0

    for seed in range(100, 130):

        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False

        while not (terminated or truncated):

            mask = env.action_masks()

            combinations = env.production_combinations

            valid = combinations[mask]

            c_actions = valid[valid[:, 2] > 0]

            total_days += 1
            total_feasible_actions += len(valid)
            feasible_actions_with_c += len(c_actions)

            if len(c_actions) > 0:
                days_with_c_feasible += 1

            obs, reward, terminated, truncated, info = env.step(
                0
            )

    print("\n========== PRODUCT C FEASIBILITY ==========")

    print("Total decisions:", total_days)

    print(
        "Days where Product C production is feasible:",
        days_with_c_feasible
    )

    print(
        "Percentage of days:",
        round(days_with_c_feasible / total_days * 100, 2),
        "%"
    )

    print(
        "Total feasible actions:",
        total_feasible_actions
    )

    print(
        "Feasible actions including Product C:",
        feasible_actions_with_c
    )

    env.close()


if __name__ == "__main__":
    main()
