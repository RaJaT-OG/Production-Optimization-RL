
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.factory_env_ppo_v4 import FactoryPPOV4Env

DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v4.zip"


def evaluate_policy(model, deterministic):
    env = FactoryPPOV4Env(
        data_path=str(DATA_PATH),
        episode_length=30
    )

    results = []

    for seed in range(100, 130):
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False

        episode_profit = 0
        total_sales = np.zeros(3)
        total_demand = np.zeros(3)
        total_production = np.zeros(3)

        while not (terminated or truncated):
            mask = env.action_masks()

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask
            )

            obs, reward, terminated, truncated, info = env.step(
                int(action)
            )

            episode_profit += info["daily_profit"]

            total_sales += info["sales"]
            total_demand += info["demand"]
            total_production += info["production"]

        fulfillment = (
            total_sales.sum() / total_demand.sum()
        ) * 100

        results.append({
            "seed": seed,
            "profit": episode_profit,
            "fulfillment": fulfillment,
            "production_A": total_production[0],
            "production_B": total_production[1],
            "production_C": total_production[2]
        })

    env.close()
    return pd.DataFrame(results)


def print_summary(name, df):
    print(f"\n========== {name} RESULTS ==========")

    print(f"Average profit: ₹{df['profit'].mean():,.2f}")
    print(f"Best profit: ₹{df['profit'].max():,.2f}")
    print(f"Worst profit: ₹{df['profit'].min():,.2f}")

    print(
        f"Average demand fulfillment: "
        f"{df['fulfillment'].mean():.2f}%"
    )

    print(
        f"Average daily Product A production: "
        f"{df['production_A'].mean() / 30:.2f}"
    )

    print(
        f"Average daily Product B production: "
        f"{df['production_B'].mean() / 30:.2f}"
    )

    print(
        f"Average daily Product C production: "
        f"{df['production_C'].mean() / 30:.2f}"
    )


def main():
    print("Loading PPO V4...")
    model = MaskablePPO.load(str(MODEL_PATH))

    print("\nEvaluating deterministic policy...")
    deterministic_df = evaluate_policy(model, True)

    print("\nEvaluating stochastic policy...")
    stochastic_df = evaluate_policy(model, False)

    print_summary("DETERMINISTIC PPO V4", deterministic_df)
    print_summary("STOCHASTIC PPO V4", stochastic_df)

    output_dir = BASE_DIR / "reports"
    output_dir.mkdir(exist_ok=True)

    deterministic_df.to_csv(
        output_dir / "ppo_v4_deterministic_comparison.csv",
        index=False
    )

    stochastic_df.to_csv(
        output_dir / "ppo_v4_stochastic_comparison.csv",
        index=False
    )

    print("\nReports saved in:", output_dir)


if __name__ == "__main__":
    main()
