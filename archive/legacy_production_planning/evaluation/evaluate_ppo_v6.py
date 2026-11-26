
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.factory_env_ppo_v6 import FactoryPPOV6Env


MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v6.zip"
DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
REPORT_PATH = BASE_DIR / "reports" / "ppo_v6_evaluation.csv"

EPISODES = 30
SEEDS = range(100, 130)


def mask_fn(env):
    return env.action_masks()


def evaluate():

    model = MaskablePPO.load(str(MODEL_PATH))

    results = []

    for episode, seed in enumerate(SEEDS, start=1):

        env = FactoryPPOV6Env(
            data_path=str(DATA_PATH),
            episode_length=30
        )

        env = ActionMasker(env, mask_fn)

        obs, info = env.reset(seed=seed)

        total_profit = 0
        total_sales = np.zeros(3)
        total_demand = np.zeros(3)
        total_production = np.zeros(3)
        total_holding_cost = 0
        valid_action_counts = []

        terminated = False
        truncated = False

        while not (terminated or truncated):

            valid_action_counts.append(
                int(env.action_masks().sum())
            )

            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                int(action)
            )

            total_profit += info["daily_profit"]
            total_sales += info["sales"]
            total_demand += info["demand"]
            total_production += info["production"]
            total_holding_cost += info["holding_cost"]

        fulfillment = (
            total_sales.sum() / total_demand.sum() * 100
            if total_demand.sum() > 0 else 0
        )

        results.append({
            "episode": episode,
            "seed": seed,
            "profit": total_profit,
            "demand_fulfillment": fulfillment,
            "holding_cost": total_holding_cost,
            "average_valid_actions": np.mean(valid_action_counts),
            "production_A": total_production[0],
            "production_B": total_production[1],
            "production_C": total_production[2]
        })

        env.close()

        print(
            f"Episode {episode:02d} | "
            f"Profit: ₹{total_profit:,.0f} | "
            f"Fulfillment: {fulfillment:.2f}%"
        )

    df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)

    print("\n========== PPO V6 EVALUATION ==========")
    print(f"Average profit: ₹{df['profit'].mean():,.2f}")
    print(f"Best profit: ₹{df['profit'].max():,.2f}")
    print(f"Worst profit: ₹{df['profit'].min():,.2f}")
    print(
        f"Demand fulfillment: "
        f"{df['demand_fulfillment'].mean():.2f}%"
    )
    print(
        f"Average holding cost: "
        f"₹{df['holding_cost'].mean():,.2f}"
    )
    print(
        f"Average valid actions: "
        f"{df['average_valid_actions'].mean():.2f}"
    )

    print("\nAverage daily production:")

    for product in ["A", "B", "C"]:
        print(
            f"Product {product}: "
            f"{df[f'production_{product}'].mean() / 30:.2f}"
        )

    print("\nReport saved:", REPORT_PATH)


if __name__ == "__main__":
    evaluate()
