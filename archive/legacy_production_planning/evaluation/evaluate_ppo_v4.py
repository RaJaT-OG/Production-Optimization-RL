
import numpy as np
import pandas as pd

from pathlib import Path
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from environment.factory_env_ppo_v4 import FactoryPPOV4Env


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v4.zip"
REPORT_PATH = BASE_DIR / "reports" / "ppo_v4_evaluation.csv"

EPISODES = 30
SEEDS = range(100, 130)


def mask_fn(env):
    return env.action_masks()


def main():

    print("\n========== PPO V4 EVALUATION ==========")

    base_env = FactoryPPOV4Env(
        data_path=str(BASE_DIR / "data" / "raw" / "production_data.csv"),
        episode_length=30
    )

    env = ActionMasker(base_env, mask_fn)

    model = MaskablePPO.load(str(MODEL_PATH))

    episode_results = []

    total_production = np.zeros(3)
    total_sales = np.zeros(3)
    total_demand = np.zeros(3)

    total_holding_cost = 0.0
    total_profit = 0.0
    total_valid_actions = 0
    total_decisions = 0

    for episode, seed in enumerate(SEEDS, start=1):

        obs, info = env.reset(seed=seed)

        episode_profit = 0.0
        episode_sales = np.zeros(3)
        episode_demand = np.zeros(3)
        episode_production = np.zeros(3)

        episode_holding_cost = 0.0
        episode_valid_actions = 0
        episode_decisions = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Get currently feasible actions.
            action_mask = env.action_masks()

            # Select only from feasible actions.
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_mask
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_profit += info["daily_profit"]

            episode_production += info["production"]
            episode_sales += info["sales"]
            episode_demand += info["demand"]

            episode_holding_cost += info["holding_cost"]

            episode_valid_actions += int(action_mask.sum())
            episode_decisions += 1

        fulfillment = (
            episode_sales.sum() / episode_demand.sum() * 100
            if episode_demand.sum() > 0 else 0
        )

        average_valid_actions = (
            episode_valid_actions / episode_decisions
            if episode_decisions > 0 else 0
        )

        episode_results.append({
            "episode": episode,
            "seed": seed,
            "profit": episode_profit,
            "demand_fulfillment": fulfillment,
            "holding_cost": episode_holding_cost,
            "average_valid_actions": average_valid_actions,
            "production_A": episode_production[0],
            "production_B": episode_production[1],
            "production_C": episode_production[2]
        })

        total_profit += episode_profit
        total_production += episode_production
        total_sales += episode_sales
        total_demand += episode_demand

        total_holding_cost += episode_holding_cost
        total_valid_actions += episode_valid_actions
        total_decisions += episode_decisions

        print(
            f"Episode {episode:02d} | "
            f"Profit: Rs.{episode_profit:,.2f} | "
            f"Fulfillment: {fulfillment:.2f}% | "
            f"Avg valid actions: {average_valid_actions:.1f}"
        )

    results_df = pd.DataFrame(episode_results)
    results_df.to_csv(REPORT_PATH, index=False)

    average_profit = total_profit / EPISODES

    overall_fulfillment = (
        total_sales.sum() / total_demand.sum() * 100
    )

    print("\n========== FINAL PPO V4 RESULTS ==========")

    print(f"Average profit: Rs.{average_profit:,.2f}")

    print(
        f"Best episode: Rs.{results_df['profit'].max():,.2f}"
    )

    print(
        f"Worst episode: Rs.{results_df['profit'].min():,.2f}"
    )

    print(f"Demand fulfillment: {overall_fulfillment:.2f}%")

    print(
        f"Average holding cost per episode: "
        f"Rs.{total_holding_cost / EPISODES:,.2f}"
    )

    print(
        f"Average feasible actions per decision: "
        f"{total_valid_actions / total_decisions:.2f}"
    )

    print("\nAverage daily production:")

    print(
        "Product A:",
        round(total_production[0] / (EPISODES * 30), 2)
    )

    print(
        "Product B:",
        round(total_production[1] / (EPISODES * 30), 2)
    )

    print(
        "Product C:",
        round(total_production[2] / (EPISODES * 30), 2)
    )

    print("\nTotal sales:", total_sales)
    print("Total demand:", total_demand)

    print("\nReport saved at:", REPORT_PATH)
    print("==========================================")

    env.close()


if __name__ == "__main__":
    main()
