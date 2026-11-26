
import numpy as np
import pandas as pd

from pathlib import Path
from stable_baselines3 import PPO

from environment.factory_env_ppo_v3 import FactoryPPOV3Env


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v3.zip"
REPORT_PATH = BASE_DIR / "reports" / "ppo_v3_evaluation.csv"

EPISODES = 30
SEEDS = range(100, 130)


def main():

    print("\n========== PPO V3 EVALUATION ==========")

    env = FactoryPPOV3Env(episode_length=30)
    model = PPO.load(str(MODEL_PATH))

    episode_results = []

    total_production = np.zeros(3)
    total_sales = np.zeros(3)
    total_demand = np.zeros(3)

    total_adjusted_actions = 0
    total_holding_cost = 0.0
    total_profit = 0.0

    for episode, seed in enumerate(SEEDS, start=1):

        obs, info = env.reset(seed=seed)

        episode_profit = 0.0
        episode_sales = np.zeros(3)
        episode_demand = np.zeros(3)
        episode_production = np.zeros(3)

        episode_holding_cost = 0.0
        episode_adjusted = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_profit += info["daily_profit"]

            episode_production += info["production"]
            episode_sales += info["sales"]
            episode_demand += info["demand"]

            episode_holding_cost += info["holding_cost"]

            if info["action_adjusted"]:
                episode_adjusted += 1

        fulfillment = (
            episode_sales.sum() / episode_demand.sum() * 100
            if episode_demand.sum() > 0 else 0
        )

        episode_results.append({
            "episode": episode,
            "seed": seed,
            "profit": episode_profit,
            "demand_fulfillment": fulfillment,
            "holding_cost": episode_holding_cost,
            "adjusted_actions": episode_adjusted,
            "production_A": episode_production[0],
            "production_B": episode_production[1],
            "production_C": episode_production[2]
        })

        total_profit += episode_profit
        total_production += episode_production
        total_sales += episode_sales
        total_demand += episode_demand

        total_holding_cost += episode_holding_cost
        total_adjusted_actions += episode_adjusted

        print(
            f"Episode {episode:02d} | "
            f"Profit: ₹{episode_profit:,.2f} | "
            f"Fulfillment: {fulfillment:.2f}% | "
            f"Adjusted actions: {episode_adjusted}"
        )

    results_df = pd.DataFrame(episode_results)
    results_df.to_csv(REPORT_PATH, index=False)

    average_profit = total_profit / EPISODES

    overall_fulfillment = (
        total_sales.sum() / total_demand.sum() * 100
    )

    print("\n========== FINAL PPO V3 RESULTS ==========")

    print(f"Average profit: ₹{average_profit:,.2f}")
    print(
        f"Best episode: ₹{results_df['profit'].max():,.2f}"
    )
    print(
        f"Worst episode: ₹{results_df['profit'].min():,.2f}"
    )

    print(f"Demand fulfillment: {overall_fulfillment:.2f}%")

    print(
        f"Total adjusted actions: {total_adjusted_actions}"
    )

    print(
        f"Average holding cost per episode: "
        f"₹{total_holding_cost / EPISODES:,.2f}"
    )

    print("\nAverage daily production:")

    print("Product A:", round(total_production[0] / (EPISODES * 30), 2))
    print("Product B:", round(total_production[1] / (EPISODES * 30), 2))
    print("Product C:", round(total_production[2] / (EPISODES * 30), 2))

    print("\nTotal sales:", total_sales)
    print("Total demand:", total_demand)

    print("\nReport saved at:", REPORT_PATH)
    print("==========================================")

    env.close()


if __name__ == "__main__":
    main()
