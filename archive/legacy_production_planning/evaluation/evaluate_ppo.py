
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from environment.factory_env_ppo import FactoryPPOEnv


def main():

    print("\n========== PPO V2 MODEL EVALUATION ==========")

    # Load PPO V2 model
    model_path = ROOT_DIR / "models" / "ppo_production_optimizer_v2.zip"

    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO V2 model not found: {model_path}\n"
            "Make sure PPO V2 training completed successfully."
        )

    model = PPO.load(str(model_path), device="cpu")

    results = []

    total_production = np.zeros(3)
    total_sales = np.zeros(3)
    total_demand = np.zeros(3)

    invalid_actions = 0
    total_holding_cost = 0

    for episode in range(30):

        env = FactoryPPOEnv(episode_length=30)
        state, info = env.reset(seed=episode + 100)

        episode_profit = 0
        episode_steps = 0

        while True:

            action, _ = model.predict(
                state,
                deterministic=True
            )

            state, reward, terminated, truncated, info = env.step(action)

            episode_profit += info["daily_profit"]
            episode_steps += 1

            total_production += info["production"]
            total_sales += info["sales"]
            total_demand += info["demand"]

            total_holding_cost += info["holding_cost"]

            if not info["feasible"]:
                invalid_actions += 1

            if terminated or truncated:
                break

        results.append({
            "Episode": episode + 1,
            "Profit": episode_profit,
            "Steps": episode_steps
        })

        print(
            f"Episode {episode + 1:02d} | "
            f"Profit: Rs. {episode_profit:,.2f}"
        )

        env.close()

    df = pd.DataFrame(results)

    average_profit = df["Profit"].mean()
    best_profit = df["Profit"].max()
    worst_profit = df["Profit"].min()

    fulfillment = (
        total_sales.sum() / total_demand.sum()
    ) * 100

    print("\n========== PPO V2 EVALUATION SUMMARY ==========")

    print(f"Average Profit: Rs. {average_profit:,.2f}")
    print(f"Best Episode Profit: Rs. {best_profit:,.2f}")
    print(f"Worst Episode Profit: Rs. {worst_profit:,.2f}")

    print(f"Demand Fulfillment: {fulfillment:.2f}%")
    print(f"Invalid Actions: {invalid_actions}")
    print(
        f"Average Holding Cost per Episode: "
        f"Rs. {total_holding_cost / 30:,.2f}"
    )

    print("\nTotal Production:")
    print("Product A:", total_production[0])
    print("Product B:", total_production[1])
    print("Product C:", total_production[2])

    print("\nAverage Daily Production:")
    print("Product A:", total_production[0] / 900)
    print("Product B:", total_production[1] / 900)
    print("Product C:", total_production[2] / 900)

    print("\nTotal Sales:", total_sales)
    print("Total Demand:", total_demand)

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "ppo_v2_evaluation.csv"
    df.to_csv(output_path, index=False)

    print("\nEpisode results saved to:", output_path)
    print("\n========== PPO V2 EVALUATION COMPLETED ==========")


if __name__ == "__main__":
    main()
