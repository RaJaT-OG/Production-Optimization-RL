
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env import FactoryEnv


# ==========================================
# 1. GREEDY ACTION SELECTION
# ==========================================

def choose_action(env):

    best_action = 0
    best_profit = float("-inf")

    # Check all 125 possible production combinations
    for action in range(env.action_space.n):

        production = env.decode_action(action)

        # Resource consumption
        material_used = np.dot(
            production, env.material_required
        )

        hours_used = np.dot(
            production, env.hours_required
        )

        production_cost = np.dot(
            production, env.costs
        )

        # Calculate sales and ending inventory
        available_stock = env.inventory + production

        sales = np.minimum(
            available_stock, env.daily_demand
        )

        ending_inventory = available_stock - sales

        # Check factory constraints
        feasible = (
            material_used <= env.raw_material
            and hours_used <= env.machine_hours
            and production_cost <= env.budget
            and ending_inventory.sum() <= env.storage_capacity
        )

        if not feasible:
            continue

        # Updated profit calculation (matches FactoryEnv)
        revenue = np.dot(sales, env.prices)

        holding_cost = (
            ending_inventory.sum() * env.holding_cost
        )

        daily_profit = (
            revenue
            - production_cost
            - holding_cost
        )

        if daily_profit > best_profit:

            best_profit = daily_profit
            best_action = action

    return best_action


# ==========================================
# 2. EVALUATE GREEDY BASELINE
# ==========================================

def evaluate_baseline(episodes=30):

    env = FactoryEnv()

    episode_profits = []
    episode_rewards = []
    episode_fulfillment = []

    for episode in range(episodes):

        state, info = env.reset(seed=episode + 100)

        total_profit = 0
        total_reward = 0

        total_demand = 0
        total_sales = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = choose_action(env)

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_profit += info["daily_profit"]
            total_reward += reward

            total_demand += np.sum(info["demand"])
            total_sales += np.sum(info["sales"])

        fulfillment = (
            total_sales / total_demand * 100
            if total_demand > 0 else 0
        )

        episode_profits.append(total_profit)
        episode_rewards.append(total_reward)
        episode_fulfillment.append(fulfillment)

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Profit: Rs.{total_profit:.2f} | "
            f"Fulfillment: {fulfillment:.2f}%"
        )

    # ======================================
    # 3. FINAL RESULTS
    # ======================================

    print("\n========== UPDATED BASELINE RESULTS ==========")

    print(
        "Average Profit:",
        round(np.mean(episode_profits), 2)
    )

    print(
        "Best Episode Profit:",
        round(np.max(episode_profits), 2)
    )

    print(
        "Worst Episode Profit:",
        round(np.min(episode_profits), 2)
    )

    print(
        "Average Reward:",
        round(np.mean(episode_rewards), 2)
    )

    print(
        "Average Demand Fulfillment:",
        round(np.mean(episode_fulfillment), 2),
        "%"
    )

    print("==============================================")

    env.close()


if __name__ == "__main__":

    evaluate_baseline(episodes=30)
