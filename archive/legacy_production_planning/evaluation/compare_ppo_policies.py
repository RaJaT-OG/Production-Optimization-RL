
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO

from environment.factory_env import FactoryEnv
from environment.factory_env_ppo import FactoryPPOEnv


# ==========================================
# 1. GREEDY ACTION SELECTION
# ==========================================

def choose_greedy_action(env):

    best_action = 0
    best_profit = float("-inf")

    for action in range(env.action_space.n):

        production = env.decode_action(action)

        material_used = np.dot(
            production, env.material_required
        )

        hours_used = np.dot(
            production, env.hours_required
        )

        production_cost = np.dot(
            production, env.costs
        )

        available_stock = env.inventory + production

        sales = np.minimum(
            available_stock, env.daily_demand
        )

        ending_inventory = available_stock - sales

        feasible = (
            material_used <= env.raw_material
            and hours_used <= env.machine_hours
            and production_cost <= env.budget
            and ending_inventory.sum() <= env.storage_capacity
        )

        if not feasible:
            continue

        revenue = np.dot(sales, env.prices)

        holding_cost = (
            ending_inventory.sum() * env.holding_cost
        )

        daily_profit = (
            revenue - production_cost - holding_cost
        )

        if daily_profit > best_profit:
            best_profit = daily_profit
            best_action = action

    return best_action


# ==========================================
# 2. EVALUATE ONE POLICY
# ==========================================

def evaluate_policy(env, policy, seed):

    state, info = env.reset(seed=seed)

    total_profit = 0
    total_demand = np.zeros(3)
    total_sales = np.zeros(3)
    total_production = np.zeros(3)

    invalid_actions = 0
    holding_cost_total = 0
    action_counts = Counter()

    while True:

        if policy == "PPO":

            action, _ = model.predict(
                state,
                deterministic=True
            )

            action = np.asarray(action, dtype=int)

            action_counts[tuple(action.tolist())] += 1

        else:

            action = choose_greedy_action(env)

            production = env.decode_action(action)

            action_counts[tuple(production.tolist())] += 1

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_profit += info["daily_profit"]

        total_demand += info["demand"]
        total_sales += info["sales"]
        total_production += info["production"]

        holding_cost_total += info["holding_cost"]

        if not info["feasible"]:
            invalid_actions += 1

        if terminated or truncated:
            break

    fulfillment = (
        total_sales.sum() / total_demand.sum() * 100
        if total_demand.sum() > 0 else 0
    )

    env.close()

    return {
        "profit": total_profit,
        "fulfillment": fulfillment,
        "production": total_production,
        "sales": total_sales,
        "demand": total_demand,
        "invalid_actions": invalid_actions,
        "holding_cost": holding_cost_total,
        "action_counts": action_counts
    }


# ==========================================
# 3. MAIN COMPARISON
# ==========================================

def main():

    global model

    print("\n========== PPO V2 VS GREEDY COMPARISON ==========")

    model_path = (
        ROOT_DIR / "models" / "ppo_production_optimizer_v2.zip"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found: {model_path}"
        )

    model = PPO.load(str(model_path), device="cpu")

    episodes = 30

    ppo_results = []
    greedy_results = []

    ppo_actions = Counter()
    greedy_actions = Counter()

    ppo_production = np.zeros(3)
    greedy_production = np.zeros(3)

    ppo_sales = np.zeros(3)
    greedy_sales = np.zeros(3)

    ppo_invalid = 0
    greedy_invalid = 0

    ppo_holding = 0
    greedy_holding = 0

    # Same seeds for both policies
    for episode in range(episodes):

        seed = episode + 100

        ppo_env = FactoryPPOEnv(episode_length=30)
        greedy_env = FactoryEnv(episode_length=30)

        ppo = evaluate_policy(
            ppo_env, "PPO", seed
        )

        greedy = evaluate_policy(
            greedy_env, "Greedy", seed
        )

        ppo_results.append({
            "Episode": episode + 1,
            "Profit": ppo["profit"],
            "Fulfillment": ppo["fulfillment"]
        })

        greedy_results.append({
            "Episode": episode + 1,
            "Profit": greedy["profit"],
            "Fulfillment": greedy["fulfillment"]
        })

        ppo_production += ppo["production"]
        greedy_production += greedy["production"]

        ppo_sales += ppo["sales"]
        greedy_sales += greedy["sales"]

        ppo_invalid += ppo["invalid_actions"]
        greedy_invalid += greedy["invalid_actions"]

        ppo_holding += ppo["holding_cost"]
        greedy_holding += greedy["holding_cost"]

        ppo_actions.update(ppo["action_counts"])
        greedy_actions.update(greedy["action_counts"])

        print(
            f"Episode {episode + 1:02d} | "
            f"PPO: Rs.{ppo['profit']:,.2f} | "
            f"Greedy: Rs.{greedy['profit']:,.2f} | "
            f"Gap: Rs.{greedy['profit'] - ppo['profit']:,.2f}"
        )

    # ======================================
    # 4. PROFIT COMPARISON
    # ======================================

    ppo_df = pd.DataFrame(ppo_results)
    greedy_df = pd.DataFrame(greedy_results)

    ppo_avg = ppo_df["Profit"].mean()
    greedy_avg = greedy_df["Profit"].mean()

    print("\n========== PROFIT COMPARISON ==========")

    print(f"PPO Average Profit: Rs.{ppo_avg:,.2f}")
    print(f"Greedy Average Profit: Rs.{greedy_avg:,.2f}")

    print(
        f"Average Profit Gap: Rs.{greedy_avg - ppo_avg:,.2f}"
    )

    if greedy_avg != 0:
        gap_percent = (
            (greedy_avg - ppo_avg) / greedy_avg
        ) * 100

        print(f"PPO Below Greedy: {gap_percent:.2f}%")

    print(
        f"PPO Average Fulfillment: "
        f"{ppo_df['Fulfillment'].mean():.2f}%"
    )

    print(
        f"Greedy Average Fulfillment: "
        f"{greedy_df['Fulfillment'].mean():.2f}%"
    )

    # ======================================
    # 5. PRODUCTION COMPARISON
    # ======================================

    print("\n========== PRODUCTION COMPARISON ==========")

    products = ["A", "B", "C"]

    for i, product in enumerate(products):

        print(
            f"Product {product} | "
            f"PPO: {ppo_production[i] / 900:.2f}/day | "
            f"Greedy: {greedy_production[i] / 900:.2f}/day"
        )

    print("\nTotal PPO Production:", ppo_production)
    print("Total Greedy Production:", greedy_production)

    print("\nPPO Total Sales:", ppo_sales)
    print("Greedy Total Sales:", greedy_sales)

    # ======================================
    # 6. INVALID ACTIONS AND HOLDING COST
    # ======================================

    print("\n========== CONSTRAINT ANALYSIS ==========")

    print("PPO Invalid Actions:", ppo_invalid)
    print("Greedy Invalid Actions:", greedy_invalid)

    print(
        "PPO Average Daily Holding Cost:",
        round(ppo_holding / 900, 2)
    )

    print(
        "Greedy Average Daily Holding Cost:",
        round(greedy_holding / 900, 2)
    )

    # ======================================
    # 7. ACTION DISTRIBUTION
    # ======================================

    print("\n========== PPO TOP 10 ACTIONS ==========")

    for action, count in ppo_actions.most_common(10):

        production = np.array(action) * 10

        print(
            f"Production {production} | "
            f"Selected {count} times"
        )

    print("\n========== GREEDY TOP 10 ACTIONS ==========")

    for action, count in greedy_actions.most_common(10):

        print(
            f"Production {action} | "
            f"Selected {count} times"
        )

    # ======================================
    # 8. SAVE REPORTS
    # ======================================

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    comparison_df = pd.DataFrame({
        "Episode": range(1, episodes + 1),
        "PPO_Profit": ppo_df["Profit"],
        "Greedy_Profit": greedy_df["Profit"],
        "PPO_Fulfillment": ppo_df["Fulfillment"],
        "Greedy_Fulfillment": greedy_df["Fulfillment"]
    })

    comparison_df.to_csv(
        reports_dir / "ppo_v2_vs_greedy.csv",
        index=False
    )

    ppo_action_df = pd.DataFrame([
        {
            "Production_A": action[0] * 10,
            "Production_B": action[1] * 10,
            "Production_C": action[2] * 10,
            "Count": count
        }
        for action, count in ppo_actions.most_common()
    ])

    ppo_action_df.to_csv(
        reports_dir / "ppo_v2_action_distribution.csv",
        index=False
    )

    greedy_action_df = pd.DataFrame([
        {
            "Production_A": action[0],
            "Production_B": action[1],
            "Production_C": action[2],
            "Count": count
        }
        for action, count in greedy_actions.most_common()
    ])

    greedy_action_df.to_csv(
        reports_dir / "greedy_action_distribution_v2.csv",
        index=False
    )

    print("\nReports saved in:", reports_dir)

    print("\n========== COMPARISON COMPLETED ==========")


if __name__ == "__main__":
    main()
