
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

from sb3_contrib import MaskablePPO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.factory_env_ppo_v5 import FactoryPPOV5Env
from baseline.baseline_strategy import choose_action

MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v5.zip"
DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"

ACTION_REPORT_PATH = (
    BASE_DIR / "reports" / "ppo_v5_action_distribution.csv"
)
INVENTORY_REPORT_PATH = (
    BASE_DIR / "reports" / "ppo_v5_inventory_trajectory.csv"
)

EPISODES = 30
EPISODE_LENGTH = 30
SEEDS = range(100, 130)


# ==========================================
# 1. IMMEDIATE PROFIT OF A PRODUCTION PLAN
# ==========================================

def immediate_profit(env, production):
    """
    Profit a feasible production plan would earn today,
    without advancing the environment.
    """

    available_stock = env.inventory + production

    sales = np.minimum(
        available_stock, env.daily_demand
    )

    ending_inventory = available_stock - sales

    revenue = np.dot(sales, env.prices)
    production_cost = np.dot(production, env.costs)
    holding_cost = (
        ending_inventory.sum() * env.holding_cost
    )

    return revenue - production_cost - holding_cost


# ==========================================
# 2. RUN DIAGNOSTIC
# ==========================================

def main():

    print("\n========== PPO V5 POLICY DIAGNOSTIC ==========")
    print("Loading model:", MODEL_PATH)

    model = MaskablePPO.load(str(MODEL_PATH))

    env = FactoryPPOV5Env(
        data_path=str(DATA_PATH),
        episode_length=EPISODE_LENGTH
    )

    total_days = EPISODES * EPISODE_LENGTH

    action_counts = Counter()

    # Per-day accumulators (averaged across episodes later)
    inventory_by_day = np.zeros((EPISODE_LENGTH, 3))
    demand_by_day = np.zeros((EPISODE_LENGTH, 3))
    sales_by_day = np.zeros((EPISODE_LENGTH, 3))
    production_by_day = np.zeros((EPISODE_LENGTH, 3))
    profit_by_day = np.zeros(EPISODE_LENGTH)

    # Per-product totals
    total_demand = np.zeros(3)
    total_sales = np.zeros(3)
    total_production = np.zeros(3)

    # Resource utilisation
    total_material_used = 0.0
    total_material_available = 0.0
    total_hours_used = 0.0
    total_hours_available = 0.0
    total_production_cost = 0.0
    total_budget_available = 0.0

    # Policy behaviour
    zero_production_days = 0
    full_storage_days = 0
    greedy_match_days = 0
    profit_gap_total = 0.0
    mismatch_days = 0

    for seed in SEEDS:

        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False
        day_index = 0

        while not (terminated or truncated):

            mask = env.action_masks()

            # Greedy-optimal action for the same state
            greedy_action = choose_action(env)

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=mask
            )

            action = int(action)
            action_counts[action] += 1

            if action == greedy_action:
                greedy_match_days += 1
            else:
                mismatch_days += 1

                greedy_profit = immediate_profit(
                    env,
                    env.decode_action(greedy_action)
                )

            # Today's available resources (before stepping)
            total_material_available += env.raw_material
            total_hours_available += env.machine_hours
            total_budget_available += env.budget

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            day = min(day_index, EPISODE_LENGTH - 1)

            inventory_by_day[day] += info["inventory"]
            demand_by_day[day] += info["demand"]
            sales_by_day[day] += info["sales"]
            production_by_day[day] += info["production"]
            profit_by_day[day] += info["daily_profit"]

            total_demand += info["demand"]
            total_sales += info["sales"]
            total_production += info["production"]

            total_material_used += info["material_used"]
            total_hours_used += info["machine_hours_used"]
            total_production_cost += info["production_cost"]

            if info["production"].sum() == 0:
                zero_production_days += 1

            if info["inventory"].sum() >= env.storage_capacity:
                full_storage_days += 1

            if action != greedy_action:
                profit_gap_total += (
                    greedy_profit - info["daily_profit"]
                )

            day_index += 1

    env.close()

    # ==========================================
    # 3. ACTION FREQUENCY
    # ==========================================

    print("\n---------- ACTION FREQUENCY ----------")
    print("Total decisions:", total_days)
    print("Unique actions used:", len(action_counts))

    top_1 = action_counts.most_common(1)[0][1]
    top_5 = sum(c for _, c in action_counts.most_common(5))
    top_10 = sum(c for _, c in action_counts.most_common(10))

    print(
        "Top 1 action share:",
        f"{top_1 / total_days * 100:.2f}%"
    )
    print(
        "Top 5 actions share:",
        f"{top_5 / total_days * 100:.2f}%"
    )
    print(
        "Top 10 actions share:",
        f"{top_10 / total_days * 100:.2f}%"
    )

    print("\nTop 15 requested production combinations:")

    for action, count in action_counts.most_common(15):

        production = env.decode_action(action)

        print(
            f"Action {action:3d} | "
            f"Production ({production[0]:.0f}, "
            f"{production[1]:.0f}, {production[2]:.0f}) | "
            f"Selected {count:3d} times "
            f"({count / total_days * 100:.2f}%)"
        )

    # ==========================================
    # 4. AGREEMENT WITH GREEDY
    # ==========================================

    print("\n---------- GREEDY AGREEMENT ----------")

    print(
        "Days matching greedy-optimal action:",
        f"{greedy_match_days}/{total_days} "
        f"({greedy_match_days / total_days * 100:.2f}%)"
    )

    print(
        "Total profit lost vs greedy choices:",
        f"Rs.{profit_gap_total:,.2f}"
    )

    if mismatch_days > 0:
        print(
            "Average profit lost per mismatch day:",
            f"Rs.{profit_gap_total / mismatch_days:,.2f}"
        )

    # ==========================================
    # 5. PER-PRODUCT BALANCE
    # ==========================================

    print("\n---------- PER-PRODUCT BALANCE ----------")

    for i, product in enumerate(["A", "B", "C"]):

        avg_demand = total_demand[i] / total_days
        avg_production = total_production[i] / total_days
        avg_sales = total_sales[i] / total_days

        fulfillment = (
            total_sales[i] / total_demand[i] * 100
            if total_demand[i] > 0 else 0
        )

        print(
            f"Product {product} | "
            f"Demand {avg_demand:5.2f}/day | "
            f"Production {avg_production:5.2f}/day | "
            f"Sales {avg_sales:5.2f}/day | "
            f"Fulfillment {fulfillment:5.2f}%"
        )

    # ==========================================
    # 6. INVENTORY TRAJECTORY
    # ==========================================

    print("\n---------- INVENTORY TRAJECTORY ----------")

    print(
        "Days with zero production:",
        f"{zero_production_days}/{total_days} "
        f"({zero_production_days / total_days * 100:.2f}%)"
    )

    print(
        "Days ending at full storage capacity:",
        f"{full_storage_days}/{total_days} "
        f"({full_storage_days / total_days * 100:.2f}%)"
    )

    print("\nAverage end-of-day inventory across episodes:")
    print("Day |   Inv A |   Inv B |   Inv C |  Daily profit")

    for day in range(EPISODE_LENGTH):

        inv_a = inventory_by_day[day, 0] / EPISODES
        inv_b = inventory_by_day[day, 1] / EPISODES
        inv_c = inventory_by_day[day, 2] / EPISODES
        profit = profit_by_day[day] / EPISODES

        print(
            f"{day + 1:3d} | "
            f"{inv_a:7.2f} | "
            f"{inv_b:7.2f} | "
            f"{inv_c:7.2f} | "
            f"Rs.{profit:10,.2f}"
        )

    # ==========================================
    # 7. RESOURCE UTILISATION
    # ==========================================

    print("\n---------- RESOURCE UTILISATION ----------")

    material_util = (
        total_material_used
        / total_material_available * 100
    )

    hours_util = (
        total_hours_used
        / total_hours_available * 100
    )

    budget_util = (
        total_production_cost
        / total_budget_available * 100
    )

    print(f"Raw material utilisation: {material_util:.2f}%")
    print(f"Machine hours utilisation: {hours_util:.2f}%")
    print(f"Budget utilisation: {budget_util:.2f}%")

    # ==========================================
    # 8. SAVE REPORTS
    # ==========================================

    action_rows = []

    for action, count in action_counts.most_common():

        production = env.decode_action(action)

        action_rows.append({
            "action": action,
            "production_A": production[0],
            "production_B": production[1],
            "production_C": production[2],
            "count": count,
            "frequency": count / total_days * 100
        })

    action_df = pd.DataFrame(action_rows)

    ACTION_REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    action_df.to_csv(ACTION_REPORT_PATH, index=False)

    trajectory_df = pd.DataFrame({
        "day": range(1, EPISODE_LENGTH + 1),
        "avg_inventory_A": inventory_by_day[:, 0] / EPISODES,
        "avg_inventory_B": inventory_by_day[:, 1] / EPISODES,
        "avg_inventory_C": inventory_by_day[:, 2] / EPISODES,
        "avg_demand_A": demand_by_day[:, 0] / EPISODES,
        "avg_demand_B": demand_by_day[:, 1] / EPISODES,
        "avg_demand_C": demand_by_day[:, 2] / EPISODES,
        "avg_sales_A": sales_by_day[:, 0] / EPISODES,
        "avg_sales_B": sales_by_day[:, 1] / EPISODES,
        "avg_sales_C": sales_by_day[:, 2] / EPISODES,
        "avg_production_A": production_by_day[:, 0] / EPISODES,
        "avg_production_B": production_by_day[:, 1] / EPISODES,
        "avg_production_C": production_by_day[:, 2] / EPISODES,
        "avg_daily_profit": profit_by_day / EPISODES
    })

    trajectory_df.to_csv(
        INVENTORY_REPORT_PATH, index=False
    )

    print("\nAction report saved:", ACTION_REPORT_PATH)
    print("Inventory report saved:", INVENTORY_REPORT_PATH)
    print("\n==============================================")


if __name__ == "__main__":
    main()
