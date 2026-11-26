import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env import FactoryEnv
from environment.factory_env_ppo_v3 import FactoryPPOV3Env

EPISODES = 30
SEEDS = range(100, 130)


def choose_greedy_action(env):
    best_action = 0
    best_profit = float("-inf")

    for action in range(env.action_space.n):
        production = env.decode_action(action)
        material = np.dot(production, env.material_required)
        hours = np.dot(production, env.hours_required)
        cost = np.dot(production, env.costs)

        stock = env.inventory + production
        sales = np.minimum(stock, env.daily_demand)
        ending_inventory = stock - sales

        feasible = (
            material <= env.raw_material
            and hours <= env.machine_hours
            and cost <= env.budget
            and ending_inventory.sum() <= env.storage_capacity
        )
        if not feasible:
            continue

        profit = (
            np.dot(sales, env.prices)
            - cost
            - ending_inventory.sum() * env.holding_cost
        )
        if profit > best_profit:
            best_profit = profit
            best_action = action

    return best_action


def evaluate_v3(env, model, seed):
    obs, _ = env.reset(seed=seed)
    totals = {
        "profit": 0.0,
        "demand": np.zeros(3),
        "sales": np.zeros(3),
        "production": np.zeros(3),
        "holding": 0.0,
        "adjusted": 0,
    }
    requested_counts = Counter()
    executed_counts = Counter()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        requested = env.decode_action(action)
        requested_counts[tuple(requested.astype(int))] += 1

        obs, _, terminated, truncated, info = env.step(action)
        executed = np.asarray(info["production"], dtype=float)
        executed_counts[tuple(executed.astype(int))] += 1

        totals["profit"] += info["daily_profit"]
        totals["demand"] += info["demand"]
        totals["sales"] += info["sales"]
        totals["production"] += executed
        totals["holding"] += info["holding_cost"]
        totals["adjusted"] += int(info.get("action_adjusted", not info["feasible"]))

        if terminated or truncated:
            break

    totals["fulfillment"] = (
        totals["sales"].sum() / totals["demand"].sum() * 100
        if totals["demand"].sum() else 0.0
    )
    totals["requested_counts"] = requested_counts
    totals["executed_counts"] = executed_counts
    return totals


def evaluate_greedy(env, seed):
    obs, _ = env.reset(seed=seed)
    totals = {
        "profit": 0.0,
        "demand": np.zeros(3),
        "sales": np.zeros(3),
        "production": np.zeros(3),
        "holding": 0.0,
        "invalid": 0,
    }
    actions = Counter()

    while True:
        action = choose_greedy_action(env)
        production = env.decode_action(action)
        actions[tuple(production.astype(int))] += 1

        obs, _, terminated, truncated, info = env.step(action)
        totals["profit"] += info["daily_profit"]
        totals["demand"] += info["demand"]
        totals["sales"] += info["sales"]
        totals["production"] += info["production"]
        totals["holding"] += info["holding_cost"]
        totals["invalid"] += int(not info["feasible"])

        if terminated or truncated:
            break

    totals["fulfillment"] = (
        totals["sales"].sum() / totals["demand"].sum() * 100
        if totals["demand"].sum() else 0.0
    )
    totals["actions"] = actions
    return totals


def save_distribution(counter, path):
    pd.DataFrame([
        {
            "Production_A": action[0],
            "Production_B": action[1],
            "Production_C": action[2],
            "Count": count,
        }
        for action, count in counter.most_common()
    ]).to_csv(path, index=False)


def main():
    model_path = ROOT_DIR / "models" / "ppo_production_optimizer_v3.zip"
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    ppo_rows, greedy_rows = [], []

    ppo_requested, ppo_executed, greedy_actions = Counter(), Counter(), Counter()
    ppo_production = np.zeros(3)
    greedy_production = np.zeros(3)
    ppo_sales = np.zeros(3)
    greedy_sales = np.zeros(3)
    ppo_adjusted = greedy_invalid = 0
    ppo_holding = greedy_holding = 0.0

    print("\\n========== PPO V3 VS GREEDY COMPARISON ==========")

    for episode, seed in enumerate(SEEDS, start=1):
        ppo_env = FactoryPPOV3Env(episode_length=30)
        greedy_env = FactoryEnv(episode_length=30)

        ppo = evaluate_v3(ppo_env, model, seed)
        greedy = evaluate_greedy(greedy_env, seed)
        ppo_env.close()
        greedy_env.close()

        ppo_rows.append({
            "Episode": episode, "Seed": seed,
            "Profit": ppo["profit"],
            "Fulfillment": ppo["fulfillment"],
            "Adjusted_Actions": ppo["adjusted"],
            "Holding_Cost": ppo["holding"],
        })
        greedy_rows.append({
            "Episode": episode, "Seed": seed,
            "Profit": greedy["profit"],
            "Fulfillment": greedy["fulfillment"],
            "Invalid_Actions": greedy["invalid"],
            "Holding_Cost": greedy["holding"],
        })

        ppo_production += ppo["production"]
        greedy_production += greedy["production"]
        ppo_sales += ppo["sales"]
        greedy_sales += greedy["sales"]
        ppo_adjusted += ppo["adjusted"]
        greedy_invalid += greedy["invalid"]
        ppo_holding += ppo["holding"]
        greedy_holding += greedy["holding"]
        ppo_requested.update(ppo["requested_counts"])
        ppo_executed.update(ppo["executed_counts"])
        greedy_actions.update(greedy["actions"])

        print(
            f"Episode {episode:02d} | PPO V3: ₹{ppo['profit']:,.2f} | "
            f"Greedy: ₹{greedy['profit']:,.2f} | "
            f"Difference: ₹{ppo['profit'] - greedy['profit']:,.2f}"
        )

    ppo_df = pd.DataFrame(ppo_rows)
    greedy_df = pd.DataFrame(greedy_rows)
    ppo_avg = ppo_df["Profit"].mean()
    greedy_avg = greedy_df["Profit"].mean()
    difference = ppo_avg - greedy_avg
    pct = difference / greedy_avg * 100 if greedy_avg else 0.0

    wins = int((ppo_df["Profit"] > greedy_df["Profit"]).sum())
    ties = int((ppo_df["Profit"] == greedy_df["Profit"]).sum())

    print("\\n========== PAIRED COMPARISON ==========")
    print(f"PPO V3 average profit: ₹{ppo_avg:,.2f}")
    print(f"Greedy average profit: ₹{greedy_avg:,.2f}")
    print(f"PPO V3 minus Greedy: ₹{difference:,.2f} ({pct:.2f}%)")
    print(f"PPO V3 average fulfillment: {ppo_df['Fulfillment'].mean():.2f}%")
    print(f"Greedy average fulfillment: {greedy_df['Fulfillment'].mean():.2f}%")
    print(f"Episode wins | PPO V3: {wins} | Greedy: {EPISODES - wins - ties} | Ties: {ties}")

    print("\\n========== PRODUCTION COMPARISON ==========")
    for i, product in enumerate(["A", "B", "C"]):
        print(
            f"Product {product} average/day | "
            f"PPO V3: {ppo_production[i] / 900:.2f} | "
            f"Greedy: {greedy_production[i] / 900:.2f}"
        )
    print("PPO V3 total production:", ppo_production)
    print("Greedy total production:", greedy_production)
    print("PPO V3 total sales:", ppo_sales)
    print("Greedy total sales:", greedy_sales)

    print("\\n========== CONSTRAINT ANALYSIS ==========")
    print(f"PPO V3 adjusted actions: {ppo_adjusted}/900")
    print(f"Greedy invalid actions: {greedy_invalid}")
    print(f"PPO V3 average daily holding cost: ₹{ppo_holding / 900:.2f}")
    print(f"Greedy average daily holding cost: ₹{greedy_holding / 900:.2f}")

    print("\\nTop PPO V3 requested actions:")
    for action, count in ppo_requested.most_common(10):
        print(f"{action}: {count}")

    print("\\nTop PPO V3 executed actions:")
    for action, count in ppo_executed.most_common(10):
        print(f"{action}: {count}")

    print("\\nTop Greedy actions:")
    for action, count in greedy_actions.most_common(10):
        print(f"{action}: {count}")

    comparison = pd.DataFrame({
        "Episode": range(1, EPISODES + 1),
        "Seed": list(SEEDS),
        "PPO_V3_Profit": ppo_df["Profit"],
        "Greedy_Profit": greedy_df["Profit"],
        "Profit_Difference_PPO_Minus_Greedy": ppo_df["Profit"] - greedy_df["Profit"],
        "PPO_V3_Fulfillment": ppo_df["Fulfillment"],
        "Greedy_Fulfillment": greedy_df["Fulfillment"],
        "PPO_V3_Adjusted_Actions": ppo_df["Adjusted_Actions"],
        "Greedy_Invalid_Actions": greedy_df["Invalid_Actions"],
    })

    comparison.to_csv(reports_dir / "ppo_v3_vs_greedy.csv", index=False)
    ppo_df.to_csv(reports_dir / "ppo_v3_paired_episodes.csv", index=False)
    greedy_df.to_csv(reports_dir / "greedy_v3_paired_episodes.csv", index=False)
    save_distribution(ppo_requested, reports_dir / "ppo_v3_requested_action_distribution.csv")
    save_distribution(ppo_executed, reports_dir / "ppo_v3_executed_action_distribution.csv")
    save_distribution(greedy_actions, reports_dir / "greedy_action_distribution_v3.csv")

    print("\\nV3 reports saved in:", reports_dir)
    print("========== COMPARISON COMPLETED ==========")


if __name__ == "__main__":
    main()
