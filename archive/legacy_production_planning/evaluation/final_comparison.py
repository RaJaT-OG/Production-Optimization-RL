
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.factory_env import FactoryEnv
from environment.factory_env_ppo_v3 import FactoryPPOV3Env
from environment.factory_env_ppo_v6 import FactoryPPOV6Env
from baseline.baseline_strategy import choose_action

DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"

DQN_MODEL_PATH = BASE_DIR / "models" / "dqn_production_optimizer_v2.zip"
PPO_V3_MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v3.zip"
PPO_V6_MODEL_PATH = BASE_DIR / "models" / "ppo_production_optimizer_v6.zip"

REPORT_PATH = BASE_DIR / "reports" / "final_comparison.csv"
EPISODES_PATH = BASE_DIR / "reports" / "final_comparison_episodes.csv"
FIGURE_PATH = BASE_DIR / "reports" / "figures" / "final_comparison.png"

EPISODES = 30
SEEDS = range(100, 130)


# ==========================================
# 1. GENERIC EPISODE RUNNER
# ==========================================

def run_episode(env, action_fn):
    """
    Run one episode with a policy that returns an action
    for the current observation. Returns per-episode totals.
    """

    total_profit = 0
    total_holding_cost = 0

    total_sales = np.zeros(3)
    total_demand = np.zeros(3)
    total_production = np.zeros(3)

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = action_fn(env)

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_profit += info["daily_profit"]
        total_holding_cost += info["holding_cost"]

        total_sales += info["sales"]
        total_demand += info["demand"]
        total_production += info["production"]

    fulfillment = (
        total_sales.sum() / total_demand.sum() * 100
        if total_demand.sum() > 0 else 0
    )

    return {
        "profit": total_profit,
        "demand_fulfillment": fulfillment,
        "holding_cost": total_holding_cost,
        "production_A": total_production[0],
        "production_B": total_production[1],
        "production_C": total_production[2]
    }


# ==========================================
# 2. STRATEGY EVALUATORS
# ==========================================

def evaluate_greedy():
    """Greedy baseline on the base FactoryEnv."""

    results = []

    for seed in SEEDS:
        env = FactoryEnv(
            data_path=str(DATA_PATH),
            episode_length=30
        )
        env.reset(seed=seed)
        results.append(run_episode(env, choose_action))
        env.close()

    return results


def evaluate_dqn():
    """DQN V2 on the base FactoryEnv."""

    model = DQN.load(str(DQN_MODEL_PATH))
    results = []

    def action_fn(env):
        state = env.get_state()
        action, _ = model.predict(state, deterministic=True)
        return int(action)

    for seed in SEEDS:
        env = FactoryEnv(
            data_path=str(DATA_PATH),
            episode_length=30
        )
        env.reset(seed=seed)
        results.append(run_episode(env, action_fn))
        env.close()

    return results


def evaluate_ppo_v3():
    """PPO V3 on the feasibility-correction environment."""

    model = PPO.load(str(PPO_V3_MODEL_PATH))
    results = []

    def action_fn(env):
        state = env.get_state()
        action, _ = model.predict(state, deterministic=True)
        return int(action)

    for seed in SEEDS:
        env = FactoryPPOV3Env(
            data_path=str(DATA_PATH),
            episode_length=30
        )
        env.reset(seed=seed)
        results.append(run_episode(env, action_fn))
        env.close()

    return results


def evaluate_ppo_v6():
    """PPO V6 on the action-masking environment."""

    model = MaskablePPO.load(str(PPO_V6_MODEL_PATH))
    results = []

    def action_fn(env):
        mask = env.action_masks()
        state = env.get_state()
        action, _ = model.predict(
            state,
            deterministic=True,
            action_masks=mask
        )
        return int(action)

    for seed in SEEDS:
        env = FactoryPPOV6Env(
            data_path=str(DATA_PATH),
            episode_length=30
        )
        env.reset(seed=seed)
        results.append(run_episode(env, action_fn))
        env.close()

    return results


# ==========================================
# 3. MAIN COMPARISON
# ==========================================

def main():

    strategies = {}

    print("\nEvaluating Greedy baseline...")
    strategies["Greedy"] = evaluate_greedy()
    print("Done.")

    print("\nEvaluating DQN V2...")
    strategies["DQN V2"] = evaluate_dqn()
    print("Done.")

    print("\nEvaluating PPO V3...")
    strategies["PPO V3"] = evaluate_ppo_v3()
    print("Done.")

    print("\nEvaluating PPO V6...")
    strategies["PPO V6"] = evaluate_ppo_v6()
    print("Done.")

    # ======================================
    # 4. SUMMARY TABLE
    # ======================================

    summary_rows = []
    episode_rows = []

    for strategy, episodes in strategies.items():

        profits = [e["profit"] for e in episodes]
        fulfillment = [
            e["demand_fulfillment"] for e in episodes
        ]
        holding = [e["holding_cost"] for e in episodes]

        production_A = np.mean(
            [e["production_A"] for e in episodes]
        ) / 30
        production_B = np.mean(
            [e["production_B"] for e in episodes]
        ) / 30
        production_C = np.mean(
            [e["production_C"] for e in episodes]
        ) / 30

        summary_rows.append({
            "strategy": strategy,
            "avg_profit": np.mean(profits),
            "std_profit": np.std(profits),
            "best_profit": np.max(profits),
            "worst_profit": np.min(profits),
            "avg_fulfillment": np.mean(fulfillment),
            "avg_holding_cost": np.mean(holding),
            "production_A_per_day": production_A,
            "production_B_per_day": production_B,
            "production_C_per_day": production_C
        })

        for seed, episode in zip(SEEDS, episodes):
            episode_rows.append({
                "strategy": strategy,
                "seed": seed,
                **episode
            })

    summary = pd.DataFrame(summary_rows)
    episodes_df = pd.DataFrame(episode_rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(REPORT_PATH, index=False)
    episodes_df.to_csv(EPISODES_PATH, index=False)

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )
    print("\nSummary saved:", REPORT_PATH)
    print("Per-episode results saved:", EPISODES_PATH)

    # ======================================
    # 5. COMPARISON FIGURE
    # ======================================

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5)
    )

    names = summary["strategy"].tolist()
    x = np.arange(len(names))

    # Average profit
    bars = ax1.bar(
        x,
        summary["avg_profit"],
        yerr=summary["std_profit"],
        capsize=5,
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    )

    for bar, value in zip(bars, summary["avg_profit"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4000,
            f"₹{value:,.0f}",
            ha="center",
            fontsize=10
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Average profit per 30-day episode (₹)")
    ax1.set_title("Average Profit Comparison")
    ax1.set_ylim(0, summary["avg_profit"].max() * 1.15)

    # Demand fulfillment
    bars = ax2.bar(
        x,
        summary["avg_fulfillment"],
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    )

    for bar, value in zip(bars, summary["avg_fulfillment"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=10
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Demand fulfillment (%)")
    ax2.set_title("Demand Fulfillment Comparison")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print("Comparison figure saved:", FIGURE_PATH)
    print("\n=============================================")


if __name__ == "__main__":
    main()
