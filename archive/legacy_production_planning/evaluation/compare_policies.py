
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env import FactoryEnv
from baseline.baseline_strategy import choose_action


# ==========================================
# 1. LOAD MODEL
# ==========================================

MODEL_PATH = ROOT_DIR / "models" / "dqn_production_optimizer_v2.zip"

model = DQN.load(str(MODEL_PATH))

EPISODES = 30


# ==========================================
# 2. EVALUATE AN ACTION WITHOUT STEPPING
# ==========================================

def calculate_action_profit(env, action):

    production = env.decode_action(int(action))

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

    invalid_penalty = 0

    if not feasible:

        production = np.zeros(3)

        production_cost = 0

        available_stock = env.inventory.copy()

        sales = np.minimum(
            available_stock, env.daily_demand
        )

        ending_inventory = available_stock - sales

        invalid_penalty = env.invalid_action_penalty

    revenue = np.dot(sales, env.prices)

    holding_cost = (
        ending_inventory.sum() * env.holding_cost
    )

    daily_profit = (
        revenue
        - production_cost
        - holding_cost
        - invalid_penalty
    )

    return {
        "profit": float(daily_profit),
        "production": production.copy(),
        "feasible": bool(feasible)
    }


# ==========================================
# 3. INITIALIZE METRICS
# ==========================================

results = []

dqn_action_counts = np.zeros(125, dtype=int)
greedy_action_counts = np.zeros(125, dtype=int)

dqn_total_production = np.zeros(3)
greedy_total_production = np.zeros(3)

dqn_actual_profit = 0
greedy_actual_profit = 0

dqn_state_matches = 0
greedy_state_matches = 0

dqn_state_profit_gaps = []
greedy_state_profit_gaps = []


# ==========================================
# 4. RUN PAIRED POLICY COMPARISON
# ==========================================

for episode in range(EPISODES):

    # Separate environments with identical reset seeds
    dqn_env = FactoryEnv()
    greedy_env = FactoryEnv()

    dqn_state, _ = dqn_env.reset(seed=episode + 100)
    greedy_state, _ = greedy_env.reset(seed=episode + 100)

    # Confirm identical starting states
    assert np.allclose(dqn_state, greedy_state), (
        "Environment states do not match!"
    )

    dqn_episode_profit = 0
    greedy_episode_profit = 0

    dqn_episode_matches = 0
    greedy_episode_matches = 0

    dqn_episode_gaps = []
    greedy_episode_gaps = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # ----------------------------------
        # A. COMPARE ACTIONS ON DQN STATE
        # ----------------------------------

        dqn_action, _ = model.predict(
            dqn_state,
            deterministic=True
        )

        dqn_action = int(dqn_action)

        greedy_action_on_dqn_state = choose_action(dqn_env)

        dqn_action_counts[dqn_action] += 1
        greedy_action_counts[greedy_action_on_dqn_state] += 1

        dqn_result = calculate_action_profit(
            dqn_env, dqn_action
        )

        greedy_result_on_dqn_state = calculate_action_profit(
            dqn_env, greedy_action_on_dqn_state
        )

        dqn_gap = (
            greedy_result_on_dqn_state["profit"]
            - dqn_result["profit"]
        )

        dqn_episode_gaps.append(dqn_gap)
        dqn_state_profit_gaps.append(dqn_gap)

        if dqn_action == greedy_action_on_dqn_state:
            dqn_episode_matches += 1
            dqn_state_matches += 1

        # Execute DQN's own action
        dqn_state, reward, dqn_terminated, dqn_truncated, dqn_info = (
            dqn_env.step(dqn_action)
        )

        dqn_episode_profit += dqn_info["daily_profit"]

        dqn_total_production += np.array(
            dqn_info["production"]
        )

        # ----------------------------------
        # B. COMPARE ACTIONS ON GREEDY STATE
        # ----------------------------------

        greedy_action = choose_action(greedy_env)

        dqn_action_on_greedy_state, _ = model.predict(
            greedy_state,
            deterministic=True
        )

        dqn_action_on_greedy_state = int(
            dqn_action_on_greedy_state
        )

        greedy_action_counts[greedy_action] += 1

        dqn_action_counts[dqn_action_on_greedy_state] += 1

        greedy_result = calculate_action_profit(
            greedy_env, greedy_action
        )

        dqn_result_on_greedy_state = calculate_action_profit(
            greedy_env, dqn_action_on_greedy_state
        )

        greedy_gap = (
            greedy_result["profit"]
            - dqn_result_on_greedy_state["profit"]
        )

        greedy_episode_gaps.append(greedy_gap)
        greedy_state_profit_gaps.append(greedy_gap)

        if greedy_action == dqn_action_on_greedy_state:
            greedy_episode_matches += 1
            greedy_state_matches += 1

        # Execute Greedy's own action
        greedy_state, reward, greedy_terminated, greedy_truncated, greedy_info = (
            greedy_env.step(greedy_action)
        )

        greedy_episode_profit += greedy_info["daily_profit"]

        greedy_total_production += np.array(
            greedy_info["production"]
        )

        # Both episodes should end together
        assert (
            dqn_terminated == greedy_terminated
            and dqn_truncated == greedy_truncated
        ), "Episode termination mismatch!"

        terminated = dqn_terminated
        truncated = dqn_truncated

    dqn_actual_profit += dqn_episode_profit
    greedy_actual_profit += greedy_episode_profit

    results.append({

        "episode": episode + 1,

        "dqn_profit": dqn_episode_profit,
        "greedy_profit": greedy_episode_profit,

        "profit_difference": (
            greedy_episode_profit - dqn_episode_profit
        ),

        "dqn_action_match_percent": (
            dqn_episode_matches / 30 * 100
        ),

        "greedy_action_match_percent": (
            greedy_episode_matches / 30 * 100
        ),

        "dqn_state_immediate_gap": np.mean(dqn_episode_gaps),
        "greedy_state_immediate_gap": np.mean(greedy_episode_gaps)

    })

    print(
        f"Episode {episode + 1}/{EPISODES} | "
        f"DQN: Rs.{dqn_episode_profit:.2f} | "
        f"Greedy: Rs.{greedy_episode_profit:.2f} | "
        f"Difference: Rs.{greedy_episode_profit - dqn_episode_profit:.2f}"
    )

    dqn_env.close()
    greedy_env.close()


# ==========================================
# 5. FINAL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

total_comparisons = EPISODES * 30

print("\n========== PAIRED POLICY COMPARISON ==========")

print("\nACTUAL EPISODE PROFIT")

print(
    "DQN Average Profit:",
    round(results_df["dqn_profit"].mean(), 2)
)

print(
    "Greedy Average Profit:",
    round(results_df["greedy_profit"].mean(), 2)
)

print(
    "Average Profit Difference:",
    round(results_df["profit_difference"].mean(), 2)
)

print("\nACTION AGREEMENT")

print(
    "DQN trajectory:",
    round(dqn_state_matches / total_comparisons * 100, 2),
    "%"
)

print(
    "Greedy trajectory:",
    round(greedy_state_matches / total_comparisons * 100, 2),
    "%"
)

print("\nIMMEDIATE PROFIT ADVANTAGE OF GREEDY")

print(
    "On DQN states:",
    round(np.mean(dqn_state_profit_gaps), 2)
)

print(
    "On Greedy states:",
    round(np.mean(greedy_state_profit_gaps), 2)
)

print("\nAVERAGE DAILY PRODUCTION")

print(
    "DQN:",
    np.round(dqn_total_production / total_comparisons, 2)
)

print(
    "Greedy:",
    np.round(greedy_total_production / total_comparisons, 2)
)


# ==========================================
# 6. MOST FREQUENTLY SELECTED ACTIONS
# ==========================================

print("\nMOST FREQUENT DQN ACTIONS")

for action in np.argsort(dqn_action_counts)[-10:][::-1]:

    print(
        f"Action {action}: {dqn_action_counts[action]} selections "
        f"| Production: {FactoryEnv().decode_action(int(action))}"
    )

print("\nMOST FREQUENT GREEDY ACTIONS")

for action in np.argsort(greedy_action_counts)[-10:][::-1]:

    print(
        f"Action {action}: {greedy_action_counts[action]} selections "
        f"| Production: {FactoryEnv().decode_action(int(action))}"
    )


# ==========================================
# 7. SAVE COMPARISON
# ==========================================

output_path = ROOT_DIR / "reports" / "action_comparison.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)

print("\n========== COMPARISON COMPLETED ==========")
