
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env import FactoryEnv


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = ROOT_DIR / "models" / "dqn_production_optimizer_v2.zip"
model = DQN.load(str(MODEL_PATH))

env = FactoryEnv()

results = []

# ==========================================
# 2. INITIALIZE DIAGNOSTICS
# ==========================================

total_production = np.zeros(3)
action_counts = np.zeros(125, dtype=int)

total_material_used = 0
total_machine_hours_used = 0
total_production_cost = 0

total_steps = 0


# ==========================================
# 3. EVALUATE OVER 30 EPISODES
# ==========================================

for episode in range(30):

    state, info = env.reset(seed=episode + 100)

    total_profit = 0
    total_reward = 0

    total_demand = 0
    total_sales = 0

    total_holding_cost = 0
    total_shortage_cost = 0

    invalid_actions = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Predict action
        action, _ = model.predict(
            state,
            deterministic=True
        )

        action = int(action)

        # Count selected actions
        action_counts[action] += 1

        # Take action in environment
        state, reward, terminated, truncated, info = env.step(action)

        # ======================================
        # COLLECT PRODUCTION DIAGNOSTICS
        # ======================================

        total_production += np.array(info["production"])

        total_material_used += info["material_used"]
        total_machine_hours_used += info["machine_hours_used"]
        total_production_cost += info["production_cost"]

        total_steps += 1

        # ======================================
        # COLLECT EPISODE METRICS
        # ======================================

        total_profit += info["daily_profit"]
        total_reward += reward

        total_demand += np.sum(info["demand"])
        total_sales += np.sum(info["sales"])

        total_holding_cost += info["holding_cost"]
        total_shortage_cost += info["shortage_cost"]

        if not info["feasible"]:
            invalid_actions += 1

    fulfillment = (
        total_sales / total_demand * 100
        if total_demand > 0 else 0
    )

    results.append({

        "episode": episode + 1,
        "total_profit": total_profit,
        "total_reward": total_reward,
        "demand_fulfillment": fulfillment,
        "holding_cost": total_holding_cost,
        "shortage_cost": total_shortage_cost,
        "invalid_actions": invalid_actions

    })

    print(
        f"Episode {episode + 1}/30 | "
        f"Profit: Rs.{total_profit:.2f} | "
        f"Fulfillment: {fulfillment:.2f}%"
    )


# ==========================================
# 4. DQN EVALUATION SUMMARY
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== DQN EVALUATION ==========")

print("Average Profit:",
      round(results_df["total_profit"].mean(), 2))

print("Best Profit:",
      round(results_df["total_profit"].max(), 2))

print("Average Demand Fulfillment:",
      round(results_df["demand_fulfillment"].mean(), 2), "%")

print("Average Holding Cost:",
      round(results_df["holding_cost"].mean(), 2))

print("Average Shortage Cost:",
      round(results_df["shortage_cost"].mean(), 2))

print("Total Invalid Actions:",
      results_df["invalid_actions"].sum())


# ==========================================
# 5. PRODUCTION DIAGNOSTICS
# ==========================================

print("\n========== DQN PRODUCTION DIAGNOSTICS ==========")

print("\nTotal Production Across All Episodes:")

print("Product A:", total_production[0])
print("Product B:", total_production[1])
print("Product C:", total_production[2])

print("\nAverage Daily Production:")

if total_steps > 0:
    print("Product A:", round(total_production[0] / total_steps, 2))
    print("Product B:", round(total_production[1] / total_steps, 2))
    print("Product C:", round(total_production[2] / total_steps, 2))

print("\nResource Usage:")

print("Total Material Used:", total_material_used)
print("Total Machine Hours Used:", total_machine_hours_used)
print("Total Production Cost:", round(total_production_cost, 2))

print("\nMost Frequently Selected Actions:")

for action in np.argsort(action_counts)[-10:][::-1]:

    print(
        f"Action {action}: {action_counts[action]} selections "
        f"| Production: {env.decode_action(int(action))}"
    )


# ==========================================
# 6. SAVE RESULTS
# ==========================================

output_path = ROOT_DIR / "reports" / "dqn_evaluation.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)

print("\n========== EVALUATION COMPLETED ==========")

env.close()
