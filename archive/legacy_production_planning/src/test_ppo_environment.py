
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from environment.factory_env_ppo import FactoryPPOEnv


env = FactoryPPOEnv()

state, info = env.reset(seed=42)

print("\n========== PPO ENVIRONMENT TEST ==========")

print("Initial state:", state)
print("State shape:", state.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

assert state.shape == (10,)
assert env.observation_space.contains(state)

total_profit = 0
feasible_days = 0
infeasible_days = 0

for day in range(30):

    # Action [1, 2, 2] means production:
    # A = 10, B = 20, C = 20
    action = np.array([1, 2, 2])

    state, reward, terminated, truncated, info = env.step(action)

    daily_profit = info["daily_profit"]
    total_profit += daily_profit

    if info["feasible"]:
        feasible_days += 1
    else:
        infeasible_days += 1

    print(f"\nDay {day + 1}")
    print("Production:", info["production"])
    print("Sales:", info["sales"])
    print("Daily profit:", daily_profit)
    print("Feasible:", info["feasible"])
    print("Inventory:", info["inventory"])

    if not info["feasible"]:
        print("Invalid penalty:", info["invalid_penalty"])
        print("Material used:", info["material_used"])
        print("Machine hours used:", info["machine_hours_used"])

    assert state.shape == (10,)
    assert env.observation_space.contains(state)
    assert np.all(info["inventory"] >= 0)
    assert info["inventory"].sum() <= env.storage_capacity

    if terminated or truncated:
        break

print("\n========== TEST SUMMARY ==========")
print("Total profit:", total_profit)
print("Feasible days:", feasible_days)
print("Infeasible days:", infeasible_days)
print("Days completed:", day + 1)

print("\n========== PPO ENVIRONMENT TEST COMPLETED ==========")

env.close()
