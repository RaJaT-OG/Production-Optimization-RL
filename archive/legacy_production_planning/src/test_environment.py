
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from environment.factory_env import FactoryEnv

# Create environment
env = FactoryEnv()

# Reset environment
state, info = env.reset(seed=42)

print("\n========== ENVIRONMENT TEST ==========")

print("Initial state:", state)
print("State shape:", state.shape)

print("\nAction space:", env.action_space)
print("Observation space:", env.observation_space)

# Verify initial observation
assert state.shape == (10,)
assert env.observation_space.contains(state)

# Test 30 production days
total_reward = 0

for day in range(30):

    # Test a valid no-production action
    action = 0

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(f"\nDay {day + 1}")

    print("Production:", info["production"])
    print("Sales:", info["sales"])
    print("Revenue:", info["revenue"])
    print("Daily profit:", info["daily_profit"])
    print("Inventory:", info["inventory"])
    print("Feasible:", info["feasible"])

    # Validate observation
    assert state.shape == (10,)
    assert env.observation_space.contains(state)

    # Validate inventory
    assert np.all(info["inventory"] >= 0)
    assert info["inventory"].sum() <= env.storage_capacity

    if terminated or truncated:
        break

print("\nTotal reward:", total_reward)
print("Total profit:", env.total_profit)

print("\n========== TEST COMPLETED ==========")

env.close()
