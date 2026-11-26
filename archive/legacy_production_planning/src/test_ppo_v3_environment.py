
from stable_baselines3.common.env_checker import check_env

from environment.factory_env_ppo_v3 import FactoryPPOV3Env


env = FactoryPPOV3Env(episode_length=30)

check_env(env, warn=True)

obs, info = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Initial observation:", obs)

total_reward = 0

for day in range(30):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"Day {day + 1}: "
        f"Requested={info['requested_production']}, "
        f"Actual={info['production']}, "
        f"Profit=₹{info['daily_profit']:.2f}, "
        f"Adjusted={info['action_adjusted']}"
    )

    if terminated or truncated:
        break

print("\n========== TEST SUMMARY ==========")
print("Total profit: ₹", round(info["total_profit"], 2))
print("Total reward:", round(total_reward, 2))
print("==================================")

env.close()
