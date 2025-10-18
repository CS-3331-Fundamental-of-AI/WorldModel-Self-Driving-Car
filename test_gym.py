import gymnasium as gym
import highway_env  
env = gym.make("highway-v0", render_mode='human') #gym.make("highway-v0", render_mode='human')

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()