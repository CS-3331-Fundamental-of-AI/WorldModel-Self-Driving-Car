import gymnasium as gym
from env.triangular_env import TriangularEnv

# Create environment
env = TriangularEnv(
    config={
        "controlled_vehicles": 1,
        "initial_vehicle_count": 6,
        "spawn_probability": 0.5,
        "show_trajectories": True,
    },
    render_mode="human",
)

# Initialize pygame window
obs, info = env.reset()
env.render()  # render initial frame

for step in range(2000):
    # Sample random action (you can change this later to a policy)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # draw frame
    if terminated or truncated:
        obs, info = env.reset()

env.close()
