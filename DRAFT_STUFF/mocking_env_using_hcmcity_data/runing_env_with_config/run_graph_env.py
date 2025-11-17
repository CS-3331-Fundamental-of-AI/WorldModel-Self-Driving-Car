import json

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import highway_env  # ensure dependency installed
import numpy as np

from env.graph_env import GraphEnv

file_path = 'roundabound_test.json'
with open(file_path, "r", encoding="utf-8") as fh:
    raw = json.load(fh)

config = GraphEnv.default_config()
if "graph" in raw:
    config.update(raw)
else:
    config["graph"] = raw
config["action"]["type"] = "ContinuousAction"

config.update(
    {
        "render_mode": "human",
        "duration": config.get("duration", 120.0),
        "initial_vehicle_count": config.get("initial_vehicle_count", 6),
        "controlled_vehicles": config.get("controlled_vehicles", 1),
        "spawn_probability": config.get("spawn_probability", 0.6),
        "simulation_frequency": config.get("simulation_frequency", 15),
    }
)

import numpy as np
import math

env = GraphEnv(config=config, render_mode="human")
obs, info = env.reset(seed=42)

# Basic parameters
target_speed = 25.0  # m/s desired cruise speed (~90 km/h)
k_speed = 0.2        # proportional gain for speed
k_heading = 1.2      # proportional gain for steering

for step in range(20000):
    ego = env.vehicle
    lane = ego.lane

    if lane is None:
        action = env.action_space.sample()
    else:
        # Convert global position to lane coordinates (longitudinal, lateral)
        lon, lat = lane.local_coordinates(ego.position)

        # Compute desired heading based on longitudinal position
        lane_heading = lane.heading_at(lon)

        # Compute steering + speed control
        heading_error = (lane_heading - ego.heading + np.pi) % (2 * np.pi) - np.pi
        speed_error = target_speed - ego.speed

        steering = np.clip(k_heading * heading_error, -1.0, 1.0)
        accel = np.clip(k_speed * speed_error, -1.0, 1.0)

        action = np.array([steering, accel])

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        print(f"Episode finished at step {step+1}, reward={reward:.2f}")
        obs, info = env.reset(seed=42)

env.close()