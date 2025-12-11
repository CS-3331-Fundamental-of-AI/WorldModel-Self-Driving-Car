"""
Local variants of highway-env environments with custom registrations.

These are adapted from upstream highway_env but registered under local IDs to avoid
clashing with the package defaults. They keep the same interfaces and rewards but
can be referenced via the `local-...` environment IDs below.
"""

import numpy as np
from gymnasium.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.graphics import EnvViewer

Observation = np.ndarray

"THE CODE WAS BASED FROM: "
"https://github.com/wuxiyang1996/Heterogeneous_Highway_Env/blob/master/envs/highway_env.pys"

class LocalHighwayEnv(AbstractEnv):
    """
    Straight highway driving with standard reward shaping.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,
                "reward_speed_range": [20, 30],
                "offroad_terminal": False,
                "offscreen_rendering": True,
                "screen_width": 600,
                "screen_height": 300,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        for i, veh in enumerate(self.road.vehicles):
            veh.vehicle_id = i

    def get_state(self):
        state = []
        features = self.config["observation"]["observation_config"]["features"]
        for veh in self.road.vehicles:
            vec_raw = veh.to_dict()
            state.append([vec_raw[key] for key in features])
        return state

    def _reward(self, action: Action) -> float:
        rewards = [self.agent_reward(v, action) for v in self.controlled_vehicles]
        self.rewards = rewards.copy()
        return float(sum(rewards))

    def _info(self, obs: Observation, action: Action) -> dict:
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        # Optional fields only if they exist (avoid attribute errors during early reset).
        if hasattr(self, "costs"):
            info["cost"] = self._cost(action)
        if hasattr(self, "rewards"):
            info["agents_rewards"] = self.rewards
        if hasattr(self, "dones"):
            info["agents_dones"] = self.dones
        if hasattr(self, "terminated"):
            info["agents_terminated"] = self.terminated
        return info

    def agent_reward(self, vehicle, action: Action) -> float:
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = (
            self.config["collision_reward"] * vehicle.crashed
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1)
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        )
        reward = utils.lmap(
            reward,
            [
                self.config["collision_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )
        reward = 0 if not vehicle.on_road else reward
        return float(reward)

    def _is_terminated(self) -> bool:
        terminated = []
        for v in self.controlled_vehicles:
            term = (
                v.crashed
                or (self.config["offroad_terminal"] and not v.on_road)
                or self.time >= self.config["duration"]
            )
            terminated.append(term)
        self.terminated = terminated.copy()
        return bool(np.all(terminated))

    def _is_truncated(self) -> bool:
        # No additional truncation beyond duration handled in _is_terminated.
        return False

    def _cost(self, action: int) -> float:
        costs = [float(v.crashed) for v in self.controlled_vehicles]
        self.costs = costs.copy()
        return float(sum(costs))

    def render(self, mode: str = "human"):
        self.rendering_mode = mode
        # Fast path: offscreen requested, skip pygame viewer.
        if mode != "human" and self.config.get("offscreen_rendering", True):
            h = int(self.config.get("screen_height", 300))
            w = int(self.config.get("screen_width", 600))
            return np.zeros((h, w, 3), dtype=np.uint8)
        if self.viewer is None:
            self.viewer = EnvViewer(self)
        self.enable_auto_render = True
        offscreen = mode != "human"
        self.viewer.offscreen = offscreen
        self.viewer.observer_vehicle = self.controlled_vehicles[0]
        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == "rgb_array":
            imgs = []
            for agent in self.controlled_vehicles:
                self.viewer.observer_vehicle = agent
                self.viewer.offscreen = True
                self.viewer.display()
                imgs.append(self.viewer.get_image())
            return np.concatenate(imgs, axis=1)


class LocalHighwayEnvHetero(LocalHighwayEnv):
    """Heterogeneous traffic mix."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "normal_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "aggressive_vehicle_type": "highway_env.vehicle.behavior.AggressiveVehicle",
                "defensive_vehicle_type": "highway_env.vehicle.behavior.DefensiveVehicle",
                "ratio_aggressive": 0.1,
                "ratio_defensive": 0.1,
            }
        )
        return config

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["normal_vehicles_type"])
        aggro_type = utils.class_from_path(self.config["aggressive_vehicle_type"])
        defen_type = utils.class_from_path(self.config["defensive_vehicle_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            controlled_vehicle = self.action_type.vehicle_class(
                self.road,
                controlled_vehicle.position,
                controlled_vehicle.heading,
                controlled_vehicle.speed,
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                random_num = self.np_random.random()
                if random_num < self.config["ratio_aggressive"]:
                    vehicle = aggro_type.create_random(
                        self.road, spacing=1 / self.config["vehicles_density"]
                    )
                elif random_num > 1 - self.config["ratio_defensive"]:
                    vehicle = defen_type.create_random(
                        self.road, spacing=1 / self.config["vehicles_density"]
                    )
                else:
                    vehicle = other_vehicles_type.create_random(
                        self.road, spacing=1 / self.config["vehicles_density"]
                    )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        for i, veh in enumerate(self.road.vehicles):
            veh.vehicle_id = i


class LocalHighwayEnvHeteroH(LocalHighwayEnvHetero):
    """Heterogeneous traffic with higher aggressive/defensive ratios."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"ratio_aggressive": 0.3, "ratio_defensive": 0.3})
        return config


class LocalHighwayEnvHeteroVH(LocalHighwayEnvHetero):
    """Very high aggression ratio; no defensive vehicles."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"ratio_aggressive": 0.5, "ratio_defensive": 0.0})
        return config


class LocalHighwayEnvFast(LocalHighwayEnv):
    """Faster, lighter variant with fewer vehicles and lanes."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class LocalMOHighwayEnv(LocalHighwayEnv):
    """Multi-objective reward decomposition."""

    def _rewards(self, action: Action) -> dict:
        rewards = {}
        rewards["collision"] = self.vehicle.crashed
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        rewards["right_lane"] = lane / max(len(neighbours) - 1, 1)
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        rewards["high_speed"] = np.clip(scaled_speed, 0, 1)
        return rewards

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = (
            self.config["collision_reward"] * rewards["collision"]
            + self.config["right_lane_reward"] * rewards["right_lane"]
            + self.config["high_speed_reward"] * rewards["high_speed"]
        )
        reward = utils.lmap(
            reward,
            [
                self.config["collision_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )
        reward = 0 if not self.vehicle.on_road else reward
        return float(reward)

    def _info(self, obs: Observation, action: Action) -> dict:
        return self._rewards(action)


# Local registrations to avoid clobbering upstream highway_env IDs.
register(id="local-highway-v0", entry_point="envs.highway_custom:LocalHighwayEnv")
register(id="local-highway-hetero-v0", entry_point="envs.highway_custom:LocalHighwayEnvHetero")
register(id="local-highway-hetero-H-v0", entry_point="envs.highway_custom:LocalHighwayEnvHeteroH")
register(id="local-highway-hetero-VH-v0", entry_point="envs.highway_custom:LocalHighwayEnvHeteroVH")
register(id="local-highway-fast-v0", entry_point="envs.highway_custom:LocalHighwayEnvFast")
register(id="local-mo-highway-v0", entry_point="envs.highway_custom:LocalMOHighwayEnv")
