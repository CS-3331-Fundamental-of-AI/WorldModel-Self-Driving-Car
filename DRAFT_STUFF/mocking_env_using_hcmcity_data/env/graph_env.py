from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import (
    AbstractLane,
    CircularLane,
    LineType,
    PolyLaneFixedWidth,
    StraightLane,
)
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import LaneIndex, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class GraphEnv(AbstractEnv):
    """Environment whose road network is loaded from a graph configuration."""

    def __init__(self, *args, **kwargs):
        self._node_positions: dict[str, np.ndarray] = {}
        self._incoming_map: dict[str, list[dict[str, Any]]] = {}
        self._outgoing_map: dict[str, list[dict[str, Any]]] = {}
        super().__init__(*args, **kwargs)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "destination": None,
                "initial_vehicle_count": 8,
                "controlled_vehicles": 1,
                "spawn_probability": 0.6,
                "duration": 40.0,
                "collision_reward": -5,
                "high_speed_reward": 1.0,
                "arrived_reward": 1.0,
                "reward_speed_range": [6.0, 12.0],
                "auto_curve": True,
                "corner_radius": 25.0,
                "curve_samples": 8,
                "exit_distance": 5.0,
                "graph": {
                    "nodes": {
                        "n0": (0.0, 0.0),
                        "n1": (60.0, 0.0),
                        "n2": (60.0, 60.0),
                        "n3": (0.0, 60.0),
                    },
                    "edges": [
                        {"start": "n0", "end": "n1", "reverse": True},
                        {"start": "n1", "end": "n2", "reverse": True},
                        {"start": "n2", "end": "n3", "reverse": True},
                        {"start": "n3", "end": "n0", "reverse": True},
                    ],
                },
            }
        )
        return cfg

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        self._ensure_controlled_vehicle()

    # ------------------------------------------------------------------ #
    # Road construction
    # ------------------------------------------------------------------ #

    def _make_road(self) -> None:
        graph_cfg = self.config.get("graph", {})
        nodes = graph_cfg.get("nodes") or {}
        edges = graph_cfg.get("edges") or []
        if not nodes or not edges:
            raise ValueError("GraphEnv requires a non-empty 'graph' configuration.")

        self._node_positions = {
            node_id: np.array(position, dtype=float) for node_id, position in nodes.items()
        }
        self._incoming_map = {node: [] for node in nodes}
        self._outgoing_map = {node: [] for node in nodes}
        for edge in edges:
            self._incoming_map[edge["end"]].append(edge)
            self._outgoing_map[edge["start"]].append(edge)

        net = RoadNetwork()
        for edge in edges:
            start, end = edge["start"], edge["end"]
            lane = self._build_lane(
                edge,
                self._node_positions[start],
                self._node_positions[end],
                reverse=False,
            )
            net.add_lane(start, end, lane)

            if edge.get("reverse", False):
                reverse_lane = self._build_lane(
                    edge,
                    self._node_positions[end],
                    self._node_positions[start],
                    reverse=True,
                )
                net.add_lane(end, start, reverse_lane)

        self.road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _build_lane(
        self,
        edge: dict[str, Any],
        start: np.ndarray,
        end: np.ndarray,
        reverse: bool,
    ) -> AbstractLane:
        shape = edge.get("shape")
        line_types = self._parse_line_types(
            shape.get("line_types") if shape else None
        )
        width = (
            shape.get("width", edge.get("width", AbstractLane.DEFAULT_WIDTH))
            if shape
            else edge.get("width", AbstractLane.DEFAULT_WIDTH)
        )
        speed_limit = (
            shape.get("speed_limit", edge.get("speed_limit", 15.0))
            if shape
            else edge.get("speed_limit", 15.0)
        )

        if shape:
            lane_type = shape.get("type", "straight").lower()
            if lane_type == "straight":
                return StraightLane(
                    start,
                    end,
                    width=width,
                    line_types=line_types,
                    speed_limit=speed_limit,
                )

            if lane_type == "circular":
                center = np.array(shape["center"], dtype=float)
                radius = float(shape["radius"])
                start_angle = float(shape["start_angle"])
                end_angle = float(shape["end_angle"])
                clockwise = bool(shape.get("clockwise", False))
                if reverse:
                    clockwise = not clockwise
                    start_angle, end_angle = end_angle, start_angle
                return CircularLane(
                    center=center,
                    radius=radius,
                    start_phase=start_angle,
                    end_phase=end_angle,
                    clockwise=clockwise,
                    width=width,
                    line_types=line_types,
                    speed_limit=speed_limit,
                )

            if lane_type == "polyline":
                points = list(shape["points"])
                if not points:
                    raise ValueError("Polyline lane requires points.")
                if reverse:
                    points = list(reversed(points))

                first = np.array(points[0], dtype=float)
                if not np.allclose(first, start):
                    points.insert(0, tuple(start))
                last = np.array(points[-1], dtype=float)
                if not np.allclose(last, end):
                    points.append(tuple(end))

                return PolyLaneFixedWidth(
                    lane_points=[tuple(point) for point in points],
                    width=width,
                    line_types=line_types,
                    speed_limit=speed_limit,
                )

            raise ValueError(f"Unsupported lane shape type '{lane_type}'.")

        if not self.config.get("auto_curve", True):
            return StraightLane(
                start, end, width=width, line_types=line_types, speed_limit=speed_limit
            )

        lane_points = self._auto_lane_points(edge, start, end, reverse)
        if len(lane_points) <= 2 or np.linalg.norm(
            np.array(lane_points[-1]) - np.array(lane_points[0])
        ) < 1e-3:
            return StraightLane(
                lane_points[0],
                lane_points[-1],
                width=width,
                line_types=line_types,
                speed_limit=speed_limit,
            )

        return PolyLaneFixedWidth(
            lane_points=[tuple(point) for point in lane_points],
            width=width,
            line_types=line_types,
            speed_limit=speed_limit,
        )

    def _auto_lane_points(
        self,
        edge: dict[str, Any],
        start: np.ndarray,
        end: np.ndarray,
        reverse: bool,
    ) -> list[tuple[float, float]]:
        start_id = edge["start"]
        end_id = edge["end"]
        exclude_start = end_id
        exclude_end = start_id
        if reverse:
            start_id, end_id = end_id, start_id
            start, end = end, start
            exclude_start, exclude_end = exclude_end, exclude_start

        radius = float(edge.get("corner_radius", self.config["corner_radius"]))
        samples = max(3, int(edge.get("curve_samples", self.config["curve_samples"])))

        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return [tuple(start), tuple(end)]
        direction_unit = direction / length

        points: list[tuple[float, float]] = []

        start_arc_points: list[tuple[float, float]] | None = None
        start_trim = start
        incoming_candidates = [
            candidate
            for candidate in self._incoming_map.get(start_id, [])
            if candidate["start"] != exclude_start and not candidate.get("shape")
        ]
        if len(incoming_candidates) == 1:
            prev_id = incoming_candidates[0]["start"]
            prev_pos = self._node_positions[prev_id]
            fillet = self._compute_fillet(prev_pos, start, end, radius, samples)
            if fillet:
                _, exit_point, arc_pts = fillet
                start_trim = exit_point
                start_arc_points = [
                    tuple(pt)
                    for pt in arc_pts
                    if np.linalg.norm(pt - start_trim) > 1e-6
                ]
                start_arc_points.insert(0, tuple(start_trim))

        if start_arc_points:
            points.extend(start_arc_points)
        else:
            points.append(tuple(start_trim))

        outgoing_candidates = [
            candidate
            for candidate in self._outgoing_map.get(end_id, [])
            if candidate["end"] != exclude_end and not candidate.get("shape")
        ]
        end_trim = end
        if len(outgoing_candidates) == 1:
            next_id = outgoing_candidates[0]["end"]
            next_pos = self._node_positions[next_id]
            fillet = self._compute_fillet(start, end, next_pos, radius, samples)
            if fillet:
                entry_point, _, _ = fillet
                end_trim = entry_point

        if not points:
            points.append(tuple(start_trim))
        if not np.allclose(points[-1], end_trim):
            points.append(tuple(end_trim))

        deduped: list[tuple[float, float]] = []
        for pt in points:
            if not deduped:
                deduped.append(pt)
                continue
            if np.linalg.norm(np.array(pt) - np.array(deduped[-1])) > 1e-6:
                deduped.append(pt)

        cleaned: list[tuple[float, float]] = [deduped[0]]
        min_step = max(radius * 0.05, 0.5)
        for pt in deduped[1:]:
            if (
                np.linalg.norm(np.array(pt) - np.array(cleaned[-1]))
                >= min_step
            ):
                cleaned.append(pt)

        if len(cleaned) < 2:
            fallback_dir = end - start
            if np.linalg.norm(fallback_dir) < 1e-6:
                fallback_dir = np.array([1.0, 0.0])
            fallback_dir = fallback_dir / np.linalg.norm(fallback_dir)
            cleaned.append(
                tuple(
                    np.array(cleaned[0])
                    + fallback_dir * max(radius * 0.2, 1.0)
                )
            )

        return cleaned

    def _compute_fillet(
        self,
        prev: np.ndarray,
        corner: np.ndarray,
        nxt: np.ndarray,
        radius: float,
        samples: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]] | None:
        vec_in = corner - prev
        vec_out = nxt - corner
        len_in = np.linalg.norm(vec_in)
        len_out = np.linalg.norm(vec_out)
        if len_in < 1e-6 or len_out < 1e-6:
            return None

        dir_in = vec_in / len_in
        dir_out = vec_out / len_out

        dot_prod = np.clip(np.dot(-dir_in, dir_out), -1.0, 1.0)
        angle = np.arccos(dot_prod)
        if angle < 1e-2 or np.isclose(angle, np.pi):
            return None

        offset = radius / np.tan(angle / 2.0)
        max_offset = min(len_in, len_out) * 0.5
        offset = min(offset, max_offset)
        if offset <= 1e-3:
            return None

        entry = corner - dir_in * offset
        exit_point = corner + dir_out * offset

        bisector = (-dir_in) + dir_out
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < 1e-6:
            return None
        bisector /= bisector_norm
        center = corner + bisector * (radius / np.sin(angle / 2.0))

        start_angle = np.arctan2(entry[1] - center[1], entry[0] - center[0])
        end_angle = np.arctan2(exit_point[1] - center[1], exit_point[0] - center[0])

        diff = end_angle - start_angle
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        cross = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0]
        if cross > 0 and diff < 0:
            diff += 2 * np.pi
        elif cross < 0 and diff > 0:
            diff -= 2 * np.pi

        angles = np.linspace(start_angle, start_angle + diff, samples, endpoint=True)
        arc_points = [
            (
                center[0] + radius * np.cos(angle_val),
                center[1] + radius * np.sin(angle_val),
            )
            for angle_val in angles
        ]

        return entry, exit_point, arc_points

    def _parse_line_types(
        self, specification: Iterable[str] | None
    ) -> tuple[LineType, LineType]:
        if specification is None:
            return (LineType.STRIPED, LineType.CONTINUOUS)
        return tuple(
            getattr(LineType, item.upper()) for item in specification
        )  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Vehicle management
    # ------------------------------------------------------------------ #

    def _make_vehicles(self, n_vehicles: int | None = None) -> None:
        if n_vehicles is None:
            n_vehicles = self.config["initial_vehicle_count"]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        lanes = self._available_edges()
        if not lanes:
            raise RuntimeError("No lanes available for vehicle spawning.")

        background_target = max(n_vehicles - self.config["controlled_vehicles"], 0)
        for _ in range(background_target):
            self._spawn_vehicle(
                longitudinal=self.np_random.uniform(0.0, 30.0),
                spawn_probability=1.0,
                spawn_candidates=lanes,
            )

        for _ in range(3):
            for _ in range(self.config["simulation_frequency"]):
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])

        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            lane_index = lanes[self.np_random.integers(len(lanes))]
            lane = self.road.network.get_lane(lane_index)
            frac = np.clip(self.np_random.uniform(0.15, 0.35), 0.05, 0.9)
            longitudinal = np.clip(frac * lane.length, 0.0, max(lane.length - 5.0, 1.0))

            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                lane.position(longitudinal, 0.0),
                speed=lane.speed_limit,
                heading=lane.heading_at(longitudinal),
            )

            destination = self._pick_destination(lane_index[1])
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                # Plan-route may fail if not implemented — assign fallback route
                if not hasattr(ego_vehicle, "route") or ego_vehicle.route is None:
                    ego_vehicle.route = [lane_index]  # already (start_node, end_node, lane_id)

            # Ensure route exists for all vehicles
            if not hasattr(ego_vehicle, "route") or ego_vehicle.route is None:
                ego_vehicle.route = [lane_index]  # already (start_node, end_node, lane_id)

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            self._prevent_initial_overlap(ego_vehicle)

    def _ensure_controlled_vehicle(self) -> None:
        """Ensure at least one controlled vehicle exists for observations."""
        if self.controlled_vehicles:
            self.vehicle = self.controlled_vehicles[0]
            self.observer_vehicle = self.controlled_vehicles[0]
            return

        edges = self._available_edges()
        if not edges:
            raise RuntimeError("Graph contains no lanes; cannot spawn vehicles.")

        lane_index = edges[0]
        lane = self.road.network.get_lane(lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            lane.position(5.0, 0.0),
            speed=lane.speed_limit * 0.5,
            heading=lane.heading_at(5.0),
        )
        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles = [ego_vehicle]
        self.vehicle = ego_vehicle
        self.observer_vehicle = ego_vehicle

    def _spawn_vehicle(
        self,
        longitudinal: float = 0.0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float | None = None,
        spawn_candidates: list[LaneIndex] | None = None,
    ) -> Vehicle | None:
        spawn_probability = (
            spawn_probability
            if spawn_probability is not None
            else self.config["spawn_probability"]
        )
        if self.np_random.uniform() > spawn_probability:
            return None

        candidates = spawn_candidates or self._available_edges()
        lane_index = candidates[self.np_random.integers(len(candidates))]
        lane = self.road.network.get_lane(lane_index)

        longitudinal_sample = (
            longitudinal + 5.0 + self.np_random.normal() * position_deviation
        )
        longitudinal_sample = np.clip(longitudinal_sample, 0.0, lane.length - 5.0)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            lane_index,
            longitudinal=longitudinal_sample,
            speed=np.clip(
                lane.speed_limit
                + self.np_random.normal() * speed_deviation,
                1.0,
                lane.speed_limit * 1.4,
            ),
        )

        for other in self.road.vehicles:
            if np.linalg.norm(other.position - vehicle.position) < 15.0:
                return None

        destination = self._pick_destination(lane_index[1])
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _available_edges(self) -> list[LaneIndex]:
        edges: list[LaneIndex] = []
        for start, destinations in self.road.network.graph.items():
            for end, lanes in destinations.items():
                for lane_id in range(len(lanes)):
                    edges.append((start, end, lane_id))
        return edges

    def _pick_destination(self, current_node: str) -> str:
        nodes = list(self.config["graph"]["nodes"].keys())
        configured = self.config.get("destination")
        if configured and configured in nodes:
            return configured
        candidates = [node for node in nodes if node != current_node]
        if not candidates:
            return current_node
        return str(self.np_random.choice(candidates))

    # ------------------------------------------------------------------ #
    # Reward and termination logic
    # ------------------------------------------------------------------ #

    def _reward(self, action) -> float:
        rewards = self._agent_rewards(self.vehicle)
        reward = sum(self.config.get(name, 0.0) * value for name, value in rewards.items())
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action) -> dict[str, float]:
        return self._agent_rewards(self.vehicle)

    def _agent_rewards(self, vehicle: Vehicle | None) -> dict[str, float]:
        if vehicle is None:
            return {
                "collision_reward": 0.0,
                "high_speed_reward": 0.0,
                "arrived_reward": 0.0,
                "on_road_reward": 0.0,
            }

        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0.0, 1.0]
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "high_speed_reward": float(np.clip(scaled_speed, 0.0, 1.0)),
            "arrived_reward": float(self.has_arrived(vehicle)),
            "on_road_reward": float(vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        return bool(
            self.vehicle
            and (self.vehicle.crashed or self.has_arrived(self.vehicle))
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    def _clear_vehicles(self) -> None:
        def is_leaving(vehicle: Vehicle) -> bool:
            if vehicle in self.controlled_vehicles or vehicle.lane is None:
                return False
            longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
            return (
                (vehicle.route is None or len(vehicle.route) == 0)
                and longitudinal >= vehicle.lane.length - 4 * vehicle.LENGTH
            )

        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles or not is_leaving(vehicle)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float | None = None) -> bool:
        if vehicle.lane is None:
            return False
        longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
        threshold = exit_distance
        if threshold is None:
            threshold = self.config.get("exit_distance", 10.0)
        threshold = min(threshold, vehicle.lane.length * 0.5)
        return (
            vehicle.route is None
            and longitudinal >= vehicle.lane.length - threshold
        )

    def _prevent_initial_overlap(self, ego_vehicle: Vehicle) -> None:
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle is ego_vehicle
            or np.linalg.norm(vehicle.position - ego_vehicle.position) >= 20.0
        ]
