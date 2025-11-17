from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.road.lane import AbstractLane, LineType, PolyLaneFixedWidth
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class TriangularEnv(IntersectionEnv):
    """Three-way cyclic driving scenario derived from the intersection environment."""

    VERTEX_ORDER: tuple[str, str, str] = ("n0", "n1", "n2")

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "destination": cls.VERTEX_ORDER[0],
                "initial_vehicle_count": 6,
                "spawn_probability": 0.5,
                "centering_position": [0.5, 0.5],
                "scaling": 6.0,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [6.0, 12.0],
            }
        )
        return config

    def _make_road(self) -> None:
        """Build an equilateral triangular loop with continuous curvature."""
        net = RoadNetwork()
        striped, continuous = LineType.STRIPED, LineType.CONTINUOUS

        outer_radius = 120.0
        corner_radius = 25.0
        vertex_angles = np.deg2rad([90.0, 210.0, 330.0])
        vertices = [
            outer_radius * np.array([np.cos(a), np.sin(a)]) for a in vertex_angles
        ]

        entries = []
        exits = []
        arc_centers = []
        start_angles = []
        end_angles = []

        for i in range(3):
            prev_vertex = vertices[i - 1]
            current_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % 3]

            incoming = current_vertex - prev_vertex
            outgoing = next_vertex - current_vertex
            incoming /= np.linalg.norm(incoming)
            outgoing /= np.linalg.norm(outgoing)

            interior_angle = np.arccos(
                np.clip(np.dot(-incoming, outgoing), -1.0, 1.0)
            )
            offset = corner_radius / np.tan(interior_angle / 2.0)

            entry = current_vertex - incoming * offset
            exit_point = current_vertex + outgoing * offset

            bisector = -incoming + outgoing
            bisector /= np.linalg.norm(bisector)
            center = current_vertex + bisector * (
                corner_radius / np.sin(interior_angle / 2.0)
            )

            start_angle = np.arctan2(entry[1] - center[1], entry[0] - center[0])
            end_angle = np.arctan2(
                exit_point[1] - center[1], exit_point[0] - center[0]
            )
            delta = ((end_angle - start_angle) + 2 * np.pi) % (2 * np.pi)
            if delta <= 0:
                delta += 2 * np.pi

            entries.append(entry)
            exits.append(exit_point)
            arc_centers.append(center)
            start_angles.append(start_angle)
            end_angles.append(start_angle + delta)

        for i in range(3):
            start_label = self.VERTEX_ORDER[i]
            end_label = self.VERTEX_ORDER[(i + 1) % 3]

            arc_center = arc_centers[i]
            start_angle = start_angles[i]
            end_angle = end_angles[i]

            arc_samples = np.linspace(0.0, end_angle - start_angle, num=8)
            arc_points = [
                arc_center
                + corner_radius
                * np.array(
                    [
                        np.cos(start_angle + step),
                        np.sin(start_angle + step),
                    ]
                )
                for step in arc_samples
            ]

            straight_start = arc_points[-1]
            straight_end = entries[(i + 1) % 3]
            straight_samples = [
                straight_start
                + (straight_end - straight_start) * t
                for t in np.linspace(0.0, 1.0, num=5)[1:]
            ]

            lane_points = arc_points + straight_samples

            lane = PolyLaneFixedWidth(
                [tuple(point) for point in lane_points],
                width=AbstractLane.DEFAULT_WIDTH,
                line_types=(striped, continuous),
                speed_limit=15.0,
            )
            net.add_lane(start_label, end_label, lane)

            reverse_lane = PolyLaneFixedWidth(
                [tuple(point) for point in reversed(lane_points)],
                width=AbstractLane.DEFAULT_WIDTH,
                line_types=(striped, continuous),
                speed_limit=15.0,
            )
            net.add_lane(end_label, start_label, reverse_lane)

        self.road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )

    def _make_vehicles(self, n_vehicles: int | None = None) -> None:
        """Populate the triangular loop with background traffic and controlled vehicles."""
        if n_vehicles is None:
            n_vehicles = self.config["initial_vehicle_count"]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        spawn_edges = self._spawn_edges()
        background_target = max(n_vehicles - self.config["controlled_vehicles"], 0)
        for _ in range(background_target):
            self._spawn_vehicle(
                longitudinal=self.np_random.uniform(0.0, 30.0),
                spawn_probability=1.0,
                spawn_edges=spawn_edges,
            )

        for _ in range(3):
            for _ in range(self.config["simulation_frequency"]):
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])

        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            start, end = spawn_edges[self.np_random.integers(len(spawn_edges))]
            lane = self.road.network.get_lane((start, end, 0))
            longitudinal = np.clip(
                30.0 + 10.0 * self.np_random.normal(), 0.0, lane.length - 5.0
            )

            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                lane.position(longitudinal, 0.0),
                speed=lane.speed_limit,
                heading=lane.heading_at(longitudinal),
            )
            destination = self.config["destination"] or self._next_node(end)
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            self._prevent_initial_overlap(ego_vehicle)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0.0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float | None = None,
        spawn_edges: list[tuple[str, str]] | None = None,
    ) -> Vehicle | None:
        spawn_probability = (
            spawn_probability
            if spawn_probability is not None
            else self.config["spawn_probability"]
        )
        if self.np_random.uniform() > spawn_probability:
            return None

        edges = spawn_edges or self._spawn_edges()
        start, end = edges[self.np_random.integers(len(edges))]
        lane = self.road.network.get_lane((start, end, 0))

        longitudinal_sample = (
            longitudinal + 5.0 + self.np_random.normal() * position_deviation
        )
        longitudinal_sample = np.clip(longitudinal_sample, 0.0, lane.length - 5.0)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            (start, end, 0),
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

        destination = self._next_node(end)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

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

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 10.0) -> bool:
        if vehicle.lane is None:
            return False
        longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
        return (
            vehicle.route is None
            and longitudinal >= vehicle.lane.length - exit_distance
        )

    def _prevent_initial_overlap(self, ego_vehicle: Vehicle) -> None:
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle is ego_vehicle
            or np.linalg.norm(vehicle.position - ego_vehicle.position) >= 20.0
        ]

    def _spawn_edges(self) -> list[tuple[str, str]]:
        ordered_vertices = self.VERTEX_ORDER
        edges: list[tuple[str, str]] = []
        for index, start in enumerate(ordered_vertices):
            end = ordered_vertices[(index + 1) % len(ordered_vertices)]
            edges.append((start, end))
            edges.append((end, start))
        return edges

    def _next_node(self, node: str) -> str:
        index = self.VERTEX_ORDER.index(node)
        return self.VERTEX_ORDER[(index + 1) % len(self.VERTEX_ORDER)]
