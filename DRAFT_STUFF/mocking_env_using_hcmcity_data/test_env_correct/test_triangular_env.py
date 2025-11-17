import pytest

from highway_env.road.lane import PolyLaneFixedWidth

from env.triangular_env import TriangularEnv


@pytest.fixture
def triangular_env():
    env = TriangularEnv(
        config={
            "controlled_vehicles": 1,
            "initial_vehicle_count": 4,
            "spawn_probability": 0.0,
        }
    )
    yield env
    env.close()


def test_triangular_network_structure(triangular_env: TriangularEnv):
    nodes = set(triangular_env.road.network.graph.keys())
    assert nodes == {"n0", "n1", "n2"}

    for start in nodes:
        outgoing = set(triangular_env.road.network.graph[start].keys())
        assert outgoing == nodes - {start}

    sample_lane = triangular_env.road.network.graph["n0"]["n1"][0]
    assert isinstance(sample_lane, PolyLaneFixedWidth)


def test_triangular_controlled_vehicle_routes(triangular_env: TriangularEnv):
    assert len(triangular_env.controlled_vehicles) == triangular_env.config[
        "controlled_vehicles"
    ]
    for vehicle in triangular_env.controlled_vehicles:
        assert vehicle.route is None or len(vehicle.route) >= 1


def test_triangular_env_step(triangular_env: TriangularEnv):
    obs, info = triangular_env.reset(seed=42)
    assert triangular_env.observation_space.contains(obs)

    for _ in range(5):
        action = triangular_env.action_space.sample()
        obs, reward, terminated, truncated, info = triangular_env.step(action)
        assert triangular_env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_spawn_vehicle_assigns_triangle_route():
    env = TriangularEnv(
        config={
            "controlled_vehicles": 1,
            "initial_vehicle_count": 1,
            "spawn_probability": 1.0,
        }
    )
    try:
        # ensure background vehicles do not interfere with spacing checks
        controlled = list(env.controlled_vehicles)
        env.controlled_vehicles.clear()
        env.road.vehicles = [
            vehicle for vehicle in env.road.vehicles if vehicle not in controlled
        ]

        vehicle = env._spawn_vehicle(longitudinal=10.0, spawn_probability=1.0)
        assert vehicle is not None
        assert vehicle.lane_index[0] in env.VERTEX_ORDER
        assert vehicle.lane_index[1] in env.VERTEX_ORDER
        assert vehicle.lane_index[0] != vehicle.lane_index[1]
        assert vehicle.route is None or len(vehicle.route) >= 1
    finally:
        env.close()
