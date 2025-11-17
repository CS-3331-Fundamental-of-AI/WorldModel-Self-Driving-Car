import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np
import pytest

from env.graph_env import GraphEnv


def sample_graph():
    return {
        "nodes": {
            "a": (0.0, 0.0),
            "b": (40.0, 0.0),
            "c": (40.0, 35.0),
            "d": (0.0, 35.0),
        },
        "edges": [
            {"start": "a", "end": "b", "reverse": True},
            {"start": "b", "end": "c", "reverse": True},
            {"start": "c", "end": "d", "reverse": True},
            {"start": "d", "end": "a", "reverse": True},
        ],
    }


@pytest.fixture
def env():
    environment = GraphEnv(
        config={
            "graph": sample_graph(),
            "controlled_vehicles": 1,
            "initial_vehicle_count": 4,
            "spawn_probability": 0.0,
        }
    )
    yield environment
    environment.close()


def test_network_built(env: GraphEnv):
    graph = env.road.network.graph
    assert set(graph.keys()) == {"a", "b", "c", "d"}
    assert "b" in graph["a"]
    lane = graph["a"]["b"][0]
    positions = np.array(
        [lane.position(s, 0.0) for s in np.linspace(0.0, lane.length, num=5)]
    )
    assert not np.allclose(positions[0], sample_graph()["nodes"]["a"])


def test_step(env: GraphEnv):
    observation, _ = env.reset(seed=123)
    assert env.observation_space.contains(observation)

    for _ in range(5):
        observation, _, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break


def test_manual_registration():
    try:
        gym.register(id="custom-graph-v0", entry_point="graph_env:GraphEnv")
    except gym_error.Error:
        pass
    environment = gym.make(
        "custom-graph-v0",
        config={
            "graph": sample_graph(),
            "controlled_vehicles": 1,
            "initial_vehicle_count": 3,
        },
    )
    try:
        obs, _ = environment.reset(seed=0)
        assert environment.observation_space.contains(obs)
    finally:
        environment.close()
