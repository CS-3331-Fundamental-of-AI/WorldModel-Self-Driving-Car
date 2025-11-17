## World-Model-Project

This repository explores world models for autonomous driving. Alongside the upstream `highway-env` scenarios, I ship two custom environments—`GraphEnv` and `TriangularEnv`—designed for flexible road layouts and quick experimentation.

### GraphEnv (`graph_env.py`)
- **Purpose:** construct a HighwayEnv-compatible road network from a user-specified graph.
- **Configuration**
  - `graph.nodes`: mapping of `{node_id: (x, y)}` coordinates.
  - `graph.edges`: list of lane specs containing `start`, `end`, optional `reverse`, and optional `shape`.
    - `shape={"type": "straight"}` (default) builds a straight lane.
    - `shape={"type": "polyline", "points": [...]}` lets you supply explicit curved geometry.
    - `shape={"type": "circular", ...}` creates a `CircularLane`.
  - `auto_curve`: when `True`, GraphEnv inserts simple fillets for nodes with a single inbound/outbound edge. Set it to `False` when you provide custom polylines for complex junctions.
  - Standard HighwayEnv parameters (`collision_reward`, `spawn_probability`, etc.) still apply.
- **Usage**
  ```python
  from graph_env import GraphEnv

  config = GraphEnv.default_config()
  config.update({
      "render_mode": "human",
      "graph": custom_graph_definition,
  })

  env = GraphEnv(config=config)
  obs, info = env.reset()
  ```
  When creating the env directly (not through `gym.make`), pass `render_mode="human"` to get a pygame window.
- **Tests** (`test_graph_env.py`)
  - `test_network_built` verifies the sample graph creates non-overlapping lanes.
  - `test_step` runs a short rollout, ensuring observations stay within bounds.
  - `test_manual_registration` registers `graph_env:GraphEnv` with Gym and confirms `gym.make` works.
  Run them with `pytest -v test_graph_env.py` inside the project’s virtualenv.

### TriangularEnv (`triangular_env.py`)
- **Purpose:** a rounded, Y-shaped loop derived from `IntersectionEnv`, useful as a starting point for bespoke scenarios.
- **Highlights**
  - Predefined `PolyLaneFixedWidth` curves keep the ego vehicle on a smooth triangular cycle.
  - Inherits HighwayEnv observation/action spaces, so agents trained on stock environments plug in immediately.
- **Usage**
  ```python
  from triangular_env import TriangularEnv

  env = TriangularEnv(config={"render_mode": "human"})
  obs, info = env.reset()
  ```
  Register manually if you prefer `gym.make`, e.g.:
  ```python
  import gymnasium as gym
  gym.register("triangular-v0", entry_point="triangular_env:TriangularEnv")
  ```
- **Tests** (`test_triangular_env.py`)
  - `test_triangular_network_structure` checks the rounded lanes are `PolyLaneFixedWidth`.
  - `test_triangular_controlled_vehicle_routes` and `test_spawn_vehicle_assigns_triangle_route` validate routing logic.
  - `test_triangular_env_step` steps through several actions and ensures observation validity.

### Running the Tests
Activate the virtual environment first:
```bash
source worldmodel31/bin/activate
```
Then execute:
```bash
pytest -v test_graph_env.py
pytest -v test_triangular_env.py
```
Both suites complete quickly and catch geometry or spawning regressions.

### Tips
- Use explicit `polyline` shapes for complex junctions. Automatic fillets only work when a node has a single inbound/outbound option.
- Set `auto_curve=False` in `GraphEnv` whenever you hand-author curves to prevent unwanted adjustments.
- The usual HighwayEnv reward knobs (`collision_reward`, `high_speed_reward`, `arrived_reward`) remain configurable through `config`.
