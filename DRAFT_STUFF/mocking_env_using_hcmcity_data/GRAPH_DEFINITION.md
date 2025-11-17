## Graph Definition Guide

This note describes how to build the `graph` configuration used by `graph_env.GraphEnv`.

### 1. Overall Structure

A graph is a dictionary with two keys:

```python
"graph": {
    "nodes": {...},
    "edges": [...],
}
```

### 2. Nodes
- Maps node IDs to coordinates in metres: `{node_id: (x, y)}`.
- Coordinates are in the same plane HighwayEnv uses (x to the right, y up).
- They mark junctions, lane starts/ends, or helper points for curved paths.

Example:
```python
"nodes": {
    "A": (0.0, 0.0),
    "B": (60.0, 0.0),
    "C": (60.0, 45.0),
    "D": (0.0, 45.0),
}
```

### 3. Edges (Lanes)
Each entry describes one lane from `start` node to `end` node. Keys:

| Key        | Type    | Meaning                                                         |
|------------|---------|-----------------------------------------------------------------|
| `start`    | string  | Source node id                                                  |
| `end`      | string  | Destination node id                                             |
| `reverse`  | bool    | If `True`, creates a mirrored lane from `end` back to `start`  |
| `shape`    | dict    | Optional geometry override                                      |
| `width`    | float   | Optional lane width override (default is HighwayEnv width)      |
| `speed_limit` | float | Optional lane speed limit                                      |

#### Shape options
- **Straight** (default): `shape` omitted or `{"type": "straight"}`.
- **Polyline**:
  ```python
  "shape": {
      "type": "polyline",
      "points": [(x0, y0), (x1, y1), ...],
  }
  ```
  Use polylines to hand-design curves or complex geometry. The first/last points do not need to exactly match the node coordinates; GraphEnv will insert them if missing.
- **Circular**:
  ```python
  "shape": {
      "type": "circular",
      "center": (cx, cy),
      "radius": r,
      "start_angle": angle0,
      "end_angle": angle1,
      "clockwise": False,  # optional
  }
  ```

### 4. Auto-curving vs manual polylines
- `auto_curve=True` (default) attempts to add fillets whenever a node has exactly one incoming and one outgoing edge. Use it for simple chains.
- For intersections with multiple options, set `auto_curve=False` and supply `polyline` shapes to avoid overlapping turns.

### 5. Complete Example (Y-shaped loop)

```python
graph_config = {
    "nodes": {
        "Left_far": (-100.0, -10.0),
        "Right_far": (100.0, -10.0),
        "Top_far": (0.0, 130.0),
        "Left_mid": (-50.0, -5.0),
        "Right_mid": (50.0, -5.0),
        "Top_mid": (0.0, 80.0),
    },
    "edges": [
        {"start": "Left_far", "end": "Left_mid", "reverse": True},
        {"start": "Right_far", "end": "Right_mid", "reverse": True},
        {"start": "Top_far", "end": "Top_mid", "reverse": True},
        {
            "start": "Left_mid",
            "end": "Top_mid",
            "reverse": True,
            "shape": {
                "type": "polyline",
                "points": [
                    (-50.0, -5.0),
                    (-38.0, 8.0),
                    (-26.0, 20.0),
                    (-14.0, 34.0),
                    (-4.0, 48.0),
                    (0.0, 80.0),
                ],
            },
        },
        # ... additional edges ...
    ],
}
```

The example creates a smooth Y-cycle by assigning explicit polylines for each turn. Set `auto_curve=False` in the GraphEnv config to ensure these polylines are used without adjustment.

### 6. Tips
- For bidirectional roads, add `reverse=True` instead of duplicating the edge by hand.
- When designing curves, sample the polyline points every ~5–10 metres to keep the lane smooth.
- Keep enough distance between consecutive points (`> 0.5m`) to avoid numerical issues.
- If a lane still appears straight, double-check that your `auto_curve` setting matches the intention and that the polyline points start/end near the node coordinates.
