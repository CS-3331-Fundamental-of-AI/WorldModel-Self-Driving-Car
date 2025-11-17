#!/usr/bin/env python3
"""
Convert HCMC map tiles into GraphEnv configuration files.

Usage:
    python convert_tiles_to_graph.py \
        --input-dir hcmc_map_tiles \
        --output-dir graph_configs

Each input JSON should contain a "nodes" list (with x/y coordinates) and a
"links" list describing source/target pairs. The script normalises coordinates,
optionally rescales them, and writes a GraphEnv config JSON suitable for the
`graph_env.GraphEnv` environment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import isclose
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from env.graph_env import GraphEnv


def build_graph(
    tile: dict[str, Any],
    *,
    recenter: bool = True,
    scale: float = 1000.0,
    connection_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a GraphEnv-compatible graph from a tile description."""
    raw_nodes: Iterable[dict[str, Any]] = tile.get("nodes", [])
    if not raw_nodes:
        raise ValueError("Tile has no nodes.")

    xs = [node["x"] for node in raw_nodes]
    ys = [node["y"] for node in raw_nodes]
    offset_x = min(xs) if recenter else 0.0
    offset_y = min(ys) if recenter else 0.0

    def transform(x_val: float, y_val: float) -> tuple[float, float]:
        tx = (x_val - offset_x) * scale
        ty = (y_val - offset_y) * scale
        return round(tx, 5), round(ty, 5)

    nodes: dict[str, tuple[float, float]] = {}
    for node in raw_nodes:
        node_id = str(node["id"])
        nodes[node_id] = transform(float(node["x"]), float(node["y"]))

    default_reverse = not tile.get("directed", True)

    street_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    link_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    edges_raw: list[dict[str, Any]] = []
    raw_edge_street: list[str] = []

    for link in tile.get("links", []):
        street_id = link.get("street_id")
        street_key = (
            str(street_id)
            if street_id is not None
            else f"segment_{link.get('segment_id', 'unknown')}"
        )
        street_groups[street_key].append(link)

        start = str(link["source"])
        end = str(link["target"])
        link_map[(start, end)].append(link)

        edge: dict[str, Any] = {"start": start, "end": end, "reverse": default_reverse}

        if link.get("max_velocity") is not None:
            edge["speed_limit"] = float(link["max_velocity"]) * (1000.0 / 3600.0)

        if link.get("length") is not None:
            edge["length_hint"] = float(link["length"])

        metadata_keys = [
            "segment_id",
            "street_id",
            "street_name",
            "street_type",
            "street_level",
        ]
        metadata = {key: link[key] for key in metadata_keys if key in link}
        metadata["street_group"] = street_key
        edge["metadata"] = metadata

        edges_raw.append(edge)
        raw_edge_street.append(street_key)

    base_graph = nx.DiGraph()
    for (src, dst), link_list in link_map.items():
        for link in link_list:
            street_id = link.get("street_id")
            street_key = (
                str(street_id)
                if street_id is not None
                else f"segment_{link.get('segment_id', 'unknown')}"
            )
            base_graph.add_edge(src, dst, street_id=street_key)

    curved_edges: list[dict[str, Any]] = []
    processed_streets: set[str] = set()

    for street_key, links in street_groups.items():
        sub_nodes = {str(link["source"]) for link in links} | {
            str(link["target"]) for link in links
        }
        subgraph = base_graph.subgraph(sub_nodes).copy()
        if subgraph.number_of_edges() == 0:
            continue

        visited_edges: set[tuple[str, str]] = set()
        street_curved: list[dict[str, Any]] = []

        cycles = list(nx.simple_cycles(subgraph))
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            path_nodes = [str(node) for node in cycle]
            path_nodes.append(path_nodes[0])
            polyline = build_polyline_from_nodes(path_nodes, nodes)
            polyline = dedupe_polyline(polyline)
            if len(polyline) < 2:
                continue
            metadata, speed_mps = fetch_link_metadata(
                path_nodes[0], path_nodes[1], link_map
            )
            street_curved.append(
                make_curved_edge(
                    start=path_nodes[0],
                    end=path_nodes[-1],
                    points=polyline,
                    reverse=default_reverse,
                    metadata=metadata,
                    speed_mps=speed_mps,
                    street_key=street_key,
                )
            )
            visited_edges.update(
                (path_nodes[i], path_nodes[i + 1])
                for i in range(len(path_nodes) - 1)
            )

        street_curved.extend(
            build_chained_edges(
                subgraph,
                nodes,
                link_map,
                visited_edges,
                street_key=street_key,
                default_reverse=default_reverse,
            )
        )

        if street_curved:
            curved_edges.extend(street_curved)
            processed_streets.add(street_key)

    if curved_edges:
        remaining_edges = [
            edge
            for edge, street in zip(edges_raw, raw_edge_street)
            if street not in processed_streets
        ]
        edges = curved_edges + remaining_edges
    else:
        edges = edges_raw

    if connection_overrides:
        edges.extend(
            build_connection_edges(
                nodes,
                connection_overrides,
                radius_default=float(connection_overrides.get("_radius", 25.0)),
                samples_default=int(connection_overrides.get("_samples", 8)),
            )
        )

    return {"nodes": nodes, "edges": edges}


def build_chained_edges(
    subgraph: nx.DiGraph,
    nodes: dict[str, tuple[float, float]],
    link_map: dict[tuple[str, str], list[dict[str, Any]]],
    visited_edges: set[tuple[str, str]],
    *,
    street_key: str,
    default_reverse: bool,
) -> list[dict[str, Any]]:
    """Construct composite edges by chaining straight segments for a street."""

    def successors_unvisited(node: str) -> list[str]:
        return [nbr for nbr in subgraph.successors(node) if (node, nbr) not in visited_edges]

    chain_edges: list[dict[str, Any]] = []

    starts = [
        node
        for node in subgraph.nodes
        if subgraph.in_degree(node) != 1 or subgraph.out_degree(node) != 1
    ]

    for start in starts:
        if not successors_unvisited(start):
            continue
        path = [start]
        current = start
        while True:
            nxt_candidates = successors_unvisited(current)
            if len(nxt_candidates) != 1:
                break
            nxt = nxt_candidates[0]
            visited_edges.add((current, nxt))
            path.append(nxt)
            if subgraph.out_degree(nxt) != 1 or subgraph.in_degree(nxt) != 1:
                current = nxt
                break
            current = nxt

        if len(path) < 2:
            continue

        polyline = build_polyline_from_nodes(path, nodes)
        polyline = dedupe_polyline(polyline)
        if len(polyline) < 2:
            continue

        metadata, speed_mps = fetch_link_metadata(path[0], path[1], link_map)
        chain_edges.append(
            make_curved_edge(
                start=path[0],
                end=path[-1],
                points=polyline,
                reverse=default_reverse,
                metadata=metadata,
                speed_mps=speed_mps,
                street_key=street_key,
            )
        )

    for start, end in list(subgraph.edges()):
        if (start, end) in visited_edges:
            continue
        visited_edges.add((start, end))
        polyline = build_polyline_from_nodes([start, end], nodes)
        metadata, speed_mps = fetch_link_metadata(start, end, link_map)
        chain_edges.append(
            make_curved_edge(
                start=start,
                end=end,
                points=polyline,
                reverse=default_reverse,
                metadata=metadata,
                speed_mps=speed_mps,
                street_key=street_key,
            )
        )

    return chain_edges


def build_polyline_from_nodes(
    node_sequence: Iterable[str],
    nodes: dict[str, tuple[float, float]],
) -> list[tuple[float, float]]:
    return [tuple(nodes[node_id]) for node_id in node_sequence if node_id in nodes]


def fetch_link_metadata(
    src: str,
    dst: str,
    link_map: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any], float | None]:
    links = link_map.get((src, dst))
    if not links:
        return {}, None
    link = links[0]
    metadata = {
        "segment_id": link.get("segment_id"),
        "street_id": link.get("street_id"),
        "street_name": link.get("street_name"),
        "street_type": link.get("street_type"),
        "street_level": link.get("street_level"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    max_vel = link.get("max_velocity")
    speed_mps = None
    if max_vel is not None:
        speed_mps = float(max_vel) * (1000.0 / 3600.0)
    return metadata, speed_mps


def make_curved_edge(
    *,
    start: str,
    end: str,
    points: list[tuple[float, float]],
    reverse: bool,
    metadata: dict[str, Any],
    speed_mps: float | None,
    street_key: str,
) -> dict[str, Any]:
    poly_points = points if len(points) >= 2 else points + points
    length = 0.0
    for idx in range(len(poly_points) - 1):
        p0 = np.array(poly_points[idx])
        p1 = np.array(poly_points[idx + 1])
        length += float(np.linalg.norm(p1 - p0))

    edge = {
        "start": start,
        "end": end,
        "reverse": reverse,
        "shape": {
            "type": "polyline",
            "points": poly_points,
        },
        "metadata": {**metadata, "street_group": street_key, "composite": True},
        "length_hint": length,
    }
    if speed_mps is not None:
        edge["speed_limit"] = speed_mps
    return edge


def build_connection_edges(
    nodes: dict[str, tuple[float, float]],
    overrides: dict[str, Any],
    *,
    radius_default: float,
    samples_default: int,
) -> list[dict[str, Any]]:
    """Create additional edges with curved polylines based on connection metadata."""

    def get_coord(node_id: str) -> np.ndarray:
        if node_id not in nodes:
            raise KeyError(f"Connection references unknown node '{node_id}'.")
        return np.array(nodes[node_id], dtype=float)

    extra_edges: list[dict[str, Any]] = []
    for via_node, incoming_map in overrides.items():
        if via_node.startswith("_"):
            # Special keys such as _radius are handled elsewhere.
            continue
        via_coord = get_coord(via_node)

        for incoming_node, outgoing_map in incoming_map.items():
            in_coord = get_coord(incoming_node)
            for outgoing_node, params in outgoing_map.items():
                out_coord = get_coord(outgoing_node)

                radius = float(params.get("radius", radius_default))
                samples = int(params.get("samples", samples_default))
                reverse = bool(params.get("reverse", False))

                polyline = make_connection_polyline(
                    in_coord, via_coord, out_coord, radius=radius, samples=samples
                )
                polyline = dedupe_polyline(
                    [tuple(in_coord)] + polyline + [tuple(out_coord)]
                )

                extra_edges.append(
                    {
                        "start": incoming_node,
                        "end": outgoing_node,
                        "reverse": reverse,
                        "shape": {
                            "type": "polyline",
                            "points": polyline,
                        },
                        "metadata": {
                            "connection_via": via_node,
                            "radius": radius,
                        },
                    }
                )

    return extra_edges


def make_connection_polyline(
    prev: np.ndarray,
    corner: np.ndarray,
    nxt: np.ndarray,
    *,
    radius: float,
    samples: int,
) -> list[tuple[float, float]]:
    """Return points describing a curved turn from prev→corner→next."""
    vec_in = corner - prev
    vec_out = nxt - corner
    len_in = np.linalg.norm(vec_in)
    len_out = np.linalg.norm(vec_out)
    if len_in < 1e-6 or len_out < 1e-6:
        return []

    dir_in = vec_in / len_in
    dir_out = vec_out / len_out

    dot_prod = np.clip(np.dot(-dir_in, dir_out), -1.0, 1.0)
    angle = np.arccos(dot_prod)
    if angle < 1e-3 or isclose(angle, np.pi, rel_tol=1e-3):
        return []

    offset = radius / np.tan(angle / 2.0)
    offset = min(offset, len_in * 0.5, len_out * 0.5)
    if offset <= 1e-3:
        return []

    entry = corner - dir_in * offset
    exit_point = corner + dir_out * offset

    bisector = (-dir_in) + dir_out
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm < 1e-6:
        return []
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
        (float(center[0] + radius * np.cos(theta)), float(center[1] + radius * np.sin(theta)))
        for theta in angles
    ]
    return arc_points


def dedupe_polyline(points: Iterable[tuple[float, float]], min_step: float = 0.5) -> list[tuple[float, float]]:
    """Remove consecutive points closer than min_step."""
    cleaned: list[tuple[float, float]] = []
    last = None
    for pt in points:
        if last is None:
            cleaned.append(pt)
            last = np.array(pt)
            continue
        if np.linalg.norm(np.array(pt) - last) >= min_step:
            cleaned.append(pt)
            last = np.array(pt)
    if len(cleaned) < 2:
        if last is None:
            return []
        direction = np.array([1.0, 0.0])
        cleaned.append(tuple(last + direction * max(min_step, 1.0)))
    return cleaned


def convert_tile(
    path: Path,
    out_dir: Path,
    *,
    recenter: bool,
    scale: float,
    connection_overrides: dict[str, Any] | None = None,
) -> Path:
    """Convert a single tile file."""
    with path.open("r", encoding="utf-8") as fh:
        tile = json.load(fh)

    graph = build_graph(
        tile,
        recenter=recenter,
        scale=scale,
        connection_overrides=connection_overrides,
    )

    config = GraphEnv.default_config()
    config.update(
        {
            "auto_curve": False,
            "graph": graph,
            "spawn_probability": 0.0,
            "initial_vehicle_count": 0,
            "controlled_vehicles": 0,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_graph.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HCMC map tiles to GraphEnv configuration files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("hcmc_map_tiles"),
        help="Directory containing *.json map tiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graph_configs"),
        help="Directory to write converted graph configurations.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Scale factor applied to the x/y coordinates (default: 1000 -> metres).",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help="Keep original coordinates instead of translating them to start at (0, 0).",
    )
    parser.add_argument(
        "--connections",
        type=Path,
        help=(
            "Optional JSON file describing curved connections. "
            "Structure: {\"default\": {<node>: {<incoming>: {<outgoing>: {\"radius\": r}}}}}."
        ),
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    tile_files = sorted(input_dir.glob("*.json"))
    if not tile_files:
        raise FileNotFoundError(f"No JSON tiles found under {input_dir}")

    connection_specs: dict[str, Any] | None = None
    if args.connections:
        with args.connections.open("r", encoding="utf-8") as fh:
            connection_specs = json.load(fh)

    produced = []
    for tile_path in tile_files:
        overrides = None
        if connection_specs:
            overrides = connection_specs.get(
                tile_path.stem, connection_specs.get("default")
            )
        out_path = convert_tile(
            tile_path,
            output_dir,
            recenter=not args.no_recenter,
            scale=args.scale,
            connection_overrides=overrides,
        )
        produced.append(out_path)
        print(f"Converted {tile_path.name} -> {out_path}")

    print(f"Generated {len(produced)} GraphEnv configs in {output_dir}")


if __name__ == "__main__":
    main()
