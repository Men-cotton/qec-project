#!/usr/bin/env python3
"""
Reconstruct detector coordinates for graph JSON files and visualize them.

The graph JSON files in `graph/` do not store detector coordinates directly.
This script rebuilds the corresponding Stim rotated-memory circuit, recovers
detector coordinates, maps graph node ids back onto detector ids, writes a CSV
table, and saves a plot of the reconstructed graph.

The mapping rule used here is an inference from the stored ids:

- detector-backed graph ids are treated as 1-based Stim detector ids
- `num_detectors + 1` is treated as a virtual boundary node with no Stim coord

This matches the observed id ranges in `graph/`, but it is still an inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import stim


GRAPH_KEY_MAP = {
    "short": "short_subgraph",
    "long": "long_subgraph",
    "full": "fullgraph",
}

NODE_LIST_KEYS = (
    "short_subgraph_node_ids",
    "short_subgraph_boundary_node_ids",
    "long_subgraph_node_ids",
    "long_subgraph_boundary_node_ids",
    "fullgraph_node_ids",
    "fullgraph_boundary_node_ids",
)

FILENAME_RE = re.compile(
    r"graph_data_d(?P<distance>\d+)_r(?P<rounds>\d+)_case_(?P<case>\d+)\.json"
)


@dataclass(frozen=True)
class GraphSpec:
    path: Path
    distance: int
    rounds: int
    case_index: int | None


@dataclass(frozen=True)
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float


@dataclass(frozen=True)
class NodeRecord:
    node_id: int
    role: str
    detector_index: int | None
    x: float | None
    y: float | None
    t: float | None
    x_raw: float | None
    y_raw: float | None
    t_raw: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct detector coordinates for graph JSON files and visualize the result."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Graph JSON file(s), directories, or glob patterns to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graph/reconstructed"),
        help="Directory for reconstructed CSV tables and plots.",
    )
    parser.add_argument(
        "--graph-kind",
        choices=("short", "long", "full", "all"),
        default="all",
        help="Which graph variant to draw.",
    )
    parser.add_argument(
        "--code-task",
        default="surface_code:rotated_memory_z",
        help="Stim generated-circuit task used to rebuild detector coordinates.",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=1e-3,
        help="Dummy error rate passed to Stim. Coordinates are unaffected by this value.",
    )
    parser.add_argument(
        "--include-all-detectors",
        action="store_true",
        help="Write every detector-backed id to the CSV, not just ids referenced by the graph JSON.",
    )
    parser.add_argument(
        "--edge-limit",
        type=int,
        default=5000,
        help="Maximum number of edges to draw per subplot. Use 0 to draw all edges.",
    )
    parser.add_argument(
        "--annotate-ids",
        action="store_true",
        help="Annotate plotted nodes with graph ids.",
    )
    return parser.parse_args()


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        candidate = Path(raw)
        if candidate.is_dir():
            matched = sorted(candidate.glob("*.json"))
        elif candidate.exists():
            matched = [candidate]
        else:
            matched = sorted(Path().glob(raw))
        for path in matched:
            resolved = path.resolve()
            if resolved.suffix != ".json" or resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return sorted(paths)


def load_graph_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def infer_graph_spec(path: Path, graph_data: dict) -> GraphSpec:
    match = FILENAME_RE.fullmatch(path.name)
    if match is not None:
        return GraphSpec(
            path=path,
            distance=int(match.group("distance")),
            rounds=int(match.group("rounds")),
            case_index=int(match.group("case")),
        )
    return GraphSpec(
        path=path,
        distance=int(graph_data["code_distance"]),
        rounds=int(graph_data["measurement_rounds"]),
        case_index=None,
    )


def build_circuit(spec: GraphSpec, code_task: str, error_rate: float) -> stim.Circuit:
    return stim.Circuit.generated(
        code_task,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate,
    )


def reconstruct_detector_table(
    circuit: stim.Circuit,
) -> tuple[dict[int, NodeRecord], int]:
    coords = circuit.get_detector_coordinates()
    table: dict[int, NodeRecord] = {}
    for detector_index, coord in coords.items():
        if len(coord) < 3:
            raise ValueError(f"Expected 3D detector coordinates, got {coord!r}")
        x_raw, y_raw, t_raw = float(coord[0]), float(coord[1]), float(coord[2])
        table[detector_index + 1] = NodeRecord(
            node_id=detector_index + 1,
            role="detector",
            detector_index=detector_index,
            x=x_raw / 2.0,
            y=y_raw / 2.0,
            t=t_raw,
            x_raw=x_raw,
            y_raw=y_raw,
            t_raw=t_raw,
        )
    return table, circuit.num_detectors


def collect_used_node_ids(graph_data: dict) -> set[int]:
    used: set[int] = set()
    for key in GRAPH_KEY_MAP.values():
        for src, dst, _weight in graph_data.get(key, []):
            used.add(int(src))
            used.add(int(dst))
    for key in NODE_LIST_KEYS:
        used.update(int(node_id) for node_id in graph_data.get(key, []))
    if "boundary_node_id" in graph_data:
        used.add(int(graph_data["boundary_node_id"]))
    return used


def classify_node_id(node_id: int, boundary_start_id: int, num_detectors: int) -> str:
    if node_id == num_detectors + 1:
        return "virtual_boundary"
    if node_id >= boundary_start_id:
        return "boundary"
    return "node"


def build_node_records(
    graph_data: dict,
    detector_table: dict[int, NodeRecord],
    num_detectors: int,
    include_all_detectors: bool,
) -> tuple[dict[int, NodeRecord], int]:
    boundary_start_id = int(
        graph_data.get("boundary_node_id", (num_detectors // 2) + 1)
    )
    used_ids = collect_used_node_ids(graph_data)
    if include_all_detectors:
        used_ids.update(range(1, num_detectors + 2))

    records: dict[int, NodeRecord] = {}
    for node_id in sorted(used_ids):
        role = classify_node_id(node_id, boundary_start_id, num_detectors)
        if node_id == num_detectors + 1:
            records[node_id] = NodeRecord(
                node_id=node_id,
                role=role,
                detector_index=None,
                x=None,
                y=None,
                t=None,
                x_raw=None,
                y_raw=None,
                t_raw=None,
            )
            continue
        if node_id not in detector_table:
            raise ValueError(
                f"Node id {node_id} is outside the reconstructed Stim detector range 1..{num_detectors + 1}."
            )
        detector_record = detector_table[node_id]
        records[node_id] = NodeRecord(
            node_id=node_id,
            role=role,
            detector_index=detector_record.detector_index,
            x=detector_record.x,
            y=detector_record.y,
            t=detector_record.t,
            x_raw=detector_record.x_raw,
            y_raw=detector_record.y_raw,
            t_raw=detector_record.t_raw,
        )
    return records, boundary_start_id


def write_mapping_csv(records: dict[int, NodeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "node_id",
                "role",
                "detector_index",
                "x",
                "y",
                "t",
                "x_raw",
                "y_raw",
                "t_raw",
            ),
        )
        writer.writeheader()
        for node_id in sorted(records):
            record = records[node_id]
            writer.writerow(
                {
                    "node_id": record.node_id,
                    "role": record.role,
                    "detector_index": record.detector_index,
                    "x": record.x,
                    "y": record.y,
                    "t": record.t,
                    "x_raw": record.x_raw,
                    "y_raw": record.y_raw,
                    "t_raw": record.t_raw,
                }
            )


def compute_bounds(records: dict[int, NodeRecord]) -> Bounds3D:
    real_records = [record for record in records.values() if record.x is not None]
    if not real_records:
        raise ValueError("No real detector coordinates were reconstructed.")
    return Bounds3D(
        x_min=min(record.x for record in real_records if record.x is not None),
        x_max=max(record.x for record in real_records if record.x is not None),
        y_min=min(record.y for record in real_records if record.y is not None),
        y_max=max(record.y for record in real_records if record.y is not None),
        t_min=min(record.t for record in real_records if record.t is not None),
        t_max=max(record.t for record in real_records if record.t is not None),
    )


def synthetic_virtual_boundary_point(
    bounds: Bounds3D, time_value: float
) -> tuple[float, float, float]:
    return (bounds.x_max + 1.5, (bounds.y_min + bounds.y_max) / 2.0, time_value)


def limit_edges(
    edges: list[list[int]], edge_limit: int
) -> tuple[list[list[int]], bool]:
    if edge_limit <= 0 or len(edges) <= edge_limit:
        return edges, False
    stride = len(edges) / float(edge_limit)
    kept = []
    for index in range(edge_limit):
        kept.append(edges[min(int(index * stride), len(edges) - 1)])
    return kept, True


def edge_endpoints(
    src_id: int,
    dst_id: int,
    records: dict[int, NodeRecord],
    bounds: Bounds3D,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    src = records[src_id]
    dst = records[dst_id]

    if src.t is None and dst.t is None:
        mid_t = bounds.t_max
        return synthetic_virtual_boundary_point(
            bounds, mid_t
        ), synthetic_virtual_boundary_point(bounds, mid_t)
    if src.t is None:
        assert dst.t is not None
        return synthetic_virtual_boundary_point(bounds, dst.t), (dst.x, dst.y, dst.t)
    if dst.t is None:
        return (src.x, src.y, src.t), synthetic_virtual_boundary_point(bounds, src.t)
    return (src.x, src.y, src.t), (dst.x, dst.y, dst.t)


def collect_used_records(
    edges: list[list[int]], records: dict[int, NodeRecord]
) -> list[NodeRecord]:
    used_ids: set[int] = set()
    for src_id, dst_id, _weight in edges:
        used_ids.add(int(src_id))
        used_ids.add(int(dst_id))
    return [records[node_id] for node_id in sorted(used_ids)]


def plot_graph_variant(
    ax,
    name: str,
    edges: list[list[int]],
    records: dict[int, NodeRecord],
    bounds: Bounds3D,
    annotate_ids: bool,
    edge_limit: int,
) -> None:
    limited_edges, was_limited = limit_edges(edges, edge_limit)
    used_records = collect_used_records(limited_edges, records)

    role_style = {
        "node": ("tab:blue", "o"),
        "boundary": ("tab:orange", "s"),
        "virtual_boundary": ("tab:red", "^"),
    }

    for role, (color, marker) in role_style.items():
        xs: list[float] = []
        ys: list[float] = []
        ts: list[float] = []
        if role == "virtual_boundary":
            times = sorted(
                {
                    records[src_id].t
                    if records[src_id].t is not None
                    else records[dst_id].t
                    for src_id, dst_id, _weight in limited_edges
                    if records[src_id].role == "virtual_boundary"
                    or records[dst_id].role == "virtual_boundary"
                }
            )
            for time_value in times:
                if time_value is None:
                    continue
                x, y, t = synthetic_virtual_boundary_point(bounds, time_value)
                xs.append(x)
                ys.append(y)
                ts.append(t)
        else:
            for record in used_records:
                if record.role != role or record.x is None:
                    continue
                xs.append(record.x)
                ys.append(record.y)
                ts.append(record.t if record.t is not None else 0.0)
        if xs:
            ax.scatter(xs, ys, ts, color=color, marker=marker, s=28, label=role)

    if annotate_ids:
        for record in used_records:
            if record.x is None or record.t is None:
                continue
            ax.text(record.x, record.y, record.t, str(record.node_id), fontsize=6)

    for src_id, dst_id, weight in limited_edges:
        (x0, y0, t0), (x1, y1, t1) = edge_endpoints(
            int(src_id), int(dst_id), records, bounds
        )
        line_alpha = min(0.85, 0.18 + 0.1 * math.log1p(float(weight)))
        ax.plot(
            (x0, x1), (y0, y1), (t0, t1), color="0.35", alpha=line_alpha, linewidth=0.8
        )

    subtitle = name
    if was_limited:
        subtitle += f" (showing {len(limited_edges)}/{len(edges)} edges)"
    ax.set_title(subtitle)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_xlim(bounds.x_min - 0.5, bounds.x_max + 2.5)
    ax.set_ylim(bounds.y_min - 0.5, bounds.y_max + 0.5)
    ax.set_zlim(bounds.t_min - 0.2, bounds.t_max + 0.2)


def plot_reconstruction(
    graph_data: dict,
    records: dict[int, NodeRecord],
    graph_kind: str,
    annotate_ids: bool,
    edge_limit: int,
    output_path: Path,
) -> None:
    bounds = compute_bounds(records)
    selected_kinds = list(GRAPH_KEY_MAP) if graph_kind == "all" else [graph_kind]
    figure = plt.figure(figsize=(7 * len(selected_kinds), 6))

    for subplot_index, kind in enumerate(selected_kinds, start=1):
        ax = figure.add_subplot(1, len(selected_kinds), subplot_index, projection="3d")
        plot_graph_variant(
            ax=ax,
            name=kind,
            edges=graph_data[GRAPH_KEY_MAP[kind]],
            records=records,
            bounds=bounds,
            annotate_ids=annotate_ids,
            edge_limit=edge_limit,
        )

    handles, labels = figure.axes[0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        figure.legend(unique.values(), unique.keys(), loc="upper right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def process_file(args: argparse.Namespace, path: Path) -> tuple[Path, Path]:
    graph_data = load_graph_json(path)
    spec = infer_graph_spec(path, graph_data)
    circuit = build_circuit(spec, args.code_task, args.error_rate)
    detector_table, num_detectors = reconstruct_detector_table(circuit)
    records, boundary_start_id = build_node_records(
        graph_data=graph_data,
        detector_table=detector_table,
        num_detectors=num_detectors,
        include_all_detectors=args.include_all_detectors,
    )

    output_stem = args.output_dir / path.stem
    mapping_path = output_stem.with_suffix(".mapping.csv")
    plot_suffix = (
        ".overview.png" if args.graph_kind == "all" else f".{args.graph_kind}.png"
    )
    plot_path = output_stem.with_suffix(plot_suffix)

    write_mapping_csv(records, mapping_path)
    plot_reconstruction(
        graph_data=graph_data,
        records=records,
        graph_kind=args.graph_kind,
        annotate_ids=args.annotate_ids,
        edge_limit=args.edge_limit,
        output_path=plot_path,
    )

    print(
        f"{path.name}: distance={spec.distance} rounds={spec.rounds} "
        f"num_detectors={num_detectors} boundary_start_id={boundary_start_id} "
        f"mapping={mapping_path} plot={plot_path}"
    )
    return mapping_path, plot_path


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No JSON inputs matched.")
    for path in input_paths:
        process_file(args, path)


if __name__ == "__main__":
    main()
