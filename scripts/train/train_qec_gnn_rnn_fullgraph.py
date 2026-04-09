#!/usr/bin/env python3
"""
Train the QEC_GNN-RNN GRU decoder on graph/fullgraph samples without
modifying files under NN-based/QEC_GNN-RNN.

This script reimplements only the dataset and training loop in scripts/.
The model definition is imported from NN-based/QEC_GNN-RNN at runtime.

Default labels use `imperfect_fmu_is_valid` because the graph JSON files do
not contain logical-observable labels. To train on logical flips instead,
provide `--label-manifest`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GRAPH_KEY_MAP = {
    "short": "short_subgraph",
    "long": "long_subgraph",
    "full": "fullgraph",
}

NODE_LIST_KEYS = {
    "short": ("short_subgraph_node_ids", "short_subgraph_boundary_node_ids"),
    "long": ("long_subgraph_node_ids", "long_subgraph_boundary_node_ids"),
    "full": ("fullgraph_node_ids", "fullgraph_boundary_node_ids"),
}

FILENAME_RE = re.compile(
    r"graph_data_d(?P<distance>\d+)_r(?P<rounds>\d+)_case_(?P<case>\d+)\.json"
)


@dataclass(frozen=True)
class GraphSpec:
    path: Path
    distance: int
    rounds: int


@dataclass(frozen=True)
class GraphSample:
    path: Path
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QEC_GNN-RNN on graph/fullgraph samples from scripts/."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Graph JSON files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--graph-kind",
        choices=("short", "long", "full"),
        default="full",
        help="Which stored graph variant to use.",
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        help=(
            "Optional JSON file containing labels keyed by absolute path, relative path, "
            "filename, or stem. If omitted, --target-field is read from each graph JSON."
        ),
    )
    parser.add_argument(
        "--target-field",
        default="imperfect_fmu_is_valid",
        help=(
            "Boolean or 0/1 field to use as the training target when --label-manifest "
            "is not provided. Default is imperfect_fmu_is_valid so the script can run "
            "against the existing repo data."
        ),
    )
    parser.add_argument(
        "--weight-mode",
        choices=("raw", "inverse", "inverse_square", "unit"),
        default="inverse",
        help="How to convert stored graph weights into edge_attr.",
    )
    parser.add_argument(
        "--code-task",
        default="surface_code:rotated_memory_z",
        help="Stim generated-circuit task used to reconstruct detector coordinates.",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=1e-3,
        help="Dummy error rate passed to Stim during coordinate reconstruction.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of samples used for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Shuffle seed for the train/validation split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of graph JSON files to load after shuffling.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional output path for torch.save(model.state_dict(), ...).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build and summarize samples. Does not import torch or train.",
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


def require_module(name: str):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(
            f"Missing dependency `{name}` in the current environment. "
            f"Run this script via `uv run` in an environment where `{name}` is installed."
        )


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
        )
    return GraphSpec(
        path=path,
        distance=int(graph_data["code_distance"]),
        rounds=int(graph_data["measurement_rounds"]),
    )


def build_circuit(spec: GraphSpec, code_task: str, error_rate: float):
    require_module("stim")
    import stim

    return stim.Circuit.generated(
        code_task,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=error_rate,
        after_reset_flip_probability=error_rate,
        before_measure_flip_probability=error_rate,
        before_round_data_depolarization=error_rate,
    )


def reconstruct_detector_table(circuit) -> tuple[dict[int, tuple[float, float, float]], int]:
    coords = circuit.get_detector_coordinates()
    table: dict[int, tuple[float, float, float]] = {}
    for detector_index, coord in coords.items():
        if len(coord) < 3:
            raise ValueError(f"Expected 3D detector coordinates, got {coord!r}")
        table[detector_index + 1] = (
            float(coord[0]) / 2.0,
            float(coord[1]) / 2.0,
            float(coord[2]),
        )
    return table, circuit.num_detectors


def compute_syndrome_mask(distance: int) -> np.ndarray:
    sz = distance + 1
    syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
    syndrome_x[::2, 1 : sz - 1 : 2] = 1
    syndrome_x[1::2, 2::2] = 1
    syndrome_z = np.rot90(syndrome_x) * 3
    return syndrome_x + syndrome_z


def collect_used_node_ids(graph_data: dict, graph_kind: str) -> set[int]:
    used: set[int] = set()
    graph_key = GRAPH_KEY_MAP[graph_kind]
    for src, dst, _weight in graph_data.get(graph_key, []):
        used.add(int(src))
        used.add(int(dst))
    node_key, boundary_key = NODE_LIST_KEYS[graph_kind]
    used.update(int(node_id) for node_id in graph_data.get(node_key, []))
    used.update(int(node_id) for node_id in graph_data.get(boundary_key, []))
    return used


def collect_neighbors(edges: list[list[int]]) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {}
    for src, dst, _weight in edges:
        src = int(src)
        dst = int(dst)
        neighbors.setdefault(src, []).append(dst)
        neighbors.setdefault(dst, []).append(src)
    return neighbors


def synthetic_virtual_coord(
    neighbors: list[int],
    real_coords: dict[int, tuple[float, float, float]],
) -> tuple[float, float, float]:
    real_neighbor_coords = [real_coords[n] for n in neighbors if n in real_coords]
    if not real_neighbor_coords:
        return (0.0, 0.0, 0.0)
    xs = [coord[0] for coord in real_neighbor_coords]
    ys = [coord[1] for coord in real_neighbor_coords]
    ts = [coord[2] for coord in real_neighbor_coords]
    return (max(xs) + 1.0, float(sum(ys) / len(ys)), float(sum(ts) / len(ts)))


def infer_stabilizer_type(
    x: float,
    y: float,
    mask: np.ndarray,
) -> tuple[float, float]:
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
        value = int(mask[yi, xi])
        if value == 3:
            return (1.0, 0.0)
        if value == 1:
            return (0.0, 1.0)
    parity = (xi + yi) % 2
    return (float(parity == 0), float(parity == 1))


def transform_edge_weight(weight: float, mode: str) -> float:
    weight = float(weight)
    if mode == "raw":
        return weight
    if mode == "inverse":
        return 1.0 / max(weight, 1e-6)
    if mode == "inverse_square":
        return 1.0 / max(weight, 1e-6) ** 2
    if mode == "unit":
        return 1.0
    raise ValueError(f"Unknown weight mode: {mode}")


def load_label_manifest(path: Path) -> dict[str, float]:
    with path.open() as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for row in raw:
            if not isinstance(row, dict) or "label" not in row:
                raise ValueError("Label manifest rows must be objects with a `label` field.")
            key = row.get("path") or row.get("name") or row.get("stem")
            if key is None:
                raise ValueError("Each label manifest row must define path, name, or stem.")
            items.append((key, row["label"]))
    else:
        raise ValueError("Label manifest must be a JSON object or a JSON list.")
    labels: dict[str, float] = {}
    for key, value in items:
        labels[str(key)] = coerce_label(value, source=f"label manifest entry {key!r}")
    return labels


def coerce_label(value, *, source: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
        return float(value)
    raise ValueError(f"{source} must be boolean or 0/1, got {value!r}.")


def resolve_label(path: Path, graph_data: dict, manifest: dict[str, float] | None, target_field: str) -> float:
    if manifest is not None:
        candidates = (
            str(path.resolve()),
            str(path),
            path.name,
            path.stem,
        )
        for candidate in candidates:
            if candidate in manifest:
                return manifest[candidate]
        raise KeyError(f"No label found in manifest for {path}.")
    if target_field not in graph_data:
        raise KeyError(f"{path} does not define target field {target_field!r}.")
    return coerce_label(graph_data[target_field], source=f"{path.name}:{target_field}")


def build_sample(
    path: Path,
    graph_kind: str,
    code_task: str,
    error_rate: float,
    weight_mode: str,
    manifest: dict[str, float] | None,
    target_field: str,
) -> GraphSample:
    graph_data = load_graph_json(path)
    spec = infer_graph_spec(path, graph_data)
    circuit = build_circuit(spec, code_task, error_rate)
    real_coords, num_detectors = reconstruct_detector_table(circuit)

    graph_key = GRAPH_KEY_MAP[graph_kind]
    edges = graph_data[graph_key]
    neighbors = collect_neighbors(edges)
    used_node_ids = sorted(collect_used_node_ids(graph_data, graph_kind))
    mask = compute_syndrome_mask(spec.distance)

    coords_by_node: dict[int, tuple[float, float, float]] = {}
    for node_id in used_node_ids:
        if node_id in real_coords:
            coords_by_node[node_id] = real_coords[node_id]
        elif node_id == num_detectors + 1:
            coords_by_node[node_id] = synthetic_virtual_coord(neighbors.get(node_id, []), real_coords)
        else:
            raise ValueError(
                f"{path.name}: node id {node_id} is outside reconstructed detector range 1..{num_detectors + 1}."
            )

    node_id_to_local = {node_id: index for index, node_id in enumerate(used_node_ids)}
    node_features = []
    for node_id in used_node_ids:
        x, y, t = coords_by_node[node_id]
        z_type, x_type = infer_stabilizer_type(x, y, mask)
        node_features.append([x, y, t, z_type, x_type])

    directed_edges = []
    directed_weights = []
    for src, dst, weight in edges:
        src_index = node_id_to_local[int(src)]
        dst_index = node_id_to_local[int(dst)]
        edge_attr = transform_edge_weight(weight, weight_mode)
        directed_edges.append((src_index, dst_index))
        directed_weights.append(edge_attr)
        directed_edges.append((dst_index, src_index))
        directed_weights.append(edge_attr)

    if not directed_edges:
        if not used_node_ids:
            raise ValueError(f"{path.name}: graph contains neither nodes nor edges for {graph_kind}.")
        for local_index in range(len(used_node_ids)):
            directed_edges.append((local_index, local_index))
            directed_weights.append(0.0)

    edge_index = np.asarray(directed_edges, dtype=np.int64).T
    edge_attr = np.asarray(directed_weights, dtype=np.float32).reshape(-1, 1)
    x = np.asarray(node_features, dtype=np.float32)
    label = resolve_label(path, graph_data, manifest, target_field)
    return GraphSample(path=path, x=x, edge_index=edge_index, edge_attr=edge_attr, label=label)


def build_samples(args: argparse.Namespace) -> list[GraphSample]:
    manifest = load_label_manifest(args.label_manifest) if args.label_manifest else None
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No graph JSON files matched the provided inputs.")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    if args.limit is not None:
        paths = paths[: args.limit]

    samples = [
        build_sample(
            path=path,
            graph_kind=args.graph_kind,
            code_task=args.code_task,
            error_rate=args.error_rate,
            weight_mode=args.weight_mode,
            manifest=manifest,
            target_field=args.target_field,
        )
        for path in paths
    ]
    return samples


def summarize_samples(samples: list[GraphSample]) -> None:
    labels = np.asarray([sample.label for sample in samples], dtype=np.float32)
    node_counts = np.asarray([sample.x.shape[0] for sample in samples], dtype=np.int64)
    edge_counts = np.asarray([sample.edge_attr.shape[0] // 2 for sample in samples], dtype=np.int64)
    print(f"Loaded {len(samples)} samples")
    print(f"labels: mean={labels.mean():.4f}, positives={int(labels.sum())}, negatives={len(labels) - int(labels.sum())}")
    print(
        "nodes per graph: "
        f"min={node_counts.min()}, median={int(np.median(node_counts))}, max={node_counts.max()}"
    )
    print(
        "edges per graph: "
        f"min={edge_counts.min()}, median={int(np.median(edge_counts))}, max={edge_counts.max()}"
    )


def split_samples(samples: list[GraphSample], train_fraction: float) -> tuple[list[GraphSample], list[GraphSample]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    split_index = max(1, min(len(samples) - 1, int(round(len(samples) * train_fraction))))
    return samples[:split_index], samples[split_index:]


def collate_graph_samples(samples: list[GraphSample]):
    import torch

    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_labels = []
    label_map = []
    flips = []
    node_offset = 0

    for batch_index, sample in enumerate(samples):
        x = torch.from_numpy(sample.x)
        edge_index = torch.from_numpy(sample.edge_index)
        edge_attr = torch.from_numpy(sample.edge_attr)

        x_list.append(x)
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        batch_labels.append(torch.full((x.shape[0],), batch_index, dtype=torch.long))
        label_map.append(torch.tensor([[batch_index, 0]], dtype=torch.long))
        flips.append(torch.tensor([[sample.label]], dtype=torch.float32))
        node_offset += x.shape[0]

    return (
        torch.cat(x_list, dim=0),
        torch.cat(edge_index_list, dim=1),
        torch.cat(batch_labels, dim=0),
        torch.cat(label_map, dim=0),
        torch.cat(edge_attr_list, dim=0),
        torch.cat(flips, dim=0),
    )


def build_model(batch_size: int, lr: float):
    require_module("torch")
    require_module("torch_geometric")

    qec_dir = Path(__file__).resolve().parents[1] / "NN-based" / "QEC_GNN-RNN"
    sys.path.insert(0, str(qec_dir))

    from args import Args  # type: ignore
    from gru_decoder import GRUDecoder  # type: ignore

    args = Args()
    args.batch_size = batch_size
    args.lr = lr
    model = GRUDecoder(args)
    return args, model


def iterate_minibatches(samples: list[GraphSample], batch_size: int, rng: random.Random):
    order = list(range(len(samples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        yield [samples[index] for index in indices]


def move_batch_to_device(batch, device):
    x, edge_index, batch_labels, label_map, edge_attr, flips = batch
    return (
        x.to(device),
        edge_index.to(device),
        batch_labels.to(device),
        label_map.to(device),
        edge_attr.to(device),
        flips.to(device),
    )


def evaluate(model, samples: list[GraphSample], batch_size: int, device) -> tuple[float, float]:
    import torch
    import torch.nn as nn

    loss_fn = nn.BCELoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    if not samples:
        return math.nan, math.nan

    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = collate_graph_samples(samples[start : start + batch_size])
            x, edge_index, batch_labels, label_map, edge_attr, flips = move_batch_to_device(batch, device)
            out = model(x, edge_index, edge_attr, batch_labels, label_map)
            loss = loss_fn(out, flips)
            total_loss += loss.item() * flips.shape[0]
            total_correct += int((torch.round(out) == flips).sum().item())
            total_count += int(flips.numel())
    return total_loss / total_count, total_correct / total_count


def train(args: argparse.Namespace, samples: list[GraphSample]) -> None:
    import torch
    import torch.nn as nn

    train_samples, val_samples = split_samples(samples, args.train_fraction)
    _, model = build_model(batch_size=args.batch_size, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    rng = random.Random(args.seed)
    best_val_acc = -1.0
    best_state = None

    print(f"training samples={len(train_samples)}, validation samples={len(val_samples)}, device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        for minibatch in iterate_minibatches(train_samples, args.batch_size, rng):
            batch = collate_graph_samples(minibatch)
            x, edge_index, batch_labels, label_map, edge_attr, flips = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr, batch_labels, label_map)
            loss = loss_fn(out, flips)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * flips.shape[0]
            epoch_correct += int((torch.round(out) == flips).sum().item())
            epoch_count += int(flips.numel())

        train_loss = epoch_loss / epoch_count
        train_acc = epoch_correct / epoch_count
        val_loss, val_acc = evaluate(model, val_samples, args.batch_size, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"saved model to {args.save_model}")


def main() -> None:
    args = parse_args()
    samples = build_samples(args)
    summarize_samples(samples)
    if args.prepare_only:
        return
    train(args, samples)


if __name__ == "__main__":
    main()
