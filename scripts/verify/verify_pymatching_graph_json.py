#!/usr/bin/env python3
"""
Verify that graph JSON samples can be decoded by PyMatching from scripts/ alone.

This script does not modify or import code from non-NN-based/PyMatching. It uses
the installed `pymatching` package, reconstructs a matching graph directly from
`graph/*.json`, decodes the active syndrome implied by `*_node_ids`, and compares
the result against the stored `*_MWPM_weight` and `*_MWPM_matching`.

The primary correctness check is the MWPM solution weight. Exact matching edges
are also compared, but ties can make edge-level disagreement possible even when
the decoded weight is correct.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GRAPH_KEY_MAP = {
    "short": "short_subgraph",
    "long": "long_subgraph",
    "full": "fullgraph",
}

NODE_KEY_MAP = {
    "short": ("short_subgraph_node_ids", "short_subgraph_boundary_node_ids"),
    "long": ("long_subgraph_node_ids", "long_subgraph_boundary_node_ids"),
    "full": ("fullgraph_node_ids", "fullgraph_boundary_node_ids"),
}


@dataclass(frozen=True)
class DecodeResult:
    path: Path
    stored_weight: float
    decoded_weight: float
    stored_matching: list[tuple[int, int]]
    decoded_matching: list[tuple[int, int]]
    weight_matches: bool
    exact_matches: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify graph JSON decoding by reconstructing PyMatching graphs in scripts/."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["graph/graph_data_d5_r5_case_*.json"],
        help="Graph JSON files, directories, or glob patterns. Defaults to the d=5/r=5 set.",
    )
    parser.add_argument(
        "--graph-kind",
        choices=("short", "long", "full"),
        default="full",
        help="Which stored graph variant to verify.",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path("scripts/results/pymatching_d5_fullgraph_verification.json"),
        help="Path to write the verification summary JSON.",
    )
    parser.add_argument(
        "--max-mismatch-details",
        type=int,
        default=20,
        help="Maximum number of mismatch details to keep in the JSON summary.",
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


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(
            f"Missing dependency `{name}` in the current environment. "
            f"Run this script via `uv run` after installing `{name}`."
        )


def load_graph_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def normalize_pair(src: int, dst: int, boundary_nodes: set[int]) -> tuple[int, int]:
    src_boundary = src in boundary_nodes
    dst_boundary = dst in boundary_nodes
    if src_boundary and dst_boundary:
        raise ValueError(f"Boundary-boundary edge is not supported: {(src, dst)}")
    if src_boundary:
        return (dst, src)
    if dst_boundary:
        return (src, dst)
    return tuple(sorted((src, dst)))


def build_matching_graph(graph_data: dict, graph_kind: str):
    require_module("pymatching")
    import pymatching

    graph_key = GRAPH_KEY_MAP[graph_kind]
    node_key, boundary_key = NODE_KEY_MAP[graph_kind]
    regular_nodes = sorted({int(node_id) for node_id in graph_data.get(node_key, [])})
    boundary_nodes = {int(node_id) for node_id in graph_data.get(boundary_key, [])}

    if not regular_nodes and not graph_data.get(graph_key, []):
        raise ValueError("graph contains neither nodes nor edges")

    index_of = {node_id: index for index, node_id in enumerate(regular_nodes)}
    matching = pymatching.Matching()
    edge_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    boundary_edge_by_regular: dict[int, tuple[int, int]] = {}

    for src_raw, dst_raw, weight_raw in graph_data.get(graph_key, []):
        src = int(src_raw)
        dst = int(dst_raw)
        weight = float(weight_raw)
        normalized = normalize_pair(src, dst, boundary_nodes)
        if normalized in seen_pairs:
            raise ValueError(f"duplicate edge is not supported: {normalized}")
        seen_pairs.add(normalized)
        edge_id = len(edge_pairs)
        edge_pairs.append(normalized)

        src_boundary = src in boundary_nodes
        dst_boundary = dst in boundary_nodes
        if src_boundary is dst_boundary:
            if src not in index_of or dst not in index_of:
                raise ValueError(f"edge references unknown regular nodes: {(src, dst)}")
            matching.add_edge(index_of[src], index_of[dst], fault_ids=edge_id, weight=weight)
            continue

        regular = dst if src_boundary else src
        boundary = src if src_boundary else dst
        if regular not in index_of:
            raise ValueError(f"boundary edge references unknown regular node: {(src, dst)}")
        if regular in boundary_edge_by_regular:
            raise ValueError(
                f"regular node {regular} has multiple boundary edges: "
                f"{boundary_edge_by_regular[regular]} and {(regular, boundary)}"
            )
        boundary_edge_by_regular[regular] = (regular, boundary)
        matching.add_boundary_edge(index_of[regular], fault_ids=edge_id, weight=weight)

    syndrome = np.ones(len(regular_nodes), dtype=np.uint8)
    return matching, syndrome, edge_pairs, boundary_nodes


def decode_sample(path: Path, graph_kind: str) -> DecodeResult:
    graph_data = load_graph_json(path)
    matching, syndrome, edge_pairs, boundary_nodes = build_matching_graph(graph_data, graph_kind)
    correction, decoded_weight = matching.decode(syndrome, return_weight=True)

    decoded_pairs = [
        edge_pairs[index]
        for index, value in enumerate(np.asarray(correction, dtype=np.uint8).tolist())
        if value
    ]
    decoded_pairs = sorted(decoded_pairs)

    stored_matching = [
        normalize_pair(int(src), int(dst), boundary_nodes)
        for src, dst in graph_data[f"{GRAPH_KEY_MAP[graph_kind]}_MWPM_matching"]
    ]
    stored_matching = sorted(stored_matching)
    stored_weight = float(graph_data[f"{GRAPH_KEY_MAP[graph_kind]}_MWPM_weight"])

    return DecodeResult(
        path=path,
        stored_weight=stored_weight,
        decoded_weight=float(decoded_weight),
        stored_matching=stored_matching,
        decoded_matching=decoded_pairs,
        weight_matches=math.isclose(float(decoded_weight), stored_weight, rel_tol=0.0, abs_tol=1e-9),
        exact_matches=decoded_pairs == stored_matching,
    )


def summarize_results(
    results: list[DecodeResult],
    skipped: list[dict],
    errors: list[dict],
    graph_kind: str,
    inputs: list[str],
    max_mismatch_details: int,
) -> dict:
    weight_match_count = sum(result.weight_matches for result in results)
    exact_match_count = sum(result.exact_matches for result in results)
    mismatches = [
        {
            "path": str(result.path),
            "stored_weight": result.stored_weight,
            "decoded_weight": result.decoded_weight,
            "stored_matching": result.stored_matching,
            "decoded_matching": result.decoded_matching,
            "weight_matches": result.weight_matches,
            "exact_matches": result.exact_matches,
        }
        for result in results
        if (not result.weight_matches) or (not result.exact_matches)
    ]

    return {
        "config": {
            "inputs": inputs,
            "graph_kind": graph_kind,
        },
        "counts": {
            "decoded": len(results),
            "skipped": len(skipped),
            "errors": len(errors),
            "weight_match_count": weight_match_count,
            "exact_match_count": exact_match_count,
        },
        "skipped": skipped,
        "errors": errors,
        "mismatches": mismatches[:max_mismatch_details],
    }


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit(f"No graph JSON files matched: {args.inputs!r}")

    results: list[DecodeResult] = []
    skipped: list[dict] = []
    errors: list[dict] = []

    for path in paths:
        try:
            result = decode_sample(path, args.graph_kind)
        except ValueError as exc:
            if "contains neither nodes nor edges" in str(exc):
                skipped.append({"path": str(path), "reason": str(exc)})
                continue
            errors.append({"path": str(path), "reason": str(exc)})
            continue
        results.append(result)

    summary = summarize_results(
        results=results,
        skipped=skipped,
        errors=errors,
        graph_kind=args.graph_kind,
        inputs=args.inputs,
        max_mismatch_details=args.max_mismatch_details,
    )

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(summary, indent=2))

    counts = summary["counts"]
    print(f"decoded={counts['decoded']} skipped={counts['skipped']} errors={counts['errors']}")
    print(
        f"weight_matches={counts['weight_match_count']}/{counts['decoded']} "
        f"exact_matches={counts['exact_match_count']}/{counts['decoded']}"
    )
    print(f"saved summary to {args.results_out}")

    if errors:
        print("encountered unsupported graph samples")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
