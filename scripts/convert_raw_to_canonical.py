import argparse
import json
import math
import sys
from pathlib import Path


REQUIRED_KEYS = (
    "code_distance",
    "measurement_rounds",
    "long_subgraph",
    "fullgraph",
    "long_subgraph_node_ids",
    "fullgraph_node_ids",
    "long_subgraph_boundary_node_ids",
    "fullgraph_boundary_node_ids",
    "long_subgraph_MWPM_weight",
    "long_subgraph_MWPM_is_valid",
    "long_subgraph_MWPM_matching",
    "fullgraph_MWPM_weight",
    "fullgraph_MWPM_is_valid",
    "fullgraph_MWPM_matching",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_keys(raw, source_file):
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"[{source_file}] missing required keys: {', '.join(sorted(missing))}"
        )


def _require_object(value, context, source_file):
    if not isinstance(value, dict):
        raise ValueError(
            f"[{source_file}] {context} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _require_array(value, name, source_file):
    if not isinstance(value, list):
        raise ValueError(
            f"[{source_file}] '{name}' must be a JSON array, got {type(value).__name__}"
        )
    return value


def _require_int(value, context, source_file):
    if type(value) is not int:
        raise ValueError(
            f"[{source_file}] {context} must be an integer, got {value!r}"
        )
    return value


def _require_number(value, context, source_file):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"[{source_file}] {context} must be a number, got {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"[{source_file}] {context} must be finite, got {value!r}"
        )
    return value


def _require_bool(value, context, source_file):
    if type(value) is not bool:
        raise ValueError(
            f"[{source_file}] {context} must be a boolean, got {value!r}"
        )
    return value


def _require_unique_int_array(values, context, source_file):
    values = _require_array(values, context, source_file)
    seen = {}
    unique_values = set()
    for i, value in enumerate(values):
        value = _require_int(value, f"{context}[{i}]", source_file)
        if value in seen:
            raise ValueError(
                f"[{source_file}] duplicate value {value} in {context}[{i}] "
                f"(first seen at index {seen[value]})"
            )
        seen[value] = i
        unique_values.add(value)
    return unique_values


def _normalize_node_sets(node_ids, boundary_node_ids, graph_name, source_file):
    regular_nodes = _require_unique_int_array(
        node_ids, f"{graph_name}.node_ids", source_file
    )
    boundary_nodes = _require_unique_int_array(
        boundary_node_ids, f"{graph_name}.boundary_node_ids", source_file
    )

    overlap = regular_nodes & boundary_nodes
    if overlap:
        raise ValueError(
            f"[{source_file}] {graph_name} node_ids and boundary_node_ids must be "
            f"disjoint, got overlap: {sorted(overlap)}"
        )

    all_declared_nodes = regular_nodes | boundary_nodes
    return regular_nodes, boundary_nodes, all_declared_nodes


def kind_of_node(node_id, regular_node_ids, boundary_node_ids, graph_name, source_file):
    node_id = _require_int(node_id, f"{graph_name}.node_id", source_file)
    if node_id in boundary_node_ids:
        return "boundary"
    if node_id in regular_node_ids:
        return "regular"
    raise ValueError(
        f"[{source_file}] {graph_name} references undeclared node_id={node_id}"
    )


def convert_graph(edge_list, node_ids, boundary_node_ids, graph_name, source_file):
    regular_nodes, boundary_nodes, _ = _normalize_node_sets(
        node_ids, boundary_node_ids, graph_name, source_file
    )
    edge_list = _require_array(edge_list, f"{graph_name}.edges", source_file)
    edges_out = []

    for i, edge in enumerate(edge_list):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(
                f"[{source_file}] invalid edge at index {i} in {graph_name}: "
                f"expected [u, v, w], got {edge!r}"
            )
        u, v, w = edge
        w = _require_number(w, f"{graph_name}.edges[{i}].weight", source_file)
        u_kind = kind_of_node(
            u, regular_nodes, boundary_nodes, graph_name, source_file
        )
        v_kind = kind_of_node(
            v, regular_nodes, boundary_nodes, graph_name, source_file
        )

        edges_out.append({
            "u": u,
            "v": v,
            "weight": w,
            "u_kind": u_kind,
            "v_kind": v_kind
        })

    return {
        "name": graph_name,
        "regular_nodes": sorted(regular_nodes),
        "boundary_nodes": sorted(boundary_nodes),
        "edges": edges_out
    }


def convert_matching(
    matching_pairs,
    regular_node_ids,
    boundary_node_ids,
    graph_name,
    source_file,
):
    matching_pairs = _require_array(
        matching_pairs, f"{graph_name}.matching", source_file
    )
    out = []
    used_endpoints = {}
    for i, pair in enumerate(matching_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"[{source_file}] invalid matching pair at index {i} in {graph_name}: "
                f"expected [u, v], got {pair!r}"
            )
        u, v = pair
        if u == v:
            raise ValueError(
                f"[{source_file}] invalid matching pair at index {i} in {graph_name}: "
                f"self-loop {(u, v)} is not allowed"
            )
        u_kind = kind_of_node(
            u, regular_node_ids, boundary_node_ids, graph_name, source_file
        )
        v_kind = kind_of_node(
            v, regular_node_ids, boundary_node_ids, graph_name, source_file
        )

        if u_kind == "boundary" and v_kind == "regular":
            u, v = v, u
            u_kind, v_kind = v_kind, u_kind

        if u_kind == "regular" and v_kind == "regular":
            kind = "regular-regular"
        elif u_kind == "regular" and v_kind == "boundary":
            kind = "regular-boundary"
        else:
            raise ValueError(
                f"[{source_file}] invalid matching pair at index {i} in {graph_name}: "
                f"{(u, v)} classified as ({u_kind}, {v_kind}); allowed kinds are "
                "regular-regular and regular-boundary only"
            )

        for node_id in (u, v):
            if node_id in used_endpoints:
                raise ValueError(
                    f"[{source_file}] invalid matching pair at index {i} in {graph_name}: "
                    f"node_id={node_id} is reused (first used at index "
                    f"{used_endpoints[node_id]})"
                )
            used_endpoints[node_id] = i

        out.append({
            "u": u,
            "v": v,
            "kind": kind
        })
    return out


def _build_undirected_edge_lookup(graph, source_file):
    lookup = {}
    first_seen = {}
    for i, edge in enumerate(graph["edges"]):
        key = tuple(sorted((edge["u"], edge["v"])))
        if key in lookup:
            raise ValueError(
                f"[{source_file}] duplicate undirected edge {key} in {graph['name']} "
                f"(indices {first_seen[key]} and {i})"
            )
        lookup[key] = edge["weight"]
        first_seen[key] = i
    return lookup


def _read_mwpm_fields(raw, prefix, label, source_file):
    weight_key = f"{prefix}_MWPM_weight"
    valid_key = f"{prefix}_MWPM_is_valid"
    matching_key = f"{prefix}_MWPM_matching"
    weight = _require_number(raw[weight_key], weight_key, source_file)
    is_valid = _require_bool(raw[valid_key], valid_key, source_file)
    if not is_valid:
        raise ValueError(
            f"[{source_file}] {valid_key} is false; invalid {label} is not allowed"
        )
    return weight, is_valid, raw[matching_key]


def _validate_teacher_matching(
    matching,
    graph,
    regular_node_ids,
    boundary_node_ids,
    declared_weight,
    source_file,
    label,
):
    edge_lookup = _build_undirected_edge_lookup(graph, source_file)
    computed_weight = 0.0
    regular_counts = {node_id: 0 for node_id in regular_node_ids}
    boundary_counts = {node_id: 0 for node_id in boundary_node_ids}

    for i, pair in enumerate(matching):
        u = pair["u"]
        v = pair["v"]
        edge_key = tuple(sorted((u, v)))
        if edge_key not in edge_lookup:
            raise ValueError(
                f"[{source_file}] {label} matching pair at index {i} does not exist "
                f"in {graph['name']}: {(u, v)}"
            )
        computed_weight += edge_lookup[edge_key]

        for node_id in (u, v):
            if node_id in regular_counts:
                regular_counts[node_id] += 1
            elif node_id in boundary_counts:
                boundary_counts[node_id] += 1

    missing_regular = sorted(
        node_id for node_id, count in regular_counts.items() if count == 0
    )
    reused_regular = sorted(
        node_id for node_id, count in regular_counts.items() if count > 1
    )
    reused_boundary = sorted(
        node_id for node_id, count in boundary_counts.items() if count > 1
    )

    if missing_regular:
        raise ValueError(
            f"[{source_file}] {label} matching does not cover all regular nodes; "
            f"missing: {missing_regular}"
        )
    if reused_regular:
        raise ValueError(
            f"[{source_file}] {label} matching reuses regular nodes: {reused_regular}"
        )
    if reused_boundary:
        raise ValueError(
            f"[{source_file}] {label} matching reuses boundary nodes: {reused_boundary}"
        )
    if not math.isclose(computed_weight, declared_weight, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"[{source_file}] {label} matching weight mismatch: declared "
            f"{declared_weight}, computed {computed_weight}"
        )

def convert_raw_to_canonical(raw, source_file):
    raw = _require_object(raw, "root JSON", source_file)
    _require_keys(raw, source_file)
    code_distance = _require_int(raw["code_distance"], "code_distance", source_file)
    measurement_rounds = _require_int(
        raw["measurement_rounds"], "measurement_rounds", source_file
    )
    long_mwpm_weight, long_mwpm_is_valid, long_mwpm_matching = _read_mwpm_fields(
        raw, "long_subgraph", "teacher", source_file
    )
    fullgraph_mwpm_weight, fullgraph_mwpm_is_valid, fullgraph_mwpm_matching = (
        _read_mwpm_fields(raw, "fullgraph", "reference", source_file)
    )

    long_regular_ids, long_boundary_ids, _ = _normalize_node_sets(
        raw["long_subgraph_node_ids"],
        raw["long_subgraph_boundary_node_ids"],
        "long_subgraph",
        source_file,
    )
    full_regular_ids, full_boundary_ids, _ = _normalize_node_sets(
        raw["fullgraph_node_ids"],
        raw["fullgraph_boundary_node_ids"],
        "fullgraph",
        source_file,
    )

    input_graph = convert_graph(
        raw["long_subgraph"],
        raw["long_subgraph_node_ids"],
        raw["long_subgraph_boundary_node_ids"],
        "long_subgraph",
        source_file,
    )
    reference_graph = convert_graph(
        raw["fullgraph"],
        raw["fullgraph_node_ids"],
        raw["fullgraph_boundary_node_ids"],
        "fullgraph",
        source_file,
    )
    teacher_matching = convert_matching(
        long_mwpm_matching,
        long_regular_ids,
        long_boundary_ids,
        "long_subgraph_MWPM_matching",
        source_file,
    )
    reference_matching = convert_matching(
        fullgraph_mwpm_matching,
        full_regular_ids,
        full_boundary_ids,
        "fullgraph_MWPM_matching",
        source_file,
    )
    _validate_teacher_matching(
        teacher_matching,
        input_graph,
        long_regular_ids,
        long_boundary_ids,
        long_mwpm_weight,
        source_file,
        "teacher",
    )
    _validate_teacher_matching(
        reference_matching,
        reference_graph,
        full_regular_ids,
        full_boundary_ids,
        fullgraph_mwpm_weight,
        source_file,
        "reference",
    )

    canonical = {
        "meta": {
            "schema_version": "phase1-v0.2",
            "source_file": source_file,
            "code_distance": code_distance,
            "measurement_rounds": measurement_rounds,
            "input_graph_name": "long_subgraph",
            "teacher_graph_name": "long_subgraph",
            "reference_graph_name": "fullgraph",
            "notes": []
        },
        "graphs": {
            "input_graph": input_graph,
            "reference_graph": reference_graph
        },
        "teacher": {
            "matching_weight": long_mwpm_weight,
            "is_valid": long_mwpm_is_valid,
            "matching": teacher_matching
        },
        "reference": {
            "matching_weight": fullgraph_mwpm_weight,
            "is_valid": fullgraph_mwpm_is_valid,
            "matching": reference_matching
        },
        "ignored": {
            "imperfect_fmu_matching": None
        }
    }

    if "boundary_node_id" in raw:
        canonical["meta"]["raw_boundary_node_id"] = _require_int(
            raw["boundary_node_id"], "boundary_node_id", source_file
        )

    return canonical

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw graph JSON into canonical JSON format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input raw JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output canonical JSON file.",
    )
    args = parser.parse_args()

    raw_path = Path(args.input)
    out_path = Path(args.output)

    raw = load_json(raw_path)
    canonical = convert_raw_to_canonical(raw, raw_path.name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: file not found: {exc.filename}", file=sys.stderr)
        raise SystemExit(2) from exc
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except OSError as exc:
        print(f"ERROR: OS error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
