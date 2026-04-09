#!/usr/bin/env python3
"""
Run a substantial d=5/r=5 training check for the fullgraph -> QEC_GNN-RNN path.

This script stays entirely under scripts/ and reuses the helper functions in
`train_qec_gnn_rnn_fullgraph.py`. Because the graph JSON files do not contain
logical-observable labels, it derives a balanced-enough binary target from the
stored fullgraph MWPM weight:

    label = 1 if fullgraph_MWPM_weight >= threshold else 0

The default threshold is 3, which yields a 64/36 class split on the current
`graph_data_d5_r5_case_*.json` set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that fullgraph-based QEC_GNN-RNN training on d=5 reduces loss."
    )
    parser.add_argument(
        "--pattern",
        default="graph/graph_data_d5_r5_case_*.json",
        help="Glob pattern for the training set.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Binary target is 1 when fullgraph_MWPM_weight >= threshold.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size.",
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
        help="Shuffle seed.",
    )
    parser.add_argument(
        "--graph-kind",
        choices=("short", "long", "full"),
        default="full",
        help="Stored graph variant to train on.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("raw", "inverse", "inverse_square", "unit"),
        default="inverse",
        help="How to map stored graph weights into edge_attr.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("scripts/results/qec_gnn_rnn_d5_training_metrics.json"),
        help="Path to save the metrics JSON.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("scripts/results/qec_gnn_rnn_d5_fullgraph_model.pt"),
        help="Path to save the best model state dict.",
    )
    return parser.parse_args()


def load_base_module():
    script_path = Path(__file__).resolve().parent / "train_qec_gnn_rnn_fullgraph.py"
    spec = importlib.util.spec_from_file_location("fullgraph_train_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_manifest(paths: list[Path], threshold: int) -> tuple[dict[str, float], dict[str, int]]:
    manifest: dict[str, float] = {}
    class_counts = {"negative": 0, "positive": 0}
    for path in paths:
        with path.open() as fh:
            graph_data = json.load(fh)
        label = float(graph_data["fullgraph_MWPM_weight"] >= threshold)
        manifest[str(path.resolve())] = label
        class_counts["positive" if label == 1.0 else "negative"] += 1
    return manifest, class_counts


def iterate_minibatches(samples, batch_size: int, rng: random.Random):
    order = list(range(len(samples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [samples[index] for index in order[start : start + batch_size]]


def evaluate(base, model, samples, batch_size, device):
    import torch
    import torch.nn as nn

    loss_fn = nn.BCELoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = base.collate_graph_samples(samples[start : start + batch_size])
            x, edge_index, batch_labels, label_map, edge_attr, flips = base.move_batch_to_device(batch, device)
            out = model(x, edge_index, edge_attr, batch_labels, label_map)
            loss = loss_fn(out, flips)
            total_loss += loss.item() * flips.shape[0]
            total_correct += int((torch.round(out) == flips).sum().item())
            total_count += int(flips.numel())
    return total_loss / total_count, total_correct / total_count


def run_training(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn

    base = load_base_module()
    paths = base.expand_inputs([args.pattern])
    if not paths:
        raise SystemExit(f"No files matched {args.pattern!r}.")

    manifest, class_counts = build_manifest(paths, args.threshold)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    samples = []
    skipped = []
    for path in paths:
        try:
            sample = base.build_sample(
                path=path,
                graph_kind=args.graph_kind,
                code_task="surface_code:rotated_memory_z",
                error_rate=1e-3,
                weight_mode=args.weight_mode,
                manifest=manifest,
                target_field="imperfect_fmu_is_valid",
            )
        except ValueError as exc:
            if "contains neither nodes nor edges" in str(exc):
                skipped.append(path.name)
                continue
            raise
        samples.append(sample)
    if skipped:
        print(f"skipped {len(skipped)} empty samples: {', '.join(skipped)}")
    class_counts = {
        "positive": int(sum(sample.label == 1.0 for sample in samples)),
        "negative": int(sum(sample.label == 0.0 for sample in samples)),
    }
    base.summarize_samples(samples)
    train_samples, val_samples = base.split_samples(samples, args.train_fraction)
    _, model = base.build_model(batch_size=args.batch_size, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    batch_rng = random.Random(args.seed)

    history = []
    best_val_loss = float("inf")
    best_state = None

    print(
        "derived target: "
        f"fullgraph_MWPM_weight >= {args.threshold} "
        f"(positive={class_counts['positive']}, negative={class_counts['negative']})"
    )
    print(f"training samples={len(train_samples)}, validation samples={len(val_samples)}, device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        for minibatch in iterate_minibatches(train_samples, args.batch_size, batch_rng):
            batch = base.collate_graph_samples(minibatch)
            x, edge_index, batch_labels, label_map, edge_attr, flips = base.move_batch_to_device(batch, device)

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
        val_loss, val_acc = evaluate(base, model, val_samples, args.batch_size, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.save_model)

    first_train_loss = history[0]["train_loss"]
    last_train_loss = history[-1]["train_loss"]
    best_train_loss = min(item["train_loss"] for item in history)
    first_val_loss = history[0]["val_loss"]
    best_val_loss = min(item["val_loss"] for item in history)

    loss_decreased = last_train_loss < first_train_loss
    best_val_improved = best_val_loss < first_val_loss

    result = {
        "config": {
            "pattern": args.pattern,
            "threshold": args.threshold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
            "graph_kind": args.graph_kind,
            "weight_mode": args.weight_mode,
            "lr": args.lr,
        },
        "class_counts": class_counts,
        "history": history,
        "verification": {
            "loss_decreased": loss_decreased,
            "best_val_improved": best_val_improved,
            "first_train_loss": first_train_loss,
            "last_train_loss": last_train_loss,
            "best_train_loss": best_train_loss,
            "first_val_loss": first_val_loss,
            "best_val_loss": best_val_loss,
        },
        "model_path": str(args.save_model),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(result, indent=2))
    print(f"saved metrics to {args.metrics_out}")
    print(
        "verification: "
        f"loss_decreased={loss_decreased}, "
        f"best_val_improved={best_val_improved}"
    )
    return result


def main() -> None:
    args = parse_args()
    result = run_training(args)
    if not result["verification"]["loss_decreased"]:
        raise SystemExit("Training loss did not decrease over the run.")


if __name__ == "__main__":
    main()
