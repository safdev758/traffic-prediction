import argparse
import json
from pathlib import Path

import mlflow


def numeric_leaves(obj, prefix=""):
    """Yield (dotted.path, float) for every int/float leaf; strings/bools become params elsewhere."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            yield from numeric_leaves(x, f"{prefix}{i}.")
    else:
        leaf = (prefix.rstrip('.') if prefix else "value"), obj
        try:
            yield leaf[0], float(leaf[1])
        except (TypeError, ValueError):
            pass


def all_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from all_leaves(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            yield from all_leaves(x, f"{prefix}{i}.")
    else:
        yield prefix.rstrip(".") or "value", obj


def main():
    p = argparse.ArgumentParser(description="Log a training run's metrics.json into MLflow")
    p.add_argument("--metrics", default="output/train/output/metrics.json")
    p.add_argument("--project", required=True)
    p.add_argument("--kernel", required=True, help="Kaggle kernel slug this run came from")
    p.add_argument("--experiment", default=None)
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"metrics file not found: {metrics_path}")

    data = json.loads(metrics_path.read_text())
    experiment = args.experiment or f"{args.project}_training"
