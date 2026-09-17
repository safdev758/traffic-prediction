import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="PEMS-BAY multi-horizon forecasting")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=6000)
    return p.parse_args()


def main():
    from src.data import load_pemsh5, MultiHorizonDataset, find_data_file, HORIZONS
    from src.models import RNNWrapper, GraphGRU

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_root) if args.data_root else Path("/kaggle/input")
    data_path = find_data_file(root)
    print(f"[ENV] device={device} file={data_path}")
    data = load_pemsh5(str(data_path))
    print(f"[DATA] shape={data.shape} mean={data.mean():.2f}")
    ds = MultiHorizonDataset(data)
    n = len(ds)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_ds = torch.utils.data.Subset(ds, range(n_train))
