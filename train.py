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
    val_ds = torch.utils.data.Subset(ds, range(n_train, n_train + n_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    metrics = {"meta": vars(args), "horizons_min": [h * 3 for h in HORIZONS], "samples": n}
    y_scale = float(data.std())
    y_mu = float(data.mean())
    for kind in ("rnn", "lstm", "gru", "graph_gru"):
        if kind == "graph_gru":
            net = GraphGRU(args.hidden, len(HORIZONS)).to(device)
        else:
            net = RNNWrapper(kind, data.shape[1], args.hidden, len(HORIZONS)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()
        steps = 0
        for epoch in range(1, args.epochs + 1):
            net.train()
            total = 0.0
            seen = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * xb.size(0)
                seen += xb.size(0)
                steps += 1
                if steps > args.max_steps:
                    break
            print(f"[TRAFFIC] {kind} epoch {epoch}/{args.epochs} sMAE={total/seen:.4f}")
            if steps > args.max_steps:
                break
        net.eval()
        per_hor = {h: [] for h in HORIZONS}
        per_gt = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = net(xb.to(device))
                for j, h in enumerate(HORIZONS):
                    per_hor[h] += pred[:, j].cpu().tolist()
                    per_gt[h] += yb[:, j].tolist()
        per_rng = {}
        for h in HORIZONS:
            diff = (np.asarray(per_hor[h]) - np.asarray(per_gt[h])) * y_scale
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            per_rng[h] = {"rmse": round(rmse, 2), "mae": round(mae, 2)}
        per_rng["minutes"] = [int(h * 3) for h in HORIZONS]
        metrics[kind] = per_rng
        print(f"[TRAFFIC-EVAL] {kind}: {per_rng}")
    Path("output").mkdir(exist_ok=True)
    Path("output/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL:", json.dumps(metrics))


if __name__ == "__main__":
    main()
