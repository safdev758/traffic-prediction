import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (5, 10, 20)  # 15/30/60 min at 3-min PEMS-BAY cadence
HISTORY = 24


def find_data_file(root):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in (".h5", ".hdf5") and "meta" not in p.name.lower():
            return p
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in (".h5", ".hdf5"):
            return p
    raise SystemExit(f"no h5 in {root}")


def load_pemsh5(path: str):
    import h5py
    with h5py.File(path, "r") as f:
        if "speed/block0_values" in f:
            data = np.asarray(f["speed/block0_values"]).astype(np.float32)
        else:
            group = "block0_values" if "block0_values" in f else list(f.keys())[0]
            data = np.asarray(f[group]).astype(np.float32)
    return data


def build_windows(data, history=HISTORY, horizons=HORIZONS, sensor_idx=0):
    X, Y = [], []
    for i in range(len(data) - history - max(horizons)):
        X.append(data[i:i + history])
        Y.append([data[i + history - 1 + h, sensor_idx] for h in horizons])
    return torch.from_numpy(np.stack(X).astype(np.float32)), torch.from_numpy(np.asarray(Y, dtype=np.float32))


class MultiHorizonDataset:
    def __init__(self, data, history=HISTORY, horizons=HORIZONS, sensor_idx=0):
        self.X, self.y = build_windows(data, history, horizons, sensor_idx)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
