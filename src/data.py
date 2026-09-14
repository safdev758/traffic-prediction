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
