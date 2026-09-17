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
