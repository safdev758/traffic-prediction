import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (5, 10, 20)  # 15/30/60 min at 3-min PEMS-BAY cadence
HISTORY = 24


