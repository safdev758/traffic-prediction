import torch
import torch.nn as nn


class RNNWrapper(nn.Module):
    """Per-sensor RNN/LSTM/GRU on the sensor-axis (S) — single head per future horizon."""

    def __init__(self, kind, n_sensors, hidden=64, n_horizons=3):
        super().__init__()
