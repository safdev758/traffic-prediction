import torch
import torch.nn as nn


class RNNWrapper(nn.Module):
    """Per-sensor RNN/LSTM/GRU on the sensor-axis (S) — single head per future horizon."""

    def __init__(self, kind, n_sensors, hidden=64, n_horizons=3):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.seq = rnn_cls(n_sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        out, _ = self.seq(x)
        return self.head(out[:, -1, :])


