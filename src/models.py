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


class GraphGRU(nn.Module):
    """Multi-station GRU over the TIME axis with sensor-spatial attention over states
    (a light stand-in for STGCN spatial aggregation — attention over ALL sensor hidden
    states at the last time step, then per-horizon head)."""

    def __init__(self, hidden=64, n_horizons=3):
        super().__init__()
        self.emb = nn.Linear(1, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        # x: (B, T, S)
        B, T, S = x.shape
        h = self.emb(x.transpose(1, 2).unsqueeze(-1))  # (B, T, S, hidden)
        states, _ = self.gru(h.reshape(B * S, T, -1))  # (B*S, T, hidden)
        last = states.reshape(B, S, T, -1)[:, :, -1]
        attn_out, _ = self.attn(last, last, last)
        pooled = attn_out.mean(dim=1)
        return self.head(pooled)
