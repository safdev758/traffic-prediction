# Traffic Prediction (PEMS-BAY)

Sensor-network traffic speed forecasting: classic GateRecurrent baselines vs. a graph-conditioned GRU.

## What it does
- **Dataset**: PEMS-BAY speed matrix (325 sensors, 5-min sampling, 52k timesteps).
- **Task**: 3 horizons — 5/10/20 steps ahead (15/30/60 minutes) — from the past 12 steps (1 h).
- **Models**: RNN, LSTM, GRU + GraphGRU (spatial-attention weighting over neighbor sensors).

## Result (kernel `traffic-prediction-v5`, 52,072 samples, horizons 15/30/60 min)
| model | best RMSE | best MAE |
|-------|-----------|----------|
| RNN   | 86.29 | 39.58 |
| GRU   | **83.21** | 42.74 |
