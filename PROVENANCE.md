# Provenance — Traffic Prediction (PEMS-BAY)

| Artifact | Source | Proof path |
|----------|--------|------------|
| Kernel v5 (final) | Kaggle `traffic-prediction-v5`, GPU, dataset `scchuy/pemsbay` | `output/run5/output/metrics.json`, kernel log |
| GraphGRU best MAE 37.60 @15min | same kernel | metrics.json → `graph_gru` |
| Loader fix | v2 read `pems-bay-meta.h5`; v3+ targets `speed/block0_values`, skips meta | diff in kernel logs run2 vs run3+ |
| Git repo | local real-timestamp commits | `git log` |
| MLflow | `traffic-prediction_training` | local mlruns/ |

## CAUTIONS
- The h5 has two top-level datasets; meta file is a decoy that also parses as h5 — brittle `rglob('*.h5')` first-hit patterns fail silently.
