# Spacing review (5-rule framework) — Traffic Prediction

## D+2
1. What do the horizons 5/10/20 steps mean in minutes for PEMS-BAY (5-min sampling)?
2. How does GraphGRU weight neighbor sensors (see src/models.py attention block)?
3. Why does GraphGRU improve MAE but not RMSE compared to plain GRU?

## D+7
1. Re-derive the mask logic for missing sensors; what happens with all-zero sensors?
2. Diagnose tail errors: which times of day drive RMSE 83–86 divergence?
3. Ablation locally: horizon 1 vs 5 vs 20 steps with GRU.

## D+21 (construct)
1. Implement a real DCRNN-style chebyshev graph conv and compare.
2. Deploy two-sentence resume blurb: travel-time API w/ uncertainty bands.
3. Extend to a second dataset (PEMSD4) with domain-adaptive normalization.

## Pointers
- loader src/data.py load_pemsh5 reads speed/block0_values (52116, 325).
