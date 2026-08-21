# H-FraudGT

Research prototype based on the official
[FraudGT](https://github.com/junhongmit/FraudGT) implementation.

The project investigates leakage-safe historical features for financial fraud
detection:

- **H** - leakage-safe historical transaction features;
- **HG** - reliability-gated history features (implemented; pending final run).

Current status:

| Variant | Status |
|---|---|
| A - FraudGT baseline | Evaluated |
| H - History-Augmented FraudGT | Evaluated, 3 seeds |
| HG - Reliability-Gated H-FraudGT | Implemented; pending evaluation |
| T / TH temporal prototype | Known-invalid; do not report |

On AML Small-HI at a fixed threshold of 0.50, A obtains test F1
`0.6975 ± 0.0207`; H obtains `0.6876 ± 0.0836` and higher recall
(`0.7278 ± 0.0704` vs. `0.6858 ± 0.0125`). H therefore demonstrates a
precision-recall trade-off, not an overall improvement over A.

Current Kaggle notebooks:

- `notebooks/kaggle/06_H_FraudGT_History_T4.ipynb`
- `notebooks/kaggle/07_AH_Seeds43_44_T4x2.ipynb`
- `notebooks/kaggle/11_History_Final_Seed43_Checkpoints_T4x2.ipynb`
- `notebooks/kaggle/11A_History_Final_Part1_Seed43_T4x2.ipynb`
- `notebooks/kaggle/11B_History_Final_Part2_Seed43_T4x2.ipynb`

The final notebook runs the complete R/F/M factorial ablation, saves
validation-selected `best.ckpt` files, records the dataset/environment
manifest, and creates one reproducibility ZIP.

The temporal notebook is retained only as an invalid prototype. Its sampler
removes the target transaction while the original edge head expects that edge,
which produces an empty target mask and NaN loss.

See `HANDOFF_CODEX.md` for the experiment history and next steps.

The upstream framework, loaders, baselines, and configurations are retained
for reproducibility. See [NOTICE.md](NOTICE.md) for attribution.
