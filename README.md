# TH-FraudGT

Research prototype based on the official
[FraudGT](https://github.com/junhongmit/FraudGT) implementation.

The project investigates two extensions for financial fraud detection:

- **H** — leakage-safe historical transaction features;
- **T** — past-only temporal neighbor sampling.

Current status:

| Variant | Status |
|---|---|
| A — FraudGT baseline | Evaluated |
| H — History-Augmented FraudGT | Implemented; multi-seed experiment running |
| T — Temporal FraudGT | Planned |
| TH — H + T | Planned |

Current Kaggle notebooks:

- `notebooks/kaggle/06_H_FraudGT_History_T4.ipynb`
- `notebooks/kaggle/07_AH_Seeds43_44_T4x2.ipynb`

The upstream framework, loaders, baselines, and configurations are retained
for reproducibility. See [NOTICE.md](NOTICE.md) for attribution.
