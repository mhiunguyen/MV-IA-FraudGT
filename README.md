# TH-FraudGT

**History-Augmented and Temporal Fraud Graph Transformer** is a research
extension of [FraudGT](https://github.com/junhongmit/FraudGT) for financial
fraud detection on directed transaction multigraphs.

## Project status

The project is evaluated as a controlled ablation study.  The original
FraudGT encoder and edge-classification head are retained.

| Variant | Historical edge features (H) | Temporal neighbor sampling (T) | Status |
|---|---:|---:|---|
| **A — FraudGT baseline** | No | No | Implemented and evaluated |
| **H — H-FraudGT** | Yes | No | Implemented; seed 42 completed |
| **T — T-FraudGT** | No | Yes | Planned; not implemented yet |
| **TH — TH-FraudGT** | Yes | Yes | Planned after the T ablation |

The repository name describes the complete research direction.  Results must
not label T or TH as completed until their code, leakage audit, and experiments
are available.

## Implemented contribution: H-FraudGT

H-FraudGT augments each transaction edge with eight behavioral features built
only from strictly earlier transactions:

1. time since the source account last sent money;
2. time since the destination account last received money;
3. whether the source has sent before;
4. whether the destination has received before;
5. prior outgoing transaction count;
6. prior incoming transaction count;
7. prior transaction count for the ordered account pair; and
8. current amount relative to the source's historical outgoing mean.

For an edge observed at time `t`, its history is restricted to:

```text
H(t) = { e_k | timestamp(e_k) < t }
E_history = concat(E_original, eight past-only history features)
```

Edges sharing the same timestamp cannot observe one another, and continuous
history features are normalized with training-split statistics only.

Main implementation files:

- `fraudGT/datasets/history_features.py` — past-only feature construction;
- `fraudGT/datasets/aml_dataset.py` — optional history integration and cache;
- `fraudGT/config/dataset_config.py` — `dataset.add_history` switch;
- `configs/AML-Small-HI/AML-Small-HI-History-T4.yaml` — H experiment;
- `notebooks/kaggle/06_H_FraudGT_History_T4.ipynb` — seed 42;
- `notebooks/kaggle/07_AH_Seeds43_44_T4x2.ipynb` — A/H seeds 43–44.

## Planned contribution: T-FraudGT

T will replace the ordinary link-neighbor sampler with a past-only temporal
sampler.  For a target transaction at time `t`, only neighboring edges with
timestamps earlier than `t` will be eligible.  A recency policy may then rank
or sample those eligible edges.  The exact policy must be implemented and
audited before reporting T or TH results.

TH combines the two independent changes:

```text
TH-FraudGT = H historical edge features + T past-only temporal sampling
```

The intended final ablation is `A / H / T / TH`, which separates the effect of
each component from their interaction.

## Seed-42 result on AML-Small-HI

At the fixed threshold 0.50, the current controlled one-seed experiment is:

| Model | Validation F1 | Test F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.61983 | 0.67442 | 0.66514 | 0.68396 | 0.98688 |
| H | **0.66667** | **0.71642** | **0.69421** | **0.74009** | **0.99023** |

This is preliminary evidence from one seed, not a final mean ± standard
deviation result.  Seeds 43 and 44 must be completed for both A and H.

## Reproducibility protocol

1. Use the same AML-Small-HI chronological train/validation/test split.
2. Match batch size, fanout, iterations, optimizer, encoder, head, and budget.
3. Train independently with seeds 42, 43, and 44.
4. Select the epoch with validation F1 only.
5. Report fixed-threshold 0.50 as the primary protocol.
6. Report a validation-selected threshold only as a secondary analysis.
7. Apply the selected epoch and threshold to test once.
8. Report mean ± standard deviation across seeds.

After a multi-seed run:

```bash
python scripts/summarize_thresholds.py results/<experiment-directory> \
  --fixed-threshold 0.50
```

## Legacy exploratory experiments

The earlier Multi-view and imbalance-aware heads remain in the repository as
exploratory ablations.  They are not the primary proposed method after the
project moved to the H/T direction.  Keeping them preserves experiment history
and does not mean they are required for the final A/H/T/TH comparison.

## Upstream attribution

This repository retains the official FraudGT framework, dataset loaders,
baseline models, and configurations for reproducibility.  The repository is
an extension, not a claim of authorship over the complete upstream codebase.
See [NOTICE.md](NOTICE.md) for attribution and redistribution notes.
