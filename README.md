# Demonstration-Order Robustness in Few-Shot In-Context Text Classification

> Demonstration-Order Robustness in Few-Shot In-Context Text Classification<br>
> Yaowen Sun

## Overview

This repository contains a sanitized processed-results verification package for
a finite-grid study of demonstration-order robustness in few-shot in-context
text classification. It is intended to verify the reported aggregate tables,
figures, per-sample original-grid predictions, and checksum records. It is not
a full model-inference rerun package.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── checksums_sha256.txt
├── requirements.txt
├── data/
│   ├── formal_aggregate.csv
│   ├── formal_predictions_original_360.csv
│   ├── results_2shot_exhaustive.csv
│   ├── results_4shot_exhaustive.csv
│   ├── results_8shot_extended.csv
│   ├── permutation_mapping.json
│   ├── spread_by_dataset_shot.csv
│   ├── descriptive_table_iv_v2.csv
│   ├── descriptive_statistics.csv
│   ├── cluster_sign_flip_tests_v2.csv
│   ├── cluster_diagnostics_v2.csv
│   ├── cluster_diagnostics_v2.json
│   ├── cluster_per_shot_differences_v2.csv
│   ├── random_order_dispersion.csv
│   ├── random_permutation_diagnostics_v2.csv
│   ├── support_permutation_uniqueness.csv
│   ├── permutation_multiplicity_by_shot.csv
│   ├── two_shot_deterministic_permutation_diagnostics_v2.csv
│   ├── two_shot_label_grouped_alternating_equivalence_v2.csv
│   ├── worst_order_regret_by_order.csv
│   ├── model_environment_record_v2.json
│   ├── public_summary.json
│   ├── public_v2_manifest.json
│   └── validation_summary.csv
├── figures/
├── src/
│   ├── verify_public_results.py
│   └── plot_figures.py
└── tests/
    └── test_public_results.py
```

## Data Scope

| Component | Scope | Rows |
| --- | --- | ---: |
| Original named-order aggregate grid | 3 datasets x 3 shot budgets x 5 seeds x 8 orders | 360 |
| 2-shot exhaustive extension | 3 datasets x 5 seeds x 2 unique permutations | 30 |
| 4-shot exhaustive extension | 3 datasets x 5 seeds x 24 unique permutations | 360 |
| 8-shot evaluated extension | 3 datasets x 5 seeds x 30 additional unique permutations | 450 |
| Public aggregate evidence package | Original grid plus extensions | 1,200 |
| Per-sample prediction file | Original 360 aggregate cells only | 72,000 |

Each aggregate cell contains 200 held-out evaluation examples, so the public
aggregate evidence package accounts for 240,000 evaluated instances. The
per-sample CSV intentionally covers only the original 360-cell named-order grid;
the exhaustive and extended cells are released as aggregate rows.

## Experimental Setup

| Dimension | Values |
| --- | --- |
| Model | `Qwen/Qwen3-4B` |
| Datasets | `sst2`, `rte`, `mrpc` |
| Shot budgets | `2`, `4`, `8` |
| Seeds | `1`, `2`, `3`, `4`, `5` |
| Original named orders | `seeded_base`, `label_grouped`, `alternating`, `random_0`-`random_4` |
| Held-out examples per aggregate cell | `200` |

Candidate answer letters `A` and `B` are scored directly by mean token
log-probability; no free-form generation parser is used in the reported
measurements.

## Permutation Coverage

The deterministic and random named orders are evaluated on the original grid.
The extension adds exhaustive or expanded unique-permutation coverage:

| Shot budget | Evaluated unique permutations per dataset-seed stratum |
| --- | ---: |
| 2-shot | 2 of 2 |
| 4-shot | 24 of 24 |
| 8-shot | 38 of 40,320 |

The 8-shot count combines the 8 original named-order permutations with 30
additional evaluated unique permutations.

## Key Results

- The current aggregate verification package contains 1,200 aggregate cells and
  240,000 evaluated instances.
- The per-sample prediction CSV contains 72,000 sanitized rows for the original
  360-cell named-order grid.
- The maximum observed accuracy spread is `0.0700`.
- The maximum observed macro-F1 spread is `0.0790`.
- Table IV reports descriptive finite-grid differences against the evaluated
  unique-permutation mean, not against the random-order-name mean.
- The planned exact cluster sign-flip tests use 15 dataset-seed clusters and
  six planned tests; all Holm-adjusted p-values are `1.0`.
- Random-order dispersion diagnostics remain scoped to the original named-order
  grid and are provided separately from the exhaustive extension.

## Verification

The main verifier uses only the Python standard library:

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

The verifier checks row counts, schema, per-sample-to-aggregate closure for the
original grid, the 1,200-cell spread table, Table IV descriptive values,
cluster sign-flip p-values, manifest hashes, figure hashes, and package
checksums.

`src/plot_figures.py` is retained as an optional plotting helper and requires
Matplotlib. The committed figure files are the publication-facing artifacts
verified by hash.

## Citation

```bibtex
@article{sun2026demonstrationorderrobustness,
  title = {Demonstration-Order Robustness in Few-Shot In-Context Text Classification},
  author = {Sun, Yaowen},
  year = {2026}
}
```
