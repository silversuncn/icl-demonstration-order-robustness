# Demonstration-Order Robustness in Few-Shot In-Context Text Classification

> **Demonstration-Order Robustness in Few-Shot In-Context Text Classification**  
> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for a finite-grid
measurement study of demonstration-order robustness in few-shot in-context text
classification. The bundle focuses on aggregate result verification rather than
rerunning model inference.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── public_summary.json
│   ├── formal_aggregate.csv
│   ├── spread_by_dataset_shot.csv
│   ├── statistical_tests_holm.csv
│   ├── random_order_dispersion.csv
│   ├── parser_success_summary.csv
│   ├── validation_summary.csv
│   └── worst_order_regret_by_order.csv
├── figures/
│   ├── order_spread_by_dataset_shot.png
│   └── random_order_dispersion_by_dataset_shot.png
├── src/
│   └── verify_public_results.py
└── tests/
    └── test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Model | `Qwen/Qwen3-4B` | 1 |
| Datasets | `sst2`, `rte`, `mrpc` | 3 |
| Total shots | `2`, `4`, `8` | 3 |
| Seeds | `1`, `2`, `3`, `4`, `5` | 5 |
| Orders | `canonical`, `label_grouped`, `alternating`, `random_0`-`random_4` | 8 |
| Held-out examples per aggregate cell | `200` | 1 |

Row-count check:

```text
3 datasets x 3 shot budgets x 5 seeds x 8 orders = 360 aggregate rows
360 aggregate rows x 200 held-out examples = 72,000 raw evaluations
```

## Hardware & Environment

| Component | Value |
| --- | --- |
| Runtime target | cached local model inference |
| Device record | `cuda:0` |
| Model | `Qwen/Qwen3-4B` |
| Python | `3.11.15` |
| Torch | `2.11.0+cu128` |
| Transformers | `5.10.1` |
| Datasets | `5.0.0` |

## Key Results

- The matrix contains `360` aggregate cells and `72,000` raw evaluations.
- Parser success is `1.0` across all dataset/shot groups.
- The maximum observed accuracy spread is `0.0550`.
- The maximum observed macro-F1 spread is `0.0571`.
- Planned deterministic-order comparisons have Holm-corrected p-values of `1.0`, so deterministic orders are not supported as superior to the random-order mean in this finite grid.

## Requirements

The public verification script uses only the Python standard library.

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

## Citation

```bibtex
@article{sun2026demonstrationorderrobustness,
  title = {Demonstration-Order Robustness in Few-Shot In-Context Text Classification},
  author = {Sun, Yaowen},
  year = {2026}
}
```
