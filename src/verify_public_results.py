#!/usr/bin/env python3
"""Verify public aggregate results for the demonstration-order robustness study."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify() -> dict[str, object]:
    aggregate = read_csv("formal_aggregate.csv")
    spread = read_csv("spread_by_dataset_shot.csv")
    random_dispersion = read_csv("random_order_dispersion.csv")
    uniqueness = read_csv("support_permutation_uniqueness.csv")
    multiplicity = read_csv("permutation_multiplicity_by_shot.csv")
    tests = read_csv("statistical_tests_holm.csv")
    summary = json.loads((DATA / "public_summary.json").read_text(encoding="utf-8"))

    strata: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in aggregate:
        key = (row["dataset"], int(row["total_shots"]), int(row["seed"]))
        strata.setdefault(key, []).append(row)

    accuracy_spreads = []
    macro_f1_spreads = []
    for rows in strata.values():
        acc = [float(row["accuracy"]) for row in rows]
        f1 = [float(row["macro_f1"]) for row in rows]
        accuracy_spreads.append(max(acc) - min(acc))
        macro_f1_spreads.append(max(f1) - min(f1))

    result = {
        "aggregate_rows": len(aggregate),
        "raw_evaluations_from_aggregate": sum(int(float(row["n_eval"])) for row in aggregate),
        "dataset_count": len({row["dataset"] for row in aggregate}),
        "shot_values": sorted({int(row["total_shots"]) for row in aggregate}),
        "seed_values": sorted({int(row["seed"]) for row in aggregate}),
        "order_count": len({row["order_id"] for row in aggregate}),
        "all_n_eval_200": all(int(float(row["n_eval"])) == 200 for row in aggregate),
        "max_accuracy_spread": max(accuracy_spreads),
        "max_macro_f1_spread": max(macro_f1_spreads),
        "spread_rows": len(spread),
        "test_rows": len(tests),
        "holm_p_values": sorted({float(row["p_holm"]) for row in tests}),
        "max_random_order_macro_f1_range": max(float(row["macro_f1_random_range"]) for row in random_dispersion),
        "uniqueness_rows": len(uniqueness),
        "two_shot_named_unique": sorted({int(row["named_unique_count"]) for row in uniqueness if int(row["total_shots"]) == 2}),
        "four_shot_named_unique": sorted({int(row["named_unique_count"]) for row in uniqueness if int(row["total_shots"]) == 4}),
        "eight_shot_named_unique": sorted({int(row["named_unique_count"]) for row in uniqueness if int(row["total_shots"]) == 8}),
        "multiplicity_rows": len(multiplicity),
        "multiplicity_random_duplicate_strata": [int(row["random_duplicate_strata"]) for row in multiplicity],
        "legacy_parser_fields": sorted(field for field in aggregate[0] if "parser" in field.lower()),
    }

    expected = {
        "aggregate_rows": 360,
        "raw_evaluations_from_aggregate": 72000,
        "dataset_count": 3,
        "shot_values": [2, 4, 8],
        "seed_values": [1, 2, 3, 4, 5],
        "order_count": 8,
        "all_n_eval_200": True,
        "max_accuracy_spread": 0.05500000000000005,
        "max_macro_f1_spread": 0.05708452539254005,
        "spread_rows": 9,
        "test_rows": 6,
        "holm_p_values": [1.0],
        "max_random_order_macro_f1_range": 0.05708452539254005,
        "uniqueness_rows": 45,
        "two_shot_named_unique": [2],
        "four_shot_named_unique": [5, 6, 7, 8],
        "eight_shot_named_unique": [8],
        "multiplicity_rows": 3,
        "multiplicity_random_duplicate_strata": [15, 5, 0],
        "legacy_parser_fields": [],
    }
    failures = {}
    for key, value in expected.items():
        actual = result[key]
        if isinstance(value, float):
            ok = abs(float(actual) - value) < 1e-12
        else:
            ok = actual == value
        if not ok:
            failures[key] = {"expected": value, "actual": actual}

    summary_checks = {
        "summary_aggregate_rows": summary["aggregate_rows"] == 360,
        "summary_raw_evaluations": summary["raw_evaluations"] == 72000,
        "summary_max_accuracy_spread": abs(float(summary["max_accuracy_spread"]) - 0.05500000000000005) < 1e-12,
        "summary_max_macro_f1_spread": abs(float(summary["max_macro_f1_spread"]) - 0.05708452539254005) < 1e-12,
    }
    for key, ok in summary_checks.items():
        if not ok:
            failures[key] = "public_summary.json mismatch"

    return {
        "status": "PASS" if not failures else "FAIL",
        "computed": result,
        "expected": expected,
        "failures": failures,
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
