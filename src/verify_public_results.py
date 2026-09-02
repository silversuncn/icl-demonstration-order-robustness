#!/usr/bin/env python3
"""Verify the public processed-results bundle for the order-robustness study."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

FORMAL_AGGREGATE_SHA256 = "9c53c07020257dab1130a316092d77706bcdcadd32c4c3089207f2c411a18620"
EXPECTED_ORDER_IDS = [
    "alternating",
    "label_grouped",
    "random_0",
    "random_1",
    "random_2",
    "random_3",
    "random_4",
    "seeded_base",
]
PER_SAMPLE_FIELDS = [
    "dataset",
    "model_id",
    "total_shots",
    "seed",
    "order_id",
    "eval_id",
    "gold_label",
    "predicted_label",
    "correct",
    "score_A",
    "score_B",
]
AGGREGATE_FIELDS = [
    "dataset",
    "model_id",
    "total_shots",
    "seed",
    "order_family",
    "order_id",
    "n_eval",
    "accuracy",
    "macro_f1",
    "mean_prompt_tokens",
    "total_latency_seconds",
]
EXPECTED_DATA_HASH_FILES = {
    "cluster_diagnostics_v2.csv",
    "cluster_diagnostics_v2.json",
    "cluster_per_shot_differences_v2.csv",
    "cluster_sign_flip_tests_v2.csv",
    "descriptive_statistics.csv",
    "descriptive_table_iv_v2.csv",
    "formal_aggregate.csv",
    "formal_predictions_original_360.csv",
    "model_environment_record_v2.json",
    "permutation_mapping.json",
    "permutation_multiplicity_by_shot.csv",
    "public_summary.json",
    "random_order_dispersion.csv",
    "random_permutation_diagnostics_v2.csv",
    "results_2shot_exhaustive.csv",
    "results_4shot_exhaustive.csv",
    "results_8shot_extended.csv",
    "spread_by_dataset_shot.csv",
    "support_permutation_uniqueness.csv",
    "two_shot_deterministic_permutation_diagnostics_v2.csv",
    "two_shot_label_grouped_alternating_equivalence_v2.csv",
    "validation_summary.csv",
    "worst_order_regret_by_order.csv",
}
EXPECTED_FIGURE_HASH_FILES = {
    "order_accuracy_heatmap.png",
    "order_accuracy_heatmap_v2.pdf",
    "order_accuracy_heatmap_v2.png",
    "order_spread_by_dataset_shot.pdf",
    "order_spread_by_dataset_shot.png",
    "order_spread_by_dataset_shot_v2.pdf",
    "order_spread_by_dataset_shot_v2.png",
    "random_order_dispersion_by_dataset_shot.png",
    "random_order_dispersion_by_dataset_shot_v2.pdf",
    "random_order_dispersion_by_dataset_shot_v2.png",
}
STALE_HOLM_FILE = "statistical_tests_" + "holm.csv"
STALE_ALIAS_FILE = "formal_aggregate_" + "seeded_base_v2.csv"
EXPECTED_TABLE_IV_DISPLAY = {
    ("seeded_base_minus_evaluated_unique_permutation_mean", "accuracy"): ("-0.0009", "-0.0002"),
    ("seeded_base_minus_evaluated_unique_permutation_mean", "macro_f1"): ("-0.0011", "-0.0000"),
    ("label_grouped_minus_evaluated_unique_permutation_mean", "accuracy"): ("-0.0023", "-0.0017"),
    ("label_grouped_minus_evaluated_unique_permutation_mean", "macro_f1"): ("-0.0024", "-0.0008"),
    ("alternating_minus_evaluated_unique_permutation_mean", "accuracy"): ("-0.0008", "-0.0008"),
    ("alternating_minus_evaluated_unique_permutation_mean", "macro_f1"): ("-0.0007", "-0.0008"),
}
EXPECTED_SPREAD_DISPLAY = {
    ("mrpc", 2): ("0.0230", "0.0450", "0.0253", "0.0542"),
    ("mrpc", 4): ("0.0570", "0.0700", "0.0608", "0.0790"),
    ("mrpc", 8): ("0.0480", "0.0600", "0.0509", "0.0632"),
    ("rte", 2): ("0.0060", "0.0100", "0.0059", "0.0098"),
    ("rte", 4): ("0.0260", "0.0350", "0.0265", "0.0350"),
    ("rte", 8): ("0.0370", "0.0400", "0.0380", "0.0414"),
    ("sst2", 2): ("0.0080", "0.0150", "0.0080", "0.0150"),
    ("sst2", 4): ("0.0340", "0.0550", "0.0341", "0.0552"),
    ("sst2", 8): ("0.0310", "0.0400", "0.0311", "0.0403"),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def display4(value: float) -> str:
    return f"{float(value):.4f}"


def add_failure(failures: dict[str, object], key: str, expected: object, actual: object) -> None:
    failures[key] = {"expected": expected, "actual": actual}


def macro_f1(golds: list[str], preds: list[str]) -> float:
    labels = sorted(set(golds) | set(preds))
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall))
    return statistics.mean(scores)


def aggregate_key(row: dict[str, str]) -> tuple[str, str, int, int, str]:
    return (
        row["dataset"],
        row["model_id"],
        as_int(row["total_shots"]),
        as_int(row["seed"]),
        row["order_id"],
    )


def public_aggregate_key(row: dict[str, str]) -> tuple[str, str, int, int, str, str]:
    return (
        row["dataset"],
        row["model_id"],
        as_int(row["total_shots"]),
        as_int(row["seed"]),
        row["order_family"],
        row["order_id"],
    )


def spread_key(row: dict[str, str]) -> tuple[str, int]:
    return (row["dataset"], as_int(row["total_shots"]))


def compute_spread_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, object]]:
    by_stratum: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[(row["dataset"], as_int(row["total_shots"]), as_int(row["seed"]))].append(row)

    result: dict[tuple[str, int], dict[str, object]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        for shots in [2, 4, 8]:
            acc_spreads = []
            f1_spreads = []
            aggregate_cells = []
            for seed in [1, 2, 3, 4, 5]:
                stratum_rows = by_stratum[(dataset, shots, seed)]
                aggregate_cells.append(len(stratum_rows))
                acc = [as_float(row["accuracy"]) for row in stratum_rows]
                f1 = [as_float(row["macro_f1"]) for row in stratum_rows]
                acc_spreads.append(max(acc) - min(acc))
                f1_spreads.append(max(f1) - min(f1))
            result[(dataset, shots)] = {
                "dataset": dataset,
                "total_shots": shots,
                "seed_count": 5,
                "evaluated_unique_permutations_per_seed": {2: 2, 4: 24, 8: 38}[shots],
                "aggregate_cells_used_per_seed_in_public_package": aggregate_cells[0],
                "mean_accuracy_spread": statistics.mean(acc_spreads),
                "median_accuracy_spread": statistics.median(acc_spreads),
                "max_accuracy_spread": max(acc_spreads),
                "mean_macro_f1_spread": statistics.mean(f1_spreads),
                "median_macro_f1_spread": statistics.median(f1_spreads),
                "max_macro_f1_spread": max(f1_spreads),
            }
    return result


def exact_sign_flip(values: list[float]) -> tuple[float, int, int, float]:
    observed = statistics.mean(values)
    threshold = abs(observed) - 1e-15
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = sum(sign * value for sign, value in zip(signs, values)) / len(values)
        if abs(statistic) >= threshold:
            extreme += 1
    return observed, extreme, total, extreme / total


def holm_adjust(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adjusted = [dict(row) for row in rows]
    previous = 0.0
    for rank, row in enumerate(sorted(adjusted, key=lambda item: (item["p_raw"], item["comparison"], item["metric"])), start=1):
        value = min(1.0, (len(adjusted) - rank + 1) * float(row["p_raw"]))
        previous = max(previous, value)
        row["p_holm"] = previous
        row["holm_rank"] = rank
    order_index = {"seeded_base": 0, "label_grouped": 1, "alternating": 2}
    metric_index = {"accuracy": 0, "macro_f1": 1}
    return sorted(
        adjusted,
        key=lambda row: (order_index[row["comparison"].split("_minus_")[0]], metric_index[row["metric"]]),
    )


def compute_cluster_tests(per_shot: list[dict[str, str]]) -> list[dict[str, object]]:
    by_cluster: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    for row in per_shot:
        by_cluster[
            (
                row["dataset"],
                as_int(row["seed"]),
                row["comparison"],
                row["order_id"],
                row["metric"],
            )
        ].append(as_float(row["difference"]))

    values_by_test: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_dataset, _seed, comparison, _order_id, metric), values in by_cluster.items():
        values_by_test[(comparison, metric)].append(statistics.mean(values))

    tests = []
    for comparison, metric in sorted(values_by_test):
        values = values_by_test[(comparison, metric)]
        observed, extreme, total, p_raw = exact_sign_flip(values)
        tests.append(
            {
                "comparison": comparison,
                "metric": metric,
                "n_clusters": len(values),
                "n_nonzero_clusters": sum(abs(value) > 1e-15 for value in values),
                "mean_difference": observed,
                "median_difference": statistics.median(values),
                "positive_clusters": sum(value > 1e-15 for value in values),
                "negative_clusters": sum(value < -1e-15 for value in values),
                "zero_clusters": sum(abs(value) <= 1e-15 for value in values),
                "test_method": "exact_cluster_sign_flip_mean_2^15_evaluated_unique_permutation_mean",
                "p_raw": p_raw,
                "p_holm": None,
                "holm_rank": None,
                "extreme_assignments": extreme,
                "total_assignments": total,
            }
        )
    return holm_adjust(tests)


def checksum_public_files() -> list[str]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        parts = set(path.relative_to(ROOT).parts)
        if ".git" in parts or "__pycache__" in parts or rel == "checksums_sha256.txt":
            continue
        if path.suffix == ".pyc":
            continue
        files.append(rel)
    return sorted(files)


def parse_checksums() -> dict[str, str]:
    checksums = {}
    path = ROOT / "checksums_sha256.txt"
    if not path.exists():
        return checksums
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        checksums[rel.strip()] = digest
    return checksums


def verify() -> dict[str, object]:
    failures: dict[str, object] = {}

    formal = read_csv("formal_aggregate.csv")
    exhaustive_2 = read_csv("results_2shot_exhaustive.csv")
    exhaustive_4 = read_csv("results_4shot_exhaustive.csv")
    exhaustive_8 = read_csv("results_8shot_extended.csv")
    combined = formal + exhaustive_2 + exhaustive_4 + exhaustive_8
    per_sample = read_csv("formal_predictions_original_360.csv")
    spread = read_csv("spread_by_dataset_shot.csv")
    multiplicity = read_csv("permutation_multiplicity_by_shot.csv")
    descriptive = read_csv("descriptive_table_iv_v2.csv")
    descriptive_copy = (DATA / "descriptive_statistics.csv").read_bytes()
    cluster_tests = read_csv("cluster_sign_flip_tests_v2.csv")
    per_shot = read_csv("cluster_per_shot_differences_v2.csv")
    cluster_diagnostics = read_csv("cluster_diagnostics_v2.csv")
    validation_summary = read_csv("validation_summary.csv")
    public_summary = read_json("public_summary.json")
    cluster_summary = read_json("cluster_diagnostics_v2.json")
    manifest = read_json("public_v2_manifest.json")
    checksums = parse_checksums()

    duplicate_public_keys = len(combined) - len({public_aggregate_key(row) for row in combined})
    computed_spread = compute_spread_rows(combined)
    spread_by_key = {spread_key(row): row for row in spread}
    descriptive_by_key = {(row["comparison"], row["metric"]): row for row in descriptive}
    expected_cluster_tests = {
        (row["comparison"], row["metric"]): row
        for row in compute_cluster_tests(per_shot)
    }
    cluster_tests_by_key = {(row["comparison"], row["metric"]): row for row in cluster_tests}

    by_cell: dict[tuple[str, str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_sample:
        by_cell[aggregate_key(row)].append(row)
    aggregate_by_cell = {aggregate_key(row): row for row in formal}
    aggregate_mismatches = []
    for key, rows in by_cell.items():
        aggregate_row = aggregate_by_cell.get(key)
        if aggregate_row is None:
            aggregate_mismatches.append({"key": key, "reason": "missing_aggregate_row"})
            continue
        golds = [row["gold_label"] for row in rows]
        preds = [row["predicted_label"] for row in rows]
        accuracy = sum(as_int(row["correct"]) for row in rows) / len(rows)
        f1 = macro_f1(golds, preds)
        if not close(accuracy, as_float(aggregate_row["accuracy"])) or not close(f1, as_float(aggregate_row["macro_f1"])):
            aggregate_mismatches.append(
                {
                    "key": key,
                    "accuracy": accuracy,
                    "aggregate_accuracy": aggregate_row["accuracy"],
                    "macro_f1": f1,
                    "aggregate_macro_f1": aggregate_row["macro_f1"],
                }
            )

    computed = {
        "formal_aggregate_rows": len(formal),
        "exhaustive_2shot_rows": len(exhaustive_2),
        "exhaustive_4shot_rows": len(exhaustive_4),
        "exhaustive_8shot_rows": len(exhaustive_8),
        "aggregate_rows_in_public_package": len(combined),
        "evaluation_instances_in_public_package": sum(as_int(row["n_eval"]) for row in combined),
        "per_sample_rows": len(per_sample),
        "per_sample_cells": len(by_cell),
        "dataset_count": len({row["dataset"] for row in formal}),
        "shot_values": sorted({as_int(row["total_shots"]) for row in formal}),
        "seed_values": sorted({as_int(row["seed"]) for row in formal}),
        "order_ids": sorted({row["order_id"] for row in formal}),
        "all_aggregate_n_eval_200": all(as_int(row["n_eval"]) == 200 for row in combined),
        "duplicate_public_aggregate_keys": duplicate_public_keys,
        "formal_aggregate_sha256": sha256_path(DATA / "formal_aggregate.csv"),
        "formal_aggregate_schema": list(formal[0].keys()),
        "legacy_parser_fields": sorted(field for field in formal[0] if "parser" in field.lower()),
        "per_sample_schema": list(per_sample[0].keys()),
        "per_sample_cell_sizes": sorted({len(rows) for rows in by_cell.values()}),
        "per_sample_order_ids": sorted({row["order_id"] for row in per_sample}),
        "per_sample_forbidden_columns": sorted(
            field
            for field in per_sample[0]
            if field.lower() in {"prompt", "support_examples", "label_text"} or "path" in field.lower()
        ),
        "score_fields_finite": all(
            math.isfinite(as_float(row["score_A"])) and math.isfinite(as_float(row["score_B"]))
            for row in per_sample
        ),
        "aggregate_mismatch_count": len(aggregate_mismatches),
        "spread_rows": len(spread),
        "max_accuracy_spread": max(row["max_accuracy_spread"] for row in computed_spread.values()),
        "max_macro_f1_spread": max(row["max_macro_f1_spread"] for row in computed_spread.values()),
        "descriptive_rows": len(descriptive),
        "descriptive_copy_matches": descriptive_copy == (DATA / "descriptive_table_iv_v2.csv").read_bytes(),
        "descriptive_p_value_columns": sorted(field for field in descriptive[0] if field.startswith("p_") or field.endswith("_p")),
        "descriptive_significance_columns": sorted(field for field in descriptive[0] if ("signifi" + "cant") in field.lower()),
        "cluster_per_shot_rows": len(per_shot),
        "cluster_diagnostic_rows": len(cluster_diagnostics),
        "cluster_test_rows": len(cluster_tests),
        "cluster_test_holm_p_values": sorted({as_float(row["p_holm"]) for row in cluster_tests}),
        "cluster_count": cluster_summary["cluster_design"]["cluster_count"],
        "multiplicity_rows": len(multiplicity),
        "multiplicity_random_duplicate_strata": [as_int(row["random_duplicate_strata"]) for row in multiplicity],
        "legacy_holm_table_absent": not (DATA / STALE_HOLM_FILE).exists(),
        "seeded_base_alias_file_absent": not (DATA / STALE_ALIAS_FILE).exists(),
        "manifest_status": manifest.get("status"),
        "manifest_data_files": sorted((manifest.get("data_sha256") or {}).keys()),
        "manifest_figure_files": sorted((manifest.get("figure_sha256") or {}).keys()),
        "checksum_entries": len(checksums),
    }

    expected_scalar = {
        "formal_aggregate_rows": 360,
        "exhaustive_2shot_rows": 30,
        "exhaustive_4shot_rows": 360,
        "exhaustive_8shot_rows": 450,
        "aggregate_rows_in_public_package": 1200,
        "evaluation_instances_in_public_package": 240000,
        "per_sample_rows": 72000,
        "per_sample_cells": 360,
        "dataset_count": 3,
        "shot_values": [2, 4, 8],
        "seed_values": [1, 2, 3, 4, 5],
        "order_ids": EXPECTED_ORDER_IDS,
        "all_aggregate_n_eval_200": True,
        "duplicate_public_aggregate_keys": 0,
        "formal_aggregate_sha256": FORMAL_AGGREGATE_SHA256,
        "formal_aggregate_schema": AGGREGATE_FIELDS,
        "legacy_parser_fields": [],
        "per_sample_schema": PER_SAMPLE_FIELDS,
        "per_sample_cell_sizes": [200],
        "per_sample_order_ids": EXPECTED_ORDER_IDS,
        "per_sample_forbidden_columns": [],
        "score_fields_finite": True,
        "aggregate_mismatch_count": 0,
        "spread_rows": 9,
        "descriptive_rows": 6,
        "descriptive_copy_matches": True,
        "descriptive_p_value_columns": [],
        "descriptive_significance_columns": [],
        "cluster_per_shot_rows": 270,
        "cluster_diagnostic_rows": 90,
        "cluster_test_rows": 6,
        "cluster_test_holm_p_values": [1.0],
        "cluster_count": 15,
        "multiplicity_rows": 3,
        "multiplicity_random_duplicate_strata": [15, 5, 0],
        "legacy_holm_table_absent": True,
        "seeded_base_alias_file_absent": True,
        "manifest_status": "PASS",
    }
    for key, expected in expected_scalar.items():
        if computed.get(key) != expected:
            add_failure(failures, key, expected, computed.get(key))

    for key, expected in EXPECTED_SPREAD_DISPLAY.items():
        actual = spread_by_key.get(key)
        recalculated = computed_spread.get(key)
        if actual is None or recalculated is None:
            add_failure(failures, f"spread_missing_{key}", expected, actual)
            continue
        display = (
            display4(actual["mean_accuracy_spread"]),
            display4(actual["max_accuracy_spread"]),
            display4(actual["mean_macro_f1_spread"]),
            display4(actual["max_macro_f1_spread"]),
        )
        if display != expected:
            add_failure(failures, f"spread_display_{key}", expected, display)
        for field in [
            "mean_accuracy_spread",
            "median_accuracy_spread",
            "max_accuracy_spread",
            "mean_macro_f1_spread",
            "median_macro_f1_spread",
            "max_macro_f1_spread",
        ]:
            if not close(as_float(actual[field]), float(recalculated[field])):
                add_failure(failures, f"spread_recomputed_{key}_{field}", recalculated[field], actual[field])

    if set(descriptive_by_key) != set(EXPECTED_TABLE_IV_DISPLAY):
        add_failure(failures, "table_iv_comparisons", sorted(EXPECTED_TABLE_IV_DISPLAY), sorted(descriptive_by_key))
    for key, expected in EXPECTED_TABLE_IV_DISPLAY.items():
        row = descriptive_by_key.get(key)
        if row is None:
            continue
        actual = (row["display_mean_difference"], row["display_median_difference"])
        if actual != expected:
            add_failure(failures, f"table_iv_display_{key}", expected, actual)
        if row["description_policy"] != "mean_over_45_dataset_shot_seed_strata":
            add_failure(failures, f"table_iv_policy_{key}", "mean_over_45_dataset_shot_seed_strata", row["description_policy"])

    if set(cluster_tests_by_key) != set(EXPECTED_TABLE_IV_DISPLAY):
        add_failure(failures, "cluster_test_comparisons", sorted(EXPECTED_TABLE_IV_DISPLAY), sorted(cluster_tests_by_key))
    for key, expected_row in expected_cluster_tests.items():
        actual = cluster_tests_by_key.get(key)
        if actual is None:
            continue
        for field in [
            "n_clusters",
            "n_nonzero_clusters",
            "positive_clusters",
            "negative_clusters",
            "zero_clusters",
            "holm_rank",
            "extreme_assignments",
            "total_assignments",
        ]:
            if as_int(actual[field]) != int(expected_row[field]):
                add_failure(failures, f"cluster_{key}_{field}", expected_row[field], actual[field])
        for field in ["mean_difference", "median_difference", "p_raw", "p_holm"]:
            if not close(as_float(actual[field]), float(expected_row[field])):
                add_failure(failures, f"cluster_{key}_{field}", expected_row[field], actual[field])
        if actual["test_method"] != expected_row["test_method"]:
            add_failure(failures, f"cluster_{key}_test_method", expected_row["test_method"], actual["test_method"])

    expected_validation = {
        "aggregate_cells_in_public_package": "1200",
        "original_named_grid_cells": "360",
        "exhaustive_extension_cells": "840",
        "evaluation_instances_in_public_package": "240000",
        "per_sample_prediction_rows": "72000",
        "per_sample_scope": "original_named_grid_360_aggregate_cells_only",
        "duplicate_public_aggregate_keys": "0",
        "all_aggregate_n_eval_200": "True",
        "raw_aggregate_mismatch_count": "0",
        "legacy_holm_table_absent": "True",
        "formal_aggregate_sha256_unchanged": "True",
    }
    if len(validation_summary) != 1:
        add_failure(failures, "validation_summary_rows", 1, len(validation_summary))
    else:
        for key, expected in expected_validation.items():
            if validation_summary[0].get(key) != expected:
                add_failure(failures, f"validation_summary_{key}", expected, validation_summary[0].get(key))

    summary_checks = {
        "aggregate_rows_in_public_package": 1200,
        "original_named_grid_aggregate_rows": 360,
        "exhaustive_extension_aggregate_rows": 840,
        "evaluation_instances_in_public_package": 240000,
        "per_sample_prediction_rows": 72000,
        "per_sample_scope": "original_named_grid_360_aggregate_cells_only",
        "table_iv_comparison_baseline": "evaluated_unique_permutation_mean",
    }
    for key, expected in summary_checks.items():
        if public_summary.get(key) != expected:
            add_failure(failures, f"public_summary_{key}", expected, public_summary.get(key))
    if display4(public_summary.get("max_accuracy_spread", 0.0)) != "0.0700":
        add_failure(failures, "public_summary_max_accuracy_spread", "0.0700", public_summary.get("max_accuracy_spread"))
    if display4(public_summary.get("max_macro_f1_spread", 0.0)) != "0.0790":
        add_failure(failures, "public_summary_max_macro_f1_spread", "0.0790", public_summary.get("max_macro_f1_spread"))
    if cluster_summary.get("status") != "PASS":
        add_failure(failures, "cluster_summary_status", "PASS", cluster_summary.get("status"))

    data_hashes = manifest.get("data_sha256") or {}
    figure_hashes = manifest.get("figure_sha256") or {}
    if set(data_hashes) != EXPECTED_DATA_HASH_FILES:
        add_failure(failures, "manifest_data_file_set", sorted(EXPECTED_DATA_HASH_FILES), sorted(data_hashes))
    if set(figure_hashes) != EXPECTED_FIGURE_HASH_FILES:
        add_failure(failures, "manifest_figure_file_set", sorted(EXPECTED_FIGURE_HASH_FILES), sorted(figure_hashes))
    stale_names = {STALE_HOLM_FILE, STALE_ALIAS_FILE}
    stale_manifest_names = sorted(stale_names & (set(data_hashes) | set(figure_hashes)))
    if stale_manifest_names:
        add_failure(failures, "manifest_stale_names", [], stale_manifest_names)
    for name, digest in data_hashes.items():
        actual = sha256_path(DATA / name)
        if actual != digest:
            add_failure(failures, f"manifest_data_hash_{name}", actual, digest)
    for name, digest in figure_hashes.items():
        actual = sha256_path(FIGURES / name)
        if actual != digest:
            add_failure(failures, f"manifest_figure_hash_{name}", actual, digest)

    public_files = checksum_public_files()
    if set(checksums) != set(public_files):
        add_failure(failures, "checksum_file_set", public_files, sorted(checksums))
    for rel, digest in checksums.items():
        actual = sha256_path(ROOT / rel)
        if actual != digest:
            add_failure(failures, f"checksum_hash_{rel}", actual, digest)

    return {
        "status": "PASS" if not failures else "FAIL",
        "computed": computed,
        "failures": failures,
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
