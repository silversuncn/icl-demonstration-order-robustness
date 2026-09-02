#!/usr/bin/env python3
"""Regenerate demonstration-order figures and order-uniqueness evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


FIGURE_STEMS = (
    "order_spread_by_dataset_shot",
    "order_accuracy_heatmap",
    "random_order_dispersion_by_dataset_shot",
)
DATASET_ORDER = ("mrpc", "rte", "sst2")
SHOT_ORDER = (2, 4, 8)
ORDER_IDS = ("seeded_base", "label_grouped", "alternating", "random_0", "random_1", "random_2", "random_3", "random_4")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stratum_labels() -> list[str]:
    return [f"{dataset.upper()}-{shots}" for dataset in DATASET_ORDER for shots in SHOT_ORDER]


def order_accuracy_grid(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[list[float]]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["total_shots"]), row["order_id"])].append(float(row["accuracy"]))
    values = [
        [statistics.mean(grouped[(dataset, shots, order_id)]) for dataset in DATASET_ORDER for shots in SHOT_ORDER]
        for order_id in ORDER_IDS
    ]
    return list(ORDER_IDS), stratum_labels(), values


def random_accuracy_range_groups(rows: list[dict[str, str]]) -> tuple[list[str], list[list[float]]]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["total_shots"]))].append(
            (int(row["seed"]), float(row["accuracy_random_range"]))
        )
    groups = [
        [value for _, value in sorted(grouped[(dataset, shots)])]
        for dataset in DATASET_ORDER
        for shots in SHOT_ORDER
    ]
    return stratum_labels(), groups


def support_permutation_uniqueness(predictions_jsonl: Path) -> list[dict[str, int | str]]:
    orders: dict[tuple[str, int, int, str], tuple[int, ...]] = {}
    with predictions_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["dataset"], int(row["total_shots"]), int(row["seed"]), row["order_id"])
            orders.setdefault(key, tuple(int(value) for value in row["support_example_ids"]))

    grouped: dict[tuple[str, int, int], dict[str, tuple[int, ...]]] = defaultdict(dict)
    for (dataset, shots, seed, order_id), permutation in orders.items():
        grouped[(dataset, shots, seed)][order_id] = permutation

    result = []
    for (dataset, shots, seed), variants in sorted(grouped.items()):
        named = [variants[order_id] for order_id in ORDER_IDS]
        random = [variants[order_id] for order_id in ORDER_IDS if order_id.startswith("random_")]
        result.append(
            {
                "dataset": dataset,
                "total_shots": shots,
                "seed": seed,
                "named_variant_count": len(named),
                "named_unique_count": len(set(named)),
                "random_variant_count": len(random),
                "random_unique_count": len(set(random)),
            }
        )
    return result


def require_path(path: Path | None, option: str) -> Path:
    if path is None:
        raise SystemExit(f"{option} is required unless --aggregate-only is used")
    return path


def permutation_multiplicity_by_shot(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["total_shots"])].append(row)

    result = []
    for shots in SHOT_ORDER:
        shot_rows = grouped[shots]
        named = [int(row["named_unique_count"]) for row in shot_rows]
        random = [int(row["random_unique_count"]) for row in shot_rows]
        result.append(
            {
                "total_shots": shots,
                "strata": len(shot_rows),
                "named_unique_min": min(named),
                "named_unique_mean": statistics.mean(named),
                "named_unique_max": max(named),
                "random_unique_min": min(random),
                "random_unique_mean": statistics.mean(random),
                "random_unique_max": max(random),
                "random_duplicate_strata": sum(value < 5 for value in random),
            }
        )
    return result


def apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")


def render_figures(aggregate_csv: Path, spread_csv: Path, random_csv: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)

    spread_rows = read_csv(spread_csv)
    x = list(range(9))
    mean_accuracy = [float(row["mean_accuracy_spread"]) for row in spread_rows]
    max_accuracy = [float(row["max_accuracy_spread"]) for row in spread_rows]
    fig, ax = plt.subplots(figsize=(7.16, 2.25))
    width = 0.38
    ax.bar([value - width / 2 for value in x], mean_accuracy, width, label="Mean across seeds", color="#2563a6")
    ax.bar([value + width / 2 for value in x], max_accuracy, width, label="Maximum across seeds", color="#d97706")
    ax.set_xticks(x, stratum_labels())
    ax.set_ylabel("Accuracy spread")
    ax.grid(axis="y", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_both(fig, output_dir, "order_spread_by_dataset_shot")
    plt.close(fig)

    order_ids, strata, grid = order_accuracy_grid(read_csv(aggregate_csv))
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    image = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=min(map(min, grid)), vmax=max(map(max, grid)))
    ax.set_xticks(range(len(strata)), strata)
    ax.set_yticks(range(len(order_ids)), [value.replace("_", " ") for value in order_ids])
    for y, row in enumerate(grid):
        for x_index, value in enumerate(row):
            normalized = (value - image.norm.vmin) / (image.norm.vmax - image.norm.vmin)
            ax.text(x_index, y, f"{value:.3f}", ha="center", va="center", fontsize=6.2, color="white" if normalized > 0.58 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean accuracy", fontsize=8)
    fig.tight_layout()
    save_both(fig, output_dir, "order_accuracy_heatmap")
    plt.close(fig)

    labels, groups = random_accuracy_range_groups(read_csv(random_csv))
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    boxes = ax.boxplot(groups, patch_artist=True, widths=0.58, medianprops={"color": "black"})
    for patch in boxes["boxes"]:
        patch.set_facecolor("#8fb9d8")
    ax.scatter(
        [index for index, group in enumerate(groups, start=1) for _ in group],
        [value for group in groups for value in group],
        s=9,
        color="#b43c39",
        zorder=3,
    )
    ax.set_xticks(range(1, 10), labels)
    ax.set_ylabel("Within-seed random-order\naccuracy range")
    ax.grid(axis="y", linewidth=0.35, alpha=0.28)
    fig.tight_layout()
    save_both(fig, output_dir, "random_order_dispersion_by_dataset_shot")
    plt.close(fig)


def write_uniqueness_csv(rows: list[dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--spread-csv", type=Path, required=True)
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uniqueness-csv", type=Path)
    parser.add_argument("--multiplicity-csv", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    render_figures(args.aggregate_csv, args.spread_csv, args.random_csv, args.output_dir)
    if args.aggregate_only:
        print(f"PASS: generated {len(FIGURE_STEMS) * 2} figure files in aggregate-only mode")
        return 0

    predictions_jsonl = require_path(args.predictions_jsonl, "--predictions-jsonl")
    uniqueness_csv = require_path(args.uniqueness_csv, "--uniqueness-csv")
    uniqueness = support_permutation_uniqueness(predictions_jsonl)
    write_uniqueness_csv(uniqueness, uniqueness_csv)
    if args.multiplicity_csv:
        write_rows_csv(permutation_multiplicity_by_shot(uniqueness), args.multiplicity_csv)
    print(f"PASS: generated {len(FIGURE_STEMS) * 2} figure files and {uniqueness_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
