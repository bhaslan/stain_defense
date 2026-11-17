from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.stage05_C_vmatch import config_vmatch as cfg

PANEL_CLASSES = ("direct", "palindromic_inverted")
REQUIRED_FIELDS = {"repeat_class", "length_1", "length_2", "evalue"}

@dataclass(frozen=True, slots=True)
class MatchPoint:

    repeat_class: str
    shorter_arm_length: int
    evalue: float

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gridsize",
        type=int,
        default=60,
        help="Number of hexagons across each panel (default: 60).",
    )
    parser.add_argument(
        "--max-shorter-arm-length",
        type=int,
        default=60,
        help=(
            "Plot only rows whose shorter arm is at most this many nucleotides "
            "(default: 60)."
        ),
    )
    parser.add_argument(
        "--max-neg-log10-evalue",
        type=float,
        default=3.0,
        help=(
            "Plot only rows at or below this −log10(E-value), filtering before "
            "hexbinning (default: 3)."
        ),
    )
    return parser.parse_args()

def _required_value(row: dict[str, str], field: str, line_number: int) -> str:
    value = row[field].strip()
    if not value:
        raise ValueError(f"Missing {field} at {line_number} in the matches table")
    return value

def _load_points(path: Path) -> list[MatchPoint]:
    if not path.exists():
        raise FileNotFoundError(path)

    points: list[MatchPoint] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(
                f"{path.name} lacks required columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            repeat_class = _required_value(row, "repeat_class", line_number)
            if repeat_class not in PANEL_CLASSES:
                continue
            try:
                length_1 = int(_required_value(row, "length_1", line_number))
                length_2 = int(_required_value(row, "length_2", line_number))
                evalue = float(_required_value(row, "evalue", line_number))
            except ValueError as error:
                raise ValueError(f"Invalid numeric value at {path}:{line_number}") from error
            if length_1 <= 0 or length_2 <= 0:
                raise ValueError(f"Arm lengths must be positive at {path}:{line_number}")
            if not math.isfinite(evalue) or evalue < 0:
                raise ValueError(
                    f"E-value must be finite and non-negative at {path}:{line_number}"
                )
            points.append(MatchPoint(repeat_class, min(length_1, length_2), evalue))

    if not points:
        raise ValueError(f"{path} contains no active Vmatch repeat rows")
    return points

def _zero_plot_value(points: list[MatchPoint]) -> float:
    positive_values = [point.evalue for point in points if point.evalue > 0]
    return min(positive_values) / 10 if positive_values else 1e-300

def _neg_log10_evalue(point: MatchPoint) -> float:
    return -math.log10(point.evalue) if point.evalue > 0 else math.inf

def _plot_panel(
    axis,
    points: list[MatchPoint],
    repeat_class: str,
    zero_plot_value: float,
    gridsize: int,
):
    panel_points = [point for point in points if point.repeat_class == repeat_class]
    title = repeat_class.replace("_", " ")
    if not panel_points:
        axis.text(
            0.5,
            0.5,
            "No selected rows",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
        return None

    lengths = [point.shorter_arm_length for point in panel_points]
    strengths = [
        _neg_log10_evalue(point)
        if point.evalue > 0
        else -math.log10(zero_plot_value)
        for point in panel_points
    ]
    hexagons = axis.hexbin(
        lengths,
        strengths,
        gridsize=gridsize,
        mincnt=1,
        cmap="magma_r",
        linewidths=0,
    )
    axis.set_title(f"{title} (n = {len(panel_points):,})")
    axis.axhline(
        2,
        color="#39d0c3",
        linestyle="--",
        linewidth=1.2,
        label="E-value = 0.01",
    )
    axis.legend(frameon=False, loc="lower right")
    return hexagons

def plot(
    points: list[MatchPoint],
    gridsize: int,
    maximum_shorter_arm_length: int,
    maximum_neg_log10_evalue: float,
):
    if gridsize < 2:
        raise ValueError("--gridsize must be at least 2")

    zero_plot_value = _zero_plot_value(points)
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    hexagons = [
        _plot_panel(axis, points, repeat_class, zero_plot_value, gridsize)
        for axis, repeat_class in zip(axes, PANEL_CLASSES)
    ]
    for axis in axes:
        axis.set_xlabel("Shorter arm length (nt)")
        axis.grid(False)
    axes[0].set_ylabel("−log₁₀(E-value)  (higher = stronger)")
    plotted = [item for item in hexagons if item is not None]
    if plotted:
        maximum_count = max(item.get_array().max() for item in plotted)
        common_norm = LogNorm(vmin=1, vmax=max(2, maximum_count))
        for item in plotted:
            item.set_norm(common_norm)
        colorbar = figure.colorbar(plotted[0], ax=axes, pad=0.02)
        colorbar.set_label("Selected Step 3C row count per hexagon (log scale)")
    figure.suptitle(
        "Selected Step 3C Vmatch calls: E-value versus arm length\n"
        f"shorter arm ≤ {maximum_shorter_arm_length} nt; "
        f"−log₁₀(E-value) ≤ {maximum_neg_log10_evalue:g}",
        fontsize=13,
    )
    plt.show()

def main():
    args = _arguments()
    if args.max_shorter_arm_length < 1:
        raise ValueError("--max-shorter-arm-length must be positive")
    if not math.isfinite(args.max_neg_log10_evalue):
        raise ValueError("--max-neg-log10-evalue must be finite")

    all_points = _load_points(cfg.STEP3_MATCHES_PATH)
    length_filtered = [
        point
        for point in all_points
        if point.shorter_arm_length <= args.max_shorter_arm_length
    ]
    points = [
        point
        for point in length_filtered
        if _neg_log10_evalue(point) <= args.max_neg_log10_evalue
    ]
    if not points:
        raise ValueError(
            "No active Vmatch rows remain after the requested plotting filters."
        )
    plot(
        points,
        args.gridsize,
        args.max_shorter_arm_length,
        args.max_neg_log10_evalue,
    )
    counts = Counter(point.repeat_class for point in points)
    print(f"Input: {cfg.STEP3_MATCHES_PATH}")
    print(f"Maximum shorter-arm length: {args.max_shorter_arm_length} nt")
    print(f"Maximum −log10(E-value): {args.max_neg_log10_evalue:g}")
    print(f"Rows excluded by arm length: {len(all_points) - len(length_filtered):,}")
    print(f"Rows excluded by E-value strength: {len(length_filtered) - len(points):,}")
    print(f"Direct rows: {counts['direct']:,}")
    print(f"Palindromic/inverted rows: {counts['palindromic_inverted']:,}")

if __name__ == "__main__":
    main()
