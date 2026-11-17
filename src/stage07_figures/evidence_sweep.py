import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage07_figures import config_figures as cfg

REFERENCE_X = 3e-2

def _plot_overall(axis, rows, x_values) -> None:

    axis.plot(x_values, [row.pair_share for row in rows], marker="o", label="Pairs")
    axis.plot(x_values, [row.bases_share for row in rows], marker="o", label="Bases")
    axis.plot(x_values, [row.occupied_share for row in rows], marker="o", label="Occupied units")
    axis.plot(x_values, [row.chance_share for row in rows],
              marker="o", linestyle="--", label="Chance / surviving pairs")
    axis.set_title("Overall survival")
    axis.set_ylabel("Share of no-cutoff total")
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#D4D4D4", linewidth=0.6, alpha=0.5)

def _plot_strata(axis, rows_by_stratum, x_values, title: str) -> None:

    palette = ("#246A9A", "#D17B0F", "#43895D", "#8B5AA5", "#B84A4A")

    for index, rows in enumerate(rows_by_stratum):

        axis.plot(x_values, [row.pair_share for row in rows], marker="o", color=palette[index],
                  label=f"Q{index + 1}")

    axis.set_title(title)
    axis.set_ylabel("Pair survival")
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False, fontsize=8, title="Quintile")
    axis.grid(color="#D4D4D4", linewidth=0.6, alpha=0.5)

def _render_figure(output_path: Path, overall, entropy_rows, at_rows, dust_rows) -> None:

    x_values = [REFERENCE_X if row.label == "none" else float(row.label) for row in overall]
    labels = [row.label for row in overall]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    _plot_overall(axes[0, 0], overall, x_values)
    _plot_strata(axes[0, 1], entropy_rows, x_values, "Pair survival by entropy-rate quintile")
    _plot_strata(axes[1, 0], at_rows, x_values, "Pair survival by AT-content quintile")
    _plot_strata(axes[1, 1], dust_rows, x_values, "Pair survival by DUST quintile")

    for axis in axes.ravel():

        axis.set_xticks(x_values)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    for axis in axes[1, :]: axis.set_xlabel("Pair E-value cutoff")

    fig.suptitle(f"Step 5C evidence E-value threshold sweep ({vmatch_cfg.STEP5_PRIMARY_REPEAT_SET} set)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _atomic_save_figure(fig, output_path)

def _atomic_save_figure(fig, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:

        fig.savefig(temporary, format="png", dpi=cfg.DPI, bbox_inches="tight")
        os.replace(temporary, path)

    finally:

        temporary.unlink(missing_ok=True)
        plt.close(fig)

def draw():

    table = pd.read_csv(vmatch_cfg.STEP5_EVIDENCE_THRESHOLD_SWEEP_PATH, sep="\t")
    strata = {stratum: list(rows.itertuples()) for stratum, rows in table.groupby("stratum", sort=False)}
    quintiles = {name: [strata[f"{name}_q{quintile}"] for quintile in range(1, 6)] for name in ("entropy", "at", "dust")}
    path = cfg.EVIDENCE_SWEEP_DIR / "step5c_evidence_threshold_sweep.png"
    _render_figure(path, strata["overall"], quintiles["entropy"], quintiles["at"], quintiles["dust"])

    return [path]
