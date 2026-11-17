import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.stage05_C_vmatch import background as panel
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage06_repeat_enrichment import background
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg
from src.stage07_figures.enrichment import REPEAT_LABELS

REPEAT_SET = "all"
KINDS = {"all": ("All repeats", "-", 1.2), **{name: (REPEAT_LABELS[name], style, width) for name, style, width in (("tandem", "--", 1.2), ("direct", ":", 1.0))}}
MIN_SAMPLES = 10
POOLED = "any HTH"
GROUPS = {**{name: (name,) for name in cfg.ENRICHMENT_CLASSES}, POOLED: cfg.ENRICHMENT_CLASSES}
DRAWN = cfg.ENRICHMENT_CLASSES
COLOURS = {**cfg.HTH_CLASS_COLOURS, POOLED: "black"}

def curves(flanks, units, repeat_set, window):

    focal, reference = background.matched(flanks, units, repeat_set, window)
    bases = f"repeat_bases_{window}"
    thresholds = np.arange(1, window + 1)
    groups = reference.groupby(["genome", "side"])[bases].apply(np.sort)
    shares = np.vstack([1 - np.searchsorted(values, thresholds) / len(values) for values in groups])
    rows = {key: index for index, key in enumerate(groups.index)}
    frames = []

    for name, members in GROUPS.items():

        chosen = focal.loc[focal["hth_class"].isin(members)]
        per_sample = chosen.groupby("group_accession")["group_accession"].transform("size")
        weight = (1 / per_sample / chosen["group_accession"].nunique()).to_numpy()
        above = chosen[bases].to_numpy()[:, None] >= thresholds
        matched_shares = shares[[rows[key] for key in zip(chosen["genome"], chosen["side"])]]
        reaching = pd.DataFrame(above).groupby(chosen["group_accession"].to_numpy()).any().sum().to_numpy()
        frames.append(pd.DataFrame({"repeat_set": repeat_set, "window": window, "hth_class": name, "repeat_bases_at_least": thresholds,
                                    "focal_share": weight @ above, "background_share": weight @ matched_shares, "samples_reaching": reaching}))

    table = pd.concat(frames, ignore_index=True)
    table["ratio"] = table["focal_share"] / table["background_share"]

    return table

STYLE = {"font.size": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
LEGEND = [*(Line2D([], [], color=COLOURS[name], linewidth=1.2, label=" and ".join(members) + " pooled" * (len(members) > 1))
            for name, members in GROUPS.items() if name in DRAWN),
          Line2D([], [], color="#666666", linewidth=0.8, linestyle="--", label="Background genes, same genomes")]

def window_panels(shares, ratios, table, window, shown=REPEAT_SET, kinds=(REPEAT_SET,)):

    rows = table.loc[table["window"] == window]

    for name in DRAWN:

        colour = COLOURS[name]
        chosen = rows.loc[(rows["hth_class"] == name) & (rows["repeat_set"] == shown)]
        x = chosen["repeat_bases_at_least"]
        shares.plot(x, chosen["focal_share"].where(chosen["focal_share"] > 0), color=colour, linewidth=1.2)
        shares.plot(x, chosen["background_share"].where(chosen["background_share"] > 0), color=colour, linewidth=0.8, linestyle="--")

        for kind in kinds:

            chosen = rows.loc[(rows["hth_class"] == name) & (rows["repeat_set"] == kind)]
            _, style, width = KINDS[kind] if len(kinds) > 1 else KINDS[REPEAT_SET]
            ratios.plot(chosen["repeat_bases_at_least"], chosen["ratio"].where(chosen["samples_reaching"] >= MIN_SAMPLES), color=colour,
                        linewidth=width, linestyle=style)

    for axis in (shares, ratios):

        axis.set_yscale("log")
        axis.set_xlim(0, window)

    ratios.axhline(1, color="black", linewidth=0.6, linestyle="--")
    shares.set_ylim(1e-3, 0.5)
    shares.set_yticks([0.001, 0.01, 0.1, 0.2, 0.3], ["0.1%", "1%", "10%", "20%", "30%"])
    ratios.set_yticks([1, 10, 20, 30], ["1×", "10×", "20×", "30×"])
    ratios.yaxis.set_minor_formatter("")

def plot(table, path):

    windows = sorted(table["window"].unique())

    with plt.rc_context(STYLE):

        fig, axes = plt.subplots(2, len(windows), figsize=cfg.REPEAT_SIZE_SIZE, sharey="row", squeeze=False)

        for column, window in enumerate(windows):

            window_panels(axes[0, column], axes[1, column], table, window)
            axes[0, column].set_title(f"Within {window} bp of the gene", fontsize=6.5)
            axes[1, column].set_xlabel("Repeat DNA in the window, at least (bp)")

        axes[0, 0].set_ylabel("Fraction of flanks")
        axes[1, 0].set_ylabel("Times more often than\nbackground genes")
        axes[0, -1].legend(handles=LEGEND, fontsize=4, loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=cfg.DPI)

def plot_paper(table, path, shown, kinds, window=cfg.REPEAT_SIZE_PAPER_WINDOW):

    with plt.rc_context(STYLE):

        fig, (shares, ratios) = plt.subplots(2, 1, figsize=cfg.REPEAT_SIZE_PAPER_SIZE, sharex=True)
        window_panels(shares, ratios, table, window, shown, kinds)
        shares.set_title(KINDS[shown][0], fontsize=6.5)
        shares.set_ylabel("Fraction of flanks")
        ratios.set_ylabel("Times more often\nthan background genes")
        ratios.set_xlabel(f"Repeat DNA within {window} bp of the gene,\nat least (bp)")
        shares.legend(handles=LEGEND, fontsize=4, loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.3)

        if len(kinds) > 1:

            handles = [Line2D([], [], color="#666666", linewidth=KINDS[kind][2], linestyle=KINDS[kind][1], label=KINDS[kind][0]) for kind in kinds]
            ratios.legend(handles=handles, fontsize=4, loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=cfg.DPI)

def draw():

    flanks = background.read_flanks()
    units = pd.read_csv(panel.panel_paths(enrichment_cfg.BACKGROUND_PANEL)["units_path"], sep="\t")
    paper = {kind for shown, kinds in cfg.REPEAT_SIZE_PAPER_FIGURES.values() for kind in (shown, *kinds)} - {REPEAT_SET}
    table = pd.concat([*(curves(flanks, units, REPEAT_SET, window) for window in vmatch_cfg.STEP5_UNIT_WINDOWS),
                       *(curves(flanks, units, kind, cfg.REPEAT_SIZE_PAPER_WINDOW) for kind in sorted(paper))], ignore_index=True)
    cfg.REPEAT_SIZE_DIR.mkdir(parents=True, exist_ok=True)
    plot(table.loc[table["repeat_set"] == REPEAT_SET], cfg.REPEAT_SIZE_DIR / "repeat_size.png")

    for name, (shown, kinds) in cfg.REPEAT_SIZE_PAPER_FIGURES.items(): plot_paper(table, cfg.REPEAT_SIZE_DIR / name, shown, kinds)

    table.to_csv(cfg.REPEAT_SIZE_DIR / "repeat_size.csv", index=False)

    return [cfg.REPEAT_SIZE_DIR / name for name in ("repeat_size.png", *cfg.REPEAT_SIZE_PAPER_FIGURES, "repeat_size.csv")]
