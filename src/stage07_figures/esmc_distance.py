import h5py
import matplotlib
import numpy as np
import pandas as pd
from Bio import Phylo

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src import config_main as main_cfg
from src.protein_tree import config_tree as tree_cfg
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage03_concordance import config_concordance as concordance_cfg
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg
from src.stage07_figures.tree import prepare

ANCHORS = {anchor["protein_acc"]: anchor for anchor in main_cfg.the_nine.values()}
REFERENCE = next(accession for accession, anchor in ANCHORS.items() if anchor["Name"] == "StaN")
ACTIVITY = {1: ("green", "tRNase activity"), 0: ("red", "no tRNase activity")}

DNA_BINDING = {1: (True, "binds DNA"), 2: (False, "DNA binding not determined")}

ANCHOR_LEGEND = [*(Line2D([], [], marker="o", linestyle="", markersize=7, color=colour, label=label) for colour, label in ACTIVITY.values()),
                 *(Line2D([], [], marker="o", linestyle="", markersize=7, markeredgecolor="black", markeredgewidth=1.4,
                          markerfacecolor="black" if filled else "white", label=label) for filled, label in DNA_BINDING.values())]

MM = 1 / 25.4
FULL_SIZE = (64 * MM, 90 * MM)
TIP_SIZE = 1
REPEATS = LinearSegmentedColormap.from_list("repeats", matplotlib.colormaps["inferno_r"](np.linspace(0.15, 0.9, 256)))
REPEAT_SET = "all"
NO_REPEATS = "#E2E2E2"
SLICE, SLICE_MIN_TIPS = 0.05, 20
SLICE_COLOUR = "#666666"
PANEL_RC = {"font.size": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
PANEL_MARGINS = {"left": 0.16, "right": 0.92, "bottom": 0.12, "top": 0.98}
LABEL_PAD, LABEL_GAP, LABEL_FONT = 0.1, 0.022, 4.0

def embeddings():

    with h5py.File(concordance_cfg.MEANPOOLED_L2_PATH) as handle:

        ids = handle["ids"].asstr()[:]
        vectors = handle["embeddings"][:]

    return dict(zip(ids, vectors))

def distances():

    vectors = embeddings()
    frames = []

    for name, path in tree_cfg.PRUNED_TREE_PATHS.items():

        tree = Phylo.read(path, "newick")
        tips = sorted(tip.name for tip in tree.get_terminals() if tip.name != REFERENCE)
        frames.append(pd.DataFrame({"tree": name, "protein_accession": tips,
                                    "esmc_distance": [np.linalg.norm(vectors[tip] - vectors[REFERENCE]) for tip in tips],
                                    "patristic_distance": [tree.distance(REFERENCE, tip) for tip in tips]}))

    frame = pd.concat(frames, ignore_index=True)
    frame["anchor"] = frame["protein_accession"].map(lambda accession: ANCHORS.get(accession, {}).get("Name"))

    return frame

def mark_anchors(ax, tips, size=36, font=8, width=1.4):

    for row in tips.dropna(subset=["anchor"]).itertuples():

        anchor = ANCHORS[row.protein_accession]
        colour, filled = ACTIVITY[anchor["tRNAse_activity"]][0], DNA_BINDING[anchor["DNA-binding_activity"]][0]
        point = (row.patristic_distance, row.esmc_distance)
        ax.scatter(*point, s=size, facecolors=colour if filled else "white", edgecolors=colour, linewidths=width, zorder=3)
        ax.annotate(row.anchor, point, xytext=(0, size ** 0.5 * 0.6), textcoords="offset points", ha="center", fontsize=font, color=colour, zorder=4)

def ring_color(anchor):

    return "black" if anchor["tRNAse_activity"] == 1 else ACTIVITY[0][0]

def anchor_label(row):

    return short(row.anchor) + ("\u2605" if DNA_BINDING[ANCHORS[row.protein_accession]["DNA-binding_activity"]][0] else "")

def ring_anchors(ax, tips):

    for row in tips.dropna(subset=["anchor"]).itertuples():

        ax.scatter(row.patristic_distance, row.esmc_distance, s=10, facecolors="none",
                   edgecolors=ring_color(ANCHORS[row.protein_accession]), linewidths=0.6, zorder=3)

def name_beside(ax, tips):

    for row in tips.dropna(subset=["anchor"]).itertuples():

        ax.annotate(anchor_label(row), (row.patristic_distance, row.esmc_distance), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=4.5, color="black", zorder=4)

def name_in_column(ax, tips):

    anchors = tips.dropna(subset=["anchor"]).sort_values("esmc_distance")
    heights = []

    for height in anchors["esmc_distance"]: heights.append(max(height, heights[-1] + LABEL_GAP) if heights else height)

    for row, height in zip(anchors.itertuples(), heights):

        colour = ring_color(ANCHORS[row.protein_accession])
        ax.annotate(anchor_label(row), (row.patristic_distance, row.esmc_distance), xytext=(ax.get_xlim()[1] - LABEL_PAD, height), textcoords="data",
                    ha="right", va="center", fontsize=LABEL_FONT, color=colour, zorder=4,
                    arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.5, "linestyle": ":", "relpos": (0, 0.5), "shrinkA": 1, "shrinkB": 2.2})

def short(anchor): return anchor.removesuffix("N")

def plot_core(tips, path):

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    ax.scatter(tips["patristic_distance"], tips["esmc_distance"], s=6, color="0.6", linewidths=0)
    mark_anchors(ax, tips)
    ax.set_title(f"core tree, frozen ({len(tips):,} tips besides StaN)")
    ax.set_xlabel("Patristic distance to StaN (substitutions per site)")
    ax.set_ylabel("ESM-C distance to StaN")
    tip_legend = Line2D([], [], marker="o", linestyle="", markersize=5, color="0.6", label="tree tips")
    ax.legend(handles=[tip_legend, *ANCHOR_LEGEND], fontsize=8)
    fig.tight_layout()
    fig.savefig(path)

def background_rate():

    table = pd.read_csv(enrichment_cfg.BACKGROUND_MODEL_TABLE_PATH, sep="\t")
    row = table.loc[(table["panel"] == "full") & (table["method"] == "counting") & (table["hth_class"] == "any HTH")
                    & (table["repeat_set"] == REPEAT_SET) & (table["window"] == cfg.ENRICHMENT_WINDOW)]

    return row["background_rate"].item()

def with_repeats(tips):

    return tips.assign(repeats=prepare.sample_repeats(REPEAT_SET).reindex(tips["protein_accession"]).to_numpy())

def with_hth_class(tips):

    classes = pd.read_csv(vmatch_cfg.STEP6_SAMPLE_SUMMARY_PATH, sep="\t").set_index("protein_accession")["hth_class"]

    return tips.assign(hth_class=classes.reindex(tips["protein_accession"]).to_numpy())

def slice_rates(tips):

    slices = tips.dropna(subset=["repeats"]).groupby((tips["esmc_distance"] // SLICE) * SLICE)["repeats"]

    return slices.mean()[slices.size() >= SLICE_MIN_TIPS]

def new_panel():

    return plt.subplots(1, 2, figsize=FULL_SIZE, dpi=2 * cfg.DPI, sharey=True, gridspec_kw={"width_ratios": (4, 1), "wspace": 0.06})

def label_axes(ax, tips):

    ax.set_xlim(right=tips["patristic_distance"].max() + 0.9)
    ax.set_xlabel(f"Patristic distance to {short('StaN')}\n(substitutions per site)")
    ax.set_ylabel(f"ESM-C distance to {short('StaN')}")

def key_anchors(ax):

    keys = [Line2D([], [], marker="o", linestyle="", markersize=3, markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.6, label="tRNase activity"),
            Line2D([], [], marker="o", linestyle="", markersize=3, markerfacecolor="none", markeredgecolor=ACTIVITY[0][0], markeredgewidth=0.6, label="no tRNase activity"),
            Line2D([], [], marker="$\u2605$", linestyle="", markersize=4, color="black", label="binds DNA")]
    ax.add_artist(ax.legend(handles=keys, fontsize=4.5, frameon=False, loc="lower right", handletextpad=0.2))

def draw_strip(side, rates):

    side.barh(rates.index + SLICE / 2, rates, height=SLICE, color=SLICE_COLOUR, edgecolor="white", linewidth=0.3)
    side.axvline(background_rate(), color="black", linestyle="--", linewidth=0.6)
    side.set_xlim(0, max(0.6, rates.max() * 1.05))
    side.set_xticks([0, 0.25, 0.5], ["0", ".25", ".5"])
    side.yaxis.set_label_position("right")
    side.set_ylabel(f"Average repeat share of the tips, per {SLICE:g} slice", rotation=270, labelpad=6, fontsize=5)
    side.tick_params(axis="y", left=False)

def plot_full(tips, path):

    tips = with_repeats(tips)
    zero = tips[tips["repeats"] == 0]
    some = tips[tips["repeats"] > 0].sort_values("repeats")

    with plt.rc_context(PANEL_RC):

        fig, (ax, side) = new_panel()
        ax.scatter(zero["patristic_distance"], zero["esmc_distance"], s=TIP_SIZE, color=NO_REPEATS, linewidths=0)
        ax.scatter(some["patristic_distance"], some["esmc_distance"], s=TIP_SIZE, c=some["repeats"], cmap=REPEATS, vmin=0, vmax=1, linewidths=0)
        label_axes(ax, tips)
        ring_anchors(ax, tips)
        name_beside(ax, tips)
        key_anchors(ax)
        bar = ax.inset_axes((0.03, 0.955, 0.32, 0.018))
        fig.colorbar(plt.cm.ScalarMappable(cmap=REPEATS, norm=plt.Normalize(0, 1)), cax=bar, orientation="horizontal")
        bar.tick_params(labelsize=4, length=1.5, width=0.4, pad=1)
        bar.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        bar.set_title(f"Fraction of flanks with a repeat within {cfg.ENRICHMENT_WINDOW} bp", fontsize=4, pad=1.5, loc="left")
        handles = [Line2D([], [], marker="o", linestyle="", markersize=2, color=NO_REPEATS, label="none")]
        ax.add_artist(ax.legend(handles=handles, fontsize=4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.925), ncols=2,
                  handletextpad=0.1, columnspacing=0.8, borderaxespad=0.3))
        draw_strip(side, slice_rates(tips))
        fig.subplots_adjust(**PANEL_MARGINS)
        fig.savefig(path)

def plot_full_hth(tips, path):

    tips = with_repeats(with_hth_class(tips))

    with plt.rc_context(PANEL_RC):

        fig, (ax, side) = new_panel()

        for hth_class, colour in cfg.HTH_CLASS_COLOURS.items():

            members = tips[tips["hth_class"] == hth_class]
            ax.scatter(members["patristic_distance"], members["esmc_distance"], s=TIP_SIZE, color=colour, linewidths=0)

        label_axes(ax, tips)
        ring_anchors(ax, tips)
        name_in_column(ax, tips)
        key_anchors(ax)
        classes = [Line2D([], [], marker="o", linestyle="", markersize=2.5, color=colour, label=hth_class.replace("no_HTH", "no HTH"))
                   for hth_class, colour in cfg.HTH_CLASS_COLOURS.items()]
        ax.legend(handles=classes, fontsize=4.5, frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.3)
        draw_strip(side, slice_rates(tips))
        fig.subplots_adjust(**PANEL_MARGINS)
        fig.savefig(path)

def draw():

    frame = distances()
    cfg.ESMC_DISTANCE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.ESMC_DISTANCE_DIR / "esmc_phylo_distance.csv", index=False)
    anchors = frame.dropna(subset=["anchor"]).sort_values(["tree", "patristic_distance"])
    anchors[["anchor", *frame.columns.drop("anchor")]].to_csv(cfg.ESMC_DISTANCE_DIR / "esmc_phylo_distance_anchors.csv", index=False)
    paths = {"core": cfg.ESMC_DISTANCE_DIR / "esmc_phylo_distance_core.png", "full": cfg.ESMC_DISTANCE_DIR / "MAIN_esmc_phylo_distance_full.png",
             "full_hth": cfg.ESMC_DISTANCE_DIR / "MAIN_esmc_phylo_distance_full_v2.png"}
    plot_core(frame[frame["tree"] == "core"], paths["core"])
    plot_full(frame[frame["tree"] == "full"], paths["full"])
    plot_full_hth(frame[frame["tree"] == "full"], paths["full_hth"])

    return list(paths.values())
