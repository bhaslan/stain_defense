import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.stage06_repeat_enrichment import background, lineages
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg
from src.stage07_figures.context_gc import measure

SIDE_NAMES = {"left": "upstream", "right": "downstream"}
GC_SIDE = "both"
ORDINARY = "#7F7F7F"
MULTIPLES = "#C0392B"
PARTNER_NAMES = {"same cluster, other genus": "same cluster,\nanother genus", "other cluster, same genus": "another cluster,\nsame genus"}
GAPS = background.GAP_LABELS[:-1]
STYLE = {"font.size": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5}

def _plain(axis):

    for spine in ("top", "right"): axis.spines[spine].set_visible(False)

def _save(fig, panel, dpi=cfg.DPI):

    name, _ = cfg.REPEAT_CHARACTER_PANELS[panel]
    path = cfg.REPEAT_CHARACTER_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path

def mirror_spectrum(axis, spectrum, side):

    largest = enrichment_cfg.UNIT_SPECTRUM_LARGEST
    axis.axvspan(10, 13, color="#EEEEEE", zorder=0, linewidth=0)
    for multiple in range(11, largest + 1, 11): axis.axvline(multiple + 0.5, color=MULTIPLES, linewidth=0.5, linestyle=":", zorder=0)

    for name, sign in zip(cfg.ENRICHMENT_CLASSES, (1, -1)):

        chosen = spectrum.loc[(spectrum["hth_class"] == name) & (spectrum["side"] == side)].sort_values("unit_size")
        edges = [*chosen["unit_size"], chosen["unit_size"].iloc[-1] + 1]
        axis.stairs(sign * 100 * chosen["focal_rate"], edges, color=cfg.HTH_CLASS_COLOURS[name], fill=True, alpha=0.9, baseline=0)
        axis.stairs(sign * 100 * chosen["background_rate"], edges, color="black", linewidth=0.6, linestyle="--", baseline=None)

    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    axis.set_ylim(-cfg.REPEAT_CHARACTER_MIRROR_FLOOR, 15)
    ticks = [-1, -0.1, 0, 0.1, 1, 10]
    axis.set_yticks(ticks, [f"{abs(tick):g}%" for tick in ticks])
    axis.set_ylabel(f"{SIDE_NAMES[side]}\n% of flanks")

def units_panel(spectrum):

    largest = enrichment_cfg.UNIT_SPECTRUM_LARGEST

    with plt.rc_context(STYLE):

        fig, axes = plt.subplots(2, 1, figsize=cfg.REPEAT_CHARACTER_PANELS["units"][1], sharex=True)
        for axis, side in zip(axes, ("left", "right")): mirror_spectrum(axis, spectrum, side)
        axes[1].set_xlim(10, largest + 1)
        axes[1].set_xticks(range(10, largest + 1, 10), [f"{size}" if size % 20 == 10 else "" for size in range(10, largest + 1, 10)])
        axes[1].set_xlabel("unit size (bp)")
        fig.suptitle(f"Tandem arrays within {enrichment_cfg.UNIT_SIZE_WINDOW} bp of the gene", fontsize=7, y=0.985)
        fig.legend(handles=[*(Patch(color=cfg.HTH_CLASS_COLOURS[name], label=f"{name} ({where})") for name, where in zip(cfg.ENRICHMENT_CLASSES, ("above zero", "below zero"))),
                            Line2D([], [], color="black", linewidth=0.6, linestyle="--", label="ordinary genes, same genomes"),
                            Line2D([], [], color=MULTIPLES, linewidth=0.6, linestyle=":", label="multiples of 11 bp")],
                   loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=5.5, columnspacing=1.2, handlelength=1.8)
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.1, top=0.83, hspace=0.1)

        return _save(fig, "units")

def _against_shuffles(axis, x, rows, colour):

    low, high = (np.array(values) for values in zip(*map(background._pair, rows["shuffled_ci"])))
    axis.bar(x, 100 * rows["observed"], 0.36, color=colour)
    axis.errorbar(x, 100 * rows["shuffled"], yerr=[100 * (rows["shuffled"] - low), 100 * (high - rows["shuffled"])], fmt="_", color="black",
                  markersize=7, elinewidth=0.7, capsize=0)

def all_or_none(axis, table):

    clusters = []

    for offset, (name, side) in zip((-0.19, 0.19), cfg.REPEAT_CHARACTER_SIDES.items()):

        rows = table.loc[(table["hth_class"] == name) & (table["side"] == side)].set_index("category").loc[list(lineages.CATEGORIES)]
        _against_shuffles(axis, np.arange(len(rows)) + offset, rows, cfg.HTH_CLASS_COLOURS[name])
        clusters.append(f"{name} {rows['clusters'].iloc[0]}")

    axis.set_xticks(range(len(lineages.CATEGORIES)), ["none", "some", "all"])
    axis.set_xlabel(f"how many of a cluster's flanks carry a repeat\n(clusters with {enrichment_cfg.LINEAGE_MIN_FLANKS} flanks or more: {', '.join(clusters)})")
    axis.set_ylabel("% of clusters")
    axis.set_title("All or none", fontsize=7, fontweight="bold")

def follows_the_protein(axis, table):

    for offset, (name, side) in zip((-0.19, 0.19), cfg.REPEAT_CHARACTER_SIDES.items()):

        rows = table.loc[(table["hth_class"] == name) & (table["side"] == side)].set_index("partner").loc[list(PARTNER_NAMES)]
        x = np.arange(len(rows)) + offset
        _against_shuffles(axis, x, rows, cfg.HTH_CLASS_COLOURS[name])

        for position, row in zip(x, rows.itertuples()):

            axis.text(position, 100 * max(row.observed, background._pair(row.shuffled_ci)[1]) + 2, f"{row.clusters}", ha="center", va="bottom", fontsize=5)

    axis.set_xticks(range(len(PARTNER_NAMES)), list(PARTNER_NAMES.values()))
    axis.set_ylim(0, 105)
    axis.set_ylabel("% of partner flanks with a repeat,\nwhen the flank has one")
    axis.text(0.99, 0.99, "numbers: clusters", transform=axis.transAxes, ha="right", va="top", fontsize=5)
    axis.set_title("The repeat follows the protein", fontsize=7, fontweight="bold")

def lineages_panel(tables):

    with plt.rc_context(STYLE):

        fig, (left, right) = plt.subplots(1, 2, figsize=cfg.REPEAT_CHARACTER_PANELS["lineages"][1])
        all_or_none(left, tables["all_or_none"])
        follows_the_protein(right, tables["follows"])
        for axis in (left, right): _plain(axis)
        fig.legend(handles=[*(Patch(color=cfg.HTH_CLASS_COLOURS[name], label=f"{name} {SIDE_NAMES[side]}") for name, side in cfg.REPEAT_CHARACTER_SIDES.items()),
                            Line2D([], [], color="black", marker="_", markersize=7, linewidth=0.7, label="shuffled within host genus (mean, 95% range)")],
                   loc="upper center", ncol=3, frameon=False, fontsize=6)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.82, wspace=0.3)

        return _save(fig, "lineages")

def gc_by_gap(axis, by_gap, overall, side):

    x = np.arange(len(GAPS))
    reference = by_gap.loc[by_gap["side"] == side].drop_duplicates("gap").set_index("gap").loc[list(GAPS)]
    axis.plot(x, 100 * reference["ordinary_offset"], color=ORDINARY, linewidth=0.9, linestyle="--", marker="o", markersize=2.5)
    axis.axhline(0, color="black", linewidth=0.5)
    axis.text(0.98, 0.98, "gap-adjusted, points:", transform=axis.transAxes, ha="right", va="top", fontsize=5.5, color="#555555")

    for line, (offset, name) in enumerate(zip((-0.12, 0.12), cfg.ENRICHMENT_CLASSES), start=1):

        rows = by_gap.loc[(by_gap["hth_class"] == name) & (by_gap["side"] == side)].set_index("gap").reindex(list(GAPS))
        low, high = (np.array(values) for values in zip(*map(background._pair, rows["offset_ci"])))
        colour = cfg.HTH_CLASS_COLOURS[name]
        axis.errorbar(x + offset, 100 * rows["offset"], yerr=[100 * (rows["offset"] - low), 100 * (high - rows["offset"])], color=colour, marker="o", markersize=3,
                      linewidth=1.0, elinewidth=0.7, capsize=0)
        adjusted = overall.loc[(overall["hth_class"] == name) & (overall["side"] == side) & (overall["scope"] == "all flanks") & (overall["adjustment"] == "genome and gap")].iloc[0]
        low, high = background._pair(adjusted["offset_ci"])
        axis.text(0.98, 0.98 - 0.075 * line, f"{name} {100 * adjusted['offset']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]", transform=axis.transAxes,
                  ha="right", va="top", fontsize=5.5, color=colour, fontweight="bold")

    axis.set_ylim(-7, 4.5)
    axis.set_xticks(x, [gap.replace("over ", ">") for gap in GAPS], rotation=35, ha="right", rotation_mode="anchor")
    axis.set_xlabel("gap to the next gene on the flank's side (bp)")
    _plain(axis)

def gc_panel(tables):

    with plt.rc_context(STYLE):

        fig, axis = plt.subplots(figsize=cfg.REPEAT_CHARACTER_PANELS["gc"][1])
        gc_by_gap(axis, tables["gaps"], tables["gc"], GC_SIDE)
        axis.set_ylabel("GC minus own genome's\nordinary flanks (points)")
        fig.legend(handles=[*(Line2D([], [], color=cfg.HTH_CLASS_COLOURS[name], marker="o", markersize=3, linewidth=1.0, label=name) for name in cfg.ENRICHMENT_CLASSES),
                            Line2D([], [], color=ORDINARY, marker="o", markersize=2.5, linewidth=0.9, linestyle="--", label="ordinary genes")],
                   loc="upper center", ncol=3, frameon=False, fontsize=6)
        fig.subplots_adjust(left=0.2, right=0.98, bottom=0.25, top=0.85)

        return _save(fig, "gc")

PART_NAMES = {"upstream": "upstream flank", "downstream": "downstream flank", "gc3": "gene GC3"}
GROUPS = ("ordinary", *cfg.ENRICHMENT_CLASSES)
GROUP_COLOURS = {"ordinary": "#BDBDBD", **cfg.HTH_CLASS_COLOURS}
GC_LIMITS = (-35, 30)

def context_gc_panel(offsets):

    kinds = {"ordinary": offsets["kind"] == "ordinary", **{name: (offsets["kind"] == "focal") & (offsets["hth_class"] == name) for name in cfg.ENRICHMENT_CLASSES}}

    with plt.rc_context({**STYLE, "font.size": 5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "xtick.major.size": 2}):

        fig = plt.figure(figsize=cfg.REPEAT_CHARACTER_PANELS["context_gc"][1])
        axis = fig.add_axes([0.08, 0.13, 0.84, 0.85])
        axis.axvline(0, color=ORDINARY, linewidth=0.4, linestyle="--", zorder=0)
        row = 0.0

        for part, label in PART_NAMES.items():

            axis.text(GC_LIMITS[0] + 3, row, label, fontsize=4.5, va="center", ha="left")
            row += 1.1

            for group in GROUPS:

                values = offsets.loc[kinds[group], part].to_numpy()
                body = axis.violinplot(values, [row], vert=False, widths=0.9, showextrema=False, points=200, bw_method="scott")["bodies"][0]
                body.set_facecolor(GROUP_COLOURS[group]); body.set_edgecolor("none"); body.set_alpha(0.7 if group == "ordinary" else 0.85)
                low, median, high = np.percentile(values, [25, 50, 75])
                axis.plot([low, high], [row, row], color="black", linewidth=0.8, solid_capstyle="butt")
                axis.plot(median, row, marker="|", color="white", markersize=2, markeredgewidth=0.6)
                row += 1.0

            row += 0.5

        axis.set_xlim(*GC_LIMITS)
        axis.set_xticks(range(-30, 31, 10), ["-30", "", "", "0", "", "", "30"])
        axis.set_ylim(row - 0.6, -1.0)
        axis.set_yticks([])
        axis.set_xlabel("GC offset (points)", fontsize=5, labelpad=1)

        return _save(fig, "context_gc", cfg.REPEAT_CHARACTER_C_ALL_DPI)

def draw():

    paths = {"spectrum": enrichment_cfg.UNIT_SPECTRUM_PATH, "all_or_none": enrichment_cfg.ALL_OR_NONE_PATH, "follows": enrichment_cfg.FOLLOWS_PATH,
             "gaps": enrichment_cfg.FLANK_GC_GAPS_PATH, "gc": enrichment_cfg.FLANK_GC_PATH}
    tables = {name: pd.read_csv(path, sep="\t") for name, path in paths.items()}

    name, _ = cfg.REPEAT_CHARACTER_PANELS["context_gc"]
    compared = cfg.REPEAT_CHARACTER_DIR / name.replace(".png", ".tsv")
    table = measure.contexts()
    measure.compare(table, compared)

    return [units_panel(tables["spectrum"]), lineages_panel(tables), gc_panel(tables), context_gc_panel(measure.offsets(table))]
