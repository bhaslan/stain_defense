import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

from src.stage06_repeat_enrichment import background
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg
from src.stage07_figures.enrichment import REPEAT_LABELS

GENE = 450
BACKGROUND_COLOUR = "#7F7F7F"

def extent(side, inner, outer):

    return (-outer, -inner) if side == "left" else (GENE + inner, GENE + outer)

def profile_panel(axis, rows, colour):

    for side, chosen in rows.groupby("side"):

        chosen = chosen.sort_values("inner", ascending=side == "right")
        spans = [extent(side, inner, outer) for inner, outer in zip(chosen["inner"], chosen["outer"])]
        edges = sorted({edge for span in spans for edge in span})
        ordered = chosen.iloc[sorted(range(len(spans)), key=lambda index: spans[index][0])]
        axis.stairs(100 * ordered["background_rate"], edges, baseline=None, color=BACKGROUND_COLOUR, linestyle="--", linewidth=0.9)
        axis.stairs(100 * ordered["focal_rate"], edges, baseline=None, color=colour, linewidth=1.4)

        for row in ordered.itertuples():

            middle = sum(extent(side, row.inner, row.outer)) / 2
            low, high = (100 * value for value in background._pair(row.focal_rate_ci))
            axis.plot([middle, middle], [low, high], color=colour, linewidth=0.8)
            if row.p_value < cfg.SIDE_PROFILE_LABEL_P: axis.annotate(f"{row.ratio:.1f}×", (middle, high), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)

    axis.axvspan(0, GENE, color="#F2F2F2", zorder=0, linewidth=0)
    axis.annotate("", xy=(GENE * 0.92, 0.5), xytext=(GENE * 0.08, 0.5), xycoords=("data", "axes fraction"),
                  arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", linewidth=2.5, mutation_scale=7))
    axis.text(GENE / 2, 0.56, "gene", transform=axis.get_xaxis_transform(), ha="center", va="bottom", color="#888888", fontsize=5.5)

def side_profile(table, path):

    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(len(cfg.ENRICHMENT_CLASSES), len(cfg.ENRICHMENT_REPEAT_SETS), figsize=cfg.SIDE_PROFILE_SIZE, sharex=True, sharey="col", squeeze=False)

    for (row, name), (column, repeat_set) in ((pair, other) for pair in enumerate(cfg.ENRICHMENT_CLASSES) for other in enumerate(cfg.ENRICHMENT_REPEAT_SETS)):

        axis = axes[row, column]
        profile_panel(axis, table.loc[(table["hth_class"] == name) & (table["repeat_set"] == repeat_set)], cfg.HTH_CLASS_COLOURS[name])
        axis.set_ylim(0, axis.get_ylim()[1] * 1.08)
        axis.tick_params(length=2.5)
        for spine in ("top", "right"): axis.spines[spine].set_visible(False)
        if row == 0: axis.set_title(REPEAT_LABELS[repeat_set], fontsize=7)
        if column == 0: axis.set_ylabel(f"{name}\n% of flanks with a repeat in the ring", fontsize=6.5)

    outers = [outer for _, outer in enrichment_cfg.SIDE_RINGS]
    axis.set_xticks([*(-outer for outer in outers), 0, GENE, *(GENE + outer for outer in outers)])
    axis.set_xticklabels([*map(str, outers), "", "", *map(str, outers)])
    fig.text(0.535, 0.02, "upstream: bp before the start codon          |          downstream: bp after the stop codon", ha="center", fontsize=6.5)
    fig.legend(handles=[plt.Line2D([], [], color="black", linewidth=1.4, label="StaN-like class (bar: 95% interval, samples and genera resampled)"),
                        plt.Line2D([], [], color=BACKGROUND_COLOUR, linestyle="--", linewidth=0.9, label="ordinary genes of the same genomes, same side")],
               loc="upper center", ncol=2, frameon=False, fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.88, hspace=0.25, wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI)
    plt.close(fig)

    return path

LOW, HIGH = -3, np.log2(20)
HEAT = LinearSegmentedColormap.from_list("depletion_enrichment", [(0, "#636363"), (-LOW / (HIGH - LOW) - 0.001, "#F2F2F2"),
                                                                  (-LOW / (HIGH - LOW), "#FFFFB2"), (0.55, "#FD8D3C"), (0.8, "#E31A1C"), (1, "#800026")])
HEAT_NORM = Normalize(LOW, HIGH)
STRIPE_INCHES = {"high": 0.058, "ratio": 0.183, "low": 0.058}
TRACK = sum(STRIPE_INCHES.values())
STRIPES = (("high", 1 - STRIPE_INCHES["high"] / TRACK, 1.0), ("ratio", STRIPE_INCHES["low"] / TRACK, 1 - STRIPE_INCHES["high"] / TRACK), ("low", 0.0, STRIPE_INCHES["low"] / TRACK))
HEAT_GENE = 0.8
HEAT_ROW_LABELS = {"tandem": "tandem", "direct": "direct", "inverted": "inverted"}
LEFT, BRACKET = 0.27, 0.14
GAP, CLASS_GAP, TOP, BOTTOM = 0.07, 0.14, 0.92, 0.45

def heat(ratio): return np.array(HEAT(HEAT_NORM(np.log2(max(ratio, 2.0 ** LOW)))))[:3]

def slot(side, inner):

    index = [ring[0] for ring in enrichment_cfg.SIDE_RINGS].index(inner)

    return (-index - 1, -index) if side == "left" else (HEAT_GENE + index, HEAT_GENE + index + 1)

def heat_track(axis, rows, hth_class):

    rings = len(enrichment_cfg.SIDE_RINGS)
    axis.set_xlim(-rings, HEAT_GENE + rings)
    axis.set_ylim(0, 1)
    axis.axvspan(0, HEAT_GENE, color="#F2F2F2", linewidth=0)
    head = HEAT_GENE * 0.25
    axis.add_patch(Polygon([(HEAT_GENE * 0.15, 0.44), (HEAT_GENE * 0.85 - head, 0.44), (HEAT_GENE * 0.85 - head, 0.36), (HEAT_GENE * 0.85, 0.5), (HEAT_GENE * 0.85 - head, 0.64),
                            (HEAT_GENE * 0.85 - head, 0.56), (HEAT_GENE * 0.15, 0.56)], closed=True, facecolor=cfg.HTH_CLASS_COLOURS[hth_class], edgecolor="none"))

    for row in rows.itertuples():

        x0, x1 = slot(row.side, row.inner)
        values = {"ratio": row.ratio, **dict(zip(("low", "high"), background._pair(row.ratio_ci)))}

        for field, y0, y1 in STRIPES:

            if np.isnan(values[field]): continue
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=heat(values[field]), edgecolor="white", linewidth=0.8))

        established = row.p_value < cfg.SIDE_PROFILE_LABEL_P
        axis.text((x0 + x1) / 2, 0.5, f"{row.ratio:.1f}×", ha="center", va="center", fontsize=6, color="black" if established else "#A0A0A0",
                  fontweight="bold" if established else "normal")

    axis.set_yticks([])
    axis.set_xticks([])

def heat_tracks(table, path):

    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.7})
    kinds = len(cfg.ENRICHMENT_REPEAT_SETS)
    count = kinds * len(cfg.ENRICHMENT_CLASSES)
    tall = TOP + count * TRACK + (count - len(cfg.ENRICHMENT_CLASSES)) * GAP + (len(cfg.ENRICHMENT_CLASSES) - 1) * CLASS_GAP + BOTTOM
    fig = plt.figure(figsize=(cfg.SIDE_HEAT_WIDTH, tall))

    for index, (name, repeat_set) in enumerate((name, repeat_set) for name in cfg.ENRICHMENT_CLASSES for repeat_set in cfg.ENRICHMENT_REPEAT_SETS):

        above = TOP + (index + 1) * TRACK + (index - index // kinds) * GAP + (index // kinds) * CLASS_GAP
        axis = fig.add_axes((LEFT, 1 - above / tall, 0.955 - LEFT, TRACK / tall))
        heat_track(axis, table.loc[(table["hth_class"] == name) & (table["repeat_set"] == repeat_set)], name)
        axis.text(-0.02, 0.5, HEAT_ROW_LABELS[repeat_set], transform=axis.transAxes, ha="right", va="center", fontsize=6)

        if index % kinds == 0: first = 1 - (above - TRACK) / tall
        if index % kinds != kinds - 1: continue
        last = 1 - above / tall
        fig.add_artist(plt.Line2D([BRACKET, BRACKET], [last, first], color=cfg.HTH_CLASS_COLOURS[name], linewidth=1.5))
        fig.text(BRACKET - 0.015, (first + last) / 2, name, ha="right", va="center", fontsize=6.5, fontweight="bold", color=cfg.HTH_CLASS_COLOURS[name])

    rings = len(enrichment_cfg.SIDE_RINGS)
    outers = [outer for _, outer in enrichment_cfg.SIDE_RINGS]
    axis.set_xticks([*(-index - 1 for index in range(rings)), *(HEAT_GENE + index + 1 for index in range(rings))])
    axis.set_xticklabels([*map(str, outers), *map(str, outers)], fontsize=5)
    axis.tick_params(axis="x", length=2, pad=1.5)
    axis.set_xlabel("bp before the start codon   |   bp after the stop codon\n(rings not to scale)", fontsize=5.5)

    bar = fig.add_axes((LEFT + 0.12, 1 - 0.27 / tall, 0.955 - LEFT - 0.24, 0.07 / tall))
    bar.imshow(np.linspace(LOW, HIGH, 256)[None, :], cmap=HEAT, norm=HEAT_NORM, aspect="auto", extent=(LOW, HIGH, 0, 1))
    bar.set_xticks(np.log2([1 / 8, 1 / 2, 1, 4, 16]))
    bar.set_xticklabels(["1/8×", "1/2×", "1×", "4×", "16×"], fontsize=5)
    bar.set_yticks([])
    bar.tick_params(length=1.5, pad=1)
    bar.set_title("ratio to ordinary genes of the same genomes", fontsize=5.5, pad=2)

    key = fig.add_axes((LEFT, 1 - (0.42 + TRACK * 1.3) / tall, 0.955 - LEFT, TRACK * 1.3 / tall))
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")

    for (field, y0, y1), label in zip(STRIPES, ("95% interval, top", "estimate", "95% interval, bottom")):

        key.add_patch(Rectangle((0, y0), 0.07, y1 - y0, facecolor=heat(8 if field == "ratio" else 3), edgecolor="white", linewidth=0.8))
        key.text(0.09, (y0 + y1) / 2 + {"high": 0.08, "low": -0.08}.get(field, 0), label, va="center", fontsize=5)

    key.text(0.62, 0.5, "faded number:\np ≥ 0.05", va="center", fontsize=5, color="#A0A0A0")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI)
    plt.close(fig)

    return path

def draw():

    tables = {panel: pd.read_csv(background.panel_path(enrichment_cfg.BACKGROUND_SIDES_PATH, panel), sep="\t") for panel in {*cfg.SIDE_PROFILE_PANELS, *cfg.SIDE_HEAT_PANELS}}

    return ([side_profile(tables[panel], cfg.SIDE_PROFILE_DIR / name) for panel, name in cfg.SIDE_PROFILE_PANELS.items()]
            + [heat_tracks(tables[panel], cfg.SIDE_PROFILE_DIR / name) for panel, name in cfg.SIDE_HEAT_PANELS.items()])
