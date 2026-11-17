import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.stage06_repeat_enrichment import background
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg

REPEAT_LABELS = {"tandem": "Tandem direct", "direct": "Spaced direct", "inverted": "Spaced inverted"}
GENOME_SETS = {"full": ("RefSeq", 1.0, None), "genbank": ("GenBank-only", 1.0, "////")}

def enrichment_panel(table, genomes, panels, path):

    rows = table.loc[(table["window"] == cfg.ENRICHMENT_WINDOW) & table["repeat_set"].isin(cfg.ENRICHMENT_REPEAT_SETS)
                     & table["hth_class"].isin(cfg.ENRICHMENT_CLASSES) & table["panel"].isin(panels)]
    if rows.empty: raise RuntimeError(f"Nothing to plot at {cfg.ENRICHMENT_WINDOW} bp.")

    fig, axis = plt.subplots(figsize=cfg.ENRICHMENT_SIZE)
    slots = [(name, panel) for name in cfg.ENRICHMENT_CLASSES for panel in panels]
    width = 0.76 / len(slots)
    offsets = {slot: index - (len(slots) - 1) / 2 for index, slot in enumerate(slots)}

    for group, repeat_set in enumerate(cfg.ENRICHMENT_REPEAT_SETS):

        for (name, panel), offset in offsets.items():

            chosen = rows.loc[(rows["repeat_set"] == repeat_set) & (rows["hth_class"] == name) & (rows["panel"] == panel)].set_index("method")
            _, alpha, hatch = GENOME_SETS[panel]
            x = group + offset * width
            bayes, glmm = chosen.loc["bayes"], chosen.loc["glmmTMB"]
            axis.bar(x, bayes["ratio"] - 1, bottom=1, width=width * 0.92, color=cfg.HTH_CLASS_COLOURS[name], alpha=alpha, hatch=hatch,
                     edgecolor="white", linewidth=0)
            axis.errorbar(x, bayes["ratio"], yerr=[[bayes["ratio"] - bayes["ratio_low"]], [bayes["ratio_high"] - bayes["ratio"]]],
                          fmt="none", ecolor="#333333", elinewidth=0.8, capsize=1.8, zorder=3)
            axis.plot(x + 0.3 * width, glmm["ratio"], marker="^", markersize=3.2, color="black", linestyle="none", zorder=4)

            if panel != "full": continue
            above = bayes["ratio"] >= 1
            axis.annotate(f"{bayes['ratio']:.1f}×", (x, bayes["ratio_high"] if above else bayes["ratio_low"]),
                          xytext=(0, 2 if above else -2), textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=6.5)

    axis.axhline(1, color="#666666", linestyle="--", linewidth=0.7, zorder=1)
    axis.set_yscale("log", base=2)
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    bayes = rows.loc[rows["method"] == "bayes"]
    low, high = bayes["ratio_low"].min(), bayes["ratio_high"].max()
    axis.set_ylim(low / 1.3 if low / 1.3 >= 0.25 else 0.125, min(max(ticks) * 1.5, high * 1.6))
    axis.set_yticks([tick for tick in ticks if axis.get_ylim()[0] <= tick <= axis.get_ylim()[1]])
    axis.set_yticklabels([f"{tick:g}×" for tick in axis.get_yticks()], fontsize=7)
    axis.set_xticks(range(len(cfg.ENRICHMENT_REPEAT_SETS)))
    axis.set_xticklabels([REPEAT_LABELS[name] for name in cfg.ENRICHMENT_REPEAT_SETS], fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_ylabel(f"Repeat within {cfg.ENRICHMENT_WINDOW} bp,\nratio to the genome's background genes", fontsize=7.5)
    axis.tick_params(axis="both", length=2.5, width=0.6)
    for spine in axis.spines.values(): spine.set_linewidth(0.6)

    handles = [Patch(color=cfg.HTH_CLASS_COLOURS[name], label=name) for name in cfg.ENRICHMENT_CLASSES]
    handles += [Line2D([], [], marker="^", linestyle="none", color="black", markersize=4, label="glmmTMB")]
    if len(panels) > 1:

        handles.append(Line2D([], [], linestyle="none", label=f"Solid: RefSeq ({genomes['full']:,} genomes)\nHatched: GenBank-only ({genomes['genbank']:,})"))
        axis.legend(handles=handles, fontsize=5, frameon=False, loc="lower left", handlelength=1.0)

    else: axis.legend(handles=handles, fontsize=6, frameon=False, loc="upper right", handlelength=1.2)

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.19, top=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI)
    plt.close(fig)

    return path

def _family_rows(families, classes, repeat_sets):

    families = families.loc[(families["window"] == cfg.ENRICHMENT_WINDOW) & families["repeat_set"].isin(repeat_sets)
                            & families["hth_class"].isin(cfg.ENRICHMENT_CLASSES)]
    classes = classes.loc[(classes["panel"] == "full") & (classes["window"] == cfg.ENRICHMENT_WINDOW) & classes["repeat_set"].isin(repeat_sets)
                          & classes["hth_class"].isin(cfg.ENRICHMENT_CLASSES)]
    if families.empty: raise RuntimeError(f"No family to plot at {cfg.ENRICHMENT_WINDOW} bp.")

    samples = families.loc[families["method"] == "counting"].groupby(["family", "hth_class"])["samples"].max().unstack()
    order = samples.sum(axis=1).sort_values(ascending=False).index
    labels = {"class": "All proteins of the class", **{family: f"{family[2:].removesuffix(' domain')}, no family" if family.startswith("T:") else family
                                                       for family in order}}
    labels[cfg.STAN_HTH_FAMILY] += " (StaN)"

    return pd.concat([classes.assign(family="class"), families]), samples.loc[order], labels

def _counts(samples, family):

    return " | ".join("–" if pd.isna(samples.loc[family].get(name)) else f"{int(samples.loc[family, name])}" for name in cfg.ENRICHMENT_CLASSES)

def family_column(families, classes, path):

    rows, samples, labels = _family_rows(families, classes, cfg.FAMILY_COLUMN_REPEAT_SETS)
    names = ["class", *samples.index[samples.notna().all(axis=1)]]
    rows = rows.loc[rows["family"].isin(names) & (rows["method"] == "bayes")]
    ticks = [0.0625, 0.25, 1, 4, 16, 64]
    limits = (max(ticks[0], rows["ratio_low"].min() / 1.25), min(ticks[-1], rows["ratio_high"].max() * 1.25))
    label_y, class_y = -0.30, {name: offset for name, offset in zip(cfg.ENRICHMENT_CLASSES, (-0.05, 0.20))}
    blank = {"facecolor": "white", "edgecolor": "none", "pad": 0.4}

    fig, axes = plt.subplots(len(cfg.FAMILY_COLUMN_REPEAT_SETS), 1, figsize=cfg.FAMILY_COLUMN_SIZE, sharex=True)

    for axis, repeat_set in zip(axes, cfg.FAMILY_COLUMN_REPEAT_SETS):

        for (family, name), chosen in rows.loc[rows["repeat_set"] == repeat_set].groupby(["family", "hth_class"]):

            point = chosen.iloc[0]
            y = names.index(family) + class_y[name]
            axis.plot([point["ratio_low"], point["ratio_high"]], [y, y], color=cfg.HTH_CLASS_COLOURS[name], linewidth=1.0, solid_capstyle="butt", zorder=2)
            axis.plot(point["ratio"], y, marker="o", markersize=2.8, color=cfg.HTH_CLASS_COLOURS[name], linestyle="none", zorder=3)

        for index, family in enumerate(names):

            axis.text(0.01, index + label_y, labels[family], transform=axis.get_yaxis_transform(), fontsize=5.5, va="center", ha="left", bbox=blank,
                      fontweight="bold" if family == cfg.STAN_HTH_FAMILY else "normal")
            if index > 1: axis.axhline(index - 0.55, color="#e3e3e3", linewidth=0.4, zorder=0)

        axis.axvline(1, color="#666666", linestyle="--", linewidth=0.6, zorder=1)
        axis.axhline(0.45, color="#bbbbbb", linewidth=0.5, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_xlim(limits)
        axis.set_ylim(len(names) - 0.6, -0.5)
        axis.set_yticks([])
        axis.set_title(REPEAT_LABELS[repeat_set], fontsize=6.5, pad=3)
        axis.tick_params(axis="x", length=2, width=0.5, labelsize=5.5, pad=1.5)
        for side in ("top", "right", "left"): axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.5)

    axes[-1].set_xticks([tick for tick in ticks if limits[0] <= tick <= limits[1]])
    axes[-1].set_xticklabels([f"{tick:g}×" for tick in axes[-1].get_xticks()])
    axes[-1].set_xlabel(f"Repeat within {cfg.ENRICHMENT_WINDOW} bp, ratio to\nthe genome's background genes", fontsize=6, labelpad=2)
    handles = [Line2D([], [], marker="o", markersize=2.8, linewidth=1.0, color=cfg.HTH_CLASS_COLOURS[name], label=name) for name in cfg.ENRICHMENT_CLASSES]
    fig.legend(handles=handles, fontsize=5.5, frameon=False, loc="lower center", ncol=len(handles), handlelength=1.4, columnspacing=1.0)

    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.095, top=0.975, hspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.FAMILY_COLUMN_DPI)
    plt.close(fig)

    return path

def family_panel(families, classes, path):

    rows, samples, labels = _family_rows(families, classes, cfg.ENRICHMENT_REPEAT_SETS)
    names = ["class", *samples.index]
    labels = [labels["class"], *(f"{labels[family]}  {_counts(samples, family)}" for family in samples.index)]

    fig, axes = plt.subplots(1, len(cfg.ENRICHMENT_REPEAT_SETS), figsize=cfg.FAMILY_SUPPLEMENT_SIZE, sharey=True)
    shift = {name: (index - (len(cfg.ENRICHMENT_CLASSES) - 1) / 2) * 0.32 for index, name in enumerate(cfg.ENRICHMENT_CLASSES)}
    bayes = rows.loc[rows["method"] == "bayes"]
    ticks = [0.0625, 0.25, 1, 4, 16, 64]
    limits = (max(ticks[0], bayes["ratio_low"].min() / 1.3), min(ticks[-1], bayes["ratio_high"].max() * 1.3))

    for axis, repeat_set in zip(axes, cfg.ENRICHMENT_REPEAT_SETS):

        for (family, name), chosen in rows.loc[rows["repeat_set"] == repeat_set].groupby(["family", "hth_class"]):

            chosen = chosen.set_index("method")
            y = names.index(family) + shift[name]
            point, glmm = chosen.loc["bayes"], chosen.loc["glmmTMB"]
            axis.plot([point["ratio_low"], point["ratio_high"]], [y, y], color=cfg.HTH_CLASS_COLOURS[name], linewidth=1.1, solid_capstyle="butt", zorder=2)
            axis.plot(point["ratio"], y, marker="o", markersize=4, color=cfg.HTH_CLASS_COLOURS[name], linestyle="none", zorder=3)
            axis.plot(glmm["ratio"], y, marker="^", markersize=2.4, color="black", linestyle="none", zorder=4)

        axis.axvline(1, color="#666666", linestyle="--", linewidth=0.7, zorder=1)
        axis.axhline(0.5, color="#bbbbbb", linewidth=0.6, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_xlim(limits)
        axis.set_xticks([tick for tick in ticks if limits[0] <= tick <= limits[1]])
        axis.set_xticklabels([f"{tick:g}×" for tick in axis.get_xticks()], fontsize=6.5)
        axis.set_title(REPEAT_LABELS[repeat_set], fontsize=7.5)
        axis.tick_params(axis="both", length=2.5, width=0.6)
        for side in ("top", "right"): axis.spines[side].set_visible(False)
        for side in ("left", "bottom"): axis.spines[side].set_linewidth(0.6)

    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(labels, fontsize=6.5)
    axes[0].get_yticklabels()[names.index(cfg.STAN_HTH_FAMILY)].set_fontweight("bold")
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    fig.supxlabel(f"Repeat within {cfg.ENRICHMENT_WINDOW} bp, ratio to the genome's background genes", fontsize=7.5, y=0.07)
    handles = [Line2D([], [], marker="o", markersize=4, color=cfg.HTH_CLASS_COLOURS[name], label=name) for name in cfg.ENRICHMENT_CLASSES]
    handles += [Line2D([], [], marker="^", linestyle="none", color="black", markersize=3.5, label="glmmTMB"),
                Line2D([], [], linestyle="none", label="Numbers: samples, " + " | ".join(cfg.ENRICHMENT_CLASSES))]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower center", ncol=len(handles), handlelength=1.4)

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.16, top=0.94, wspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI)
    plt.close(fig)

    return path

TERMINATOR_LAYOUTS = {"convergent": "Convergent\nneighbour", "same strand, far": "Same strand,\n> 50 bp", "same strand, near": "Same strand,\n≤ 50 bp"}

def terminator_panel(table, path):

    rows = table.loc[table["hth_class"].isin(cfg.ENRICHMENT_CLASSES) & table["layout"].isin(TERMINATOR_LAYOUTS)]
    if rows.empty: raise RuntimeError("Nothing to plot in the terminator check.")

    fig, axis = plt.subplots(figsize=cfg.TERMINATOR_SIZE)
    bars = {"background genes": "#9a9a9a", **{name: cfg.HTH_CLASS_COLOURS[name] for name in cfg.ENRICHMENT_CLASSES}}
    width = 0.26

    for group, layout in enumerate(TERMINATOR_LAYOUTS):

        chosen = rows.loc[rows["layout"] == layout].set_index("hth_class")
        background = chosen["background_rate"].iloc[0]

        for index, (name, colour) in enumerate(bars.items()):

            x = group + (index - 1) * width
            value = background if name == "background genes" else chosen.loc[name, "focal_rate"]
            axis.bar(x, 100 * value, width=width * 0.92, color=colour, linewidth=0)
            if name == "background genes": continue
            row = chosen.loc[name]
            label_top = 100 * value

            if isinstance(row["difference_ci"], str):

                low, high = (background + float(end) for end in row["difference_ci"].strip("[]").split(","))
                axis.errorbar(x, 100 * value, yerr=[[100 * (value - max(low, 0))], [100 * (high - value)]], fmt="none", ecolor="#333333",
                              elinewidth=0.8, capsize=1.8, zorder=3)
                label_top = 100 * high

            axis.annotate(f"{row['positive_flanks']}/{row['flanks']}", (x, label_top), xytext=(0, 2), textcoords="offset points",
                          ha="center", va="bottom", fontsize=5.5, rotation=90)

    axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
    axis.set_xticks(range(len(TERMINATOR_LAYOUTS)))
    axis.set_xticklabels(TERMINATOR_LAYOUTS.values(), fontsize=7)
    axis.set_xlabel("Gene after the focal gene", fontsize=7.5)
    axis.set_ylabel(f"Inverted repeat within {enrichment_cfg.TERMINATOR_WINDOW} bp downstream,\n% of flanks, samples weighted equally", fontsize=7.5)
    axis.tick_params(axis="both", length=2.5, width=0.6, labelsize=7)
    for side in ("top", "right"): axis.spines[side].set_visible(False)
    for side in ("left", "bottom"): axis.spines[side].set_linewidth(0.6)
    handles = [Patch(color=colour, label=name.capitalize() if name == "background genes" else name) for name, colour in bars.items()]
    handles.append(Line2D([], [], linestyle="none", label="a/b: flanks with a repeat / all"))
    axis.legend(handles=handles, fontsize=6, frameon=False, loc="upper right", handlelength=1.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)

    return path

def genome_count(panel):

    table = pd.read_csv(background.panel_path(enrichment_cfg.BACKGROUND_COMPARISON_PATH, panel), sep="\t")
    row = table.loc[(table["matching"] == "genome") & (table["samples_scope"] == "all") & (table["repeat_set"] == "all")
                    & (table["window"] == cfg.ENRICHMENT_WINDOW) & (table["hth_class"] == "any HTH")]

    return int(row["genomes"].item())

def draw():

    table = pd.read_csv(enrichment_cfg.BACKGROUND_MODEL_TABLE_PATH, sep="\t")
    genomes = {panel: genome_count(panel) for panel in GENOME_SETS}
    paths = [enrichment_panel(table, genomes, panels, cfg.ENRICHMENT_DIR / name) for name, panels in cfg.ENRICHMENT_PANELS.items()]
    paths += [terminator_panel(pd.read_csv(background.panel_path(enrichment_cfg.TERMINATOR_CHECK_PATH, panel), sep="\t"), cfg.ENRICHMENT_DIR / name)
              for panel, name in cfg.TERMINATOR_PANELS.items()]

    return paths

def draw_families():

    families = pd.read_csv(enrichment_cfg.FAMILY_MODEL_TABLE_PATH, sep="\t")
    classes = pd.read_csv(enrichment_cfg.BACKGROUND_MODEL_TABLE_PATH, sep="\t")

    return [family_column(families, classes, cfg.FAMILY_ENRICHMENT_PATH), family_panel(families, classes, cfg.FAMILY_SUPPLEMENT_PATH)]
