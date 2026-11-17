import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.stage03_concordance import config_concordance as concordance_cfg
from src.stage03_concordance.pipeline import EmbeddingConfig, EvidenceConfig
from src.stage07_figures import config_figures as cfg

EVIDENCE_TIER_COLORS = {
    "high_confidence_hth": "#1b9e77",
    "likely_non_hth": "#377eb8",
    "potential_missed_or_degenerate_hth": "#d95f02",
    "ambiguous_glpr_only": "#7570b3",
    "ambiguous": "#8c8c8c"}

ESMC_SIGNAL_COLORS = {
    "hth_rich": "#2ca25f",
    "mixed": "#fdae61",
    "hth_poor": "#3182bd"}

def plot_concordance_summary(concordance, evidence_config=None, embedding_config=None,
                             top_architectures=8, title=None, figsize=(18, 12)):

    import matplotlib.pyplot as plt

    if evidence_config is None:

        evidence_config = EvidenceConfig()

    df = concordance.copy()
    tier_order = [
        "high_confidence_hth",
        "potential_missed_or_degenerate_hth",
        "likely_non_hth",
        "ambiguous_glpr_only",
        "ambiguous"]
    tier_order = [t for t in tier_order if t in set(df["evidence_tier"])]
    tier_colors = [EVIDENCE_TIER_COLORS.get(t, "#999999") for t in tier_order]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    _plot_neighbor_histogram(axes[0], df, evidence_config)
    _plot_length_vs_neighbor_score(axes[1], df, evidence_config, tier_order)
    _plot_tier_counts(axes[2], df, tier_order, tier_colors)
    _plot_architecture_tier_mix(axes[3], df, top_architectures, tier_order)
    _plot_local_status_vs_neighbor_signal(axes[4], df)
    _plot_length_by_tier(axes[5], df, tier_order, tier_colors)
    _add_evidence_tier_legend(fig, tier_order)

    if title is None:

        title = "ESMC nearest-neighbor concordance with local HTH architecture calls"

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _add_config_note(fig, df, evidence_config, embedding_config)

    return fig, axes

def _plot_neighbor_histogram(ax, df, evidence_config):

    bins = np.linspace(0, 1, 21)
    groups = [
        ("HTH_positive", "#1b9e77", "Local HTH+"),
        ("HTH_negative", "#377eb8", "Local HTH-")]

    for status, color, label in groups:

        values = df.loc[df["local_HTH_status"] == status, "hth_neighbor_fraction"]
        ax.hist(values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.35,
                color=color,
                density=True,
                label=f"{label} (n={len(values):,})")
        ax.axvline(values.median(), color=color, linestyle="-", linewidth=2)

    _draw_thresholds(ax, evidence_config, axis="x")
    summary = (df.groupby("local_HTH_status")["hth_neighbor_fraction"]
               .agg(["mean", "median"])
               .round(2))
    text = "\n".join(f"{status.replace('HTH_', 'HTH ')}: "
                         f"mean={row['mean']:.2f}, median={row['median']:.2f}"
                         for status, row in summary.iterrows())
    ax.text(0.98,
            0.96,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "alpha": 0.8, "edgecolor": "none"})
    ax.set_title("A. ESMC neighbor support for local HTH calls")
    neighbor_k = _neighbor_k_label(df)
    ax.set_xlabel(f"Fraction of {neighbor_k} nearest ESMC neighbors that are local HTH+")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)

def _plot_length_vs_neighbor_score(ax, df, evidence_config, tier_order):

    for tier in tier_order:

        x = df.loc[df["evidence_tier"] == tier, "length"]
        y = df.loc[df["evidence_tier"] == tier, "hth_neighbor_fraction"]
        ax.scatter(x,
                   y,
                   s=12,
                   alpha=0.45,
                   linewidths=0,
                   color=EVIDENCE_TIER_COLORS.get(tier, "#999999"),
                   label=_pretty_label(tier))

    _draw_thresholds(ax, evidence_config, axis="y")
    ax.set_title("B. Neighbor support versus protein length")
    ax.set_xlabel("Protein length (aa)")
    ax.set_ylabel("HTH+ neighbor fraction")
    ax.set_ylim(-0.04, 1.04)
    corr = df[["length", "hth_neighbor_fraction"]].corr().iloc[0, 1]
    ax.text(0.02,
            0.98,
            f"Pearson r = {corr:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "alpha": 0.8, "edgecolor": "none"})
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

def _plot_tier_counts(ax, df, tier_order, tier_colors):

    counts = df["evidence_tier"].value_counts().reindex(tier_order).dropna().astype(int)
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=tier_colors[:len(counts)])
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_label(x) for x in counts.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("C. Evidence tier counts")
    ax.set_xlabel("Proteins")

    for yi, value in zip(y, counts.values):

        ax.text(value, yi, f" {value:,}", va="center", fontsize=9)

def _plot_architecture_tier_mix(ax, df, top_architectures, tier_order):

    top = df["architecture"].value_counts().head(top_architectures).index
    tab = pd.crosstab(df.loc[df["architecture"].isin(top), "architecture"],
                      df.loc[df["architecture"].isin(top), "evidence_tier"])
    tab = tab.reindex(top).fillna(0)
    tab = tab[[t for t in tier_order if t in tab.columns]]
    tab = tab.iloc[::-1]

    left = np.zeros(len(tab))
    y = np.arange(len(tab))

    for tier in tab.columns:

        values = tab[tier].to_numpy()
        ax.barh(y,
                values,
                left=left,
                color=EVIDENCE_TIER_COLORS.get(tier, "#999999"),
                label=_pretty_label(tier))
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([_short_architecture_label(x) for x in tab.index], fontsize=8)
    ax.set_title("D. Major architectures split across evidence tiers")
    ax.set_xlabel("Proteins")
    ax.text(0.98,
            0.02,
            "Colors = evidence tiers",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444")

def _plot_local_status_vs_neighbor_signal(ax, df):

    signal_order = ["hth_rich", "mixed", "hth_poor"]
    status_order = ["HTH_positive", "HTH_negative"]
    tab = pd.crosstab(df["local_HTH_status"], df["esmc_hth_signal"])
    tab = tab.reindex(status_order).fillna(0)
    tab = tab[[s for s in signal_order if s in tab.columns]]

    left = np.zeros(len(tab))
    y = np.arange(len(tab))

    for signal in tab.columns:

        values = tab[signal].to_numpy()
        ax.barh(y,
                values,
                left=left,
                color=ESMC_SIGNAL_COLORS.get(signal, "#999999"),
                label=_pretty_label(signal))
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(["Local HTH+", "Local HTH-"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("E. Local HTH calls versus kNN signal")
    ax.set_xlabel("Proteins")
    ax.legend(frameon=False, fontsize=8)

    totals = tab.sum(axis=1).to_numpy()

    for yi, total in zip(y, totals):

        ax.text(total, yi, f" {int(total):,}", va="center", fontsize=9)

def _plot_length_by_tier(ax, df, tier_order, tier_colors):

    data = [
        df.loc[df["evidence_tier"] == tier, "length"].dropna().to_numpy()
        for tier in tier_order]
    labels = [_pretty_label(tier) for tier in tier_order]
    box = ax.boxplot(data, patch_artist=True, showfliers=False)

    for patch, color in zip(box["boxes"], tier_colors):

        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("F. Length distributions by evidence tier")
    ax.set_ylabel("Protein length (aa)")
    ax.text(0.98,
            0.98,
            "Colors = evidence tiers",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#444444")

def _add_evidence_tier_legend(fig, tier_order):

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=EVIDENCE_TIER_COLORS.get(tier, "#999999"),
                       label=_pretty_label(tier))
        for tier in tier_order]
    fig.legend(handles=handles,
               title="Evidence tier colors",
               loc="upper center",
               bbox_to_anchor=(0.5, 0.965),
               ncol=min(len(handles), 5),
               frameon=False,
               fontsize=9,
               title_fontsize=9)

def _add_config_note(fig, df, evidence_config, embedding_config):

    lines = []

    if "hth_neighbor_k" in df.columns:

        values = df["hth_neighbor_k"].dropna().unique()

        if len(values) == 1:

            lines.append(f"kNN k = {int(values[0])}")

    lines.extend([
        f"HTH-rich >= {evidence_config.hth_rich:.2f}",
        f"HTH-poor <= {evidence_config.hth_poor:.2f}"])

    if embedding_config is not None:

        lines.append(f"embedding space = {embedding_config.embedding_space}")

    fig.text(0.995,
             0.006,
             "\n".join(lines),
             ha="right",
             va="bottom",
             fontsize=8,
             color="#444444",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                   "alpha": 0.85, "edgecolor": "#dddddd"})

def _neighbor_k_label(df):

    if "hth_neighbor_k" not in df.columns:

        return "k"

    values = df["hth_neighbor_k"].dropna().unique()

    if len(values) == 1:

        return int(values[0])

    return "k"

def _draw_thresholds(ax, evidence_config, axis):

    if axis == "x":

        ax.axvline(evidence_config.hth_poor, color="black", linestyle=":", linewidth=1)
        ax.axvline(evidence_config.hth_rich, color="black", linestyle=":", linewidth=1)
        trans = ax.get_xaxis_transform()
        ax.text(evidence_config.hth_poor, 0.02, "poor", transform=trans, ha="center", va="bottom", fontsize=8)
        ax.text(evidence_config.hth_rich, 0.02, "rich", transform=trans, ha="center", va="bottom", fontsize=8)
    else:

        ax.axhline(evidence_config.hth_poor, color="black", linestyle=":", linewidth=1)
        ax.axhline(evidence_config.hth_rich, color="black", linestyle=":", linewidth=1)
        trans = ax.get_yaxis_transform()
        ax.text(0.02, evidence_config.hth_poor, "poor", transform=trans, ha="left", va="bottom", fontsize=8)
        ax.text(0.02, evidence_config.hth_rich, "rich", transform=trans, ha="left", va="bottom", fontsize=8)

def _pretty_label(value):

    return str(value).replace("_", " ")

def _short_architecture_label(value, max_len=44):

    value = str(value).replace(" > GLPR", " >G")

    if len(value) <= max_len:

        return value

    return value[:max_len - 3] + "..."

def draw():

    import matplotlib.pyplot as plt

    concordance = pd.read_csv(concordance_cfg.CONCORDANCE_PATH, sep="\t")
    evidence = EvidenceConfig(hth_rich=concordance_cfg.HTH_RICH, hth_poor=concordance_cfg.HTH_POOR)
    embedding = EmbeddingConfig(path_h5=concordance_cfg.EMBEDDING_ARCHIVE_PATH, embedding_space=concordance_cfg.EMBEDDING_SPACE)
    title = f"ESMC nearest-neighbor concordance: {concordance_cfg.MODEL_ID} ({concordance_cfg.EMBEDDING_SPACE}, kNN={concordance_cfg.NEIGHBOR_K})"
    figure, _ = plot_concordance_summary(concordance, evidence_config=evidence, embedding_config=embedding, title=title)
    path = cfg.CONCORDANCE_DIR / f"concordance_summary_{concordance_cfg.ANALYSIS_NAME}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return [path]
