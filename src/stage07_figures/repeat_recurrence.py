from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage05_C_vmatch.step5c.measures import field_column, sources
from src.stage07_figures import config_figures as cfg

FIGURE_FILENAMES = {
    "opportunity_recurrence": "step5c_opportunity_recurrence.png",
    "recurrent_structure": "step5c_recurrent_structure.png",
}

_SCOPE_COLORS = {
    "within_flank": "#3B82B8",
    "all": "#78716C",
}
_SCOPE_LABELS = {
    "within_flank": "Within flank",
    "all": "All",
}
_GROUP_AXIS_LABELS = {
    "no_HTH": "no HTH",
    "HTH+": "one HTH",
    "HTH++": "two or\nmore HTH",
}
_REPEAT_SET_PALETTE = ("#1BAF7A", "#4A3AA7", "#EDA100", "#E87BA4", "#2A78D6", "#E34948")

def draw() -> list[Path]:

    output_dir = Path(cfg.REPEAT_RECURRENCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename
        for name, filename in FIGURE_FILENAMES.items()
    }

    proteins = _read_tsv(Path(vmatch_cfg.STEP5_PROTEIN_SUMMARY_PATH))
    copies = _read_tsv(Path(vmatch_cfg.STEP5_COPIES_PATH))
    architectures = _read_tsv(Path(vmatch_cfg.STEP5_ARCHITECTURES_PATH))
    matches = _read_tsv(Path(vmatch_cfg.STEP5_COPY_MATCHES_PATH))
    recurrence = _read_tsv(Path(vmatch_cfg.STEP5_ARCHITECTURE_RECURRENCE_PATH))

    _plot_opportunity_and_recurrence(proteins, paths["opportunity_recurrence"])
    _plot_recurrent_structure(copies, architectures, matches, recurrence, paths["recurrent_structure"])
    return list(paths.values())

def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path} is empty and has no table header") from error

def _hth_class_groups() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    groups = tuple(vmatch_cfg.HTH_CLASSES)
    labels = tuple(_GROUP_AXIS_LABELS.get(group, group) for group in groups)
    return "hth_class", groups, labels

def _group_colors(groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(cfg.HTH_CLASS_COLOURS[group] for group in groups)

def _arch_repeat_sets() -> tuple[vmatch_cfg.RepeatSet, ...]:

    return tuple(repeat_set for repeat_set in vmatch_cfg.STEP5_REPEAT_SETS if "arch" in sources(repeat_set))

def _repeat_set_colors() -> dict[str, str]:
    return {repeat_set.name: _REPEAT_SET_PALETTE[index]
            for index, repeat_set in enumerate(_arch_repeat_sets())}

def _copy_length_bin_order() -> tuple[str, ...]:
    return tuple(label for _, _, label in vmatch_cfg.STEP5_COPY_LENGTH_BINS)

def _plot_opportunity_and_recurrence(
    proteins: pd.DataFrame,
    output_path: Path,
) -> None:
    group_column, groups, labels = _hth_class_groups()
    set_colors = _repeat_set_colors()
    required = {"selected_protein", group_column, "searched_units"}
    required.update(field_column(repeat_set, "arch", metric)
                    for repeat_set in _arch_repeat_sets()
                    for metric in ("recurrence_prevalence", "recurrence_breadth"))
    missing = required - set(proteins.columns)
    if missing:
        raise ValueError(
            f"{Path(vmatch_cfg.STEP5_PROTEIN_SUMMARY_PATH).name} lacks figure fields: "
            f"{', '.join(sorted(missing))}"
        )
    if proteins.empty:
        _placeholder_figure(
            output_path,
            "Opportunity and recurrence",
            "No protein rows are available.",
        )
        return

    selected = proteins.loc[
        proteins[group_column].astype(str).isin(groups)
    ].copy()
    if selected.empty:
        _placeholder_figure(
            output_path,
            "Opportunity and recurrence",
            "No proteins carry a declared HTH class.",
        )
        return

    opportunity_rows = selected.loc[
        selected["selected_protein"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(("true", "1", "yes"))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    _grouped_boxplot(
        axes[0],
        opportunity_rows,
        group_column,
        groups,
        labels,
        {"searched_units": _SCOPE_COLORS["within_flank"]},
    )
    axes[0].set_title("Searched units among selected proteins")
    axes[0].set_ylabel("Searched units per protein")

    _opportunity_histogram(
        axes[1],
        opportunity_rows,
        group_column,
        groups,
        labels,
    )
    axes[1].set_title("Within-flank searched-unit distribution")
    axes[1].legend(frameon=False, fontsize=8)

    _grouped_prevalence(
        axes[2],
        selected,
        group_column,
        groups,
        labels,
        {field_column(repeat_set, "arch", "recurrence_prevalence"): color
         for repeat_set, color in zip(_arch_repeat_sets(), set_colors.values(), strict=True)},
    )
    axes[2].set_title("Architecture recurrence")
    axes[2].set_ylabel("Fraction of assessable proteins")
    axes[2].set_ylim(0, 1)

    _grouped_boxplot(
        axes[3],
        selected,
        group_column,
        groups,
        labels,
        {field_column(repeat_set, "arch", "recurrence_breadth"): color
         for repeat_set, color in zip(_arch_repeat_sets(), set_colors.values(), strict=True)},
    )
    axes[3].set_title("Normalized architecture breadth")
    axes[3].set_ylabel("Participating / searched units")
    axes[3].set_ylim(-0.03, 1.03)

    _repeat_set_legend(axes[3], set_colors)
    fig.suptitle("Step 5C searched units and cross-representative recurrence")
    fig.tight_layout()
    _atomic_save_figure(fig, output_path)

def _opportunity_histogram(
    axis,
    rows: pd.DataFrame,
    group_column: str,
    groups: Sequence[str],
    labels: Sequence[str],
) -> None:

    bins = np.arange(-0.5, 21.5, 1.0)
    any_values = False
    for group, label, color in zip(
        groups, labels, _group_colors(groups), strict=True
    ):
        values = _numeric(
            rows.loc[
                rows[group_column].astype(str) == group,
                "searched_units",
            ]
        ).dropna()
        if values.empty:
            continue
        any_values = True
        clipped = values.clip(upper=20).to_numpy()
        axis.hist(
            clipped,
            bins=bins,
            weights=np.full(len(clipped), 100.0 / len(clipped)),
            histtype="step",
            linewidth=1.8,
            color=color,
            label=str(label).replace("\n", " "),
        )
    if not any_values:
        axis.text(
            0.5,
            0.5,
            "No opportunity values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlim(-0.5, 20.5)
    axis.set_xticks((0, 2, 5, 10, 15, 20), ("0", "2", "5", "10", "15", "20+"))
    axis.set_xlabel("Searched units per protein")
    axis.set_ylabel("Selected proteins (%)")
    _clean_axis(axis)

def _series_offsets(count: int) -> tuple[np.ndarray, float]:
    width = 0.8 / count
    return (np.arange(count) - (count - 1) / 2) * width, width

def _grouped_boxplot(
    axis,
    rows: pd.DataFrame,
    group_column: str,
    groups: Sequence[str],
    labels: Sequence[str],
    series: Mapping[str, str],
) -> None:
    positions = np.arange(len(groups), dtype=float)
    offsets, width = _series_offsets(len(series))
    any_values = False
    for (column, color), offset in zip(series.items(), offsets, strict=True):
        values_by_group = []
        plotted_positions = []
        for position, group in zip(positions, groups, strict=True):
            values = _numeric(
                rows.loc[
                    rows[group_column].astype(str) == group,
                    column,
                ]
            ).dropna()
            if values.empty:
                continue
            values_by_group.append(values.to_numpy())
            plotted_positions.append(position + offset)
        if not values_by_group:
            continue
        any_values = True
        boxes = axis.boxplot(
            values_by_group,
            positions=plotted_positions,
            widths=0.7 * width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.8)
    if not any_values:
        axis.text(
            0.5,
            0.5,
            "No assessable values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=9, pad=5)
    _clean_axis(axis)

def _grouped_prevalence(
    axis,
    rows: pd.DataFrame,
    group_column: str,
    groups: Sequence[str],
    labels: Sequence[str],
    series: Mapping[str, str],
) -> None:
    positions = np.arange(len(groups), dtype=float)
    offsets, width = _series_offsets(len(series))
    any_values = False
    for (column, color), offset in zip(series.items(), offsets, strict=True):
        means = []
        counts = []
        for group in groups:
            values = _numeric(
                rows.loc[
                    rows[group_column].astype(str) == group,
                    column,
                ]
            ).dropna()
            means.append(float(values.mean()) if not values.empty else np.nan)
            counts.append(len(values))
        if not any(math.isfinite(value) for value in means):
            continue
        any_values = True
        bars = axis.bar(
            positions + offset,
            means,
            width,
            color=color,
            alpha=0.85,
        )
        for bar, count, mean in zip(bars, counts, means, strict=True):
            if not math.isfinite(mean):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(mean + 0.025, 0.97),
                f"n={count}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )
    if not any_values:
        axis.text(
            0.5,
            0.5,
            "No assessable values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=9, pad=5)
    _clean_axis(axis)

_AGREEMENT_FLAGS = (
    ("full_copy_match", "All Copies Matched"),
    ("same_copy_number", "Same Copy Number"),
    ("same_topology", "Same Topology"),
    ("same_biological_relation", "Same Relation"),
    ("same_side_pattern", "Same Side Pattern"),
    ("same_spacing", "Same Spacing"),
    ("same_position_pattern", "Same Position Pattern"),
    ("same_boundary_pattern", "Same Boundary Pattern"),
)

def _plot_recurrent_structure(
    copies: pd.DataFrame,
    architectures: pd.DataFrame,
    matches: pd.DataFrame,
    recurrence: pd.DataFrame,
    output_path: Path,
) -> None:
    if recurrence.empty:
        _placeholder_figure(
            output_path,
            "Structure of recurrent architectures",
            "No direct recurrent architecture comparisons were found.",
        )
        return
    (
        recurrent_architectures,
        recurrent_copies,
        selected_matches,
        recurrent_comparisons,
    ) = _recurrent_structure_rows(
        copies, architectures, matches, recurrence
    )

    fig = plt.figure(figsize=(16, 17))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(1, 1, 1, 1.25),
        hspace=0.62,
        wspace=0.38,
    )
    architecture_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    copy_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    match_axes = [fig.add_subplot(grid[2, index]) for index in range(3)]
    agreement_axis = fig.add_subplot(grid[3, :])

    architecture_categories = (
        (
            "topology",
            "Architecture topology",
            ("simple_pair", "linear_array"),
        ),
        (
            "copy_class",
            "Architecture copy class",
            ("pair", "small_array", "large_array"),
        ),
        (
            "biological_relation",
            "Architecture relation",
            ("direct",),
        ),
    )
    for axis, (column, title, order) in zip(
        architecture_axes, architecture_categories, strict=True
    ):
        _stacked_category_bars(axis, recurrent_architectures, column, order)
        axis.set_title(title)
        axis.set_xlabel("Recurrent architectures")

    copy_categories = (
        (
            "length_class",
            "Copy length",
            _copy_length_bin_order(),
        ),
        (
            "distance_class",
            "Distance from central CDS",
            tuple(label for _, _, label in vmatch_cfg.STEP5_DISTANCE_BINS),
        ),
        (
            "boundary_class",
            "Copy boundary",
            (
                "intergenic",
                "boundary_spanning",
                "neighbor_cds",
                "boundary_unknown",
            ),
        ),
    )
    for axis, (column, title, order) in zip(
        copy_axes, copy_categories, strict=True
    ):
        _stacked_category_bars(axis, recurrent_copies, column, order)
        axis.set_title(title)
        axis.set_xlabel("Unique copies")

    _match_similarity_boxplot(match_axes[0], selected_matches)
    _stacked_category_bars(
        match_axes[1],
        selected_matches,
        "orientation",
        ("direct", "reverse_complement", "ambiguous"),
    )
    match_axes[1].set_title("Selected-match orientation")
    match_axes[1].set_xlabel("Unique selected matches")
    _stacked_category_bars(
        match_axes[2],
        selected_matches,
        "side_correspondence",
        ("same_side", "opposite_side"),
    )
    match_axes[2].set_title("Selected-match side correspondence")
    match_axes[2].set_xlabel("Unique selected matches")

    _agreement_bars(agreement_axis, recurrent_comparisons)
    _scope_legend(architecture_axes[-1])
    fig.suptitle(
        "Architecture, copy, and match structure in direct recurrence",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.95)
    _atomic_save_figure(fig, output_path)

def _recurrent_structure_rows(
    copies: pd.DataFrame,
    architectures: pd.DataFrame,
    matches: pd.DataFrame,
    recurrence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    copy_fields = {
        "step5c_copy_id",
        "length_class",
        "distance_class",
        "boundary_class",
    }
    architecture_fields = {
        "step5c_architecture_id",
        "scope",
        "copy_ids",
        "topology",
        "copy_class",
        "biological_relation",
        "recurrent_comparison_count",
    }
    match_fields = {
        "step5c_match_id",
        "comparison_scope",
        "side_correspondence",
        "direct_similarity",
        "reverse_complement_similarity",
        "similarity",
        "orientation",
        "shared_component_reuse",
        "independent",
    }
    recurrence_fields = {
        "comparison_scope",
        "architecture_1_id",
        "architecture_2_id",
        "selected_match_ids",
    }
    recurrence_fields.update(field for field, _ in _AGREEMENT_FLAGS)
    _require_figure_fields(
        copies, Path(vmatch_cfg.STEP5_COPIES_PATH), copy_fields
    )
    _require_figure_fields(
        architectures, Path(vmatch_cfg.STEP5_ARCHITECTURES_PATH), architecture_fields
    )
    _require_figure_fields(
        matches, Path(vmatch_cfg.STEP5_COPY_MATCHES_PATH), match_fields
    )
    _require_figure_fields(
        recurrence,
        Path(vmatch_cfg.STEP5_ARCHITECTURE_RECURRENCE_PATH),
        recurrence_fields,
    )

    copy_index = _figure_index(
        copies,
        "step5c_copy_id",
        Path(vmatch_cfg.STEP5_COPIES_PATH),
    )
    architecture_index = _figure_index(
        architectures,
        "step5c_architecture_id",
        Path(vmatch_cfg.STEP5_ARCHITECTURES_PATH),
    )
    match_index = _figure_index(
        matches,
        "step5c_match_id",
        Path(vmatch_cfg.STEP5_COPY_MATCHES_PATH),
    )

    recurrent_ids = set(recurrence["architecture_1_id"].astype(str)) | set(
        recurrence["architecture_2_id"].astype(str)
    )
    unknown_architectures = recurrent_ids - set(architecture_index.index)
    if unknown_architectures:
        raise ValueError(
            "Recurrence table references unknown Step 5C architectures: "
            + ", ".join(sorted(unknown_architectures)[:5])
        )
    recurrent_count_ids = set(
        architecture_index.loc[
            pd.to_numeric(
                architecture_index["recurrent_comparison_count"],
                errors="raise",
            )
            > 0
        ].index
    )
    if recurrent_count_ids != recurrent_ids:
        raise ValueError(
            "Architecture recurrent_comparison_count disagrees with the "
            "recurrence table"
        )

    recurrent_copy_ids: set[str] = set()
    for architecture_id in sorted(recurrent_ids):
        row = architecture_index.loc[architecture_id]
        scope = str(row["scope"])
        if scope != "within_flank":
            raise ValueError(
                f"Architecture {architecture_id} has unknown scope {scope!r}"
            )
        copy_ids = _figure_json_ids(
            row["copy_ids"],
            f"architecture {architecture_id} copy_ids",
        )
        unknown_copies = set(copy_ids) - set(copy_index.index)
        if unknown_copies:
            raise ValueError(
                f"Architecture {architecture_id} references unknown copies: "
                + ", ".join(sorted(unknown_copies)[:5])
            )
        recurrent_copy_ids.update(copy_ids)

    selected_match_ids = {
        match_id
        for row_index, value in recurrence["selected_match_ids"].items()
        for match_id in _figure_json_ids(
            value, f"recurrence row {row_index} selected_match_ids"
        )
    }
    if not selected_match_ids:
        raise ValueError("Recurrent comparisons contain no selected matches")
    unknown_matches = selected_match_ids - set(match_index.index)
    if unknown_matches:
        raise ValueError(
            "Recurrence table references unknown copy matches: "
            + ", ".join(sorted(unknown_matches)[:5])
        )

    recurrent_architectures = architecture_index.loc[
        sorted(recurrent_ids)
    ].copy().reset_index(drop=True)
    recurrent_copy_rows = []
    for copy_id in sorted(recurrent_copy_ids):
        copy = copy_index.loc[copy_id]
        recurrent_copy_rows.append(
            {
                "step5c_copy_id": copy_id,
                "scope": "within_flank",
                "length_class": copy["length_class"],
                "distance_class": copy["distance_class"],
                "boundary_class": copy["boundary_class"],
            }
        )
    recurrent_copies = pd.DataFrame(recurrent_copy_rows)
    selected_matches = match_index.loc[sorted(selected_match_ids)].copy()
    selected_matches["scope"] = selected_matches["comparison_scope"].astype(str)
    for field in (
        "direct_similarity",
        "reverse_complement_similarity",
        "similarity",
    ):
        selected_matches[f"_{field}"] = pd.to_numeric(
            selected_matches[field], errors="raise"
        )
    independent = selected_matches["independent"].map(
        {"True": True, "False": False, "true": True, "false": False}
    )
    shared = selected_matches["shared_component_reuse"].map(
        {"True": True, "False": False, "true": True, "false": False}
    )
    if independent.isna().any() or shared.isna().any():
        raise ValueError("Selected copy matches have invalid status flags")
    if not independent.all() or shared.any():
        raise ValueError("Recurrence references a shared-component copy match")

    comparison_rows = recurrence.copy()
    comparison_rows["scope"] = comparison_rows["comparison_scope"].astype(str)
    unknown_scopes = set(comparison_rows["scope"]) - {"within_flank"}
    if unknown_scopes:
        raise ValueError(
            "Recurrence table has unknown comparison scopes: "
            + ", ".join(sorted(unknown_scopes))
        )
    for field, _ in _AGREEMENT_FLAGS:
        values = comparison_rows[field].map(
            {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}
        )
        if values.isna().any():
            raise ValueError(f"Recurrence table has invalid {field} values")
        comparison_rows[field] = values
    return (
        recurrent_architectures,
        recurrent_copies,
        selected_matches.reset_index(drop=True),
        comparison_rows,
    )

def _require_figure_fields(
    rows: pd.DataFrame, path: Path, required: set[str]
) -> None:
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"{path.name} lacks recurrent-structure fields: "
            + ", ".join(sorted(missing))
        )

def _figure_index(rows: pd.DataFrame, field: str, path: Path) -> pd.DataFrame:
    values = rows[field].astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"{path.name} contains an empty {field}")
    if values.duplicated().any():
        duplicate = values.loc[values.duplicated(keep=False)].iloc[0]
        raise ValueError(f"{path.name} contains duplicate {field} {duplicate}")
    indexed = rows.copy()
    indexed[field] = values
    return indexed.set_index(field, drop=False)

def _figure_json_ids(value: object, label: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item for item in parsed
    ):
        raise ValueError(f"{label} must be a JSON list of nonempty IDs")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} contains duplicate IDs")
    return tuple(parsed)

def _match_similarity_boxplot(axis, matches: pd.DataFrame) -> None:
    columns = (
        ("_direct_similarity", "Direct", "#A3A3A3"),
        ("_reverse_complement_similarity", "Reverse Complement", "#8B5CF6"),
        ("_similarity", "Selected Maximum", "#16A34A"),
    )
    values = [matches[column].to_numpy(dtype=float) for column, _, _ in columns]
    boxes = axis.boxplot(
        values,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for box, (_, _, color) in zip(boxes["boxes"], columns, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.82)
    axis.axhline(
        float(vmatch_cfg.STEP5_SIMILARITY_THRESHOLD),
        color="#525252",
        linestyle="--",
        linewidth=1,
    )
    axis.set_xticks(
        np.arange(1, len(columns) + 1),
        [label for _, label, _ in columns],
        rotation=12,
    )
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Levenshtein ratio")
    axis.set_title("Selected-match similarity scores")
    _clean_axis(axis)

def _agreement_bars(axis, recurrence: pd.DataFrame) -> None:
    positions = np.arange(len(_AGREEMENT_FLAGS), dtype=float)
    selected = recurrence.loc[recurrence["scope"] == "within_flank"]
    rates = [
        float(selected[field].astype(float).mean())
        for field, _ in _AGREEMENT_FLAGS
    ]
    axis.barh(
        positions,
        rates,
        color=_SCOPE_COLORS["within_flank"],
        alpha=0.85,
    )
    axis.set_yticks(positions, [label for _, label in _AGREEMENT_FLAGS])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    ticks = np.linspace(0, 1, 6)
    axis.set_xticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.set_xlabel("Fraction of direct recurrent comparisons agreeing")
    axis.set_title("Copy completeness and structural agreement")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D4D4D4", linewidth=0.6, alpha=0.5)
    axis.set_axisbelow(True)

def _stacked_category_bars(
    axis,
    rows: pd.DataFrame,
    category_column: str,
    preferred_order: Sequence[str],
) -> None:
    values = rows[[category_column]].copy()
    values[category_column] = values[category_column].astype(str).str.strip()
    values = values.loc[values[category_column] != ""]
    values["scope"] = rows.loc[values.index, "scope"].astype(str)
    unknown_scopes = set(values["scope"]) - {"within_flank"}
    if unknown_scopes:
        raise ValueError(
            "Recurrent-structure rows have unknown comparison scopes: "
            + ", ".join(sorted(unknown_scopes))
        )
    observed = tuple(values[category_column].unique())
    categories = [item for item in preferred_order if item in observed]
    categories.extend(sorted(set(observed) - set(categories)))
    counts = values[category_column].value_counts().reindex(categories, fill_value=0)
    axis.barh(
        np.arange(len(counts)),
        counts.to_numpy(dtype=float),
        color=_SCOPE_COLORS["within_flank"],
        alpha=0.85,
    )
    axis.set_yticks(np.arange(len(counts)), [_display(item) for item in counts.index])
    axis.invert_yaxis()
    _clean_axis(axis)

def _numeric(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip()
    normalized = normalized.mask(
        normalized.str.lower().isin(
            {
                "",
                "na",
                "nan",
                "none",
                "missing",
                "not_assessable",
                "not_estimable",
            }
        )
    )
    normalized = normalized.replace(
        {"True": "1", "False": "0", "true": "1", "false": "0"}
    )
    return pd.to_numeric(normalized, errors="coerce")

def _scope_legend(axis) -> None:
    handles = [
        Patch(
            facecolor=_SCOPE_COLORS["within_flank"],
            alpha=0.85,
            label=_SCOPE_LABELS["within_flank"],
        )
    ]
    axis.legend(handles=handles, frameon=False, fontsize=9)

def _repeat_set_legend(axis, colors: Mapping[str, str]) -> None:
    handles = [
        Patch(facecolor=color, alpha=0.85, label=name)
        for name, color in colors.items()
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        title="Repeat set",
        title_fontsize=9,
    )

def _clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D4D4D4", linewidth=0.6, alpha=0.5)
    axis.set_axisbelow(True)

def _display(value: object) -> str:
    return str(value).replace("_", " ").strip().title()

def _placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.axis("off")
    axis.set_title(title, pad=16)
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        transform=axis.transAxes,
    )
    fig.tight_layout()
    _atomic_save_figure(fig, path)

def _atomic_save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".png",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=cfg.DPI,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
