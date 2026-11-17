from __future__ import annotations

import pandas as pd

from src.stage05_C_vmatch import config_vmatch as cfg

from .measures import field_column, measures_for, presence_count_field, sources

NO_HTH_FAMILY_LABEL = "(no HTH)"

SHARED_MEAN_METRICS = ("searched_units", "searched_bases")

UNWEIGHTED_ONLY = ("median_nearest_distance_to_cds", "mean_of_protein_mean_copy_number")

BASE_RESULT_FIELDS = ("repeat_set", "unit", "grouped_by", "group", "protein_rows", "distinct_clusters",
                      "recurrence_assessable_rows", "evidence_recurrence_assessable_rows", "min_units")

def _set_rows(proteins, architectures, repeat_set):

    active = sources(repeat_set)
    assignments: dict[str, pd.Series] = {}
    mean_metrics = list(SHARED_MEAN_METRICS)
    positive_columns: list[str] = []

    for source in active:

        presence = field_column(repeat_set, source, presence_count_field(source))
        positive = f"{source}_positive"
        assignments[positive] = proteins[presence].gt(0).astype(float)
        positive_columns.append(positive)

        for measure in measures_for(repeat_set):

            if measure.source != source: continue
            metric = f"{source}_{measure.name}"
            assignments[metric] = proteins[measure.column(repeat_set)]
            mean_metrics.append(metric)

    if "arch" in active:

        assignments["arch_recurrent"] = proteins[field_column(repeat_set, "arch", "recurrence_prevalence")]
        mean_metrics.append("arch_recurrent")
        counted = architectures.loc[architectures["repeat_set"] == repeat_set.name].groupby("protein_accession")
        accessions = proteins["protein_accession"]
        assignments["nearest_distance_to_cds"] = accessions.map(counted["distance_to_cds"].min())
        assignments["mean_copy_number"] = accessions.map(counted["copy_number"].mean())

    return proteins.assign(**assignments), tuple(mean_metrics), tuple(positive_columns)

def _unit_rows(frame, unit):

    if unit == "protein_cluster_representative":

        return frame.loc[frame["protein_accession"] == frame["representative_accession"]]

    return frame

def _weights(frame, column, unit):

    defined = frame.loc[frame[column].notna()]

    if unit == "protein_cluster_weighted":

        return defined, 1.0 / defined.groupby("cluster_id")[column].transform("size")

    return defined, pd.Series(1.0, index=defined.index)

def _group_labels(frame, view):

    if view == "hth_presence":

        return frame["hth_class"].map(cfg.STEP5_HTH_PRESENCE)

    if view == "hth_families":

        return frame["hth_families"].fillna("").replace("", NO_HTH_FAMILY_LABEL)

    return frame[view].fillna("")

def measure_view(frame, view, repeat_set, unit, mean_metrics, positive_columns):

    rows = _unit_rows(frame, unit)
    labels = _group_labels(rows, view)
    grouped = rows.assign(group=labels).groupby("group", sort=True)
    summary = pd.DataFrame({"protein_rows": grouped.size(),
                            "distinct_clusters": grouped["cluster_id"].nunique(),
                            "recurrence_assessable_rows": grouped["recurrence_assessable"].sum().astype(int),
                            "evidence_recurrence_assessable_rows": grouped["evidence_recurrence_assessable"].sum().astype(int)})

    for positive in positive_columns:

        summary[f"{positive}_rows"] = grouped[positive].sum().astype(int)

    cluster_weight_column = positive_columns[0] if positive_columns else None

    for column in mean_metrics:

        defined, weight = _weights(rows, column, unit)
        keyed = defined.assign(group=_group_labels(defined, view), weight=weight,
                               weighted=defined[column] * weight)
        totals = keyed.groupby("group", sort=True)[["weight", "weighted"]].sum()
        summary[f"mean_{column}"] = totals["weighted"] / totals["weight"]

        if column == cluster_weight_column: summary["cluster_weight"] = totals["weight"]

    if unit == "protein_cluster_weighted":

        for column in UNWEIGHTED_ONLY: summary[column] = pd.NA

    elif "nearest_distance_to_cds" in frame.columns:

        summary["median_nearest_distance_to_cds"] = grouped["nearest_distance_to_cds"].median()
        summary["mean_of_protein_mean_copy_number"] = grouped["mean_copy_number"].mean()

    else:

        for column in UNWEIGHTED_ONLY: summary[column] = pd.NA

    summary = summary.reset_index()
    summary.insert(0, "grouped_by", view)
    summary.insert(0, "unit", unit)
    summary.insert(0, "repeat_set", repeat_set)
    summary["min_units"] = cfg.STEP5_MIN_UNITS

    return summary

def _result_fields(frames):

    positive_fields: list[str] = []
    mean_fields = [f"mean_{column}" for column in SHARED_MEAN_METRICS]

    for repeat_set in cfg.STEP5_REPEAT_SETS:

        _, mean_metrics, positive_columns = frames[repeat_set.name]

        for column in positive_columns:

            name = f"{column}_rows"

            if name not in positive_fields: positive_fields.append(name)

        for column in mean_metrics:

            name = f"mean_{column}"

            if name not in mean_fields: mean_fields.append(name)

    return (*BASE_RESULT_FIELDS, *positive_fields, "cluster_weight", *mean_fields, *UNWEIGHTED_ONLY)

def build_views(frames):

    built = {}
    result_fields = _result_fields(frames)

    for view in cfg.STEP5_RESULT_VIEWS:

        parts = []

        for repeat_set in cfg.STEP5_REPEAT_SETS:

            frame, mean_metrics, positive_columns = frames[repeat_set.name]

            for unit in cfg.STEP5_RESULT_UNITS:

                parts.append(measure_view(frame, view, repeat_set.name, unit, mean_metrics, positive_columns))

        built[view] = pd.concat(parts, ignore_index=True).reindex(columns=list(result_fields))

    return built

def run_results(logger):

    proteins = pd.read_csv(cfg.STEP5_PROTEIN_SUMMARY_PATH, sep="\t",
                           dtype={"eligible": bool, "recurrence_assessable": bool})
    eligible = proteins.loc[proteins["eligible"]]
    architectures = pd.read_csv(cfg.STEP5_MEASURED_ARCHITECTURES_PATH, sep="\t")
    frames = {repeat_set.name: _set_rows(eligible, architectures, repeat_set)
              for repeat_set in cfg.STEP5_REPEAT_SETS}

    for view, table in build_views(frames).items():

        path = cfg.RESULT_MEASURE_PATHS[view]
        table.to_csv(path, sep="\t", index=False, float_format="%.12g")
        logger.info("%s: %s rows grouped by %s", path.name, f"{len(table):,}", view)

    logger.info("Eligible proteins: %s of %s", f"{len(eligible):,}", f"{len(proteins):,}")
