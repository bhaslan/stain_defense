import argparse
import sys

import numpy as np
import pandas as pd

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.evidence import read_evidence

from ..measures import COUNTS, field_column, find_measure, measures_for, sources, union_length

_SHARED_COLUMNS = ("protein_accession", "hth_class", "searched_units", "searched_bases", "eligible")

def _spearman(left, right):

    paired = pd.concat([left, right], axis=1).dropna()

    if len(paired) < 2: return float("nan")

    return float(paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank()))

def _stats(values):

    values = values.dropna().astype(float)

    if values.empty: return "n=0"

    qs = values.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

    return (f"n={len(values)} mean={values.mean():.3f} min={qs.loc[0]:.3f} q25={qs.loc[0.25]:.3f} "
            f"median={qs.loc[0.5]:.3f} q75={qs.loc[0.75]:.3f} p90={qs.loc[0.9]:.3f} p95={qs.loc[0.95]:.3f} "
            f"p99={qs.loc[0.99]:.3f} max={qs.loc[1]:.3f}")

def _pct(numerator, denominator): return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")

def _arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-set", choices=[repeat_set.name for repeat_set in cfg.STEP5_REPEAT_SETS],
                        help="Limit the report to one repeat set (default: all of them).")
    parser.add_argument("--verbose", action="store_true", help="Print quantile and rank-correlation summaries.")
    parser.add_argument("--examples", type=int, default=8,
                        help="Top concentration rows to print (default: 8).")
    arguments = parser.parse_args()

    if arguments.examples < 1: parser.error("--examples must be at least 1")

    return arguments

def _set_rows(summary, repeat_set):

    renames = {}

    for source in sources(repeat_set):

        for field in COUNTS[source]: renames[field_column(repeat_set, source, field)] = f"{source}_{field}"

    for measure in measures_for(repeat_set): renames[measure.column(repeat_set)] = f"{measure.source}_{measure.name}"

    return summary[[*_SHARED_COLUMNS, *renames]].rename(columns=renames).assign(repeat_set=repeat_set.name)

def _evidence_repeat_bases(counted_ids, evidence_by_id):

    unit_spans: dict[tuple[str, str, str], list[tuple[int, int]]] = {}

    for evidence_id in counted_ids:

        item = evidence_by_id[evidence_id]
        key = (item.protein_accession, item.scope, item.search_unit_id)

        for stretch in item.stretches: unit_spans.setdefault(key, []).append((stretch.start, stretch.end))

    return sum(union_length(spans) for spans in unit_spans.values())

def _load_tables(names):

    summary = pd.read_csv(cfg.STEP5_PROTEIN_SUMMARY_PATH, sep="\t")
    measured_arch = pd.read_csv(cfg.STEP5_MEASURED_ARCHITECTURES_PATH, sep="\t")
    measured_evidence = pd.read_csv(cfg.STEP5_MEASURED_EVIDENCE_PATH, sep="\t")
    source = pd.read_csv(cfg.STEP5_ARCHITECTURES_PATH, sep="\t",
                         usecols=["step5c_architecture_id", "topology", "biological_relation", "side_pattern",
                                  "boundary_pattern", "recurrent_comparison_count"])
    architectures = (measured_arch.loc[measured_arch["repeat_set"].isin(names)]
                     .merge(summary[["protein_accession", "hth_class"]], on="protein_accession", how="left",
                            validate="many_to_one")
                     .merge(source, on="step5c_architecture_id", how="left", validate="many_to_one"))
    evidence = measured_evidence.loc[measured_evidence["repeat_set"].isin(names)]
    evidence_by_id = {item.evidence_id: item for item in read_evidence(cfg.STEP4_EVIDENCE_PATH)}
    repeat_sets = [repeat_set for repeat_set in cfg.STEP5_REPEAT_SETS if repeat_set.name in names]
    proteins = pd.concat([_set_rows(summary, repeat_set) for repeat_set in repeat_sets], ignore_index=True)

    return architectures, evidence, evidence_by_id, proteins

def _integrity(architectures, evidence, evidence_by_id, proteins):

    issues = []
    arch_sets = {repeat_set.name for repeat_set in cfg.STEP5_REPEAT_SETS
                 if repeat_set.name in proteins["repeat_set"].unique() and "arch" in sources(repeat_set)}

    if arch_sets:

        per_protein = architectures.groupby(["protein_accession", "repeat_set"], as_index=False)
        aggregated = per_protein.agg(rows=("step5c_architecture_id", "size"))
        arch_proteins = proteins.loc[proteins["repeat_set"].isin(arch_sets)]
        check = arch_proteins.merge(aggregated, on=["protein_accession", "repeat_set"], how="outer")
        check["rows"] = check["rows"].fillna(0).astype(int)
        duplicated = architectures.duplicated(["repeat_set", "step5c_architecture_id"])
        in_window = architectures["distance_to_cds"].between(1, cfg.STEP5_PROXIMAL_CUTOFF, inclusive="both")
        arch_checks = {"duplicate (repeat set, architecture) rows": int(duplicated.sum()),
                       "architecture distances outside [1, STEP5_PROXIMAL_CUTOFF]": int((~in_window).sum()),
                       "negative recurrence_partners values": int((architectures["recurrence_partners"] < 0).sum()),
                       "architecture-count disagreements": int((check["arch_architectures"] != check["rows"]).sum())}

        print("architecture rows", len(architectures), "protein-set rows", len(arch_proteins))

        for label, count in arch_checks.items(): print(label, count)

        issues += [f"{count} {label}" for label, count in arch_checks.items() if count]

    evidence_in_window = evidence["distance_to_cds"].between(1, cfg.STEP5_PROXIMAL_CUTOFF, inclusive="both")
    evidence_duplicated = evidence.duplicated(["repeat_set", "evidence_id"])
    evidence_checks = {"duplicate (repeat set, evidence) rows": int(evidence_duplicated.sum()),
                       "evidence distances outside [1, STEP5_PROXIMAL_CUTOFF]": int((~evidence_in_window).sum())}
    item_disagreements = 0
    bases_disagreements = 0

    for repeat_set in cfg.STEP5_REPEAT_SETS:

        if repeat_set.name not in proteins["repeat_set"].unique(): continue

        items_column = field_column(repeat_set, "evidence", "items")
        bases_column = field_column(repeat_set, "evidence", "repeat_bases")
        subset = proteins.loc[proteins["repeat_set"] == repeat_set.name]
        counted = evidence.loc[evidence["repeat_set"] == repeat_set.name]
        item_counts = counted.groupby("protein_accession").size()
        recomputed_bases = {protein: _evidence_repeat_bases(group["evidence_id"], evidence_by_id)
                            for protein, group in counted.groupby("protein_accession")}

        for _, row in subset.iterrows():

            protein = row["protein_accession"]
            expected_items = int(item_counts.get(protein, 0))
            expected_bases = recomputed_bases.get(protein, 0)

            if row["evidence_items"] != expected_items: item_disagreements += 1
            if row["evidence_repeat_bases"] != expected_bases: bases_disagreements += 1

    evidence_checks["evidence-item disagreements"] = item_disagreements
    evidence_checks["evidence-repeat-bases disagreements"] = bases_disagreements

    print("evidence rows", len(evidence))

    for label, count in evidence_checks.items(): print(label, count)

    issues += [f"{count} {label}" for label, count in evidence_checks.items() if count]

    return issues

def _recurrence_integrity(label, summary_path, pairs_path, repeat_sets):

    summary = pd.read_csv(summary_path, sep="\t", dtype={"protein_accession": str})
    pairs = pd.read_csv(pairs_path, sep="\t")
    breadth = find_measure("evidence", "recurrence_breadth")
    groups, gated = summary["protein_accession"], summary[breadth.gate].astype(bool)
    allowed = (pairs["matched_pieces"] >= cfg.STEP5_MIN_INDEPENDENT_MATCHES) | (pairs["decided_by_tandem"]
                                                                                & (pairs["matched_pieces"] == 1))
    checks = {"rows naming groups absent from the summary": int((~pairs["group_accession"].isin(groups)).sum()),
              "duplicate (repeat set, group, unit pair) rows":
                  int(pairs.duplicated(["repeat_set", "group_accession", "unit_1_id", "unit_2_id"]).sum()),
              "rows with unordered unit IDs": int((pairs["unit_1_id"] >= pairs["unit_2_id"]).sum()),
              "rows with too few matched pieces": int((~allowed).sum())}
    disagreements = {"recurrent units": 0, "recurrent unit pairs": 0, "prevalence": 0, "breadth": 0}

    for repeat_set in repeat_sets:

        set_pairs = pairs.loc[pairs["repeat_set"] == repeat_set.name]
        ends = pd.concat([set_pairs[["group_accession", unit]].set_axis(["group", "unit"], axis=1)
                          for unit in ("unit_1_id", "unit_2_id")]).drop_duplicates()
        units = ends.groupby("group").size().reindex(groups, fill_value=0).to_numpy()
        unit_pairs = set_pairs.groupby("group_accession").size().reindex(groups, fill_value=0).to_numpy()
        prevalence = summary[field_column(repeat_set, "evidence", "recurrence_prevalence")]
        expected_breadth = np.where(gated, units / summary["searched_units"].where(gated, 1), np.nan)
        saved_breadth = summary[breadth.column(repeat_set)].to_numpy(dtype=float)
        same_breadth = np.isclose(saved_breadth, expected_breadth, rtol=0, atol=1e-12) | (np.isnan(saved_breadth)
                                                                                           & np.isnan(expected_breadth))
        same_prevalence = np.where(gated, prevalence == (unit_pairs > 0).astype(int), prevalence.isna())
        disagreements["recurrent units"] += int((summary[field_column(repeat_set, "evidence", "recurrent_units")] != units).sum())
        disagreements["recurrent unit pairs"] += int((summary[field_column(repeat_set, "evidence", "recurrent_unit_pairs")]
                                                      != unit_pairs).sum())
        disagreements["prevalence"] += int((~same_prevalence).sum())
        disagreements["breadth"] += int((~same_breadth).sum())

    checks.update({f"evidence recurrence {name} disagreements": count for name, count in disagreements.items()})

    print(f"{label} evidence recurrence rows", len(pairs), "groups", len(summary), "gated", int(gated.sum()))

    for name, count in checks.items(): print(label, name, count)

    return [f"{label}: {count} {name}" for name, count in checks.items() if count]

def _print_mix(architectures):

    if architectures.empty: return

    copy_bins = pd.cut(architectures["copy_number"], bins=[1.5, 2.5, 3.5, 5.5, 10.5, np.inf],
                       labels=["2", "3", "4-5", "6-10", "11+"])
    neighbor_cds = architectures["boundary_pattern"].astype(str).str.contains("neighbor_cds", regex=False)
    architectures = architectures.assign(copy_bin=copy_bins, has_neighbor_cds=neighbor_cds)

    for title, column in (("COPY NUMBER BINS", "copy_bin"), ("TOPOLOGY", "topology"),
                          ("BIOLOGICAL RELATION MIX", "biological_relation"), ("BOUNDARY MIX", "has_neighbor_cds")):

        grouped = architectures.groupby(["repeat_set", column], observed=True, dropna=False)
        table = grouped.agg(n=("step5c_architecture_id", "size"),
                            median_distance=("distance_to_cds", "median"),
                            recurrence_positive=("recurrence_partners", lambda x: 100.0 * float((x > 0).mean())))
        table["fraction_pct"] = table["n"] / table.groupby(level=0)["n"].transform("sum") * 100

        print(f"\n{title}")
        print(table.round(2).to_string())

def _print_recurrence(architectures, examples):

    if architectures.empty: return

    partners, raw = architectures["recurrence_partners"], architectures["recurrent_comparison_count"]
    comparison = (architectures.assign(counted_less_than_raw=partners < raw, equal=partners == raw)
                  .groupby("repeat_set")
                  .agg(n=("step5c_architecture_id", "size"),
                       counted_partner_positive=("recurrence_partners", lambda x: int((x > 0).sum())),
                       raw_partner_positive=("recurrent_comparison_count", lambda x: int((x > 0).sum())),
                       counted_less_than_raw=("counted_less_than_raw", "sum"),
                       equal=("equal", "sum")))

    print("\nRECURRENCE FILTERING COMPARISON")
    print(comparison.to_string())
    print("\nRECURRENCE DEGREE CONCENTRATION")

    for name, group in architectures.groupby("repeat_set"):

        by_protein = (group.groupby("protein_accession")
                      .agg(architectures=("step5c_architecture_id", "size"),
                           positive_architectures=("recurrence_partners", lambda x: int((x > 0).sum())),
                           degree_sum=("recurrence_partners", "sum"),
                           max_degree=("recurrence_partners", "max"))
                      .sort_values(["degree_sum", "positive_architectures"], ascending=False))
        total_degree = int(by_protein["degree_sum"].sum())

        print(name, "total degree", total_degree, "implied pairs", total_degree // 2)

        for top_n in (1, 5, 10, 20):

            print(f"top {top_n} proteins degree share {_pct(by_protein['degree_sum'].head(top_n).sum(), total_degree):.2f}%")

        print(by_protein.head(examples).to_string())

def _print_verbose(architectures, proteins, examples):

    eligible = proteins.loc[proteins["eligible"]]

    for name in dict.fromkeys(proteins["repeat_set"]):

        aa = architectures.loc[architectures["repeat_set"] == name]

        if aa.empty: continue

        pp = eligible.loc[eligible["repeat_set"] == name]
        positive = int((aa["recurrence_partners"] > 0).sum())
        empty = int((pp["arch_architectures"] == 0).sum())

        print(f"\nREPEAT SET {name}")
        print("architectures", len(aa), "proteins represented", aa.protein_accession.nunique())
        print("nearest-start distance (1-based)", _stats(aa["distance_to_cds"]))
        print("recurrence partners", _stats(aa["recurrence_partners"]))
        print("recurrence-positive", positive, f"({_pct(positive, len(aa)):.2f}%)")
        print("rank correlations distance~partners",
              f"{_spearman(aa['distance_to_cds'], aa['recurrence_partners']):.3f}")
        print("eligible proteins", len(pp), "of", int((proteins["repeat_set"] == name).sum()))
        print("zero-architecture proteins", empty, f"({_pct(empty, len(pp)):.2f}%)")
        print("U_p", _stats(pp["searched_units"]))
        print("occupancy", _stats(pp["arch_occupancy"]))
        print("yield", _stats(pp["arch_yield"]))

    if architectures.empty: return

    bands = pd.cut(architectures["distance_to_cds"], bins=[0, 50, 100, 250, 500, 750, 1000, np.inf],
                   labels=["1-50", "51-100", "101-250", "251-500", "501-750", "751-1000", "1001+"])
    banded = (architectures.assign(distance_band=bands)
              .groupby(["repeat_set", "distance_band"], observed=True)
              .agg(n=("step5c_architecture_id", "size"),
                   recurrence_positive=("recurrence_partners", lambda x: 100.0 * float((x > 0).mean())),
                   median_copy_number=("copy_number", "median"))
              .round(2))

    print("\nDISTANCE-BAND RECURRENCE")
    print(banded.to_string())

def main():

    arguments = _arguments()
    repeat_sets = [repeat_set for repeat_set in cfg.STEP5_REPEAT_SETS if arguments.repeat_set in (None, repeat_set.name)]
    architectures, evidence, evidence_by_id, proteins = _load_tables([repeat_set.name for repeat_set in repeat_sets])

    print("CONFIG")

    for repeat_set in repeat_sets:

        print(repeat_set.name, "kinds", ",".join(repeat_set.kinds))

    print("min_copy_length", cfg.STEP5_MIN_COPY_LENGTH, "proximal_cutoff", cfg.STEP5_PROXIMAL_CUTOFF,
          "min_units", cfg.STEP5_MIN_UNITS)
    print("\nINTEGRITY")

    issues = _integrity(architectures, evidence, evidence_by_id, proteins)
    issues += _recurrence_integrity("Step 5C", cfg.STEP5_PROTEIN_SUMMARY_PATH, cfg.STEP5_EVIDENCE_RECURRENCE_PATH,
                                    repeat_sets)
    issues += _recurrence_integrity("Step 6C", cfg.STEP6_SAMPLE_SUMMARY_PATH, cfg.STEP6_EVIDENCE_RECURRENCE_PATH,
                                    repeat_sets)

    _print_mix(architectures)
    _print_recurrence(architectures, arguments.examples)

    if arguments.verbose: _print_verbose(architectures, proteins, arguments.examples)

    if issues:

        print("\nINTEGRITY FAILED")

        for issue in issues: print(f"- {issue}")

        sys.exit(1)

if __name__ == "__main__": main()
