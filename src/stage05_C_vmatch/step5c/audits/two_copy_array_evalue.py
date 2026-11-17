from __future__ import annotations

import Levenshtein
import numpy as np
import pandas as pd

from src.stage05_C_vmatch import background as panel
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step3c.trf_evidence import read_trf_arrays
from src.stage05_C_vmatch.step4c.architectures.models import Arm
from src.stage05_C_vmatch.step4c.evidence import (
    EvidenceItem,
    composition_from_sequence,
    transition_matrix_from_composition_row,
)
from src.stage05_C_vmatch.step5c.audits.tandem_array_evalue_bound import tandem_evalue
from src.stage05_C_vmatch.step5c.measures import counts_evidence
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage06_repeat_enrichment.background import CLASSES, _estimates, matched, read_flanks

WINDOW = 200
REPORTED_CLASSES = ("HTH+", "HTH++")
PAIR_SETS = {"tandem": (), "all": ("direct", "inverted")}
REPEAT_SETS = {repeat_set.name: repeat_set for repeat_set in cfg.STEP5_REPEAT_SETS}

def _flank_sequence(fasta_path: str) -> str:

    with open(fasta_path) as handle: return "".join(line.strip() for line in handle if not line.startswith(">")).upper()

def scored_arrays(population: str, arrays_path, contexts_path) -> pd.DataFrame:

    arrays = [array for array in read_trf_arrays(arrays_path) if array.full_copy_count == 2]
    keys = {(array.protein_accession, array.scope, array.search_unit_id) for array in arrays}
    contexts = pd.read_csv(contexts_path, sep="\t", dtype=str, usecols=["protein_accession", "scope", "search_unit_id", "database_fasta_path"])
    paths = contexts.set_index(["protein_accession", "scope", "search_unit_id"])["database_fasta_path"]
    transitions, lengths, rows = {}, {}, []

    for key in keys:

        sequence = _flank_sequence(paths[key])
        transitions[key] = transition_matrix_from_composition_row(composition_from_sequence(*key, sequence))
        lengths[key] = len(sequence)

    for array in arrays:

        key = (array.protein_accession, array.scope, array.search_unit_id)
        first, second = (copy.aligned_repeat.replace("-", "") for copy in array.full_copies)
        mismatches = Levenshtein.distance(first, second)
        item = EvidenceItem(array.trf_array_key, *key, "tandem_array", (Arm(array.side, array.start, array.end),))
        rows.append({"population": population, "trf_array_key": array.trf_array_key, "protein_accession": array.protein_accession,
                     "scope": array.scope, "search_unit_id": array.search_unit_id, "side": array.side, "start": array.start,
                     "end": array.end, "period": array.period, "mismatches": mismatches, "sequence_length": lengths[key],
                     "counted": counts_evidence(REPEAT_SETS["tandem"], item),
                     "markov_evalue": tandem_evalue(transitions[key], lengths[key], 2, array.period, mismatches)})

    return pd.DataFrame(rows)

def _flank_key(units: pd.DataFrame) -> pd.Series:

    return pd.Series(list(zip(units["protein_accession"], "within_" + units["side"], units["search_unit_id"])), index=units.index)

def without_weak(units: pd.DataFrame, arrays: pd.DataFrame, all_arrays: pd.DataFrame) -> pd.DataFrame:

    occupied = f"occupied_{WINDOW}"
    near = all_arrays.loc[all_arrays["start"] < WINDOW]
    weak = set(arrays.loc[arrays["counted"] & (arrays["start"] < WINDOW) & (arrays["markov_evalue"] > cfg.STEP5_MAX_PAIR_EVALUE), "trf_array_key"])
    kept_keys = set(map(tuple, near.loc[~near["trf_array_key"].isin(weak), ["protein_accession", "scope", "search_unit_id"]].to_numpy()))
    all_keys = set(map(tuple, near[["protein_accession", "scope", "search_unit_id"]].to_numpy()))
    keys = _flank_key(units)
    tandem = units["repeat_set"] == "tandem"
    if not (units.loc[tandem, occupied].astype(bool) == keys[tandem].isin(all_keys)).all(): raise ValueError("Counted arrays do not reproduce the tandem set's occupancy.")
    flank_index = ["representative_unit_id", "group_accession", "protein_accession", "side"] if "group_accession" in units else ["protein_accession", "side"]
    wide = units.pivot_table(index=flank_index, columns="repeat_set", values=occupied)
    if not (wide["all"].astype(bool) == wide[["tandem", "direct", "inverted"]].any(axis=1)).all(): raise ValueError("The all set's occupancy is not the union of its kinds'.")
    result = units.copy()

    for repeat_set, pair_sets in PAIR_SETS.items():

        rows = result["repeat_set"] == repeat_set
        pairs = wide[list(pair_sets)].any(axis=1) if pair_sets else pd.Series(False, index=wide.index)
        pairs_occupied = result.loc[rows].join(pairs.rename("pairs"), on=flank_index)["pairs"]
        result.loc[rows, occupied] = (pairs_occupied | keys[rows].isin(kept_keys)).astype(int)

    return result

def counted_arrays(arrays_path) -> pd.DataFrame:

    rows = [{"trf_array_key": array.trf_array_key, "protein_accession": array.protein_accession, "scope": array.scope,
             "search_unit_id": array.search_unit_id, "start": array.start}
            for array in read_trf_arrays(arrays_path)
            if counts_evidence(REPEAT_SETS["tandem"], EvidenceItem(array.trf_array_key, array.protein_accession, array.scope, array.search_unit_id,
                                                                    "tandem_array", (Arm(array.side, array.start, array.end),)))]

    return pd.DataFrame(rows)

def _weak_count(arrays: pd.DataFrame, units: pd.DataFrame) -> int:

    weak = arrays.loc[arrays["counted"] & (arrays["start"] < WINDOW) & (arrays["markov_evalue"] > cfg.STEP5_MAX_PAIR_EVALUE)]
    keys = set(_flank_key(units))

    return int(sum(key in keys for key in zip(weak["protein_accession"], weak["scope"], weak["search_unit_id"])))

def main() -> None:

    paths = panel.panel_paths(enrichment_cfg.BACKGROUND_PANEL)
    focal_arrays = scored_arrays("focal", cfg.STEP3_TRF_ARRAYS_PATH, cfg.STEP3_CONTEXTS_PATH)
    background_arrays = scored_arrays("background", paths["trf_arrays_path"], paths["contexts_path"])
    arrays = pd.concat([focal_arrays, background_arrays], ignore_index=True)
    arrays.to_csv(cfg.STEP5_TWO_COPY_ARRAY_EVALUE_PATH, sep="\t", index=False)

    print(f"TABLE {cfg.STEP5_TWO_COPY_ARRAY_EVALUE_PATH}")
    print(arrays.groupby("population").agg(arrays=("markov_evalue", "size"),
                                           above_cutoff=("markov_evalue", lambda values: int((values > cfg.STEP5_MAX_PAIR_EVALUE).sum()))))

    flanks = read_flanks(enrichment_cfg.BACKGROUND_PANEL)
    background = pd.read_csv(paths["units_path"], sep="\t")
    variants = {"as_counted": (flanks, background),
                "without_weak_two_copy": (without_weak(flanks, focal_arrays, counted_arrays(cfg.STEP3_TRF_ARRAYS_PATH)),
                                          without_weak(background, background_arrays, counted_arrays(paths["trf_arrays_path"])))}
    occupied = f"occupied_{WINDOW}"
    rows = []

    for repeat_set in (*PAIR_SETS, "all_strict"):

        for variant, (focal_units, background_units) in variants.items():

            if variant != "as_counted" and repeat_set not in PAIR_SETS: continue
            focal, reference = matched(focal_units, background_units, repeat_set, WINDOW)
            counted_focal, counted_reference = matched(flanks, background, repeat_set, WINDOW)

            for name in REPORTED_CLASSES:

                chosen = focal.loc[focal["hth_class"].isin(CLASSES[name])]
                as_counted = counted_focal.loc[counted_focal["hth_class"].isin(CLASSES[name])]
                rows.append({"repeat_set": repeat_set, "window": WINDOW, "hth_class": name, "variant": variant,
                             "samples": chosen["group_accession"].nunique(), "flanks": len(chosen),
                             "weak_arrays": _weak_count(focal_arrays, chosen), "background_weak_arrays": _weak_count(background_arrays, reference),
                             "positive_flanks": int(chosen[occupied].sum()), "background_positive_flanks": int(reference[occupied].sum()),
                             "lost_flanks": int(as_counted[occupied].sum() - chosen[occupied].sum()),
                             "background_lost_flanks": int(counted_reference[occupied].sum() - reference[occupied].sum()),
                             **_estimates(chosen, occupied)})
                print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(cfg.STEP5_TWO_COPY_ARRAY_SENSITIVITY_PATH, sep="\t", index=False)

    print(f"TABLE {cfg.STEP5_TWO_COPY_ARRAY_SENSITIVITY_PATH}")

if __name__ == "__main__": main()
