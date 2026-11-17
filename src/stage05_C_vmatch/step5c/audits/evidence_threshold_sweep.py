from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.repeat_architecture_inference.evalue_markov import BASES, stationary_distribution
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.evidence import (
    PAIR_REPEAT_CLASSES,
    EvidenceItem,
    read_evidence,
    read_unit_composition,
    transition_matrix_from_composition_row,
)

from ..measures import counts_evidence, union_length

CUTOFFS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)

UnitKey = tuple[str, str, str]

@dataclass(frozen=True, slots=True)
class CutoffStats:

    label: str
    pairs: int
    pair_share: float
    pair_share_direct: float
    pair_share_inverted: float
    bases: int
    bases_share: float
    occupied: int
    occupied_share: float
    chance_pairs: float | None
    chance_share: float | None

def _primary_set() -> cfg.RepeatSet:

    return next(repeat_set for repeat_set in cfg.STEP5_REPEAT_SETS
                if repeat_set.name == cfg.STEP5_PRIMARY_REPEAT_SET)

def _unit_key(item: EvidenceItem) -> UnitKey:

    return item.protein_accession, item.scope, item.search_unit_id

def _entropy_rate(row: dict[str, object]) -> float:

    transition = transition_matrix_from_composition_row(row)
    distribution = stationary_distribution(transition)
    rate = 0.0

    for first in BASES:

        for second in BASES:

            probability = transition[first][second]

            if probability > 0.0: rate -= distribution[first] * probability * math.log2(probability)

    return rate

def _at_content(row: dict[str, object]) -> float:

    acgt = sum(int(row[f"count_{base}"]) for base in BASES)

    if acgt == 0: raise ValueError(f"Unit {row['protein_accession']} has zero ACGT counts")

    return (int(row["count_A"]) + int(row["count_T"])) / acgt

def _pair_dust(item: EvidenceItem) -> float:

    if item.arm_dust is None: raise ValueError(f"Pair {item.evidence_id} lacks DUST scores")

    return max(item.arm_dust)

def _quintile_edges(values: np.ndarray) -> np.ndarray:

    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for index in range(1, len(edges)):

        if edges[index] <= edges[index - 1]: edges[index] = np.nextafter(edges[index - 1], float("inf"))

    return edges

def _quintile_index(value: float, edges: np.ndarray) -> int:

    return int(np.searchsorted(edges[1:-1], value, side="right"))

def _items_at_cutoff(base_pairs: tuple[EvidenceItem, ...],
                     base_arrays: tuple[EvidenceItem, ...],
                     cutoff: float | None) -> tuple[EvidenceItem, ...]:

    if cutoff is None: return base_pairs + base_arrays

    surviving_pairs = tuple(item for item in base_pairs if item.markov_evalue <= cutoff)

    return surviving_pairs + base_arrays

def _pair_share(base_pairs: tuple[EvidenceItem, ...], cutoff: float | None, repeat_class: str | None) -> float:

    subset = tuple(item for item in base_pairs
                   if repeat_class is None or item.repeat_class == repeat_class)

    if not subset: return float("nan")

    if cutoff is None: return 1.0

    return sum(1 for item in subset if item.markov_evalue <= cutoff) / len(subset)

def _evidence_bases(items: tuple[EvidenceItem, ...]) -> int:

    unit_spans: dict[UnitKey, list[tuple[int, int]]] = defaultdict(list)

    for item in items:

        for stretch in item.stretches: unit_spans[_unit_key(item)].append((stretch.start, stretch.end))

    return sum(union_length(spans) for spans in unit_spans.values())

def _occupied_units(items: tuple[EvidenceItem, ...]) -> int:

    return len({_unit_key(item) for item in items})

def _cutoff_stats(label: str,
                  base_pairs: tuple[EvidenceItem, ...],
                  base_arrays: tuple[EvidenceItem, ...],
                  reference_bases: int,
                  reference_occupied: int,
                  searches: int | None,
                  cutoff: float | None) -> CutoffStats:

    items = _items_at_cutoff(base_pairs, base_arrays, cutoff)
    surviving_pairs = tuple(item for item in items if item.repeat_class in PAIR_REPEAT_CLASSES)
    bases = _evidence_bases(items)
    occupied = _occupied_units(items)
    chance_pairs = None if cutoff is None or searches is None else cutoff * searches
    chance_share = None

    if chance_pairs is not None and surviving_pairs:

        chance_share = chance_pairs / len(surviving_pairs)

    return CutoffStats(label, len(surviving_pairs),
                       _pair_share(base_pairs, cutoff, None),
                       _pair_share(base_pairs, cutoff, "direct"),
                       _pair_share(base_pairs, cutoff, "palindromic_inverted"),
                       bases, bases / reference_bases if reference_bases else float("nan"),
                       occupied, occupied / reference_occupied if reference_occupied else float("nan"),
                       chance_pairs, chance_share)

def _format_share(value: float) -> str:

    return "—" if math.isnan(value) else f"{100.0 * value:6.2f}%"

def _format_count(value: float | None) -> str:

    if value is None: return "—"

    return f"{value:,.0f}"

def _print_block(title: str, rows: list[CutoffStats]) -> None:

    print(f"\n{title}")
    print("cutoff      pairs   pair%  direct% invert%    bases  base%  units unit%  chance  chance%")
    print("-" * 88)

    for row in rows:

        chance_share = "—" if row.chance_share is None else f"{100.0 * row.chance_share:6.2f}%"

        print(f"{row.label:>10} {row.pairs:7,} {_format_share(row.pair_share)} "
              f"{_format_share(row.pair_share_direct)} {_format_share(row.pair_share_inverted)} "
              f"{row.bases:7,} {_format_share(row.bases_share)} {row.occupied:5,} "
              f"{_format_share(row.occupied_share)} {_format_count(row.chance_pairs):>7} {chance_share:>7}")

def _stratum_rows(base_pairs: tuple[EvidenceItem, ...],
                  base_arrays: tuple[EvidenceItem, ...],
                  membership: dict[UnitKey, int] | None,
                  pair_membership: dict[str, int] | None,
                  stratum_count: int) -> list[list[CutoffStats]]:

    rows_by_stratum: list[list[CutoffStats]] = [[] for _ in range(stratum_count)]
    grid = ((None, "none"), *[(cutoff, f"{cutoff:.0e}") for cutoff in CUTOFFS])

    for cutoff, label in grid:

        for stratum in range(stratum_count):

            if pair_membership is not None:

                stratum_pairs = tuple(item for item in base_pairs if pair_membership[item.evidence_id] == stratum)
                stratum_arrays = ()
                stratum_searches = None

            else:

                stratum_pairs = tuple(item for item in base_pairs if membership[_unit_key(item)] == stratum)
                stratum_arrays = tuple(item for item in base_arrays if membership[_unit_key(item)] == stratum)
                stratum_searches = 2 * sum(value == stratum for value in membership.values())

            stratum_reference = _items_at_cutoff(stratum_pairs, stratum_arrays, None)
            reference_bases_stratum = _evidence_bases(stratum_reference)
            reference_occupied_stratum = _occupied_units(stratum_reference)
            rows_by_stratum[stratum].append(_cutoff_stats(label, stratum_pairs, stratum_arrays,
                                                           reference_bases_stratum, reference_occupied_stratum,
                                                           stratum_searches, cutoff))

    return rows_by_stratum

def main() -> None:

    primary = _primary_set()
    unlimited = dataclasses.replace(primary, max_pair_evalue=math.inf)
    evidence = tuple(read_evidence(cfg.STEP4_EVIDENCE_PATH))
    composition = tuple(read_unit_composition(cfg.STEP4_UNIT_COMPOSITION_PATH))
    base_items = tuple(item for item in evidence if counts_evidence(unlimited, item))
    base_pairs = tuple(item for item in base_items if item.repeat_class in PAIR_REPEAT_CLASSES)
    base_arrays = tuple(item for item in base_items if item.repeat_class == "tandem_array")
    searches = 2 * len(composition)
    reference_items = _items_at_cutoff(base_pairs, base_arrays, None)
    reference_bases = _evidence_bases(reference_items)
    reference_occupied = _occupied_units(reference_items)

    composition_by_unit = {(row["protein_accession"], row["scope"], row["search_unit_id"]): row
                           for row in composition}
    entropy_values = np.array([_entropy_rate(composition_by_unit[key]) for key in composition_by_unit])
    at_values = np.array([_at_content(composition_by_unit[key]) for key in composition_by_unit])
    dust_values = np.array([_pair_dust(item) for item in base_pairs])
    entropy_edges = _quintile_edges(entropy_values)
    at_edges = _quintile_edges(at_values)
    dust_edges = _quintile_edges(dust_values)
    unit_entropy = {key: _quintile_index(_entropy_rate(row), entropy_edges) for key, row in composition_by_unit.items()}
    unit_at = {key: _quintile_index(_at_content(row), at_edges) for key, row in composition_by_unit.items()}
    pair_dust = {item.evidence_id: _quintile_index(_pair_dust(item), dust_edges) for item in base_pairs}
    grid = ((None, "none"), *[(cutoff, f"{cutoff:.0e}") for cutoff in CUTOFFS])
    overall = [_cutoff_stats(label, base_pairs, base_arrays, reference_bases, reference_occupied, searches, cutoff)
               for cutoff, label in grid]

    print("CONFIG")
    print("repeat_set", primary.name, "kinds", ",".join(primary.kinds))
    print("min_copy_length", cfg.STEP5_MIN_COPY_LENGTH, "proximal_cutoff", cfg.STEP5_PROXIMAL_CUTOFF)
    print("searches", f"{searches:,}", "(2 × composition rows)")

    print("\nBASE SIZES")
    print(f"base pairs overall     {len(base_pairs):7,}  direct {sum(item.repeat_class == 'direct' for item in base_pairs):,}"
          f"  inverted {sum(item.repeat_class == 'palindromic_inverted' for item in base_pairs):,}")
    print(f"TRF arrays             {len(base_arrays):7,}")
    print(f"composition units      {len(composition):7,}")
    print(f"no-cutoff bases        {reference_bases:7,}")
    print(f"no-cutoff occupied     {reference_occupied:7,}")

    _print_block("OVERALL", overall)

    entropy_rows = _stratum_rows(base_pairs, base_arrays, unit_entropy, None, 5)
    at_rows = _stratum_rows(base_pairs, base_arrays, unit_at, None, 5)
    dust_rows = _stratum_rows(base_pairs, base_arrays, None, pair_dust, 5)

    for title, edges, rows in (("ENTROPY-RATE QUINTILES (bits/base)", entropy_edges, entropy_rows),
                               ("AT-CONTENT QUINTILES", at_edges, at_rows),
                               ("DUST QUINTILES (max arm)", dust_edges, dust_rows)):

        print(f"\n{title}")
        print("edges:", "  ".join(f"Q{index + 1}[{edges[index]:.4f}, {edges[index + 1]:.4f}]"
                                   for index in range(5)))

        for quintile, stratum_rows in enumerate(rows, start=1):

            _print_block(f"  Q{quintile}", stratum_rows)

    strata = {"overall": overall}

    for name, rows in (("entropy", entropy_rows), ("at", at_rows), ("dust", dust_rows)):

        strata.update({f"{name}_q{quintile}": stratum_rows for quintile, stratum_rows in enumerate(rows, start=1)})

    table = pd.DataFrame([{"stratum": stratum, **dataclasses.asdict(row)} for stratum, rows in strata.items() for row in rows])
    table.to_csv(cfg.STEP5_EVIDENCE_THRESHOLD_SWEEP_PATH, sep="\t", index=False)

    print(f"\nTABLE {cfg.STEP5_EVIDENCE_THRESHOLD_SWEEP_PATH}")

if __name__ == "__main__": main()
