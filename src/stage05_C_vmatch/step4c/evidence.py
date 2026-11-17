from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from src.repeat_architecture_inference.evalue_markov import (
    BASES,
    repeat_evalue_markov,
    transition_matrix_from_counts,
)
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step3c.trf_evidence import TrfArray
from src.stage05_C_vmatch.step4c.architectures.models import Arm, Repeat, overlap_length
from src.stage05_C_vmatch.step4c.architectures.tandem import _boundary

WITHIN_FLANK_SCOPES = frozenset({"within_left", "within_right"})
PAIR_REPEAT_CLASSES = frozenset({"direct", "palindromic_inverted"})
DINUCLEOTIDE_FIELDS = tuple(f"count_{first}{second}" for first in BASES for second in BASES)
COMPOSITION_FIELDS = (
    "protein_accession",
    "scope",
    "search_unit_id",
    "sequence_length",
    "count_A",
    "count_C",
    "count_G",
    "count_T",
    "count_other",
    *DINUCLEOTIDE_FIELDS,
)
EVIDENCE_FIELDS = (
    "evidence_id",
    "protein_accession",
    "scope",
    "search_unit_id",
    "repeat_class",
    "stretches",
    "arm_overlap_length",
    "mismatches",
    "markov_evalue",
    "arm_1_dust",
    "arm_2_dust",
)

def composition_from_sequence(
    protein_accession: str,
    scope: str,
    search_unit_id: str,
    sequence: str,
) -> dict[str, object]:

    sequence = sequence.upper()
    monomer_counts = {base: 0 for base in BASES}
    dinucleotide_counts = {field: 0 for field in DINUCLEOTIDE_FIELDS}
    count_other = 0

    for symbol in sequence:

        if symbol in BASES:
            monomer_counts[symbol] += 1
        else:
            count_other += 1

    for first, second in zip(sequence, sequence[1:]):

        if first in BASES and second in BASES:
            dinucleotide_counts[f"count_{first}{second}"] += 1

    return {
        "protein_accession": protein_accession,
        "scope": scope,
        "search_unit_id": search_unit_id,
        "sequence_length": len(sequence),
        "count_A": monomer_counts["A"],
        "count_C": monomer_counts["C"],
        "count_G": monomer_counts["G"],
        "count_T": monomer_counts["T"],
        "count_other": count_other,
        **dinucleotide_counts,
    }

def transition_matrix_from_composition_row(row: Mapping[str, object]) -> dict[str, dict[str, float]]:

    counts = {
        first: {second: float(row[f"count_{first}{second}"]) for second in BASES}
        for first in BASES
    }

    return transition_matrix_from_counts(counts, pseudocount=cfg.STEP4_MARKOV_PSEUDOCOUNT)

def build_composition_rows(
    units: Mapping[tuple[str, str, str], Mapping[str, str]],
    representative_sequences: Mapping[tuple[str, str, str], str],
) -> tuple[dict[str, object], ...]:

    rows: list[dict[str, object]] = []

    for key in sorted(units):

        unit = units[key]
        if unit["scope"] not in WITHIN_FLANK_SCOPES:
            continue
        side = unit["database_side"]
        sequence = representative_sequences[(unit["protein_accession"], unit["search_unit_id"], side)]
        rows.append(composition_from_sequence(unit["protein_accession"], unit["scope"],
                                              unit["search_unit_id"], sequence))

    return tuple(rows)

def dust_score(sequence: str) -> float:

    sequence = sequence.upper()
    triplet_count = len(sequence) - 2
    if triplet_count < 2:
        raise ValueError(f"DUST requires at least two triplets, got length {len(sequence)}")
    counts = Counter(sequence[index : index + 3] for index in range(triplet_count))
    numerator = sum(count * (count - 1) for count in counts.values()) / 2

    return numerator / (triplet_count - 1)

def _classified_stretch(
    unit: Mapping[str, str],
    side: str,
    start: int,
    end: int,
) -> Arm:

    boundary = _boundary(unit, side)

    return Arm(side, start, end, classification=boundary.classify(Arm(side, start, end)))

def _pair_markov_evalue(
    repeat: Repeat,
    composition_row: Mapping[str, object],
    cache: dict[tuple[tuple[str, str, str], int, int], float],
) -> float:

    if repeat.distance > 0:
        raise ValueError(
            f"{repeat.source_match_id} has positive distance {repeat.distance}; "
            "within-flank pairs must be substitution-only"
        )
    if repeat.arm_1.span != repeat.arm_2.span:
        raise ValueError(
            f"{repeat.source_match_id} has unequal arm lengths "
            f"{repeat.arm_1.span} and {repeat.arm_2.span}"
        )
    mismatches = -repeat.distance
    cache_key = (repeat.unit_key, repeat.arm_1.span, mismatches)

    if cache_key not in cache:

        transition = transition_matrix_from_composition_row(composition_row)
        cache[cache_key] = float(repeat_evalue_markov(transition,
                                                      n=int(composition_row["sequence_length"]),
                                                      r=2, l=repeat.arm_1.span, k=mismatches))

    return cache[cache_key]

@dataclass(frozen=True, slots=True)
class EvidenceItem:

    evidence_id: str
    protein_accession: str
    scope: str
    search_unit_id: str
    repeat_class: str
    stretches: tuple[Arm, ...]
    arm_overlap_length: int | None = None
    mismatches: int | None = None
    markov_evalue: float | None = None
    arm_dust: tuple[float, float] | None = None

    def to_row(self) -> dict[str, object]:

        row: dict[str, object] = {
            "evidence_id": self.evidence_id,
            "protein_accession": self.protein_accession,
            "scope": self.scope,
            "search_unit_id": self.search_unit_id,
            "repeat_class": self.repeat_class,
            "stretches": json.dumps([{"side": stretch.side, "start": stretch.start,
                                      "end": stretch.end, "classification": stretch.classification}
                                     for stretch in self.stretches], separators=(",", ":")),
            "arm_overlap_length": "",
            "mismatches": "",
            "markov_evalue": "",
            "arm_1_dust": "",
            "arm_2_dust": "",
        }

        if self.repeat_class in PAIR_REPEAT_CLASSES:

            row["arm_overlap_length"] = self.arm_overlap_length
            row["mismatches"] = self.mismatches
            row["markov_evalue"] = repr(float(self.markov_evalue))
            row["arm_1_dust"] = repr(float(self.arm_dust[0]))
            row["arm_2_dust"] = repr(float(self.arm_dust[1]))

        return row

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> EvidenceItem:

        missing = [field for field in EVIDENCE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Missing evidence fields: {', '.join(missing)}")
        repeat_class = row["repeat_class"]
        stretches = tuple(Arm(str(item["side"]), int(item["start"]), int(item["end"]),
                              classification=str(item["classification"]))
                          for item in json.loads(row["stretches"]))
        is_pair = repeat_class in PAIR_REPEAT_CLASSES
        is_array = repeat_class == "tandem_array"
        if is_pair == is_array:
            raise ValueError(f"Invalid repeat class: {repeat_class!r}")
        expected_stretches = 2 if is_pair else 1
        if len(stretches) != expected_stretches:
            raise ValueError(
                f"{row['evidence_id']} has {len(stretches)} stretches; expected {expected_stretches}"
            )
        pair_fields = ("arm_overlap_length", "mismatches", "markov_evalue", "arm_1_dust", "arm_2_dust")

        if is_pair:

            if any(row[field] == "" for field in pair_fields):
                raise ValueError(f"{row['evidence_id']} lacks pair-only evidence fields")
            arm_dust = (float(row["arm_1_dust"]), float(row["arm_2_dust"]))

            return cls(row["evidence_id"], row["protein_accession"], row["scope"], row["search_unit_id"],
                       repeat_class, stretches, int(row["arm_overlap_length"]), int(row["mismatches"]),
                       float(row["markov_evalue"]), arm_dust)

        if any(row[field] != "" for field in pair_fields):
            raise ValueError(f"{row['evidence_id']} carries pair-only fields on a tandem array")

        return cls(row["evidence_id"], row["protein_accession"], row["scope"], row["search_unit_id"],
                   repeat_class, stretches)

def _evidence_from_pair(
    repeat: Repeat,
    unit: Mapping[str, str],
    composition_row: Mapping[str, object],
    cache: dict[tuple[tuple[str, str, str], int, int], float],
) -> EvidenceItem:

    stretches = (_classified_stretch(unit, repeat.arm_1.side, repeat.arm_1.start, repeat.arm_1.end),
                 _classified_stretch(unit, repeat.arm_2.side, repeat.arm_2.start, repeat.arm_2.end))

    return EvidenceItem(repeat.source_match_id, repeat.protein_accession, repeat.scope,
                        repeat.search_unit_id, repeat.repeat_class, stretches,
                        overlap_length(repeat.arm_1.interval, repeat.arm_2.interval), -repeat.distance,
                        _pair_markov_evalue(repeat, composition_row, cache),
                        (dust_score(repeat.arm_1.sequence), dust_score(repeat.arm_2.sequence)))

def _evidence_from_array(
    array: TrfArray,
    unit: Mapping[str, str],
) -> EvidenceItem:

    stretch = _classified_stretch(unit, array.side, array.start, array.end)

    return EvidenceItem(array.trf_array_key, array.protein_accession, array.scope,
                        array.search_unit_id, "tandem_array", (stretch,))

def build_evidence_items(
    retained_pairs: Iterable[Repeat],
    arrays: Iterable[TrfArray],
    units: Mapping[tuple[str, str, str], Mapping[str, str]],
    composition_rows: Iterable[Mapping[str, object]],
) -> tuple[EvidenceItem, ...]:

    composition_by_unit = {
        (str(row["protein_accession"]), str(row["scope"]), str(row["search_unit_id"])): row
        for row in composition_rows
    }
    within_pairs = tuple(repeat for repeat in retained_pairs
                         if repeat.scope in WITHIN_FLANK_SCOPES and repeat.repeat_class in PAIR_REPEAT_CLASSES)
    cache: dict[tuple[tuple[str, str, str], int, int], float] = {}
    pair_items = [_evidence_from_pair(repeat, units[repeat.unit_key],
                                      composition_by_unit[repeat.unit_key], cache)
                  for repeat in within_pairs]
    array_items = [_evidence_from_array(array, units[(array.protein_accession, array.scope, array.search_unit_id)])
                   for array in arrays]

    return tuple(sorted([*pair_items, *array_items],
                        key=lambda item: (item.protein_accession, item.scope,
                                          item.search_unit_id, item.evidence_id)))

def _read_rows(
    path: str | Path,
    expected_fields: tuple[str, ...],
) -> Iterator[dict[str, str]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8", newline="") as handle:

        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{path.name} header differs from the canonical contract")
        yield from reader

def read_evidence(path: str | Path) -> Iterator[EvidenceItem]:

    return map(EvidenceItem.from_row, _read_rows(path, EVIDENCE_FIELDS))

def read_unit_composition(path: str | Path) -> Iterator[dict[str, str]]:

    yield from _read_rows(path, COMPOSITION_FIELDS)
