from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.evidence import PAIR_REPEAT_CLASSES, EvidenceItem

from .persistence import write_tsv_atomic
from .recurrence import RecurrentPair, recurrent_pairs, unit_pieces

SCOPE = "within_flank"

ARCHITECTURE_KINDS = ("tandem_array", "direct")

SHARED_FIELDS = ("searched_units", "searched_bases", "near_cds_searched_bases",
                 "unit_pairs", "eligible", "recurrence_assessable", "evidence_recurrence_assessable")

RECURRENCE_FIELDS = ("recurrent_units", "recurrent_unit_pairs", "recurrence_prevalence")

COUNTS = {"arch": ("architectures", "occupied_units", "repeat_bases", "near_cds_repeat_bases",
                   "recurrent_pairs", *RECURRENCE_FIELDS),
          "evidence": ("items", "occupied_units", "repeat_bases", "near_cds_repeat_bases", *RECURRENCE_FIELDS)}

ARCH_RECURRENCE_FIELDS = ("recurrent_pairs", *RECURRENCE_FIELDS)

@dataclass(frozen=True, slots=True)
class Measure:

    name: str
    source: str
    numerator: str
    denominator: str
    label: str
    bounded: bool
    gate: str = ""

    @property
    def architecture_recurrence(self) -> bool:

        return self.source == "arch" and self.numerator in ARCH_RECURRENCE_FIELDS

    def column(self, repeat_set: cfg.RepeatSet) -> str:

        return field_column(repeat_set, self.source, self.name)

    def numerator_column(self, repeat_set: cfg.RepeatSet) -> str:

        return field_column(repeat_set, self.source, self.numerator)

    def denominator_column(self, repeat_set: cfg.RepeatSet) -> str:

        return field_column(repeat_set, self.source, self.denominator)

MEASURES = (Measure("occupancy", "arch", "occupied_units", "searched_units", "Architecture occupancy", True),
            Measure("yield", "arch", "architectures", "searched_units", "Architecture yield", False),
            Measure("multiplicity", "arch", "architectures", "occupied_units", "Architecture multiplicity", False),
            Measure("burden", "arch", "repeat_bases", "searched_bases", "Architecture burden", True),
            Measure("near_cds_burden", "arch", "near_cds_repeat_bases", "near_cds_searched_bases",
                    "Architecture near-CDS burden", True),
            Measure("recurrence_breadth", "arch", "recurrent_units", "searched_units",
                    "Architecture recurrence breadth", True, gate="recurrence_assessable"),
            Measure("recurrence_pair_fraction", "arch", "recurrent_unit_pairs", "unit_pairs",
                    "Architecture recurrence pair fraction", True, gate="recurrence_assessable"),
            Measure("occupancy", "evidence", "occupied_units", "searched_units", "Evidence occupancy", True),
            Measure("burden", "evidence", "repeat_bases", "searched_bases", "Evidence burden", True),
            Measure("near_cds_burden", "evidence", "near_cds_repeat_bases", "near_cds_searched_bases",
                    "Evidence near-CDS burden", True),
            Measure("recurrence_breadth", "evidence", "recurrent_units", "searched_units",
                    "Evidence recurrence breadth", True, gate="evidence_recurrence_assessable"))

ARCHITECTURE_FIELDS = ("repeat_set", "step5c_architecture_id", "protein_accession",
                       "repeat_class", "copy_number", "distance_to_cds", "recurrence_partners")

MEASURED_EVIDENCE_FIELDS = ("repeat_set", "evidence_id", "protein_accession",
                            "repeat_class", "distance_to_cds")

EVIDENCE_RECURRENCE_FIELDS = ("repeat_set", "group_accession", "side", "unit_1_id", "unit_2_id",
                              "matched_pieces", "decided_by_tandem", "matches")

UNIT_FIELDS = ("repeat_set", "group_accession", "protein_accession", "side", "representative_unit_id",
               "search_unit_id", "source_context_id", "represented_contexts",
               *(f"{field}_{window}" for window in cfg.STEP5_UNIT_WINDOWS
                 for field in ("searched_bases", "occupied", "repeat_bases")))

def field_column(repeat_set: cfg.RepeatSet, source: str, field: str) -> str:

    if field in SHARED_FIELDS: return field

    return f"{repeat_set.name}_{source}_{field}"

def sources(repeat_set: cfg.RepeatSet) -> tuple[str, ...]:

    if "arch" in repeat_set.sources and not set(repeat_set.kinds) & set(ARCHITECTURE_KINDS):

        raise ValueError(f"Repeat set {repeat_set.name!r} declares architecture measures but no architecture kind")

    return repeat_set.sources

def find_measure(source: str, name: str) -> Measure:

    for measure in MEASURES:
        if measure.source == source and measure.name == name: return measure

    raise ValueError(f"No declared measure for source={source!r} name={name!r}")

def presence_count_field(source: str) -> str:

    return "architectures" if source == "arch" else "items"

def measures_for(repeat_set: cfg.RepeatSet) -> tuple[Measure, ...]:

    active = sources(repeat_set)

    return tuple(measure for measure in MEASURES if measure.source in active)

def set_columns(repeat_set: cfg.RepeatSet) -> tuple[str, ...]:

    columns: list[str] = []

    for source in sources(repeat_set):

        for field in COUNTS[source]: columns.append(field_column(repeat_set, source, field))
        for measure in measures_for(repeat_set):
            if measure.source == source: columns.append(measure.column(repeat_set))

    return tuple(columns)

def architecture_recurrence_columns(repeat_set: cfg.RepeatSet) -> frozenset[str]:

    if "arch" not in sources(repeat_set): return frozenset()

    return frozenset((*(field_column(repeat_set, "arch", field) for field in ARCH_RECURRENCE_FIELDS),
                      *(measure.column(repeat_set) for measure in measures_for(repeat_set)
                        if measure.architecture_recurrence)))

def union_length(intervals: Iterable[tuple[int, int]]) -> int:

    covered = reach = 0

    for start, end in sorted(intervals):

        if end > reach: covered += end - max(start, reach)
        reach = max(reach, end)

    return covered

def _ratio(count: int | float, denominator: int | float) -> float | str:

    return count / denominator if denominator else ""

def _stretch_passes(length: int, end: int) -> bool:

    return length >= cfg.STEP5_MIN_COPY_LENGTH and end <= cfg.STEP5_PROXIMAL_CUTOFF

def _covered_within(spans: Iterable[tuple[int, int]], cutoff: int) -> int:

    return union_length((start, min(end, cutoff)) for start, end in spans if start < cutoff)

def _unit_bases(spans: Iterable[tuple[int, int]]) -> tuple[int, int]:

    spans = tuple(spans)

    return union_length(spans), _covered_within(spans, cfg.STEP5_NEAR_CDS_CUTOFF)

def _span_counts(unit_spans: Mapping[str, list[tuple[int, int]]]) -> dict[str, int]:

    bases = [_unit_bases(spans) for spans in unit_spans.values()]

    return {"occupied_units": len(unit_spans),
            "repeat_bases": sum(repeat for repeat, _ in bases),
            "near_cds_repeat_bases": sum(near for _, near in bases)}

def _apply_measures(repeat_set: cfg.RepeatSet,
                    source: str,
                    shared: Mapping[str, object],
                    counts: Mapping[str, int | float | str]) -> dict[str, object]:

    values = {**shared, **counts}
    row = {field_column(repeat_set, source, field): counts[field] for field in COUNTS[source] if field in counts}

    for measure in measures_for(repeat_set):

        if measure.source != source or measure.numerator not in counts: continue
        if measure.gate and not shared[measure.gate]: row[measure.column(repeat_set)] = ""
        else: row[measure.column(repeat_set)] = _ratio(values[measure.numerator], values[measure.denominator])

    return row

def _shared(units: list) -> dict[str, object]:

    count = len(units)
    same_side_pairs = sum(n * (n - 1) // 2 for n in Counter(unit.database_side for unit in units).values())

    return {"searched_units": count,
            "searched_bases": sum(unit.searched_bases_within(cfg.STEP5_PROXIMAL_CUTOFF) for unit in units),
            "near_cds_searched_bases": sum(unit.searched_bases_within(cfg.STEP5_NEAR_CDS_CUTOFF) for unit in units),
            "unit_pairs": same_side_pairs,
            "eligible": count >= cfg.STEP5_MIN_UNITS,
            "recurrence_assessable": same_side_pairs > 0,
            "evidence_recurrence_assessable": count >= cfg.STEP5_MIN_RECURRENCE_UNITS}

def _counts_in(repeat_set: cfg.RepeatSet, architecture, copies) -> bool:

    return (architecture.repeat_class in repeat_set.kinds
            and all(_stretch_passes(copy.complete_length, copy.end) for copy in copies))

def counts_evidence(repeat_set: cfg.RepeatSet, item: EvidenceItem) -> bool:

    if item.repeat_class not in repeat_set.kinds: return False
    if item.repeat_class in PAIR_REPEAT_CLASSES:
        if item.arm_overlap_length != 0: return False
        if item.markov_evalue > repeat_set.max_pair_evalue: return False

    return all(_stretch_passes(stretch.span, stretch.end) for stretch in item.stretches)

def _recurrence_counts(source: str,
                       shared: Mapping[str, object],
                       unit_pairs: Iterable[tuple[str, str]]) -> dict[str, int | str]:

    pairs = {frozenset(pair) for pair in unit_pairs}
    assessable = shared[find_measure(source, "recurrence_breadth").gate]

    return {"recurrent_units": len(set().union(*pairs)),
            "recurrent_unit_pairs": len(pairs),
            "recurrence_prevalence": int(bool(pairs)) if assessable else ""}

def _arch_counts(shared: Mapping[str, object],
                 counted,
                 copies,
                 recurrences) -> dict[str, int | float | str]:

    unit_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for architecture in counted:

        for copy_id in architecture.copy_ids:
            copy = copies[copy_id]
            unit_spans[architecture.representative_unit_id].append((copy.start, copy.end))

    counts = _span_counts(unit_spans)
    counts["architectures"] = len(counted)

    if recurrences is None: return counts

    counts["recurrent_pairs"] = len(recurrences)
    counts.update(_recurrence_counts("arch", shared, recurrences))

    return counts

def _evidence_spans(counted: list[EvidenceItem],
                    opportunity_by_upstream: Mapping[tuple[str, str, str], object]) -> dict[str, list[tuple[int, int]]]:

    unit_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for item in counted:
        opportunity = opportunity_by_upstream[(item.protein_accession, item.scope, item.search_unit_id)]
        for stretch in item.stretches:
            unit_spans[opportunity.representative_unit_id].append((stretch.start, stretch.end))

    return unit_spans

def _evidence_counts(shared: Mapping[str, object],
                     unit_spans: Mapping[str, list[tuple[int, int]]],
                     items: int,
                     pairs: Iterable[RecurrentPair]) -> dict[str, int | str]:

    counts = _span_counts(unit_spans)
    counts["items"] = items
    counts.update(_recurrence_counts("evidence", shared, ((pair.unit_1_id, pair.unit_2_id) for pair in pairs)))

    return counts

def searched_unit_rows(repeat_set: cfg.RepeatSet,
                       group: str,
                       units: Iterable,
                       unit_spans: Mapping[str, list[tuple[int, int]]]) -> list[dict[str, object]]:

    rows = []

    for unit in units:

        spans = tuple(unit_spans.get(unit.representative_unit_id, ()))
        row = {"repeat_set": repeat_set.name,
               "group_accession": group,
               "protein_accession": unit.protein_accession,
               "side": unit.database_side,
               "representative_unit_id": unit.representative_unit_id,
               "search_unit_id": unit.search_unit_id,
               "source_context_id": unit.source_context_id,
               "represented_contexts": unit.represented_context_count}

        for window in cfg.STEP5_UNIT_WINDOWS:

            searched, covered = unit.searched_bases_within(window), _covered_within(spans, window)
            if covered > searched: raise ValueError(f"Unit {unit.representative_unit_id} has {covered} covered bases within {window} bp but {searched} searched")
            row.update({f"searched_bases_{window}": searched, f"occupied_{window}": int(covered > 0), f"repeat_bases_{window}": covered})

        rows.append(row)

    return rows

def _recurrence_row(repeat_set: cfg.RepeatSet, group: str, pair: RecurrentPair) -> dict[str, object]:

    matches = ";".join(f"{match.piece_1.start}-{match.piece_1.end}:{match.piece_2.start}-{match.piece_2.end}:"
                       f"{match.similarity:.4f}:{match.orientation}" for match in pair.matches)

    return {"repeat_set": repeat_set.name,
            "group_accession": group,
            "side": pair.side,
            "unit_1_id": pair.unit_1_id,
            "unit_2_id": pair.unit_2_id,
            "matched_pieces": len(pair.matches),
            "decided_by_tandem": pair.decided_by_tandem,
            "matches": matches}

@dataclass(frozen=True, slots=True)
class Measured:

    rows: list[dict[str, object]]
    architecture_rows: list[dict[str, object]]
    evidence_rows: list[dict[str, object]]
    recurrence_rows: list[dict[str, object]]
    unit_rows: list[dict[str, object]]
    architecture_recurrence: bool

    def summary_fields(self, label_fields: Iterable[str]) -> tuple[str, ...]:

        measure_fields = [*SHARED_FIELDS]

        for repeat_set in cfg.STEP5_REPEAT_SETS:

            absent = frozenset() if self.architecture_recurrence else architecture_recurrence_columns(repeat_set)
            measure_fields.extend(column for column in set_columns(repeat_set) if column not in absent)

        if overlap := set(label_fields) & set(measure_fields):

            raise ValueError(f"Summary labels collide with Step 5C measures: {', '.join(sorted(overlap))}")

        return (*label_fields, *measure_fields)

def measure(groups: Mapping[str, Mapping[str, object]],
            group_of: Mapping[str, str],
            opportunities,
            copies,
            architectures,
            evidence_items,
            sequences,
            recurrences=None) -> Measured:

    if unknown := set(group_of.values()) - set(groups):

        raise ValueError(f"Proteins are mapped to undeclared groups: {', '.join(sorted(unknown))}")

    copies = {copy.step5c_copy_id: copy for copy in copies}

    if unknown := ({architecture.repeat_class for architecture in architectures if architecture.comparison_scope == SCOPE}
                   - set(cfg.STEP5_REPEAT_KINDS)):

        raise ValueError(f"Step 5C architectures carry repeat classes not in STEP5_REPEAT_KINDS: {', '.join(sorted(unknown))}")

    if unknown := {item.repeat_class for item in evidence_items} - set(cfg.STEP5_REPEAT_KINDS):

        raise ValueError(f"Step 4C evidence carries repeat classes not in STEP5_REPEAT_KINDS: {', '.join(sorted(unknown))}")

    units = defaultdict(list)

    for opportunity in opportunities:
        if opportunity.comparison_scope == SCOPE and opportunity.protein_accession in group_of:
            units[group_of[opportunity.protein_accession]].append(opportunity)

    within = [architecture for architecture in architectures
              if architecture.comparison_scope == SCOPE and architecture.protein_accession in group_of]
    within_evidence = tuple(item for item in evidence_items if item.protein_accession in group_of)
    with_recurrence = recurrences is not None
    pairs = ([(item.architecture_1_id, item.architecture_2_id) for item in recurrences if item.comparison_scope == SCOPE]
             if with_recurrence else [])
    opportunity_by_upstream = {opportunity.upstream_key: opportunity for opportunity in opportunities}
    shared = {group: _shared(units[group]) for group in groups}
    rows = {group: {**base, **shared[group]} for group, base in groups.items()}
    architecture_rows = []
    evidence_rows = []
    recurrence_rows = []
    unit_rows = []

    for repeat_set in cfg.STEP5_REPEAT_SETS:

        active_sources = sources(repeat_set)
        counted = {architecture.step5c_architecture_id: architecture for architecture in within
                   if "arch" in active_sources
                   and _counts_in(repeat_set, architecture, [copies[copy_id] for copy_id in architecture.copy_ids])}
        counted_evidence = tuple(item for item in within_evidence if counts_evidence(repeat_set, item))
        by_group, unit_pairs, partners = defaultdict(list), defaultdict(list), defaultdict(set)

        for architecture in counted.values(): by_group[group_of[architecture.protein_accession]].append(architecture)

        for first, second in pairs:

            if first not in counted or second not in counted: continue
            unit_pairs[group_of[counted[first].protein_accession]].append((counted[first].representative_unit_id,
                                                                           counted[second].representative_unit_id))
            partners[first].add(second)
            partners[second].add(first)

        evidence_by_group = defaultdict(list)

        for item in counted_evidence: evidence_by_group[group_of[item.protein_accession]].append(item)

        pieces = unit_pieces(opportunities, counted_evidence, sequences)

        for group, row in rows.items():

            if "arch" in active_sources:
                arch_counts = _arch_counts(shared[group], by_group[group], copies,
                                           unit_pairs[group] if with_recurrence else None)
                row.update(_apply_measures(repeat_set, "arch", shared[group], arch_counts))
            group_pairs = recurrent_pairs(units[group], pieces)
            unit_spans = _evidence_spans(evidence_by_group[group], opportunity_by_upstream)
            evidence_counts = _evidence_counts(shared[group], unit_spans, len(evidence_by_group[group]), group_pairs)
            row.update(_apply_measures(repeat_set, "evidence", shared[group], evidence_counts))
            recurrence_rows.extend(_recurrence_row(repeat_set, group, pair) for pair in group_pairs)
            unit_rows.extend(searched_unit_rows(repeat_set, group, units[group], unit_spans))

        architecture_rows.extend({"repeat_set": repeat_set.name,
                                  "step5c_architecture_id": architecture_id,
                                  "protein_accession": architecture.protein_accession,
                                  "repeat_class": architecture.repeat_class,
                                  "copy_number": architecture.copy_number,
                                  "distance_to_cds": min(copies[copy_id].distance_to_cds
                                                         for copy_id in architecture.copy_ids),
                                  **({"recurrence_partners": len(partners[architecture_id])} if with_recurrence else {})}
                                 for architecture_id, architecture in counted.items())

        evidence_rows.extend({"repeat_set": repeat_set.name,
                              "evidence_id": item.evidence_id,
                              "protein_accession": item.protein_accession,
                              "repeat_class": item.repeat_class,
                              "distance_to_cds": min(stretch.start + 1 for stretch in item.stretches)}
                             for item in counted_evidence)

    return Measured(list(rows.values()), architecture_rows, evidence_rows, recurrence_rows, unit_rows, with_recurrence)

def _per_set_counts(rows: list[Mapping[str, object]]) -> str:

    return ", ".join(f"{repeat_set.name} {sum(row['repeat_set'] == repeat_set.name for row in rows):,}"
                     for repeat_set in cfg.STEP5_REPEAT_SETS)

def write_measures(observed, logger):

    candidates = observed["candidates"]
    groups = {candidate.protein_accession: candidate.values for candidate in candidates}
    measured = measure(groups, {protein: protein for protein in groups}, observed["opportunities"], observed["copies"],
                       observed["architectures"], observed["evidence"], observed["sequences"], observed["recurrences"])

    write_tsv_atomic(cfg.STEP5_PROTEIN_SUMMARY_PATH, measured.summary_fields(candidates[0].field_order), measured.rows)
    write_tsv_atomic(cfg.STEP5_MEASURED_ARCHITECTURES_PATH, ARCHITECTURE_FIELDS, measured.architecture_rows)
    write_tsv_atomic(cfg.STEP5_MEASURED_EVIDENCE_PATH, MEASURED_EVIDENCE_FIELDS, measured.evidence_rows)
    write_tsv_atomic(cfg.STEP5_EVIDENCE_RECURRENCE_PATH, EVIDENCE_RECURRENCE_FIELDS, measured.recurrence_rows)

    logger.info("Step 5C measures complete: %s proteins; counted architectures %s; counted evidence %s; "
                "evidence recurrent unit pairs %s", f"{len(measured.rows):,}", _per_set_counts(measured.architecture_rows),
                _per_set_counts(measured.evidence_rows), _per_set_counts(measured.recurrence_rows))
