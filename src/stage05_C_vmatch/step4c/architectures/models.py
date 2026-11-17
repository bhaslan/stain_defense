from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, NamedTuple

ARCHITECTURE_FIELDS = tuple(
    (
        "step4c_architecture_id step3_snapshot_id "
        "protein_accession scope search_unit_id source_sequence_id left_cluster_id "
        "right_cluster_id paired_cluster_id represented_context_count "
        "represented_context_kind repeat_class trf_array_key copies copy_number unit_length "
        "minimum_spacer_length repeat_span_classification distance_to_cds "
        "sequence_covered_bases log_evalue"
    ).split()
)
REPEAT_METADATA_FIELDS = (
    "source_sequence_id",
    "left_cluster_id",
    "left_cluster_size",
    "right_cluster_id",
    "right_cluster_size",
    "paired_cluster_id",
    "pair_support",
)
Interval = tuple[int, int]

def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:length]}"

def source_match_id(row: Mapping[str, str]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return stable_id("match", payload, length=20)

def _required_int(row: Mapping[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing required integer field: {key}")
    return int(float(value))

def _optional_int(value: object) -> int | None:
    return None if value in {None, ""} else int(float(value))

def merge_intervals(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    merged: list[list[int]] = []
    for start, end in sorted(set(intervals)):
        if start >= end:
            raise ValueError(f"Invalid half-open interval: {(start, end)}")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    if not merged:
        raise ValueError("An arm must contain at least one supported block")
    return tuple(map(tuple, merged))

def overlap_length(first: Interval, second: Interval) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))

@dataclass(frozen=True, slots=True)
class Arm:

    side: str
    start: int
    end: int
    sequence: str = ""
    classification: str = "boundary_unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", self.sequence.upper())
        if self.start >= self.end:
            raise ValueError(f"Invalid half-open interval: {(self.start, self.end)}")
        if self.sequence and len(self.sequence) != self.span:
            raise ValueError(
                f"{self.side} arm spans {self.span} bases but has a {len(self.sequence)}-nt sequence"
            )

    @property
    def interval(self) -> Interval: return self.start, self.end

    @property
    def span(self) -> int: return self.end - self.start

    def reciprocal_overlap(self, other: Arm) -> float:
        overlap = (
            0 if self.side != other.side else overlap_length(self.interval, other.interval)
        )
        return min(overlap / self.span, overlap / other.span) if overlap else 0.0

class BoundarySide(NamedTuple):
    length: int
    status: str
    intergenic_length: int | None
    neighbor_start: int | None
    neighbor_stop: int | None

@dataclass(frozen=True, slots=True)
class BoundaryContext:

    central_start: int
    central_stop: int
    strand: str
    sides: Mapping[str, BoundarySide]

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> BoundaryContext:
        suffixes = ("boundary_status", "intergenic_length", "neighbor_cds_start", "neighbor_cds_stop")
        missing = [
            f"arm_{arm}_{suffix}"
            for arm in (1, 2)
            for suffix in suffixes
            if f"arm_{arm}_{suffix}" not in row
        ]
        if missing:
            raise ValueError(f"Missing boundary fields: {', '.join(missing)}")
        def side_values(number: int) -> tuple[str, BoundarySide]:
            side = row[f"arm_{number}_side"]
            return side, BoundarySide(
                _required_int(row, f"{side}_flank_length"),
                row[f"arm_{number}_boundary_status"],
                _optional_int(row[f"arm_{number}_intergenic_length"]),
                _optional_int(row[f"arm_{number}_neighbor_cds_start"]),
                _optional_int(row[f"arm_{number}_neighbor_cds_stop"]),
            )
        pairs = [side_values(number) for number in (1, 2)]
        if pairs[0][0] == pairs[1][0] and pairs[0][1] != pairs[1][1]:
            raise ValueError(f"Conflicting {pairs[0][0]} boundary metadata")
        strand = row["central_cds_strand"]
        if strand not in {"+", "-"}:
            raise ValueError(f"Invalid central CDS strand: {strand!r}")
        return cls(_required_int(row, "central_cds_start"), _required_int(row, "central_cds_stop"), strand, MappingProxyType(dict(pairs)))

    def _neighbor_interval(self, side: str) -> tuple[int, int] | None:
        boundary = self.sides[side]
        if boundary.neighbor_start is None or boundary.neighbor_stop is None:
            return None
        if (self.strand == "+" and side == "left") or (
            self.strand == "-" and side == "right"
        ):
            first = self.central_start - boundary.neighbor_stop
            last = self.central_start - boundary.neighbor_start
        else:
            first = boundary.neighbor_start - self.central_stop
            last = boundary.neighbor_stop - self.central_stop
        start = max(1, min(first, last))
        end = min(boundary.length, max(first, last))
        return None if start > end else (start, end)

    def classify(self, arm: Arm) -> str:
        boundary = self.sides[arm.side]
        if not boundary.status:
            return "boundary_unknown"
        start, end = arm.start + 1, arm.end
        neighbor = self._neighbor_interval(arm.side)
        if boundary.status in {
            "annotated",
            "no_cds_within_flank",
            "uncertain_boundary",
        }:
            intergenic_length = boundary.intergenic_length
            if (
                intergenic_length is None
                and boundary.status == "no_cds_within_flank"
            ):
                intergenic_length = boundary.length
            if intergenic_length is None:
                return "boundary_unknown"
            if end <= intergenic_length:
                return "intergenic"
            if start <= intergenic_length < end:
                return "boundary_spanning"
            if neighbor and start >= neighbor[0] and end <= neighbor[1]:
                return "neighbor_cds"
        elif boundary.status == "overlapping" and neighbor:
            if start >= neighbor[0] and end <= neighbor[1]:
                return "neighbor_cds"
        return "boundary_unknown"

@dataclass(frozen=True, slots=True)
class Repeat:

    metadata: Mapping[str, str]
    source_match_id: str
    protein_accession: str
    scope: str
    search_unit_id: str
    repeat_class: str
    stored_relation: str
    chromosome_relation: str
    block_order: str
    arm_1: Arm
    arm_2: Arm
    boundary: BoundaryContext
    distance: int
    evalue: float

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Repeat:
        raw = dict(row)
        source_id = source_match_id(raw)
        coordinates: list[Interval] = []
        sequences: list[str] = []
        for number in (1, 2):
            start = _required_int(raw, f"start_{number}_0based")
            end = _required_int(raw, f"end_{number}_0based") + 1
            length = _required_int(raw, f"length_{number}")
            sequence = raw.get(f"arm_{number}_sequence", "")
            if end - start != length or len(sequence) != length:
                raise ValueError(
                    f"{source_id} arm {number} has inconsistent coordinates, length, or sequence"
                )
            coordinates.append((start, end))
            sequences.append(sequence)
        repeat_class = raw.get("repeat_class", "")
        if repeat_class not in {
            "direct",
            "palindromic_inverted",
        }:
            raise ValueError(f"{source_id} has an invalid repeat class: {repeat_class!r}")
        match_type = raw.get("match_type", "")
        if (repeat_class == "palindromic_inverted") != (match_type == "P"):
            raise ValueError(f"{source_id} has inconsistent repeat class and match type")
        relation = "inverted" if repeat_class == "palindromic_inverted" else "direct"
        scope = raw["scope"]
        if repeat_class == "palindromic_inverted" and scope != "cross_flank":
            reported_overlap = _optional_int(raw.get("overlap_length"))
            if reported_overlap is None or bool(reported_overlap) != bool(overlap_length(*coordinates)):
                raise ValueError(f"{source_id} has an overlap class inconsistent with its coordinates")
        chromosome_relation = relation
        if scope == "cross_flank":
            chromosome_relation = "direct" if relation == "inverted" else "inverted"
            block_order = "left_then_right"
        else:
            block_order = "arm_1_then_arm_2" if coordinates[0][0] <= coordinates[1][0] else "arm_2_then_arm_1"
        return cls(
            MappingProxyType({field: raw.get(field, "") for field in REPEAT_METADATA_FIELDS}),
            source_id,
            raw["protein_accession"],
            scope,
            raw["search_unit_id"],
            repeat_class,
            relation,
            chromosome_relation,
            block_order,
            Arm(raw["arm_1_side"], *coordinates[0], sequences[0]),
            Arm(raw["arm_2_side"], *coordinates[1], sequences[1]),
            BoundaryContext.from_row(raw),
            _required_int(raw, "distance"),
            float(raw["evalue"]),
        )

    @property
    def unit_key(self) -> tuple[str, str, str]:
        return self.protein_accession, self.scope, self.search_unit_id

@dataclass(frozen=True, slots=True)
class Copy:

    side: str
    start: int
    end: int
    sequence: str
    classification: str
    mismatches: int | None = None
    support: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", self.sequence.upper())
        if not 0 <= self.start < self.end or len(self.sequence) != self.end - self.start:
            raise ValueError(f"Copy {self.side}:{self.start}-{self.end} has inconsistent coordinates or sequence")
        if (self.mismatches is None) != (self.support is None):
            raise ValueError("A copy carries mismatches and support together or not at all")

    @property
    def interval(self) -> Interval: return self.start, self.end

    def to_dict(self) -> dict[str, object]:
        return {"side": self.side, "start": self.start, "end": self.end, "sequence": self.sequence,
                "classification": self.classification, "mismatches": self.mismatches,
                "support": None if self.support is None else list(self.support)}

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> Copy:
        if set(row) != {"side", "start", "end", "sequence", "classification", "mismatches", "support"}:
            raise ValueError("Copy fields differ from the canonical catalog contract")
        support = row["support"]
        return cls(str(row["side"]), int(row["start"]), int(row["end"]), str(row["sequence"]),
                   str(row["classification"]), row["mismatches"], None if support is None else tuple(map(str, support)))

def _minimum_spacer(copies: tuple[Copy, ...]) -> int:
    return min(max(0, second.start - first.end) for first, second in zip(copies, copies[1:]))

def _covered_bases(copies: tuple[Copy, ...]) -> int:
    return sum(end - start for start, end in merge_intervals(copy.interval for copy in copies))

class Architecture:

    __slots__ = ("_row", "copies")

    def __init__(self, row: Mapping[str, object], copies: tuple[Copy, ...]) -> None:
        self._row = {field: row.get(field, "") for field in ARCHITECTURE_FIELDS}
        self.copies = copies

    @staticmethod
    def _architecture_id(row: Mapping[str, object], copies: tuple[Copy, ...]) -> str:
        identity = json.dumps(
            [[copy.side, copy.start, copy.end, copy.sequence] for copy in copies], separators=(",", ":")
        )
        return stable_id(
            "step4c_arch",
            row["protein_accession"],
            row["scope"],
            row["search_unit_id"],
            row["source_sequence_id"],
            row["repeat_class"],
            row.get("trf_array_key", ""),
            identity,
            length=20,
        )

    @classmethod
    def build(
        cls,
        copies: Iterable[Copy],
        step3_snapshot_id: str,
        *,
        protein_accession: str,
        scope: str,
        search_unit_id: str,
        source_sequence_id: str,
        left_cluster_id: str,
        right_cluster_id: str,
        paired_cluster_id: str,
        represented_context_count: int,
        repeat_class: str,
        repeat_span_classification: str,
        distance_to_cds: int,
        trf_array_key: str = "",
        unit_length: int | None = None,
        log_evalue: float | None = None,
    ) -> Architecture:

        copies = tuple(sorted(copies, key=lambda copy: copy.start))
        if scope not in {"within_left", "within_right"}:
            raise ValueError(f"An architecture cannot use scope {scope!r}")
        if len(copies) < 2 or len({copy.side for copy in copies}) != 1:
            raise ValueError("An architecture requires at least two copies on one side")
        if represented_context_count < 1:
            raise ValueError("An architecture requires positive context multiplicity")
        if (repeat_class == "tandem_array") != bool(trf_array_key):
            raise ValueError("Only tandem-array architectures may carry trf_array_key")
        row: dict[str, object] = {
            "step3_snapshot_id": step3_snapshot_id,
            "protein_accession": protein_accession,
            "scope": scope,
            "search_unit_id": search_unit_id,
            "source_sequence_id": source_sequence_id,
            "left_cluster_id": left_cluster_id,
            "right_cluster_id": right_cluster_id,
            "paired_cluster_id": paired_cluster_id,
            "represented_context_count": represented_context_count,
            "represented_context_kind": "sequence_cluster_size",
            "repeat_class": repeat_class,
            "trf_array_key": trf_array_key,
            "copies": json.dumps([copy.to_dict() for copy in copies], separators=(",", ":")),
            "copy_number": len(copies),
            "unit_length": "" if unit_length is None else unit_length,
            "minimum_spacer_length": _minimum_spacer(copies),
            "repeat_span_classification": repeat_span_classification,
            "distance_to_cds": distance_to_cds,
            "sequence_covered_bases": _covered_bases(copies),
            "log_evalue": "" if log_evalue is None else repr(log_evalue),
        }
        row["step4c_architecture_id"] = cls._architecture_id(row, copies)
        return cls(row, copies)

    def _value(self, field: str) -> str:
        return str(self._row[field])

    step4c_architecture_id = property(lambda self: self._value("step4c_architecture_id"))
    step3_snapshot_id = property(lambda self: self._value("step3_snapshot_id"))
    protein_accession = property(lambda self: self._value("protein_accession"))
    scope = property(lambda self: self._value("scope"))
    search_unit_id = property(lambda self: self._value("search_unit_id"))
    repeat_class = property(lambda self: self._value("repeat_class"))
    trf_array_key = property(lambda self: self._value("trf_array_key"))

    def to_row(self) -> dict[str, object]:
        return dict(self._row)

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Architecture:
        missing = [field for field in ARCHITECTURE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Missing architecture fields: {', '.join(missing)}")
        copies = tuple(Copy.from_dict(item) for item in json.loads(row["copies"]))
        if list(copies) != sorted(copies, key=lambda copy: copy.start):
            raise ValueError("Architecture copies are not in start order")
        if (row["repeat_class"] == "tandem_array") != bool(row["trf_array_key"]):
            raise ValueError("Only tandem-array architectures may carry trf_array_key")
        for field_name, expected in (
            ("copy_number", len(copies)),
            ("minimum_spacer_length", _minimum_spacer(copies)),
            ("sequence_covered_bases", _covered_bases(copies)),
        ):
            if _required_int(row, field_name) != expected:
                raise ValueError(f"{field_name} disagrees with copies")
        if row["unit_length"] and any(copy.end - copy.start != int(row["unit_length"]) for copy in copies):
            raise ValueError("unit_length disagrees with copies")
        if cls._architecture_id(row, copies) != row["step4c_architecture_id"]:
            raise ValueError("Architecture ID disagrees with canonical copy content")
        return cls(row, copies)

def _read_rows(
    path: str | Path, expected_fields: tuple[str, ...] | None = None
) -> Iterator[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if expected_fields is not None and tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{path.name} header differs from the canonical contract")
        yield from reader

def read_repeats(path: str | Path) -> Iterator[Repeat]:
    return map(Repeat.from_row, _read_rows(path))

def read_architectures(path: str | Path) -> Iterator[Architecture]:
    return map(Architecture.from_row, _read_rows(path, ARCHITECTURE_FIELDS))

def read_audit_records(path: str | Path) -> Iterator[dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Audit line {line_number} is not a JSON object")
            yield record
