from __future__ import annotations

import csv
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Mapping

TRF_ARRAY_FIELDS = tuple(
    (
        "trf_array_key parent_trf_array_key protein_accession scope "
        "search_unit_id source_sequence_id side start_0based end_0based "
        "period consensus_size reported_copy_number full_copy_count score "
        "percent_matches percent_indels consensus_sequence aligned_copies_json"
    ).split()
)

_COORDINATE_SEQUENCES = re.compile(
    r"^\s*(\d+)\s+([ACGTN-]+(?:\s+[ACGTN-]+)*)\s*$", re.I
)
_CALL_HEADER = re.compile(r"Indices:\s*(\d+)--(\d+)\s+Score:\s*(\d+)")
_PERIOD_HEADER = re.compile(
    r"Period size:\s*(\d+)\s+Copynumber:\s*([0-9.]+)\s+"
    r"Consensus size:\s*(\d+)"
)

SPAN_SCORE_SMALLEST_PERIOD_POLICY = "span_score_smallest_period_v1"
DEFAULT_TRF_TAU_SPAN = 0.20
DEFAULT_TRF_TAU_SCORE = 0.30
DEFAULT_TRF_MIN_RECIPROCAL_OVERLAP = 0.70
_TRF_UNIT_FIELDS = ("protein_accession", "scope", "search_unit_id", "side")

def trf_array_key(
    protein_accession: str,
    search_unit_id: str,
    side: str,
    start: int,
    end: int,
    period: int,
) -> str:

    return (
        f"{protein_accession}/{search_unit_id}/"
        f"{side}:{start}-{end}:p{period}"
    )

@dataclass(frozen=True, slots=True)
class TrfAlignedCopy:

    start: int
    end: int
    consensus_start: int
    consensus_end: int
    aligned_repeat: str
    aligned_consensus: str
    is_full: bool

    def __post_init__(self) -> None:
        repeat = self.aligned_repeat.upper()
        consensus = self.aligned_consensus.upper()
        object.__setattr__(self, "aligned_repeat", repeat)
        object.__setattr__(self, "aligned_consensus", consensus)
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid TRF copy interval: {(self.start, self.end)}")
        if len(repeat) != len(consensus):
            raise ValueError("TRF aligned repeat and consensus lengths disagree")
        if len(repeat.replace("-", "")) != self.end - self.start:
            raise ValueError("TRF aligned repeat disagrees with its source interval")
        if self.consensus_start < 1 or self.consensus_end < self.consensus_start:
            raise ValueError("Invalid TRF consensus coordinates")

    @property
    def interval(self) -> tuple[int, int]:
        return self.start, self.end

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "consensus_start": self.consensus_start,
            "consensus_end": self.consensus_end,
            "aligned_repeat": self.aligned_repeat,
            "aligned_consensus": self.aligned_consensus,
            "is_full": self.is_full,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> TrfAlignedCopy:
        return cls(
            int(row["start"]),
            int(row["end"]),
            int(row["consensus_start"]),
            int(row["consensus_end"]),
            str(row["aligned_repeat"]),
            str(row["aligned_consensus"]),
            bool(row["is_full"]),
        )

@dataclass(frozen=True, slots=True)
class TrfArray:

    trf_array_key: str
    parent_trf_array_key: str
    protein_accession: str
    scope: str
    search_unit_id: str
    source_sequence_id: str
    side: str
    start: int
    end: int
    period: int
    consensus_size: int
    reported_copy_number: float
    full_copy_count: int
    score: int
    percent_matches: int
    percent_indels: int
    consensus_sequence: str
    aligned_copies: tuple[TrfAlignedCopy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "consensus_sequence", self.consensus_sequence.upper())
        if self.scope not in {"within_left", "within_right"}:
            raise ValueError(f"TRF arrays require a within-flank scope: {self.scope}")
        if self.side not in {"left", "right"}:
            raise ValueError(f"Invalid TRF side: {self.side}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid TRF array interval: {(self.start, self.end)}")
        if self.period < 1 or self.consensus_size < 1:
            raise ValueError("TRF period and consensus size must be positive")
        if not self.aligned_copies:
            raise ValueError("A TRF array requires aligned structural segments")
        if len(self.consensus_sequence) != self.consensus_size:
            raise ValueError("TRF consensus sequence length disagrees with metadata")
        if any(
            copy.start < self.start or copy.end > self.end
            for copy in self.aligned_copies
        ):
            raise ValueError("A TRF copy lies outside its array")
        if any(
            copy.consensus_end > self.consensus_size
            for copy in self.aligned_copies
        ):
            raise ValueError("A TRF copy lies outside its consensus")
        if any(
            copy.aligned_consensus.replace("-", "")
            != self.consensus_sequence[
                copy.consensus_start - 1 : copy.consensus_end
            ]
            for copy in self.aligned_copies
        ):
            raise ValueError("TRF copy-to-consensus alignment disagrees with consensus")
        if (
            self.aligned_copies[0].start != self.start
            or self.aligned_copies[-1].end != self.end
            or any(
                first.end != second.start
                for first, second in zip(
                    self.aligned_copies,
                    self.aligned_copies[1:],
                )
            )
        ):
            raise ValueError("TRF aligned copies do not tile the array span")
        observed_full = sum(copy.is_full for copy in self.aligned_copies)
        if observed_full != self.full_copy_count:
            raise ValueError("TRF full-copy count disagrees with aligned copies")
        expected_key = trf_array_key(
            self.protein_accession,
            self.search_unit_id,
            self.side,
            self.start,
            self.end,
            self.period,
        )
        if self.trf_array_key != expected_key:
            raise ValueError(
                f"TRF array key {self.trf_array_key!r} != {expected_key!r}"
            )

    @property
    def interval(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def full_copies(self) -> tuple[TrfAlignedCopy, ...]:
        return tuple(copy for copy in self.aligned_copies if copy.is_full)

    @property
    def partial_copies(self) -> tuple[TrfAlignedCopy, ...]:
        return tuple(copy for copy in self.aligned_copies if not copy.is_full)

    @property
    def span(self) -> int:
        return self.end - self.start

    def to_row(self) -> dict[str, object]:
        return {
            "trf_array_key": self.trf_array_key,
            "parent_trf_array_key": self.parent_trf_array_key,
            "protein_accession": self.protein_accession,
            "scope": self.scope,
            "search_unit_id": self.search_unit_id,
            "source_sequence_id": self.source_sequence_id,
            "side": self.side,
            "start_0based": self.start,
            "end_0based": self.end,
            "period": self.period,
            "consensus_size": self.consensus_size,
            "reported_copy_number": f"{self.reported_copy_number:g}",
            "full_copy_count": self.full_copy_count,
            "score": self.score,
            "percent_matches": self.percent_matches,
            "percent_indels": self.percent_indels,
            "consensus_sequence": self.consensus_sequence,
            "aligned_copies_json": json.dumps(
                [copy.to_dict() for copy in self.aligned_copies],
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> TrfArray:
        copies = json.loads(row["aligned_copies_json"])
        return cls(
            row["trf_array_key"],
            row.get("parent_trf_array_key", ""),
            row["protein_accession"],
            row["scope"],
            row["search_unit_id"],
            row["source_sequence_id"],
            row["side"],
            int(row["start_0based"]),
            int(row["end_0based"]),
            int(row["period"]),
            int(row["consensus_size"]),
            float(row["reported_copy_number"]),
            int(row["full_copy_count"]),
            int(row["score"]),
            int(row["percent_matches"]),
            int(row["percent_indels"]),
            row["consensus_sequence"],
            tuple(TrfAlignedCopy.from_dict(item) for item in copies),
        )

@dataclass(frozen=True, slots=True)
class _DatCall:
    start_1based: int
    end_1based: int
    period: int
    reported_copy_number: float
    consensus_size: int
    percent_matches: int
    percent_indels: int
    score: int
    consensus_sequence: str

    @property
    def join_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.start_1based,
            self.end_1based,
            self.period,
            self.consensus_size,
            self.score,
        )

def _read_dat_calls(path: Path) -> list[_DatCall]:
    calls: list[_DatCall] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 15 or not fields[0].isdigit():
                continue
            calls.append(
                _DatCall(
                    int(fields[0]),
                    int(fields[1]),
                    int(fields[2]),
                    float(fields[3]),
                    int(fields[4]),
                    int(fields[5]),
                    int(fields[6]),
                    int(fields[7]),
                    fields[13].upper(),
                )
            )
    return calls

def _plain_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<BR\s*/?>", "\n", raw, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", raw))

def _finalize_copy(
    rows: list[tuple[int, str, int, str]], consensus_size: int
) -> TrfAlignedCopy:
    start = min(source_start - 1 for source_start, _, _, _ in rows)
    end = max(
        source_start - 1 + len(source_text.replace("-", ""))
        for source_start, source_text, _, _ in rows
    )
    consensus_start = min(item[2] for item in rows)
    consensus_end = max(
        consensus_start_row + len(consensus_text.replace("-", "")) - 1
        for _, _, consensus_start_row, consensus_text in rows
    )
    aligned_repeat = "".join(item[1] for item in rows)
    aligned_consensus = "".join(item[3] for item in rows)
    return TrfAlignedCopy(
        start,
        end,
        consensus_start,
        consensus_end,
        aligned_repeat,
        aligned_consensus,
        consensus_start == 1 and consensus_end == consensus_size,
    )

def _split_alignment_token(
    source_token: str,
    consensus_token: str,
    source_start: int,
    consensus_start: int,
    consensus_size: int,
) -> list[tuple[int, str, int, str]]:

    if len(source_token) != len(consensus_token):
        raise ValueError("TRF aligned source and consensus token lengths disagree")
    if not 1 <= consensus_start <= consensus_size:
        raise ValueError("TRF alignment starts outside its consensus")
    segments: list[tuple[int, str, int, str]] = []
    segment_source_start = source_start
    segment_consensus_start = consensus_start
    source_cursor = source_start
    phase = consensus_start
    boundary_reached = False
    source_buffer: list[str] = []
    consensus_buffer: list[str] = []

    def finish_segment() -> None:
        if source_buffer:
            segments.append(
                (
                    segment_source_start,
                    "".join(source_buffer),
                    segment_consensus_start,
                    "".join(consensus_buffer),
                )
            )

    for source_base, consensus_base in zip(
        source_token,
        consensus_token,
        strict=True,
    ):
        if boundary_reached and consensus_base != "-":
            finish_segment()
            source_buffer.clear()
            consensus_buffer.clear()
            segment_source_start = source_cursor
            segment_consensus_start = 1
            boundary_reached = False
        source_buffer.append(source_base)
        consensus_buffer.append(consensus_base)
        if source_base != "-":
            source_cursor += 1
        if consensus_base != "-":
            if phase == consensus_size:
                phase = 1
                boundary_reached = True
            else:
                phase += 1
    finish_segment()
    return segments

def _alignment_copies(block: str, consensus_size: int) -> tuple[TrfAlignedCopy, ...]:
    lines = block.splitlines()
    paired: list[tuple[int, str, int, str]] = []
    index = 0
    while index + 1 < len(lines):
        source = _COORDINATE_SEQUENCES.match(lines[index])
        consensus = _COORDINATE_SEQUENCES.match(lines[index + 1])
        if source and consensus:
            source_tokens = source.group(2).upper().split()
            consensus_tokens = consensus.group(2).upper().split()
            if len(source_tokens) != len(consensus_tokens):
                raise ValueError(
                    "TRF source and consensus rows contain different copy counts"
                )
            source_start = int(source.group(1))
            consensus_start = int(consensus.group(1))
            for token_number, (source_token, consensus_token) in enumerate(
                zip(source_tokens, consensus_tokens, strict=True)
            ):
                paired.extend(
                    _split_alignment_token(
                        source_token,
                        consensus_token,
                        source_start,
                        consensus_start if token_number == 0 else 1,
                        consensus_size,
                    )
                )
                source_start += len(source_token.replace("-", ""))
            index += 2
        else:
            index += 1
    if not paired:
        raise ValueError("TRF alignment block contains no aligned copy rows")

    groups: list[list[tuple[int, str, int, str]]] = []
    for row in paired:
        if groups and row[2] > groups[-1][-1][2]:
            groups[-1].append(row)
        else:
            groups.append([row])
    return tuple(_finalize_copy(group, consensus_size) for group in groups)

def _read_html_alignments(
    path: Path,
) -> dict[tuple[int, int, int, int, int], list[tuple[TrfAlignedCopy, ...]]]:
    text = _plain_html(path)
    headers = list(_CALL_HEADER.finditer(text))
    alignments: dict[
        tuple[int, int, int, int, int], list[tuple[TrfAlignedCopy, ...]]
    ] = {}
    for number, header in enumerate(headers):
        stop = headers[number + 1].start() if number + 1 < len(headers) else len(text)
        block = text[header.start() : stop]
        period_header = _PERIOD_HEADER.search(block)
        if period_header is None:
            raise ValueError("TRF alignment block lacks period metadata")
        key = (
            int(header.group(1)),
            int(header.group(2)),
            int(period_header.group(1)),
            int(period_header.group(3)),
            int(header.group(3)),
        )
        alignments.setdefault(key, []).append(
            _alignment_copies(block, int(period_header.group(3)))
        )
    return alignments

def parse_trf_outputs(
    dat_path: str | Path,
    alignment_path: str | Path,
    *,
    protein_accession: str,
    scope: str,
    search_unit_id: str,
    source_sequence_id: str,
    side: str,
    source_sequence: str,
) -> tuple[TrfArray, ...]:

    dat_path = Path(dat_path)
    alignment_path = Path(alignment_path)
    sequence = source_sequence.upper()
    alignments = _read_html_alignments(alignment_path)
    arrays: list[TrfArray] = []
    for call in _read_dat_calls(dat_path):
        matching = alignments.get(call.join_key, [])
        if not matching:
            raise ValueError(f"TRF data call lacks an alignment: {call.join_key}")
        copies = matching.pop(0)
        start, end = call.start_1based - 1, call.end_1based
        for copy in copies:
            observed = copy.aligned_repeat.replace("-", "")
            if sequence[copy.start : copy.end] != observed:
                raise ValueError(
                    f"TRF aligned copy disagrees with source sequence at "
                    f"{side}:{copy.start}-{copy.end}"
                )
        arrays.append(
            TrfArray(
                trf_array_key(
                    protein_accession,
                    search_unit_id,
                    side,
                    start,
                    end,
                    call.period,
                ),
                "",
                protein_accession,
                scope,
                search_unit_id,
                source_sequence_id,
                side,
                start,
                end,
                call.period,
                call.consensus_size,
                call.reported_copy_number,
                sum(copy.is_full for copy in copies),
                call.score,
                call.percent_matches,
                call.percent_indels,
                call.consensus_sequence,
                copies,
            )
        )
    unmatched = [key for key, values in alignments.items() if values]
    if unmatched:
        raise ValueError(f"TRF alignments lack matching data calls: {unmatched[:3]}")
    return tuple(arrays)

def normalize_trf_arrays_legacy(
    arrays: Iterable[TrfArray],
    *,
    minimum_period: int,
    minimum_full_copies: int,
) -> tuple[TrfArray, ...]:

    candidates = [
        array
        for array in arrays
        if array.period >= minimum_period
        and array.full_copy_count >= minimum_full_copies
    ]
    candidates.sort(
        key=lambda array: (
            -array.span,
            -array.full_copy_count,
            -array.score,
            -array.percent_matches,
            array.period,
            array.start,
            array.end,
            array.trf_array_key,
        )
    )
    accepted: list[TrfArray] = []
    known_keys: set[str] = set()
    for candidate in candidates:
        if candidate.trf_array_key in known_keys:
            continue
        containers = [
            parent
            for parent in accepted
            if parent.start <= candidate.start and candidate.end <= parent.end
        ]
        if not containers:
            accepted.append(candidate)
            known_keys.add(candidate.trf_array_key)
            continue
        eligible_parents = [
            parent
            for parent in containers
            if candidate.period < parent.period
            and any(
                copy.start <= candidate.start and candidate.end <= copy.end
                for copy in parent.aligned_copies
            )
        ]
        if eligible_parents:
            eligible_parents.sort(
                key=lambda item: (item.span, item.period, item.trf_array_key)
            )
            parent = eligible_parents[0]
            accepted.append(
                replace(candidate, parent_trf_array_key=parent.trf_array_key)
            )
            known_keys.add(candidate.trf_array_key)
    return tuple(
        sorted(
            accepted,
            key=lambda item: (
                item.protein_accession,
                item.scope,
                item.search_unit_id,
                item.start,
                item.end,
                item.period,
            ),
        )
    )

def _copy_contained_in_parent_copy(child: TrfArray, parent: TrfArray) -> bool:
    return (
        child.period < parent.period
        and parent.start <= child.start
        and child.end <= parent.end
        and any(
            copy.start <= child.start and child.end <= copy.end
            for copy in parent.aligned_copies
        )
    )

def _trf_arrays_compete(
    first: TrfArray,
    second: TrfArray,
    minimum_full_copies: int,
    minimum_reciprocal_overlap: float,
) -> bool:

    if any(
        getattr(first, field) != getattr(second, field)
        for field in _TRF_UNIT_FIELDS
    ):
        return False
    if _copy_contained_in_parent_copy(
        first, second
    ) or _copy_contained_in_parent_copy(second, first):
        return False
    shared_start = max(first.start, second.start)
    shared_end = min(first.end, second.end)
    if shared_start >= shared_end:
        return False
    shared_span = shared_end - shared_start
    if (
        first.start <= second.start and second.end <= first.end
    ) or (
        second.start <= first.start and first.end <= second.end
    ):
        return True
    if all(
        shared_span / array.span >= minimum_reciprocal_overlap
        for array in (first, second)
    ):
        return True
    if first.period == second.period:
        return False
    smaller, larger = sorted((first, second), key=lambda array: array.period)
    if larger.period % smaller.period:
        return False
    return all(
        sum(
            copy.is_full
            and 2 * shared_start <= copy.start + copy.end < 2 * shared_end
            for copy in array.aligned_copies
        )
        >= minimum_full_copies
        for array in (first, second)
    )

@dataclass(frozen=True, slots=True)
class SpanScoreSmallestPeriodStrategy:

    tau_span: float = DEFAULT_TRF_TAU_SPAN
    tau_score: float = DEFAULT_TRF_TAU_SCORE
    minimum_reciprocal_overlap: float = DEFAULT_TRF_MIN_RECIPROCAL_OVERLAP

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_span < 1.0:
            raise ValueError("TRF span-loss tolerance must be in [0, 1)")
        if not 0.0 <= self.tau_score < 1.0:
            raise ValueError("TRF score-loss tolerance must be in [0, 1)")
        if not 0.0 <= self.minimum_reciprocal_overlap <= 1.0:
            raise ValueError("TRF reciprocal overlap must be in [0, 1]")

    def __call__(
        self,
        arrays: Iterable[TrfArray],
        *,
        minimum_period: int,
        minimum_full_copies: int,
    ) -> tuple[TrfArray, ...]:
        winners, _ = self._arbitrate(
            tuple(arrays), minimum_period, minimum_full_copies
        )
        return normalize_trf_arrays_legacy(
            winners,
            minimum_period=minimum_period,
            minimum_full_copies=minimum_full_copies,
        )

    def diagnose(
        self,
        arrays: Iterable[TrfArray],
        *,
        minimum_period: int,
        minimum_full_copies: int,
    ) -> list[dict[str, object]]:
        _, diagnostics = self._arbitrate(
            tuple(arrays), minimum_period, minimum_full_copies
        )
        return diagnostics

    def _arbitrate(
        self,
        arrays: tuple[TrfArray, ...],
        minimum_period: int,
        minimum_full_copies: int,
    ) -> tuple[tuple[TrfArray, ...], list[dict[str, object]]]:
        eligible = tuple(
            array
            for array in arrays
            if array.period >= minimum_period
            and array.full_copy_count >= minimum_full_copies
        )
        by_index = {id(array): index for index, array in enumerate(eligible)}
        adjacent: dict[int, set[int]] = defaultdict(set)
        for first_index, first in enumerate(eligible):
            for second_index in range(first_index + 1, len(eligible)):
                if _trf_arrays_compete(
                    first,
                    eligible[second_index],
                    minimum_full_copies,
                    self.minimum_reciprocal_overlap,
                ):
                    adjacent[first_index].add(second_index)
                    adjacent[second_index].add(first_index)

        components: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for index in range(len(eligible)):
            if index in seen or index not in adjacent:
                continue
            pending = [index]
            seen.add(index)
            component = []
            while pending:
                current = pending.pop()
                component.append(current)
                for neighbor in adjacent[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
            components.append(tuple(sorted(component)))
        components.sort(
            key=lambda component: tuple(
                sorted(eligible[index].trf_array_key for index in component)
            )
        )

        retained = set(range(len(eligible)))
        diagnostics: list[dict[str, object]] = []
        for group_number, component in enumerate(components, start=1):
            members = sorted(
                (eligible[index] for index in component),
                key=lambda array: array.trf_array_key,
            )
            fully_competing = all(
                _trf_arrays_compete(
                    first,
                    second,
                    minimum_full_copies,
                    self.minimum_reciprocal_overlap,
                )
                for first, second in combinations(members, 2)
            )
            if not fully_competing:
                for array in members:
                    diagnostics.append(
                        self._diagnostic(
                            array,
                            group_number,
                            retained=True,
                            reason="not_arbitrated_without_pairwise_competition",
                            passes_span=None,
                            passes_score=None,
                            selected_period=None,
                        )
                    )
                continue

            span_max = max(array.span for array in members)
            span_cutoff = (1.0 - self.tau_span) * span_max
            span_pass = [array for array in members if array.span >= span_cutoff]
            score_max = max(array.score / array.span for array in span_pass)
            score_cutoff = (1.0 - self.tau_score) * score_max
            score_pass = [
                array
                for array in span_pass
                if array.score / array.span >= score_cutoff
            ]
            selected_period = min(array.period for array in score_pass)
            period_pass = [
                array for array in score_pass if array.period == selected_period
            ]
            winner = min(
                period_pass,
                key=lambda array: (
                    -(array.score / array.span),
                    -array.span,
                    -array.score,
                    -array.percent_matches,
                    array.percent_indels,
                    array.start,
                    array.end,
                    array.trf_array_key,
                ),
            )
            for array in members:
                passes_span = array in span_pass
                passes_score = array in score_pass
                retained_call = array.trf_array_key == winner.trf_array_key
                reason = (
                    "retained_selected_call"
                    if retained_call
                    else "rejected_span_below_cutoff"
                    if not passes_span
                    else "rejected_score_per_base_below_cutoff"
                    if not passes_score
                    else "rejected_larger_period"
                    if array.period != selected_period
                    else "rejected_same_period_tie_break"
                )
                diagnostics.append(
                    self._diagnostic(
                        array,
                        group_number,
                        retained=retained_call,
                        reason=reason,
                        span_max=span_max,
                        span_cutoff=span_cutoff,
                        score_per_base_max=score_max,
                        score_per_base_cutoff=score_cutoff,
                        passes_span=passes_span,
                        passes_score=passes_score,
                        selected_period=selected_period,
                        selected_trf_array_key=winner.trf_array_key,
                    )
                )
                if not retained_call:
                    retained.discard(by_index[id(array)])
        diagnostics.sort(
            key=lambda item: (int(item["group"]), str(item["trf_array_key"]))
        )
        return tuple(eligible[index] for index in sorted(retained)), diagnostics

    def _diagnostic(
        self,
        array: TrfArray,
        group: int,
        *,
        retained: bool,
        reason: str,
        **details: object,
    ) -> dict[str, object]:
        return {
            "group": group,
            "same_array_rule": (
                "non_nested_containment_or_reciprocal_overlap_"
                "or_harmonic_shared_multi_copy_evidence"
            ),
            "tau_span": self.tau_span,
            "tau_score": self.tau_score,
            "minimum_reciprocal_overlap": self.minimum_reciprocal_overlap,
            "trf_array_key": array.trf_array_key,
            "start": array.start,
            "end": array.end,
            "span": array.span,
            "raw_score": array.score,
            "score_per_base": array.score / array.span,
            "period": array.period,
            "retained": retained,
            "reason": reason,
            **details,
        }

def normalize_trf_arrays(
    arrays: Iterable[TrfArray],
    *,
    minimum_period: int,
    minimum_full_copies: int,
    tau_span: float = DEFAULT_TRF_TAU_SPAN,
    tau_score: float = DEFAULT_TRF_TAU_SCORE,
    minimum_reciprocal_overlap: float = DEFAULT_TRF_MIN_RECIPROCAL_OVERLAP,
) -> tuple[TrfArray, ...]:

    return SpanScoreSmallestPeriodStrategy(
        tau_span=tau_span,
        tau_score=tau_score,
        minimum_reciprocal_overlap=minimum_reciprocal_overlap,
    )(
        arrays,
        minimum_period=minimum_period,
        minimum_full_copies=minimum_full_copies,
    )

def read_trf_arrays(path: str | Path) -> Iterator[TrfArray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TRF_ARRAY_FIELDS:
            raise ValueError(f"{path.name} header differs from the TRF array contract")
        for row in reader:
            yield TrfArray.from_row(row)
