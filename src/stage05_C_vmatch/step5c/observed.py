from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from Bio.Seq import Seq
from Levenshtein import ratio as levenshtein_ratio

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.architectures.models import (
    Architecture as SourceArchitecture,
    read_architectures,
)
from src.stage05_C_vmatch.step4c.evidence import EvidenceItem, read_evidence

from .models import (
    ArchitectureRecurrence,
    Candidate,
    CopyMatch,
    CopyMembership,
    NormalizedArchitecture,
    Opportunity,
    PhysicalCopy,
    PhysicalKey,
)
from .persistence import read_tsv, write_observed_outputs
from .recurrence import load_unit_sequences, optimal_assignment

_BOUNDARY_CLASSES = {
    "intergenic",
    "boundary_spanning",
    "neighbor_cds",
    "boundary_unknown",
}

_CANDIDATE_LABEL_FIELDS = (
    "protein_accession",
    "candidate_group",
    "selection_roles",
    "selected_protein",
    "eligibility_reason",
    "is_experimental_anchor",
    "anchor_name",
    "trnase_activity",
    "dna_binding_activity",
    "architecture",
    "hth_class",
    "hth_families",
    "n_hth",
    "architecture_source",
    "min_judge",
    "cluster_id",
    "representative_accession",
    "cluster_size",
    "interpro_architecture",
    "base_architecture",
    "has_local_hth",
    "local_hth_status",
    "hth_neighbor_fraction",
    "protein_length",
)

@dataclass(frozen=True, slots=True)
class _CopyFacts:
    step5c_copy_id: str
    physical_key: PhysicalKey
    protein_accession: str
    source_sequence_id: str
    side: str
    start: int
    end: int
    complete_sequence: str
    boundary_class: str
    component_cluster_id: str
    length_class: str
    distance_class: str

@dataclass(frozen=True, slots=True)
class _AssignmentCandidate:
    match: CopyMatch
    copy_1_id: str
    copy_2_id: str

def _stable_id(prefix: str, payload: object, length: int = 20) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"

def _required_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field, "")
    if value in {None, ""}:
        raise ValueError(f"Missing required integer field: {field}")
    return int(float(str(value)))

def _optional_int(value: object) -> int | None:
    return None if value in {None, ""} else int(float(str(value)))

def _optional_text(value: object) -> str | None:
    text = "" if value is None else str(value)
    return text or None

def _bin_label(
    value: int,
    bins: Sequence[tuple[int, int | None, str]],
    setting_name: str,
) -> str:
    for lower, upper, label in bins:
        if lower <= value and (upper is None or value <= upper):
            return str(label)
    raise ValueError(
        f"{setting_name} has no category covering the observed value {value}"
    )

def load_candidates() -> tuple[Candidate, ...]:

    _, rows = read_tsv(
        cfg.STEP5_CANDIDATES_PATH,
        required_fields=_CANDIDATE_LABEL_FIELDS,
    )
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for row in rows:
        protein_accession = row["protein_accession"]
        if protein_accession in seen:
            raise ValueError(f"Duplicate candidate protein: {protein_accession}")
        seen.add(protein_accession)
        values = {field: row[field] for field in _CANDIDATE_LABEL_FIELDS}
        candidates.append(
            Candidate(protein_accession, values, _CANDIDATE_LABEL_FIELDS)
        )
    if not candidates:
        raise ValueError("The complete candidate table is empty")
    return tuple(candidates)

def _validate_complete_step3_runs(opportunities: Sequence[Opportunity]) -> None:

    from src.stage05_C_vmatch.step3c.trf import (
        profile_summary as trf_profile_summary,
    )
    from src.stage05_C_vmatch.step3c.vmatch import (
        profile_summary as vmatch_profile_summary,
    )

    for caller, path, summary in (
        ("Vmatch", cfg.STEP3_RUNS_PATH, vmatch_profile_summary()),
        ("TRF", cfg.STEP3_TRF_RUNS_PATH, trf_profile_summary()),
    ):
        profile_contract: dict[tuple[str, str], str] = {}
        for row in summary:
            key = str(row["scope"]), str(row["profile"])
            repeat_class = str(row["repeat_class"])
            previous = profile_contract.setdefault(key, repeat_class)
            if previous != repeat_class:
                raise ValueError(
                    f"Conflicting {caller} Step 3C profile contract for {key}"
                )

        _, run_rows = read_tsv(
            path,
            required_fields=(
                "protein_accession",
                "scope",
                "search_unit_id",
                "profile",
                "repeat_class",
            ),
        )
        observed: dict[tuple[str, str, str, str], str] = {}
        for row in run_rows:
            if row["scope"] not in {"within_left", "within_right"}:
                continue
            profile_key = row["scope"], row["profile"]
            expected_repeat_class = profile_contract.get(profile_key)
            if expected_repeat_class is None:
                raise ValueError(
                    f"Unexpected {caller} Step 3C run profile: {profile_key}"
                )
            if row["repeat_class"] != expected_repeat_class:
                raise ValueError(
                    f"{caller} Step 3C run repeat class disagrees for "
                    f"{profile_key}"
                )
            key = (
                row["protein_accession"],
                row["scope"],
                row["search_unit_id"],
                row["profile"],
            )
            if key in observed:
                raise ValueError(f"Duplicate {caller} Step 3C unit/profile run: {key}")
            observed[key] = row["repeat_class"]

        expected = {
            (
                opportunity.protein_accession,
                opportunity.upstream_scope,
                opportunity.search_unit_id,
                profile,
            )
            for opportunity in opportunities
            for scope, profile in profile_contract
            if scope == opportunity.upstream_scope
        }
        observed_keys = set(observed)
        if expected != observed_keys:
            missing = sorted(expected - observed_keys)
            extra = sorted(observed_keys - expected)
            details = []
            if missing:
                details.append(f"missing={missing[:5]}")
            if extra:
                details.append(f"extra={extra[:5]}")
            raise ValueError(
                f"{caller} Step 3C run rows do not exactly cover all "
                "opportunity/profile pairs: " + "; ".join(details)
            )

def load_opportunities() -> tuple[Opportunity, ...]:

    required = (
        "protein_accession",
        "scope",
        "search_unit_id",
        "source_context_id",
        "source_sequence_id",
        "database_side",
        "query_side",
        "left_flank_length",
        "right_flank_length",
        "left_intergenic_length",
        "right_intergenic_length",
        "left_cluster_id",
        "left_cluster_size",
        "right_cluster_id",
        "right_cluster_size",
        "paired_cluster_id",
        "pair_support",
    )
    _, rows = read_tsv(cfg.STEP3_CONTEXTS_PATH, required_fields=required)
    opportunities: list[Opportunity] = []
    upstream_keys: set[tuple[str, str, str]] = set()
    representative_ids: set[str] = set()
    for row in rows:
        upstream_scope = row["scope"]
        if upstream_scope not in {"within_left", "within_right"}:
            continue
        representative_unit_id = _stable_id(
            "step5c_unit",
            [
                row["protein_accession"],
                upstream_scope,
                row["search_unit_id"],
            ],
        )
        opportunity = Opportunity(
            protein_accession=row["protein_accession"],
            upstream_scope=upstream_scope,
            comparison_scope="within_flank",
            search_unit_id=row["search_unit_id"],
            representative_unit_id=representative_unit_id,
            source_context_id=row["source_context_id"],
            source_sequence_id=row["source_sequence_id"],
            database_side=row["database_side"],
            query_side=_optional_text(row["query_side"]),
            left_flank_length=_required_int(row, "left_flank_length"),
            right_flank_length=_required_int(row, "right_flank_length"),
            left_intergenic_length=_optional_int(row["left_intergenic_length"]),
            right_intergenic_length=_optional_int(row["right_intergenic_length"]),
            left_cluster_id=_optional_text(row["left_cluster_id"]),
            right_cluster_id=_optional_text(row["right_cluster_id"]),
            left_cluster_size=_optional_int(row["left_cluster_size"]),
            right_cluster_size=_optional_int(row["right_cluster_size"]),
            paired_cluster_id=_optional_text(row["paired_cluster_id"]),
            pair_support=_optional_int(row["pair_support"]),
        )
        if opportunity.upstream_key in upstream_keys:
            raise ValueError(
                f"Duplicate Step 3C opportunity: {opportunity.upstream_key}"
            )
        if representative_unit_id in representative_ids:
            raise ValueError(
                "Step 5C representative-unit ID collision: "
                f"{representative_unit_id}"
            )
        upstream_keys.add(opportunity.upstream_key)
        representative_ids.add(representative_unit_id)
        opportunities.append(opportunity)
    opportunities.sort(key=lambda item: item.upstream_key)
    _validate_complete_step3_runs(opportunities)
    return tuple(opportunities)

def load_evidence(opportunities: Sequence[Opportunity]) -> tuple[EvidenceItem, ...]:

    upstream_keys = {opportunity.upstream_key for opportunity in opportunities}
    items = tuple(read_evidence(cfg.STEP4_EVIDENCE_PATH))
    for item in items:
        key = item.protein_accession, item.scope, item.search_unit_id
        if key not in upstream_keys:
            raise ValueError(f"Step 4C evidence item lacks an opportunity: {key}")

    return items

def _source_architecture_objects(
    source_architectures: Iterable[SourceArchitecture] | None,
) -> tuple[SourceArchitecture, ...]:
    if source_architectures is None:
        return tuple(read_architectures(cfg.STEP4_ARCHITECTURES_PATH))
    architectures = tuple(source_architectures)
    if any(not isinstance(item, SourceArchitecture) for item in architectures):
        raise TypeError("Step 5C requires typed Step 4C Architecture objects")
    return architectures

def _copy_class(copy_number: int) -> str:
    if copy_number == 2:
        return "pair"
    if 3 <= copy_number <= 4:
        return "small_array"
    if copy_number >= 5:
        return "large_array"
    raise ValueError("Architecture copy number must be at least two")

def _topology(copy_ids: tuple[str, ...]) -> str:
    if len(copy_ids) == 2:
        return "simple_pair"
    if len(copy_ids) >= 3:
        return "linear_array"
    raise ValueError("Architecture copy number must be at least two")

def _biological_relation(repeat_class: str) -> str:
    if repeat_class in {"tandem_array", "direct"}:
        return "direct"
    raise ValueError(
        f"Unknown repeat class for biological relation: {repeat_class!r}"
    )

def _structure_patterns(
    copy_ids: tuple[str, ...],
    copy_facts: Mapping[str, _CopyFacts],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    side_pattern = tuple(copy_facts[item].side for item in copy_ids)
    position_pattern = tuple(copy_facts[item].start + 1 for item in copy_ids)
    boundary_pattern = tuple(copy_facts[item].boundary_class for item in copy_ids)
    signatures: list[str] = []
    categories: list[str] = []
    for first_id, second_id in zip(copy_ids, copy_ids[1:]):
        first = copy_facts[first_id]
        second = copy_facts[second_id]
        if first.side != second.side:
            signatures.append("cross_side")
            categories.append("cross_side")
            continue
        overlap = max(0, min(first.end, second.end) - max(first.start, second.start))
        if overlap:
            signatures.append(f"overlap:{overlap}")
            categories.append("overlap")
            continue
        separation = max(first.start, second.start) - min(first.end, second.end)
        if separation == 0:
            signatures.append("touching")
            categories.append("touching")
        else:
            signatures.append(f"separation:{separation}")
            categories.append(
                _bin_label(
                    separation,
                    cfg.STEP5_EDGE_SEPARATION_BINS,
                    "STEP5_EDGE_SEPARATION_BINS",
                )
            )
    return (
        side_pattern,
        tuple(signatures),
        tuple(categories),
        position_pattern,
        boundary_pattern,
    )

def normalize_architectures(
    opportunities: Iterable[Opportunity],
    source_architectures: Iterable[SourceArchitecture] | None = None,
) -> tuple[tuple[PhysicalCopy, ...], tuple[NormalizedArchitecture, ...]]:

    opportunity_index: dict[tuple[str, str, str], Opportunity] = {}
    for opportunity in opportunities:
        if opportunity.upstream_key in opportunity_index:
            raise ValueError(f"Duplicate opportunity: {opportunity.upstream_key}")
        opportunity_index[opportunity.upstream_key] = opportunity

    physical_by_key: dict[PhysicalKey, _CopyFacts] = {}
    physical_by_id: dict[str, _CopyFacts] = {}
    memberships_by_copy: dict[
        str, dict[tuple[str, str, str], set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    normalized_by_id: dict[str, NormalizedArchitecture] = {}
    step4c_ids_by_architecture: dict[str, list[str]] = defaultdict(list)

    for source in _source_architecture_objects(source_architectures):
        row = source.to_row()
        upstream_key = (
            str(row["protein_accession"]),
            str(row["scope"]),
            str(row["search_unit_id"]),
        )
        opportunity = opportunity_index.get(upstream_key)
        if opportunity is None:
            raise ValueError(f"Step 4C architecture lacks an opportunity: {upstream_key}")
        expected_metadata = {
            "source_sequence_id": opportunity.source_sequence_id,
            "left_cluster_id": opportunity.left_cluster_id or "",
            "right_cluster_id": opportunity.right_cluster_id or "",
            "paired_cluster_id": opportunity.paired_cluster_id or "",
            "represented_context_count": str(opportunity.represented_context_count),
            "represented_context_kind": opportunity.represented_context_kind,
        }
        for field, expected in expected_metadata.items():
            if str(row.get(field, "")) != expected:
                raise ValueError(
                    f"Step 4C {field} disagrees with opportunity {upstream_key}"
                )

        copy_ids: list[str] = []
        for copy in source.copies:
            if copy.side not in opportunity.searched_sides:
                raise ValueError(
                    f"Step 4C copy uses unsearched side in {upstream_key}"
                )
            if copy.start < 0 or copy.end > opportunity.side_length(copy.side):
                raise ValueError(
                    f"Step 4C copy is outside its searched flank in {upstream_key}"
                )
            if copy.classification not in _BOUNDARY_CLASSES:
                raise ValueError(
                    f"Unknown Step 4C boundary class: {copy.classification!r}"
                )
            physical_key: PhysicalKey = (
                opportunity.protein_accession,
                opportunity.source_sequence_id,
                copy.side,
                copy.start,
                copy.end,
            )
            copy_id = _stable_id("step5c_copy", physical_key)
            component_cluster_id = opportunity.cluster_id(copy.side)
            if not component_cluster_id:
                raise ValueError(f"Copy lacks a side cluster in {upstream_key}")
            facts = _CopyFacts(
                step5c_copy_id=copy_id,
                physical_key=physical_key,
                protein_accession=opportunity.protein_accession,
                source_sequence_id=opportunity.source_sequence_id,
                side=copy.side,
                start=copy.start,
                end=copy.end,
                complete_sequence=copy.sequence,
                boundary_class=copy.classification,
                component_cluster_id=component_cluster_id,
                length_class=_bin_label(
                    copy.end - copy.start,
                    cfg.STEP5_COPY_LENGTH_BINS,
                    "STEP5_COPY_LENGTH_BINS",
                ),
                distance_class=_bin_label(
                    copy.start + 1,
                    cfg.STEP5_DISTANCE_BINS,
                    "STEP5_DISTANCE_BINS",
                ),
            )
            previous = physical_by_key.setdefault(physical_key, facts)
            if previous != facts:
                raise ValueError(
                    "Duplicate physical interval disagrees on sequence, boundary, "
                    f"or cluster facts: {physical_key}"
                )
            id_previous = physical_by_id.setdefault(copy_id, facts)
            if id_previous.physical_key != physical_key:
                raise ValueError(f"Step 5C physical copy ID collision: {copy_id}")
            copy_ids.append(copy_id)

        if len(set(copy_ids)) != len(copy_ids):
            raise ValueError(f"Step 4C architecture duplicates a physical interval: {upstream_key}")

        membership_key = (
            opportunity.comparison_scope,
            opportunity.upstream_scope,
            opportunity.representative_unit_id,
        )
        for copy_id in copy_ids:
            memberships_by_copy[copy_id][membership_key]

        repeat_class = str(row["repeat_class"])
        identity = {
            "protein_accession": opportunity.protein_accession,
            "scope": opportunity.comparison_scope,
            "representative_unit_id": opportunity.representative_unit_id,
            "repeat_class": repeat_class,
            "copy_ids": copy_ids,
        }
        architecture_id = _stable_id("step5c_arch", identity)
        side_pattern, spacing_signature, separation_categories, position_pattern, boundary_pattern = (
            _structure_patterns(tuple(copy_ids), physical_by_id)
        )
        architecture = NormalizedArchitecture(
            step5c_architecture_id=architecture_id,
            protein_accession=opportunity.protein_accession,
            comparison_scope=opportunity.comparison_scope,
            upstream_scope=opportunity.upstream_scope,
            representative_unit_id=opportunity.representative_unit_id,
            search_unit_id=opportunity.search_unit_id,
            source_sequence_id=opportunity.source_sequence_id,
            left_cluster_id=opportunity.left_cluster_id,
            right_cluster_id=opportunity.right_cluster_id,
            paired_cluster_id=opportunity.paired_cluster_id,
            represented_context_count=opportunity.represented_context_count,
            represented_context_kind=opportunity.represented_context_kind,
            copy_ids=tuple(copy_ids),
            copy_class=_copy_class(len(copy_ids)),
            topology=_topology(tuple(copy_ids)),
            biological_relation=_biological_relation(repeat_class),
            repeat_class=repeat_class,
            side_pattern=side_pattern,
            spacing_signature=spacing_signature,
            edge_separation_categories=separation_categories,
            position_pattern=position_pattern,
            boundary_pattern=boundary_pattern,
        )
        previous_architecture = normalized_by_id.setdefault(
            architecture_id, architecture
        )
        if previous_architecture != architecture:
            raise ValueError(
                "Duplicate normalized architecture identity disagrees on facts: "
                f"{architecture_id}"
            )
        step4c_ids_by_architecture[architecture_id].append(
            source.step4c_architecture_id
        )
        for copy_id in copy_ids:
            memberships_by_copy[copy_id][membership_key].add(architecture_id)

    architectures = tuple(
        sorted(
            (
                replace(
                    architecture,
                    step4c_architecture_ids=tuple(
                        sorted(set(step4c_ids_by_architecture[architecture.step5c_architecture_id]))
                    ),
                )
                for architecture in normalized_by_id.values()
            ),
            key=lambda item: (
                item.protein_accession,
                item.comparison_scope,
                item.representative_unit_id,
                item.step5c_architecture_id,
            ),
        )
    )
    copies = tuple(
        PhysicalCopy(
            step5c_copy_id=facts.step5c_copy_id,
            protein_accession=facts.protein_accession,
            source_sequence_id=facts.source_sequence_id,
            side=facts.side,
            start=facts.start,
            end=facts.end,
            complete_sequence=facts.complete_sequence,
            boundary_class=facts.boundary_class,
            component_cluster_id=facts.component_cluster_id,
            length_class=facts.length_class,
            distance_class=facts.distance_class,
            memberships=tuple(
                CopyMembership(
                    comparison_scope=membership_key[0],
                    upstream_scope=membership_key[1],
                    representative_unit_id=membership_key[2],
                    step5c_architecture_ids=tuple(sorted(architecture_ids)),
                )
                for membership_key, architecture_ids in sorted(
                    memberships_by_copy[facts.step5c_copy_id].items()
                )
            ),
        )
        for facts in sorted(
            physical_by_id.values(), key=lambda item: item.physical_key
        )
    )
    return copies, architectures

def _reverse_complement(sequence: str) -> str:
    return str(Seq(sequence).reverse_complement())

def build_copy_matches(
    copies: Iterable[PhysicalCopy],
    group_of: Mapping[str, str] | None = None,
) -> tuple[CopyMatch, ...]:

    copies = tuple(copies)
    if group_of is None: group_of = {copy.protein_accession: copy.protein_accession for copy in copies}
    grouped: dict[tuple[str, str], list[PhysicalCopy]] = defaultdict(list)
    units_by_copy_scope: dict[tuple[str, str], tuple[str, ...]] = {}
    for copy in copies:
        if copy.protein_accession not in group_of: continue
        by_scope: dict[str, set[str]] = defaultdict(set)
        for membership in copy.memberships:
            by_scope[membership.comparison_scope].add(
                membership.representative_unit_id
            )
        for scope, units in by_scope.items():
            grouped[(group_of[copy.protein_accession], scope)].append(copy)
            units_by_copy_scope[(copy.step5c_copy_id, scope)] = tuple(
                sorted(units)
            )

    matches_by_id: dict[str, CopyMatch] = {}
    threshold = float(cfg.STEP5_SIMILARITY_THRESHOLD)
    for (group_accession, scope), group in sorted(grouped.items()):
        group.sort(key=lambda item: item.step5c_copy_id)
        for first, second in combinations(group, 2):
            if first.side != second.side: continue
            first_units = units_by_copy_scope[(first.step5c_copy_id, scope)]
            second_units = units_by_copy_scope[(second.step5c_copy_id, scope)]
            eligible_unit_pairs = [
                (first_unit, second_unit)
                for first_unit in first_units
                for second_unit in second_units
                if first_unit != second_unit
            ]
            if not eligible_unit_pairs:
                continue
            direct = levenshtein_ratio(
                first.complete_sequence, second.complete_sequence
            )
            reverse = levenshtein_ratio(
                first.complete_sequence,
                _reverse_complement(second.complete_sequence),
            )
            similarity = max(direct, reverse)
            if similarity < threshold:
                continue
            orientation = (
                "direct"
                if direct > reverse
                else "reverse_complement"
                if reverse > direct
                else "ambiguous"
            )
            for first_unit, second_unit in sorted(eligible_unit_pairs):
                match_id = _stable_id(
                    "step5c_match",
                    [
                        group_accession,
                        scope,
                        first.step5c_copy_id,
                        second.step5c_copy_id,
                        first_unit,
                        second_unit,
                    ],
                )
                match = CopyMatch(
                    step5c_match_id=match_id,
                    protein_accession=group_accession,
                    comparison_scope=scope,
                    copy_1_id=first.step5c_copy_id,
                    copy_2_id=second.step5c_copy_id,
                    representative_unit_1_id=first_unit,
                    representative_unit_2_id=second_unit,
                    source_sequence_1_id=first.source_sequence_id,
                    source_sequence_2_id=second.source_sequence_id,
                    side_1=first.side,
                    side_2=second.side,
                    component_1_id=f"{first.side}:{first.component_cluster_id}",
                    component_2_id=f"{second.side}:{second.component_cluster_id}",
                    length_1=first.complete_length,
                    length_2=second.complete_length,
                    direct_similarity=direct,
                    reverse_complement_similarity=reverse,
                    similarity=similarity,
                    orientation=orientation,
                    shared_component_reuse=False,
                )
                previous = matches_by_id.setdefault(match_id, match)
                if previous != match:
                    raise ValueError(f"Step 5C copy-match ID collision: {match_id}")

    return tuple(
        sorted(
            matches_by_id.values(),
            key=lambda item: (
                item.protein_accession,
                item.comparison_scope,
                item.representative_unit_1_id,
                item.representative_unit_2_id,
                item.copy_1_id,
                item.copy_2_id,
            ),
        )
    )

def _optimal_assignment(
    candidates: Iterable[_AssignmentCandidate],
    copies: Mapping[str, PhysicalCopy],
) -> tuple[_AssignmentCandidate, ...]:

    candidate_by_pair: dict[tuple[str, str], _AssignmentCandidate] = {}
    for candidate in candidates:
        key = candidate.copy_1_id, candidate.copy_2_id
        previous = candidate_by_pair.get(key)
        if previous is None or (
            candidate.match.similarity,
            candidate.match.step5c_match_id,
        ) > (
            previous.match.similarity,
            previous.match.step5c_match_id,
        ):
            candidate_by_pair[key] = candidate
    ordered = sorted(
        candidate_by_pair.values(),
        key=lambda item: (
            copies[item.copy_1_id].physical_key,
            copies[item.copy_2_id].physical_key,
            item.match.step5c_match_id,
        ),
    )
    chosen = optimal_assignment([(item.copy_1_id, item.copy_2_id, item.match.similarity) for item in ordered])
    selected = [ordered[index] for index in chosen]
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                copies[item.copy_1_id].physical_key,
                copies[item.copy_2_id].physical_key,
            ),
        )
    )

def build_architecture_recurrences(
    copies: Iterable[PhysicalCopy],
    architectures: Iterable[NormalizedArchitecture],
    matches: Iterable[CopyMatch],
    group_of: Mapping[str, str] | None = None,
) -> tuple[ArchitectureRecurrence, ...]:

    copy_by_id = {item.step5c_copy_id: item for item in copies}
    architecture_by_id = {
        item.step5c_architecture_id: item for item in architectures
    }
    if group_of is None:
        group_of = {item.protein_accession: item.protein_accession for item in architecture_by_id.values()}
    membership_index: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for copy in copy_by_id.values():
        for membership in copy.memberships:
            membership_index[
                (
                    copy.step5c_copy_id,
                    membership.comparison_scope,
                    membership.representative_unit_id,
                )
            ].update(membership.step5c_architecture_ids)

    candidates_by_architecture_pair: dict[
        tuple[str, str], dict[tuple[str, str], _AssignmentCandidate]
    ] = defaultdict(dict)
    for match in matches:
        if not match.independent:
            continue
        first_architectures = membership_index[
            (
                match.copy_1_id,
                match.comparison_scope,
                match.representative_unit_1_id,
            )
        ]
        second_architectures = membership_index[
            (
                match.copy_2_id,
                match.comparison_scope,
                match.representative_unit_2_id,
            )
        ]
        for first_id in sorted(first_architectures):
            for second_id in sorted(second_architectures):
                if first_id == second_id:
                    continue
                if first_id < second_id:
                    pair = first_id, second_id
                    copy_pair = match.copy_1_id, match.copy_2_id
                else:
                    pair = second_id, first_id
                    copy_pair = match.copy_2_id, match.copy_1_id
                candidate = _AssignmentCandidate(
                    match, copy_pair[0], copy_pair[1]
                )
                previous = candidates_by_architecture_pair[pair].get(copy_pair)
                if previous is None or (
                    candidate.match.similarity,
                    candidate.match.step5c_match_id,
                ) > (
                    previous.match.similarity,
                    previous.match.step5c_match_id,
                ):
                    candidates_by_architecture_pair[pair][copy_pair] = candidate

    minimum_matches = int(cfg.STEP5_MIN_INDEPENDENT_MATCHES)
    recurrences: list[ArchitectureRecurrence] = []
    for (first_id, second_id), candidate_map in sorted(
        candidates_by_architecture_pair.items()
    ):
        first = architecture_by_id[first_id]
        second = architecture_by_id[second_id]
        group = group_of[first.protein_accession]
        if (
            group != group_of[second.protein_accession]
            or first.comparison_scope != second.comparison_scope
        ):
            raise ValueError("Copy match crossed a group or comparison scope")
        if first.representative_unit_id == second.representative_unit_id:
            raise ValueError("Copy match compared architectures from one unit")
        if len(candidate_map) < minimum_matches:
            continue
        selected = _optimal_assignment(candidate_map.values(), copy_by_id)
        if len(selected) < minimum_matches:
            continue
        selected_pairs = tuple(
            (item.copy_1_id, item.copy_2_id) for item in selected
        )
        selected_match_ids = tuple(
            item.match.step5c_match_id for item in selected
        )
        used_first = {pair[0] for pair in selected_pairs}
        used_second = {pair[1] for pair in selected_pairs}
        unmatched_first = tuple(
            copy_id for copy_id in first.copy_ids if copy_id not in used_first
        )
        unmatched_second = tuple(
            copy_id for copy_id in second.copy_ids if copy_id not in used_second
        )
        recurrence_id = _stable_id(
            "step5c_recur", [first_id, second_id]
        )
        orientations = tuple(item.match.orientation for item in selected)
        same_side_count = sum(
            copy_by_id[item.copy_1_id].side
            == copy_by_id[item.copy_2_id].side
            for item in selected
        )
        recurrences.append(
            ArchitectureRecurrence(
                step5c_recurrence_id=recurrence_id,
                protein_accession=group,
                comparison_scope=first.comparison_scope,
                architecture_1_id=first_id,
                architecture_2_id=second_id,
                representative_unit_1_id=first.representative_unit_id,
                representative_unit_2_id=second.representative_unit_id,
                selected_match_ids=selected_match_ids,
                selected_copy_pairs=selected_pairs,
                architecture_1_copy_number=first.copy_number,
                architecture_2_copy_number=second.copy_number,
                unmatched_copy_ids_1=unmatched_first,
                unmatched_copy_ids_2=unmatched_second,
                topology_1=first.topology,
                topology_2=second.topology,
                biological_relation_1=first.biological_relation,
                biological_relation_2=second.biological_relation,
                side_pattern_1=first.side_pattern,
                side_pattern_2=second.side_pattern,
                spacing_signature_1=first.spacing_signature,
                spacing_signature_2=second.spacing_signature,
                position_pattern_1=first.position_pattern,
                position_pattern_2=second.position_pattern,
                boundary_pattern_1=first.boundary_pattern,
                boundary_pattern_2=second.boundary_pattern,
                match_orientations=orientations,
                same_side_match_count=same_side_count,
                opposite_side_match_count=len(selected) - same_side_count,
            )
        )
    return tuple(
        sorted(
            recurrences,
            key=lambda item: (
                item.protein_accession,
                item.comparison_scope,
                item.architecture_1_id,
                item.architecture_2_id,
            ),
        )
    )

def _annotate_recurrence_counts(
    architectures: Iterable[NormalizedArchitecture],
    recurrences: Iterable[ArchitectureRecurrence],
) -> tuple[NormalizedArchitecture, ...]:
    counts: Counter[str] = Counter()
    for recurrence in recurrences:
        counts[recurrence.architecture_1_id] += 1
        counts[recurrence.architecture_2_id] += 1
    return tuple(
        replace(
            architecture,
            recurrent_comparison_count=counts[architecture.step5c_architecture_id],
        )
        for architecture in architectures
    )

def load_measured_inputs(logger) -> dict[str, object]:

    logger.info("Stage 05C measured inputs: loading complete candidates and opportunities")
    candidates = load_candidates()
    opportunities = load_opportunities()
    if unknown := {item.protein_accession for item in opportunities} - {
        item.protein_accession for item in candidates
    }:
        raise ValueError(
            "Step 3C opportunities name proteins absent from the candidate table: "
            + ", ".join(sorted(unknown))
        )
    evidence = load_evidence(opportunities)
    logger.info("Stage 05C measured inputs: loaded %s evidence items", f"{len(evidence):,}")
    logger.info("Stage 05C measured inputs: reading the units' Step 3C sequences")
    sequences = load_unit_sequences(opportunities)
    logger.info("Stage 05C measured inputs: normalizing canonical Step 4C architectures")
    copies, architectures = normalize_architectures(opportunities)
    return {
        "candidates": candidates,
        "opportunities": opportunities,
        "sequences": sequences,
        "copies": copies,
        "architectures": architectures,
        "evidence": evidence,
    }

def run_observed_analysis(logger) -> dict[str, object]:

    inputs = load_measured_inputs(logger)
    candidates, opportunities = inputs["candidates"], inputs["opportunities"]
    copies, architectures = inputs["copies"], inputs["architectures"]
    logger.info("Step 5C observed: comparing finalized copy sequences")
    matches = build_copy_matches(copies)
    logger.info("Step 5C observed: assigning direct recurrent architectures")
    recurrences = build_architecture_recurrences(
        copies, architectures, matches
    )
    architectures = _annotate_recurrence_counts(architectures, recurrences)
    write_observed_outputs(copies, architectures, matches, recurrences)
    logger.info(
        "Step 5C observed complete: %s candidates, %s opportunities, %s "
        "copies, %s architectures, %s qualifying matches, %s recurrent pairs",
        f"{len(candidates):,}",
        f"{len(opportunities):,}",
        f"{len(copies):,}",
        f"{len(architectures):,}",
        f"{len(matches):,}",
        f"{len(recurrences):,}",
    )
    return {
        **inputs,
        "architectures": architectures,
        "matches": matches,
        "recurrences": recurrences,
    }
