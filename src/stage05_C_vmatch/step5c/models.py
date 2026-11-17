from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

PhysicalKey = tuple[str, str, str, int, int]
ComponentKey = tuple[str, str, str]

@dataclass(frozen=True, slots=True)
class Candidate:

    protein_accession: str
    values: Mapping[str, str]
    field_order: tuple[str, ...]

    def __post_init__(self) -> None:
        values = dict(self.values)
        if not self.protein_accession:
            raise ValueError("Candidate protein_accession cannot be empty")
        if values.get("protein_accession") != self.protein_accession:
            raise ValueError("Candidate accession disagrees with its source row")
        if tuple(values) != self.field_order:
            raise ValueError("Candidate field order differs from its source header")
        object.__setattr__(self, "values", MappingProxyType(values))

@dataclass(frozen=True, slots=True)
class Opportunity:

    protein_accession: str
    upstream_scope: str
    comparison_scope: str
    search_unit_id: str
    representative_unit_id: str
    source_context_id: str
    source_sequence_id: str
    database_side: str
    query_side: str | None
    left_flank_length: int
    right_flank_length: int
    left_intergenic_length: int | None
    right_intergenic_length: int | None
    left_cluster_id: str | None
    right_cluster_id: str | None
    left_cluster_size: int | None
    right_cluster_size: int | None
    paired_cluster_id: str | None
    pair_support: int | None

    def __post_init__(self) -> None:
        if self.upstream_scope not in {"within_left", "within_right"}:
            raise ValueError(f"Unknown Step 3C scope: {self.upstream_scope!r}")
        if self.comparison_scope != "within_flank":
            raise ValueError("Opportunity comparison scope is inconsistent")
        expected_database_side = (
            "left" if self.upstream_scope == "within_left" else "right"
        )
        if self.database_side != expected_database_side:
            raise ValueError("Opportunity database side is inconsistent with scope")
        if self.query_side is not None:
            raise ValueError("Within-flank opportunities cannot carry a query side")
        if not all(
            (
                self.protein_accession,
                self.search_unit_id,
                self.representative_unit_id,
                self.source_sequence_id,
            )
        ):
            raise ValueError("Opportunity identifiers cannot be empty")
        for side in self.searched_sides:
            if self.side_length(side) <= 0:
                raise ValueError("Searched flank lengths must be positive")
            intergenic_length = self.intergenic_length(side)
            if intergenic_length is None or intergenic_length < 0:
                raise ValueError(
                    f"Searched {side} side lacks a nonnegative intergenic length"
                )
            if not self.cluster_id(side):
                raise ValueError(f"Searched {side} side lacks its cluster ID")
        if self.paired_cluster_id is not None or self.pair_support is not None:
            raise ValueError("Within-flank units cannot carry pair metadata")

    @property
    def upstream_key(self) -> tuple[str, str, str]:
        return self.protein_accession, self.upstream_scope, self.search_unit_id

    @property
    def searched_sides(self) -> tuple[str, ...]:
        return (self.database_side,)

    def side_length(self, side: str) -> int:
        if side == "left":
            return self.left_flank_length
        if side == "right":
            return self.right_flank_length
        raise ValueError(f"Unknown flank side: {side!r}")

    def intergenic_length(self, side: str) -> int | None:
        if side == "left":
            return self.left_intergenic_length
        if side == "right":
            return self.right_intergenic_length
        raise ValueError(f"Unknown flank side: {side!r}")

    def cluster_id(self, side: str) -> str | None:
        if side == "left":
            return self.left_cluster_id
        if side == "right":
            return self.right_cluster_id
        raise ValueError(f"Unknown flank side: {side!r}")

    def component_key(self, side: str) -> ComponentKey:
        cluster_id = self.cluster_id(side)
        if not cluster_id:
            raise ValueError(f"Opportunity has no {side} cluster")
        return self.protein_accession, side, cluster_id

    def searched_bases_within(self, cutoff: int) -> int:
        return sum(min(self.side_length(side), cutoff) for side in self.searched_sides)

    @property
    def represented_context_count(self) -> int:
        size = (
            self.left_cluster_size
            if self.upstream_scope == "within_left"
            else self.right_cluster_size
        )
        if size is None or size <= 0:
            raise ValueError("Within-flank cluster size must be positive")
        return size

    @property
    def represented_context_kind(self) -> str:
        return "sequence_cluster_size"

@dataclass(frozen=True, slots=True, order=True)
class CopyMembership:
    comparison_scope: str
    upstream_scope: str
    representative_unit_id: str
    step5c_architecture_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.step5c_architecture_ids))) != (
            self.step5c_architecture_ids
        ):
            raise ValueError(
                "Copy architecture memberships must be sorted and unique"
            )
        if any(not item for item in self.step5c_architecture_ids):
            raise ValueError("Copy architecture IDs cannot be empty")

@dataclass(frozen=True, slots=True)
class PhysicalCopy:

    step5c_copy_id: str
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
    memberships: tuple[CopyMembership, ...]

    def __post_init__(self) -> None:
        sequence = self.complete_sequence.upper()
        object.__setattr__(self, "complete_sequence", sequence)
        if self.side not in {"left", "right"}:
            raise ValueError(f"Unknown copy side: {self.side!r}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Copy coordinates must be a nonempty half-open interval")
        if len(sequence) != self.end - self.start:
            raise ValueError("Copy sequence length disagrees with its interval")
        if not self.component_cluster_id:
            raise ValueError("Copy component cluster cannot be empty")
        if not self.memberships:
            raise ValueError("A physical copy needs an opportunity membership")
        if tuple(sorted(set(self.memberships))) != self.memberships:
            raise ValueError("Copy memberships must be sorted and unique")

    @property
    def physical_key(self) -> PhysicalKey:
        return (
            self.protein_accession,
            self.source_sequence_id,
            self.side,
            self.start,
            self.end,
        )

    @property
    def component_key(self) -> ComponentKey:
        return self.protein_accession, self.side, self.component_cluster_id

    @property
    def complete_length(self) -> int:
        return self.end - self.start

    @property
    def distance_to_cds(self) -> int:
        return self.start + 1

@dataclass(frozen=True, slots=True)
class NormalizedArchitecture:

    step5c_architecture_id: str
    protein_accession: str
    comparison_scope: str
    upstream_scope: str
    representative_unit_id: str
    search_unit_id: str
    source_sequence_id: str
    left_cluster_id: str | None
    right_cluster_id: str | None
    paired_cluster_id: str | None
    represented_context_count: int
    represented_context_kind: str
    copy_ids: tuple[str, ...]
    copy_class: str
    topology: str
    biological_relation: str
    repeat_class: str
    side_pattern: tuple[str, ...]
    spacing_signature: tuple[str, ...]
    edge_separation_categories: tuple[str, ...]
    position_pattern: tuple[int, ...]
    boundary_pattern: tuple[str, ...]
    recurrent_comparison_count: int = 0
    step4c_architecture_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.comparison_scope != "within_flank":
            raise ValueError("Architecture has an invalid comparison scope")
        if len(set(self.copy_ids)) != len(self.copy_ids):
            raise ValueError("Architecture copy IDs must be unique")
        if len(self.copy_ids) < 2:
            raise ValueError("An architecture requires at least two copies")
        if self.copy_class not in {"pair", "small_array", "large_array"}:
            raise ValueError("Architecture has an invalid copy class")
        if self.topology not in {"simple_pair", "linear_array"}:
            raise ValueError("Architecture has an invalid topology")
        if self.biological_relation != "direct":
            raise ValueError("Architecture has an invalid biological relation")
        if self.recurrent_comparison_count < 0:
            raise ValueError("Recurrent comparison count cannot be negative")
        if tuple(sorted(set(self.step4c_architecture_ids))) != self.step4c_architecture_ids:
            raise ValueError(
                "Step 4C architecture IDs must be sorted and unique"
            )
        if any(not item for item in self.step4c_architecture_ids):
            raise ValueError("Step 4C architecture IDs cannot be empty")

    @property
    def copy_number(self) -> int:
        return len(self.copy_ids)

@dataclass(frozen=True, slots=True)
class CopyMatch:

    step5c_match_id: str
    protein_accession: str
    comparison_scope: str
    copy_1_id: str
    copy_2_id: str
    representative_unit_1_id: str
    representative_unit_2_id: str
    source_sequence_1_id: str
    source_sequence_2_id: str
    side_1: str
    side_2: str
    component_1_id: str
    component_2_id: str
    length_1: int
    length_2: int
    direct_similarity: float
    reverse_complement_similarity: float
    similarity: float
    orientation: str
    shared_component_reuse: bool

    def __post_init__(self) -> None:
        if self.copy_1_id >= self.copy_2_id:
            raise ValueError("Copy-match endpoints must be canonically ordered")
        if self.representative_unit_1_id == self.representative_unit_2_id:
            raise ValueError("Copy matches must cross representative units")
        if self.orientation not in {"direct", "reverse_complement", "ambiguous"}:
            raise ValueError("Copy match has an invalid orientation")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("Copy similarity must lie in [0, 1]")
        if self.similarity != max(
            self.direct_similarity, self.reverse_complement_similarity
        ):
            raise ValueError("Copy similarity is not the maximum orientation score")

    @property
    def independent(self) -> bool:
        return not self.shared_component_reuse

    @property
    def side_correspondence(self) -> str:
        return "same_side" if self.side_1 == self.side_2 else "opposite_side"

@dataclass(frozen=True, slots=True)
class ArchitectureRecurrence:

    step5c_recurrence_id: str
    protein_accession: str
    comparison_scope: str
    architecture_1_id: str
    architecture_2_id: str
    representative_unit_1_id: str
    representative_unit_2_id: str
    selected_match_ids: tuple[str, ...]
    selected_copy_pairs: tuple[tuple[str, str], ...]
    architecture_1_copy_number: int
    architecture_2_copy_number: int
    unmatched_copy_ids_1: tuple[str, ...]
    unmatched_copy_ids_2: tuple[str, ...]
    topology_1: str
    topology_2: str
    biological_relation_1: str
    biological_relation_2: str
    side_pattern_1: tuple[str, ...]
    side_pattern_2: tuple[str, ...]
    spacing_signature_1: tuple[str, ...]
    spacing_signature_2: tuple[str, ...]
    position_pattern_1: tuple[int, ...]
    position_pattern_2: tuple[int, ...]
    boundary_pattern_1: tuple[str, ...]
    boundary_pattern_2: tuple[str, ...]
    match_orientations: tuple[str, ...]
    same_side_match_count: int
    opposite_side_match_count: int

    def __post_init__(self) -> None:
        if self.architecture_1_id >= self.architecture_2_id:
            raise ValueError("Recurrence architecture IDs must be canonically ordered")
        if self.representative_unit_1_id == self.representative_unit_2_id:
            raise ValueError("Recurrence must cross representative units")
        if len(self.selected_match_ids) != len(self.selected_copy_pairs):
            raise ValueError("Selected match IDs and copy pairs disagree")
        if len(set(self.selected_match_ids)) != len(self.selected_match_ids):
            raise ValueError("A recurrence cannot reuse a selected match")
        first = [pair[0] for pair in self.selected_copy_pairs]
        second = [pair[1] for pair in self.selected_copy_pairs]
        if len(first) != len(set(first)) or len(second) != len(set(second)):
            raise ValueError("A recurrence assignment cannot reuse a copy")

    @property
    def matched_copy_count(self) -> int:
        return len(self.selected_match_ids)

    @property
    def matched_fraction_1(self) -> float:
        return self.matched_copy_count / self.architecture_1_copy_number

    @property
    def matched_fraction_2(self) -> float:
        return self.matched_copy_count / self.architecture_2_copy_number

    @property
    def full_copy_match(self) -> bool:
        return not self.unmatched_copy_ids_1 and not self.unmatched_copy_ids_2
