from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.stage05_C_vmatch import config_vmatch as cfg

from .models import (
    ArchitectureRecurrence,
    CopyMatch,
    NormalizedArchitecture,
    PhysicalCopy,
)

COPY_FIELDS = (
    "step5c_copy_id",
    "protein_accession",
    "source_sequence_id",
    "side",
    "start",
    "end",
    "complete_length",
    "complete_sequence",
    "boundary_class",
    "length_class",
    "distance_to_cds",
    "distance_class",
    "component_cluster_id",
    "architecture_memberships",
)

ARCHITECTURE_FIELDS = (
    "step5c_architecture_id",
    "protein_accession",
    "scope",
    "upstream_scope",
    "representative_unit_id",
    "search_unit_id",
    "source_sequence_id",
    "left_cluster_id",
    "right_cluster_id",
    "paired_cluster_id",
    "represented_context_count",
    "represented_context_kind",
    "copy_ids",
    "copy_number",
    "copy_class",
    "topology",
    "biological_relation",
    "repeat_class",
    "side_pattern",
    "spacing_signature",
    "edge_separation_categories",
    "position_pattern",
    "boundary_pattern",
    "recurrent_comparison_count",
    "step4c_architecture_ids",
)

MATCH_FIELDS = (
    "step5c_match_id",
    "protein_accession",
    "comparison_scope",
    "copy_1_id",
    "copy_2_id",
    "representative_unit_1_id",
    "representative_unit_2_id",
    "source_sequence_1_id",
    "source_sequence_2_id",
    "side_1",
    "side_2",
    "side_correspondence",
    "component_1_id",
    "component_2_id",
    "shared_component_reuse",
    "independent",
    "length_1",
    "length_2",
    "direct_similarity",
    "reverse_complement_similarity",
    "similarity",
    "orientation",
)

RECURRENCE_FIELDS = (
    "step5c_recurrence_id",
    "protein_accession",
    "comparison_scope",
    "architecture_1_id",
    "architecture_2_id",
    "representative_unit_1_id",
    "representative_unit_2_id",
    "selected_match_ids",
    "selected_copy_pairs",
    "matched_copy_count",
    "architecture_1_copy_number",
    "architecture_2_copy_number",
    "matched_fraction_1",
    "matched_fraction_2",
    "unmatched_copy_ids_1",
    "unmatched_copy_ids_2",
    "copy_number_difference",
    "absolute_copy_number_difference",
    "full_copy_match",
    "same_copy_number",
    "topology_1",
    "topology_2",
    "same_topology",
    "biological_relation_1",
    "biological_relation_2",
    "same_biological_relation",
    "side_pattern_1",
    "side_pattern_2",
    "same_side_pattern",
    "spacing_signature_1",
    "spacing_signature_2",
    "same_spacing",
    "position_pattern_1",
    "position_pattern_2",
    "same_position_pattern",
    "boundary_pattern_1",
    "boundary_pattern_2",
    "same_boundary_pattern",
    "match_orientations",
    "direct_match_count",
    "reverse_complement_match_count",
    "ambiguous_match_count",
    "same_side_match_count",
    "opposite_side_match_count",
)

def read_tsv(
    path: str | Path,
    required_fields: Iterable[str] = (),
) -> tuple[tuple[str, ...], list[dict[str, str]]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        if not fields:
            raise ValueError(f"{path.name} has no header")
        missing = set(required_fields) - set(fields)
        if missing:
            raise ValueError(
                f"{path.name} lacks required fields: {', '.join(sorted(missing))}"
            )
        return fields, list(reader)

def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

def _float(value: float) -> str:
    return f"{value:.12g}"

def _bool(value: bool) -> str:
    return "True" if value else "False"

def copy_row(item: PhysicalCopy) -> dict[str, object]:
    memberships = [
        {
            "comparison_scope": membership.comparison_scope,
            "upstream_scope": membership.upstream_scope,
            "representative_unit_id": membership.representative_unit_id,
            "step5c_architecture_ids": membership.step5c_architecture_ids,
        }
        for membership in item.memberships
    ]
    return {
        "step5c_copy_id": item.step5c_copy_id,
        "protein_accession": item.protein_accession,
        "source_sequence_id": item.source_sequence_id,
        "side": item.side,
        "start": item.start,
        "end": item.end,
        "complete_length": item.complete_length,
        "complete_sequence": item.complete_sequence,
        "boundary_class": item.boundary_class,
        "length_class": item.length_class,
        "distance_to_cds": item.distance_to_cds,
        "distance_class": item.distance_class,
        "component_cluster_id": item.component_cluster_id,
        "architecture_memberships": _json(memberships),
    }

def architecture_row(item: NormalizedArchitecture) -> dict[str, object]:
    return {
        "step5c_architecture_id": item.step5c_architecture_id,
        "protein_accession": item.protein_accession,
        "scope": item.comparison_scope,
        "upstream_scope": item.upstream_scope,
        "representative_unit_id": item.representative_unit_id,
        "search_unit_id": item.search_unit_id,
        "source_sequence_id": item.source_sequence_id,
        "left_cluster_id": item.left_cluster_id or "",
        "right_cluster_id": item.right_cluster_id or "",
        "paired_cluster_id": item.paired_cluster_id or "",
        "represented_context_count": item.represented_context_count,
        "represented_context_kind": item.represented_context_kind,
        "copy_ids": _json(item.copy_ids),
        "copy_number": item.copy_number,
        "copy_class": item.copy_class,
        "topology": item.topology,
        "biological_relation": item.biological_relation,
        "repeat_class": item.repeat_class,
        "side_pattern": _json(item.side_pattern),
        "spacing_signature": _json(item.spacing_signature),
        "edge_separation_categories": _json(item.edge_separation_categories),
        "position_pattern": _json(item.position_pattern),
        "boundary_pattern": _json(item.boundary_pattern),
        "recurrent_comparison_count": item.recurrent_comparison_count,
        "step4c_architecture_ids": _json(item.step4c_architecture_ids),
    }

def match_row(item: CopyMatch) -> dict[str, object]:
    return {
        "step5c_match_id": item.step5c_match_id,
        "protein_accession": item.protein_accession,
        "comparison_scope": item.comparison_scope,
        "copy_1_id": item.copy_1_id,
        "copy_2_id": item.copy_2_id,
        "representative_unit_1_id": item.representative_unit_1_id,
        "representative_unit_2_id": item.representative_unit_2_id,
        "source_sequence_1_id": item.source_sequence_1_id,
        "source_sequence_2_id": item.source_sequence_2_id,
        "side_1": item.side_1,
        "side_2": item.side_2,
        "side_correspondence": item.side_correspondence,
        "component_1_id": item.component_1_id,
        "component_2_id": item.component_2_id,
        "shared_component_reuse": _bool(item.shared_component_reuse),
        "independent": _bool(item.independent),
        "length_1": item.length_1,
        "length_2": item.length_2,
        "direct_similarity": _float(item.direct_similarity),
        "reverse_complement_similarity": _float(
            item.reverse_complement_similarity
        ),
        "similarity": _float(item.similarity),
        "orientation": item.orientation,
    }

def recurrence_row(item: ArchitectureRecurrence) -> dict[str, object]:
    orientation_counts = {
        orientation: item.match_orientations.count(orientation)
        for orientation in ("direct", "reverse_complement", "ambiguous")
    }
    copy_difference = (
        item.architecture_2_copy_number - item.architecture_1_copy_number
    )
    return {
        "step5c_recurrence_id": item.step5c_recurrence_id,
        "protein_accession": item.protein_accession,
        "comparison_scope": item.comparison_scope,
        "architecture_1_id": item.architecture_1_id,
        "architecture_2_id": item.architecture_2_id,
        "representative_unit_1_id": item.representative_unit_1_id,
        "representative_unit_2_id": item.representative_unit_2_id,
        "selected_match_ids": _json(item.selected_match_ids),
        "selected_copy_pairs": _json(item.selected_copy_pairs),
        "matched_copy_count": item.matched_copy_count,
        "architecture_1_copy_number": item.architecture_1_copy_number,
        "architecture_2_copy_number": item.architecture_2_copy_number,
        "matched_fraction_1": _float(item.matched_fraction_1),
        "matched_fraction_2": _float(item.matched_fraction_2),
        "unmatched_copy_ids_1": _json(item.unmatched_copy_ids_1),
        "unmatched_copy_ids_2": _json(item.unmatched_copy_ids_2),
        "copy_number_difference": copy_difference,
        "absolute_copy_number_difference": abs(copy_difference),
        "full_copy_match": _bool(item.full_copy_match),
        "same_copy_number": _bool(copy_difference == 0),
        "topology_1": item.topology_1,
        "topology_2": item.topology_2,
        "same_topology": _bool(item.topology_1 == item.topology_2),
        "biological_relation_1": item.biological_relation_1,
        "biological_relation_2": item.biological_relation_2,
        "same_biological_relation": _bool(
            item.biological_relation_1 == item.biological_relation_2
        ),
        "side_pattern_1": _json(item.side_pattern_1),
        "side_pattern_2": _json(item.side_pattern_2),
        "same_side_pattern": _bool(item.side_pattern_1 == item.side_pattern_2),
        "spacing_signature_1": _json(item.spacing_signature_1),
        "spacing_signature_2": _json(item.spacing_signature_2),
        "same_spacing": _bool(
            item.spacing_signature_1 == item.spacing_signature_2
        ),
        "position_pattern_1": _json(item.position_pattern_1),
        "position_pattern_2": _json(item.position_pattern_2),
        "same_position_pattern": _bool(
            item.position_pattern_1 == item.position_pattern_2
        ),
        "boundary_pattern_1": _json(item.boundary_pattern_1),
        "boundary_pattern_2": _json(item.boundary_pattern_2),
        "same_boundary_pattern": _bool(
            item.boundary_pattern_1 == item.boundary_pattern_2
        ),
        "match_orientations": _json(item.match_orientations),
        "direct_match_count": orientation_counts["direct"],
        "reverse_complement_match_count": orientation_counts[
            "reverse_complement"
        ],
        "ambiguous_match_count": orientation_counts["ambiguous"],
        "same_side_match_count": item.same_side_match_count,
        "opposite_side_match_count": item.opposite_side_match_count,
    }

def write_tsv_atomic(
    path: str | Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(fields),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

def write_observed_outputs(
    copies: Sequence[PhysicalCopy],
    architectures: Sequence[NormalizedArchitecture],
    matches: Sequence[CopyMatch],
    recurrences: Sequence[ArchitectureRecurrence],
) -> None:

    write_tsv_atomic(
        cfg.STEP5_COPIES_PATH,
        COPY_FIELDS,
        map(copy_row, copies),
    )
    write_tsv_atomic(
        cfg.STEP5_ARCHITECTURES_PATH,
        ARCHITECTURE_FIELDS,
        map(architecture_row, architectures),
    )
    write_tsv_atomic(
        cfg.STEP5_COPY_MATCHES_PATH,
        MATCH_FIELDS,
        map(match_row, matches),
    )
    write_tsv_atomic(
        cfg.STEP5_ARCHITECTURE_RECURRENCE_PATH,
        RECURRENCE_FIELDS,
        map(recurrence_row, recurrences),
    )
