from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step3c.trf_evidence import TrfArray

from .models import (
    Architecture,
    Arm,
    BoundaryContext,
    BoundarySide,
    Copy,
    Repeat,
    overlap_length,
)

def trf_overlap_annotations(
    repeat: Repeat,
    arrays: Iterable[TrfArray],
    *,
    minimum_arm_coverage: float = cfg.TRF_VMATCH_MIN_ARM_COVERAGE,
    minimum_overlap_bp: int = cfg.TRF_VMATCH_MIN_OVERLAP_BP,
) -> tuple[dict[str, object], ...]:

    annotations: list[dict[str, object]] = []
    for arm_number, arm in ((1, repeat.arm_1), (2, repeat.arm_2)):
        corresponding_unit = (
            repeat.metadata.get(f"{arm.side}_cluster_id", "")
            or repeat.search_unit_id
        )
        for array in arrays:
            if (
                array.protein_accession != repeat.protein_accession
                or array.search_unit_id != corresponding_unit
                or array.side != arm.side
            ):
                continue
            overlap_bp = overlap_length(arm.interval, array.interval)
            if not overlap_bp:
                continue
            arm_coverage = overlap_bp / arm.span
            annotations.append(
                {
                    "arm_number": arm_number,
                    "trf_array_key": array.trf_array_key,
                    "overlap_bp": overlap_bp,
                    "arm_coverage": arm_coverage,
                    "explained": (
                        overlap_bp >= minimum_overlap_bp
                        and arm_coverage >= minimum_arm_coverage
                    ),
                }
            )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                int(item["arm_number"]),
                str(item["trf_array_key"]),
            ),
        )
    )

def trf_suppression_evidence(
    repeat: Repeat,
    arrays: Iterable[TrfArray],
    *,
    minimum_arm_coverage: float = cfg.TRF_VMATCH_MIN_ARM_COVERAGE,
    minimum_overlap_bp: int = cfg.TRF_VMATCH_MIN_OVERLAP_BP,
) -> tuple[dict[str, object], ...]:

    if not 0.0 <= minimum_arm_coverage <= 1.0:
        raise ValueError("minimum_arm_coverage must be between zero and one")
    if minimum_overlap_bp < 1:
        raise ValueError("minimum_overlap_bp must be positive")
    if not (
        repeat.repeat_class == "direct"
        and repeat.scope in {"within_left", "within_right"}
    ):
        return ()
    arrays = tuple(arrays)
    arrays_by_key = {array.trf_array_key: array for array in arrays}
    by_array: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for annotation in trf_overlap_annotations(
        repeat,
        arrays,
        minimum_arm_coverage=minimum_arm_coverage,
        minimum_overlap_bp=minimum_overlap_bp,
    ):
        if annotation["explained"]:
            by_array[str(annotation["trf_array_key"])][
                int(annotation["arm_number"])
            ] = annotation
    evidence: list[dict[str, object]] = []
    for array_key, by_arm in sorted(by_array.items()):
        if set(by_arm) != {1, 2}:
            continue
        array = arrays_by_key[array_key]
        evidence.append(
            {
                "trf_array_key": array_key,
                "array_side": array.side,
                "array_start_0based": array.start,
                "array_end_0based": array.end,
                "arm_1_overlap_bp": int(by_arm[1]["overlap_bp"]),
                "arm_1_coverage": float(by_arm[1]["arm_coverage"]),
                "arm_2_overlap_bp": int(by_arm[2]["overlap_bp"]),
                "arm_2_coverage": float(by_arm[2]["arm_coverage"]),
            }
        )
    return tuple(evidence)

def partition_vmatch_repeats(
    repeats: Iterable[Repeat],
    arrays: Iterable[TrfArray],
    *,
    minimum_arm_coverage: float = cfg.TRF_VMATCH_MIN_ARM_COVERAGE,
    minimum_overlap_bp: int = cfg.TRF_VMATCH_MIN_OVERLAP_BP,
) -> tuple[tuple[Repeat, ...], tuple[Repeat, ...]]:

    if not 0.0 <= minimum_arm_coverage <= 1.0:
        raise ValueError("minimum_arm_coverage must be between zero and one")
    if minimum_overlap_bp < 1:
        raise ValueError("minimum_overlap_bp must be positive")
    by_unit: dict[tuple[str, str], list[TrfArray]] = defaultdict(list)
    for array in arrays:
        by_unit[(array.protein_accession, array.search_unit_id)].append(array)
    retained: list[Repeat] = []
    suppressed: list[Repeat] = []
    for repeat in repeats:
        evidence = trf_suppression_evidence(
            repeat,
            by_unit.get((repeat.protein_accession, repeat.search_unit_id), ()),
            minimum_arm_coverage=minimum_arm_coverage,
            minimum_overlap_bp=minimum_overlap_bp,
        )
        destination = suppressed if evidence else retained
        destination.append(repeat)
    return tuple(retained), tuple(suppressed)

def _optional_int(value: object) -> int | None:
    return None if value in {None, ""} else int(str(value))

def _boundary(unit: Mapping[str, str], side: str) -> BoundaryContext:
    strand = unit["central_cds_strand"]
    if strand not in {"+", "-"}:
        raise ValueError(f"Invalid central CDS strand: {strand!r}")
    return BoundaryContext(
        int(unit["central_cds_start"]),
        int(unit["central_cds_stop"]),
        strand,
        {
            side: BoundarySide(
                int(unit[f"{side}_flank_length"]),
                unit[f"{side}_boundary_status"],
                _optional_int(unit[f"{side}_intergenic_length"]),
                _optional_int(unit[f"{side}_neighbor_cds_start"]),
                _optional_int(unit[f"{side}_neighbor_cds_stop"]),
            )
        },
    )

def build_trf_architecture(
    array: TrfArray,
    source_sequence: str,
    unit: Mapping[str, str],
    step3_snapshot_id: str,
) -> Architecture:

    for field, expected in (
        ("protein_accession", array.protein_accession),
        ("scope", array.scope),
        ("search_unit_id", array.search_unit_id),
        ("source_sequence_id", array.source_sequence_id),
    ):
        if unit[field] != expected:
            raise ValueError(
                f"TRF array {array.trf_array_key} disagrees with unit {field}"
            )
    if unit["database_side"] != array.side or unit["query_side"]:
        raise ValueError(f"TRF array {array.trf_array_key} is not a within-side unit")
    sequence = source_sequence.upper()
    boundary = _boundary(unit, array.side)
    copies: list[Copy] = []
    for number, copy in enumerate(array.full_copies, start=1):
        observed = sequence[copy.start : copy.end]
        if observed != copy.aligned_repeat.replace("-", ""):
            raise ValueError(
                f"TRF copy {number} disagrees with {array.trf_array_key} sequence"
            )
        classification = boundary.classify(Arm(array.side, copy.start, copy.end))
        copies.append(Copy(array.side, copy.start, copy.end, observed, classification))
    if len(copies) < 2:
        raise ValueError("An accepted TRF array requires at least two full copies")
    multiplicity_field = (
        "left_cluster_size" if array.side == "left" else "right_cluster_size"
    )
    return Architecture.build(
        copies,
        step3_snapshot_id,
        protein_accession=array.protein_accession,
        scope=array.scope,
        search_unit_id=array.search_unit_id,
        source_sequence_id=array.source_sequence_id,
        left_cluster_id=unit["left_cluster_id"],
        right_cluster_id=unit["right_cluster_id"],
        paired_cluster_id="",
        represented_context_count=int(unit[multiplicity_field]),
        repeat_class="tandem_array",
        repeat_span_classification=boundary.classify(Arm(array.side, array.start, array.end)),
        distance_to_cds=array.start + 1,
        trf_array_key=array.trf_array_key,
    )
