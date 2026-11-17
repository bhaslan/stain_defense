from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.repeat_architecture_inference.architecture_inference_deterministic_v5 import ArchitectureDetector
from src.repeat_architecture_inference.architecture_model import Architecture as Claim, Evidence
from src.repeat_architecture_inference.evalue import log_repeat_evalue
from src.stage05_C_vmatch import config_vmatch as cfg

from .models import Architecture, Arm, Copy, Repeat
from .tandem import _boundary

def build_spaced_architectures(
    repeats: Iterable[Repeat],
    representative_sequences: Mapping[tuple[str, str, str], str],
    units: Mapping[tuple[str, str, str], Mapping[str, str]],
    step3_snapshot_id: str,
    settings: Mapping[str, object],
) -> tuple[list[Architecture], list[dict[str, object]]]:

    by_unit: dict[tuple[str, str, str], list[Repeat]] = defaultdict(list)
    for repeat in repeats:
        if repeat.repeat_class == "direct" and repeat.scope in {"within_left", "within_right"}:
            by_unit[repeat.unit_key].append(repeat)

    architectures: list[Architecture] = []
    records: list[dict[str, object]] = []
    for key, unit_repeats in by_unit.items():
        built, record = _build_unit(key, unit_repeats, representative_sequences, units[key], step3_snapshot_id, settings)
        architectures.extend(built)
        records.append(record)
    return architectures, records

def _build_unit(key, repeats, representative_sequences, unit, step3_snapshot_id, settings):

    protein_accession, scope, search_unit_id = key
    side = unit["database_side"]
    sequence = representative_sequences[(protein_accession, search_unit_id, side)]
    source_ids = {number: repeat.source_match_id for number, repeat in enumerate(repeats, start=1)}
    matches = []

    for number, repeat in enumerate(repeats, start=1):

        if {repeat.arm_1.side, repeat.arm_2.side} != {side}:
            raise ValueError(f"{repeat.source_match_id} does not lie on {':'.join(key)}'s {side} flank")
        if repeat.arm_1.span != repeat.arm_2.span:
            raise ValueError(f"{repeat.source_match_id} has arms of different lengths; V5 needs substitution-only pairs")
        matches.append(dict(match_id=number, start1=repeat.arm_1.start, end1=repeat.arm_1.end,
                            start2=repeat.arm_2.start, end2=repeat.arm_2.end,
                            distance=repeat.distance, evalue=repeat.evalue))

    try:
        evidence = Evidence(sequence, matches, search_identity=cfg.VMATCH_WITHIN_MIN_IDENTITY)
        detector = ArchitectureDetector(evidence, **settings)
        claims = [Claim.from_proposal(proposal, evidence) for proposal in detector.detect()]
    except ValueError as error:
        raise ValueError(f"V5 failed on unit {':'.join(key)}: {error}") from error

    boundary = _boundary(unit, side)
    metadata = repeats[0].metadata
    for repeat in repeats[1:]:
        if repeat.metadata != metadata or repeat.boundary != repeats[0].boundary:
            raise ValueError(f"Step 3C rows of unit {':'.join(key)} disagree on their unit metadata")
    multiplicity_field = "left_cluster_size" if side == "left" else "right_cluster_size"
    architectures = []

    for claim in claims:

        copies = [Copy(side, copy.start, copy.end, copy.sequence, boundary.classify(Arm(side, copy.start, copy.end)),
                       copy.mismatches, tuple(source_ids[number] for number in copy.support))
                  for copy in claim.copies]
        hull = Arm(side, min(copy.start for copy in copies), max(copy.end for copy in copies))
        log_evalue = log_repeat_evalue(evidence.iid.background, len(evidence.sequence), claim.copy_number,
                                       claim.unit_length, claim.max_mismatches)
        architectures.append(Architecture.build(
            copies,
            step3_snapshot_id,
            protein_accession=protein_accession,
            scope=scope,
            search_unit_id=search_unit_id,
            source_sequence_id=metadata["source_sequence_id"],
            left_cluster_id=metadata["left_cluster_id"],
            right_cluster_id=metadata["right_cluster_id"],
            paired_cluster_id=metadata["paired_cluster_id"],
            represented_context_count=int(metadata[multiplicity_field]),
            repeat_class="direct",
            repeat_span_classification=boundary.classify(hull),
            distance_to_cds=hull.start + 1,
            unit_length=claim.unit_length,
            log_evalue=log_evalue,
        ))

    used = {source for architecture in architectures for copy in architecture.copies for source in copy.support}
    decisions = [{name: value for name, value in decision.items() if name != "profile"}
                 for decision in detector.diagnostics["decisions"]]
    record = {
        "step3_snapshot_id": step3_snapshot_id,
        "protein_accession": protein_accession,
        "scope": scope,
        "search_unit_id": search_unit_id,
        "notes": list(detector.notes),
        "decisions": decisions,
        "pairs": [{"source_match_id": source, "match_id": number, "used": source in used}
                  for number, source in source_ids.items()],
        "architecture_ids": [architecture.step4c_architecture_id for architecture in architectures],
    }
    return architectures, record
