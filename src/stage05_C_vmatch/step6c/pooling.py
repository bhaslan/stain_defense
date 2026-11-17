from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.protein_tree.samples import sha256
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step5c.measures import EVIDENCE_RECURRENCE_FIELDS, UNIT_FIELDS, measure
from src.stage05_C_vmatch.step5c.observed import build_architecture_recurrences, build_copy_matches, load_measured_inputs
from src.stage05_C_vmatch.step5c.persistence import read_tsv, write_tsv_atomic
from src.stage05_C_vmatch.step6c import near_copies

def read_samples() -> dict[str, str]:

    _, rows = read_tsv(cfg.STEP6_SAMPLES_PATH, required_fields=("protein_accession", "sample_accession"))
    sample_of: dict[str, str] = {}

    for row in rows:

        protein, sample = row["protein_accession"], row["sample_accession"]
        if not protein or not sample: raise ValueError(f"Sample table row lacks an accession: {row}")
        if protein in sample_of: raise ValueError(f"Protein listed twice in the sample table: {protein}")
        sample_of[protein] = sample

    if unlisted := {sample for sample in sample_of.values() if sample_of.get(sample) != sample}:

        raise ValueError(f"Samples not listed as their own protein: {', '.join(sorted(unlisted))}")

    return sample_of

def _kept_inputs(inputs, kept) -> tuple[tuple, tuple, tuple, tuple]:

    keys = {unit.upstream_key for unit in kept}
    unit_ids = {unit.representative_unit_id for unit in kept}
    evidence = tuple(item for item in inputs["evidence"] if (item.protein_accession, item.scope, item.search_unit_id) in keys)
    copies = []

    for copy in inputs["copies"]:

        held = {membership.representative_unit_id in unit_ids for membership in copy.memberships}
        if len(held) > 1: raise ValueError(f"Copy {copy.step5c_copy_id} lies on kept and set-aside units at once")
        if held == {True}: copies.append(copy)

    architectures = tuple(architecture for architecture in inputs["architectures"]
                          if architecture.representative_unit_id in unit_ids)

    return tuple(kept), evidence, tuple(copies), architectures

def write_sample_summary(logger,
                         path: Path = cfg.STEP6_SAMPLE_SUMMARY_PATH,
                         recurrence_path: Path = cfg.STEP6_EVIDENCE_RECURRENCE_PATH,
                         near_copies_path: Path = cfg.STEP6_NEAR_COPIES_PATH,
                         units_path: Path = cfg.STEP6_UNIT_MEASURES_PATH,
                         manifest_path: Path = cfg.STEP6_SAMPLE_SUMMARY_MANIFEST_PATH) -> None:

    inputs = load_measured_inputs(logger)
    samples_sha256 = sha256(cfg.STEP6_SAMPLES_PATH)
    sample_of = read_samples()
    candidates = inputs["candidates"]

    if unknown := set(sample_of) - {candidate.protein_accession for candidate in candidates}:

        raise ValueError(f"Sample table names proteins absent from the candidate table: {', '.join(sorted(unknown))}")

    samples = set(sample_of.values())
    groups = {candidate.protein_accession: candidate.values for candidate in candidates
              if candidate.protein_accession in samples}
    kept, near_copy_rows = near_copies.select(inputs["opportunities"], sample_of, inputs["sequences"])
    units, evidence, copies, architectures = _kept_inputs(inputs, kept)
    recurrences = build_architecture_recurrences(copies, architectures, build_copy_matches(copies, sample_of), sample_of)
    measured = measure(groups, sample_of, units, copies, architectures, evidence, inputs["sequences"], recurrences)
    rows = measured.rows
    pooled = defaultdict(set)
    removed = Counter(row["sample_accession"] for row in near_copy_rows if not row["kept"])

    for unit in units: pooled[sample_of[unit.protein_accession]].add(unit.protein_accession)

    for row in rows:

        row[cfg.STEP6_POOLED_PROTEINS_FIELD] = len(pooled[row["protein_accession"]])
        row[cfg.STEP6_NEAR_COPIES_FIELD] = removed[row["protein_accession"]]

    labels = (*candidates[0].field_order, cfg.STEP6_POOLED_PROTEINS_FIELD, cfg.STEP6_NEAR_COPIES_FIELD)
    write_tsv_atomic(path, measured.summary_fields(labels), rows)
    write_tsv_atomic(recurrence_path, EVIDENCE_RECURRENCE_FIELDS, measured.recurrence_rows)
    write_tsv_atomic(near_copies_path, near_copies.NEAR_COPY_FIELDS, near_copy_rows)
    write_tsv_atomic(units_path, UNIT_FIELDS, measured.unit_rows)
    manifest_path.write_text(json.dumps({"samples_sha256": samples_sha256}, indent=2) + "\n")

    logger.info("Step 6C sample summary: %s samples pooling %s proteins, %s of %s pooled units set aside as near-copies, "
                "%s samples with no searched unit, %s recurrent architecture pairs, %s evidence recurrent unit pairs "
                "over every set; %s; %s; %s; %s",
                f"{len(rows):,}", f"{len(sample_of):,}", f"{sum(removed.values()):,}", f"{len(near_copy_rows):,}",
                f"{sum(not row['searched_units'] for row in rows):,}", f"{len(recurrences):,}",
                f"{len(measured.recurrence_rows):,}", path, recurrence_path, near_copies_path, units_path)
