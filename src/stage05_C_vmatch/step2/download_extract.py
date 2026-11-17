import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import edlib
from Bio import SeqIO
from Bio.Data.CodonTable import TranslationError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step1 import inventory_candidates as step1
from src.stage05_C_vmatch.step2 import segment_downloader

CONTEXT_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "context_id",
    "sequence_id",
    "selection_rank",
    "canonical_context_id",
    "raw_context_id",
    "canonical_assembly",
    "assembly",
    "nucleotide_accession",
    "start",
    "stop",
    "strand",
    "authoritative_context",
    "target_protein_block",
    "member_protein_accession",
    "taxid",
    "organism",
    "strain",
    "manifest_selection_reason",
    "requested_fetch_start",
    "requested_fetch_stop",
    "raw_path",
    "download_status",
    "central_cds_annotated",
    "left_boundary_source",
    "left_boundary_status",
    "left_intergenic_length",
    "left_neighbor_cds_start",
    "left_neighbor_cds_stop",
    "left_neighbor_cds_strand",
    "left_neighbor_boundary_type",
    "left_neighbor_partial",
    "right_boundary_source",
    "right_boundary_status",
    "right_intergenic_length",
    "right_neighbor_cds_start",
    "right_neighbor_cds_stop",
    "right_neighbor_cds_strand",
    "right_neighbor_boundary_type",
    "right_neighbor_partial",
    "status",
    "error",
    "retrieved_length",
    "cds_length",
    "left_length",
    "right_length",
    "translation_table",
    "translation_mode",
    "translated_length",
]

PROTEIN_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "requested_contexts",
    "downloaded_contexts",
    "cached_contexts",
    "valid_contexts",
    "usable_left_flanks",
    "usable_right_flanks",
    "failed_contexts",
    "translation_mismatches",
    "left_clusters",
    "left_collapsed_flanks",
    "left_sequence_eligible",
    "left_eligibility_reason",
    "left_authoritative_usable",
    "left_authoritative_representative",
    "left_finalized_flanks",
    "right_clusters",
    "right_collapsed_flanks",
    "right_sequence_eligible",
    "right_eligibility_reason",
    "right_authoritative_usable",
    "right_authoritative_representative",
    "right_finalized_flanks",
]

CLUSTER_FIELDS = [
    "protein_accession",
    "context_id",
    "sequence_id",
    "selection_rank",
    "authoritative_context",
    "flank_length",
    "cluster_id",
    "cluster_number",
    "cluster_size",
    "selected_representative",
    "representative_sequence_id",
    "clustering_method",
    "minimum_prefix_similarity",
    "ambiguity_policy",
    "prefix_edit_distance",
]

def active_manifest_summary():
    manifest = _load_active_manifest()
    group_counts = Counter(row["candidate_group"] for row in manifest)
    return {
        "proteins": len(manifest),
        "contexts": sum(len(row["selected_contexts"]) for row in manifest),
        "groups": dict(sorted(group_counts.items())),
        "flank_length": cfg.FLANK_LENGTH,
        "minimum_flank_length": cfg.MIN_FLANK_LENGTH,
        "flank_clustering_method": cfg.FLANK_CLUSTER_METHOD,
        "minimum_prefix_similarity": (
            cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY
        ),
        "ambiguity_policy": cfg.FLANK_CLUSTER_AMBIGUITY_POLICY,
    }

def download_and_extract():
    manifest = _load_active_manifest()
    if not cfg.NCBI_EMAIL:
        raise ValueError("Set NCBI_EMAIL before downloading genomic segments.")

    cfg.SEGMENT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cfg.EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    total_contexts = sum(len(row["selected_contexts"]) for row in manifest)
    all_context_rows = []
    protein_rows = []

    with tqdm(total=total_contexts, desc="Stage 05C contexts") as progress:
        for protein in manifest:
            protein_accession = protein["protein_accession"]
            progress.set_postfix_str(protein_accession)
            context_rows = []
            valid_records = []

            contexts = sorted(
                protein["selected_contexts"],
                key=lambda row: (
                    int(row["selection_rank"]),
                    row["canonical_context_id"],
                ),
            )
            for context in contexts:
                result, records = _process_context(
                    protein,
                    context,
                )

                context_rows.append(result)
                if records is not None:
                    valid_records.append(records)
                progress.update()

            paths = _protein_paths(protein_accession)
            _write_protein_fastas(paths, valid_records)
            _write_tsv_atomic(
                paths["contexts"],
                CONTEXT_FIELDS,
                context_rows,
            )

            all_context_rows.extend(context_rows)
            protein_rows.append(_protein_summary(protein, context_rows))

    _write_tsv_atomic(
        cfg.STEP2_CONTEXTS_PATH,
        CONTEXT_FIELDS,
        all_context_rows,
    )
    _write_tsv_atomic(
        cfg.STEP2_PROTEINS_PATH,
        PROTEIN_FIELDS,
        protein_rows,
    )
    print(f"Context summary: {cfg.STEP2_CONTEXTS_PATH}")
    print(f"Protein summary: {cfg.STEP2_PROTEINS_PATH}")
    return protein_rows

def cluster_usable_flanks():
    _validate_cluster_configuration()
    manifest = _load_active_manifest()
    protein_rows = {
        row["protein_accession"]: row
        for row in _read_tsv(cfg.STEP2_PROTEINS_PATH)
    }
    all_cluster_rows = {"left": [], "right": []}
    cache_counts = Counter()

    for protein in tqdm(manifest, desc="Edlib proteins"):
        protein_accession = protein["protein_accession"]
        if protein_accession not in protein_rows:
            raise ValueError(
                f"Step 2 download summary is missing {protein_accession}."
            )

        paths = _protein_paths(protein_accession)
        context_rows = _read_tsv(paths["contexts"])
        for side in ("left", "right"):
            usable_rows = _usable_side_rows(context_rows, side)
            records_by_id = _records_by_id(paths[f"usable_{side}"])
            rows_by_id = {
                row["sequence_id"]: row
                for row in usable_rows
            }
            if set(records_by_id) != set(rows_by_id):
                raise ValueError(
                    f"Usable {side} FASTA and contexts.tsv disagree for "
                    f"{protein_accession}."
                )

            expected_state = _cluster_state(
                protein_accession,
                side,
                rows_by_id,
                records_by_id,
            )
            cluster_rows = _cached_cluster_rows(
                paths[f"{side}_clusters"],
                paths[f"{side}_cluster_state"],
                expected_state,
                records_by_id,
            )
            if cluster_rows is None:
                if not records_by_id:
                    cache_counts["empty"] += 1
                elif len(records_by_id) == 1:
                    cache_counts["singleton"] += 1
                else:
                    cache_counts["edlib"] += 1
                cluster_rows = _cluster_rows(
                    protein_accession,
                    rows_by_id,
                    records_by_id,
                )
                _write_tsv_atomic(
                    paths[f"{side}_clusters"],
                    CLUSTER_FIELDS,
                    cluster_rows,
                )
                _write_json_atomic(
                    paths[f"{side}_cluster_state"],
                    expected_state,
                )
            else:
                cache_counts["reused"] += 1

            all_cluster_rows[side].extend(cluster_rows)
            _update_cluster_summary(
                protein,
                protein_rows[protein_accession],
                side,
                usable_rows,
                cluster_rows,
            )

    cfg.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        _write_tsv_atomic(
            getattr(cfg, f"STEP2_{side.upper()}_CLUSTERS_PATH"),
            CLUSTER_FIELDS,
            all_cluster_rows[side],
        )
    _write_tsv_atomic(
        cfg.STEP2_PROTEINS_PATH,
        PROTEIN_FIELDS,
        [protein_rows[row["protein_accession"]] for row in manifest],
    )
    print(f"Left cluster membership: {cfg.STEP2_LEFT_CLUSTERS_PATH}")
    print(f"Right cluster membership: {cfg.STEP2_RIGHT_CLUSTERS_PATH}")
    print(f"Updated protein summary: {cfg.STEP2_PROTEINS_PATH}")
    print(
        f"Cluster sides: {cache_counts['reused']} reused, "
        f"{cache_counts['edlib']} clustered with Edlib, "
        f"{cache_counts['singleton']} singleton-assigned, "
        f"{cache_counts['empty']} empty"
    )
    return all_cluster_rows

def finalize_nonredundant_fastas():
    _validate_cluster_configuration()
    manifest = _load_active_manifest()
    protein_rows = {
        row["protein_accession"]: row
        for row in _read_tsv(cfg.STEP2_PROTEINS_PATH)
    }

    for protein in manifest:
        protein_accession = protein["protein_accession"]
        if protein_accession not in protein_rows:
            raise ValueError(
                f"Step 2 protein summary is missing {protein_accession}."
            )

        paths = _protein_paths(protein_accession)
        context_rows = _read_tsv(paths["contexts"])
        for side in ("left", "right"):
            usable_rows = _usable_side_rows(context_rows, side)
            rows_by_id = {
                row["sequence_id"]: row
                for row in usable_rows
            }
            records_by_id = _records_by_id(paths[f"usable_{side}"])
            if set(records_by_id) != set(rows_by_id):
                raise ValueError(
                    f"Usable {side} FASTA and contexts.tsv disagree for "
                    f"{protein_accession}."
                )
            expected_state = _cluster_state(
                protein_accession,
                side,
                rows_by_id,
                records_by_id,
            )
            cluster_rows = _cached_cluster_rows(
                paths[f"{side}_clusters"],
                paths[f"{side}_cluster_state"],
                expected_state,
                records_by_id,
            )
            if cluster_rows is None:
                raise ValueError(
                    f"{side.capitalize()} cluster membership is stale or "
                    f"missing for {protein_accession}; rerun clustering."
                )
            cluster_ids = {row["cluster_id"] for row in cluster_rows}
            representative_counts = Counter(
                row["cluster_id"]
                for row in cluster_rows
                if _as_bool(row["selected_representative"])
            )
            if any(
                representative_counts[cluster_id] != 1
                for cluster_id in cluster_ids
            ):
                raise ValueError(
                    f"Expected one selected {side} representative per cluster "
                    f"for {protein_accession}."
                )
            representatives = [
                row
                for row in cluster_rows
                if _as_bool(row["selected_representative"])
            ]
            representatives.sort(key=lambda row: int(row["cluster_number"]))
            representative_ids = [
                row["sequence_id"]
                for row in representatives
            ]

            _write_fasta_atomic(
                paths[f"nr_{side}"],
                [records_by_id[record_id] for record_id in representative_ids],
            )
            protein_rows[protein_accession][f"{side}_finalized_flanks"] = len(
                representative_ids
            )

    _write_tsv_atomic(
        cfg.STEP2_PROTEINS_PATH,
        PROTEIN_FIELDS,
        [protein_rows[row["protein_accession"]] for row in manifest],
    )
    print(f"Finalized nonredundant FASTAs under {cfg.EXTRACTIONS_DIR}")
    return protein_rows

def run_step2():
    download_and_extract()
    return cluster_usable_flanks()

def _process_context(protein, context):
    row = _context_row(protein, context)
    try:
        raw_path, downloaded = segment_downloader.fetch_genbank_segment(
            protein_accession=protein["protein_accession"],
            context=context,
            download_root=cfg.SEGMENT_DOWNLOAD_DIR,
            flank_length=cfg.FLANK_LENGTH,
            email=cfg.NCBI_EMAIL,
            api_key=cfg.NCBI_API_KEY,
        )
    except Exception as error:
        row["status"] = "fetch_error"
        row["error"] = f"{type(error).__name__}: {error}"
        return row, None

    row["raw_path"] = str(raw_path)
    row["download_status"] = "downloaded" if downloaded else "cached"

    raw_record = segment_downloader.read_genbank_segment(raw_path)
    return process_record(row, raw_record, context, protein["expected_protein_sequence"])

def process_record(row, raw_record, context, expected_protein_sequence):

    row["retrieved_length"] = len(raw_record.seq)
    row.update(_boundary_annotations(raw_record, context))
    try:
        extracted = _extract_sequences(raw_record.seq, context)
    except ValueError as error:
        row["status"] = "coordinate_error"
        row["error"] = str(error)
        return row, None

    row["cds_length"] = len(extracted["cds"])
    row["left_length"] = len(extracted["left"])
    row["right_length"] = len(extracted["right"])
    expected = expected_protein_sequence.strip().upper().rstrip("*")
    translation = _matching_translation(extracted["cds"], expected)
    row["translated_length"] = translation["translated_length"]
    if not translation["matched"]:
        row["status"] = "translation_mismatch"
        row["error"] = (
            "CDS did not exactly match the expected protein with translation "
            f"tables {cfg.TRANSLATION_TABLES}."
        )
        return row, None

    row["status"] = "valid"
    row["translation_table"] = translation["table"]
    row["translation_mode"] = translation["mode"]
    records = _sequence_records(row, context, extracted)
    return row, records

def _boundary_annotations(record, context):
    fetch_start, _ = segment_downloader.fetch_bounds(
        context,
        cfg.FLANK_LENGTH,
    )
    record_length = len(record.seq)
    central_raw_start = int(context["start"]) - fetch_start
    central_raw_end = int(context["stop"]) - fetch_start + 1
    central_strand = 1 if context["strand"] == "+" else -1

    if central_strand == 1:
        central_start = central_raw_start
        central_end = central_raw_end
    else:
        central_start = record_length - central_raw_end
        central_end = record_length - central_raw_start

    features = []
    central_annotated = False
    for feature in record.features:
        if feature.type != "CDS":
            continue
        raw_start_bounds = _position_bounds(
            feature.location.start,
            record_length,
        )
        raw_end_bounds = _position_bounds(
            feature.location.end,
            record_length,
        )
        start_unknown = _position_is_unknown(feature.location.start)
        end_unknown = _position_is_unknown(feature.location.end)
        if start_unknown and end_unknown:
            continue
        raw_start_min, raw_start_max = raw_start_bounds
        raw_end_min, raw_end_max = raw_end_bounds
        raw_start_max = min(raw_start_max, raw_end_max)
        raw_end_min = max(raw_end_min, raw_start_min)
        raw_start = (
            raw_start_max
            if start_unknown
            else int(feature.location.start)
        )
        raw_end = (
            raw_end_min
            if end_unknown
            else int(feature.location.end)
        )
        raw_strand = feature.location.strand
        if (
            raw_start == central_raw_start
            and raw_end == central_raw_end
            and raw_strand == central_strand
        ):
            central_annotated = True
            continue

        start_exact = _position_is_exact(feature.location.start)
        end_exact = _position_is_exact(feature.location.end)
        if central_strand == 1:
            oriented_start = raw_start
            oriented_end = raw_end
            oriented_start_min = raw_start_min
            oriented_start_max = raw_start_max
            oriented_end_min = raw_end_min
            oriented_end_max = raw_end_max
            oriented_strand = raw_strand
            oriented_start_exact = start_exact
            oriented_end_exact = end_exact
        else:
            oriented_start = record_length - raw_end
            oriented_end = record_length - raw_start
            oriented_start_min = record_length - raw_end_max
            oriented_start_max = record_length - raw_end_min
            oriented_end_min = record_length - raw_start_max
            oriented_end_max = record_length - raw_start_min
            oriented_strand = (
                -raw_strand if raw_strand is not None else None
            )
            oriented_start_exact = end_exact
            oriented_end_exact = start_exact

        features.append(
            {
                "start": oriented_start,
                "end": oriented_end,
                "start_min": oriented_start_min,
                "start_max": oriented_start_max,
                "end_min": oriented_end_min,
                "end_max": oriented_end_max,
                "strand": oriented_strand,
                "start_exact": oriented_start_exact,
                "end_exact": oriented_end_exact,
                "accession_start": (
                    "" if start_unknown else fetch_start + raw_start
                ),
                "accession_stop": (
                    "" if end_unknown else fetch_start + raw_end - 1
                ),
                "accession_strand": raw_strand,
                "partial": not (start_exact and end_exact),
            }
        )

    annotations = {"central_cds_annotated": central_annotated}
    annotations.update(
        _side_boundary_annotation(
            "left",
            features,
            central_start,
            central_end,
            central_start,
        )
    )
    annotations.update(
        _side_boundary_annotation(
            "right",
            features,
            central_start,
            central_end,
            record_length - central_end,
        )
    )
    return annotations

def _side_boundary_annotation(
    side,
    features,
    central_start,
    central_end,
    flank_length,
):
    prefix = f"{side}_"
    result = {
        f"{prefix}boundary_source": "genbank_cds",
        f"{prefix}boundary_status": "no_cds_within_flank",
        f"{prefix}intergenic_length": flank_length,
        f"{prefix}neighbor_cds_start": "",
        f"{prefix}neighbor_cds_stop": "",
        f"{prefix}neighbor_cds_strand": "",
        f"{prefix}neighbor_boundary_type": "",
        f"{prefix}neighbor_partial": "",
    }

    if side == "left":
        overlapping = [
            feature
            for feature in features
            if feature["start_max"] < central_start < feature["end_min"]
        ]
        candidates = [
            feature
            for feature in features
            if feature["end_min"] <= central_start
        ]
        if overlapping:
            neighbor = max(
                overlapping,
                key=lambda feature: (feature["start"], feature["end"]),
            )
            status = "overlapping"
            intergenic_length = 0
        elif candidates:
            neighbor = max(
                candidates,
                key=lambda feature: (
                    feature["end_min"],
                    feature["start_min"],
                ),
            )
            intergenic_length = _maximal_intergenic_length(
                side,
                neighbor,
                central_start,
                central_end,
                flank_length,
            )
            status = (
                "annotated"
                if neighbor["end_exact"] and neighbor["strand"] is not None
                else "uncertain_boundary"
            )
        else:
            return result
        proximal_type = {
            1: "stop",
            -1: "start",
        }.get(neighbor["strand"], "")
    else:
        overlapping = [
            feature
            for feature in features
            if feature["start_max"] < central_end < feature["end_min"]
        ]
        candidates = [
            feature
            for feature in features
            if feature["start_max"] >= central_end
        ]
        if overlapping:
            neighbor = min(
                overlapping,
                key=lambda feature: (feature["end"], feature["start"]),
            )
            status = "overlapping"
            intergenic_length = 0
        elif candidates:
            neighbor = min(
                candidates,
                key=lambda feature: (
                    feature["start_max"],
                    feature["end_max"],
                ),
            )
            intergenic_length = _maximal_intergenic_length(
                side,
                neighbor,
                central_start,
                central_end,
                flank_length,
            )
            status = (
                "annotated"
                if neighbor["start_exact"] and neighbor["strand"] is not None
                else "uncertain_boundary"
            )
        else:
            return result
        proximal_type = {
            1: "start",
            -1: "stop",
        }.get(neighbor["strand"], "")

    result.update(
        {
            f"{prefix}boundary_status": status,
            f"{prefix}intergenic_length": intergenic_length,
            f"{prefix}neighbor_cds_start": neighbor["accession_start"],
            f"{prefix}neighbor_cds_stop": neighbor["accession_stop"],
            f"{prefix}neighbor_cds_strand": (
                "+"
                if neighbor["accession_strand"] == 1
                else "-"
                if neighbor["accession_strand"] == -1
                else ""
            ),
            f"{prefix}neighbor_boundary_type": proximal_type,
            f"{prefix}neighbor_partial": neighbor["partial"],
        }
    )
    return result

def _maximal_intergenic_length(
    side,
    neighbor,
    central_start,
    central_end,
    flank_length,
):

    if side == "left":
        inferred = central_start - neighbor["end_min"]
    else:
        inferred = neighbor["start_max"] - central_end
    return max(0, min(flank_length, inferred))

def _position_bounds(position, record_length):

    kind = type(position).__name__
    if kind == "UnknownPosition":
        return 0, record_length
    value = int(position)
    if kind == "ExactPosition":
        lower, upper = value, value
    elif kind == "BeforePosition":
        lower, upper = 0, value
    elif kind == "AfterPosition":
        lower, upper = value, record_length
    elif kind in {"WithinPosition", "BetweenPosition"}:
        lower, upper = int(position._left), int(position._right)
    elif kind == "OneOfPosition":
        choices = [
            _position_bounds(choice, record_length)
            for choice in position.position_choices
        ]
        choices = [choice for choice in choices if choice is not None]
        if not choices:
            return None
        lower = min(choice[0] for choice in choices)
        upper = max(choice[1] for choice in choices)
    else:
        lower, upper = 0, record_length
    lower = max(0, min(record_length, lower))
    upper = max(lower, min(record_length, upper))
    return lower, upper

def _position_is_exact(position):
    return type(position).__name__ == "ExactPosition"

def _position_is_unknown(position):
    return type(position).__name__ == "UnknownPosition"

def _extract_sequences(raw_sequence, context):
    sequence = Seq(str(raw_sequence).upper())
    start = int(context["start"])
    stop = int(context["stop"])
    cds_length = stop - start + 1
    fetch_start, _ = segment_downloader.fetch_bounds(
        context,
        cfg.FLANK_LENGTH,
    )
    actual_fetch_stop = fetch_start + len(sequence) - 1

    if start < fetch_start or stop > actual_fetch_stop:
        raise ValueError(
            f"Fetched interval {fetch_start}-{actual_fetch_stop} does not "
            f"contain CDS {start}-{stop}."
        )

    if context["strand"] == "+":
        cds_offset = start - fetch_start
        gene_oriented = sequence
    elif context["strand"] == "-":
        cds_offset = actual_fetch_stop - stop
        gene_oriented = sequence.reverse_complement()
    else:
        raise ValueError(f"Unsupported strand: {context['strand']}")

    cds = gene_oriented[cds_offset:cds_offset + cds_length]
    if len(cds) != cds_length:
        raise ValueError(
            f"Expected {cds_length} CDS bases; extracted {len(cds)}."
        )

    gene_left = gene_oriented[:cds_offset]
    gene_right = gene_oriented[cds_offset + cds_length:]
    return {
        "segment": gene_oriented,
        "cds": cds,
        "left": gene_left.reverse_complement(),
        "right": gene_right,
    }

def _matching_translation(cds, expected):
    first_translation = ""
    for table in cfg.TRANSLATION_TABLES:
        try:
            strict = str(cds.translate(table=table, cds=True))
        except TranslationError:
            strict = ""
        if not first_translation and strict:
            first_translation = strict
        if strict == expected:
            return {
                "matched": True,
                "table": table,
                "mode": "complete_cds",
                "translated_length": len(strict),
            }

        relaxed = str(cds.translate(table=table))
        if relaxed.endswith("*"):
            relaxed = relaxed[:-1]
        if not first_translation:
            first_translation = relaxed
        if relaxed == expected:
            return {
                "matched": True,
                "table": table,
                "mode": "exact_translation",
                "translated_length": len(relaxed),
            }

    return {
        "matched": False,
        "table": "",
        "mode": "",
        "translated_length": len(first_translation),
    }

def _sequence_records(row, context, extracted):
    description = (
        f"assembly={context['canonical_assembly']} "
        f"nucleotide={context['nucleotide_accession']} "
        f"cds={context['start']}-{context['stop']} "
        f"strand={context['strand']} "
        f"authoritative={str(context['authoritative_context']).lower()} "
        f"left_length={row['left_length']} "
        f"right_length={row['right_length']}"
    )
    return {
        "row": row,
        "segment": SeqRecord(
            extracted["segment"],
            id=row["sequence_id"],
            description=f"{description} artifact=segment gene_oriented=true",
        ),
        "cds": SeqRecord(
            extracted["cds"],
            id=row["sequence_id"],
            description=f"{description} artifact=cds",
        ),
        "left": SeqRecord(
            extracted["left"],
            id=row["sequence_id"],
            description=f"{description} artifact=left proximal_first=true",
        ),
        "right": SeqRecord(
            extracted["right"],
            id=row["sequence_id"],
            description=f"{description} artifact=right proximal_first=true",
        ),
    }

def _write_protein_fastas(paths, records):
    records.sort(key=lambda item: int(item["row"]["selection_rank"]))
    usable_left = [
        item
        for item in records
        if int(item["row"]["left_length"]) >= cfg.MIN_FLANK_LENGTH
    ]
    usable_right = [
        item
        for item in records
        if int(item["row"]["right_length"]) >= cfg.MIN_FLANK_LENGTH
    ]

    _write_fasta_atomic(
        paths["validated_segment"],
        [item["segment"] for item in records],
    )
    _write_fasta_atomic(
        paths["validated_cds"],
        [item["cds"] for item in records],
    )
    _write_fasta_atomic(
        paths["validated_left"],
        [item["left"] for item in records],
    )
    _write_fasta_atomic(
        paths["validated_right"],
        [item["right"] for item in records],
    )
    _write_fasta_atomic(
        paths["usable_left"],
        [item["left"] for item in usable_left],
    )
    _write_fasta_atomic(
        paths["usable_right"],
        [item["right"] for item in usable_right],
    )

def _validate_cluster_configuration(minimum_similarity=None):
    if cfg.FLANK_CLUSTER_METHOD != "edlib_shw_prefix_star":
        raise ValueError(
            "FLANK_CLUSTER_METHOD must be 'edlib_shw_prefix_star'."
        )
    if cfg.FLANK_CLUSTER_AMBIGUITY_POLICY != "non_acgt_counts_as_edit":
        raise ValueError(
            "FLANK_CLUSTER_AMBIGUITY_POLICY must be "
            "'non_acgt_counts_as_edit'."
        )
    similarity = (
        float(cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY)
        if minimum_similarity is None
        else float(minimum_similarity)
    )
    if not 0.0 < similarity <= 1.0:
        raise ValueError(
            "FLANK_CLUSTER_MIN_PREFIX_SIMILARITY must be in (0, 1]."
        )
    return similarity

def _comparison_sequence(sequence, ambiguity_symbol):
    return "".join(
        base if base in "ACGT" else ambiguity_symbol
        for base in str(sequence).upper()
    )

def _prefix_edit_distance(member, representative, maximum_distance=None):
    member = str(member).upper()
    representative = str(representative).upper()
    if not member:
        raise ValueError("Cannot cluster an empty flank sequence.")
    if len(member) > len(representative):
        raise ValueError(
            "Prefix-cluster members cannot be longer than their representative."
        )

    arguments = {}
    if maximum_distance is not None:
        arguments["k"] = int(maximum_distance)
    result = edlib.align(
        _comparison_sequence(member, "N"),
        _comparison_sequence(representative, "X"),
        mode="SHW",
        task="distance",
        **arguments,
    )
    return int(result["editDistance"])

def _maximum_prefix_edits(member_length, minimum_similarity):
    if member_length <= 0:
        raise ValueError("Cannot cluster an empty flank sequence.")
    return math.floor(
        (1.0 - float(minimum_similarity)) * member_length + 1e-12
    )

def _qualifying_prefix_distance(member, representative, minimum_similarity):
    maximum_distance = _maximum_prefix_edits(
        len(member),
        minimum_similarity,
    )
    distance = _prefix_edit_distance(
        member,
        representative,
        maximum_distance=maximum_distance,
    )
    return None if distance < 0 else distance

def redundant_flank_distance(member, representative):
    return _qualifying_prefix_distance(
        member,
        representative,
        _validate_cluster_configuration(),
    )

def _cluster_state(protein_accession, side, rows_by_id, records_by_id):
    minimum_similarity = _validate_cluster_configuration()
    return {
        "format_version": 1,
        "protein_accession": protein_accession,
        "side": side,
        "minimum_flank_length": int(cfg.MIN_FLANK_LENGTH),
        "sequences": [
            {
                "sequence_id": sequence_id,
                "context_id": rows_by_id[sequence_id]["context_id"],
                "selection_rank": int(
                    rows_by_id[sequence_id]["selection_rank"]
                ),
                "authoritative_context": _as_bool(
                    rows_by_id[sequence_id]["authoritative_context"]
                ),
                "sha256": hashlib.sha256(
                    str(records_by_id[sequence_id].seq).upper().encode("ascii")
                ).hexdigest(),
            }
            for sequence_id in sorted(records_by_id)
        ],
        "parameters": {
            "method": cfg.FLANK_CLUSTER_METHOD,
            "minimum_prefix_similarity": minimum_similarity,
            "ambiguity_policy": cfg.FLANK_CLUSTER_AMBIGUITY_POLICY,
            "membership_rule": "direct_representative_star",
            "representative_order": (
                "length_desc_authoritative_desc_rank_asc_id_asc"
            ),
        },
    }

def _cached_cluster_rows(
    cluster_path,
    state_path,
    expected_state,
    records_by_id,
):
    if not cluster_path.exists() or not state_path.exists():
        return None
    with state_path.open(encoding="utf-8") as handle:
        if json.load(handle) != expected_state:
            return None

    rows = _read_tsv(cluster_path)
    sequence_ids = [row["sequence_id"] for row in rows]
    if (
        len(sequence_ids) != len(records_by_id)
        or set(sequence_ids) != set(records_by_id)
    ):
        return None
    _validate_cluster_rows(
        rows,
        records_by_id,
        expected_state["protein_accession"],
    )
    return rows

def _cluster_rows(
    protein_accession,
    rows_by_id,
    records_by_id,
    minimum_similarity=None,
):
    minimum_similarity = _validate_cluster_configuration(minimum_similarity)
    if set(rows_by_id) != set(records_by_id):
        raise ValueError(
            f"Context rows and flank sequences disagree for {protein_accession}."
        )

    ordered = sorted(
        records_by_id,
        key=lambda sequence_id: (
            -len(records_by_id[sequence_id].seq),
            not _as_bool(
                rows_by_id[sequence_id]["authoritative_context"]
            ),
            int(rows_by_id[sequence_id]["selection_rank"]),
            sequence_id,
        ),
    )
    assigned = set()
    clusters = []
    for representative in ordered:
        if representative in assigned:
            continue
        assigned.add(representative)
        members = [(representative, 0)]
        representative_sequence = str(records_by_id[representative].seq)
        for member in ordered:
            if member in assigned:
                continue
            distance = _qualifying_prefix_distance(
                str(records_by_id[member].seq),
                representative_sequence,
                minimum_similarity,
            )
            if distance is None:
                continue
            assigned.add(member)
            members.append((member, distance))
        clusters.append((representative, members))

    clusters.sort(
        key=lambda item: (
            int(rows_by_id[item[0]]["selection_rank"]),
            item[0],
        )
    )

    rows = []
    for cluster_number, (representative, members) in enumerate(
        clusters,
        start=1,
    ):
        cluster_id = f"cluster_{cluster_number:04d}"
        member_distances = dict(members)
        for member in sorted(
            member_distances,
            key=lambda sequence_id: (
                int(rows_by_id[sequence_id]["selection_rank"]),
                sequence_id,
            ),
        ):
            context = rows_by_id[member]
            rows.append(
                {
                    "protein_accession": protein_accession,
                    "context_id": context["context_id"],
                    "sequence_id": member,
                    "selection_rank": context["selection_rank"],
                    "authoritative_context": context[
                        "authoritative_context"
                    ],
                    "flank_length": len(records_by_id[member].seq),
                    "cluster_id": cluster_id,
                    "cluster_number": cluster_number,
                    "cluster_size": len(member_distances),
                    "selected_representative": member == representative,
                    "representative_sequence_id": representative,
                    "clustering_method": cfg.FLANK_CLUSTER_METHOD,
                    "minimum_prefix_similarity": minimum_similarity,
                    "ambiguity_policy": (
                        cfg.FLANK_CLUSTER_AMBIGUITY_POLICY
                    ),
                    "prefix_edit_distance": member_distances[member],
                }
            )
    return rows

def _update_cluster_summary(
    protein,
    summary,
    side,
    usable_rows,
    cluster_rows,
):
    cluster_count = len({row["cluster_id"] for row in cluster_rows})
    is_anchor = bool(protein["experimental_anchor"])
    if cluster_count == 0:
        eligible = False
        reason = "no_usable_flanks"
    elif is_anchor:
        eligible = True
        reason = "experimental_anchor_minimum_bypass"
    elif cluster_count >= cfg.MIN_CONTEXTS:
        eligible = True
        reason = "sequence_flank_minimum_met"
    else:
        eligible = False
        reason = "below_sequence_flank_minimum"

    summary[f"{side}_clusters"] = cluster_count
    summary[f"{side}_collapsed_flanks"] = len(usable_rows) - cluster_count
    summary[f"{side}_sequence_eligible"] = eligible
    summary[f"{side}_eligibility_reason"] = reason
    summary[f"{side}_authoritative_usable"] = sum(
        _as_bool(row["authoritative_context"])
        for row in usable_rows
    )
    summary[f"{side}_authoritative_representative"] = sum(
        _as_bool(row["authoritative_context"])
        and _as_bool(row["selected_representative"])
        for row in cluster_rows
    )
    summary[f"{side}_finalized_flanks"] = ""

def _usable_side_rows(context_rows, side):
    length_field = f"{side}_length"
    return [
        row
        for row in context_rows
        if row["status"] == "valid"
        and int(row[length_field]) >= cfg.MIN_FLANK_LENGTH
    ]

def _validate_cluster_rows(cluster_rows, records_by_id, protein_accession):
    expected = (
        cfg.FLANK_CLUSTER_METHOD,
        float(cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY),
        cfg.FLANK_CLUSTER_AMBIGUITY_POLICY,
    )
    groups = {}
    for row in cluster_rows:
        current = (
            row["clustering_method"],
            float(row["minimum_prefix_similarity"]),
            row["ambiguity_policy"],
        )
        if current != expected:
            raise ValueError(
                f"Flank clustering settings changed for {protein_accession}; "
                "rerun clustering before finalization."
            )
        sequence_id = row["sequence_id"]
        if sequence_id not in records_by_id:
            raise ValueError(
                f"Unknown cluster member for {protein_accession}: {sequence_id}"
            )
        flank_length = len(records_by_id[sequence_id].seq)
        if int(row["flank_length"]) != flank_length:
            raise ValueError(
                f"Stored flank length is stale for {protein_accession}: "
                f"{sequence_id}"
            )
        distance = int(row["prefix_edit_distance"])
        if not 0 <= distance <= _maximum_prefix_edits(
            flank_length,
            expected[1],
        ):
            raise ValueError(
                f"Invalid prefix edit distance for {protein_accession}: "
                f"{sequence_id}"
            )
        groups.setdefault(row["cluster_id"], []).append(row)

    for cluster_id, members in groups.items():
        representative_ids = {
            row["representative_sequence_id"] for row in members
        }
        selected = [
            row for row in members if _as_bool(row["selected_representative"])
        ]
        if (
            len(representative_ids) != 1
            or len(selected) != 1
            or selected[0]["sequence_id"] not in representative_ids
            or int(selected[0]["prefix_edit_distance"]) != 0
            or any(int(row["cluster_size"]) != len(members) for row in members)
        ):
            raise ValueError(
                f"Invalid cluster membership for {protein_accession}: "
                f"{cluster_id}"
            )

def _context_row(protein, context):
    context_id = f"context_{int(context['selection_rank']):04d}"
    sequence_id = f"{protein['protein_accession']}|{context_id}"
    fetch_start, fetch_stop = segment_downloader.fetch_bounds(
        context,
        cfg.FLANK_LENGTH,
    )
    return {
        "protein_accession": protein["protein_accession"],
        "candidate_group": protein["candidate_group"],
        "experimental_anchor": protein["experimental_anchor"],
        "context_id": context_id,
        "sequence_id": sequence_id,
        "selection_rank": context["selection_rank"],
        "canonical_context_id": context["canonical_context_id"],
        "raw_context_id": context["raw_context_id"],
        "canonical_assembly": context["canonical_assembly"],
        "assembly": context["assembly"],
        "nucleotide_accession": context["nucleotide_accession"],
        "start": context["start"],
        "stop": context["stop"],
        "strand": context["strand"],
        "authoritative_context": context["authoritative_context"],
        "target_protein_block": context["target_protein_block"],
        "member_protein_accession": context[
            "member_protein_accession"
        ],
        "taxid": context["taxid"],
        "organism": context["organism"],
        "strain": context["strain"],
        "manifest_selection_reason": context[
            "manifest_selection_reason"
        ],
        "requested_fetch_start": fetch_start,
        "requested_fetch_stop": fetch_stop,
        "raw_path": "",
        "download_status": "",
        "central_cds_annotated": "",
        "left_boundary_source": "",
        "left_boundary_status": "",
        "left_intergenic_length": "",
        "left_neighbor_cds_start": "",
        "left_neighbor_cds_stop": "",
        "left_neighbor_cds_strand": "",
        "left_neighbor_boundary_type": "",
        "left_neighbor_partial": "",
        "right_boundary_source": "",
        "right_boundary_status": "",
        "right_intergenic_length": "",
        "right_neighbor_cds_start": "",
        "right_neighbor_cds_stop": "",
        "right_neighbor_cds_strand": "",
        "right_neighbor_boundary_type": "",
        "right_neighbor_partial": "",
        "status": "",
        "error": "",
        "retrieved_length": "",
        "cds_length": "",
        "left_length": "",
        "right_length": "",
        "translation_table": "",
        "translation_mode": "",
        "translated_length": "",
    }

def _protein_summary(protein, context_rows):
    valid = [row for row in context_rows if row["status"] == "valid"]
    usable_left = _usable_side_rows(valid, "left")
    usable_right = _usable_side_rows(valid, "right")
    return {
        "protein_accession": protein["protein_accession"],
        "candidate_group": protein["candidate_group"],
        "experimental_anchor": protein["experimental_anchor"],
        "requested_contexts": len(context_rows),
        "downloaded_contexts": sum(
            row["download_status"] == "downloaded"
            for row in context_rows
        ),
        "cached_contexts": sum(
            row["download_status"] == "cached"
            for row in context_rows
        ),
        "valid_contexts": len(valid),
        "usable_left_flanks": len(usable_left),
        "usable_right_flanks": len(usable_right),
        "failed_contexts": len(context_rows) - len(valid),
        "translation_mismatches": sum(
            row["status"] == "translation_mismatch"
            for row in context_rows
        ),
        "left_clusters": "",
        "left_collapsed_flanks": "",
        "left_sequence_eligible": "",
        "left_eligibility_reason": "",
        "left_authoritative_usable": "",
        "left_authoritative_representative": "",
        "left_finalized_flanks": "",
        "right_clusters": "",
        "right_collapsed_flanks": "",
        "right_sequence_eligible": "",
        "right_eligibility_reason": "",
        "right_authoritative_usable": "",
        "right_authoritative_representative": "",
        "right_finalized_flanks": "",
    }

def _protein_paths(protein_accession):
    directory = cfg.EXTRACTIONS_DIR / protein_accession
    prefix = directory / protein_accession
    return {
        "directory": directory,
        "contexts": directory / "contexts.tsv",
        "validated_segment": Path(f"{prefix}.validated.segment.fna"),
        "validated_cds": Path(f"{prefix}.validated.cds.fna"),
        "validated_left": Path(f"{prefix}.validated.left.fna"),
        "validated_right": Path(f"{prefix}.validated.right.fna"),
        "usable_left": Path(f"{prefix}.usable.left.fna"),
        "usable_right": Path(f"{prefix}.usable.right.fna"),
        "left_clusters": directory / "left_clusters.tsv",
        "right_clusters": directory / "right_clusters.tsv",
        "left_cluster_state": directory / "left_cluster_state.json",
        "right_cluster_state": directory / "right_cluster_state.json",
        "nr_left": Path(f"{prefix}.nr.left.fna"),
        "nr_right": Path(f"{prefix}.nr.right.fna"),
    }

def _load_active_manifest():
    validation = step1.validate_active_manifest()
    if validation is None:
        raise FileNotFoundError(
            f"Active manifest does not exist: {cfg.ACTIVE_MANIFEST_PATH}"
        )

    rows = []
    with cfg.ACTIVE_MANIFEST_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _records_by_id(path):
    if not path.exists():
        raise FileNotFoundError(path)
    records = list(SeqIO.parse(path, "fasta"))
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate FASTA IDs in {path}")
    return by_id

def _read_tsv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def _write_tsv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
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
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: "" if row.get(field) is None else row.get(field, "")
                        for field in fieldnames
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def _write_fasta_atomic(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            SeqIO.write(records, handle, "fasta")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download, validate, and deduplicate Stage 05C flanks."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("download", "cluster", "finalize", "run"),
    )
    return parser.parse_args()

def main():
    action = _parse_args().action
    if action == "download":
        download_and_extract()
    elif action == "cluster":
        cluster_usable_flanks()
    elif action == "finalize":
        finalize_nonredundant_fastas()
    else:
        run_step2()

if __name__ == "__main__":
    main()
