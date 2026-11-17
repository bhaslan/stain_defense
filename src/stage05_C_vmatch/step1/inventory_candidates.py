import argparse
import csv
import hashlib
import json
import pathlib
import shutil
from collections import Counter, defaultdict

import pandas as pd
from Bio import SeqIO

from src import config_main as main_cfg
from src import roster
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step1 import report_ipg_contexts as ipg

CANDIDATE_GROUP_ORDER = {
    "canonical_hth_architecture": 0,
    "canonical_two_hth_architecture": 1,
    "canonical_without_hth_architecture": 2,
    "other_schlafen_architecture": 3,
    "experimental_anchor_only": 4,
}

ARCHITECTURE_CANDIDATE_GROUPS = frozenset(CANDIDATE_GROUP_ORDER) - {
    "experimental_anchor_only"
}

BLAST_FIELDS = [
    "in_blast",
    "blast_hsp_count",
    "blast_evalue",
    "blast_bits",
    "blast_score",
    "blast_qcov",
    "blast_pident",
    "blast_start",
    "blast_end",
    "blast_description",
]

AVAILABILITY_FIELDS = [
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
    *BLAST_FIELDS,
    "in_mapped_1",
    "in_addendum",
    "mapped_context_source",
    "mapped_nucleotide_accession",
    "mapped_assembly",
    "addendum_nucleotide_accession",
    "addendum_assembly",
    "expected_sequence_found",
    "expected_sequence_length",
    "ipg_status",
    "ipg_path",
    "ipg_report_count",
    "target_protein_block_count",
    "raw_context_count",
    "valid_locus_count",
    "assembly_locus_count",
    "raw_unique_nucleotide_count",
    "raw_unique_assembly_count",
    "canonical_assembly_count",
    "paired_assembly_reduction",
    "selected_context_count",
    "contexts_capped",
]

CONTEXT_FIELDS = [
    "protein_accession",
    "candidate_group",
    "is_experimental_anchor",
    "raw_context_id",
    "context_source",
    "ipg_path",
    "ipg_report_index",
    "ipg_id",
    "product_accession",
    "product_name",
    "member_protein_accession",
    "member_source",
    "member_name",
    "member_priority",
    "nucleotide_accession",
    "start",
    "stop",
    "strand",
    "assembly",
    "canonical_assembly",
    "canonical_context_id",
    "taxid",
    "organism",
    "strain",
    "has_valid_interval",
    "has_assembly",
    "target_protein_block",
    "refseq_member",
    "refseq_nucleotide",
    "authoritative_context",
    "canonical_role",
    "primary_selection_reason",
    "canonical_selected_for_manifest",
    "selected_for_manifest",
    "manifest_selection_reason",
]

CANDIDATE_SUMMARY_FIELDS = [
    "protein_accession",
    "candidate_group",
    "selection_roles",
    "is_experimental_anchor",
    "anchor_name",
    "architecture",
    "hth_class",
    "hth_families",
    "architecture_source",
    "canonical_assembly_count",
    "selected_context_count",
    "contexts_capped",
    "in_blast",
    "blast_bits",
    "blast_qcov",
    "blast_pident",
    "expected_sequence_length",
]

def input_coverage():
    study_ids = set(roster.accessions())
    concordance_ids = set(_read_concordance()["seqid"])
    blast_ids = set(_read_blast_metrics()["protein_accession"])
    mapped_ids = set(_read_mapped()["blast_accession"])
    fasta_ids = set(_read_fasta())
    anchor_ids = set(_experimental_anchors())
    missing_ipgs = ipg.missing_ipg_accessions(study_ids)
    return {
        "study_proteins": len(study_ids),
        "concordance_proteins": len(concordance_ids),
        "blast_proteins": len(blast_ids),
        "mapped_1_proteins": len(mapped_ids),
        "fasta_proteins": len(fasta_ids),
        "experimental_anchors": len(anchor_ids),
        "study_not_in_concordance": len(study_ids - concordance_ids),
        "study_not_in_blast": len(study_ids - blast_ids),
        "study_not_in_mapped_1": len(study_ids - mapped_ids),
        "study_not_in_fasta": len(study_ids - fasta_ids),
        "concordance_not_in_study": len(concordance_ids - study_ids),
        "anchors_not_in_study": sorted(anchor_ids - study_ids),
        "missing_ipg_reports": len(missing_ipgs),
        "missing_ipg_accessions": missing_ipgs,
    }

def fetch_missing_ipgs():
    return ipg.fetch_missing_ipgs(sorted(roster.accessions()))

def build_context_inventory(destination=None):
    study = roster.read()
    concordance = _read_concordance()
    blast = _read_blast_metrics()
    mapped = _read_mapped()
    addendum = _read_addendum()
    sequences = roster.sequences()
    anchors = _experimental_anchors()
    pair_map = ipg.load_assembly_pair_map()

    dpam_by_id = _read_dpam_architectures(set(study["protein_accession"]))
    concordance_by_id = concordance.set_index("seqid").to_dict("index")
    blast_by_id = blast.set_index("protein_accession").to_dict("index")
    mapped_by_id = mapped.set_index("blast_accession").to_dict("index")
    addendum_by_id = addendum.set_index("blast_accession").to_dict("index")

    availability_rows = []
    context_rows = []

    for protein_accession in study["protein_accession"]:
        concordance_row = concordance_by_id.get(protein_accession, {})
        anchor = anchors.get(protein_accession)
        dpam_row = dpam_by_id[protein_accession]
        candidate_group = _candidate_group(concordance_row)
        if anchor and not candidate_group:
            candidate_group = "experimental_anchor_only"

        report = ipg.parse_ipg_report(
            ipg.resolve_ipg_path(protein_accession),
            protein_accession,
        )
        contexts = report["contexts"]

        if anchor:
            addendum_row = addendum_by_id.get(protein_accession)
            if addendum_row is None:
                raise ValueError(
                    f"Experimental anchor is absent from addendum.csv: "
                    f"{protein_accession}"
                )
            _mark_or_add_authoritative_context(
                contexts,
                protein_accession,
                anchor,
                addendum_row,
            )

        _canonicalize_contexts(contexts, pair_map)
        primary_contexts = _choose_canonical_primaries(contexts)

        canonical_count = len(primary_contexts)
        selected_protein = bool(anchor) or canonical_count >= cfg.MIN_CONTEXTS
        selected_contexts = []
        selection_reasons = {}
        if selected_protein:
            selected_contexts, selection_reasons = _select_contexts(
                protein_accession,
                primary_contexts,
            )
        if anchor and not any(
            context["authoritative_context"]
            for context in selected_contexts
        ):
            raise ValueError(
                f"Authoritative context was not selected for "
                f"{protein_accession}"
            )

        selected_ids = {
            context["raw_context_id"] for context in selected_contexts
        }
        selected_assemblies = {
            context["canonical_assembly"] for context in selected_contexts
        }
        for context in contexts:
            canonical = context["canonical_assembly"]
            context["protein_accession"] = protein_accession
            context["candidate_group"] = candidate_group
            context["is_experimental_anchor"] = bool(anchor)
            context["canonical_selected_for_manifest"] = (
                canonical in selected_assemblies
            )
            context["selected_for_manifest"] = (
                context["raw_context_id"] in selected_ids
            )
            context["manifest_selection_reason"] = selection_reasons.get(
                canonical,
                "",
            )
            context_rows.append(context)

        blast_row = blast_by_id.get(protein_accession, {})
        mapped_row = mapped_by_id.get(protein_accession, {})
        addendum_row = addendum_by_id.get(protein_accession, {})
        sequence = sequences[protein_accession]
        raw_assemblies = {
            context["assembly"]
            for context in contexts
            if context["has_valid_interval"] and context["assembly"]
        }
        raw_nucleotides = {
            context["nucleotide_accession"]
            for context in contexts
            if context["has_valid_interval"]
        }

        availability_rows.append(
            {
                "protein_accession": protein_accession,
                "candidate_group": candidate_group,
                "selection_roles": _selection_roles(
                    candidate_group,
                    bool(anchor),
                ),
                "selected_protein": selected_protein,
                "eligibility_reason": _eligibility_reason(
                    bool(anchor),
                    canonical_count,
                    report["status"],
                    sum(context["has_valid_interval"] for context in contexts),
                ),
                "is_experimental_anchor": bool(anchor),
                "anchor_name": anchor.get("name", "") if anchor else "",
                "trnase_activity": (
                    anchor.get("trnase_activity", "") if anchor else ""
                ),
                "dna_binding_activity": (
                    anchor.get("dna_binding_activity", "") if anchor else ""
                ),
                "architecture": dpam_row["architecture"],
                "hth_class": dpam_row["hth_class"],
                "hth_families": dpam_row["hth_families"],
                "n_hth": dpam_row["n_hth"],
                "architecture_source": dpam_row["architecture_source"],
                "min_judge": dpam_row["min_judge"],
                "cluster_id": dpam_row["cluster_id"],
                "representative_accession": dpam_row["representative_accession"],
                "cluster_size": dpam_row["cluster_size"],
                "interpro_architecture": concordance_row.get("architecture", ""),
                "base_architecture": concordance_row.get(
                    "base_architecture",
                    "",
                ),
                "has_local_hth": concordance_row.get("has_local_hth", ""),
                "local_hth_status": concordance_row.get(
                    "local_HTH_status",
                    "",
                ),
                "hth_neighbor_fraction": concordance_row.get(
                    "hth_neighbor_fraction",
                    "",
                ),
                "protein_length": concordance_row.get("length", ""),
                **_blast_metadata(blast_row),
                "in_mapped_1": bool(mapped_row),
                "in_addendum": bool(addendum_row),
                "mapped_context_source": mapped_row.get(
                    "context_source",
                    "",
                ),
                "mapped_nucleotide_accession": _text(
                    mapped_row.get("accver")
                ),
                "mapped_assembly": _text(mapped_row.get("assembly")),
                "addendum_nucleotide_accession": _text(
                    addendum_row.get("accver")
                ),
                "addendum_assembly": _text(
                    addendum_row.get("assembly")
                ),
                "expected_sequence_found": bool(sequence),
                "expected_sequence_length": len(sequence),
                "ipg_status": report["status"],
                "ipg_path": report["path"],
                "ipg_report_count": report["report_count"],
                "target_protein_block_count": report["target_block_count"],
                "raw_context_count": len(contexts),
                "valid_locus_count": sum(
                    context["has_valid_interval"] for context in contexts
                ),
                "assembly_locus_count": sum(
                    context["has_valid_interval"] and context["has_assembly"]
                    for context in contexts
                ),
                "raw_unique_nucleotide_count": len(raw_nucleotides),
                "raw_unique_assembly_count": len(raw_assemblies),
                "canonical_assembly_count": canonical_count,
                "paired_assembly_reduction": (
                    len(raw_assemblies) - canonical_count
                ),
                "selected_context_count": len(selected_contexts),
                "contexts_capped": canonical_count > cfg.MAX_CONTEXTS,
            }
        )

    availability_path = _output_path(destination, cfg.CONTEXT_AVAILABILITY_PATH)
    records_path = _output_path(destination, cfg.CONTEXT_RECORDS_PATH)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(availability_path, AVAILABILITY_FIELDS, availability_rows)
    _write_tsv(records_path, CONTEXT_FIELDS, context_rows)

    print(f"Wrote {len(availability_rows):,} proteins to {availability_path}")
    ipg_statuses = Counter(
        row["ipg_status"].split(":", 1)[0] for row in availability_rows
    )
    for status, count in sorted(ipg_statuses.items()):
        print(f"  ipg_status {status}: {count:,}")
    print(f"Wrote {len(context_rows):,} raw loci to {records_path}")
    canonical_roles = Counter(row["canonical_role"] for row in context_rows)
    for role, count in sorted(canonical_roles.items()):
        print(f"  canonical_role {role}: {count:,}")
    return pd.DataFrame(availability_rows), pd.DataFrame(context_rows)

def build_candidate_manifest(*, sync_active=True, destination=None):
    summary_path = _output_path(destination, cfg.CANDIDATE_SUMMARY_PATH)
    full_path = _output_path(destination, cfg.FULL_MANIFEST_PATH)
    active_path = _output_path(destination, cfg.ACTIVE_MANIFEST_PATH)
    availability = pd.read_csv(
        _output_path(destination, cfg.CONTEXT_AVAILABILITY_PATH),
        sep="\t",
        keep_default_na=False,
    )
    contexts = pd.read_csv(
        _output_path(destination, cfg.CONTEXT_RECORDS_PATH),
        sep="\t",
        keep_default_na=False,
    )
    selected = availability[
        availability["selected_protein"].map(_boolean)
    ].copy()
    selected["_group_order"] = (
        selected["candidate_group"]
        .map(CANDIDATE_GROUP_ORDER)
        .fillna(len(CANDIDATE_GROUP_ORDER))
    )
    selected = selected.sort_values(
        ["_group_order", "protein_accession"],
        kind="stable",
    ).drop(columns="_group_order")

    sequences = roster.sequences()
    anchors = _experimental_anchors()

    context_groups = {
        key: rows.to_dict("records")
        for key, rows in contexts.groupby(
            ["protein_accession", "canonical_assembly"],
            sort=False,
        )
    }

    manifest_rows = []
    for row in selected.to_dict("records"):
        protein_accession = row["protein_accession"]
        selected_primary_rows = contexts[
            (contexts["protein_accession"] == protein_accession)
            & contexts["selected_for_manifest"].map(_boolean)
        ].copy()
        selected_primary_rows["_reason_order"] = (
            selected_primary_rows["manifest_selection_reason"].map(
                {
                    "authoritative_context": 0,
                    "taxon_or_organism_diversity": 1,
                    "strain_diversity": 2,
                    "remaining_canonical_assembly": 3,
                    "all_available": 4,
                }
            )
        )
        selected_primary_rows = selected_primary_rows.sort_values(
            ["_reason_order", "canonical_assembly"],
            kind="stable",
        )

        manifest_contexts = []
        for selection_rank, primary in enumerate(
            selected_primary_rows.to_dict("records"),
            start=1,
        ):
            key = (
                protein_accession,
                primary["canonical_assembly"],
            )
            representations = context_groups[key]
            alternates = [
                _manifest_representation(context)
                for context in representations
                if context["raw_context_id"] != primary["raw_context_id"]
            ]
            manifest_context = _manifest_representation(primary)
            manifest_context.update(
                {
                    "canonical_context_id": primary["canonical_context_id"],
                    "selection_rank": selection_rank,
                    "primary_selection_reason": primary[
                        "primary_selection_reason"
                    ],
                    "manifest_selection_reason": primary[
                        "manifest_selection_reason"
                    ],
                    "alternate_representations": sorted(
                        alternates,
                        key=lambda value: value["raw_context_id"],
                    ),
                }
            )
            manifest_contexts.append(manifest_context)

        anchor = anchors.get(protein_accession)
        manifest_rows.append(
            {
                "protein_accession": protein_accession,
                "candidate_group": row["candidate_group"],
                "selection_roles": [
                    role
                    for role in row["selection_roles"].split(";")
                    if role
                ],
                "experimental_anchor": (
                    {
                        "name": anchor["name"],
                        "authoritative_nucleotide_accession": anchor[
                            "authoritative_nucleotide_accession"
                        ],
                        "trnase_activity": anchor["trnase_activity"],
                        "dna_binding_activity": anchor[
                            "dna_binding_activity"
                        ],
                    }
                    if anchor
                    else None
                ),
                "domains": {
                    "architecture": row["architecture"],
                    "hth_class": row["hth_class"],
                    "n_hth": _integer_or_none(row["n_hth"]),
                    "architecture_source": row["architecture_source"],
                    "min_judge": row["min_judge"],
                    "cluster_id": row["cluster_id"],
                    "representative_accession": row["representative_accession"],
                    "cluster_size": _integer_or_none(row["cluster_size"]),
                },
                "concordance": {
                    "interpro_architecture": row["interpro_architecture"],
                    "base_architecture": row["base_architecture"],
                    "has_local_hth": _boolean(row["has_local_hth"]),
                    "local_hth_status": row["local_hth_status"],
                    "hth_neighbor_fraction": _float_or_none(
                        row["hth_neighbor_fraction"]
                    ),
                    "protein_length": _integer_or_none(
                        row["protein_length"]
                    ),
                },
                "blast": _manifest_blast(row),
                "provenance": {
                    "in_mapped_1": _boolean(row["in_mapped_1"]),
                    "in_addendum": _boolean(row["in_addendum"]),
                    "mapped_context_source": row["mapped_context_source"],
                    "mapped_nucleotide_accession": row[
                        "mapped_nucleotide_accession"
                    ],
                    "mapped_assembly": row["mapped_assembly"],
                    "addendum_nucleotide_accession": row[
                        "addendum_nucleotide_accession"
                    ],
                    "addendum_assembly": row["addendum_assembly"],
                },
                "expected_protein_sequence": sequences[protein_accession],
                "expected_protein_length": len(sequences[protein_accession]),
                "context_counts": {
                    "raw_loci": _integer_or_none(row["raw_context_count"]),
                    "raw_assemblies": _integer_or_none(
                        row["raw_unique_assembly_count"]
                    ),
                    "canonical_assemblies": _integer_or_none(
                        row["canonical_assembly_count"]
                    ),
                    "selected_contexts": len(manifest_contexts),
                },
                "selected_contexts": manifest_contexts,
            }
        )

    full_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(summary_path, CANDIDATE_SUMMARY_FIELDS, selected.to_dict("records"))
    with full_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    print(f"Wrote {len(manifest_rows):,} proteins to {full_path}")
    print(f"Summary: {summary_path}")
    if sync_active:
        shutil.copyfile(full_path, active_path)
        print(
            f"Synced active manifest to full ({len(manifest_rows):,} proteins): "
            f"{active_path}"
        )
    else:
        print(
            "Active manifest was not created or modified "
            f"(--preserve-active): {active_path}"
        )
    return manifest_rows

def validate_active_manifest():
    if not cfg.ACTIVE_MANIFEST_PATH.exists():
        print(f"Active manifest does not exist: {cfg.ACTIVE_MANIFEST_PATH}")
        return None

    full_rows = _read_jsonl(cfg.FULL_MANIFEST_PATH)
    active_rows = _read_jsonl(cfg.ACTIVE_MANIFEST_PATH)
    full_by_id = {
        row["protein_accession"]: row
        for row in full_rows
    }
    active_ids = [row.get("protein_accession") for row in active_rows]

    if len(active_ids) != len(set(active_ids)):
        raise ValueError("The active manifest contains duplicate proteins.")
    missing = sorted(set(active_ids) - set(full_by_id))
    if missing:
        raise ValueError(
            "Active proteins absent from the full manifest: "
            + ", ".join(missing)
        )
    changed = [
        protein_accession
        for protein_accession, row in zip(active_ids, active_rows)
        if row != full_by_id[protein_accession]
    ]
    if changed:
        raise ValueError(
            "Active rows must be complete, unchanged full-manifest lines: "
            + ", ".join(changed)
        )

    group_counts = Counter(row["candidate_group"] for row in active_rows)
    context_count = sum(
        len(row.get("selected_contexts", [])) for row in active_rows
    )
    print(f"Active proteins: {len(active_rows):,}")
    print(f"Active selected contexts: {context_count:,}")
    for group, count in sorted(
        group_counts.items(),
        key=lambda item: CANDIDATE_GROUP_ORDER.get(item[0], len(CANDIDATE_GROUP_ORDER)),
    ):
        print(f"{group or '(no architecture class)'}: {count:,}")
    return {
        "protein_count": len(active_rows),
        "context_count": context_count,
        "group_counts": dict(group_counts),
    }

def run_step1(*, sync_active=True, destination=None):
    build_context_inventory(destination=destination)
    return build_candidate_manifest(sync_active=sync_active, destination=destination)

def _read_concordance():
    rows = pd.read_csv(cfg.CONCORDANCE_PATH, sep="\t")
    if rows["seqid"].duplicated().any():
        raise ValueError("Concordance must contain one row per seqid.")
    return rows.sort_values("seqid", kind="stable").reset_index(drop=True)

def _read_dpam_architectures(accessions):
    rows = pd.read_csv(cfg.DPAM_PROTEIN_ARCHITECTURES_PATH)
    unannotated = sorted(accessions - set(rows["protein_accession"]))
    if unannotated:
        raise ValueError(
            f"{len(unannotated)} analysed proteins have no DPAM architecture "
            f"(for example {unannotated[0]}). Run "
            f"`python -m src.stage02_domains.pipeline step6_build_dpam_architectures`."
        )

    requested = rows[rows["protein_accession"].isin(accessions)]
    unresolved = sorted(
        requested.loc[requested["architecture_source"] == "unresolved", "protein_accession"]
    )
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} requested proteins have an unresolved DPAM architecture "
            f"(for example {unresolved[0]}). Supply an AF3+DPAM batch covering them, or "
            f"remove them from the roster, then rerun "
            f"`python -m src.stage02_domains.pipeline step6_build_dpam_architectures`."
        )
    return rows.set_index("protein_accession").to_dict("index")

def _read_blast_metrics():
    rows = pd.read_csv(cfg.BLAST_ALIGNMENT_PATH)
    hsp_counts = rows["Accession"].value_counts()
    selected = (
        rows.assign(_row_order=range(len(rows)))
        .sort_values(
            [
                "Accession",
                "E Value",
                "Bits",
                "QCov",
                "Pident",
                "_row_order",
            ],
            ascending=[True, True, False, False, False, True],
        )
        .drop_duplicates("Accession")
    )
    selected["blast_hsp_count"] = selected["Accession"].map(hsp_counts)
    return selected.rename(
        columns={
            "Accession": "protein_accession",
            "E Value": "blast_evalue",
            "Bits": "blast_bits",
            "Score": "blast_score",
            "QCov": "blast_qcov",
            "Pident": "blast_pident",
            "Start": "blast_start",
            "End": "blast_end",
            "Description": "blast_description",
        }
    )[
        [
            "protein_accession",
            "blast_hsp_count",
            "blast_evalue",
            "blast_bits",
            "blast_score",
            "blast_qcov",
            "blast_pident",
            "blast_start",
            "blast_end",
            "blast_description",
        ]
    ]

def _read_mapped():
    rows = pd.read_csv(cfg.MAPPED_PATH)
    if rows["blast_accession"].duplicated().any():
        raise ValueError("mapped_1.csv must contain one row per protein.")
    return rows

def _read_addendum():
    rows = pd.read_csv(cfg.ADDENDUM_PATH, encoding="utf-8-sig")
    if rows["blast_accession"].duplicated().any():
        raise ValueError("addendum.csv must contain one row per protein.")
    return rows

def _read_fasta():
    sequences = {}
    for record in SeqIO.parse(cfg.FASTA_PATH, "fasta"):
        sequence = str(record.seq).upper()
        if record.id in sequences and sequences[record.id] != sequence:
            raise ValueError(f"Conflicting FASTA records for {record.id}.")
        sequences[record.id] = sequence
    return sequences

def _experimental_anchors():
    anchors = {}
    for nucleotide_accession, metadata in main_cfg.the_nine.items():
        protein_accession = metadata["protein_acc"]
        anchors[protein_accession] = {
            "name": metadata["Name"],
            "authoritative_nucleotide_accession": nucleotide_accession,
            "trnase_activity": metadata["tRNAse_activity"],
            "dna_binding_activity": metadata["DNA-binding_activity"],
        }
    return anchors

def _candidate_group(concordance_row):
    architecture = concordance_row.get("architecture", "")
    if architecture == cfg.CANONICAL_HTH_ARCHITECTURE:
        return "canonical_hth_architecture"
    if architecture == cfg.CANONICAL_TWO_HTH_ARCHITECTURE:
        return "canonical_two_hth_architecture"
    if architecture == cfg.CANONICAL_WITHOUT_HTH_ARCHITECTURE:
        return "canonical_without_hth_architecture"
    if cfg.SCHLAFEN_DOMAIN in architecture:
        return "other_schlafen_architecture"
    return ""

def _mark_or_add_authoritative_context(
    contexts,
    protein_accession,
    anchor,
    addendum_row,
):
    authoritative = {
        "nucleotide_accession": str(addendum_row.get("accver", "")).strip(),
        "start": _integer_or_none(addendum_row.get("start")),
        "stop": _integer_or_none(addendum_row.get("stop")),
        "strand": str(addendum_row.get("strand", "")).strip(),
        "assembly": str(addendum_row.get("assembly", "")).strip(),
    }
    expected_nucleotide = anchor["authoritative_nucleotide_accession"]
    if authoritative["nucleotide_accession"] != expected_nucleotide:
        raise ValueError(
            f"Authoritative nucleotide mismatch for {protein_accession}: "
            f"{authoritative['nucleotide_accession']} != {expected_nucleotide}"
        )
    if not (
        authoritative["nucleotide_accession"]
        and authoritative["assembly"].startswith(("GCF_", "GCA_"))
        and authoritative["start"] is not None
        and authoritative["stop"] is not None
        and 1 <= authoritative["start"] <= authoritative["stop"]
        and authoritative["strand"] in {"+", "-"}
    ):
        raise ValueError(
            f"Incomplete authoritative addendum context for "
            f"{protein_accession}"
        )

    matches = [
        context
        for context in contexts
        if all(
            context[field] == authoritative[field]
            for field in (
                "nucleotide_accession",
                "start",
                "stop",
                "strand",
                "assembly",
            )
        )
    ]
    if matches:
        for context in matches:
            context["authoritative_context"] = True
        return

    contexts.append(
        {
            "protein_accession": protein_accession,
            "raw_context_id": f"{protein_accession}|authoritative_addendum",
            "context_source": "addendum",
            "ipg_path": "",
            "ipg_report_index": "",
            "ipg_id": "",
            "product_accession": protein_accession,
            "product_name": str(addendum_row.get("blast_description", "")),
            "member_protein_accession": protein_accession,
            "member_source": "RefSeq",
            "member_name": str(addendum_row.get("blast_description", "")),
            "member_priority": "",
            **authoritative,
            "taxid": _text(addendum_row.get("taxid")),
            "organism": _text(addendum_row.get("org")),
            "strain": _text(addendum_row.get("strain")),
            "target_protein_block": True,
            "refseq_member": True,
            "refseq_nucleotide": True,
            "authoritative_context": True,
            "has_valid_interval": True,
            "has_assembly": True,
        }
    )

def _canonicalize_contexts(contexts, pair_map):
    for context in contexts:
        assembly = context["assembly"]
        canonical = ""
        if context["has_valid_interval"] and assembly:
            canonical = pair_map.get(assembly, assembly)
        context["canonical_assembly"] = canonical
        context["canonical_context_id"] = (
            f"{context['protein_accession']}|{canonical}" if canonical else ""
        )
        context["canonical_role"] = "unusable"
        context["primary_selection_reason"] = ""
        context["canonical_selected_for_manifest"] = False
        context["selected_for_manifest"] = False
        context["manifest_selection_reason"] = ""

def _choose_canonical_primaries(contexts):
    groups = defaultdict(list)
    for context in contexts:
        if context["canonical_assembly"]:
            groups[context["canonical_assembly"]].append(context)

    primaries = []
    for canonical in sorted(groups):
        group = sorted(groups[canonical], key=_primary_context_sort_key)
        primary = group[0]
        reason = _primary_selection_reason(primary)
        for context in group:
            context["canonical_role"] = (
                "primary" if context is primary else "alternate"
            )
            context["primary_selection_reason"] = reason
        primaries.append(primary)
    return primaries

def _primary_context_sort_key(context):
    refseq_representation = (
        context["refseq_member"] or context["refseq_nucleotide"]
    )
    return (
        not context["authoritative_context"],
        not context["target_protein_block"],
        not refseq_representation,
        context["nucleotide_accession"],
        context["start"] if context["start"] is not None else 10**18,
        context["stop"] if context["stop"] is not None else 10**18,
        context["strand"],
        context["member_protein_accession"],
        context["raw_context_id"],
    )

def _primary_selection_reason(context):
    if context["authoritative_context"]:
        return "authoritative_context"
    if context["target_protein_block"]:
        return "exact_target_protein_block"
    if context["refseq_member"] or context["refseq_nucleotide"]:
        return "refseq_representation"
    return "deterministic_accession_coordinate"

def _select_contexts(protein_accession, contexts):
    ordered = sorted(
        contexts,
        key=lambda context: (
            not context["authoritative_context"],
            _stable_context_key(protein_accession, context),
        ),
    )
    if len(ordered) <= cfg.MAX_CONTEXTS:
        return ordered, {
            context["canonical_assembly"]: (
                "authoritative_context"
                if context["authoritative_context"]
                else "all_available"
            )
            for context in ordered
        }

    selected = []
    reasons = {}

    def add(context, reason):
        if context in selected or len(selected) >= cfg.MAX_CONTEXTS:
            return
        selected.append(context)
        reasons[context["canonical_assembly"]] = reason

    for context in ordered:
        if context["authoritative_context"]:
            add(context, "authoritative_context")

    _select_one_per_key(
        ordered,
        selected,
        add,
        _taxon_or_organism_key,
        "taxon_or_organism_diversity",
        skip_empty=True,
    )
    _select_one_per_key(
        ordered,
        selected,
        add,
        _organism_strain_key,
        "strain_diversity",
        skip_empty=True,
    )
    for context in ordered:
        add(context, "remaining_canonical_assembly")
    return selected, reasons

def _select_one_per_key(
    contexts,
    selected,
    add,
    key_function,
    reason,
    skip_empty=False,
):
    represented = {
        key_function(context)
        for context in selected
        if key_function(context)
    }
    for context in contexts:
        key = key_function(context)
        if skip_empty and not key:
            continue
        if key in represented:
            continue
        add(context, reason)
        represented.add(key)
        if len(selected) >= cfg.MAX_CONTEXTS:
            return

def _taxon_or_organism_key(context):
    taxid = _text(context["taxid"])
    organism = _text(context["organism"]).casefold()
    if taxid:
        return f"taxid:{taxid}"
    if organism:
        return f"organism:{organism}"
    return ""

def _organism_strain_key(context):
    organism = _taxon_or_organism_key(context)
    strain = _text(context["strain"]).casefold()
    if not organism or not strain:
        return ""
    return f"{organism}|strain:{strain}"

def _stable_context_key(protein_accession, context):
    text = (
        f"{cfg.SELECTION_RANDOM_SEED}|{protein_accession}|"
        f"{context['canonical_assembly']}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _selection_roles(candidate_group, is_anchor):
    roles = []
    if candidate_group and candidate_group != "experimental_anchor_only":
        roles.append(candidate_group)
    if is_anchor:
        roles.append("experimental_anchor")
    return ";".join(roles)

def _eligibility_reason(is_anchor, canonical_count, ipg_status, interval_count):
    if is_anchor and canonical_count < cfg.MIN_CONTEXTS:
        return "experimental_anchor_minimum_bypass"
    if is_anchor:
        return "experimental_anchor"
    if canonical_count >= cfg.MIN_CONTEXTS:
        return "context_threshold_met"
    if ipg_status != "ok":
        return "ipg_report_unusable"
    if interval_count:
        return "contexts_without_assembly_identity"
    return "below_context_minimum"

def _blast_metadata(row):
    if not row:
        return {field: False if field == "in_blast" else "" for field in BLAST_FIELDS}
    return {
        "in_blast": True,
        "blast_hsp_count": row.get("blast_hsp_count", ""),
        "blast_evalue": row.get("blast_evalue", ""),
        "blast_bits": row.get("blast_bits", ""),
        "blast_score": row.get("blast_score", ""),
        "blast_qcov": row.get("blast_qcov", ""),
        "blast_pident": row.get("blast_pident", ""),
        "blast_start": row.get("blast_start", ""),
        "blast_end": row.get("blast_end", ""),
        "blast_description": row.get("blast_description", ""),
    }

def _manifest_representation(row):
    return {
        "raw_context_id": row["raw_context_id"],
        "context_source": row["context_source"],
        "ipg_id": row["ipg_id"],
        "product_accession": row["product_accession"],
        "canonical_assembly": row["canonical_assembly"],
        "assembly": row["assembly"],
        "nucleotide_accession": row["nucleotide_accession"],
        "start": _integer_or_none(row["start"]),
        "stop": _integer_or_none(row["stop"]),
        "strand": row["strand"],
        "taxid": _text(row["taxid"]),
        "organism": row["organism"],
        "strain": row["strain"],
        "member_protein_accession": row["member_protein_accession"],
        "member_source": row["member_source"],
        "member_name": row["member_name"],
        "member_priority": _integer_or_none(row["member_priority"]),
        "target_protein_block": _boolean(row["target_protein_block"]),
        "refseq_member": _boolean(row["refseq_member"]),
        "refseq_nucleotide": _boolean(row["refseq_nucleotide"]),
        "authoritative_context": _boolean(row["authoritative_context"]),
    }

def _manifest_blast(row):
    return {
        "in_blast": _boolean(row["in_blast"]),
        "blast_hsp_count": _integer_or_none(row["blast_hsp_count"]),
        "blast_evalue": _float_or_none(row["blast_evalue"]),
        "blast_bits": _float_or_none(row["blast_bits"]),
        "blast_score": _float_or_none(row["blast_score"]),
        "blast_qcov": _float_or_none(row["blast_qcov"]),
        "blast_pident": _float_or_none(row["blast_pident"]),
        "blast_start": _integer_or_none(row["blast_start"]),
        "blast_end": _integer_or_none(row["blast_end"]),
        "blast_description": _text(row["blast_description"]),
    }

def _output_path(destination, path):
    return path if destination is None else destination / path.name

def _write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

def _read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict) or not row.get("protein_accession"):
                raise ValueError(
                    f"Missing protein_accession on {path}:{line_number}"
                )
            rows.append(row)
    return rows

def _integer_or_none(value):
    if value is None or _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def _float_or_none(value):
    if value is None or _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}

def _text(value):
    if value is None or _is_missing(value):
        return ""
    return str(value).strip()

def _is_missing(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Inventory and select Stage 05C genomic contexts."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=(
            "coverage",
            "fetch-missing-ipgs",
            "build-inventory",
            "build-manifest",
            "validate-active",
            "run",
        ),
    )
    parser.add_argument(
        "--preserve-active",
        action="store_true",
        help=(
            "With build-manifest or run: write only the full manifest and "
            "leave candidate_manifest.active.jsonl untouched (old default)."
        ),
    )
    parser.add_argument(
        "--into",
        metavar="DIR",
        help=(
            "Write the inventory and manifest into DIR instead of the published "
            "selections directory, leaving the current selection untouched."
        ),
    )
    return parser.parse_args()

def main():
    args = _parse_args()
    action = args.action
    sync_active = not args.preserve_active
    destination = pathlib.Path(args.into) if args.into else None
    if action == "coverage":
        print(json.dumps(input_coverage(), indent=2))
    elif action == "fetch-missing-ipgs":
        fetch_missing_ipgs()
    elif action == "build-inventory":
        build_context_inventory(destination=destination)
    elif action == "build-manifest":
        build_candidate_manifest(sync_active=sync_active, destination=destination)
    elif action == "validate-active":
        validate_active_manifest()
    else:
        run_step1(sync_active=sync_active, destination=destination)

if __name__ == "__main__":
    main()
