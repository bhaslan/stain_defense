import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm

from src.stage05_C_vmatch import config_vmatch as cfg

BOUNDARY_SUFFIXES = [
    "boundary_source",
    "boundary_status",
    "intergenic_length",
    "neighbor_cds_start",
    "neighbor_cds_stop",
    "neighbor_cds_strand",
    "neighbor_boundary_type",
    "neighbor_partial",
]

UNIT_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "scope",
    "search_unit_id",
    "source_context_id",
    "source_sequence_id",
    "selection_rank",
    "authoritative_context",
    "left_sequence_eligible",
    "right_sequence_eligible",
    "left_cluster_id",
    "left_cluster_size",
    "left_cluster_representative_sequence_id",
    "right_cluster_id",
    "right_cluster_size",
    "right_cluster_representative_sequence_id",
    "paired_cluster_id",
    "pair_support",
    "database_side",
    "query_side",
    "nucleotide_accession",
    "central_cds_start",
    "central_cds_stop",
    "central_cds_strand",
    "central_cds_annotated",
    "left_flank_length",
    "right_flank_length",
]
for _side in ("left", "right"):
    UNIT_FIELDS.extend(f"{_side}_{suffix}" for suffix in BOUNDARY_SUFFIXES)
UNIT_FIELDS.extend(
    [
        "database_fasta_path",
        "query_fasta_path",
        "unit_state_path",
        "unit_directory",
    ]
)

CALIBRATION_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "scope",
    "search_unit_id",
    "profile",
    "repeat_class",
    "match_length",
    "match_count",
    "total_matches",
    "calibration_path",
]

RUN_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "scope",
    "search_unit_id",
    "profile",
    "repeat_class",
    "left_sequence_eligible",
    "right_sequence_eligible",
    "pair_support",
    "raw_matches",
    "vmatchselect_matches",
    "selected_matches",
    "coordinate_filtered_matches",
    "direct_matches",
    "palindromic_matches",
    "raw_bytes",
    "selected_bytes",
    "raw_path",
    "selected_path",
    "parsed_path",
]

MATCH_FIELDS = [
    "protein_accession",
    "candidate_group",
    "experimental_anchor",
    "scope",
    "search_unit_id",
    "profile",
    "repeat_class",
    "source_context_id",
    "source_sequence_id",
    "selection_rank",
    "authoritative_context",
    "left_sequence_eligible",
    "right_sequence_eligible",
    "left_cluster_id",
    "left_cluster_size",
    "left_cluster_representative_sequence_id",
    "right_cluster_id",
    "right_cluster_size",
    "right_cluster_representative_sequence_id",
    "paired_cluster_id",
    "pair_support",
    "nucleotide_accession",
    "central_cds_start",
    "central_cds_stop",
    "central_cds_strand",
    "central_cds_annotated",
    "left_flank_length",
    "right_flank_length",
    "arm_1_side",
    "length_1",
    "start_1_0based",
    "end_1_0based",
    "start_1_1based",
    "end_1_1based",
    "arm_1_sequence",
    "match_type",
    "arm_2_side",
    "length_2",
    "start_2_0based",
    "end_2_0based",
    "start_2_1based",
    "end_2_1based",
    "arm_2_sequence",
    "arm_2_match_oriented_sequence",
    "spacer_length",
    "overlap_length",
    "distance",
    "evalue",
    "score",
    "identity",
]
for _arm in ("arm_1", "arm_2"):
    MATCH_FIELDS.extend(
        f"{_arm}_{suffix}" for suffix in BOUNDARY_SUFFIXES
    )
    MATCH_FIELDS.append(f"{_arm}_classification")
MATCH_FIELDS.append("repeat_span_classification")

def configuration_summary():
    units = load_search_units()
    scopes = Counter(unit["scope"] for unit in units)
    return {
        "search_units": len(units),
        "within_left": scopes["within_left"],
        "within_right": scopes["within_right"],
        "cross_flank": scopes["cross_flank"],
        "minimum_flank_length": cfg.MIN_FLANK_LENGTH,
        "minimum_repeat_length": cfg.VMATCH_MIN_LENGTH,
        "maximum_repeat_length": "none",
        "maximum_spacer": cfg.VMATCH_MAX_SPACER,
        "within_direct_vmatch_arguments": " ".join(within_direct_profile()["arguments"]),
        "within_direct_vmatchselect_arguments": " ".join(within_direct_profile()["selection"]),
        "within_inverted_vmatch_arguments": " ".join(within_inverted_profile()["arguments"]),
        "within_inverted_vmatchselect_arguments": " ".join(within_inverted_profile()["selection"]),
        "profiles": len(profile_summary()),
        "vmatchselect_minimum_length": cfg.VMATCHSELECT_MIN_LENGTH,
        "vmatchselect_minimum_identity": cfg.VMATCHSELECT_MIN_IDENTITY,
        "vmatchselect_maximum_evalue": cfg.VMATCHSELECT_MAX_EVALUE,
        "output_root": str(cfg.PROTEINS_DIR),
        "orientation": (
            "Both stored flanks begin next to the central CDS and run outward; "
            "Vmatch D/P labels refer to those stored strings."
        ),
    }

def search_unit_summary():
    return [
        _unit_output_row(unit, output_root=cfg.PROTEINS_DIR)
        for unit in load_search_units()
    ]

def profile_summary():
    rows = []
    for scope in ("within_left", "within_right", "cross_flank"):
        for profile in _profiles(scope):
            rows.append(
                {
                    "scope": scope,
                    "profile": profile["name"],
                    "repeat_class": profile["repeat_class"],
                    "vmatch_arguments": " ".join(profile["arguments"]),
                    "vmatchselect_arguments": " ".join(profile["selection"]),
                }
            )
    return rows

def calibrate_step3c():
    units = load_search_units()
    _write_unit_summary(units, output_root=cfg.PROTEINS_DIR)
    rows = []
    total_profiles = sum(len(_profiles(unit["scope"])) for unit in units)

    with tqdm(total=total_profiles, desc="Vmatch calibration") as progress:
        for unit in units:
            paths = _prepare_unit(
                unit,
                create=True,
                output_root=cfg.PROTEINS_DIR,
            )
            index_prefix = _ensure_index(paths["database"], paths["unit_dir"])
            for profile in _profiles(unit["scope"]):
                progress.set_postfix_str(
                    f"{unit['protein_accession']} {unit['search_unit_id']} "
                    f"{profile['name']}"
                )
                output_path = (
                    paths["unit_dir"]
                    / f"{profile['name']}.calibration.txt"
                )
                if not output_path.exists():
                    _run_to_file(
                        _vmatch_command(
                            unit,
                            profile,
                            paths,
                            index_prefix,
                            count_only=True,
                        ),
                        output_path,
                    )
                rows.extend(
                    _parse_calibration(output_path, unit, profile)
                )
                progress.update()

    _write_tsv(cfg.STEP3_CALIBRATION_PATH, CALIBRATION_FIELDS, rows)
    print(f"Vmatch calibration: {cfg.STEP3_CALIBRATION_PATH}")
    return rows

def run_step3c():
    return _process_profiles(run_missing=True)

def refilter_step3c():
    return _process_profiles(run_missing=False)

def discover_units(
    units: list[dict],
    output_root: str | Path,
    *,
    run_missing: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict], list[dict]]:

    _validate_selection_thresholds()
    units = list(units)
    output_root = Path(output_root)
    all_matches = []
    run_rows = []
    total_profiles = sum(len(_profiles(unit["scope"])) for unit in units)

    label = "Vmatch discovery" if run_missing else "Vmatch selection"
    with tqdm(
        total=total_profiles,
        desc=label,
        disable=not show_progress,
    ) as progress:
        for unit in units:
            paths = _prepare_unit(
                unit,
                create=run_missing,
                output_root=output_root,
            )
            index_prefix = (
                _ensure_index(paths["database"], paths["unit_dir"])
                if run_missing
                else None
            )
            for profile in _profiles(unit["scope"]):
                progress.set_postfix_str(
                    f"{unit['protein_accession']} {unit['search_unit_id']} "
                    f"{profile['name']}"
                )
                raw_path = (
                    paths["unit_dir"] / f"{profile['name']}.raw.txt"
                )
                if run_missing and not raw_path.exists():
                    _run_to_file(
                        _vmatch_command(
                            unit,
                            profile,
                            paths,
                            index_prefix,
                            count_only=False,
                        ),
                        raw_path,
                    )
                if not raw_path.exists():
                    raise FileNotFoundError(
                        f"Raw Vmatch output is missing: {raw_path}"
                    )

                selection_tag = profile["selection_tag"]
                selected_path = (
                    paths["unit_dir"]
                    / f"{profile['name']}.{selection_tag}.selected.txt"
                )
                _run_to_file(
                    [
                        str(cfg.VMATCHSELECT_BIN),
                        *profile["selection"],
                        str(raw_path),
                    ],
                    selected_path,
                )

                matches = _parse_matches(
                    selected_path,
                    unit,
                    profile,
                )
                parsed_path = (
                    paths["unit_dir"]
                    / f"{profile['name']}.{selection_tag}.matches.tsv"
                )
                _write_tsv(parsed_path, MATCH_FIELDS, matches)
                all_matches.extend(matches)
                run_rows.append(
                    _run_summary(
                        unit,
                        profile,
                        raw_path,
                        selected_path,
                        parsed_path,
                        matches,
                    )
                )
                progress.update()

    return all_matches, run_rows

def _process_profiles(run_missing):
    units = load_search_units()
    _write_unit_summary(units, output_root=cfg.PROTEINS_DIR)
    all_matches, run_rows = discover_units(
        units,
        cfg.PROTEINS_DIR,
        run_missing=run_missing,
    )
    _write_tsv(cfg.STEP3_RUNS_PATH, RUN_FIELDS, run_rows)
    _write_tsv(cfg.STEP3_MATCHES_PATH, MATCH_FIELDS, all_matches)
    print(f"Vmatch run summary: {cfg.STEP3_RUNS_PATH}")
    print(f"Selected matches: {cfg.STEP3_MATCHES_PATH}")
    return run_rows

def load_search_units() -> list[dict]:

    protein_rows = {
        row["protein_accession"]: row
        for row in _read_required_tsv(cfg.STEP2_PROTEINS_PATH)
    }
    context_rows = _read_required_tsv(cfg.STEP2_CONTEXTS_PATH)
    contexts = {}
    contexts_by_protein = defaultdict(list)
    for row in context_rows:
        key = (row["protein_accession"], row["context_id"])
        if key in contexts:
            raise ValueError(f"Duplicate Step 2 context: {key}")
        contexts[key] = row
        contexts_by_protein[row["protein_accession"]].append(row)

    memberships = {}
    selected = defaultdict(list)
    required_clustering_fields = {
        "clustering_method",
        "minimum_prefix_similarity",
        "ambiguity_policy",
    }
    for side, path in (
        ("left", cfg.STEP2_LEFT_CLUSTERS_PATH),
        ("right", cfg.STEP2_RIGHT_CLUSTERS_PATH),
    ):
        for row in _read_required_tsv(path):
            if not required_clustering_fields <= row.keys():
                raise ValueError(
                    f"{path} uses the retired Stage 2 cluster schema. "
                    "Rerun Step 2 clustering and finalization."
                )
            observed_clustering = (
                row["clustering_method"],
                float(row["minimum_prefix_similarity"]),
                row["ambiguity_policy"],
            )
            expected_clustering = (
                cfg.FLANK_CLUSTER_METHOD,
                float(cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY),
                cfg.FLANK_CLUSTER_AMBIGUITY_POLICY,
            )
            if observed_clustering != expected_clustering:
                raise ValueError(
                    f"{path} uses flank clustering settings "
                    f"{observed_clustering}; expected {expected_clustering}. "
                    "Rerun Step 2 clustering and finalization."
                )
            key = (side, row["protein_accession"], row["sequence_id"])
            if key in memberships:
                raise ValueError(f"Duplicate Step 2 cluster membership: {key}")
            memberships[key] = row
            if _as_bool(row["selected_representative"]):
                selected[(row["protein_accession"], side)].append(row)

    units = []
    for protein_accession in sorted(protein_rows):
        protein = protein_rows[protein_accession]
        for side in ("left", "right"):
            expected = _as_int(protein[f"{side}_finalized_flanks"])
            if expected == 0:
                continue
            representatives = selected[(protein_accession, side)]
            representatives.sort(key=lambda row: int(row["cluster_number"]))
            if len(representatives) != expected:
                raise ValueError(
                    f"{protein_accession} reports {expected} finalized {side} "
                    f"flanks but has {len(representatives)} representatives."
                )

            nr_path = (
                cfg.EXTRACTIONS_DIR
                / protein_accession
                / f"{protein_accession}.nr.{side}.fna"
            )
            records = _records_by_id(nr_path)
            representative_ids = {
                row["sequence_id"] for row in representatives
            }
            if set(records) != representative_ids:
                raise ValueError(
                    f"{nr_path} and the selected {side} representatives "
                    "disagree."
                )

            for representative in representatives:
                context = contexts.get(
                    (protein_accession, representative["context_id"])
                )
                if context is None:
                    raise ValueError(
                        "Missing Step 2 context for representative "
                        f"{representative['sequence_id']}."
                    )
                unit = _base_unit(protein, context)
                unit.update(
                    {
                        "scope": f"within_{side}",
                        "search_unit_id": representative["cluster_id"],
                        "left_cluster_id": (
                            representative["cluster_id"]
                            if side == "left"
                            else ""
                        ),
                        "left_cluster_size": (
                            representative["cluster_size"]
                            if side == "left"
                            else ""
                        ),
                        "left_cluster_representative_sequence_id": (
                            representative["representative_sequence_id"]
                            if side == "left"
                            else ""
                        ),
                        "right_cluster_id": (
                            representative["cluster_id"]
                            if side == "right"
                            else ""
                        ),
                        "right_cluster_size": (
                            representative["cluster_size"]
                            if side == "right"
                            else ""
                        ),
                        "right_cluster_representative_sequence_id": (
                            representative["representative_sequence_id"]
                            if side == "right"
                            else ""
                        ),
                        "paired_cluster_id": "",
                        "pair_support": "",
                        "database_side": side,
                        "query_side": "",
                        "database_sequence_id": representative["sequence_id"],
                        "database_sequence": records[
                            representative["sequence_id"]
                        ],
                        "query_sequence_id": "",
                        "query_sequence": "",
                    }
                )
                units.append(unit)

        if (
            _as_int(protein["left_finalized_flanks"]) > 0
            and _as_int(protein["right_finalized_flanks"]) > 0
        ):
            units.extend(
                _cross_flank_units(
                    protein,
                    contexts_by_protein[protein_accession],
                    memberships,
                )
            )

    units.sort(
        key=lambda unit: (
            unit["protein_accession"],
            unit["scope"],
            unit["search_unit_id"],
        )
    )
    return units

def _cross_flank_units(protein, context_rows, memberships):
    protein_accession = protein["protein_accession"]
    grouped = defaultdict(list)
    for context in context_rows:
        if (
            context["status"] != "valid"
            or _as_int(context["left_length"]) < cfg.MIN_FLANK_LENGTH
            or _as_int(context["right_length"]) < cfg.MIN_FLANK_LENGTH
        ):
            continue
        sequence_id = context["sequence_id"]
        left = memberships.get(("left", protein_accession, sequence_id))
        right = memberships.get(("right", protein_accession, sequence_id))
        if left is None or right is None:
            raise ValueError(
                f"Usable paired context {sequence_id} lacks a side "
                "cluster membership."
            )
        grouped[(left["cluster_id"], right["cluster_id"])].append(
            (context, left, right)
        )

    if not grouped:
        return []

    extraction_dir = cfg.EXTRACTIONS_DIR / protein_accession
    left_records = _records_by_id(
        extraction_dir / f"{protein_accession}.usable.left.fna"
    )
    right_records = _records_by_id(
        extraction_dir / f"{protein_accession}.usable.right.fna"
    )
    units = []
    for (left_cluster_id, right_cluster_id), members in sorted(grouped.items()):
        members.sort(
            key=lambda member: (
                -min(
                    _as_int(member[0]["left_length"]),
                    _as_int(member[0]["right_length"]),
                ),
                -max(
                    _as_int(member[0]["left_length"]),
                    _as_int(member[0]["right_length"]),
                ),
                0 if _as_bool(member[0]["authoritative_context"]) else 1,
                int(member[0]["selection_rank"]),
                member[0]["context_id"],
            )
        )
        context, left, right = members[0]
        sequence_id = context["sequence_id"]
        if sequence_id not in left_records or sequence_id not in right_records:
            raise ValueError(
                f"Usable paired FASTAs are missing {sequence_id}."
            )

        paired_cluster_id = (
            f"left_{_path_component(left_cluster_id)}"
            f"__right_{_path_component(right_cluster_id)}"
        )
        unit = _base_unit(protein, context)
        unit.update(
            {
                "scope": "cross_flank",
                "search_unit_id": paired_cluster_id,
                "left_cluster_id": left_cluster_id,
                "left_cluster_size": left["cluster_size"],
                "left_cluster_representative_sequence_id": left[
                    "representative_sequence_id"
                ],
                "right_cluster_id": right_cluster_id,
                "right_cluster_size": right["cluster_size"],
                "right_cluster_representative_sequence_id": right[
                    "representative_sequence_id"
                ],
                "paired_cluster_id": paired_cluster_id,
                "pair_support": len(members),
                "database_side": "left",
                "query_side": "right",
                "database_sequence_id": sequence_id,
                "database_sequence": left_records[sequence_id],
                "query_sequence_id": sequence_id,
                "query_sequence": right_records[sequence_id],
            }
        )
        units.append(unit)
    return units

def _base_unit(protein, context):
    return {
        "protein_accession": protein["protein_accession"],
        "candidate_group": protein["candidate_group"],
        "experimental_anchor": protein["experimental_anchor"],
        "source_context_id": context["context_id"],
        "source_sequence_id": context["sequence_id"],
        "selection_rank": context["selection_rank"],
        "authoritative_context": context["authoritative_context"],
        "left_sequence_eligible": protein["left_sequence_eligible"],
        "right_sequence_eligible": protein["right_sequence_eligible"],
        "context": context,
    }

def within_direct_profile(length=cfg.VMATCH_MIN_LENGTH, maximum_gap=cfg.VMATCH_MAX_SPACER + 1,
                          seed_length=cfg.VMATCH_WITHIN_SEED_LENGTH, hxdrop=cfg.VMATCH_WITHIN_HXDROP,
                          identity=cfg.VMATCH_WITHIN_MIN_IDENTITY, evalue=cfg.VMATCH_WITHIN_MAX_EVALUE):

    selection_evalue = cfg.VMATCHSELECT_WITHIN_MAX_EVALUE

    return {"name": f"direct_l{length}_s{seed_length}_hx{hxdrop}_id{identity}_ev{_number_token(evalue)}_gap1-{maximum_gap}",
            "repeat_class": "direct",
            "arguments": ["-d", "-l", str(length), "1", str(maximum_gap), "-seedlength", str(seed_length),
                          "-hxdrop", str(hxdrop), "-identity", str(identity), "-evalue", str(evalue)],
            "selection": ["-evalue", str(selection_evalue), "-sort", "ea"],
            "selection_tag": f"select_ev{_number_token(selection_evalue)}_sort-ea"}

def within_inverted_profile():

    length, seed_length, hxdrop = cfg.VMATCH_MIN_LENGTH, cfg.VMATCH_WITHIN_SEED_LENGTH, cfg.VMATCH_WITHIN_HXDROP
    identity, evalue = cfg.VMATCH_WITHIN_MIN_IDENTITY, cfg.VMATCH_WITHIN_MAX_EVALUE
    direct = within_direct_profile()

    return {"name": f"palindromic_l{length}_s{seed_length}_hx{hxdrop}_id{identity}_ev{_number_token(evalue)}",
            "repeat_class": "palindromic_inverted",
            "arguments": ["-p", "-l", str(length), "-seedlength", str(seed_length),
                          "-hxdrop", str(hxdrop), "-identity", str(identity), "-evalue", str(evalue)],
            "selection": direct["selection"],
            "selection_tag": direct["selection_tag"]}

def _edit_distance_profile(kind, repeat_class, strand):

    suffix = f"l{cfg.VMATCH_MIN_LENGTH}_e{cfg.VMATCH_MAX_EDIT_DISTANCE}_id{cfg.VMATCH_MIN_IDENTITY}_ev{_number_token(cfg.VMATCH_MAX_EVALUE)}"
    selection_tag = (f"select_l{cfg.VMATCHSELECT_MIN_LENGTH}_id{cfg.VMATCHSELECT_MIN_IDENTITY}"
                     f"_ev{_number_token(cfg.VMATCHSELECT_MAX_EVALUE)}")

    return {"name": f"{kind}_{suffix}",
            "repeat_class": repeat_class,
            "arguments": [strand, "-l", str(cfg.VMATCH_MIN_LENGTH), "-e", str(cfg.VMATCH_MAX_EDIT_DISTANCE),
                          "-identity", str(cfg.VMATCH_MIN_IDENTITY), "-evalue", str(cfg.VMATCH_MAX_EVALUE)],
            "selection": ["-l", str(cfg.VMATCHSELECT_MIN_LENGTH), "-identity", str(cfg.VMATCHSELECT_MIN_IDENTITY),
                          "-evalue", str(cfg.VMATCHSELECT_MAX_EVALUE)],
            "selection_tag": selection_tag}

def _profiles(scope):

    if scope in {"within_left", "within_right"}: return [within_direct_profile(), within_inverted_profile()]

    return [_edit_distance_profile("direct", "direct", "-d"),
            _edit_distance_profile("palindromic", "palindromic_inverted", "-p")]

def _validate_selection_thresholds():
    if cfg.VMATCHSELECT_WITHIN_MAX_EVALUE > cfg.VMATCH_WITHIN_MAX_EVALUE:
        raise ValueError(
            "VMATCHSELECT_WITHIN_MAX_EVALUE cannot exceed the within-flank "
            "Vmatch discovery maximum."
        )
    if cfg.VMATCHSELECT_MIN_LENGTH < cfg.VMATCH_MIN_LENGTH:
        raise ValueError(
            "VMATCHSELECT_MIN_LENGTH cannot be lower than the Vmatch "
            "discovery minimum."
        )
    if cfg.VMATCHSELECT_MIN_IDENTITY < cfg.VMATCH_MIN_IDENTITY:
        raise ValueError(
            "VMATCHSELECT_MIN_IDENTITY cannot be lower than the Vmatch "
            "discovery minimum."
        )
    if cfg.VMATCHSELECT_MAX_EVALUE > cfg.VMATCH_MAX_EVALUE:
        raise ValueError(
            "VMATCHSELECT_MAX_EVALUE cannot exceed the Vmatch discovery "
            "maximum."
        )

def _prepare_unit(unit, create, output_root=cfg.PROTEINS_DIR):
    unit_dir = _unit_directory(unit, output_root=output_root)
    state_path = unit_dir / "unit_state.json"
    database_path = unit_dir / f"{unit['database_side']}.fna"
    query_path = (
        unit_dir / f"{unit['query_side']}.fna"
        if unit["query_side"]
        else None
    )
    expected_state = _unit_state(unit)

    if not state_path.exists():
        if not create:
            raise FileNotFoundError(
                f"Search-unit state is missing: {state_path}"
            )
        unit_dir.mkdir(parents=True, exist_ok=True)
        if any(unit_dir.iterdir()):
            raise ValueError(
                f"Search-unit directory has no state file and is not empty: "
                f"{unit_dir}"
            )
        _write_single_fasta(
            database_path,
            unit["database_sequence_id"],
            unit["database_sequence"],
        )
        if query_path is not None:
            _write_single_fasta(
                query_path,
                unit["query_sequence_id"],
                unit["query_sequence"],
            )
        _write_json_atomic(state_path, expected_state)
    else:
        with state_path.open(encoding="utf-8") as handle:
            observed_state = json.load(handle)
        if observed_state != expected_state:
            raise ValueError(
                f"Search-unit inputs or discovery settings changed for "
                f"{unit_dir}; do not reuse its stale Vmatch artifacts."
            )

    _validate_single_fasta(
        database_path,
        unit["database_sequence_id"],
        unit["database_sequence"],
    )
    if query_path is not None:
        _validate_single_fasta(
            query_path,
            unit["query_sequence_id"],
            unit["query_sequence"],
        )
    return {
        "unit_dir": unit_dir,
        "state": state_path,
        "database": database_path,
        "query": query_path,
    }

def _unit_state(unit):
    return {
        "format_version": 3,
        "scope": unit["scope"],
        "search_unit_id": unit["search_unit_id"],
        "flank_policy": {
            "intended_length": int(cfg.FLANK_LENGTH),
            "minimum_length": int(cfg.MIN_FLANK_LENGTH),
            "clustering_method": cfg.FLANK_CLUSTER_METHOD,
            "minimum_prefix_similarity": float(
                cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY
            ),
            "ambiguity_policy": cfg.FLANK_CLUSTER_AMBIGUITY_POLICY,
        },
        "database": {
            "side": unit["database_side"],
            "sequence_id": unit["database_sequence_id"],
            "sha256": _sequence_hash(unit["database_sequence"]),
        },
        "query": (
            {
                "side": unit["query_side"],
                "sequence_id": unit["query_sequence_id"],
                "sha256": _sequence_hash(unit["query_sequence"]),
            }
            if unit["query_side"]
            else None
        ),
        "index": {
            "prefix_length": cfg.VMATCH_PREFIX_LENGTH,
            "tables": [
                "tis",
                "ois",
                "suf",
                "lcp",
                "bwt",
                "bck",
                "sti1",
            ],
        },
        "profiles": [
            {
                "name": profile["name"],
                "arguments": profile["arguments"],
            }
            for profile in _profiles(unit["scope"])
        ],
        "executables": {
            "mkvtree": str(cfg.MKVTREE_BIN),
            "vmatch": str(cfg.VMATCH_BIN),
        },
    }

def index_command(database_path, index_prefix):

    return [str(cfg.MKVTREE_BIN), "-db", str(database_path), "-dna", "-indexname", str(index_prefix),
            "-pl", str(cfg.VMATCH_PREFIX_LENGTH), "-tis", "-ois", "-suf", "-lcp", "-bwt", "-bck", "-sti1"]

def _ensure_index(database_path, unit_dir):
    index_dir = unit_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_prefix = index_dir / "flank"
    if Path(f"{index_prefix}.prj").exists():
        return index_prefix

    _run_quiet(index_command(database_path, index_prefix))
    return index_prefix

def _vmatch_command(unit, profile, paths, index_prefix, count_only):
    command = [str(cfg.VMATCH_BIN)]
    if count_only:
        command.append("-i")
    if paths["query"] is not None:
        command.extend(["-q", str(paths["query"])])
    command.extend(profile["arguments"])
    command.append(str(index_prefix))
    return command

def _run_quiet(command):
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

def _run_to_file(command, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output_path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    temporary.replace(output_path)

def _parse_calibration(path, unit, profile):
    total = None
    distribution = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 3 or fields[0] != "#":
                continue
            if fields[1] == "all":
                total = int(fields[2])
            elif fields[1].isdigit():
                distribution.append((int(fields[1]), int(fields[2])))
    if total is None:
        raise ValueError(f"Vmatch calibration total is missing: {path}")
    if not distribution:
        distribution = [("", 0)]

    return [
        {
            "protein_accession": unit["protein_accession"],
            "candidate_group": unit["candidate_group"],
            "experimental_anchor": unit["experimental_anchor"],
            "scope": unit["scope"],
            "search_unit_id": unit["search_unit_id"],
            "profile": profile["name"],
            "repeat_class": profile["repeat_class"],
            "match_length": length,
            "match_count": count,
            "total_matches": total,
            "calibration_path": path,
        }
        for length, count in distribution
    ]

def _parse_matches(path, unit, profile):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 11:
                raise ValueError(
                    f"Unexpected Vmatch row {path}:{line_number}: {line}"
                )

            length_1, sequence_1, start_1 = map(int, fields[:3])
            match_type = fields[3]
            length_2, sequence_2, start_2 = map(int, fields[4:7])
            if sequence_1 != 0 or sequence_2 != 0:
                raise ValueError(
                    f"Search unit {unit['search_unit_id']} was expected to "
                    "contain one database record and at most one query record."
                )

            side_1 = unit["database_side"]
            side_2 = unit["query_side"] or unit["database_side"]
            sequence_text_1 = unit["database_sequence"]
            sequence_text_2 = (
                unit["query_sequence"]
                if unit["query_side"]
                else unit["database_sequence"]
            )
            arm_1 = sequence_text_1[start_1:start_1 + length_1]
            arm_2 = sequence_text_2[start_2:start_2 + length_2]
            arm_2_oriented = (
                str(Seq(arm_2).reverse_complement())
                if match_type == "P"
                else arm_2
            )

            separation = (
                start_2 - (start_1 + length_1)
                if not unit["query_side"]
                else None
            )
            if (
                profile["repeat_class"] == "direct"
                and separation is not None
                and not 0 <= separation <= cfg.VMATCH_MAX_SPACER
            ):
                continue
            spacer = (
                max(0, separation) if separation is not None else ""
            )
            overlap = (
                max(0, -separation) if separation is not None else ""
            )
            context = unit["context"]
            start_1_1based = start_1 + 1
            end_1_1based = start_1 + length_1
            start_2_1based = start_2 + 1
            end_2_1based = start_2 + length_2
            arm_1_boundary = _arm_boundary_values(
                context,
                side_1,
                start_1_1based,
                end_1_1based,
            )
            arm_2_boundary = _arm_boundary_values(
                context,
                side_2,
                start_2_1based,
                end_2_1based,
            )

            row = _match_unit_values(unit)
            row.update(
                {
                    "profile": profile["name"],
                    "repeat_class": profile["repeat_class"],
                    "arm_1_side": side_1,
                    "length_1": length_1,
                    "start_1_0based": start_1,
                    "end_1_0based": start_1 + length_1 - 1,
                    "start_1_1based": start_1_1based,
                    "end_1_1based": end_1_1based,
                    "arm_1_sequence": arm_1,
                    "match_type": match_type,
                    "arm_2_side": side_2,
                    "length_2": length_2,
                    "start_2_0based": start_2,
                    "end_2_0based": start_2 + length_2 - 1,
                    "start_2_1based": start_2_1based,
                    "end_2_1based": end_2_1based,
                    "arm_2_sequence": arm_2,
                    "arm_2_match_oriented_sequence": arm_2_oriented,
                    "spacer_length": spacer,
                    "overlap_length": overlap,
                    "distance": int(fields[7]),
                    "evalue": fields[8],
                    "score": int(fields[9]),
                    "identity": float(fields[10]),
                }
            )
            row.update(
                {
                    f"arm_1_{key}": value
                    for key, value in arm_1_boundary.items()
                }
            )
            row.update(
                {
                    f"arm_2_{key}": value
                    for key, value in arm_2_boundary.items()
                }
            )
            if unit["scope"] == "cross_flank":
                row["repeat_span_classification"] = (
                    _cross_flank_classification(
                        arm_1_boundary["classification"],
                        arm_2_boundary["classification"],
                    )
                )
            else:
                row["repeat_span_classification"] = _classify_interval(
                    context,
                    side_1,
                    min(start_1_1based, start_2_1based),
                    max(end_1_1based, end_2_1based),
                )
            rows.append(row)
    return rows

def _match_unit_values(unit):
    context = unit["context"]
    return {
        "protein_accession": unit["protein_accession"],
        "candidate_group": unit["candidate_group"],
        "experimental_anchor": unit["experimental_anchor"],
        "scope": unit["scope"],
        "search_unit_id": unit["search_unit_id"],
        "source_context_id": unit["source_context_id"],
        "source_sequence_id": unit["source_sequence_id"],
        "selection_rank": unit["selection_rank"],
        "authoritative_context": unit["authoritative_context"],
        "left_sequence_eligible": unit["left_sequence_eligible"],
        "right_sequence_eligible": unit["right_sequence_eligible"],
        "left_cluster_id": unit["left_cluster_id"],
        "left_cluster_size": unit["left_cluster_size"],
        "left_cluster_representative_sequence_id": unit[
            "left_cluster_representative_sequence_id"
        ],
        "right_cluster_id": unit["right_cluster_id"],
        "right_cluster_size": unit["right_cluster_size"],
        "right_cluster_representative_sequence_id": unit[
            "right_cluster_representative_sequence_id"
        ],
        "paired_cluster_id": unit["paired_cluster_id"],
        "pair_support": unit["pair_support"],
        "nucleotide_accession": context["nucleotide_accession"],
        "central_cds_start": context["start"],
        "central_cds_stop": context["stop"],
        "central_cds_strand": context["strand"],
        "central_cds_annotated": context["central_cds_annotated"],
        "left_flank_length": context["left_length"],
        "right_flank_length": context["right_length"],
    }

def _arm_boundary_values(context, side, start, end):
    values = {
        suffix: context[f"{side}_{suffix}"]
        for suffix in BOUNDARY_SUFFIXES
    }
    values["classification"] = _classify_interval(
        context,
        side,
        start,
        end,
    )
    return values

def _classify_interval(context, side, start, end):
    status = context[f"{side}_boundary_status"]
    if not status:
        return "boundary_unknown"

    neighbor_interval = _neighbor_outward_interval(context, side)
    if status in {
        "annotated",
        "no_cds_within_flank",
        "uncertain_boundary",
    }:
        raw_intergenic_length = context[f"{side}_intergenic_length"]
        if raw_intergenic_length == "":
            if status != "no_cds_within_flank":
                return "boundary_unknown"
            intergenic_length = _as_int(context[f"{side}_length"])
        else:
            intergenic_length = _as_int(raw_intergenic_length)
        if end <= intergenic_length:
            return "intergenic"
        if start <= intergenic_length < end:
            return "boundary_spanning"
        if (
            neighbor_interval is not None
            and start >= neighbor_interval[0]
            and end <= neighbor_interval[1]
        ):
            return "neighbor_cds"
        return "boundary_unknown"

    if status == "overlapping":
        if (
            neighbor_interval is not None
            and start >= neighbor_interval[0]
            and end <= neighbor_interval[1]
        ):
            return "neighbor_cds"
        return "boundary_unknown"

    return "boundary_unknown"

def _neighbor_outward_interval(context, side):
    fields = (
        context[f"{side}_neighbor_cds_start"],
        context[f"{side}_neighbor_cds_stop"],
        context["start"],
        context["stop"],
    )
    if any(value == "" for value in fields):
        return None
    neighbor_start, neighbor_stop, central_start, central_stop = map(
        int, fields
    )
    strand = context["strand"]

    if (strand == "+" and side == "left") or (
        strand == "-" and side == "right"
    ):
        first = central_start - neighbor_stop
        last = central_start - neighbor_start
    else:
        first = neighbor_start - central_stop
        last = neighbor_stop - central_stop

    interval_start = max(1, min(first, last))
    interval_end = min(
        _as_int(context[f"{side}_length"]),
        max(first, last),
    )
    if interval_start > interval_end:
        return None
    return interval_start, interval_end

def _cross_flank_classification(first, second):
    if "boundary_unknown" in {first, second}:
        return "boundary_unknown"
    if "boundary_spanning" in {first, second}:
        return "boundary_spanning"
    if first == second:
        return first
    return "boundary_spanning"

def _run_summary(
    unit,
    profile,
    raw_path,
    selected_path,
    parsed_path,
    matches,
):
    types = Counter(row["match_type"] for row in matches)
    vmatchselect_matches = _match_count(selected_path)
    return {
        "protein_accession": unit["protein_accession"],
        "candidate_group": unit["candidate_group"],
        "experimental_anchor": unit["experimental_anchor"],
        "scope": unit["scope"],
        "search_unit_id": unit["search_unit_id"],
        "profile": profile["name"],
        "repeat_class": profile["repeat_class"],
        "left_sequence_eligible": unit["left_sequence_eligible"],
        "right_sequence_eligible": unit["right_sequence_eligible"],
        "pair_support": unit["pair_support"],
        "raw_matches": _match_count(raw_path),
        "vmatchselect_matches": vmatchselect_matches,
        "selected_matches": len(matches),
        "coordinate_filtered_matches": (
            vmatchselect_matches - len(matches)
        ),
        "direct_matches": types["D"],
        "palindromic_matches": types["P"],
        "raw_bytes": raw_path.stat().st_size,
        "selected_bytes": selected_path.stat().st_size,
        "raw_path": raw_path,
        "selected_path": selected_path,
        "parsed_path": parsed_path,
    }

def _unit_output_row(unit, output_root=cfg.PROTEINS_DIR):
    context = unit["context"]
    unit_dir = _unit_directory(unit, output_root=output_root)
    database_path = unit_dir / f"{unit['database_side']}.fna"
    query_path = (
        unit_dir / f"{unit['query_side']}.fna"
        if unit["query_side"]
        else ""
    )
    row = {
        "protein_accession": unit["protein_accession"],
        "candidate_group": unit["candidate_group"],
        "experimental_anchor": unit["experimental_anchor"],
        "scope": unit["scope"],
        "search_unit_id": unit["search_unit_id"],
        "source_context_id": unit["source_context_id"],
        "source_sequence_id": unit["source_sequence_id"],
        "selection_rank": unit["selection_rank"],
        "authoritative_context": unit["authoritative_context"],
        "left_sequence_eligible": unit["left_sequence_eligible"],
        "right_sequence_eligible": unit["right_sequence_eligible"],
        "left_cluster_id": unit["left_cluster_id"],
        "left_cluster_size": unit["left_cluster_size"],
        "left_cluster_representative_sequence_id": unit[
            "left_cluster_representative_sequence_id"
        ],
        "right_cluster_id": unit["right_cluster_id"],
        "right_cluster_size": unit["right_cluster_size"],
        "right_cluster_representative_sequence_id": unit[
            "right_cluster_representative_sequence_id"
        ],
        "paired_cluster_id": unit["paired_cluster_id"],
        "pair_support": unit["pair_support"],
        "database_side": unit["database_side"],
        "query_side": unit["query_side"],
        "nucleotide_accession": context["nucleotide_accession"],
        "central_cds_start": context["start"],
        "central_cds_stop": context["stop"],
        "central_cds_strand": context["strand"],
        "central_cds_annotated": context["central_cds_annotated"],
        "left_flank_length": context["left_length"],
        "right_flank_length": context["right_length"],
        "database_fasta_path": database_path,
        "query_fasta_path": query_path,
        "unit_state_path": unit_dir / "unit_state.json",
        "unit_directory": unit_dir,
    }
    for side in ("left", "right"):
        for suffix in BOUNDARY_SUFFIXES:
            row[f"{side}_{suffix}"] = context[f"{side}_{suffix}"]
    return row

def _write_unit_summary(units, output_root=cfg.PROTEINS_DIR):
    _write_tsv(
        cfg.STEP3_CONTEXTS_PATH,
        UNIT_FIELDS,
        [
            _unit_output_row(unit, output_root=output_root)
            for unit in units
        ],
    )

def _unit_directory(unit, output_root=cfg.PROTEINS_DIR):
    return (
        Path(output_root)
        / unit["protein_accession"]
        / "vmatch"
        / unit["scope"]
        / unit["search_unit_id"]
    )

def _records_by_id(path):
    if not path.exists():
        raise FileNotFoundError(f"Required FASTA is missing: {path}")
    records = {}
    for record in SeqIO.parse(path, "fasta"):
        if record.id in records:
            raise ValueError(f"Duplicate FASTA record ID in {path}: {record.id}")
        records[record.id] = str(record.seq).upper()
    return records

def _read_required_tsv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required Step 2 summary is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def _write_single_fasta(path, sequence_id, sequence):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        SeqIO.write(
            SeqRecord(Seq(sequence), id=sequence_id, description=""),
            handle,
            "fasta",
        )
    temporary.replace(path)

def _validate_single_fasta(path, sequence_id, sequence):
    records = list(SeqIO.parse(path, "fasta"))
    if (
        len(records) != 1
        or records[0].id != sequence_id
        or str(records[0].seq).upper() != sequence
    ):
        raise ValueError(
            f"Search-unit FASTA no longer matches its recorded input: {path}"
        )

def _write_json_atomic(path, value):
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)

def _write_tsv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def _match_count(path):
    with path.open(encoding="utf-8") as handle:
        return sum(
            bool(line.strip()) and not line.startswith("#")
            for line in handle
        )

def _sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def _path_component(value):
    if not value or value in {".", ".."} or "/" in value:
        raise ValueError(f"Unsafe cluster ID for output path: {value!r}")
    return value

def _number_token(value):
    return str(value).replace(".", "p")

def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}

def _as_int(value):
    return int(value) if str(value).strip() else 0

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run representative-level Stage 05C Vmatch discovery on "
            "independent flanks and observed flank pairs."
        )
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("calibrate", "run", "select"),
    )
    return parser.parse_args()

def main():
    action = _parse_args().action
    if action == "calibrate":
        calibrate_step3c()
    elif action == "select":
        refilter_step3c()
    else:
        run_step3c()

if __name__ == "__main__":
    main()
