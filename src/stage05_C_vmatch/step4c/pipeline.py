from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from Bio import SeqIO

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.architectures.models import (
    ARCHITECTURE_FIELDS,
    read_architectures,
    read_audit_records,
    read_repeats,
    stable_id,
)
from src.stage05_C_vmatch.step4c.architectures.spaced import build_spaced_architectures
from src.stage05_C_vmatch.step4c.architectures.tandem import (
    build_trf_architecture,
    partition_vmatch_repeats,
    trf_suppression_evidence,
)
from src.stage05_C_vmatch.step3c.trf_evidence import (
    SPAN_SCORE_SMALLEST_PERIOD_POLICY,
    read_trf_arrays,
)
from src.stage05_C_vmatch.step4c.evidence import (
    COMPOSITION_FIELDS,
    EVIDENCE_FIELDS,
    PAIR_REPEAT_CLASSES,
    WITHIN_FLANK_SCOPES,
    build_composition_rows,
    build_evidence_items,
    read_evidence,
    read_unit_composition,
)

def configured_build_paths() -> dict[str, Path]:

    return {
        "matches_path": cfg.STEP3_MATCHES_PATH,
        "trf_arrays_path": cfg.STEP3_TRF_ARRAYS_PATH,
        "trf_normalization_fingerprint_path": (
            cfg.STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH
        ),
        "contexts_path": cfg.STEP3_CONTEXTS_PATH,
        "architectures_path": cfg.STEP4_ARCHITECTURES_PATH,
        "audit_path": cfg.STEP4_ARCHITECTURE_AUDIT_PATH,
        "suppression_ledger_path": cfg.STEP4_VMATCH_SUPPRESSION_LEDGER_PATH,
        "evidence_path": cfg.STEP4_EVIDENCE_PATH,
        "unit_composition_path": cfg.STEP4_UNIT_COMPOSITION_PATH,
    }

def configuration_summary() -> dict[str, object]:

    return {
        **configured_build_paths(),
        "trf_normalization_policy": cfg.TRF_NORMALIZATION_POLICY,
        "trf_normalization_tau_span": cfg.TRF_NORMALIZATION_TAU_SPAN,
        "trf_normalization_tau_score": cfg.TRF_NORMALIZATION_TAU_SCORE,
        "trf_normalization_minimum_reciprocal_overlap": (
            cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP
        ),
        "trf_vmatch_suppression_policy": cfg.TRF_VMATCH_SUPPRESSION_POLICY,
        "spaced_direct_policy": cfg.V5_POLICY,
        "spaced_direct_settings": dict(cfg.V5_SETTINGS),
        "trf_vmatch_minimum_arm_coverage": cfg.TRF_VMATCH_MIN_ARM_COVERAGE,
        "trf_vmatch_minimum_overlap_bp": cfg.TRF_VMATCH_MIN_OVERLAP_BP,
        "markov_pseudocount": cfg.STEP4_MARKOV_PSEUDOCOUNT,
    }

def _logger() -> logging.Logger:
    logger = logging.getLogger("stage05c.step4c")
    if logger.handlers:
        return logger
    cfg.STEP4_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    concise = logging.StreamHandler(sys.stdout)
    concise.setLevel(logging.INFO)
    detailed = logging.FileHandler(cfg.STEP4_LOG_PATH, encoding="utf-8")
    detailed.setLevel(logging.DEBUG)
    concise.setFormatter(logging.Formatter("%(message)s"))
    detailed.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(concise)
    logger.addHandler(detailed)
    return logger

def _write_tsv(
    path: str | Path,
    fields: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def _write_jsonl(
    path: str | Path, records: Iterable[Mapping[str, object]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)

def _read_single_fasta(
    path: Path,
    cache: dict[Path, tuple[str, str]],
) -> tuple[str, str]:
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(f"Required Step 3C FASTA is missing: {path}")
        records = list(SeqIO.parse(path, "fasta"))
        if len(records) != 1:
            raise ValueError(
                f"Expected exactly one representative sequence in {path}, "
                f"found {len(records)}"
            )
        cache[path] = records[0].id, str(records[0].seq).upper()
    return cache[path]

def _read_unit_state(
    path: Path,
    cache: dict[Path, dict[str, object]],
) -> dict[str, object]:
    if path not in cache:
        if not path.exists():
            raise FileNotFoundError(f"Required Step 3C unit state is missing: {path}")
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict) or state.get("format_version") != 3:
            raise ValueError(f"Unsupported Step 3C unit state: {path}")
        cache[path] = state
    return cache[path]

def _read_representative_sequences(
    contexts_path: str | Path,
    required_keys: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], str]:

    contexts_path = Path(contexts_path)
    if not contexts_path.exists():
        raise FileNotFoundError(contexts_path)
    required_fields = {
        "protein_accession",
        "scope",
        "search_unit_id",
        "database_side",
        "query_side",
        "database_fasta_path",
        "query_fasta_path",
        "left_flank_length",
        "right_flank_length",
        "unit_state_path",
    }
    sequences: dict[tuple[str, str, str], str] = {}
    fasta_cache: dict[Path, tuple[str, str]] = {}
    state_cache: dict[Path, dict[str, object]] = {}
    with contexts_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"{contexts_path.name} lacks required fields: "
                f"{', '.join(sorted(missing_fields))}"
            )
        for row in reader:
            protein_accession = row["protein_accession"]
            search_unit_id = row["search_unit_id"]
            required_prefixes = []
            for prefix in ("database", "query"):
                side = row[f"{prefix}_side"]
                key = (protein_accession, search_unit_id, side)
                if side and key in required_keys:
                    required_prefixes.append((prefix, side, key))
            if not required_prefixes:
                continue
            if not row["unit_state_path"]:
                raise ValueError(
                    f"{protein_accession}:{search_unit_id} lacks its Step 3C unit state path"
                )
            state_path = Path(row["unit_state_path"])
            state = _read_unit_state(state_path, state_cache)
            if (
                state.get("scope") != row["scope"]
                or state.get("search_unit_id") != search_unit_id
            ):
                raise ValueError(
                    f"Step 3C unit state disagrees with {protein_accession}:"
                    f"{search_unit_id}"
                )
            for prefix, side, key in required_prefixes:
                fasta_path = row[f"{prefix}_fasta_path"]
                if not fasta_path:
                    raise ValueError(
                        f"{':'.join(key)} lacks its Step 3C FASTA path"
                    )
                state_sequence = state.get(prefix)
                if not isinstance(state_sequence, dict):
                    raise ValueError(
                        f"{':'.join(key)} lacks {prefix} sequence state"
                    )
                sequence_id, sequence = _read_single_fasta(
                    Path(fasta_path), fasta_cache
                )
                sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if (
                    state_sequence.get("side") != side
                    or state_sequence.get("sequence_id") != sequence_id
                    or state_sequence.get("sha256") != sequence_hash
                ):
                    raise ValueError(
                        f"{':'.join(key)} FASTA disagrees with its Step 3C unit state"
                    )
                expected_length = int(row[f"{side}_flank_length"])
                if len(sequence) != expected_length:
                    raise ValueError(
                        f"{':'.join(key)} FASTA length {len(sequence)} "
                        f"does not match the recorded length {expected_length}"
                    )
                previous = sequences.setdefault(key, sequence)
                if previous != sequence:
                    raise ValueError(
                        f"Conflicting representative sequences for {':'.join(key)}"
                    )
    missing_keys = required_keys - set(sequences)
    if missing_keys:
        missing = ", ".join(":".join(key) for key in sorted(missing_keys))
        raise ValueError(f"Step 3C contexts lack representative sequences: {missing}")
    return sequences

def _read_unit_rows(
    path: str | Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in reader:
            key = (
                row["protein_accession"],
                row["scope"],
                row["search_unit_id"],
            )
            if key in rows:
                raise ValueError(f"Duplicate Step 3C search unit: {key}")
            rows[key] = row
    return rows

def _step3_snapshot_id(repeats, arrays, representative_sequences) -> str:
    return stable_id(
        "step3",
        *sorted(repeat.source_match_id for repeat in repeats),
        *sorted(
            json.dumps(array.to_row(), sort_keys=True, separators=(",", ":"))
            for array in arrays
        ),
        *(
            stable_id("sequence", *key, sequence, length=20)
            for key, sequence in sorted(representative_sequences.items())
        ),
        length=20,
    )

def _canonical_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _settings_fingerprint(name: str, settings: Mapping[str, object]) -> str:
    return _canonical_fingerprint(
        {"name": name, "settings": dict(settings)}
    )

def _file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_trf_normalization_fingerprint(
    fingerprint_path: str | Path,
    arrays_path: str | Path,
    array_count: int,
) -> dict[str, object]:

    fingerprint_path = Path(fingerprint_path)
    if not fingerprint_path.exists():
        raise FileNotFoundError(
            f"TRF normalization fingerprint is missing: {fingerprint_path}; "
            "run step3c.trf normalize before building Step 4C"
        )
    with fingerprint_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    settings = {
        "minimum_period": cfg.TRF_MIN_PERIOD,
        "minimum_full_copies": cfg.TRF_MIN_FULL_COPIES,
        "tau_span": cfg.TRF_NORMALIZATION_TAU_SPAN,
        "tau_score": cfg.TRF_NORMALIZATION_TAU_SCORE,
        "minimum_reciprocal_overlap": (
            cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP
        ),
    }
    expected_policy = {
        "name": cfg.TRF_NORMALIZATION_POLICY,
        "settings": settings,
        "settings_fingerprint": _settings_fingerprint(
            cfg.TRF_NORMALIZATION_POLICY, settings
        ),
    }
    if not isinstance(record, dict) or (
        record.get("format_version") != 1
        or record.get("kernel") != "K1_trf_candidate_resolution"
        or record.get("policy") != expected_policy
    ):
        raise ValueError(
            "TRF normalization fingerprint does not match the accepted K1 policy"
        )
    if record.get("accepted_output_fingerprint") != _file_fingerprint(
        arrays_path
    ):
        raise ValueError(
            "Accepted TRF arrays differ from their normalization fingerprint"
        )
    if record.get("accepted_array_count") != array_count:
        raise ValueError(
            "Accepted TRF array count differs from its normalization fingerprint"
        )
    for field in (
        "input_fingerprint",
        "raw_candidate_fingerprint",
        "normalization_input_fingerprint",
    ):
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"TRF normalization fingerprint lacks {field}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(
                f"TRF normalization fingerprint has invalid {field}"
            ) from exc
    expected_input_fingerprint = _canonical_fingerprint(
        {
            "input_fingerprint": record["input_fingerprint"],
            "raw_candidate_fingerprint": record["raw_candidate_fingerprint"],
            "policy": expected_policy,
        }
    )
    if record["normalization_input_fingerprint"] != expected_input_fingerprint:
        raise ValueError(
            "TRF normalization input fingerprint is internally inconsistent"
        )
    return record

def accepted_build_strategies() -> dict[str, object]:

    accepted_normalization = SPAN_SCORE_SMALLEST_PERIOD_POLICY
    if cfg.TRF_NORMALIZATION_POLICY != accepted_normalization:
        raise ValueError(
            "Production TRF normalization policy is not implemented: "
            f"{cfg.TRF_NORMALIZATION_POLICY!r}"
        )
    accepted_suppression = "legacy_array_span_both_arms_v1"
    if cfg.TRF_VMATCH_SUPPRESSION_POLICY != accepted_suppression:
        raise ValueError(
            "Production TRF/Vmatch suppression policy is not implemented: "
            f"{cfg.TRF_VMATCH_SUPPRESSION_POLICY!r}"
        )
    accepted_spaced = "architecture_inference_deterministic_v5"
    if cfg.V5_POLICY != accepted_spaced:
        raise ValueError(
            f"Production spaced-direct policy is not implemented: {cfg.V5_POLICY!r}"
        )
    return {
        "vmatch_suppression": {
            "name": accepted_suppression,
            "partition": partition_vmatch_repeats,
            "explain": trf_suppression_evidence,
            "settings": {
                "minimum_arm_coverage": cfg.TRF_VMATCH_MIN_ARM_COVERAGE,
                "minimum_overlap_bp": cfg.TRF_VMATCH_MIN_OVERLAP_BP,
            },
        },
        "spaced_direct": {"name": accepted_spaced, "settings": dict(cfg.V5_SETTINGS)},
    }

def load_inputs(
    paths: Mapping[str, str | Path],
    *,
    verify_normalization_fingerprint: bool = True,
) -> dict[str, object]:

    repeats = tuple(read_repeats(paths["matches_path"]))
    arrays = tuple(read_trf_arrays(paths["trf_arrays_path"]))
    if verify_normalization_fingerprint:
        _verify_trf_normalization_fingerprint(
            paths["trf_normalization_fingerprint_path"],
            paths["trf_arrays_path"],
            len(arrays),
        )
    units = _read_unit_rows(paths["contexts_path"])
    required_sequence_keys = {
        (repeat.protein_accession, repeat.search_unit_id, arm.side)
        for repeat in repeats
        for arm in (repeat.arm_1, repeat.arm_2)
    }
    required_sequence_keys.update(
        (array.protein_accession, array.search_unit_id, array.side)
        for array in arrays
    )
    required_sequence_keys.update(
        (unit["protein_accession"], unit["search_unit_id"], unit["database_side"])
        for unit in units.values()
        if unit["scope"] in WITHIN_FLANK_SCOPES
    )
    representative_sequences = _read_representative_sequences(
        paths["contexts_path"],
        required_sequence_keys,
    )
    return {
        "repeats": repeats,
        "arrays": arrays,
        "representative_sequences": representative_sequences,
        "units": units,
    }

def _kernel_policy_provenance(
    strategies: Mapping[str, object],
) -> dict[str, object]:
    suppression = strategies["vmatch_suppression"]
    if not isinstance(suppression, Mapping):
        raise TypeError("vmatch_suppression strategy must be a mapping")
    suppression_name = str(suppression["name"])
    suppression_settings = dict(suppression["settings"])
    spaced = strategies["spaced_direct"]
    spaced_name = str(spaced["name"])
    spaced_settings = dict(spaced["settings"])
    return {
        "K2_trf_vmatch_suppression": {
            "name": suppression_name,
            "settings": suppression_settings,
            "settings_fingerprint": _settings_fingerprint(
                suppression_name, suppression_settings
            ),
        },
        "V5_spaced_direct_inference": {
            "name": spaced_name,
            "settings": spaced_settings,
            "settings_fingerprint": _settings_fingerprint(
                spaced_name, spaced_settings
            ),
        },
    }

def _suppression_ledger_records(
    suppressed,
    arrays,
    suppression_strategy: Mapping[str, object],
    policy_provenance: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    settings = dict(suppression_strategy["settings"])
    explain = suppression_strategy["explain"]
    if not callable(explain):
        raise TypeError("vmatch_suppression explain must be callable")
    records: list[dict[str, object]] = []
    for repeat in sorted(suppressed, key=lambda item: item.source_match_id):
        evidence = tuple(
            explain(
                repeat,
                arrays,
                minimum_arm_coverage=float(settings["minimum_arm_coverage"]),
                minimum_overlap_bp=int(settings["minimum_overlap_bp"]),
            )
        )
        if not evidence:
            raise AssertionError(
                f"Suppressed sighting lacks explaining TRF evidence: "
                f"{repeat.source_match_id}"
            )
        records.append(
            {
                "source_sighting": {
                    "source_match_id": repeat.source_match_id,
                    "protein_accession": repeat.protein_accession,
                    "scope": repeat.scope,
                    "search_unit_id": repeat.search_unit_id,
                    "repeat_class": repeat.repeat_class,
                    "arm_1": {
                        "side": repeat.arm_1.side,
                        "start_0based": repeat.arm_1.start,
                        "end_0based": repeat.arm_1.end,
                    },
                    "arm_2": {
                        "side": repeat.arm_2.side,
                        "start_0based": repeat.arm_2.start,
                        "end_0based": repeat.arm_2.end,
                    },
                },
                "explaining_arrays": list(evidence),
                "policy": policy_provenance["K2_trf_vmatch_suppression"],
            }
        )
    return tuple(records)

def compute(
    inputs: Mapping[str, object],
    strategies: Mapping[str, object],
) -> dict[str, object]:

    repeats = tuple(inputs["repeats"])
    arrays = tuple(inputs["arrays"])
    representative_sequences = inputs["representative_sequences"]
    units = inputs["units"]
    suppression = strategies["vmatch_suppression"]
    if not isinstance(suppression, Mapping):
        raise TypeError("vmatch_suppression strategy must be a mapping")
    partition = suppression["partition"]
    if not callable(partition):
        raise TypeError("vmatch_suppression partition must be callable")
    settings = dict(suppression["settings"])
    retained, suppressed = partition(
        repeats,
        arrays,
        minimum_arm_coverage=float(settings["minimum_arm_coverage"]),
        minimum_overlap_bp=int(settings["minimum_overlap_bp"]),
    )
    retained = tuple(retained)
    suppressed = tuple(suppressed)
    input_ids = [repeat.source_match_id for repeat in repeats]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Selected Step 3 rows contain duplicate content hashes")
    output_repeats = (*retained, *suppressed)
    if Counter(item.source_match_id for item in output_repeats) != Counter(
        input_ids
    ):
        raise ValueError(
            "Vmatch suppression strategy must exactly partition input sightings"
        )
    input_by_id = {repeat.source_match_id: repeat for repeat in repeats}
    if any(
        input_by_id[item.source_match_id] != item for item in output_repeats
    ):
        raise ValueError(
            "Vmatch suppression strategy cannot replace input sightings"
        )
    snapshot_id = _step3_snapshot_id(repeats, arrays, representative_sequences)
    policy_provenance = _kernel_policy_provenance(strategies)
    vmatch_architectures, audit_records = build_spaced_architectures(
        retained,
        representative_sequences,
        units,
        snapshot_id,
        strategies["spaced_direct"]["settings"],
    )
    for record in audit_records:
        record["kernel_policies"] = policy_provenance
    trf_architectures = [
        build_trf_architecture(
            array,
            representative_sequences[
                (array.protein_accession, array.search_unit_id, array.side)
            ],
            units[(array.protein_accession, array.scope, array.search_unit_id)],
            snapshot_id,
        )
        for array in arrays
    ]
    architectures = sorted(
        [*vmatch_architectures, *trf_architectures],
        key=lambda item: (
            item.protein_accession,
            item.scope,
            item.search_unit_id,
            item.step4c_architecture_id,
        ),
    )
    composition_rows = build_composition_rows(units, representative_sequences)
    evidence_items = build_evidence_items(
        retained,
        arrays,
        units,
        composition_rows,
    )
    return {
        "architectures": architectures,
        "vmatch_architectures": vmatch_architectures,
        "trf_architectures": trf_architectures,
        "suppressed_repeats": suppressed,
        "audit_records": tuple(audit_records),
        "suppression_ledger_records": _suppression_ledger_records(
            suppressed,
            arrays,
            suppression,
            policy_provenance,
        ),
        "composition_rows": composition_rows,
        "evidence_items": evidence_items,
    }

def persist(
    outputs: Mapping[str, object],
    paths: Mapping[str, str | Path],
) -> None:

    output_paths = [
        Path(paths[name]).resolve()
        for name in (
            "audit_path",
            "architectures_path",
            "suppression_ledger_path",
            "evidence_path",
            "unit_composition_path",
        )
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Step 4C output paths must be distinct")

    _write_jsonl(
        paths["audit_path"],
        outputs["audit_records"],
    )
    _write_tsv(
        paths["architectures_path"],
        ARCHITECTURE_FIELDS,
        (
            architecture.to_row()
            for architecture in outputs["architectures"]
        ),
    )
    _write_jsonl(
        paths["suppression_ledger_path"],
        outputs["suppression_ledger_records"],
    )
    _write_tsv(
        paths["evidence_path"],
        EVIDENCE_FIELDS,
        (item.to_row() for item in outputs["evidence_items"]),
    )
    _write_tsv(
        paths["unit_composition_path"],
        COMPOSITION_FIELDS,
        outputs["composition_rows"],
    )

def run_build():
    logger = _logger()
    logger.info("Step 4C: reading typed repeat sightings")
    logger.debug("Configuration: %s", configuration_summary())
    paths = configured_build_paths()
    inputs = load_inputs(paths)
    outputs = compute(inputs, accepted_build_strategies())
    persist(outputs, paths)
    architectures = outputs["architectures"]
    vmatch_architectures = outputs["vmatch_architectures"]
    trf_architectures = outputs["trf_architectures"]
    suppressed = outputs["suppressed_repeats"]
    evidence_by_class = Counter(item.repeat_class for item in outputs["evidence_items"])
    logger.info(
        "Step 4C complete: %s canonical architectures (%s V5, %s TRF), "
        "%s V5 unit audit records; %s Vmatch sightings suppressed by TRF; "
        "evidence %s direct, %s inverted, %s TRF",
        f"{len(architectures):,}",
        f"{len(vmatch_architectures):,}",
        f"{len(trf_architectures):,}",
        f"{len(outputs['audit_records']):,}",
        f"{len(suppressed):,}",
        f"{evidence_by_class.get('direct', 0):,}",
        f"{evidence_by_class.get('palindromic_inverted', 0):,}",
        f"{evidence_by_class.get('tandem_array', 0):,}",
    )
    logger.info("Canonical catalog: %s", paths["architectures_path"])
    logger.info("Construction audit: %s", paths["audit_path"])
    logger.info("Suppression ledger: %s", paths["suppression_ledger_path"])
    logger.info("Evidence table: %s", paths["evidence_path"])
    logger.info("Unit composition: %s", paths["unit_composition_path"])
    return architectures

def run_summarize(paths: Mapping[str, Path] | None = None) -> None:

    logger = _logger()
    paths = configured_build_paths() if paths is None else {**configured_build_paths(), **paths}
    architectures = list(read_architectures(paths["architectures_path"]))
    audits = list(read_audit_records(paths["audit_path"]))
    arrays = list(read_trf_arrays(paths["trf_arrays_path"]))
    retained, _ = partition_vmatch_repeats(read_repeats(paths["matches_path"]), arrays)
    evidence_items = list(read_evidence(paths["evidence_path"]))

    within_direct = Counter(
        repeat.source_match_id
        for repeat in retained
        if repeat.repeat_class == "direct" and repeat.scope in {"within_left", "within_right"}
    )
    audited = Counter(pair["source_match_id"] for record in audits for pair in record["pairs"])
    if audited != within_direct or any(count != 1 for count in audited.values()):
        raise ValueError("Retained within-flank direct pairs and V5 unit records disagree")

    spaced = [item for item in architectures if item.repeat_class != "tandem_array"]
    supported = {source for item in spaced for copy in item.copies for source in copy.support}
    used = {pair["source_match_id"] for record in audits for pair in record["pairs"] if pair["used"]}
    if used != supported:
        raise ValueError("Pairs marked used differ from the union of V5 copy supports")

    audit_ids = [identifier for record in audits for identifier in record["architecture_ids"]]
    if Counter(audit_ids) != Counter(item.step4c_architecture_id for item in spaced):
        raise ValueError("V5 catalog rows and unit audit records disagree")

    trf_architectures = [item for item in architectures if item.repeat_class == "tandem_array"]
    array_keys = [array.trf_array_key for array in arrays]
    catalog_array_keys = [item.trf_array_key for item in trf_architectures]
    if (
        len(array_keys) != len(set(array_keys))
        or len(catalog_array_keys) != len(set(catalog_array_keys))
        or set(array_keys) != set(catalog_array_keys)
    ):
        raise ValueError("TRF catalog rows do not resolve accepted arrays exactly")

    within_retained_pairs = Counter(
        repeat.source_match_id
        for repeat in retained
        if repeat.scope in WITHIN_FLANK_SCOPES and repeat.repeat_class in PAIR_REPEAT_CLASSES
    )
    evidence_pairs = Counter(
        item.evidence_id
        for item in evidence_items
        if item.repeat_class in PAIR_REPEAT_CLASSES
    )
    if evidence_pairs != within_retained_pairs or any(count != 1 for count in evidence_pairs.values()):
        raise ValueError("Retained within-flank Vmatch pairs and evidence pairs disagree")

    evidence_arrays = Counter(
        item.evidence_id for item in evidence_items if item.repeat_class == "tandem_array"
    )
    if evidence_arrays != Counter(array.trf_array_key for array in arrays) or any(
        count != 1 for count in evidence_arrays.values()
    ):
        raise ValueError("Accepted TRF arrays and evidence arrays disagree")

    evidence_ids = {item.evidence_id for item in evidence_items}
    if not supported.issubset(evidence_ids):
        raise ValueError("V5 copy supports include ids missing from evidence")
    if not {item.trf_array_key for item in trf_architectures}.issubset(evidence_ids):
        raise ValueError("TRF architecture keys include ids missing from evidence")

    units = _read_unit_rows(paths["contexts_path"])
    expected_composition_units = {
        key for key, unit in units.items() if unit["scope"] in WITHIN_FLANK_SCOPES
    }
    composition_rows = list(read_unit_composition(paths["unit_composition_path"]))
    composition_units = {
        (row["protein_accession"], row["scope"], row["search_unit_id"])
        for row in composition_rows
    }
    if len(composition_rows) != len(expected_composition_units):
        raise ValueError("Unit composition row count disagrees with within-flank units")
    if composition_units != expected_composition_units:
        raise ValueError("Unit composition rows do not match within-flank units exactly")
    evidence_units = {
        (item.protein_accession, item.scope, item.search_unit_id) for item in evidence_items
    }
    if not evidence_units.issubset(composition_units):
        raise ValueError("Evidence items reference units missing from the composition table")

    evidence_by_class = Counter(item.repeat_class for item in evidence_items)
    logger.info(
        "Step 4C summary: %s architectures (%s V5, %s TRF), %s copies; "
        "%s V5 units, %s retained within-flank direct pairs, %s used; "
        "evidence %s direct, %s inverted, %s TRF, %s composition rows",
        f"{len(architectures):,}",
        f"{len(spaced):,}",
        f"{len(trf_architectures):,}",
        f"{sum(len(item.copies) for item in architectures):,}",
        f"{len(audits):,}",
        f"{len(audited):,}",
        f"{len(used):,}",
        f"{evidence_by_class.get('direct', 0):,}",
        f"{evidence_by_class.get('palindromic_inverted', 0):,}",
        f"{evidence_by_class.get('tandem_array', 0):,}",
        f"{len(composition_rows):,}",
    )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "summarize"))
    return parser.parse_args()

def main() -> None:
    action = _parse_args().action
    if action == "build": run_build()
    else: run_summarize()

if __name__ == "__main__":
    main()
