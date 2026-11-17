from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step3c.vmatch import (
    _sequence_hash,
    _validate_single_fasta,
    _write_json_atomic,
    _write_single_fasta,
    _write_tsv,
    _write_unit_summary,
    load_search_units,
)
from src.stage05_C_vmatch.step3c.trf_evidence import (
    SPAN_SCORE_SMALLEST_PERIOD_POLICY,
    TRF_ARRAY_FIELDS,
    normalize_trf_arrays,
    parse_trf_outputs,
)

PROFILE_NAME = "mismatch-7"
_LEGACY_PROFILE_NAMES = {"mismatch_7"}
PROFILE_ARGUMENTS = (
    cfg.TRF_MATCH_WEIGHT,
    cfg.TRF_MISMATCH_PENALTY,
    cfg.TRF_INDEL_PENALTY,
    cfg.TRF_MATCH_PROBABILITY,
    cfg.TRF_INDEL_PROBABILITY,
    cfg.TRF_MIN_SCORE,
    cfg.TRF_MAX_PERIOD,
)
PROFILE_TOKEN = ".".join(map(str, PROFILE_ARGUMENTS))
RUN_FIELDS = tuple(
    (
        "protein_accession candidate_group experimental_anchor scope "
        "search_unit_id profile repeat_class side source_sequence_id "
        "sequence_length raw_calls accepted_arrays dat_path alignment_path "
        "summary_html_path state_path command_path stdout_path stderr_path"
    ).split()
)

def profile_summary() -> list[dict[str, str]]:
    return [
        {
            "scope": scope,
            "profile": PROFILE_NAME,
            "repeat_class": "tandem_array",
            "trf_arguments": " ".join(map(str, PROFILE_ARGUMENTS)),
        }
        for scope in ("within_left", "within_right")
    ]

def configuration_summary() -> dict[str, object]:
    return {
        "executable": cfg.TRF_BIN,
        "profile": PROFILE_NAME,
        "arguments": PROFILE_ARGUMENTS,
        "normalization_policy": cfg.TRF_NORMALIZATION_POLICY,
        "minimum_period": cfg.TRF_MIN_PERIOD,
        "minimum_full_copies": cfg.TRF_MIN_FULL_COPIES,
        "normalization_tau_span": cfg.TRF_NORMALIZATION_TAU_SPAN,
        "normalization_tau_score": cfg.TRF_NORMALIZATION_TAU_SCORE,
        "normalization_minimum_reciprocal_overlap": (
            cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP
        ),
        "runs_path": cfg.STEP3_TRF_RUNS_PATH,
        "arrays_path": cfg.STEP3_TRF_ARRAYS_PATH,
        "normalization_fingerprint_path": (
            cfg.STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH
        ),
    }

def _canonical_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _record_fingerprint(records: list[dict[str, object]]) -> str:
    canonical_records = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return _canonical_fingerprint(canonical_records)

def _normalization_policy() -> dict[str, object]:
    accepted_name = SPAN_SCORE_SMALLEST_PERIOD_POLICY
    if cfg.TRF_NORMALIZATION_POLICY != accepted_name:
        raise ValueError(
            "Production TRF normalization policy is not implemented: "
            f"{cfg.TRF_NORMALIZATION_POLICY!r}"
        )
    settings = {
        "minimum_period": cfg.TRF_MIN_PERIOD,
        "minimum_full_copies": cfg.TRF_MIN_FULL_COPIES,
        "tau_span": cfg.TRF_NORMALIZATION_TAU_SPAN,
        "tau_score": cfg.TRF_NORMALIZATION_TAU_SCORE,
        "minimum_reciprocal_overlap": (
            cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP
        ),
    }
    return {
        "name": accepted_name,
        "settings": settings,
        "settings_fingerprint": _canonical_fingerprint(
            {"name": accepted_name, "settings": settings}
        ),
    }

def _normalization_fingerprint_record(
    evidence: dict[str, list[dict[str, object]]],
    arrays_path: str | Path,
    accepted_array_count: int,
) -> dict[str, object]:
    policy = _normalization_policy()
    input_fingerprint = _record_fingerprint(evidence["inputs"])
    raw_candidate_fingerprint = _record_fingerprint(
        evidence["raw_candidates"]
    )
    accepted_output_fingerprint = _file_fingerprint(arrays_path)
    return {
        "format_version": 1,
        "kernel": "K1_trf_candidate_resolution",
        "policy": policy,
        "input_fingerprint": input_fingerprint,
        "input_unit_count": len(evidence["inputs"]),
        "raw_candidate_fingerprint": raw_candidate_fingerprint,
        "raw_candidate_count": len(evidence["raw_candidates"]),
        "normalization_input_fingerprint": _canonical_fingerprint(
            {
                "input_fingerprint": input_fingerprint,
                "raw_candidate_fingerprint": raw_candidate_fingerprint,
                "policy": policy,
            }
        ),
        "accepted_output_path": str(Path(arrays_path)),
        "accepted_output_fingerprint": accepted_output_fingerprint,
        "accepted_array_count": accepted_array_count,
    }

def _unit_directory(unit: dict[str, object], output_root: Path) -> Path:
    return (
        output_root
        / str(unit["protein_accession"])
        / "trf"
        / str(unit["scope"])
        / str(unit["search_unit_id"])
    )

def _unit_state(unit: dict[str, object]) -> dict[str, object]:
    return {
        "format_version": 1,
        "scope": unit["scope"],
        "search_unit_id": unit["search_unit_id"],
        "source_sequence_id": unit["database_sequence_id"],
        "side": unit["database_side"],
        "sequence_sha256": _sequence_hash(unit["database_sequence"]),
        "profile": {
            "name": PROFILE_NAME,
            "arguments": list(PROFILE_ARGUMENTS),
        },
        "executable": str(cfg.TRF_BIN),
    }

def _paths(unit: dict[str, object], output_root: Path) -> dict[str, Path]:
    unit_dir = _unit_directory(unit, output_root)
    prefix = f"input.fna.{PROFILE_TOKEN}"
    return {
        "unit_dir": unit_dir,
        "input": unit_dir / "input.fna",
        "state": unit_dir / "unit_state.json",
        "dat": unit_dir / f"{prefix}.dat",
        "alignment": unit_dir / f"{prefix}.1.txt.html",
        "summary_html": unit_dir / f"{prefix}.1.html",
        "command": unit_dir / "command.txt",
        "stdout": unit_dir / "stdout.txt",
        "stderr": unit_dir / "stderr.txt",
    }

def _write_text_atomic(path: Path, text: str) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

def _prepare_unit(
    unit: dict[str, object], output_root: Path, *, create: bool
) -> dict[str, Path]:
    paths = _paths(unit, output_root)
    expected = _unit_state(unit)
    if not paths["state"].exists():
        if not create:
            raise FileNotFoundError(f"TRF unit state is missing: {paths['state']}")
        paths["unit_dir"].mkdir(parents=True, exist_ok=True)
        if any(paths["unit_dir"].iterdir()):
            raise ValueError(
                "TRF unit directory has no state but is not empty: "
                f"{paths['unit_dir']}"
            )
        _write_single_fasta(
            paths["input"],
            str(unit["database_sequence_id"]),
            str(unit["database_sequence"]),
        )
        _write_json_atomic(paths["state"], expected)
    else:
        observed = json.loads(paths["state"].read_text(encoding="utf-8"))
        observed_profile = observed.get("profile")
        rewrite_state = False
        if isinstance(observed_profile, dict):
            observed_profile.pop("minimum_period", None)
            observed_profile.pop("minimum_full_copies", None)
            if observed_profile.get("name") in _LEGACY_PROFILE_NAMES:
                observed_profile["name"] = PROFILE_NAME
                rewrite_state = True
        if observed != expected:
            raise ValueError(
                f"TRF inputs or settings changed for {paths['unit_dir']}; "
                "do not reuse its stale native output."
            )
        if rewrite_state:
            _write_json_atomic(paths["state"], expected)
    _validate_single_fasta(
        paths["input"],
        str(unit["database_sequence_id"]),
        str(unit["database_sequence"]),
    )
    return paths

def _run_trf(paths: dict[str, Path]) -> None:
    command = [str(cfg.TRF_BIN), "input.fna", *map(str, PROFILE_ARGUMENTS), "-d"]
    with tempfile.TemporaryDirectory(
        prefix="trf-run-", dir=paths["unit_dir"]
    ) as temporary_name:
        temporary = Path(temporary_name)
        shutil.copyfile(paths["input"], temporary / "input.fna")
        result = subprocess.run(
            command,
            cwd=temporary,
            check=False,
            capture_output=True,
            text=True,
        )
        produced = {
            "dat": temporary / f"input.fna.{PROFILE_TOKEN}.dat",
            "alignment": temporary / f"input.fna.{PROFILE_TOKEN}.1.txt.html",
            "summary_html": temporary / f"input.fna.{PROFILE_TOKEN}.1.html",
        }
        missing = [name for name, path in produced.items() if not path.exists()]
        if missing:
            raise RuntimeError(
                f"TRF failed ({result.returncode}) or omitted expected output "
                f"{', '.join(missing)}: {shlex.join(command)}\n"
                f"{result.stderr.strip()}"
            )
        for name, source in produced.items():
            source.replace(paths[name])
        _write_text_atomic(paths["stdout"], result.stdout)
        _write_text_atomic(paths["stderr"], result.stderr)
        _write_text_atomic(paths["command"], shlex.join(command) + "\n")

def discover_units(
    units: list[dict[str, object]],
    output_root: str | Path,
    *,
    run_missing: bool,
    show_progress: bool = True,
    normalization_evidence: dict[str, list[dict[str, object]]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _normalization_policy()
    if cfg.TRF_MIN_PERIOD < 1 or cfg.TRF_MIN_PERIOD > cfg.TRF_MAX_PERIOD:
        raise ValueError("TRF_MIN_PERIOD must lie within the searched period range")
    if cfg.TRF_MIN_FULL_COPIES < 2:
        raise ValueError("TRF_MIN_FULL_COPIES must be at least two")
    if not cfg.TRF_BIN.exists():
        raise FileNotFoundError(cfg.TRF_BIN)
    output_root = Path(output_root)
    arrays: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    within_units = [
        unit
        for unit in units
        if unit["scope"] in {"within_left", "within_right"}
    ]
    with tqdm(
        within_units,
        desc="TRF discovery" if run_missing else "TRF normalization",
        disable=not show_progress,
    ) as progress:
        for unit in progress:
            progress.set_postfix_str(
                f"{unit['protein_accession']} {unit['search_unit_id']}"
            )
            paths = _prepare_unit(unit, output_root, create=run_missing)
            outputs_exist = paths["dat"].exists() and paths["alignment"].exists()
            if run_missing and not outputs_exist:
                _run_trf(paths)
            elif not outputs_exist:
                raise FileNotFoundError(
                    f"Native TRF output is missing under {paths['unit_dir']}"
                )
            try:
                raw = parse_trf_outputs(
                    paths["dat"],
                    paths["alignment"],
                    protein_accession=str(unit["protein_accession"]),
                    scope=str(unit["scope"]),
                    search_unit_id=str(unit["search_unit_id"]),
                    source_sequence_id=str(unit["database_sequence_id"]),
                    side=str(unit["database_side"]),
                    source_sequence=str(unit["database_sequence"]),
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse native TRF outputs for {paths['unit_dir']}"
                ) from exc
            if normalization_evidence is not None:
                normalization_evidence["inputs"].append(
                    {
                        "protein_accession": str(unit["protein_accession"]),
                        "scope": str(unit["scope"]),
                        "search_unit_id": str(unit["search_unit_id"]),
                        "source_sequence_id": str(
                            unit["database_sequence_id"]
                        ),
                        "side": str(unit["database_side"]),
                        "source_sequence_sha256": _sequence_hash(
                            str(unit["database_sequence"])
                        ),
                        "native_dat_sha256": _file_fingerprint(paths["dat"]),
                        "native_alignment_sha256": _file_fingerprint(
                            paths["alignment"]
                        ),
                    }
                )
                normalization_evidence["raw_candidates"].extend(
                    item.to_row() for item in raw
                )
            accepted = normalize_trf_arrays(
                raw,
                minimum_period=cfg.TRF_MIN_PERIOD,
                minimum_full_copies=cfg.TRF_MIN_FULL_COPIES,
                tau_span=cfg.TRF_NORMALIZATION_TAU_SPAN,
                tau_score=cfg.TRF_NORMALIZATION_TAU_SCORE,
                minimum_reciprocal_overlap=(
                    cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP
                ),
            )
            arrays.extend(item.to_row() for item in accepted)
            runs.append(
                {
                    "protein_accession": unit["protein_accession"],
                    "candidate_group": unit["candidate_group"],
                    "experimental_anchor": unit["experimental_anchor"],
                    "scope": unit["scope"],
                    "search_unit_id": unit["search_unit_id"],
                    "profile": PROFILE_NAME,
                    "repeat_class": "tandem_array",
                    "side": unit["database_side"],
                    "source_sequence_id": unit["database_sequence_id"],
                    "sequence_length": len(str(unit["database_sequence"])),
                    "raw_calls": len(raw),
                    "accepted_arrays": len(accepted),
                    "dat_path": paths["dat"],
                    "alignment_path": paths["alignment"],
                    "summary_html_path": paths["summary_html"],
                    "state_path": paths["state"],
                    "command_path": paths["command"],
                    "stdout_path": paths["stdout"],
                    "stderr_path": paths["stderr"],
                }
            )
    return arrays, runs

def process_step3c(*, run_missing: bool) -> list[dict[str, object]]:
    units = load_search_units()
    _write_unit_summary(units, output_root=cfg.PROTEINS_DIR)
    normalization_evidence: dict[str, list[dict[str, object]]] = {
        "inputs": [],
        "raw_candidates": [],
    }
    arrays, runs = discover_units(
        units,
        cfg.PROTEINS_DIR,
        run_missing=run_missing,
        show_progress=sys.stderr.isatty(),
        normalization_evidence=normalization_evidence,
    )
    _write_tsv(cfg.STEP3_TRF_ARRAYS_PATH, TRF_ARRAY_FIELDS, arrays)
    _write_tsv(cfg.STEP3_TRF_RUNS_PATH, RUN_FIELDS, runs)
    _write_json_atomic(
        cfg.STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH,
        _normalization_fingerprint_record(
            normalization_evidence,
            cfg.STEP3_TRF_ARRAYS_PATH,
            len(arrays),
        ),
    )
    print(f"TRF run summary: {cfg.STEP3_TRF_RUNS_PATH}")
    print(f"Accepted TRF arrays: {cfg.STEP3_TRF_ARRAYS_PATH}")
    print(
        "TRF normalization fingerprint: "
        f"{cfg.STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH}"
    )
    return runs

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", default="run", choices=("run", "normalize"))
    return parser.parse_args()

def main() -> None:
    process_step3c(run_missing=_parse_args().action == "run")

if __name__ == "__main__":
    main()
