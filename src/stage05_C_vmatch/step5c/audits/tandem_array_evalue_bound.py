from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from math import ceil, comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.repeat_architecture_inference.evalue_markov import (
    BASES,
    repeat_evalue_markov,
    stationary_distribution,
    transition_matrix_from_counts,
)
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step3c.trf import PROFILE_ARGUMENTS, PROFILE_TOKEN
from src.stage05_C_vmatch.step3c.trf_evidence import normalize_trf_arrays, parse_trf_outputs

GC_WINDOWS = {"GC35": (0.33, 0.37), "GC50": (0.48, 0.52), "GC28": (0.26, 0.30)}
BOUND_LENGTHS = (10, 11, 15, 20, 30)
TRF_LENGTHS = range(10, 31)
TRF_SUBSTITUTIONS = range(4)
TRIALS = 10
SEED = 1

def placement_probability(transition: dict[str, dict[str, float]], n: int, copies: int, length: int, mismatches: int) -> float:

    return repeat_evalue_markov(transition, n, copies, length, mismatches) / comb(n - length + 1, copies)

def tandem_evalue(transition: dict[str, dict[str, float]], n: int, copies: int, length: int, mismatches: int) -> float:

    return (n - copies * length + 1) * placement_probability(transition, n, copies, length, mismatches)

@dataclass(frozen=True, eq=False)
class Background:

    name: str
    transition: dict[str, dict[str, float]]
    units: int

    @classmethod
    def pooled(cls, composition: pd.DataFrame, name: str) -> Background:

        low, high = GC_WINDOWS[name]
        full = composition[composition.sequence_length == cfg.FLANK_LENGTH]
        gc = (full.count_C + full.count_G) / full[[f"count_{base}" for base in BASES]].sum(axis=1)
        chosen = full[(gc >= low) & (gc < high)]
        counts = {first: {second: float(chosen[f"count_{first}{second}"].sum()) for second in BASES} for first in BASES}

        return cls(name, transition_matrix_from_counts(counts, pseudocount=cfg.STEP4_MARKOV_PSEUDOCOUNT), len(chosen))

    @property
    def stationary(self) -> dict[str, float]:

        return stationary_distribution(self.transition)

    @cache
    def placement_probability(self, copies: int, length: int, mismatches: int) -> float:

        return placement_probability(self.transition, cfg.FLANK_LENGTH, copies, length, mismatches)

    def spaced_pair_evalue(self, length: int, mismatches: int) -> float:

        return repeat_evalue_markov(self.transition, cfg.FLANK_LENGTH, 2, length, mismatches)

    def tandem_evalue(self, copies: int, length: int, mismatches: int) -> float:

        return (cfg.FLANK_LENGTH - copies * length + 1) * self.placement_probability(copies, length, mismatches)

    def iid_extra_copy(self, length: int, mismatches: int) -> float:

        p_match = sum(freq**2 for freq in self.stationary.values())

        return sum(comb(length, j) * p_match**(length - j) * (1 - p_match)**j for j in range(mismatches + 1))

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:

        cumulative = np.cumsum([[self.transition[a][b] for b in BASES] for a in BASES], axis=1)
        bases = [rng.choice(4, p=[self.stationary[base] for base in BASES])]

        for _ in range(length - 1): bases.append(min(int(np.searchsorted(cumulative[bases[-1]], rng.random(), side="right")), 3))

        return np.array(bases)

def bound_rows(background: Background) -> list[dict[str, object]]:

    rows = []

    for length in BOUND_LENGTHS:

        for mismatches in range(int(0.2 * length) + 1):

            pair = background.tandem_evalue(2, length, mismatches)
            triple = background.tandem_evalue(3, length, mismatches)
            rows.append({"background": background.name, "L": length, "m": mismatches,
                         "E_pair_spaced": background.spaced_pair_evalue(length, mismatches),
                         "E_pair_tandem": pair, "E_triple_tandem": triple, "fold_2_to_3": pair / triple,
                         "q_markov": (background.placement_probability(3, length, mismatches)
                                      / background.placement_probability(2, length, mismatches)),
                         "q_iid": background.iid_extra_copy(length, mismatches)})

    return rows

def planted_flank(background: Background, copies: int, length: int, substitutions: int,
                  rng: np.random.Generator) -> tuple[str, int, int]:

    unit = background.sample(length, rng)
    array = np.tile(unit, (copies, 1))

    for index in range(substitutions):

        copy, position = 1 + index % (copies - 1), rng.integers(length)
        array[copy, position] = (array[copy, position] + rng.integers(1, 4)) % 4

    left = (cfg.FLANK_LENGTH - copies * length) // 2
    bases = np.concatenate([background.sample(left, rng), array.ravel(),
                            background.sample(cfg.FLANK_LENGTH - left - copies * length, rng)])

    return "".join(BASES[base] for base in bases), left, left + copies * length

def trf_reports(sequence: str, start: int, end: int, length: int) -> bool:

    with tempfile.TemporaryDirectory(prefix="trf-probe-") as directory:

        workdir = Path(directory)
        (workdir / "input.fna").write_text(f">probe\n{sequence}\n")
        subprocess.run([str(cfg.TRF_BIN), "input.fna", *map(str, PROFILE_ARGUMENTS), "-d"],
                       cwd=workdir, check=False, capture_output=True)
        raw = parse_trf_outputs(workdir / f"input.fna.{PROFILE_TOKEN}.dat", workdir / f"input.fna.{PROFILE_TOKEN}.1.txt.html",
                                protein_accession="probe", scope="within_left", search_unit_id="probe",
                                source_sequence_id="probe", side="left", source_sequence=sequence)

    accepted = normalize_trf_arrays(raw, minimum_period=cfg.TRF_MIN_PERIOD, minimum_full_copies=cfg.TRF_MIN_FULL_COPIES,
                                    tau_span=cfg.TRF_NORMALIZATION_TAU_SPAN, tau_score=cfg.TRF_NORMALIZATION_TAU_SCORE,
                                    minimum_reciprocal_overlap=cfg.TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP)

    return any(array.period == length and array.start < end and start < array.end for array in accepted)

def admissible_rows(background: Background, seed: int) -> list[dict[str, object]]:

    rng = np.random.default_rng(seed)
    rows = []

    for length in TRF_LENGTHS:

        for substitutions in TRF_SUBSTITUTIONS:

            copies = cfg.TRF_MIN_FULL_COPIES

            while not (reported := sum(trf_reports(*planted_flank(background, copies, length, substitutions, rng), length)
                                       for _ in range(TRIALS))):

                copies += 1

            mismatches = ceil(substitutions / (copies - 1))
            rows.append({"background": background.name, "L": length, "s": substitutions,
                         "trf_min_copies": copies, "trf_report_rate": reported / TRIALS,
                         "m_per_copy": mismatches,
                         "E_tandem_at_min_copies": background.tandem_evalue(copies, length, mismatches)})
            print(rows[-1], flush=True)

    return rows

def main() -> None:

    composition = pd.read_csv(cfg.STEP4_UNIT_COMPOSITION_PATH, sep="\t")
    backgrounds = [Background.pooled(composition, name) for name in GC_WINDOWS]

    for background in backgrounds:

        print(f"{background.name}: {background.units:,} units, GC {background.stationary['C'] + background.stationary['G']:.3f}")

    bound = pd.DataFrame([row for background in backgrounds for row in bound_rows(background)])
    bound.to_csv(cfg.STEP5_TANDEM_EVALUE_BOUND_PATH, sep="\t", index=False, float_format="%.3g")

    print(bound.to_string(index=False, float_format="%.3g"))
    print(f"\nTABLE {cfg.STEP5_TANDEM_EVALUE_BOUND_PATH}")

    admissible = pd.DataFrame([row for background in backgrounds for row in admissible_rows(background, SEED)])
    admissible.to_csv(cfg.STEP5_TRF_ADMISSIBLE_EVALUE_PATH, sep="\t", index=False, float_format="%.3g")

    print(f"\nTABLE {cfg.STEP5_TRF_ADMISSIBLE_EVALUE_PATH}")

if __name__ == "__main__": main()
