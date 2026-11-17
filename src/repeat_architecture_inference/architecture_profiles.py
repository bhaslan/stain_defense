from dataclasses import dataclass
from itertools import accumulate
from math import comb, isfinite, log

BASES = "ACGT"

def mismatch_count(first, second):

    return sum(a != b or a not in BASES for a, b in zip(first, second))

def column_scores(agreements, pair_count, target_identity, p_match):

    if not (isfinite(target_identity) and p_match < target_identity < 1):

        raise ValueError("Column scoring requires background agreement < target identity < 1")

    reward = log(target_identity / p_match)
    penalty = log((1 - target_identity) / (1 - p_match))

    return tuple(agree * reward + (pair_count - agree) * penalty for agree in agreements)

def best_interval(scores):

    if not scores: raise ValueError("Cannot score an empty profile")
    best = running = scores[0]
    start = best_start = 0
    best_end = 1

    for index, score in enumerate(scores[1:], 1):

        if running < 0: running, start = score, index
        else: running += score

        if running > best: best, best_start, best_end = running, start, index + 1

    return best_start, best_end, best

@dataclass
class ConservationProfile:

    starts: tuple[int, ...]
    lower: int
    upper: int
    sequences: tuple[str, ...]
    disagreements: tuple[tuple[int, ...], ...]
    agreements: tuple[int, ...]
    target_identity: float | None
    unavailable: str | None
    scores: tuple[float, ...] | None
    prefix: tuple[float, ...] | None

    @classmethod
    def build(cls, evidence, starts, bounds, target_identity):

        lower, upper = bounds
        sequences = tuple(evidence.sequence[start + lower:start + upper] for start in starts)
        counts = [{base: column.count(base) for base in BASES} for column in zip(*sequences)]
        agreements = tuple(sum(comb(count, 2) for count in column.values()) for column in counts)
        disagreements = tuple(tuple(len(starts) - column.get(base, 1) for base, column in zip(sequence, counts))
                              for sequence in sequences)
        p_match = sum(p**2 for p in evidence.iid.background.values())
        unavailable = None

        if target_identity is None: unavailable = "No target identity supplied or recorded by the search"
        elif not p_match < target_identity < 1: unavailable = "Require background agreement < target identity < 1"

        scores = None if unavailable else column_scores(agreements, comb(len(starts), 2), target_identity, p_match)
        prefix = None if scores is None else tuple(accumulate(scores, initial=0.0))

        return cls(tuple(starts), lower, upper, sequences, disagreements, agreements,
                   target_identity, unavailable, scores, prefix)

    def score(self, left, right):

        if not self.lower <= left < right <= self.upper: raise ValueError("Window outside fixed profile envelope")
        if self.prefix is None: return None

        return self.prefix[right - self.lower] - self.prefix[left - self.lower]

    def window(self, left, right):

        score = self.score(left, right)
        a, b = left - self.lower, right - self.lower
        reference = self.sequences[0][a:b]
        mismatches = [mismatch_count(reference, sequence[a:b]) for sequence in self.sequences]

        return dict(left=left, right=right, column_score=score, mismatches=mismatches)

    def to_dict(self):

        preferred = None

        if self.scores is not None:

            left, right, score = best_interval(self.scores)
            preferred = dict(left=left + self.lower, right=right + self.lower, score=score)

        return dict(lower=self.lower, upper=self.upper, target_identity=self.target_identity,
                    unavailable=self.unavailable, agreeing_pairs=list(self.agreements),
                    pair_count=comb(len(self.starts), 2),
                    column_scores=None if self.scores is None else list(self.scores),
                    preferred_unconstrained=preferred,
                    copies=[dict(seed_start=start, sequence=sequence,
                                 mismatch_to_first="".join(str(mismatch_count(a, b)) for a, b in zip(sequence, self.sequences[0])),
                                 disagreements_with_others=list(disagreement))
                            for start, sequence, disagreement in zip(self.starts, self.sequences, self.disagreements)])

def write_decisions(diagnostics, out):

    out.write("\nDecision profiles (offsets relative to each seed; not confidence or indel diagnoses)\n")
    out.write(f"settings: {diagnostics['settings']}\n")

    for decision in diagnostics["decisions"]:

        out.write(f"\nSeed {decision['id']}: {decision['seed']} -> {decision['status']}\n")
        out.write(f"linked seed: {decision['related_seed']}; window rejections: {decision['rejections']}\n")
        out.write(f"seed window: {decision['seed_window']}\nchosen window: {decision['chosen_window']}\n")
        out.write(f"chosen proposal: {decision['chosen']}; column score change: {decision['column_gain']}\n")
        profile = decision["profile"]
        out.write(f"fixed envelope [{profile['lower']}:{profile['upper']}); "
                  f"unconstrained preference: {profile['preferred_unconstrained']}\n")
        if profile["unavailable"]: out.write(f"column scoring unavailable: {profile['unavailable']}\n")

        for offset in range(0, profile["upper"] - profile["lower"], 60):

            end = offset + 60
            out.write(f"offset {profile['lower'] + offset:+d}\n")
            scores = profile["column_scores"]
            if scores is not None: out.write("score: " + " ".join(f"{v:.2f}" for v in scores[offset:end]) + "\n")
            out.write(f"agree/{profile['pair_count']}: " + " ".join(map(str, profile["agreeing_pairs"][offset:end])) + "\n")

            for copy in profile["copies"]:

                out.write(f"  @{copy['seed_start']} {copy['sequence'][offset:end]}\n")
                out.write(f"    mismatch-to-first: {copy['mismatch_to_first'][offset:end]}\n")
                out.write("    disagreeing-others: " + " ".join(map(str, copy["disagreements_with_others"][offset:end])) + "\n")
