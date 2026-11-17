from dataclasses import dataclass
from math import isfinite

from collections import defaultdict
from .architecture_model import Proposal
from .evalue import log_repeat_evalue
from .architecture_profiles import ConservationProfile, mismatch_count

@dataclass(frozen=True)
class Candidate:

    unit_length: int
    copy_starts: tuple[int, ...]

    @property
    def copy_count(self):

        return len(self.copy_starts)

    @property
    def total_core_bases(self):

        return self.unit_length * self.copy_count

class UnionFind:

    def __init__(self, size):

        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):

        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.rank[a] < self.rank[b]: a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]: self.rank[a] += 1

class PositionGraph:

    def __init__(self, sequence_length, matches):

        self.uf = UnionFind(sequence_length)

        for match in matches:

            for offset in range(match.length):

                self.uf.union(match.start1 + offset, match.start2 + offset)

        self.components = defaultdict(list)

        for position in range(sequence_length):

            self.components[self.uf.find(position)].append(position)

    def positions_at(self, position):

        return self.components[self.uf.find(position)]

    def column_label_at(self, position):

        return tuple(p - position for p in self.positions_at(position))

    @property
    def largest_component(self):

        return max(len(positions) for positions in self.components.values())

@dataclass(frozen=True)
class ScoredCandidate:

    proposal: Proposal
    log_evalue: float
    column_score: float | None = None

    @property
    def unit_length(self):

        return self.proposal.unit_length

    @property
    def copy_starts(self):

        return self.proposal.copy_starts

    @property
    def copy_count(self):

        return len(self.copy_starts)

    @property
    def score_key(self):

        return self.log_evalue, -self.copy_count, -self.unit_length, self.copy_starts

@dataclass
class Decision:

    id: int
    seed: Candidate
    profile: ConservationProfile
    status: str
    chosen: ScoredCandidate | None = None
    related_seed: int | None = None
    rejections: dict | None = None

    def to_dict(self):

        seed_window = self.profile.window(0, self.seed.unit_length)
        chosen_window = None
        gain = None

        if self.chosen is not None:

            left = self.chosen.copy_starts[0] - self.seed.copy_starts[0]
            chosen_window = self.profile.window(left, left + self.chosen.unit_length)

            if seed_window["column_score"] is not None:

                gain = chosen_window["column_score"] - seed_window["column_score"]

        chosen = None if self.chosen is None else dict(
            proposal=dict(unit_length=self.chosen.unit_length, copy_starts=list(self.chosen.copy_starts)),
            log_evalue=self.chosen.log_evalue, column_score=self.chosen.column_score)

        return dict(id=self.id, seed=dict(unit_length=self.seed.unit_length, copy_starts=list(self.seed.copy_starts)),
                    status=self.status, chosen=chosen,
                    related_seed=self.related_seed, rejections=self.rejections,
                    seed_window=seed_window, chosen_window=chosen_window, column_gain=gain,
                    profile=self.profile.to_dict())

class Register:

    def __init__(self, seed, evidence):

        self.seed = seed
        self.evidence = evidence
        starts = seed.copy_starts
        indices = {start: index for index, start in enumerate(starts)}
        self.minimum_gap = min(b - a for a, b in zip(starts, starts[1:]))
        self.edges = []

        for match in evidence.matches.values():

            for index, start in enumerate(starts):

                other = indices.get(start + match.coordinate_shift)
                if other is None: continue
                self.edges.append((index, other, match.start1 - start, match.end1 - start))

    def bounds(self, mode):

        left, right = 0, self.seed.unit_length

        if mode == "trim-and-extend":

            merged = []

            for a, b in sorted((a, b) for _, _, a, b in self.edges):

                if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b)
                else: merged.append([a, b])

            for a, b in merged:

                if a <= 0 and b >= self.seed.unit_length:

                    left, right = a, b
                    break

        starts = self.seed.copy_starts

        return max(left, -starts[0]), min(right, len(self.evidence.sequence) - starts[-1])

    def connected(self, left, right):

        adjacency = [set() for _ in self.seed.copy_starts]

        for first, second, a, b in self.edges:

            if max(a, left) >= min(b, right): continue
            adjacency[first].add(second)
            adjacency[second].add(first)

        seen, pending = set(), [0]

        while pending:

            current = pending.pop()
            if current in seen: continue
            seen.add(current)
            pending.extend(adjacency[current] - seen)

        return len(seen) == len(adjacency)

    def windows(self, mode, minimum_length):

        if mode == "none":

            yield 0, self.seed.unit_length

            return

        lower, upper = self.bounds(mode)

        for left in range(lower, upper - minimum_length + 1):

            for right in range(left + minimum_length, min(upper, left + self.minimum_gap) + 1):

                if left >= self.seed.unit_length or right <= 0: continue

                yield left, right

    def refine(self, mode, minimum_length, objective, profile):

        sequence = self.evidence.sequence
        starts = self.seed.copy_starts
        background = self.evidence.iid.background
        mismatch_probability = 1 - sum(freq**2 for freq in background.values())
        best = None
        best_key = None
        rejections = dict(overlap=0, disconnected=0, high_mismatch=0, degenerate_background=0)
        examined = 0

        if mismatch_probability <= 0:

            rejections["degenerate_background"] = 1

            return best, rejections

        for left, right in self.windows(mode, minimum_length):

            examined += 1
            length = right - left
            if length > self.minimum_gap:

                rejections["overlap"] += 1
                continue

            if not self.connected(left, right):

                rejections["disconnected"] += 1
                continue
            reference = sequence[starts[0] + left:starts[0] + right]
            k = max(mismatch_count(reference, sequence[start + left:start + right]) for start in starts[1:])
            if k > length * mismatch_probability:

                rejections["high_mismatch"] += 1
                continue

            score = log_repeat_evalue(background, len(sequence), len(starts), length, k)
            proposal = Proposal(length, tuple(start + left for start in starts))
            candidate = ScoredCandidate(proposal, score, profile.score(left, right))
            key = candidate.score_key if objective == "iid" else (-candidate.column_score, -length, proposal.copy_starts)

            if best is None or key < best_key: best, best_key = candidate, key

        if not examined: rejections["no_windows_within_length_limits"] = 1

        return best, rejections

class ArchitectureDetector:

    def __init__(self, evidence, min_copies=2, min_unit_length=10, min_total_core_bases=20,
                 refinement="none", selection="legacy", objective="iid", target_identity=None):

        if refinement not in ("none", "trim", "trim-and-extend"): raise ValueError(refinement)
        if selection not in ("legacy", "score-aware"): raise ValueError(selection)
        if objective not in ("iid", "column"): raise ValueError(objective)

        if target_identity is not None and not (isfinite(target_identity) and 0 < target_identity < 100):

            raise ValueError("Target identity must be a percentage strictly between 0 and 100")

        self.target_identity = evidence.search_identity if target_identity is None else target_identity
        self.identity_source = "search" if target_identity is None else "override"
        self.objective = objective
        theta = None if self.target_identity is None else self.target_identity / 100
        p_match = sum(p**2 for p in evidence.iid.background.values())

        if objective == "column" and (theta is None or not p_match < theta < 1):

            raise ValueError("Column objective requires background agreement < target identity < 100%; supply --target-identity")

        if min_copies < 2 or min_unit_length < 1 or min_total_core_bases < 1:

            raise ValueError("Require min_copies >= 2 and positive length/core minima")

        self.sequence = evidence.sequence
        self.matches = tuple(evidence.matches.values())
        self.min_copies = min_copies
        self.min_unit_length = min_unit_length
        self.min_total_core_bases = min_total_core_bases
        self.graph = PositionGraph(len(self.sequence), self.matches)
        self.evidence = evidence
        self.refinement = refinement
        self.selection = selection
        self.notes = []
        self.decisions = []

    @property
    def diagnostics(self):

        return dict(settings=dict(objective=self.objective, target_identity=self.target_identity,
                                  identity_source=self.identity_source, refinement=self.refinement,
                                  selection=self.selection, min_copies=self.min_copies,
                                  min_unit_length=self.min_unit_length, min_total_core_bases=self.min_total_core_bases),
                    decisions=[decision.to_dict() for decision in self.decisions])

    def detect(self):

        seeds = self._find_translation_coherent_runs()
        self.decisions = []
        candidates = {}
        supported = rejected = 0
        theta = None if self.target_identity is None else self.target_identity / 100

        for index, seed in enumerate(seeds, 1):

            register = Register(seed, self.evidence)
            profile = ConservationProfile.build(self.evidence, seed.copy_starts,
                                                register.bounds("trim-and-extend"), theta)
            decision = Decision(index, seed, profile, "unsupported_seed")
            self.decisions.append(decision)
            if not register.connected(0, seed.unit_length): continue
            supported += 1
            minimum_length = max(self.min_unit_length,
                                 (self.min_total_core_bases + seed.copy_count - 1) // seed.copy_count)
            candidate, decision.rejections = register.refine(self.refinement, minimum_length, self.objective, profile)
            decision.chosen = candidate

            if candidate is None:

                decision.status = "no_admissible_window"
                rejected += 1
                continue

            if candidate.proposal in candidates:

                decision.status = "duplicate"
                decision.related_seed = candidates[candidate.proposal].id
                continue

            decision.status = "selected"
            candidates[candidate.proposal] = decision

        ordered = list(candidates.values())

        if self.selection == "legacy": ordered.sort(key=lambda d: (-d.chosen.copy_count, -d.chosen.unit_length))
        else: ordered.sort(key=lambda d: d.chosen.score_key)

        kept = []

        for decision in ordered:

            suppressor = next((better for better in kept if self._is_shadow(decision.chosen, better.chosen)), None)

            if suppressor is not None:

                decision.status = "suppressed"
                decision.related_seed = suppressor.id
                continue

            kept.append(decision)

        self.notes = [
            f"refinement={self.refinement}; selection={self.selection}; objective={self.objective}; "
            f"target_identity={self.target_identity} ({self.identity_source}); "
            f"min_copies={self.min_copies}; min_unit_length={self.min_unit_length}; "
            f"min_total_core_bases={self.min_total_core_bases}",
            f"Position graph: {len(self.graph.components)} components; largest holds "
            f"{self.graph.largest_component} of {len(self.sequence)} positions",
            f"Supported seeds={supported}; no admissible window={rejected}; "
            f"distinct refined proposals={len(ordered)}; selected={len(kept)}",
            "Admissible: non-overlapping copies, connected pair support, low-mismatch iid branch. "
            "Membership is fixed; short seeds are not rescued. Score-aware selection uses iid E-values."]

        return [decision.chosen.proposal for decision in kept]

    def _find_translation_coherent_runs(self):

        labels = [self.graph.column_label_at(p) for p in range(len(self.sequence))]
        candidates = {}
        i = 0

        while i < len(labels):

            label = labels[i]
            j = i + 1

            while j < len(labels) and labels[j] == label:

                j += 1

            if len(label) > 1:

                candidate = Candidate(j - i, tuple(self.graph.positions_at(i)))

                if (candidate.copy_count >= self.min_copies
                    and candidate.unit_length >= self.min_unit_length
                    and candidate.total_core_bases >= self.min_total_core_bases):

                    candidates[candidate.unit_length, candidate.copy_starts] = candidate

            i = j

        return list(candidates.values())

    @staticmethod
    def _is_shadow(candidate, better):

        if candidate.copy_count > better.copy_count: return False
        better_starts = set(better.copy_starts)

        for anchor in better.copy_starts:

            delta = candidate.copy_starts[0] - anchor
            mapped_back = {start - delta for start in candidate.copy_starts}

            if (mapped_back.issubset(better_starts)
                and delta + candidate.unit_length >= 0 and delta <= better.unit_length):

                return True

        return False
