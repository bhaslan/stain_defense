from dataclasses import dataclass

from .evalue import repeat_evalue
from .architecture_profiles import BASES, mismatch_count
from .evalue_markov import repeat_evalue_markov, transition_matrix_from_sequence

WILDCARDS = "NSYWRKVBDHM"

@dataclass(frozen=True)
class Match:

    match_id: int
    start1: int
    end1: int
    start2: int
    end2: int
    distance: int
    evalue: float | None

    @property
    def length(self):

        return self.end1 - self.start1

    @property
    def coordinate_shift(self):

        return self.start2 - self.start1

    @property
    def is_hamming(self):

        return self.distance is not None and self.distance <= 0

    def supports(self, start, other_start, length):

        if self.coordinate_shift == other_start - start: first = start
        elif self.coordinate_shift == start - other_start: first = other_start
        else: return False

        return max(self.start1, first) < min(self.end1, first + length)

@dataclass(frozen=True)
class EvalueModel:

    name: str
    background: object
    formula: object

    @classmethod
    def named(cls, name, sequence):

        if name == "markov": return cls(name, transition_matrix_from_sequence(sequence, 0.5), repeat_evalue_markov)

        nucleotides = sum(sequence.count(base) for base in BASES)

        return cls(name, {base: sequence.count(base) / nucleotides for base in BASES}, repeat_evalue)

    def calculate(self, n, r, l, k):

        return self.formula(self.background, n, r, l, k)

class Evidence:

    def __init__(self, sequence, matches, pair_evalue_model="iid", *, truth_path=None, search_identity=None):

        self.sequence = sequence.upper()
        self.truth_path = truth_path
        self.search_identity = search_identity
        self.matches = {}
        n = len(self.sequence)
        if set(self.sequence) - set(BASES + WILDCARDS): raise ValueError("Expected a DNA reference of ACGT and the IUPAC wildcards Vmatch accepts")
        if not set(self.sequence) & set(BASES): raise ValueError("Expected a reference with at least one A, C, G or T")

        for record in matches:

            match = Match(**record)
            if match.match_id in self.matches: raise ValueError(f"Duplicate match ID: {match.match_id}")
            if not (0 <= match.start1 < match.end1 <= n and 0 <= match.start2 < match.end2 <= n):

                raise ValueError(f"Match {match.match_id}: coordinates outside reference")

            self.matches[match.match_id] = match

        self.iid = EvalueModel.named("iid", self.sequence)
        self.pair_model = EvalueModel.named(pair_evalue_model, self.sequence)

    def calculate(self, r, l, k):

        return self.iid.calculate(len(self.sequence), r, l, k)

    def pair_evalue(self, match):

        if not match.is_hamming: return None

        return self.pair_model.calculate(len(self.sequence), 2, match.length, abs(match.distance))

    def supporting_matches(self, start, other_start, length):

        return [match.match_id for match in self.matches.values() if match.supports(start, other_start, length)]

@dataclass(frozen=True)
class Proposal:

    unit_length: int
    copy_starts: tuple[int, ...]

@dataclass(frozen=True)
class Copy:

    start: int
    end: int
    sequence: str
    mismatches: int
    support: tuple[int, ...]

@dataclass
class Architecture:

    unit_length: int
    copies: tuple[Copy, ...]
    repeat_evalue: float

    @classmethod
    def from_proposal(cls, proposal, evidence):

        length, starts = proposal.unit_length, tuple(sorted(proposal.copy_starts))
        cls.validate_claim(length, starts, evidence)
        links = {start: {} for start in starts}

        for index, start in enumerate(starts):

            for other in starts[index + 1:]:

                found = evidence.supporting_matches(start, other, length)
                if found: links[start][other] = links[other][start] = found

        cls.validate_support(links, length)

        return cls._from_coordinates(length, starts, evidence, links)

    @classmethod
    def from_truth(cls, proposal, evidence):

        length, starts = proposal.unit_length, tuple(sorted(proposal.copy_starts))
        cls.validate_claim(length, starts, evidence)

        if any(b < a + length for a, b in zip(starts, starts[1:])):

            raise ValueError("Planted copies must not overlap")

        return cls._from_coordinates(length, starts, evidence, {start: {} for start in starts})

    @classmethod
    def _from_coordinates(cls, length, starts, evidence, links):

        reference = evidence.sequence[starts[0]:starts[0] + length]
        copies = []

        for start in starts:

            sequence = evidence.sequence[start:start + length]
            support = tuple(sorted({match_id for found in links[start].values() for match_id in found}))
            mismatches = mismatch_count(reference, sequence)
            copies.append(Copy(start, start + length, sequence, mismatches, support))

        copies = tuple(copies)
        max_mismatches = max(copy.mismatches for copy in copies)

        return cls(length, copies, evidence.calculate(len(copies), length, max_mismatches))

    @staticmethod
    def validate_claim(length, starts, evidence):

        if length < 1: raise ValueError(f"Unit length must be positive: {length}")
        if len(starts) < 2: raise ValueError("An architecture requires at least two copies")
        if len(set(starts)) != len(starts): raise ValueError(f"Duplicate copy start: {starts}")

        for start in starts:

            if not (0 <= start and start + length <= len(evidence.sequence)):

                raise ValueError(f"Copy [{start}:{start + length}) has invalid coordinates")

    @staticmethod
    def validate_support(links, length):

        groups, seen = [], set()

        for start in links:

            if start in seen: continue

            group, stack = set(), [start]

            while stack:

                current = stack.pop()
                if current in group: continue
                group.add(current)
                stack.extend(links[current])

            seen |= group
            groups.append([f"[{member}:{member + length})" for member in sorted(group)])

        if len(groups) > 1:

            listed = "; ".join(" ".join(group) for group in groups)
            raise ValueError(f"Supporting pairs leave {len(groups)} separately aligned groups instead of one architecture: {listed}")

    @property
    def copy_number(self):

        return len(self.copies)

    @property
    def max_mismatches(self):

        return max(copy.mismatches for copy in self.copies)

    @property
    def spacers(self):

        return tuple(right.start - left.end for left, right in zip(self.copies, self.copies[1:]))

    @property
    def summary(self):

        return (f"r={self.copy_number} l={self.unit_length} k={self.max_mismatches} "
                f"spacers={self.spacers} repeat_evalue={self.repeat_evalue:.3e}")

    def to_dict(self):

        return dict(unit_length=self.unit_length, copy_starts=[copy.start for copy in self.copies],
                    copy_number=self.copy_number, max_mismatches=self.max_mismatches,
                    spacers=list(self.spacers), repeat_evalue=self.repeat_evalue,
                    copies=[dict(start=copy.start, end=copy.end, mismatches=copy.mismatches,
                                 support=list(copy.support), sequence=copy.sequence) for copy in self.copies],)
