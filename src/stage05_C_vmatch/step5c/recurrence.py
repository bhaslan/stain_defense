from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Hashable, Iterable, Mapping, Sequence

import edlib
from Bio.Seq import Seq

from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step4c.evidence import EvidenceItem
from src.stage05_C_vmatch.step4c.pipeline import _read_representative_sequences

from .models import Opportunity

UnitKey = tuple[str, str, str]

@dataclass(frozen=True, slots=True)
class Piece:

    start: int
    end: int
    array_spans: tuple[tuple[int, int], ...]
    sequence: str

    def array_covers(self, start: int, end: int) -> bool:

        return any(span_start <= self.start + start and self.start + end <= span_end
                   for span_start, span_end in self.array_spans)

@dataclass(frozen=True, slots=True)
class PieceMatch:

    piece_1: Piece
    piece_2: Piece
    similarity: float
    orientation: str

@dataclass(frozen=True, slots=True)
class RecurrentPair:

    unit_1_id: str
    unit_2_id: str
    side: str
    matches: tuple[PieceMatch, ...]
    decided_by_tandem: bool

def load_unit_sequences(opportunities: Iterable[Opportunity]) -> dict[UnitKey, str]:

    units = {(unit.protein_accession, unit.search_unit_id, unit.database_side): unit.upstream_key
             for unit in opportunities}
    sequences = _read_representative_sequences(cfg.STEP3_CONTEXTS_PATH, set(units))

    return {units[key]: sequence for key, sequence in sequences.items()}

def unit_pieces(opportunities: Iterable[Opportunity],
                counted: Iterable[EvidenceItem],
                sequences: Mapping[UnitKey, str]) -> dict[str, tuple[Piece, ...]]:

    stretches: dict[UnitKey, list[tuple[int, int, bool]]] = defaultdict(list)

    for item in counted:

        key = item.protein_accession, item.scope, item.search_unit_id
        is_array = item.repeat_class == "tandem_array"
        stretches[key].extend((stretch.start, stretch.end, is_array) for stretch in item.stretches)

    pieces: dict[str, tuple[Piece, ...]] = {}

    for unit in opportunities:

        if unit.upstream_key not in stretches: continue
        sequence = sequences[unit.upstream_key]
        merged: list[list] = []

        for start, end, is_array in sorted(stretches[unit.upstream_key]):

            if merged and start < merged[-1][1]: merged[-1][1] = max(merged[-1][1], end)
            else: merged.append([start, end, []])
            if is_array: merged[-1][2].append((start, end))

        pieces[unit.representative_unit_id] = tuple(Piece(start, end, _union(arrays), sequence[start:end])
                                                    for start, end, arrays in merged)

    return pieces

def _union(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:

    union: list[list[int]] = []

    for start, end in spans:

        if union and start <= union[-1][1]: union[-1][1] = max(union[-1][1], end)
        else: union.append([start, end])

    return tuple((start, end) for start, end in union)

def _infix_identity(first: str, second: str) -> float:

    query, target = (first, second) if len(first) <= len(second) else (second, first)
    distance = edlib.align(query, target, mode="HW", task="distance")["editDistance"]

    return 1 - distance / len(query)

def piece_similarity(first: Piece, second: Piece) -> tuple[float, str]:

    direct = _infix_identity(first.sequence, second.sequence)
    reverse = _infix_identity(first.sequence, str(Seq(second.sequence).reverse_complement()))
    orientation = "direct" if direct > reverse else "reverse_complement" if reverse > direct else "ambiguous"

    return max(direct, reverse), orientation

def _inside_arrays(match: PieceMatch) -> bool:

    shorter, longer = ((match.piece_1, match.piece_2) if len(match.piece_1.sequence) <= len(match.piece_2.sequence)
                       else (match.piece_2, match.piece_1))
    if not shorter.array_covers(0, len(shorter.sequence)) or not longer.array_spans: return False
    flip = match.orientation == "reverse_complement"
    flip_longer = flip and longer is match.piece_2
    query = str(Seq(shorter.sequence).reverse_complement()) if flip and not flip_longer else shorter.sequence
    target = str(Seq(longer.sequence).reverse_complement()) if flip_longer else longer.sequence
    first, last = edlib.align(query, target, mode="HW", task="locations")["locations"][0]
    start, end = (len(target) - 1 - last, len(target) - first) if flip_longer else (first, last + 1)

    return longer.array_covers(start, end)

def _candidate_matches(first: Sequence[Piece], second: Sequence[Piece]) -> list[PieceMatch]:

    threshold = float(cfg.STEP5_SIMILARITY_THRESHOLD)
    candidates: list[PieceMatch] = []

    for piece_1 in first:

        for piece_2 in second:

            similarity, orientation = piece_similarity(piece_1, piece_2)
            if similarity >= threshold: candidates.append(PieceMatch(piece_1, piece_2, similarity, orientation))

    candidates.sort(key=lambda match: (match.piece_1.start, match.piece_1.end, match.piece_2.start, match.piece_2.end))

    return candidates

def _assign(candidates: Sequence[PieceMatch]) -> tuple[PieceMatch, ...]:

    chosen = optimal_assignment([(match.piece_1, match.piece_2, match.similarity) for match in candidates])

    return tuple(candidates[index] for index in chosen)

def match_pieces(first: Sequence[Piece], second: Sequence[Piece]) -> tuple[PieceMatch, ...]:

    return _assign(_candidate_matches(first, second))

def recurrent_pairs(units: Iterable[Opportunity],
                    pieces: Mapping[str, Sequence[Piece]]) -> tuple[RecurrentPair, ...]:

    by_side: dict[str, list[str]] = defaultdict(list)

    for unit in units:

        if unit.representative_unit_id in pieces: by_side[unit.database_side].append(unit.representative_unit_id)

    pairs: list[RecurrentPair] = []

    for side in sorted(by_side):

        for unit_1_id, unit_2_id in combinations(sorted(by_side[side]), 2):

            candidates = _candidate_matches(pieces[unit_1_id], pieces[unit_2_id])
            matches = _assign(candidates)

            if len(matches) >= cfg.STEP5_MIN_INDEPENDENT_MATCHES:
                pairs.append(RecurrentPair(unit_1_id, unit_2_id, side, matches, False))
                continue

            arrays = [match for match in candidates if _inside_arrays(match)]
            if arrays: pairs.append(RecurrentPair(unit_1_id, unit_2_id, side, (max(arrays, key=lambda match: match.similarity),), True))

    return tuple(pairs)

@dataclass(slots=True)
class _FlowEdge:

    destination: int
    reverse_index: int
    capacity: int
    cost: tuple[Fraction, int]

def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, destination: int, cost: tuple[Fraction, int]) -> int:

    forward_index, reverse_index = len(graph[source]), len(graph[destination])
    graph[source].append(_FlowEdge(destination, reverse_index, 1, cost))
    graph[destination].append(_FlowEdge(source, forward_index, 0, (-cost[0], -cost[1])))

    return forward_index

def _shortest_path(graph: list[list[_FlowEdge]], source: int) -> list[tuple[int, int] | None]:

    distances: list[tuple[Fraction, int] | None] = [None] * len(graph)
    previous: list[tuple[int, int] | None] = [None] * len(graph)
    distances[source] = (Fraction(0), 0)

    for _ in range(len(graph) - 1):

        changed = False

        for node, edges in enumerate(graph):

            if distances[node] is None: continue

            for edge_index, edge in enumerate(edges):

                if edge.capacity == 0: continue
                distance = (distances[node][0] + edge.cost[0], distances[node][1] + edge.cost[1])

                if distances[edge.destination] is None or distance < distances[edge.destination]:

                    distances[edge.destination] = distance
                    previous[edge.destination] = node, edge_index
                    changed = True

        if not changed: break

    return previous

def optimal_assignment(edges: Sequence[tuple[Hashable, Hashable, float]]) -> tuple[int, ...]:

    if not edges: return ()
    left_nodes = {left: 1 + index for index, left in enumerate(dict.fromkeys(edge[0] for edge in edges))}
    right_offset = 1 + len(left_nodes)
    right_nodes = {right: right_offset + index for index, right in enumerate(dict.fromkeys(edge[1] for edge in edges))}
    source, sink = 0, right_offset + len(right_nodes)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    zero = (Fraction(0), 0)

    for node in left_nodes.values(): _add_flow_edge(graph, source, node, zero)

    for node in right_nodes.values(): _add_flow_edge(graph, node, sink, zero)

    edge_refs: list[tuple[int, int]] = []

    for rank, (left, right, similarity) in enumerate(edges):

        cost = (-Fraction(*float(similarity).as_integer_ratio()), -(1 << (len(edges) - rank - 1)))
        edge_refs.append((left_nodes[left], _add_flow_edge(graph, left_nodes[left], right_nodes[right], cost)))

    while (previous := _shortest_path(graph, source))[sink] is not None:

        node = sink

        while node != source:

            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse_index].capacity = 1
            node = parent

    return tuple(index for index, (node, edge_index) in enumerate(edge_refs) if graph[node][edge_index].capacity == 0)
