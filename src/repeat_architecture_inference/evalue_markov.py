from collections import defaultdict
from itertools import product
from math import comb

import numpy as np

BASES = "ACGT"

"""

```python
seq = "AACCGTCAGACTGGAATCGATCGATCGATCGATCGATCGATCGATCG"

T = transition_matrix_from_sequence(
    seq,
    pseudocount=0.5
)

E = repeat_evalue_markov(
    T,
    n=len(seq),
    r=3,
    l=14,
    k=2
)

print(T)
print(E)
```

This assumes a **stationary first-order Markov background**, and for an r-copy repeat it requires copies 2…r each to have **≤ `k` Hamming mismatches relative to copy 1**.


A small pseudocount such as:

```python
pseudocount=0.5
```

is sensible if the background sequence is short, because otherwise an unobserved transition gets probability exactly zero.

The important improvement over the previous function is that this no longer reduces the background to a single `p_match`. 

For example, a sequence with strong `A→T→A→T` behavior will correctly make alternating repeat patterns more likely under the null.

Also, this version computes the **full ≤k mismatch probability**, rather than the earlier REPuter-style dominant `exactly k` term. 

So statistically, this is actually closer to the event:

> “an r-copy repeat of length at least this similarity exists by chance.”

The remaining approximation is mainly the search-space multiplication: overlapping candidate arms are not independent, 
so the resulting E-value should still be viewed as a first-order analytical approximation.

"""

def transition_matrix_from_counts(counts, pseudocount=0.0):

    adjusted = {
        a: {b: float(counts[a][b]) + pseudocount for b in BASES}
        for a in BASES
    }
    T = {}

    for a in BASES:
        total = sum(adjusted[a].values())

        if total == 0:
            raise ValueError(
                f"No outgoing transitions observed from {a}. "
                "Use a small pseudocount."
            )

        T[a] = {b: adjusted[a][b] / total for b in BASES}

    return T

def transition_matrix_from_sequence(seq, pseudocount=0.0):
    seq = seq.upper()
    counts = {a: {b: 0.0 for b in BASES} for a in BASES}

    for a, b in zip(seq, seq[1:]):
        if a in BASES and b in BASES:
            counts[a][b] += 1

    return transition_matrix_from_counts(counts, pseudocount)

def stationary_distribution(T):
    M = np.array([
        [T[a][b] for b in BASES]
        for a in BASES
    ])

    A = np.vstack([
        M.T - np.eye(4),
        np.ones(4)
    ])

    b = np.array([0, 0, 0, 0, 1.0])

    pi, *_ = np.linalg.lstsq(A, b, rcond=None)

    pi = np.clip(pi, 0, None)
    pi /= pi.sum()

    return dict(zip(BASES, pi))

def repeat_evalue_markov(T, n, r, l, k):

    pi = stationary_distribution(T)

    R = {
        x: {
            c: pi[c] * T[c][x] / pi[x]
            for c in BASES
        }
        for x in BASES
    }

    base_tuples = list(product(BASES, repeat=r))

    dp = defaultdict(float)

    for x in base_tuples:

        mismatches = tuple(
            int(x[i] != x[0])
            for i in range(1, r)
        )

        if any(m > k for m in mismatches):
            continue

        p = 1.0
        for base in x:
            p *= pi[base]

        p_left_extend = 0.0

        for previous_base in BASES:
            q = 1.0

            for current_base in x:
                q *= R[current_base][previous_base]

            p_left_extend += q

        p_left_maximal = 1.0 - p_left_extend

        dp[(x, mismatches)] += p * p_left_maximal

    for _ in range(1, l):

        new_dp = defaultdict(float)

        for (current, mismatches), state_prob in dp.items():

            for nxt in base_tuples:

                new_mismatches = tuple(
                    mismatches[i - 1]
                    + int(nxt[i] != nxt[0])
                    for i in range(1, r)
                )

                if any(m > k for m in new_mismatches):
                    continue

                transition_prob = 1.0

                for a, b in zip(current, nxt):
                    transition_prob *= T[a][b]

                new_dp[(nxt, new_mismatches)] += (
                    state_prob * transition_prob
                )

        dp = new_dp

    p_repeat = 0.0

    for (current, mismatches), state_prob in dp.items():

        p_right_extend = 0.0

        for next_base in BASES:
            q = 1.0

            for current_base in current:
                q *= T[current_base][next_base]

            p_right_extend += q

        p_right_maximal = 1.0 - p_right_extend

        p_repeat += state_prob * p_right_maximal

    num_starts = n - l + 1

    search_space = comb(num_starts, r)

    return search_space * p_repeat
