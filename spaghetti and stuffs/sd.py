"""
Serial Dictatorship for TA matching.

Students are processed in a fixed priority order. Each student picks their
most-preferred course that still has remaining capacity and satisfies the
PhD constraint (if applicable). Earlier students have full priority over
later ones.
"""

import numpy as np
from typing import Optional
from DGP import Market


def serial_dictatorship(market: Market, order: Optional[np.ndarray] = None, seed: Optional[int] = None) -> np.ndarray:
    """
    Run serial dictatorship on a market.

    Parameters
    ----------
    market : Market
    order : array-like of int, optional
        Priority ordering of student indices. If None, a random order is used.
    seed : int or None
        Random seed for generating order when order is None.

    Returns
    -------
    assignment : np.ndarray, shape (n_students,)
        assignment[i] = j if student i is matched to course j, -1 if unmatched.
    """
    if order is None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(market.n_students)

    remaining = market.capacities.copy().astype(int)
    assignment = np.full(market.n_students, -1, dtype=int)

    for i in order:
        for j in market.student_prefs[i]:
            if market.phd_required[j] and not market.phd_students[i]:
                continue
            if i in market.course_rejections[j]:
                continue
            if remaining[j] > 0:
                assignment[i] = j
                remaining[j] -= 1
                break

    return assignment


if __name__ == "__main__":
    from DGP import generate_market
    from da import summarize_assignment

    m = generate_market(seed=42)
    assignment = serial_dictatorship(m, seed=0)
    summarize_assignment(assignment, m)

    print("\nFull assignment (student -> course, -1 = unmatched):")
    for i, j in enumerate(assignment):
        phd = " [PhD]" if m.phd_students[i] else ""
        if j >= 0:
            rank = m.student_rankings[i, j]
            phd_req = " [PhD required]" if m.phd_required[j] else ""
            course = f"{m.course_codes[j]}{phd_req} (rank {rank})"
        else:
            course = "unmatched"
        print(f"  {m.student_names[i]}{phd}: {course}")
