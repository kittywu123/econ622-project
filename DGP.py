"""
Data Generating Process for TA matching simulations.

Each student i has a latent skill / trait vector theta_i in R^k.
Each course j has a latent skill / trait requirement vector phi_j in R^k.
Preference scores: u_ij = theta_i @ phi_j + eps_ij, eps_ij ~ N(0, sigma^2)
Preferences are symmetric: courses also score students the same way (independent noise).
Rankings are derived from scores.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Market:
    """Container for a simulated TA matching market."""
    n_students: int
    n_courses: int
    capacities: np.ndarray          # shape (n_courses,), slots per course
    student_scores: np.ndarray      # shape (n_students, n_courses), u_ij
    course_scores: np.ndarray       # shape (n_courses, n_students), v_ji
    student_rankings: np.ndarray    # shape (n_students, n_courses), rank of course j for student i (0 = top)
    course_rankings: np.ndarray     # shape (n_courses, n_students), rank of student i for course j (0 = top)
    student_prefs: list             # student_prefs[i] = ordered list of course indices (most to least preferred)
    course_prefs: list              # course_prefs[j] = ordered list of student indices (most to least preferred)
    theta: np.ndarray               # latent student skill vectors, shape (n_students, k)
    phi: np.ndarray                 # latent course skill vectors, shape (n_courses, k)
    phd_students: np.ndarray        # bool array shape (n_students,), True if PhD student
    phd_required: np.ndarray        # bool array shape (n_courses,), True if course requires a PhD TA


def generate_market(
    n_students: int = 20,
    n_courses: int = 8,
    k: int = 3,
    sigma: float = 0.5,
    capacity_range: tuple = (1, 3),
    phd_fraction: float = 0.4,
    phd_required_fraction: float = 0.25,
    sparse_prefs: bool = False,
    sparse_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> Market:
    """
    Generate a synthetic TA matching market.

    Parameters
    ----------
    n_students : int
        Number of TA candidates.
    n_courses : int
        Number of course-instructor pairs.
    k : int
        Dimension of latent skill vectors.
    sigma : float
        Std dev of idiosyncratic noise in preference scores.
    capacity_range : tuple
        (min, max) TA slots per course (inclusive, drawn uniformly).
    phd_fraction : float
        Fraction of students who are PhD students.
    phd_required_fraction : float
        Fraction of courses that require a PhD TA.
    sparse_prefs : bool
        If True, students only rank a subset of courses.
    sparse_length : int or None
        Number of courses each student ranks when sparse_prefs=True.
        Defaults to ceil(n_courses / 2).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Market
    """
    rng = np.random.default_rng(seed)

    # Latent skill vectors
    # DISTRIBUTED NORMAL FOR NOW - MAYBE CHANGE?
    theta = rng.standard_normal((n_students, k))  # student skills
    phi = rng.standard_normal((n_courses, k))     # course skill requirements

    # Preference scores: u_ij = theta_i @ phi_j + eps_ij
    base = theta @ phi.T  # (n_students, n_courses)
    eps_student = rng.normal(0, sigma, size=(n_students, n_courses))
    eps_course = rng.normal(0, sigma, size=(n_courses, n_students))

    student_scores = base + eps_student       # u_ij
    course_scores = base.T + eps_course       # v_ji (independent noise draws)

    # PhD flags
    phd_students = rng.random(n_students) < phd_fraction
    phd_required = rng.random(n_courses) < phd_required_fraction

    # Course capacities
    lo, hi = capacity_range
    capacities = rng.integers(lo, hi + 1, size=n_courses)

    # Rankings: rank of item j in agent i's list (0 = most preferred)
    student_rankings = np.argsort(np.argsort(-student_scores, axis=1), axis=1)
    course_rankings = np.argsort(np.argsort(-course_scores, axis=1), axis=1)

    # Ordered preference lists
    if sparse_prefs:
        if sparse_length is None:
            sparse_length = int(np.ceil(n_courses / 2))
        sparse_length = min(sparse_length, n_courses)
        student_prefs = [
            list(np.argsort(-student_scores[i]))[:sparse_length]
            for i in range(n_students)
        ]
    else:
        student_prefs = [list(np.argsort(-student_scores[i])) for i in range(n_students)]

    course_prefs = [list(np.argsort(-course_scores[j])) for j in range(n_courses)]

    return Market(
        n_students=n_students,
        n_courses=n_courses,
        capacities=capacities,
        student_scores=student_scores,
        course_scores=course_scores,
        student_rankings=student_rankings,
        course_rankings=course_rankings,
        student_prefs=student_prefs,
        course_prefs=course_prefs,
        theta=theta,
        phi=phi,
        phd_students=phd_students,
        phd_required=phd_required,
    )


def summarize_market(market: Market) -> None:
    """Print a quick summary of a generated market."""
    print(f"Students : {market.n_students}  ({market.phd_students.sum()} PhD)")
    print(f"Courses  : {market.n_courses}  ({market.phd_required.sum()} require PhD TA)")
    print(f"Capacities: {market.capacities}  (total slots: {market.capacities.sum()})")
    pref_len = [len(p) for p in market.student_prefs]
    print(f"Student pref list length: min={min(pref_len)}, max={max(pref_len)}")


if __name__ == "__main__":
    m = generate_market(seed=42)
    summarize_market(m)

    print("\nStudent scores (first 3 students x all courses):")
    print(m.student_scores[:3].round(2))

    print("\nCourse scores (first 3 courses x all students):")
    print(m.course_scores[:3].round(2))

    print("\nStudent preference lists (first 3 students):")
    for i in range(3):
        print(f"  Student {i}: {m.student_prefs[i]}")

    print("\nCourse preference lists (first 3 courses):")
    for j in range(3):
        print(f"  Course {j}: {m.course_prefs[j]}")
