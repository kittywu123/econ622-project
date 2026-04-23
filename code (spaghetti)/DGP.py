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

_GREEK = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]

_COURSE_POOL = [f"ECON{num}" for num in [
    101, 102, 201, 202, 221, 301, 302, 306, 401, 402, 500, 502, 526,
]]


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
    course_rejections: list         # course_rejections[j] = set of student indices course j always rejects
    student_names: list             # student_names[i] = name string for student i
    course_codes: list              # course_codes[j] = code string for course j


def generate_market(
    n_students: int = 20,
    n_courses: int = 8,
    k: int = 3,
    sigma: float = 0.5,
    popularity_weight: float = 0.0,
    capacity_range: tuple = (1, 3),
    phd_fraction: float = 0.4,
    phd_required_fraction: float = 0.25,
    rejection_fraction: float = 0.1,
    sparse_prefs: bool = False,
    sparse_length: Optional[int] = None,
    fixed_capacities: Optional[np.ndarray] = None,
    fixed_course_codes: Optional[list] = None,
    fixed_phd_required: Optional[np.ndarray] = None,
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
    rejection_fraction : float
        Expected fraction of students each course rejects outright.
    popularity_weight : float
        Scale of a course-level popularity term shared by all students.
        0.0 (default) = pure skill match. Higher values make some courses
        universally preferred, creating competition for popular slots.
    fixed_capacities : np.ndarray or None
        If provided, overrides random capacity draws. Shape (n_courses,).
    fixed_course_codes : list or None
        If provided, overrides randomly sampled course codes.
    fixed_phd_required : np.ndarray or None
        If provided, overrides randomly drawn PhD-required flags. Shape (n_courses,).
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

    if popularity_weight > 0:
        course_popularity = rng.standard_normal(n_courses)  # μ_j, one per course
        student_scores = base + eps_student + popularity_weight * course_popularity
    else:
        student_scores = base + eps_student
    course_scores = base.T + eps_course       # v_ji (independent noise draws)

    # PhD flags
    phd_students = rng.random(n_students) < phd_fraction
    if fixed_phd_required is not None:
        phd_required = np.asarray(fixed_phd_required, dtype=bool)
    else:
        phd_required = rng.random(n_courses) < phd_required_fraction

    # Course rejection lists: each course rejects a random subset of students by index
    n_rejected = max(0, round(rejection_fraction * n_students))
    course_rejections = [
        set(rng.choice(n_students, size=n_rejected, replace=False).tolist())
        for _ in range(n_courses)
    ]

    # Course capacities
    if fixed_capacities is not None:
        capacities = np.asarray(fixed_capacities, dtype=int)
    else:
        lo, hi = capacity_range
        capacities = rng.integers(lo, hi + 1, size=n_courses)

    # Ensure enough PhD students to fill all PhD-required slots
    phd_slots_needed = int(capacities[phd_required].sum())
    shortfall = phd_slots_needed - int(phd_students.sum())
    if shortfall > 0:
        non_phd = np.where(~phd_students)[0]
        phd_students[rng.choice(non_phd, size=min(shortfall, len(non_phd)), replace=False)] = True

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

    # Student names: Greek letters, with numeric suffix if n_students > 24
    base_names = [_GREEK[i % len(_GREEK)] for i in range(n_students)]
    seen: dict = {}
    student_names = []
    for name in base_names:
        seen[name] = seen.get(name, 0) + 1
        student_names.append(name if seen[name] == 1 else f"{name}{seen[name]}")

    # Course codes
    if fixed_course_codes is not None:
        course_codes = list(fixed_course_codes)
    else:
        course_pool = list(_COURSE_POOL)
        rng.shuffle(course_pool)
        course_codes = course_pool[:n_courses]

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
        course_rejections=course_rejections,
        student_names=student_names,
        course_codes=course_codes,
    )


def summarize_market(market: Market) -> None:
    """Print a quick summary of a generated market."""
    print(f"Students : {market.n_students}  ({market.phd_students.sum()} PhD)")
    print(f"Courses  : {market.n_courses}  ({market.phd_required.sum()} require PhD TA)")
    caps = {market.course_codes[j]: market.capacities[j] for j in range(market.n_courses)}
    print(f"Capacities: {caps}  (total slots: {market.capacities.sum()})")
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
        prefs = [m.course_codes[j] for j in m.student_prefs[i]]
        print(f"  {m.student_names[i]}: {prefs}")

    print("\nCourse preference lists (first 3 courses):")
    for j in range(3):
        prefs = [m.student_names[i] for i in m.course_prefs[j]]
        print(f"  {m.course_codes[j]}: {prefs}")
