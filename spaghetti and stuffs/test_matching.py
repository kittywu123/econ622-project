import numpy as np
import pytest
from DGP import generate_market, Market, _GREEK
from da import deferred_acceptance
from sd import serial_dictatorship


# ── helpers ──────────────────────────────────────────────────────────────────

def is_stable(assignment: np.ndarray, market: Market) -> bool:
    """Return True iff no blocking pair exists (respects PhD constraint and sparse prefs)."""
    pref_set = [set(prefs) for prefs in market.student_prefs]
    for i in range(market.n_students):
        for j in range(market.n_courses):
            if j not in pref_set[i]:
                continue
            if market.phd_required[j] and not market.phd_students[i]:
                continue
            if assignment[i] == j:
                continue
            # Does i prefer j over their current match (or are they unmatched)?
            if assignment[i] >= 0:
                if market.student_rankings[i, j] >= market.student_rankings[i, assignment[i]]:
                    continue
            # i wants j — can j take i?
            assigned_to_j = np.where(assignment == j)[0]
            if len(assigned_to_j) < market.capacities[j]:
                return False  # j has a vacant slot: blocking pair
            if any(market.course_rankings[j, i] < market.course_rankings[j, k]
                   for k in assigned_to_j):
                return False  # j prefers i over someone it holds: blocking pair
    return True


# ── DGP tests ─────────────────────────────────────────────────────────────────

class TestGenerateMarket:
    def test_shapes(self):
        m = generate_market(n_students=15, n_courses=6, seed=0)
        assert m.student_scores.shape == (15, 6)
        assert m.course_scores.shape == (6, 15)
        assert m.student_rankings.shape == (15, 6)
        assert m.course_rankings.shape == (6, 15)
        assert m.capacities.shape == (6,)
        assert m.phd_students.shape == (15,)
        assert m.phd_required.shape == (6,)

    def test_names_and_codes_length(self):
        m = generate_market(n_students=10, n_courses=5, seed=1)
        assert len(m.student_names) == 10
        assert len(m.course_codes) == 5

    def test_student_names_unique_within_24(self):
        m = generate_market(n_students=24, seed=2)
        assert len(set(m.student_names)) == 24

    def test_course_codes_unique(self):
        m = generate_market(n_courses=10, seed=3)
        assert len(set(m.course_codes)) == 10

    def test_course_codes_all_econ(self):
        m = generate_market(seed=4)
        assert all(code.startswith("ECON") for code in m.course_codes)

    def test_student_names_are_greek(self):
        m = generate_market(n_students=10, seed=5)
        bases = [name.rstrip("0123456789") for name in m.student_names]
        assert all(base in _GREEK for base in bases)

    def test_rankings_are_permutations(self):
        m = generate_market(seed=6)
        for i in range(m.n_students):
            assert sorted(m.student_rankings[i]) == list(range(m.n_courses))
        for j in range(m.n_courses):
            assert sorted(m.course_rankings[j]) == list(range(m.n_students))

    def test_prefs_consistent_with_rankings(self):
        m = generate_market(seed=7)
        for i in range(m.n_students):
            ranks = [m.student_rankings[i, j] for j in m.student_prefs[i]]
            assert ranks == sorted(ranks)

    def test_reproducibility(self):
        m1 = generate_market(seed=99)
        m2 = generate_market(seed=99)
        np.testing.assert_array_equal(m1.student_scores, m2.student_scores)
        assert m1.student_names == m2.student_names
        assert m1.course_codes == m2.course_codes


# ── DA tests ──────────────────────────────────────────────────────────────────

class TestDeferredAcceptance:
    def test_output_shape(self):
        m = generate_market(seed=10)
        a = deferred_acceptance(m)
        assert a.shape == (m.n_students,)

    def test_assignment_values_in_range(self):
        m = generate_market(seed=11)
        a = deferred_acceptance(m)
        assert np.all((a >= -1) & (a < m.n_courses))

    def test_capacity_not_exceeded(self):
        m = generate_market(seed=12)
        a = deferred_acceptance(m)
        for j in range(m.n_courses):
            assert (a == j).sum() <= m.capacities[j]

    def test_matched_course_in_pref_list(self):
        m = generate_market(seed=13)
        a = deferred_acceptance(m)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert a[i] in m.student_prefs[i]

    def test_phd_constraint(self):
        m = generate_market(seed=14, phd_required_fraction=0.5)
        a = deferred_acceptance(m)
        for i in range(m.n_students):
            if a[i] >= 0 and m.phd_required[a[i]]:
                assert m.phd_students[i], (
                    f"{m.student_names[i]} is non-PhD but matched to "
                    f"PhD-required {m.course_codes[a[i]]}"
                )

    def test_stability(self):
        for seed in range(5):
            m = generate_market(seed=seed)
            a = deferred_acceptance(m)
            assert is_stable(a, m), f"Unstable matching for seed={seed}"

    def test_stability_sparse_prefs(self):
        m = generate_market(seed=20, sparse_prefs=True)
        a = deferred_acceptance(m)
        assert is_stable(a, m)

    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=30,
                            capacity_range=(10, 10), phd_required_fraction=0.0)
        a = deferred_acceptance(m)
        assert np.all(a >= 0), "Everyone should match when total slots >> n_students"

    def test_all_unmatched_when_all_phd_required_no_phd_students(self):
        m = generate_market(n_students=10, n_courses=4, seed=31,
                            phd_fraction=0.0, phd_required_fraction=1.0)
        a = deferred_acceptance(m)
        assert np.all(a == -1), "Non-PhD students should not match PhD-required courses"


# ── SD tests ──────────────────────────────────────────────────────────────────

class TestSerialDictatorship:
    def test_output_shape(self):
        m = generate_market(seed=40)
        a = serial_dictatorship(m, seed=0)
        assert a.shape == (m.n_students,)

    def test_assignment_values_in_range(self):
        m = generate_market(seed=41)
        a = serial_dictatorship(m, seed=0)
        assert np.all((a >= -1) & (a < m.n_courses))

    def test_capacity_not_exceeded(self):
        m = generate_market(seed=42)
        a = serial_dictatorship(m, seed=0)
        for j in range(m.n_courses):
            assert (a == j).sum() <= m.capacities[j]

    def test_matched_course_in_pref_list(self):
        m = generate_market(seed=43)
        a = serial_dictatorship(m, seed=0)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert a[i] in m.student_prefs[i]

    def test_phd_constraint(self):
        m = generate_market(seed=44, phd_required_fraction=0.5)
        a = serial_dictatorship(m, seed=0)
        for i in range(m.n_students):
            if a[i] >= 0 and m.phd_required[a[i]]:
                assert m.phd_students[i], (
                    f"{m.student_names[i]} is non-PhD but matched to "
                    f"PhD-required {m.course_codes[a[i]]}"
                )

    def test_first_student_gets_top_choice(self):
        # The first student in the order should always get their top valid choice
        m = generate_market(seed=45, phd_required_fraction=0.0)
        order = np.arange(m.n_students)
        a = serial_dictatorship(m, order=order)
        assert a[0] == m.student_prefs[0][0]

    def test_fixed_order_is_deterministic(self):
        m = generate_market(seed=46)
        order = np.arange(m.n_students)
        a1 = serial_dictatorship(m, order=order)
        a2 = serial_dictatorship(m, order=order)
        np.testing.assert_array_equal(a1, a2)

    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=47,
                            capacity_range=(10, 10), phd_required_fraction=0.0)
        a = serial_dictatorship(m, seed=0)
        assert np.all(a >= 0)

    def test_all_unmatched_when_all_phd_required_no_phd_students(self):
        m = generate_market(n_students=10, n_courses=4, seed=48,
                            phd_fraction=0.0, phd_required_fraction=1.0)
        a = serial_dictatorship(m, seed=0)
        assert np.all(a == -1)
