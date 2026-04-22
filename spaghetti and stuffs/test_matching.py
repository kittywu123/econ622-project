import numpy as np
import pytest
from DGP import generate_market, Market, _GREEK
from da import deferred_acceptance
from sd import serial_dictatorship
from mip import solve_mip, MIPResult

_SOLVER = "GUROBI"


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


# ── MIP tests ─────────────────────────────────────────────────────────────────

class TestMIP:
    def _solve(self, market, objective="student", lp_relax=False):
        return solve_mip(market, objective=objective, lp_relax=lp_relax, solver=_SOLVER)

    # ── structural ────────────────────────────────────────────────────────────

    def test_output_shape(self):
        m = generate_market(seed=50)
        res = self._solve(m)
        assert res.assignment.shape == (m.n_students,)

    def test_assignment_values_in_range(self):
        m = generate_market(seed=51)
        res = self._solve(m)
        assert np.all((res.assignment >= -1) & (res.assignment < m.n_courses))

    def test_capacity_not_exceeded(self):
        m = generate_market(seed=52)
        res = self._solve(m)
        for j in range(m.n_courses):
            assert (res.assignment == j).sum() <= m.capacities[j]

    def test_phd_constraint(self):
        m = generate_market(seed=53, phd_required_fraction=0.5)
        for obj in ("student", "course", "bilateral", "egalitarian"):
            res = self._solve(m, objective=obj)
            for i in range(m.n_students):
                if res.assignment[i] >= 0 and m.phd_required[res.assignment[i]]:
                    assert m.phd_students[i]

    def test_rejection_list_respected(self):
        m = generate_market(seed=54, rejection_fraction=0.3)
        res = self._solve(m)
        for i in range(m.n_students):
            if res.assignment[i] >= 0:
                assert i not in m.course_rejections[res.assignment[i]]

    def test_result_fields_populated(self):
        m = generate_market(seed=55)
        res = self._solve(m)
        assert res.status is not None
        assert not np.isnan(res.obj_value)
        assert res.solve_time > 0
        assert res.X_value.shape == (m.n_students, m.n_courses)

    # ── all objectives ────────────────────────────────────────────────────────

    def test_all_objectives_run(self):
        m = generate_market(seed=56)
        for obj in ("student", "course", "bilateral", "egalitarian"):
            res = self._solve(m, objective=obj)
            assert res.assignment.shape == (m.n_students,)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=57,
                            capacity_range=(10, 10), phd_required_fraction=0.0,
                            rejection_fraction=0.0)
        for obj in ("student", "course", "bilateral"):
            res = self._solve(m, objective=obj)
            assert np.all(res.assignment >= 0)

    def test_all_unmatched_when_blocked(self):
        m = generate_market(n_students=10, n_courses=4, seed=58,
                            phd_fraction=0.0, phd_required_fraction=1.0)
        for obj in ("student", "course", "bilateral"):
            res = self._solve(m, objective=obj)
            assert np.all(res.assignment == -1)

    # ── optimality ────────────────────────────────────────────────────────────

    def test_student_utility_at_least_da(self):
        m = generate_market(seed=59, phd_required_fraction=0.0, rejection_fraction=0.0)
        res = self._solve(m, objective="student")
        a_da = deferred_acceptance(m)

        mip_util = sum(m.student_scores[i, res.assignment[i]] for i in range(m.n_students) if res.assignment[i] >= 0)
        da_util  = sum(m.student_scores[i, a_da[i]]           for i in range(m.n_students) if a_da[i] >= 0)
        assert mip_util >= da_util - 1e-6

    def test_egalitarian_min_utility_geq_student(self):
        # With ample capacity both objectives fully match everyone, so cardinality
        # is equal and egalitarian's explicit min-maximisation must win or tie.
        m = generate_market(n_students=10, n_courses=5, seed=60,
                            capacity_range=(5, 5), phd_required_fraction=0.0,
                            rejection_fraction=0.0)
        res_egal    = self._solve(m, objective="egalitarian")
        res_student = self._solve(m, objective="student")

        assert np.all(res_egal.assignment >= 0)
        assert np.all(res_student.assignment >= 0)

        min_egal    = min(m.student_scores[i, res_egal.assignment[i]]    for i in range(m.n_students))
        min_student = min(m.student_scores[i, res_student.assignment[i]] for i in range(m.n_students))
        assert min_egal >= min_student - 1e-6

    # ── LP relaxation ─────────────────────────────────────────────────────────

    def test_lp_output_shape(self):
        m = generate_market(seed=61)
        res = self._solve(m, lp_relax=True)
        assert res.assignment.shape == (m.n_students,)

    def test_lp_capacity_not_exceeded(self):
        m = generate_market(seed=62)
        res = self._solve(m, lp_relax=True)
        for j in range(m.n_courses):
            assert (res.assignment == j).sum() <= m.capacities[j]

    def test_lp_obj_geq_mip(self):
        m = generate_market(seed=63, phd_required_fraction=0.0, rejection_fraction=0.0)
        res_mip = self._solve(m, objective="student")
        res_lp  = self._solve(m, objective="student", lp_relax=True)
        assert res_lp.obj_value >= res_mip.obj_value - 1e-6

    # ── error handling ────────────────────────────────────────────────────────

    def test_invalid_objective_raises(self):
        m = generate_market(seed=64)
        with pytest.raises(ValueError, match="Unknown objective"):
            solve_mip(m, objective="invalid", solver=_SOLVER)

    def test_invalid_solver_raises(self):
        m = generate_market(seed=65)
        with pytest.raises(ValueError, match="Unknown solver"):
            solve_mip(m, objective="student", solver="BLAH")
