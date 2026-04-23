import numpy as np
import pytest
from DGP import generate_market, Market, _GREEK
from da import deferred_acceptance
from sd import serial_dictatorship
from mip import solve_mip, MIPResult

_SOLVER = "GUROBI"


# helpers
# check stability (very similar to notebook version)
def is_stable(assignment: np.ndarray, market: Market) -> bool:
    """Return True iff no blocking pair exists (respects PhD constraint and sparse prefs)."""
    pref_set = [set(prefs) for prefs in market.student_prefs]
    for i in range(market.n_students):
        for j in range(market.n_courses):
            if j not in pref_set[i]:
                continue
            if market.phd_required[j] and not market.phd_students[i]:
                continue
            if i in market.course_rejections[j]:
                continue
            if assignment[i] == j:
                continue

            if assignment[i] >= 0:
                if market.student_rankings[i, j] >= market.student_rankings[i, assignment[i]]:
                    continue
            
            assigned_to_j = np.where(assignment == j)[0]
            if len(assigned_to_j) < market.capacities[j]:
                return False  
            if any(market.course_rankings[j, i] < market.course_rankings[j, k]
                   for k in assigned_to_j):
                return False
    return True


# DGP
class TestGenerateMarket:
    # all score/ranking/capacity/phd arrays have the expected shape
    def test_shapes(self):
        m = generate_market(n_students=15, n_courses=6, seed=0)
        assert m.student_scores.shape == (15, 6)
        assert m.course_scores.shape == (6, 15)
        assert m.student_rankings.shape == (15, 6)
        assert m.course_rankings.shape == (6, 15)
        assert m.capacities.shape == (6,)
        assert m.phd_students.shape == (15,)
        assert m.phd_required.shape == (6,)

    # student_names and course_codes lists match the requested counts
    def test_names_and_codes_length(self):
        m = generate_market(n_students=10, n_courses=5, seed=1)
        assert len(m.student_names) == 10
        assert len(m.course_codes) == 5

    # no duplicate names when n_students ≤ 24 (size of the Greek name pool)
    def test_student_names_unique_within_24(self):
        m = generate_market(n_students=24, seed=2)
        assert len(set(m.student_names)) == 24

    # no duplicate ECON course codes generated
    def test_course_codes_unique(self):
        m = generate_market(n_courses=10, seed=3)
        assert len(set(m.course_codes)) == 10

    # every generated course code starts with "ECON"
    def test_course_codes_all_econ(self):
        m = generate_market(seed=4)
        assert all(code.startswith("ECON") for code in m.course_codes)

    # student names use only Greek letter bases from _GREEK
    def test_student_names_are_greek(self):
        m = generate_market(n_students=10, seed=5)
        bases = [name.rstrip("0123456789") for name in m.student_names]
        assert all(base in _GREEK for base in bases)

    # each row of student_rankings and course_rankings is a valid permutation of indices
    def test_rankings_are_permutations(self):
        m = generate_market(seed=6)
        for i in range(m.n_students):
            assert sorted(m.student_rankings[i]) == list(range(m.n_courses))
        for j in range(m.n_courses):
            assert sorted(m.course_rankings[j]) == list(range(m.n_students))

    # student_prefs lists courses in ascending rank order (top choice first)
    def test_prefs_consistent_with_rankings(self):
        m = generate_market(seed=7)
        for i in range(m.n_students):
            ranks = [m.student_rankings[i, j] for j in m.student_prefs[i]]
            assert ranks == sorted(ranks)

    # same seed produces bit-identical scores, names, and codes
    def test_reproducibility(self):
        m1 = generate_market(seed=99)
        m2 = generate_market(seed=99)
        np.testing.assert_array_equal(m1.student_scores, m2.student_scores)
        assert m1.student_names == m2.student_names
        assert m1.course_codes == m2.course_codes

# for the actual postings, the DGP should respect fixed overrides
class TestDGPFixedOverrides:
    # fixed_capacities overrides random capacity draws
    def test_fixed_capacities(self):
        caps = np.array([1, 2, 3, 1, 2])
        m = generate_market(n_courses=5, fixed_capacities=caps, seed=90)
        np.testing.assert_array_equal(m.capacities, caps)

    # fixed_course_codes overrides randomly sampled codes
    def test_fixed_course_codes(self):
        codes = ["ECON301", "ECON401", "ECON501"]
        m = generate_market(n_courses=3, fixed_course_codes=codes, seed=91)
        assert m.course_codes == codes

    # fixed_phd_required overrides randomly drawn PhD flags
    def test_fixed_phd_required(self):
        flags = np.array([True, False, True, False])
        m = generate_market(n_courses=4, fixed_phd_required=flags, seed=92)
        np.testing.assert_array_equal(m.phd_required, flags)

# when #students > #slots market doesnt break
class TestOversubscribed:
    # when students > total slots, matched count never exceeds total capacity
    def test_match_count_bounded_by_capacity(self):
        m = generate_market(n_students=30, n_courses=5, seed=80,
                            capacity_range=(2, 3), phd_required_fraction=0.0,
                            rejection_fraction=0.0)
        assert m.n_students > m.capacities.sum(), "market must be oversubscribed"
        for a in (
            deferred_acceptance(m),
            serial_dictatorship(m, seed=0),
            solve_mip(m, objective="student", solver=_SOLVER).assignment,
        ):
            assert (a >= 0).sum() <= m.capacities.sum()

# DA
class TestDeferredAcceptance:
    # assignment has one entry per student
    def test_output_shape(self):
        m = generate_market(seed=10)
        a = deferred_acceptance(m)
        assert a.shape == (m.n_students,)

    # values are -1 (unmatched) or a valid course index
    def test_assignment_values_in_range(self):
        m = generate_market(seed=11)
        a = deferred_acceptance(m)
        assert np.all((a >= -1) & (a < m.n_courses))

    # no course is assigned more students than its capacity
    def test_capacity_not_exceeded(self):
        m = generate_market(seed=12)
        a = deferred_acceptance(m)
        for j in range(m.n_courses):
            assert (a == j).sum() <= m.capacities[j]

    # students are only matched to courses they submitted in their preference list
    def test_matched_course_in_pref_list(self):
        m = generate_market(seed=13)
        a = deferred_acceptance(m)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert a[i] in m.student_prefs[i]

    # non-PhD students are never placed in PhD-required courses
    def test_phd_constraint(self):
        m = generate_market(seed=14, phd_required_fraction=0.5)
        a = deferred_acceptance(m)
        for i in range(m.n_students):
            if a[i] >= 0 and m.phd_required[a[i]]:
                assert m.phd_students[i], (
                    f"{m.student_names[i]} is non-PhD but matched to "
                    f"PhD-required {m.course_codes[a[i]]}"
                )

    # no blocking pairs exist across 5 different seeds (DA is stable by construction)
    def test_stability(self):
        for seed in range(5):
            m = generate_market(seed=seed)
            a = deferred_acceptance(m)
            assert is_stable(a, m), f"Unstable matching for seed={seed}"

    # stability still holds when students submit only a short preference list
    def test_stability_sparse_prefs(self):
        m = generate_market(seed=20, sparse_prefs=True)
        a = deferred_acceptance(m)
        assert is_stable(a, m)

    # everyone is matched when total capacity far exceeds the number of students
    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=30,
                            capacity_range=(10, 10), phd_required_fraction=0.0)
        a = deferred_acceptance(m)
        assert np.all(a >= 0), "Everyone should match when total slots >> n_students"

    # all students remain unmatched when every course requires PhD and no student has one
    def test_all_unmatched_when_all_phd_required_no_phd_students(self):
        m = _phd_blocked_market(seed=31)
        a = deferred_acceptance(m)
        assert np.all(a == -1), "Non-PhD students should not match PhD-required courses"
class TestDAStudentOptimality:
    # DA (student-optimal stable matching) gives students weakly more total utility
    # than MIP-course (course-optimal stable matching)
    def test_da_student_utility_geq_mip_course(self):
        for seed in range(5):
            m = generate_market(seed=seed, phd_required_fraction=0.0, rejection_fraction=0.0)
            a_da  = deferred_acceptance(m)
            a_mip = solve_mip(m, objective="course", solver=_SOLVER).assignment
            da_util  = sum(m.student_scores[i, a_da[i]]  for i in range(m.n_students) if a_da[i]  >= 0)
            mip_util = sum(m.student_scores[i, a_mip[i]] for i in range(m.n_students) if a_mip[i] >= 0)
            assert da_util >= mip_util - 1e-6, f"DA not student-optimal vs MIP-course for seed={seed}"

# SD
class TestSerialDictatorship:
    # assignment has one entry per student
    def test_output_shape(self):
        m = generate_market(seed=40)
        a = serial_dictatorship(m, seed=0)
        assert a.shape == (m.n_students,)

    # values are -1 (unmatched) or a valid course index
    def test_assignment_values_in_range(self):
        m = generate_market(seed=41)
        a = serial_dictatorship(m, seed=0)
        assert np.all((a >= -1) & (a < m.n_courses))

    # no course is assigned more students than its capacity
    def test_capacity_not_exceeded(self):
        m = generate_market(seed=42)
        a = serial_dictatorship(m, seed=0)
        for j in range(m.n_courses):
            assert (a == j).sum() <= m.capacities[j]

    # students are only matched to courses they submitted in their preference list
    def test_matched_course_in_pref_list(self):
        m = generate_market(seed=43)
        a = serial_dictatorship(m, seed=0)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert a[i] in m.student_prefs[i]

    # non-PhD students are never placed in PhD-required courses
    def test_phd_constraint(self):
        m = generate_market(seed=44, phd_required_fraction=0.5)
        a = serial_dictatorship(m, seed=0)
        for i in range(m.n_students):
            if a[i] >= 0 and m.phd_required[a[i]]:
                assert m.phd_students[i], (
                    f"{m.student_names[i]} is non-PhD but matched to "
                    f"PhD-required {m.course_codes[a[i]]}"
                )

    # the student who picks first always gets their top valid choice
    def test_first_student_gets_top_choice(self):
        # The first student in the order should always get their top valid choice
        m = generate_market(seed=45, phd_required_fraction=0.0)
        order = np.arange(m.n_students)
        a = serial_dictatorship(m, order=order)
        assert a[0] == m.student_prefs[0][0]

    # same fixed order produces identical assignment on repeated calls
    def test_fixed_order_is_deterministic(self):
        m = generate_market(seed=46)
        order = np.arange(m.n_students)
        a1 = serial_dictatorship(m, order=order)
        a2 = serial_dictatorship(m, order=order)
        np.testing.assert_array_equal(a1, a2)

    # everyone is matched when total capacity far exceeds the number of students
    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=47,
                            capacity_range=(10, 10), phd_required_fraction=0.0)
        a = serial_dictatorship(m, seed=0)
        assert np.all(a >= 0)

    # all students remain unmatched when every course requires PhD and no student has one
    def test_all_unmatched_when_all_phd_required_no_phd_students(self):
        m = _phd_blocked_market(seed=48)
        a = serial_dictatorship(m, seed=0)
        assert np.all(a == -1)

class TestSDOrderMatters:
    # different random seeds produce different assignments in a competitive market
    def test_different_seeds_differ(self):
        # oversubscribed so pick order has real consequences
        m = generate_market(n_students=20, n_courses=5, seed=100,
                            capacity_range=(2, 2), phd_required_fraction=0.0,
                            rejection_fraction=0.0)
        assignments = {seed: serial_dictatorship(m, seed=seed) for seed in range(10)}
        unique = {tuple(a.tolist()) for a in assignments.values()}
        assert len(unique) > 1, "SD should produce different assignments under different orderings"


# MIP
class TestMIP:
    def _solve(self, market, objective="student", lp_relax=False):
        return solve_mip(market, objective=objective, lp_relax=lp_relax, solver=_SOLVER)

    # general tests (same as above)
    # assignment has one entry per student
    def test_output_shape(self):
        m = generate_market(seed=50)
        res = self._solve(m)
        assert res.assignment.shape == (m.n_students,)

    # values are -1 (unmatched) or a valid course index
    def test_assignment_values_in_range(self):
        m = generate_market(seed=51)
        res = self._solve(m)
        assert np.all((res.assignment >= -1) & (res.assignment < m.n_courses))

    # no course is assigned more students than its capacity
    def test_capacity_not_exceeded(self):
        m = generate_market(seed=52)
        res = self._solve(m)
        for j in range(m.n_courses):
            assert (res.assignment == j).sum() <= m.capacities[j]

    # PhD constraint is respected across all four objective types
    def test_phd_constraint(self):
        m = generate_market(seed=53, phd_required_fraction=0.5)
        for obj in ("student", "course", "bilateral", "egalitarian"):
            res = self._solve(m, objective=obj)
            for i in range(m.n_students):
                if res.assignment[i] >= 0 and m.phd_required[res.assignment[i]]:
                    assert m.phd_students[i]

    # students are never assigned to a course that explicitly rejected them
    def test_rejection_list_respected(self):
        m = generate_market(seed=54, rejection_fraction=0.3)
        res = self._solve(m)
        for i in range(m.n_students):
            if res.assignment[i] >= 0:
                assert i not in m.course_rejections[res.assignment[i]]

    # MIPResult has status, obj_value, solve_time, and X_value populated
    def test_result_fields_populated(self):
        m = generate_market(seed=55)
        res = self._solve(m)
        assert res.status is not None
        assert not np.isnan(res.obj_value)
        assert res.solve_time > 0
        assert res.X_value.shape == (m.n_students, m.n_courses)

    # all four objectives complete without error and return a valid assignment
    def test_all_objectives_run(self):
        m = generate_market(seed=56)
        for obj in ("student", "course", "bilateral", "egalitarian"):
            res = self._solve(m, objective=obj)
            assert res.assignment.shape == (m.n_students,)

    # edge cases
    # all students matched when capacity greatly exceeds students and no constraints block
    def test_all_matched_when_ample_slots(self):
        m = generate_market(n_students=10, n_courses=5, seed=57,
                            capacity_range=(10, 10), phd_required_fraction=0.0,
                            rejection_fraction=0.0)
        for obj in ("student", "course", "bilateral"):
            res = self._solve(m, objective=obj)
            assert np.all(res.assignment >= 0)

    # all students unmatched when PhD constraint blocks every possible assignment
    def test_all_unmatched_when_blocked(self):
        m = _phd_blocked_market(seed=58)
        for obj in ("student", "course", "bilateral"):
            res = self._solve(m, objective=obj)
            assert np.all(res.assignment == -1)

    # check optimality
    # MIP-student total welfare ≥ DA total welfare (global optimum dominates decentralised)
    def test_student_utility_at_least_da(self):
        m = generate_market(seed=59, phd_required_fraction=0.0, rejection_fraction=0.0)
        res = self._solve(m, objective="student")
        a_da = deferred_acceptance(m)

        mip_util = sum(m.student_scores[i, res.assignment[i]] for i in range(m.n_students) if res.assignment[i] >= 0)
        da_util  = sum(m.student_scores[i, a_da[i]]           for i in range(m.n_students) if a_da[i] >= 0)
        assert mip_util >= da_util - 1e-6

    # egalitarian min-utility ≥ student-objective min-utility when everyone is fully matched
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

    # integer assumption relaxed (LP relaxation), check the same general constraints
    # LP relaxation returns an assignment of the right length
    def test_lp_output_shape(self):
        m = generate_market(seed=61)
        res = self._solve(m, lp_relax=True)
        assert res.assignment.shape == (m.n_students,)

    # LP assignment (rounded) still respects course capacities
    def test_lp_capacity_not_exceeded(self):
        m = generate_market(seed=62)
        res = self._solve(m, lp_relax=True)
        for j in range(m.n_courses):
            assert (res.assignment == j).sum() <= m.capacities[j]

    # LP objective value ≥ MIP objective value (relaxation is an upper bound on the IP)
    def test_lp_obj_geq_mip(self):
        m = generate_market(seed=63, phd_required_fraction=0.0, rejection_fraction=0.0)
        res_mip = self._solve(m, objective="student")
        res_lp  = self._solve(m, objective="student", lp_relax=True)
        assert res_lp.obj_value >= res_mip.obj_value - 1e-6

    # errors are handled
    # unrecognised objective string raises ValueError
    def test_invalid_objective_raises(self):
        m = generate_market(seed=64)
        with pytest.raises(ValueError, match="Unknown objective"):
            solve_mip(m, objective="invalid", solver=_SOLVER)

    # unrecognised solver string raises ValueError
    def test_invalid_solver_raises(self):
        m = generate_market(seed=65)
        with pytest.raises(ValueError, match="Unknown solver"):
            solve_mip(m, objective="student", solver="BLAH")


class TestMIPCourseOptimality:
    # MIP-course total course utility ≥ DA total course utility (MIP optimises over
    # a larger feasible set that includes the DA matching)
    def test_mip_course_utility_geq_da(self):
        for seed in range(5):
            m = generate_market(seed=seed, phd_required_fraction=0.0, rejection_fraction=0.0)
            a_da  = deferred_acceptance(m)
            a_mip = solve_mip(m, objective="course", solver=_SOLVER).assignment
            da_util  = sum(m.course_scores[a_da[i],  i] for i in range(m.n_students) if a_da[i]  >= 0)
            mip_util = sum(m.course_scores[a_mip[i], i] for i in range(m.n_students) if a_mip[i] >= 0)
            assert mip_util >= da_util - 1e-6, f"MIP-course utility < DA for seed={seed}"

# testing some specific weird edge cases
# 6 students, 3 courses (cap 2 each), all students rank courses identically:
# course 0 > course 1 > course 2. Courses prefer lower-index students.
# Total slots == n_students so everyone should be matched.

def _phd_blocked_market(n_students: int = 10, n_courses: int = 4, seed: int = 0) -> Market:
    """All courses require PhD, no student is PhD — so no valid match exists.
    Built directly to bypass the DGP safeguard that promotes students to PhD."""
    rng = np.random.default_rng(seed)
    n_s, n_c = n_students, n_courses
    student_scores   = rng.standard_normal((n_s, n_c))
    course_scores    = rng.standard_normal((n_c, n_s))
    student_rankings = np.argsort(np.argsort(-student_scores, axis=1), axis=1)
    course_rankings  = np.argsort(np.argsort(-course_scores,  axis=1), axis=1)
    caps = rng.integers(1, 3, size=n_c)
    return Market(
        n_students=n_s, n_courses=n_c,
        capacities=caps,
        student_scores=student_scores, course_scores=course_scores,
        student_rankings=student_rankings, course_rankings=course_rankings,
        student_prefs=[list(np.argsort(-student_scores[i])) for i in range(n_s)],
        course_prefs=[list(np.argsort(-course_scores[j]))   for j in range(n_c)],
        theta=np.zeros((n_s, 1)), phi=np.zeros((n_c, 1)),
        phd_students=np.zeros(n_s, dtype=bool),   # no PhD students
        phd_required=np.ones(n_c,  dtype=bool),   # all courses require PhD
        course_rejections=[set() for _ in range(n_c)],
        student_names=[f"S{i}" for i in range(n_s)],
        course_codes=[f"C{j}" for j in range(n_c)],
    )


def _identical_pref_market() -> Market:
    n_s, n_c = 6, 3
    student_scores   = np.tile([3.0, 2.0, 1.0], (n_s, 1))
    course_scores    = np.tile(np.arange(n_s, 0, -1, dtype=float), (n_c, 1))
    student_rankings = np.tile([0, 1, 2], (n_s, 1))
    course_rankings  = np.tile(np.arange(n_s), (n_c, 1))
    return Market(
        n_students=n_s, n_courses=n_c,
        capacities=np.array([2, 2, 2]),
        student_scores=student_scores, course_scores=course_scores,
        student_rankings=student_rankings, course_rankings=course_rankings,
        student_prefs=[[0, 1, 2]] * n_s,
        course_prefs=[list(range(n_s))] * n_c,
        theta=np.zeros((n_s, 1)), phi=np.zeros((n_c, 1)),
        phd_students=np.zeros(n_s, dtype=bool),
        phd_required=np.zeros(n_c, dtype=bool),
        course_rejections=[set() for _ in range(n_c)],
        student_names=[f"S{i}" for i in range(n_s)],
        course_codes=[f"C{j}" for j in range(n_c)],
    )

class TestIdenticalPreferences:
    # everyone is matched when total capacity equals n_students
    def test_all_matched_da(self):
        m = _identical_pref_market()
        a = deferred_acceptance(m)
        assert np.all(a >= 0)

    # DA is stable even when all students compete for the same top course
    def test_da_stable(self):
        m = _identical_pref_market()
        a = deferred_acceptance(m)
        assert is_stable(a, m)

    # top course fills to capacity before anyone is pushed to the next choice
    def test_da_fills_top_course_first(self):
        m = _identical_pref_market()
        a = deferred_acceptance(m)
        assert (a == 0).sum() == m.capacities[0]
        assert (a == 1).sum() == m.capacities[1]
        assert (a == 2).sum() == m.capacities[2]

    # the first two students in pick order always land on their top choice
    def test_sd_first_students_get_top_choice(self):
        m = _identical_pref_market()
        order = np.arange(m.n_students)
        a = serial_dictatorship(m, order=order)
        assert a[0] == 0 and a[1] == 0

    # MIP still matches everyone and respects capacity with identical preferences
    def test_mip_all_matched(self):
        m = _identical_pref_market()
        res = solve_mip(m, objective="student", solver=_SOLVER)
        assert np.all(res.assignment >= 0)
        for j in range(m.n_courses):
            assert (res.assignment == j).sum() <= m.capacities[j]


# pure noise (no skill signal) 
# preferences are driven entirely by random noise
def _noise_market(seed: int = 0) -> Market:
    n_s, n_c = 10, 4
    rng = np.random.default_rng(seed)
    student_scores = rng.standard_normal((n_s, n_c))   # pure noise, no skill signal
    course_scores  = rng.standard_normal((n_c, n_s))
    student_rankings = np.argsort(np.argsort(-student_scores, axis=1), axis=1)
    course_rankings  = np.argsort(np.argsort(-course_scores,  axis=1), axis=1)
    return Market(
        n_students=n_s, n_courses=n_c,
        capacities=np.array([3, 3, 2, 2]),             # total slots = n_students
        student_scores=student_scores, course_scores=course_scores,
        student_rankings=student_rankings, course_rankings=course_rankings,
        student_prefs=[list(np.argsort(-student_scores[i])) for i in range(n_s)],
        course_prefs=[list(np.argsort(-course_scores[j]))   for j in range(n_c)],
        theta=np.zeros((n_s, 1)), phi=np.zeros((n_c, 1)),
        phd_students=np.zeros(n_s, dtype=bool),
        phd_required=np.zeros(n_c, dtype=bool),
        course_rejections=[set() for _ in range(n_c)],
        student_names=[f"S{i}" for i in range(n_s)],
        course_codes=[f"C{j}" for j in range(n_c)],
    )

class TestPureNoise:
    # DA is stable even when preferences are random noise
    def test_da_stable(self):
        for seed in range(5):
            m = _noise_market(seed)
            a = deferred_acceptance(m)
            assert is_stable(a, m), f"Unstable DA matching for noise seed={seed}"

    # no algorithm overfills a course with random preferences
    def test_capacity_respected_all_algorithms(self):
        m = _noise_market()
        a_da = deferred_acceptance(m)
        a_sd = serial_dictatorship(m, seed=0)
        a_mip = solve_mip(m, objective="student", solver=_SOLVER).assignment
        for a in (a_da, a_sd, a_mip):
            for j in range(m.n_courses):
                assert (a == j).sum() <= m.capacities[j]

    # all algorithms return valid indices (-1 or a course index) with random preferences
    def test_valid_assignment_values(self):
        m = _noise_market()
        for a in (
            deferred_acceptance(m),
            serial_dictatorship(m, seed=0),
            solve_mip(m, objective="student", solver=_SOLVER).assignment,
        ):
            assert np.all((a >= -1) & (a < m.n_courses))

    # MIP runs without error across all objectives with random preferences
    def test_mip_all_objectives_run(self):
        m = _noise_market()
        for obj in ("student", "course", "bilateral", "egalitarian"):
            res = solve_mip(m, objective=obj, solver=_SOLVER)
            assert res.assignment.shape == (m.n_students,)


# check rejected students are actually rejected
class TestRejectionLists:
    # DA never assigns a student to a course that rejected them
    def test_da_respects_rejections(self):
        m = generate_market(seed=70, rejection_fraction=0.3)
        a = deferred_acceptance(m)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert i not in m.course_rejections[a[i]], (
                    f"{m.student_names[i]} assigned to {m.course_codes[a[i]]} which rejected them"
                )

    # SD never assigns a student to a course that rejected them
    def test_sd_respects_rejections(self):
        m = generate_market(seed=71, rejection_fraction=0.3)
        a = serial_dictatorship(m, seed=0)
        for i in range(m.n_students):
            if a[i] >= 0:
                assert i not in m.course_rejections[a[i]], (
                    f"{m.student_names[i]} assigned to {m.course_codes[a[i]]} which rejected them"
                )

    # DA is still stable when rejection lists are present
    def test_da_stable_with_rejections(self):
        m = generate_market(seed=72, rejection_fraction=0.3)
        a = deferred_acceptance(m)
        assert is_stable(a, m)

# degenerate market
def _zero_cap_market() -> Market:
    n_s, n_c = 8, 3
    rng = np.random.default_rng(0)
    student_scores  = rng.standard_normal((n_s, n_c))
    course_scores   = rng.standard_normal((n_c, n_s))
    student_rankings = np.argsort(np.argsort(-student_scores, axis=1), axis=1)
    course_rankings  = np.argsort(np.argsort(-course_scores,  axis=1), axis=1)
    return Market(
        n_students=n_s, n_courses=n_c,
        capacities=np.zeros(n_c, dtype=int),
        student_scores=student_scores, course_scores=course_scores,
        student_rankings=student_rankings, course_rankings=course_rankings,
        student_prefs=[list(np.argsort(-student_scores[i])) for i in range(n_s)],
        course_prefs=[list(np.argsort(-course_scores[j]))   for j in range(n_c)],
        theta=np.zeros((n_s, 1)), phi=np.zeros((n_c, 1)),
        phd_students=np.zeros(n_s, dtype=bool),
        phd_required=np.zeros(n_c, dtype=bool),
        course_rejections=[set() for _ in range(n_c)],
        student_names=[f"S{i}" for i in range(n_s)],
        course_codes=[f"C{j}" for j in range(n_c)],
    )


class TestZeroCapacity:
    # all algorithms leave everyone unmatched when every course has zero slots
    def test_all_unmatched_da(self):
        assert np.all(deferred_acceptance(_zero_cap_market()) == -1)

    def test_all_unmatched_sd(self):
        assert np.all(serial_dictatorship(_zero_cap_market(), seed=0) == -1)

    def test_all_unmatched_mip(self):
        for obj in ("student", "course", "bilateral"):
            res = solve_mip(_zero_cap_market(), objective=obj, solver=_SOLVER)
            assert np.all(res.assignment == -1)


