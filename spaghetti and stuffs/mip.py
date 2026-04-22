"""
MIP / LP-relaxation based TA matching using CVXPY.

Solves the assignment as a (mixed) integer program or its LP relaxation.

Objective variants
------------------
'student'     Maximise total student utility   sum_ij u_ij x_ij
'course'      Maximise total course utility    sum_ij v_ij x_ij
'bilateral'   Maximise sum of both sides       sum_ij (u_ij + v_ij) x_ij
'egalitarian' Maximise min matched-student utility  (big-M formulation)

LP relaxation
-------------
Set lp_relax=True to relax x_ij in {0,1} to x_ij in [0,1].
The returned assignment is obtained by greedy rounding (argmax per student, threshold 0.5).

Solvers
-------
Gurobi ('GUROBI') is the default. HiGHS ('HIGHS') is a free alternative
bundled with the cvxpy[highs] install and supports both LP and MIP.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
from DGP import Market


_SOLVER_MAP: dict[str, str] = {
    "GUROBI":  cp.GUROBI, # for MIP
    "HIGHS":   cp.HIGHS, # as backup for GUROBI
}


@dataclass
class MIPResult:
    assignment: np.ndarray   # shape (n_students,), -1 if unmatched
    obj_value: float         # optimal (or relaxed) objective value
    status: str              # solver status string
    X_value: np.ndarray      # shape (n_students, n_courses), raw variable values
    solve_time: float        # wall-clock seconds


def _build_allowed(market: Market) -> np.ndarray:
    """
    Binary mask: allowed[i, j] = 1 iff student i may be assigned to course j.

    This is the single place that translates Market eligibility fields into the
    MIP. If the DGP grows new eligibility rules, update only here.
    """
    n, m = market.n_students, market.n_courses
    allowed = np.ones((n, m), dtype=float)
    for j in range(m):
        if market.phd_required[j]:
            allowed[~market.phd_students, j] = 0.0
        for i in market.course_rejections[j]:
            allowed[i, j] = 0.0
    return allowed


def solve_mip(
    market: Market,
    objective: str = "student",
    lp_relax: bool = False,
    solver: str = "GUROBI",
    **solver_kwargs,
) -> MIPResult:
    """
    Solve the TA matching problem as a MIP or LP relaxation.

    Parameters
    ----------
    market : Market
    objective : {'student', 'course', 'bilateral', 'egalitarian'}
    lp_relax : bool
        If True, relax binary constraints to [0, 1].
    solver : str
        CVXPY solver name. 'GUROBI' (default) or 'HIGHS' (free, bundled).
    **solver_kwargs
        Forwarded to problem.solve().

    Returns
    -------
    MIPResult
    """
    n, m = market.n_students, market.n_courses
    allowed = _build_allowed(market)

    # define decision variables 
    if lp_relax:
        X = cp.Variable((n, m), nonneg=True)
        constraints = [X <= allowed]
    else:
        X = cp.Variable((n, m), boolean=True)
        constraints = [X <= allowed]

    constraints += [
        cp.sum(X, axis=1) <= 1,                  # each student matched at most once
        cp.sum(X, axis=0) <= market.capacities,  # course capacity
    ]

    # optimization objectives
    U = market.student_scores       # (n, m)
    V = market.course_scores.T      # (n, m)

    # avoid errors
    cvx_solver = _SOLVER_MAP.get(solver.upper())
    if cvx_solver is None:
        raise ValueError(
            f"Unknown solver '{solver}'. Choose from: {list(_SOLVER_MAP)}"
        )

    t0 = time.perf_counter()

    if objective == "student":
        cvx_obj = cp.Maximize(cp.sum(cp.multiply(U, X)))

    elif objective == "course":
        cvx_obj = cp.Maximize(cp.sum(cp.multiply(V, X)))

    elif objective == "bilateral":
        cvx_obj = cp.Maximize(cp.sum(cp.multiply(U + V, X)))

    elif objective == "egalitarian":
        if lp_relax:
            X1 = cp.Variable((n, m), nonneg=True)
        else:
            X1 = cp.Variable((n, m), boolean=True)
        c1 = [X1 <= allowed]
        # first find maximum matching cardinality
        c1 += [cp.sum(X1, axis=1) <= 1, cp.sum(X1, axis=0) <= market.capacities] # constraint 1, same as above for capacity
        p1 = cp.Problem(cp.Maximize(cp.sum(X1)), c1)  # type: ignore[arg-type]
        p1.solve(solver=cvx_solver, **solver_kwargs)
        if p1.value is None:
            raise RuntimeError(f"Phase 1 (cardinality) infeasible — solver status: {p1.status}")
        max_matched = int(round(float(np.asarray(p1.value).item())))

        # then avoid the issue that unmatched students have trivial utility, so minimizing their utility is not helpful. 
        # disregard unmatched student when min/max optimizing
        M = float(np.abs(U).max() + 1.0) # some constant shifter for unmatched students
        t = cp.Variable()  # min utility of matched students
        
        matched_i = cp.sum(X, axis=1)  # (n,)
        student_utility = cp.sum(cp.multiply(U, X), axis=1)  # (n,)
        constraints += [
            cp.sum(X) >= max_matched,                            # require max matches
            t <= student_utility + M * (1 - matched_i),     # only matched students constrain t
        ]
        cvx_obj = cp.Maximize(t)  # min/max objective

    else:
        raise ValueError(
            f"Unknown objective '{objective}'. "
            "Choose from: 'student', 'course', 'bilateral', 'egalitarian'."
        )

    # solved
    problem = cp.Problem(cvx_obj, constraints)  # type: ignore[arg-type]
    problem.solve(solver=cvx_solver, **solver_kwargs)
    elapsed = time.perf_counter() - t0

    # didnt solve
    if X.value is None:
        raise RuntimeError(
            f"Solver '{solver}' returned status '{problem.status}'. "
            "No solution was found."
        )

    # return result
    X_val = X.value
    assignment = np.full(n, -1, dtype=int)
    for i in range(n):
        best = int(np.argmax(X_val[i]))  # best course for student i
        if X_val[i, best] > 0.5:  # check they are indeed assigned
            assignment[i] = best

    return MIPResult(
        assignment=assignment,
        obj_value=float(np.asarray(problem.value).item()) if problem.value is not None else float("nan"),
        status=problem.status,
        X_value=X_val,
        solve_time=elapsed,
    )


def summarize_mip_result(
    result: MIPResult,
    market: Market,
    objective: str,
    lp_relax: bool = False,
) -> None:
    """Print a human-readable summary of a MIPResult."""
    mode = "LP-relax" if lp_relax else "MIP"
    print(f"[{mode} | objective={objective}]")
    print(f"  Status     : {result.status}")
    print(f"  Obj value  : {result.obj_value:.4f}")
    print(f"  Solve time : {result.solve_time:.4f}s")

    a = result.assignment
    matched = a >= 0
    print(f"  Matched    : {matched.sum()} / {market.n_students}")

    ranks = [
        market.student_rankings[i, a[i]]
        for i in range(market.n_students) if a[i] >= 0
    ]
    if ranks:
        print(
            f"  Student rank of match — "
            f"mean: {np.mean(ranks):.2f}, "
            f"min: {min(ranks)}, max: {max(ranks)}  (0 = top choice)"
        )

    print()
    for i in range(market.n_students):
        name = market.student_names[i]
        if a[i] >= 0:
            course = market.course_codes[a[i]]
            rank = market.student_rankings[i, a[i]]
            print(f"  {name} -> {course}  (rank {rank})")
        else:
            print(f"  {name} -> unmatched")

# demo block
if __name__ == "__main__":
    from DGP import generate_market

    m = generate_market(seed=42)

    for obj in ("student", "course", "bilateral", "egalitarian"):
        print()
        try:
            res = solve_mip(m, objective=obj, solver="GUROBI")
            summarize_mip_result(res, m, obj)
        except Exception as e:
            print(f"[MIP | {obj}] FAILED: {e}")

    print("\n--- LP relaxation (student objective) ---")
    try:
        res_lp = solve_mip(m, objective="student", lp_relax=True, solver="GUROBI")
        summarize_mip_result(res_lp, m, "student", lp_relax=True)
    except Exception as e:
        print(f"[LP | student] FAILED: {e}")
