# VSE TA Assignment Problem
**ECON622 final project — Kitty Wu**

## Overview

The VSE currently allocates TAs each term using a deferred acceptance procedure. This project uses that setting as a backdrop to build a proof-of-concept framework that implements the current approach (deferred-acceptance) alongside two alternative matching methods, simulates preference data, and compares performance across several criteria: stability, fairness, welfare, and computational efficiency.

I'm approaching this is an exploration of various ways to solve the same matching problem, and not as a theoretical contribution on greating a 'better' algorithm. The goal is to ask: *given the same preference data, how differently do these algorithms actually perform, and what does each one optimise for?* Given the context is close to our livlihoods (for better or for worse), I thought this would be a fun application of a more specialized optimization problem for the project :D

Note that the project was completed with the help fo Claude. 


## The Matching Problem

Let $I$ = students, $J$ = courses, $c_j$ = TA slots per course. The assignment is represented as a binary matrix $X$ where $x_{ij} \in \{0,1\}$ and $x_{ij} = 1$ if student $i$ is assigned to course $j$.

**Constraints applied across all algorithms:**
- Each student is assigned at most once: $\sum_j x_{ij} \leq 1$
- Course capacity: $\sum_i x_{ij} \leq c_j$
- PhD requirement: if $d_j = 1$, only PhD students ($\text{phd}_i = 1$) may be assigned
- Rejection lists: some courses reject specific students outright always 

## Algorithms

### 1. Deferred Acceptance / Gale–Shapley (DA)

Students propose to their most-preferred course that hasn't yet rejected them. Courses tentatively accept up to capacity, holding the best applicants by their own ranking and rejecting the rest. Rejected students propose to their next choice. This repeats until no proposals remain.

At the VSE, the many-to-one problem (multiple slots per course) is converted to one-to-one by replicating each course $j$ into $c_j$ identical allocation blocks.

**Key properties:** student-optimal, strategy-proof for students, always produces a stable matching (no blocking pairs).

### 2. Serial Dictatorship (SD)

Students are ordered by a (random) priority ranking. The first student picks their top available choice, the second picks from what remains, and so on until all slots are filled. Courses have no voice in the outcome.

**Key properties:** strategy-proof, Pareto efficient among students. Not stable - can produce blocking pairs, and outcomes are sensitive to the priority order.

### 3. Mixed Integer Programming (MIP)

Reframes allocation as a constrained optimisation problem. Through the lens of how would a social planner assign TAs, assuming preferences are reported truthfully.

Implemented via CVXPY with Gurobi (HiGHS as a free fallback).

Four objective variants are included:

| Objective | Formulation |
|---|---|
| Student utilitarian | $\max \sum_{i,j} u_{ij}\, x_{ij}$ |
| Course utilitarian | $\max \sum_{i,j} v_{ji}\, x_{ij}$ |
| Bilateral | $\max \sum_{i,j} (u_{ij} + v_{ji})\, x_{ij}$ |
| Egalitarian (min-max) | $\max\; t \quad$ s.t. $\; t \leq \sum_j u_{ij} x_{ij} + M(1 - \sum_j x_{ij})\;\; \forall i$ |

The egalitarian formulation uses a constant to exclude unmatched students from binding the floor utility $t$ that is always positive. It also runs a phase-1 problem first to fix the maximum possible matches before optimising the floor.

An **LP relaxation** variant is also included: relax $x_{ij} \in \{0,1\}$ to $x_{ij} \in [0,1]$ (so no long an integer constraint), solve in polynomial time, then rounded. Computationally, this could be more efficient than enforcing the strict integer constraint for the decision variable.

## Data Generating Process

Markets are simulated using a latent skill vector model. Each student $i$ has a skill vector $\theta_i \in \mathbb{R}^k$ and each course $j$ has a requirement vector $\phi_j \in \mathbb{R}^k$, both drawn i.i.d. from $\mathcal{N}(0, I_k)$. Preference scores are:

$$u_{ij} = \theta_i \cdot \phi_j + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$$

Course scores $v_{ji}$ are generated symmetrically with independent noise draws. Rankings are derived from scores.

### The `Market` object

`generate_market()` returns a `Market` dataclass containing everything needed to run any algorithm:

```python
@dataclass
class Market:
    n_students: int
    n_courses: int
    capacities: np.ndarray          # (n_courses,)  — slots per course
    student_scores: np.ndarray      # (n_students, n_courses)  — u_ij
    course_scores: np.ndarray       # (n_courses, n_students)  — v_ji
    student_rankings: np.ndarray    # (n_students, n_courses)  — rank of course j for student i (0 = top)
    course_rankings: np.ndarray     # (n_courses, n_students)  — rank of student i for course j (0 = top)
    student_prefs: list             # student_prefs[i] = ordered list of course indices
    course_prefs: list              # course_prefs[j] = ordered list of student indices
    theta: np.ndarray               # (n_students, k)  — latent student skill vectors
    phi: np.ndarray                 # (n_courses, k)   — latent course requirement vectors
    phd_students: np.ndarray        # (n_students,) bool
    phd_required: np.ndarray        # (n_courses,) bool
    course_rejections: list         # course_rejections[j] = set of rejected student indices
    student_names: list
    course_codes: list
```

### Generating a market

```python
from DGP import generate_market

m = generate_market(
    n_students=20,
    n_courses=8,
    k=3,               # latent skill dimension
    sigma=0.5,         # idiosyncratic noise
    seed=42,
)
```

Key parameters: `phd_fraction`, `phd_required_fraction`, `rejection_fraction`, `capacity_range`, `sparse_prefs` / `sparse_length` (students only rank a subset of courses), `popularity_weight` (adds a shared course-level popularity term). Fixed course structures can be passed via `fixed_capacities`, `fixed_course_codes`, `fixed_phd_required`.

## Files

```
.
├── code (spaghetti)/
│   ├── DGP.py              data generating process; exports generate_market() and Market
│   ├── da.py               student-proposing deferred acceptance
│   ├── sd.py               serial dictatorship
│   ├── mip.py              MIP / LP relaxation via CVXPY; exports solve_mip()
│   ├── test_matching.py    pytest suite (DGP, DA stability, SD correctness, MIP)
│   └── comparisons and visualizations.ipynb   comparison notebook — all algorithms, metrics, and visualisations
├── supporting documents (garlic bread)/
│   ├── presentation.pdf        project presentation slides
│   ├── project_proposal.pdf    original project proposal
│   └── TA POSTING 2025-2026 Winter Term 2.pdf   real VSE TA postings used in the final simulation
├── notes.md                project notes and thoughts
└── pyproject.toml          dependencies (numpy, cvxpy, highs, seaborn, jupyter, pytest)
```

## Quick Overview

For a quick summary of results, `code (spaghetti)/comparisons and visualizations.ipynb` is where I run all three algorithms and visualize the comparisons. Most of the discussion lives here. See also `notes.md` for additional notes (and after thoughts).

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

To run the test suite:

```bash
cd "code (spaghetti)"
pytest test_matching.py
```

**Solver note:** the MIP uses Gurobi by default, which requires a license (but it's free to request). Otherwise use `solver="HIGHS"`, HiGHS is a free solver bundled with the install and supports all objectives. The comparison notebook uses HiGHS automatically for the winter 2026 simulation.

```python
from mip import solve_mip
res = solve_mip(market, objective="student", solver="HIGHS")
```
