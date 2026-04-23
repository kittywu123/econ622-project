"""
Student-proposing Deferred Acceptance (Gale-Shapley) for TA matching.

Follows the VSE approach: the many-to-one problem is converted to one-to-one by
expanding each course with c_j slots into c_j separate "allocation blocks". Each
block can accept exactly one student. All blocks of the same course share the same
preference ordering over students.

Student preference lists are expanded so that course j becomes blocks (j,0), (j,1), ...,
(j, c_j - 1) in sequence — students are indifferent between blocks of the same course.

PhD constraint: blocks belonging to a PhD-required course reject non-PhD students.
"""

import numpy as np
from DGP import Market


def _expand_to_blocks(market: Market):
    """
    Convert the market into a one-to-one representation using allocation blocks.

    Returns
    -------
    block_to_course : list of int
        block_to_course[b] = course index j that block b belongs to.
    student_block_prefs : list of lists
        student_block_prefs[i] = ordered list of block indices for student i.
    block_student_prefs : list of lists
        block_student_prefs[b] = ordered list of student indices (same as course's ranking).
    block_rank : np.ndarray, shape (n_blocks, n_students)
        block_rank[b, i] = rank of student i for block b (lower = more preferred).
    """
    # Map each course to its blocks
    # course_blocks[j] = [b_0, b_1, ..., b_{c_j - 1}]
    course_blocks = []
    block_to_course = []
    b = 0
    for j in range(market.n_courses):
        slots = market.capacities[j]
        course_blocks.append(list(range(b, b + slots)))
        for _ in range(slots):
            block_to_course.append(j)
        b += slots

    n_blocks = b

    # Expand student preference lists: replace each course j with its blocks in order
    student_block_prefs = []
    for i in range(market.n_students):
        block_list = []
        for j in market.student_prefs[i]:
            block_list.extend(course_blocks[j])
        student_block_prefs.append(block_list)

    # Each block inherits the course's preference list over students
    block_student_prefs = []
    block_rank = np.empty((n_blocks, market.n_students), dtype=int)
    for j in range(market.n_courses):
        for blk in course_blocks[j]:
            block_student_prefs.append(list(market.course_prefs[j]))
            block_rank[blk] = market.course_rankings[j]

    return block_to_course, student_block_prefs, block_student_prefs, block_rank, n_blocks


def deferred_acceptance(market: Market) -> np.ndarray:
    """
    Run student-proposing deferred acceptance on a market using the one-to-one
    block expansion

    Parameters
    ----------
    market : Market

    Returns
    -------
    assignment : np.ndarray, shape (n_students,)
        assignment[i] = j if student i is matched to course j, -1 if unmatched.
    """
    block_to_course, student_block_prefs, _, block_rank, n_blocks = _expand_to_blocks(market)

    # Track where each student is in their expanded proposal list
    next_proposal = np.zeros(market.n_students, dtype=int)

    # Tentative match for each block: one student index, or -1 if empty
    tentative = np.full(n_blocks, -1, dtype=int)

    # All students start free
    free = list(range(market.n_students))

    while free:
        next_free = []
        for i in free:
            prefs = student_block_prefs[i]

            # Student has exhausted their list — permanently unmatched
            if next_proposal[i] >= len(prefs):
                continue

            b = prefs[next_proposal[i]]
            next_proposal[i] += 1

            j = block_to_course[b]

            # PhD constraint: block's course requires PhD but student is not
            if market.phd_required[j] and not market.phd_students[i]:
                next_free.append(i)
                continue

            # Rejection list: course has explicitly excluded this student
            if i in market.course_rejections[j]:
                next_free.append(i)
                continue

            # Block is empty — student is tentatively accepted
            if tentative[b] == -1:
                tentative[b] = i

            # Block is occupied — course prefers the better-ranked student
            elif block_rank[b, i] < block_rank[b, tentative[b]]:
                # Current student is preferred: reject the incumbent
                next_free.append(tentative[b])
                tentative[b] = i

            else:
                # Incumbent is preferred: reject the proposing student
                next_free.append(i)

        free = next_free

    # Map blocks back to courses
    assignment = np.full(market.n_students, -1, dtype=int)
    for b, i in enumerate(tentative):
        if i >= 0:
            assignment[i] = block_to_course[b]

    return assignment


def summarize_assignment(assignment: np.ndarray, market: Market) -> None:
    """Print a summary of an assignment."""
    matched = assignment >= 0
    n_matched = matched.sum()
    print(f"Matched  : {n_matched} / {market.n_students} students")
    print(f"Unmatched: {(~matched).sum()} students")

    # Student rank of their matched course (0 = top choice)
    student_match_ranks = [
        market.student_rankings[i, assignment[i]]
        for i in range(market.n_students) if assignment[i] >= 0
    ]
    if student_match_ranks:
        print(f"Student rank of match — mean: {np.mean(student_match_ranks):.2f}, "
              f"min: {min(student_match_ranks)}, max: {max(student_match_ranks)} "
              f"(0 = top choice)")

    print("\nCourse fill rates:")
    for j in range(market.n_courses):
        assigned = (assignment == j).sum()
        phd_tag = " [PhD required]" if market.phd_required[j] else ""
        print(f"  {market.course_codes[j]}: {assigned} / {market.capacities[j]} slots{phd_tag}")

    unmatched = [market.student_names[i] for i in np.where(assignment == -1)[0]]
    print("\nUnmatched students:", unmatched)


if __name__ == "__main__":
    from DGP import generate_market

    m = generate_market(seed=42)
    assignment = deferred_acceptance(m)
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
