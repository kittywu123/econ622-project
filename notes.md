# notes

## some thoughts

The idea for this project came about when I was waiting for my TA assignment last semester. I remembered another student and I finding out we both preferred each other's assignments (so potentially a blocking pair…?) and how the current system could be improved.

Personally I find matching problems really satisfying (maybe because it's almost like putting together a puzzle and trying to best make the pieces fit together?) so I'm glad I took this opportunity to explore the topic a bit more and go through the exercise of implementing the algorithms myself. So after doing that, here are some quick thoughts:

DA sacrifices some efficiency to gain stability, but perhaps the trade-off is not very severe?

SD sacrifices both stability and some efficiency but is simpler and strategy-proof but very 'luck' dependent.

MIP-student is efficient but not stable and not strategy-proof. While the egalitarian objective should in theory be more 'fair', at least in the simulations I ran it didn't seem like MIP-student performed that much worse. So in any case if there is a 'social planner', a utilitarian one might work best?

Strategy-proofness was also something I haven't had time to look into as much but could be an interesting extension — maybe some simulations or ways to 'game the system'?

While working on the MIP part specifically, I was first introduced to the cvxpy library and Gurobi as the solver (by Jesse's suggestions). I'm still very new to it but it was really interesting to learn more about solving convex linear problems like this one using Python.

From my understanding, Gurobi is an industrial-use optimization solver. I got a free academic license to use it (as all students should have access to) and it does have some limitations regarding problem size. When I tried to apply it to the actual Winter 2026 postings, it seems like this was the constraint.

When I generated the data, I didn't limit utilities to be positive (so some students can have negative utilities from a course). While I think this makes sense, it would also mean that some students — by way of how DA and SD work — would prefer the outside option of not being assigned but were not given this option since the algorithms only work off rankings. I thought about restricting the range of utilities but ultimately decided to keep it as is and accept this as a modelling limitation.

If a MIP-like system were implemented in practice, maybe it could be something like an 'aptitude test' that represents the latent skill/trait vector, where each dimension could be something like math skill, coding skill, scheduling flexibility, GPA, etc. So instead of reporting preferences, students might be reporting (or simply getting prescribed) a score with respect to each requirement of the course. Of course, this also raises questions of how these scores would be solicited, honesty, and whether they are a truly accurate measure of 'utility' — similar to how the DGP currently has it.

There are likely many other oversimplifications and edge cases I didn't think of that could be improved upon moving forward, but as they stand all algorithms should work as intended.
