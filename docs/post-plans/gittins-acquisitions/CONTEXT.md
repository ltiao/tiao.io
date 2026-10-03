# Cost-aware acquisition functions

Terminology for the planned blog post and its running tuning example.

## Language

**Improvement threshold (tau)**: The value against which a candidate's potential improvement is measured. It is a variable in the EI curve, not the reservation index.

**Reservation index (rho)**: The threshold at which expected improvement equals the evaluation cost expressed in objective units. In independent-box search, these indices determine evaluation order and the stopping boundary.

**Incumbent (y*)**: The best available observed objective value in the exact-observation maximization model.

**Temporal approximation**: A planning approximation that limits the number of future evaluations considered.

**Spatial independence approximation**: A planning approximation that treats remaining candidate values as independent. A GP-based implementation can still update its correlated posterior between decisions.

**Likelihood-free continuation**: A separate follow-up post connecting acquisition functionals to supervised training losses.

**Hard budget**: A cap on total evaluation expenditure that must hold on every realized execution.

**Expected-budget constraint**: A bound on evaluation expenditure averaged over the uncertainty in outcomes and any policy randomization. It permits individual executions to exceed that bound.

**Cost-penalized objective**: Expected terminal solution utility minus expected cumulative evaluation cost multiplied by a conversion factor.

**Cost multiplier (lambda)**: The factor converting evaluation costs into objective utility units. Its use in a budget relaxation does not by itself guarantee feasibility or optimality under a hard cap.
