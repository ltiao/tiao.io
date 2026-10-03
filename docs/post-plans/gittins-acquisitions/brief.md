# Acquisition functions from a cost-aware search problem

## Settled direction

Start with a concrete cost-aware AutoML / hyperparameter-tuning problem. Motivate Pandora's box and the Gittins reservation index from that problem, showing how the policy chooses evaluations and stops. Relate EI, multi-step EI, and Gaussian UCB to this foundation. PI is excluded at the user’s request.

The concluding insight is about two approximations to sequential decision-making: finite lookahead truncates future decisions, while the Pandora approximation removes dependence between candidate values during planning. Multi-step EI extends the temporal horizon; a reservation-index policy solves the independent-candidate search problem across its full stopping horizon.

Audience: readers familiar with basic Bayesian optimization; the post stands alone. Understanding comes first, with implementation as a consequence rather than an unvalidated performance promise.

Likelihood-free UCB, BORE/LFBO, and elicitation losses belong in a follow-up post.

## Existing notation and presentation decisions

Use tau for a variable improvement threshold, rho for a reservation index, y* for the incumbent, and h for the standard-normal EI helper. Write positive parts explicitly as max expressions. Use $\triangleq$ for explicit “defined as” relations; use ordinary equality for identities and derived equations. Explain what is fixed and what is solved for. Existing EI-versus-threshold figure is a candidate asset, not a settled opening figure.

## Claims requiring verification

- Separate exact independent-box optimality from the correlated-GP approximation.
- Derive EI via a one-step continuation value and distinguish this from expected improvement per cost.
- Present Gaussian UCB as a small-cost asymptotic, with parameter conventions explicit.
- Distinguish cost-penalized adaptive stopping from a hard finite budget.

## Agreed narrative and scope

Target 3,000–4,000 words plus a proof appendix. Standalone for readers familiar with basic BO.

1. Open with a tuning team that has 100 GPU-hours and an incumbent model: should it spend the whole budget? Separate feasibility from whether further evaluation is worthwhile.
2. Begin with a finite shortlist, fixed training protocols, exact scores revealed on completion, and known heterogeneous evaluation costs. Independent uncertain scores give the initial model. Explain conversion between score utility and compute costs.
3. Work through a two-configuration example that shows why evaluation order matters. Derive the break-even reservation index and stopping boundary. Explain the capped-value proof idea in the main text; put the full argument in an appendix.
4. Derive EI as a one-step approximation and Gaussian UCB as a small-cost asymptotic. Distinguish the one-step planning approximation from the small-cost asymptotic; omit PI. Keep UCB's main derivation in the text and tail-expansion details in the appendix.
5. Return to the budget. Distinguish a hard cap from an expected-budget constraint. Explain the Lagrangian connection to a cost multiplier and its limits; selecting a multiplier does not generally make the index policy optimal under a hard budget.
6. Introduce dependence using two nearby learning rates and a different model family. Explain how one evaluation reveals information about another. Contrast posterior-updated PBGI with multi-step EI, which accounts for this information within its planning horizon.
7. Conclude with the spatial-versus-temporal approximation distinction. The index plans through time too: spatial independence names what it simplifies, not the only dimension in which it operates. Discuss observation noise and early termination as extensions, without implying the initial theorem covers them.

Illustrations: the tuning shortlist, the EI-versus-threshold figure, and the correlated-candidate example. Reuse existing TikZ assets where suitable; check labels and stopping regions against the final text.

Likelihood-free methods remain a separate follow-up. Additional acquisitions are outside the core scope.

## Workflow status

Beat-by-beat drafting completed; the user authorized the subsequent editorial pass. Article: content/posts/gittins-acquisition-functions/index.md. Beats 1–36 saved; beats 14–16 revised with explicit mathematical notation; beat 36 is A (closing paragraph on future planning). PI removed from scope. Editorial pass completed: headings and transitions added, full proof and tail expansion moved to linked appendices, primary-source attribution added, and the EI-threshold plot integrated. Title and summary remain editable metadata. No publication or deployment is authorized.

Audience brings basic BO, training configurations, validation scores, and evaluation costs. Assume no decision theory. Include the reservation index as the argmin over tau of the expected half-squared positive improvement plus lambda c(x) tau; group the complete objective under the argmin. Explain this representation after grounding the reservation value, and attribute it appropriately.

## Grounding ledger

- Prerequisites: training configurations, validation scores, evaluation costs, basic BO.
- Beat 1 (opening A): the 100 GPU-hour running example; heterogeneous run costs; choosing the next evaluation and deciding whether to stop; distinction between available budget and worthwhile expenditure.
- Beat 2 (cost–accuracy trade-off): willingness to spend compute for accuracy; conversion at 0.1 accuracy percentage points per GPU-hour; completed-search value as best accuracy minus cumulative cost in score units.
- Beat 3 (unknown scores as boxes): fixed scores under a fixed training protocol; probabilities express uncertainty before evaluation; opening a box reveals its score at a compute cost; retain the best score including the starting model.
- Beat 4 (policies): a contingent plan choosing whether to stop or open another box after each outcome; expected net value averages final best score minus converted cumulative costs over possible outcomes.
- Beat 5 (independence): revealing a score leaves beliefs about unopened boxes unchanged; a better available model still changes whether evaluating another is worthwhile; dependence between nearby configurations will be restored later.
- Beat 6 (last evaluation): twenty-hour run has equal chances of scores 79 and 85; at incumbent 80 its expected gain is 2.5 points versus converted cost 2; at incumbent 84 its expected gain is 0.5; a break-even score lies between. Expected improvement is grounded by calculation but not yet named.
- Beat 7 (reservation value): the example breaks even at 81%; rho(x) is the score in hand making the last evaluation indifferent to stopping; it depends on candidate score beliefs and evaluation cost, not the current best score.
- Beat 8 (break-even equation): y is the unknown candidate score, tau a hypothetical available score; EI_tau(x) is expected positive improvement; c(x) is GPU-hour cost, lambda converts it to accuracy points; EI_rho(x)(x) = lambda c(x), illustrated at rho=81.
- Beat 9 (two rankings): cheap adjustment has independent equiprobable scores 80/82, cost 1 GPU-hour (0.1 points), reservation value 81.8; larger model has reservation value 81 and greater EI at incumbent 80. Ranking disagreement established; it does not isolate temporal myopia from heterogeneous-cost handling.
- Beat 10 (cheap-first policy): score 82 leads to stopping with net value 81.9; score 80 leads to the larger run with conditional expected net value 80.4; overall expected net value 81.15.
- Beat 11 (larger-first comparison): success gives net value 83; failure followed by adjustment gives conditional expected net value 78.9; total 80.95 versus 81.15 cheap-first. Good early results avoid different amounts of later cost.
- Beat 12 (scope of the example): both policies fit the hard budget; cost-adjusted one-step gains also favor the adjustment (0.9 versus 0.5), so the example does not establish general index optimality or isolate lookahead.
- Beat 13 (index policy): open the highest-reservation-value box while its index exceeds the score in hand; stop otherwise. Independence permits sorting once, with results determining stopping depth.
- Beats 14–16 revised together: use n=1,...,N for boxes, rho_n=rho(x_n), c_n=c(x_n); introduce z_n and d_n at capping, I_n at adaptive selection, then S, M, Z at the bound. E[I_n d_n]=E[I_n]E[d_n]=lambda c_n E[I_n] makes independence explicit. Winning box is n_star. Numerical capping example retained.
- Writing preference: prose-only proof became hard to follow; use named quantities and displayed mathematical steps with concise explanations. No decision-theory familiarity still assumed.
- Beat 17: any opened box with positive excess becomes the winner and forces immediate stopping; all other opened boxes have zero excess.
- Beat 18 (numerical attainment): adjustment is box 1 and larger model box 2; E[Z]=0.5(81.8)+0.25(81)+0.25(80)=81.15 matches cheap-first and proves optimality for the running example.
- Beat 19 (general attainment): positive-excess winner caps at the maximum Z; otherwise all opened excesses vanish and stopping excludes better unopened capped scores. The policy attains E[Z], establishing independent-box cost-penalized optimality.
- Beat 20 (variational form): rho(x) minimizes expected half-squared positive improvement plus lambda c(x) tau; derivative of L_x is lambda c(x)-EI_tau(x), giving the same break-even root.
- Beat 21 (one-step EI): y* denotes current best; one remaining evaluation has net gain EI_y*(x)-lambda c(x); maximize positive gain or stop. Equal costs give the ordinary EI ranking. This truncates future decisions, unlike full independent-box index search.
- Beat 22: R remaining configurations; V_1 is maximal positive one-step net gain; Q_2 adds expected V_1 after updating the best score and removing the evaluated candidate. Second decision follows observation. Currently independent-box notation; empty-set continuation convention supplied in beat 23.
- Beat 23: V_2 maximizes Q_2 against stopping; no candidates means zero continuation. Deeper horizons repeat recursion (dynamic programming). Independent-box index attains full-horizon value without enumerating future branches.
- Beat 24: nearby learning rates illustrate dependence; D stores observations, D prime adds (x,y); continuation must condition on updated beliefs as well as the new best score.
- Beat 25: rho_D(x) solves posterior expected improvement equals cost; update the correlated model after observations and recompute indices. The selection rule assumes independence during planning and does not anticipate future cross-candidate belief updates.
- Beat 26: both approaches plan future evaluations; multi-step EI truncates lookahead, while index solves the full independent-candidate planning problem. Correlated extension omits anticipated belief updates across candidates.
- Terminology steering: do not overemphasize temporal versus spatial; prefer limited lookahead and independent-candidate planning, while keeping non-myopic/future-planning content prominent.
- Beat 27: cheaper evaluations raise the break-even threshold; Gaussian upper-tail motivation; mu predicted mean, sigma positive posterior standard deviation of fixed score, a standardized index bonus in rho=mu+a sigma. Motivate before algebra.
- Beat 28: W standard normal (distinct from benchmark Z); y=mu+sigma W, rho=mu+a sigma; expected positive excess beyond a equals lambda c/sigma; phi/Phi defined and Gaussian integral evaluated.
- Beat 29: epsilon=lambda c/sigma tends to zero; Gaussian tail expansion cancels leading terms in EI; logarithmic balance yields a asymptotic to sqrt(2 log(sigma/(lambda c))).
- Beat 30: restore x in the Gaussian small-cost bonus; equal mean and uncertainty favor the cheaper configuration; lowering lambda increases the bonus and raising c decreases it.
- Beat 31: UCB form with beta(x)=2 log(sigma(x)/(lambda c(x))); candidate-specific coefficient even at equal costs; small-cost approximation alone supplies no confidence guarantee or regret bound.
- Beat 32: solve the exact standardized Gaussian break-even equation numerically for positive costs; rho can fall below mu; comparison with incumbent still determines evaluation versus stopping.
- Beat 33: cost-penalized expected objective differs from a 100 GPU-hour cap on every outcome path; lambda does not enforce the cap; ranking affordable candidates is not covered by the independent-box proof.
- Beat 34: C denotes total evaluation cost; relaxing E[C]<=B with price lambda yields E[M-lambda C]+lambda B; fixed-price optimization recovers the penalized objective, while calibrating the price remains separate.
- Beat 35: synthesize three roles of EI: one-step gain at incumbent, break-even equation for full-horizon independent-candidate search, and Gaussian small-cost UCB form.
- Beat 36: closing paragraph connects contingent future evaluations and stopping to explicit multi-step lookahead and the full-horizon independent-candidate index policy.
- Editorial verification: production build with drafts included passes; all 41 display equations fit at 1440, 390, and 320px, without page overflow or KaTeX errors. Policy trees and EI plot render, and appendix links resolve. See editorial-review.md.
- Grounded in editorial pass: Weitzman/Pandora versus classical discounted Gittins; PBGI attribution to Xie et al.; linear-quadratic loss attribution; bounded-score and noisy-observation caveats.



- Motivating-example figures: two diagram-design policy trees embedded beside the corresponding prose; all outcomes and costs shown. See diagram-notes.md for style decisions and validation. Full-post equation overflow was resolved in the editorial pass.

- Equation typography (2026-09-07): the user rejected the centered multiline layouts. Keep compact identities on wide screens and aligned derivation steps where meaningful. Ten long equations use equivalent narrow-screen versions through equation-layout; the narrow reservation-value form defines L_x then minimizes it, while desktop retains the explicit full objective. Never split braces into disconnected centered fragments merely to pass a width check.
