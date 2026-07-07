# Pólya-Gamma series wrap-up — design

**Date:** 2026-07-07
**Status:** approved (user, this session), amended to include Part II refinement
**Scope:** finish the three-part "A Primer on Pólya-gamma Random Variables" series

## Goal

Complete draft Parts I and III and refine published Part II so the series
reads as one arc: identities (I) → augmented model + Gibbs (II) → mean-field
VI in the same augmented model, from which the Jaakkola-Jordan (JJ) local
variational bound falls out (III). Part III is restructured **VI-first**: the
equivalence is presented as a discovery, not a verification. Both drafts stay
`draft: true`; promotion is a separate checklist handed to the user.

## Current state

- **Part I** (`content/posts/polya-gamma-basic-relationships/`, draft):
  fragmentary prose; two figures miscaptioned "Hyperbolic cosine function"
  (actually the grid and Monte-Carlo plots); citation block copy-pasted from
  the knowledge-gradient post; a footnote and a reference bullet literally
  labeled "Test"; trailing `y ∈ {−1,1}` likelihood fragment duplicating
  Part II with a conflicting label convention. All figures exist.
- **Part II** (`content/posts/polya-gamma-bayesian-logistic-regression/`,
  published): complete. Minor defects: "distribition" typo; `gassian_sample`
  and `np.expand_dim` in displayed code; `\prod` index errors
  (`\prod_{n=1}^n`, `\prod_{i=1}^n` with `n`-indexed factors); no forward
  hand-off to Part III.
- **Part III** (`content/posts/polya-gamma-sigmoid-local-variational-lower-bound/`,
  draft): the real math (JJ bound; tilted-PG variational family; per-datapoint
  ELBO term reproducing the bound; induced Gaussian `q(β)`) is ~70% present in
  the first ~300 lines but nearly prose-free. Lines ~300–1024 are a verbatim
  paste of Part II (to be cut). `src/script.py` already implements the CAVI/EM
  loop and generated figures the draft never references: `xi_*`, `lambda_*`,
  `mu_*`, `class_prob_true_*`, `class_prob_pred_*`, `samples_*` (PG(1,c)
  samples with the mean function overlaid), plus the referenced `softplus_*`
  and `sigmoid_*` bound-family figures.

## Notation and conventions (all parts)

- Prior in **precision convention** matching Part II: `p(β) = N(m, S⁻¹)`.
  Part III currently writes `N(m, S)`; harmonize.
- `λ(ξ) := tanh(ξ/2)/(2ξ) = E_{PG(1,ξ)}[ω]`. Footnote in Part III: JJ and
  Bishop define `λ_JJ(ξ) = tanh(ξ/2)/(4ξ) = λ(ξ)/2`; our exponents carry
  `λ/2` where theirs carry `λ_JJ`.
- Labels `y ∈ {0,1}` everywhere; `κ_n = y_n − ½`. The `{−1,1}` fragment in
  Part I is cut.
- `Bern(y | σ(ψ))`, not `Bernoulli(y | ψ)`.
- Citation keys: Part II keeps `tiao2021polyagamma`; Part I gets
  `tiao2021polyagammabasic`; Part III gets `tiao2021polyagammalocal`.

## Part III design (main work)

Narrative order (VI-first):

1. **Recap** of the PG-augmented model from Part II, tightened (joint
   likelihood, marginalization property as a callout).
2. **Mean-field VI**: family `q(β) ∏_n q(ω_n; ξ_n)` with
   `q(ω_n; ξ_n) = PG(ω_n | 1, ξ_n)` — an exponential tilting of the prior
   (`PG(1,ξ) = cosh(ξ/2) exp(−ξ²ω/2) PG(1,0)`). ELBO; per-datapoint
   `H(y, ξ, β) = E_q[log p(y|ω,β)] − KL[q(ω;ξ) ‖ p(ω)]` computed in closed
   form (math already in draft; add prose). Closed-form
   `KL[PG(1,ξ) ‖ PG(1,0)] = log cosh(ξ/2) − (ξ²/2)λ(ξ)`.
3. **Punchline**: `exp H(y,ξ,β) = e^{yψ} ℓ(−ψ, ξ)` where
   `ℓ(ψ,ξ) = σ(ξ) exp{(ψ−ξ)/2 − (λ(ξ)/2)(ψ²−ξ²)}` — the JJ bound, emerging
   uninvited. Supporting points:
   - `λ(ξ) = E_{PG(1,ξ)}[ω]`: the mysterious JJ λ-function is the PG mean
     function. Illustrated by the existing `samples_paper_1500x927.png`.
   - **New identity** (not in draft): for all ψ, ξ:
     `σ(ψ)/ℓ(ψ,ξ) = exp( KL[PG(1,ξ) ‖ PG(1,ψ)] )`, with
     `KL[PG(1,ξ) ‖ PG(1,ψ)] = log cosh(ξ/2) − log cosh(ψ/2) + ((ψ²−ξ²)/2)λ(ξ)`.
     The bound's multiplicative slack is exactly the exponentiated KL from the
     variational PG factor to the true PG conditional posterior
     `p(ω|y,ψ) = PG(1,ψ)`; tightness at `ξ = ±ψ` is now a statement about a
     KL vanishing, not a calculus accident. Verify numerically (assert over a
     (ψ, ξ) grid, tolerance ~1e-12) before publishing.
   - λ-convention footnote (see conventions).
4. **CAVI**: optimal `q*(ω_n) ∝ p(ω_n) exp E_{q(β)}[log p(y_n|ω_n,β)]`
   stays in the family: `q*(ω_n) = PG(1, ξ_n)` with
   `ξ_n² = E_{q(β)}[ψ_n²] = φ_nᵀ(Σ + μμᵀ)φ_n`. Gaussian update
   `Σ⁻¹ = S + ΦᵀΛΦ`, `μ = Σ(Sm + Φᵀκ)`, `Λ = diag(λ(ξ_n))`. Note this is
   verbatim JJ's EM algorithm (their E-step ↔ our `q(β)` update, their
   M-step ↔ our `ξ` update). Cite Durante & Rigon (2019) for the formal
   equivalence statement — **verify the URL via web search before inserting**
   (repo rule: never fabricate URLs).
5. **Classical derivation** (existing section, reframed): softplus upper
   bound → sigmoid lower bound via convex duality, "how JJ did it in the
   mid-90s, no Pólya-Gamma required" (PG arrived with Polson et al. 2013).
   Uses existing `softplus_*` and `sigmoid_*` figures.
6. **Implementation**: CAVI loop from `src/script.py`, cleaned, presented in
   Part II's implementation style. Existing figures: `class_prob_true`,
   `xi` (per-datapoint ξ over iterations), `lambda`, `mu` (variational-mean
   trajectory PairGrid), `class_prob_pred`.
7. **New — Gibbs vs CAVI comparison** on the same seed-8888 dataset:
   - `figures/class_prob_compare_*`: predictive class-probability overlay —
     Gibbs posterior draws vs variational predictive vs true
     `p(x)/(p(x)+q(x))`.
   - `figures/beta_posterior_compare_*`: Gibbs `β` sample cloud with `q(β)`
     Gaussian ellipses overlaid (the classic VI-underestimates-variance
     picture, or honest surprise if it doesn't here).
   - New `src/compare.py`; PG sampling via the modern `polyagamma` package
     (`pypolyagamma==1.2.3` won't build on Python 3.13); fallback:
     hand-rolled Devroye sampler. Match existing figure style (seaborn
     `paper`/`ticks`/`muted`, `crest` for iteration hues, Times; `usetex` if
     LaTeX present, else mathtext).
8. **Closing**: what generalizes — `PG(b,c)` for `b ≠ 1` (binomial/negative
   binomial), Bouchard's softmax bound, GP classification (Wenzel et al.
   2019), stochastic/natural-gradient VI. Links & further reading
   (+ Durante & Rigon 2019). Own citation block (`tiao2021polyagammalocal` —
   current block wrongly says Part II).

Cuts and front matter:

- Cut the pasted Part II content (current lines ~300–1024).
- `summary`: "We …" register (repo rule), e.g. "We show that the classical
  Jaakkola-Jordan bound on the logistic sigmoid is mean-field variational
  inference in the Pólya-Gamma augmented model in disguise…" (exact wording
  at implementation).
- `categories: [technical]` (currently empty).
- Tags reordered, most-distinctive first (repo rule):
  `[Pólya-Gamma, Variational Inference, Bayesian Statistics, Probabilistic Models, Machine Learning]`.
- Keep `date: 2021-05-11`; bump `lastmod`. Keep alias. `draft: true` stays.

## Part I design

- Stitch prose through: sigmoid ↔ cosh identity; PG definition callout
  (self-contained); Laplace-transform identity; the integral representation
  of σ; Gaussian-kernel form; Monte-Carlo verification.
- **Reframe the "joint density" language.** `σ(u)` is not integrable in `u`
  and `p(u|ω) = ½exp(u/2 − u²ω/2)` is not normalized; present the result as
  an *integral representation* of the sigmoid as a scaled Gaussian-kernel
  mixture under `PG(1,0)`, which is what the grid/MC figures actually show.
  The proper probabilistic version (with `y` the random variable) is
  Part II's job — say so in the hand-off.
- Fix captions: `grid_*` → conditional Gaussian kernel over the (u, ω) grid;
  `monte_carlo_*` → MC estimate of the mixture vs σ(u). Keep other captions.
- Cut the `y ∈ {−1,1}` tail (lines ~203–246); replace with a short hand-off
  paragraph to Part II.
- Replace citation block (currently the knowledge-gradient post's) with key
  `tiao2021polyagammabasic`; fix the "Test" footnote (make it a proper
  derivation footnote) and the "Test" reference bullet (proper further-reading
  list mirroring Part II's).
- Front matter: `summary` in "We …" register; `categories: [technical]`;
  tags reordered as in Part III (minus Variational Inference);
  keep `date`; bump `lastmod`; `draft: true` stays. Series note kept but the
  "Draft — work in progress" phrasing may drop once content is complete (the
  front-matter flag is the source of truth).

## Part II refinement (published — light touch, no restructuring)

- Typo/defect pass: "distribition"; `gassian_sample` → `gaussian_sample`
  (all occurrences, prose and code); `np.expand_dim` → `np.expand_dims`;
  `\prod_{n=1}^n` → `\prod_{n=1}^N`; `\prod_{i=1}^n` with `n`-indexed factors
  → `\prod_{n=1}^N`.
- Add a short closing "What's next" transition setting up Part III: Gibbs is
  exact but sequential; next we swap sampling for optimization in the same
  augmented model and watch a classical bound fall out. Plain-text mention
  only — **no relref to Part III while it is a draft** (published→draft
  relref breaks the build).
- Tags reordered to match the series (`Pólya-Gamma` first).
- Series note stays "in preparation" until promotion.
- Summary, structure, figures, code otherwise untouched.

## Verification

- `hugo` and `hugo --buildDrafts` both build clean.
- `src/compare.py` runs deterministically (seed 8888) and writes both new
  figures; the KL-slack identity numeric check passes.
- Grep gates: no "Test" placeholders; no Part II title/key in Part III's cite
  block; no `relref` from a published page to a draft.
- Math: every equation in Parts I and III re-derived line-by-line during
  editing (several already checked this session: tilting identity, H
  computation, Gaussian-kernel normalization, KL-slack identity).

## Promotion checklist (deliverable; NOT executed in this work)

1. Flip `draft: false` on Parts I and III; bump `lastmod`.
2. Replace Part II's series note with relref links to I and III.
3. Optionally set `featured: true` on Part III.

## Out of scope

- Restructuring Part II; new hero images; executing promotion; multiclass /
  Bouchard softmax content beyond a closing mention; touching the unrelated
  untracked `content/posts/fed-prediction-market-edge/`.

## Risks

- `polyagamma` package availability on Python 3.13 → fallback Devroye
  sampler (well-documented in Polson/Windle papers).
- `usetex` requires a local LaTeX; fallback mathtext with closest style
  match. Only the two new figures are affected; existing figures untouched.
- External URLs (Durante & Rigon) must be verified via web search before
  insertion.
