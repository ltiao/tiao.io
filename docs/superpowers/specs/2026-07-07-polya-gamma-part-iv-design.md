# Pólya-Gamma series Part IV — Beyond Binary — design

**Date:** 2026-07-07
**Status:** scope approved by user (softmax-centric + brief tour; demo includes one-vs-each)
**Working mode:** PI/editor orchestrates; Opus subagents write all code.
**Bundle:** `content/posts/polya-gamma-beyond-binary/` (draft: true until review)

## Thesis

Part III ends with "ask which augmentation your bound is secretly the ELBO
of." Part IV answers: sometimes none — and you can locate exactly which part
of the slack is the obstruction. Binary-like likelihoods extend for free;
the K ≥ 3 softmax does not, and every known escape pays a stated price.

## Content plan

1. **Warm-up: exact extensions.** The general Laplace-transform identity
   (Part II callout) covers `(e^ψ)^a/(1+e^ψ)^b`:
   binomial `Binom(m, σ(ψ))` → `PG(m, ·)`, `κ = y − m/2`;
   negative binomial (r failures, success prob σ(ψ)) → `PG(y+r, ·)`,
   `κ = (y−r)/2`. Same Gibbs/CAVI machinery, `E[ω] = b·λ(c)`.
2. **The softmax problem.** Bouchard's bound is a double bound:
   (i) α-bound: `LSE(ψ) ≤ α + Σₖ ς(ψₖ−α)` from `Σeˣ ≤ ∏(1+eˣ)`;
   (ii) JJ-quadratic each softplus with per-class ξₖ.
   **Slack decomposition (the centerpiece; numerically verified):**
   `[α + Σₖ g(ψₖ−α, ξₖ)] − LSE(ψ) = Σₖ KL[PG(1,ξₖ) ‖ PG(1, ψₖ−α)] + S_α`
   where `S_α = −log P(exactly one of K independent Bernoullis fires)`,
   `pₖ = σ(ψₖ−α)`. The KL part is the familiar augmentation slack
   (vanishes at the right ξ); `S_α > 0` strictly for all α and all finite ψ
   — the price of treating classes as independent Bernoullis, and the
   precise reason this bound touches nothing and therefore cannot arise as
   a tight augmented ELBO the way JJ's does. Optimal
   `α* = (K/2 − 1 + Σₖ λ(ξₖ)ψₖ)/Σₖ λ(ξₖ)`.
3. **The escapes, each with its toll** (surveyed, one section each, no full
   derivations):
   - **Stick-breaking PG** (Linderman, Johnson & Adams, NeurIPS 2015,
     arXiv:1506.05843; stick-breaking idea due to Ren et al. 2011, Khan et
     al. 2012): multinomial → cascade of binomials `Binom(xₖ | Nₖ, σ(ψₖ))`,
     each exactly PG-augmentable. Toll: class-order dependence (a different
     model, symmetric no more).
   - **One-vs-each** (Titsias, NeurIPS 2016, arXiv:1609.07410):
     `softmax_y(ψ) ≥ ∏_{k≠y} σ(ψ_y − ψₖ)` — the SAME product inequality as
     Bouchard's step (i), applied to the reciprocal; slack
     `= −log P(at most one competitor Bernoulli fires)`,
     `pₖ = σ(ψₖ−ψ_y)` (numerically verified). Every factor is exactly
     PG-augmentable (Snell & Zemel 2020 build on this). Toll: it is a
     pseudo-likelihood; the posterior targeted is not the softmax model's.
     Note Titsias' general proposition `P(A) ≥ ∏ᵢ P(A|A∪Bᵢ)` and the
     merge-hierarchy of bounds.
   - **Logistic-softmax** (Galy-Fajou, Wenzel, Donner & Opper, UAI 2019,
     arXiv:1905.09670): change the link to `σ(fᵏ)/Σ_c σ(f^c)`; a
     Gamma → Poisson → PG chain restores exact conditional conjugacy.
     Toll: a different likelihood. Related nuance from their paper: Polson
     et al. had a softmax PG scheme usable for sampling whose ELBO is
     intractable — "augmentation exists" and "conjugate VI exists" are
     different claims.
4. **Demo (K=3, 1-D, seed 8888 house style):** three class-conditional
   Gaussians (N(−2,0.8²), N(0.5,1²), N(2.5,0.7²)), equal priors, N=150,
   degree-3 polynomial features per class, α=2 prior precision.
   Methods: Bouchard-VB (per-class Gaussians; monotonicity-checked
   updates), stick-breaking PG Gibbs (order 0,1,2), one-vs-each PG Gibbs
   (stacked 9-dim weight vector, per-pair PG draws). Figures:
   (1) per-class predictive probability vs analytic truth, 1×3 panels;
   (2) **slack decomposition** at the converged Bouchard solution: tilting
   (KL) component vs independence component per training point — the
   post's thesis as one picture (doubles as featured.png).
5. **Closing:** the series' moral, upgraded: a bound with per-datapoint
   parameters and a touching point is an augmentation's ELBO in disguise;
   a bound that touches nothing is telling you the model class itself was
   approximated. Forward pointers (price-of-conjugacy and linear-response
   posts remain specced from the wrap-up conversation).

## Mechanics

- Front matter mirrors Part III (tags with Pólya-Gamma first + Variational
  Inference; categories technical; summary in "We…" register;
  `date: 2026-07-07`; `draft: true`).
- Series notes in Parts I–III say "three-part"; they are published, so they
  cannot relref a draft Part IV. Update all three notes to four-part + link
  ONLY at Part IV promotion (checklist item).
- Cite key `tiao2026polyagammabeyond`.
- All equations obey the ~672px column discipline learned in Part III:
  `align*` only, split wide one-liners, no `\qquad`-glued multi-part
  displays.
- References verified against arXiv e-print LaTeX sources (per new house
  rule), not abstracts.

## Verification gates

- `verify_identities.py` (agent-written, PI-reviewed): softplus-side JJ
  identity, α-slack = exactly-one-Bernoulli form, full Bouchard
  decomposition, α* stationarity, OVE slack = at-most-one form,
  binomial/NB augmentation MC checks. All pass before any identity is
  stated in prose.
- Demo asserts: Bouchard expected-bound monotonicity per sweep; slack
  components nonnegative.
- Hugo prod + drafts builds; self-driven agent-browser sweep (KaTeX errors,
  display overflows > 8px, raw TeX leaks) before review handoff.
