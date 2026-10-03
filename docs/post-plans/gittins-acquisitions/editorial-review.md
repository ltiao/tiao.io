# Editorial review

The 36 selected beats form the basis of the draft. The editorial pass adds navigation and primary-source attribution, preserves the full proof and tail expansion in linked appendices, and retains the motivating policy trees. The main text has about 3,400 whitespace-delimited words including TeX and captions, plus appendices.

## Scope and mathematical checks

- Exact result: finite independent boxes, integrable scores, positive converted costs, exact revelation, adaptive stopping, and expected terminal best score minus total cost. The incumbent supplies an outside option.
- Classical discounted cumulative-reward Gittins is distinguished from the Pandora reservation index.
- EI is derived through a one-evaluation horizon; unequal costs are subtracted, not divided.
- Multi-step continuation is chosen after the next observation. Primed incumbent and candidate set are defined before the compact recursion.
- Posterior-updated PBGI anticipates stopping/continuation under independence but does not plan cross-candidate information updates. Its stopping comparison is now explicit.
- Variational expression retains the expected half-squared positive improvement and linear cost term within the argmin. Finite second moment stated.
- Gaussian UCB form is a small-cost asymptotic with candidate-dependent beta, not a confidence or regret theorem. Bounded scores and noisy/partial observations are distinguished from the initial model.
- Hard-cap feasibility is distinguished from an expected-budget relaxation. No claim that an arbitrary cost price enforces the cap.

## Primary sources

- [Weitzman (1979)](https://scholar.harvard.edu/files/weitzman/files/optimalsearchbestalternative.pdf): reservation-value search rule.
- [Xie et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.20062): Pandora's Box Gittins Index for BO. Sections 3.2–3.3 for index and posterior construction; Appendix B.3 for small-cost limit. No novelty claim for this connection.
- [Kleinberg, Waggoner, and Weyl (2016)](https://arxiv.org/abs/1603.07682): capped-value proof approach.
- [Atsidakou et al. (2022)](https://arxiv.org/abs/2205.13114): Definition 3 / equation (2), linear-quadratic loss. Sign reversal from minimization to maximization differs from the displayed loss only by a threshold-independent term.
- [Jiang et al. (2020)](https://arxiv.org/abs/2006.15779): multi-step trees with posterior updates.

## Rendering checks

Production build: `HUGO_BUILDDRAFTS=true pnpm run build`. Draft status stays true. Standard warnings concern deprecated content in other posts.

Static production preview checked at 1440×900, 390×900, and 320×900. At each width: document width equals viewport; all 41 displays have clientWidth equal to scrollWidth; zero KaTeX errors. Accessibility snapshots expose MathML, figure descriptions, and linked appendix headings. The lazy-loaded EI plot loads when scrolled into view. Browser errors were empty. Screenshots and build logs are temporary in /private/tmp.

The EI plot's editable source and SVG are in the page bundle. It uses tau as threshold, rho as index, and the full conditional EI expression on the y axis. Shading is explicitly tied to the 0.1 cost line. Its white SVG background preserves the requested light appearance; title and description are included for standalone viewing. Regeneration uses latex and dvisvgm with LIBGS=/opt/homebrew/lib/libgs.dylib, followed by adding SVG accessibility metadata and the white background.

The policy-tree style receipt documents the blog webfonts' Georgia fallback in this browser. No global typography settings were changed. The user's Hugo development server remains running; temporary validation sessions are closed afterward.

## Equation typography revision, 2026-09-07

Restored compact desktop identities and aligned proof chains after user feedback on awkward multiline formatting. Ten long displays use the equation-layout shortcode to choose mathematically equivalent wide/narrow forms. Narrow screens define L_x before its argmin, and Q_1 before V_1; wide screens retain the explicit full expressions. CSS is scoped to this post, with the inactive version display:none so it is absent from the accessibility tree. The modest mobile type adjustment applies only to display math.

Fresh production build passes. At 1440, 820, 390, and 320px there are 41 visible display equations, zero KaTeX errors, zero per-display overflows, and no page overflow. Screenshots and accessibility snapshots checked. This supersedes the earlier centered multiline typesetting; the mathematics and source attributions are unchanged.
