---
title: UCB in Disguise
subtitle: Likelihood-free Bayesian optimization beyond expected utilities, by quantile regression
summary: We derive a likelihood-free UCB acquisition function. UCB is the quantile of the predictive distribution in disguise, and quantiles come with their own scoring rule. We then show that the state-of-the-art conformal quantile method has been optimizing UCB in disguise all along.
authors:
- me
tags:
- Bayesian Optimization
- Quantile Regression
- Machine Learning
categories:
- technical
date: 2026-07-10
math: true
draft: false

image:
  placement: 1
  focal_point: Center
  preview_only: false
---

The upper confidence bound[^srinivas2010] is not the object it appears to be. Written as
$$
\alpha_{\text{UCB}}(\mathbf{x}) \triangleq \mu(\mathbf{x}) + \sqrt{\beta} \sigma(\mathbf{x}),
$$
it reads as a mean with an exploration bonus stapled on: two moments and a tuning knob. But for a Gaussian predictive $p(y \mid \mathbf{x}) = \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$, this quantity is exactly a quantile,
$$
\mu(\mathbf{x}) + \sqrt{\beta} \sigma(\mathbf{x}) = F^{-1}(q \mid \mathbf{x}),
\qquad
q = \Phi(\sqrt{\beta}),
$$
where $F^{-1}(\cdot \mid \mathbf{x})$ is the quantile function of the predictive distribution and $\Phi$ is the standard normal CDF. The identity is one line: the Gaussian quantile function is $\mu + \sigma \Phi^{-1}(q)$, so match $\Phi^{-1}(q) = \sqrt{\beta}$. Setting $\beta = 4$ says: score each candidate by the 97.7th percentile of what it might return. UCB is a quantile wearing a mean-plus-bonus costume.

## Acquisition functions as loss functions

The reason to care about rewriting UCB is a family of methods that replace the usual machinery of Bayesian optimization (fit a probabilistic surrogate, form its posterior predictive, integrate a utility against it) with a single supervised-learning step on the raw data. The opening move was ours. BORE[^tiao2021bore] thresholds the observed outcomes at a quantile of the $y$ values seen so far, labels each point by whether it beats the threshold, and trains a probabilistic classifier $\pi(\mathbf{x})$ on the resulting labels. The quantile-thresholding device itself goes back to TPE[^bergstra2011], which worked with a ratio of densities rather than a classifier. The optimal classifier output is a monotone transform of probability of improvement, so maximizing the classifier maximizes PI. No predictive distribution is ever built. The acquisition function is estimated directly, by minimizing a familiar loss on $(\mathbf{x}_i, y_i)$ pairs, which means any classifier you like (gradient-boosted trees, neural networks) can play the role the Gaussian process usually plays.

LFBO[^song2022general] turned this observation into a recipe, and the mechanism deserves a derivation rather than a citation.[^thesisexposition] Every differentiable, strictly convex function is the upper envelope of its tangent lines,
$$
f(v) = \max_{s} \left\{ v f'(s) - f^{\star}\left( f'(s) \right) \right\},
$$
where $f^{\star}$ is the convex conjugate. Apply this pointwise to any target function $\alpha$ inside an expectation over candidates, and loosen the pointwise maximization to a single function $S$:
$$
\mathbb{E}_{p(\mathbf{x})}\left[ f(\alpha(\mathbf{x})) \right]
\geq \max_{S : \mathcal{X} \to (0, \infty)}
\mathbb{E}_{p(\mathbf{x})}\left[ \alpha(\mathbf{x}) f'(S(\mathbf{x})) - f^{\star}\left( f'(S(\mathbf{x})) \right) \right],
$$
with equality exactly at $S^{\ast} = \alpha$. So far this is a variational device for eliciting any function you can evaluate under an expectation. The likelihood-free step is a single application of the tower rule. Take the target to be the expected utility
$$
\alpha_u(\mathbf{x}) \triangleq \mathbb{E}_{p(y \mid \mathbf{x})}\left[ u(y; \tau) \right]
$$
for a nonnegative utility $u$ and threshold $\tau$. Since $f'(S(\mathbf{x}))$ does not depend on $y$, the conditional expectation defining $\alpha_u$ merges into the outer one, $\mathbb{E}_{p(\mathbf{x})}[\alpha_u(\mathbf{x}) f'(S(\mathbf{x}))] = \mathbb{E}_{p(\mathbf{x}, y)}[u(y; \tau) f'(S(\mathbf{x}))]$, and the intractable predictive integral dissolves into an average over the raw pairs. Choosing the logistic $f(v) \triangleq v \log v - (v + 1) \log (v + 1)$ and reparameterizing $S = \pi / (1 - \pi)$ lands the objective on weighted binary cross-entropy,
$$
\max_{\pi}\
\mathbb{E}_{p(\mathbf{x}, y)}\left[ u(y; \tau) \log \pi(\mathbf{x}) + \log\left( 1 - \pi(\mathbf{x}) \right) \right],
$$
in which every observation participates as a negative with weight one and as a positive with weight $u(y_i; \tau)$, and the optimal classifier satisfies $\pi / (1 - \pi) = \alpha_u$ exactly. The indicator utility $u(y; \tau) \triangleq \mathbb{1}(y > \tau)$ gives PI and recovers BORE (the same population acquisition; the loss weighting differs); the hinge utility $u(y; \tau) \triangleq \max(y - \tau, 0)$ gives EI. A single loss trains a single model whose output is the acquisition function. And the logistic choice was arbitrary: any strictly convex $f$ elicits the same $\alpha_u$, a freedom that will matter shortly.

PI and EI are where the LFBO paper's tables end. UCB appears nowhere in this family, and the omission is structural rather than an oversight.

One more exhibit before the door-closing begins. The strongest recent entrant in this lineage, conformal quantile regression for HPO[^salinas2023cqr] (from several of my BORE co-authors), moved from classification to quantile regression surrogates for reasons of calibration and noise-robustness rather than anything that follows here. Its limitations section lists UCB as future work. Remember that sentence.

## An expectation, but not an expected utility

There is a door that looks open. Wilson et al.[^wilson2018maximizing] showed that UCB can be written as an expectation. Inflate the predictive variance by a factor $\beta \pi / 2$ and the half-normal mean does the rest:
$$
\alpha_{\text{UCB}}(\mathbf{x})
= \mathbb{E}_{\gamma}\left[ \mu(\mathbf{x}) + |\gamma| \right],
\qquad
\gamma \sim \mathcal{N}\left(0, \tfrac{\beta\pi}{2} \sigma^2(\mathbf{x})\right),
$$
since $\mathbb{E}|\gamma| = \sqrt{\beta\pi/2} \sigma \cdot \sqrt{2/\pi} = \sqrt{\beta} \sigma$; the inflation factor is chosen precisely so that the half-normal constant cancels.

Look closely, though, and neither ingredient conforms to the recipe. The integrand $\mu(\mathbf{x}) + |y - \mu(\mathbf{x})|$ is a different function at every $\mathbf{x}$: it contains $\mu(\mathbf{x})$, a functional of the very predictive distribution we are refusing to model. And the expectation runs over a variance-inflated law rather than over $p(y \mid \mathbf{x})$. Our observations are samples from the latter, and reweighting them toward the former requires knowing $\mu(\mathbf{x})$ and $\sigma(\mathbf{x})$, which is the same circle again. LFBO's theorem needs a fixed utility $u(y; \tau)$ and samples from the true conditional. This form supplies neither. The door is painted on.

Still, it sharpens the question rather than settling it. UCB admits *some* expectation form; perhaps a cleverer identity produces a conforming one — a fixed utility, under the true measure, up to a monotone transform (the recipe's readout $\pi / (1 - \pi)$ is already one such transform). The next section closes that possibility for good.

## No utility works

Here is what would have to exist: a single utility $u$ and a strictly increasing readout $h$ such that
$$
\mathbb{E}_{y \sim \mathcal{N}(\mu, \sigma^2)}\left[ u(y) \right]
= h\left( \mu + \sqrt{\beta} \sigma \right)
\qquad \text{for all } \mu \in \mathbb{R},\ \sigma > 0.
$$
The quantifier is not negotiable: the pairs $(\mu(\mathbf{x}), \sigma(\mathbf{x}))$ realized across a candidate pool can be anything, so the identity must hold on the whole half-plane. Restricting attention to Gaussian predictives is also a gift to the utility — if it cannot manage the friendliest two-parameter family, larger families are hopeless.

Write $g(\mu, t) \triangleq \mathbb{E}_{y \sim \mathcal{N}(\mu, t)}[u(y)]$ with $t \triangleq \sigma^2$. As a function of $(\mu, t)$, $g$ is the convolution of $u$ with a Gaussian kernel, and the Gaussian kernel is the heat kernel, so $g$ solves the heat equation
$$
\frac{\partial g}{\partial t} = \frac{1}{2} \frac{\partial^2 g}{\partial \mu^2}
$$
for any $u$ for which the expectation exists (the smoothing makes $g$ infinitely differentiable for $t > 0$, however rough $u$ is). Substituting the hypothesis $g = h(s)$ with $s = \mu + \sqrt{\beta} \sqrt{t}$, the two sides become
$$
\frac{\sqrt{\beta} h'(s)}{2 \sigma} = \frac{1}{2} h''(s).
$$
Fix $s$ and vary $\sigma$; the mean $\mu = s - \sqrt{\beta} \sigma$ adjusts to keep $s$ constant, so every $\sigma > 0$ is admissible. The right-hand side does not move. The left-hand side scales as $1/\sigma$. The only escape is $h'(s) = 0$, at every $s$: the readout is constant, and a constant acquisition function ranks nothing. (At $\beta = 0$ the argument relents exactly as it should: the equation forces $h'' = 0$, an affine readout of $\mu$, and the mean is elicitable.)

{{< figure src="figures/level-sets.png" caption="Level sets in the (σ, μ) half-plane. The level sets of a threshold utility's expectation form a fan through (0, τ); UCB's are parallel lines of slope −√β. A fan and a family of parallels agree on at most one line: the level set with value τ. PI at threshold τ impersonates UCB exactly there and nowhere else." >}}

The argument is indifferent to every knob the recipe exposes. The threshold $\tau$ is a constant once the data are fixed, so it is absorbed into $u$. The choice of $f$-divergence in LFBO's general formulation changes the shape of the training objective but never the functional it elicits (the population optimum is $\mathbb{E}[u]$ for every admissible $f$). And the monotone readout is exactly the $h$ that the heat equation just killed. Within weighted classification, that is every door.

One escape remains in principle: abandon classification losses altogether. Perhaps some other loss (squared, absolute, something more exotic) has $\mu + \sqrt{\beta} \sigma$ as its pointwise minimizer. That question turns out to have a name in the forecasting literature, and an answer.

## No loss works either

A functional $T(p)$ is *elicitable* if some loss $\ell(z, y)$ recovers it by minimization: $T(p) = \operatorname{arg min}_z \mathbb{E}_{y \sim p}[\ell(z, y)]$ for every distribution $p$ in the family of interest. The mean is elicitable (squared loss), the median is elicitable (absolute loss), and quantiles are elicitable — hold that thought. Elicitability is exactly the property a likelihood-free acquisition function needs: it is the license to estimate a functional of $p(y \mid \mathbf{x})$ by empirical risk minimization on samples, with no model of $p$ in sight.

Not every functional has the property, and there is a clean necessary condition, due to Osband[^gneiting2011making]: an elicitable functional has convex level sets in distribution space. If $T(p_1) = T(p_2) = c$, then every mixture $p_\lambda = \lambda p_1 + (1 - \lambda) p_2$ must also satisfy $T(p_\lambda) = c$. Mean-plus-standard-deviation fails this test, and fails it strictly. Means mix linearly, but variances pick up a spread term,
$$
\operatorname{Var}(p_\lambda)
= \lambda \sigma_1^2 + (1 - \lambda) \sigma_2^2 + \lambda (1 - \lambda) (\mu_1 - \mu_2)^2,
$$
so any two members of a UCB level set with different means mix to a distribution that exits the set upward. Concretely: $\mathcal{N}(0, 2^2)$ and $\mathcal{N}(2, 1^2)$ both have $\mu + 2\sigma = 4$, yet their even mixture has mean $1$ and variance $3.5$, giving $\mu + 2\sigma \approx 4.74$. So no scalar loss has $\mu + \sqrt{\beta} \sigma$ as its pointwise minimizer over any family that is closed under mixtures and contains the Gaussians. This is the same obstruction that makes standard deviation and expected shortfall non-elicitable in the risk-forecasting literature, and both qualifiers earn their keep. Restricted to the Gaussians alone, the functional survives (the next section is about the loss that elicits it there). And pairs of losses escape: $(\mathbb{E}[y], \mathbb{E}[y^2])$ is jointly elicitable, $z_1 + \sqrt{\beta} \sqrt{z_2 - z_1^2}$ is a readout, and the result is the two-headed heteroscedastic regressor practitioners already fit — the same route by which expected shortfall becomes elicitable in a pair with VaR.[^fissler2016] What follows is the door that needs a single output.

> **How close does classification get?** The family is not barren of optimism. Take $u(y) = e^{ky}$: its expectation under $\mathcal{N}(\mu, \sigma^2)$ is $e^{k\mu + k^2 \sigma^2 / 2}$, strictly increasing in $\mu + \tfrac{k}{2} \sigma^2$ — the entropic risk measure, a genuine mean-plus-spread acquisition sitting inside the recipe all along. What the impossibility forbids is specifically the *scale-equivariant* kind: double the units of $y$ and $\sqrt{\beta} \sigma$ doubles along with $\mu$, while $\tfrac{k}{2} \sigma^2$ does not — $k$ carries units of $1/y$, so the exponential utility must be re-tuned whenever the outputs are rescaled (and its weights make for a noisy estimator besides, but that belongs to the follow-up). Variance-style optimism is available. Standard-deviation-style optimism is what cannot be had.

So the tombstone reads: within the classification recipe, no utility; outside it, no scalar loss. What survives is the observation the post opened with.

## Quantiles come with their own loss

The verdict above condemns the functional $\mu + \sqrt{\beta} \sigma$: the map from a distribution to its mean-plus-standard-deviation is what no loss can elicit. But the opening identity says that under a Gaussian predictive, the UCB acquisition is equally the map from a distribution to its $q$-quantile at $q = \Phi(\sqrt{\beta})$. The two functionals agree on every Gaussian and part ways beyond, and elicitability theory is unambiguous about which one to keep. Quantile level sets are convex (mixing two distributions with the same $q$-quantile cannot move it), and the loss that elicits them is the *pinball loss* from quantile regression,
$$
\rho_q(r) \triangleq q \max(r, 0) + (1 - q) \max(-r, 0),
\qquad r \triangleq y - z,
$$
an absolute loss with asymmetric slopes. The derivation that it works is two lines: $\frac{\partial}{\partial z} \mathbb{E}_{y \sim p}\left[ \rho_q(y - z) \right] = F(z) - q$, which vanishes exactly at $z = F^{-1}(q)$.

{{< figure src="figures/pinball.png" caption="The pinball loss at q = 0.977. Underestimates cost 0.977 per unit and overestimates cost 0.023, so the minimizer settles where only 2.3 percent of the mass lies above." >}}

That is the whole rescue. Fit a regressor $m_{\boldsymbol{\theta}}$ by minimizing $\sum_i \rho_q(y_i - m_{\boldsymbol{\theta}}(\mathbf{x}_i))$ at level $q = \Phi(\sqrt{\beta})$, and maximize its output: $m_{\boldsymbol{\theta}}(\mathbf{x})$ *is* the acquisition function, in the units of $y$. The readout is simpler than the classification recipe's $\pi / (1 - \pi)$, and the threshold machinery disappears entirely, with no $\tau$ to recompute and no labels to flip between iterations. BORE's quantile fraction $\gamma$ and UCB's exploration parameter $\beta$, two knobs that looked unrelated, collapse into a single one: the level $q$. (For minimization problems, train the lower quantile $q = \Phi(-\sqrt{\beta})$ and minimize the output.)

There is also a quiet upgrade hiding in the swap. When the predictive is not Gaussian, $\mu + \sqrt{\beta} \sigma$ and the $q$-quantile genuinely differ, and it is the quantile that keeps meaning what UCB was supposed to mean (the 97.7th percentile of what this candidate might return) while the moment formula drifts. The pinball regressor targets the right object under every noise distribution, which is more than can be said for the formula it replaces.

## The catch

Honesty requires saying which $\sigma$ this recipe buys. The distribution a pinball regressor learns about is the data-generating conditional $p(y \mid \mathbf{x})$, so its spread is observation noise: aleatoric uncertainty. The $\sigma(\mathbf{x})$ in GP-UCB is a different animal wearing the same letter: posterior uncertainty about the latent function, which grows away from the data and shrinks wherever observations accumulate. That epistemic $\sigma$ is the one the phrase "optimism in the face of uncertainty" refers to, and no amount of quantile regression on raw pairs recovers it.

{{< figure src="figures/two-sigmas.png" caption="Two σ's wearing the same letter. The epistemic band of a GP posterior balloons where data is absent; the aleatoric band of the noise itself tracks σ(x) and does not care where you have looked." >}}

The consequence is easiest to see at the extreme. On a noiseless objective, $p(y \mid \mathbf{x})$ is a point mass, every quantile equals the objective's value at $\mathbf{x}$, and the level $q$ does nothing: the rule degenerates to greedy maximization of a fitted surface, which will happily sit in the first decent basin it finds, because nothing in the objective rewards leaving. Under noise, the rule does something meaningful: it seeks the input whose *upper tail* is best, a risk-seeking target that is exactly right when outcomes are stochastic and you care about the best draws. But its exploration is optimism about noise rather than optimism about ignorance.

To be fair to the recipe, this is not a new debt, and I am poorly placed to complain. BORE and LFBO carry the same one: their classifiers are point estimates too, and the exploration they exhibit comes from threshold updates and model misfit rather than from any represented uncertainty. But PI and EI never advertised an epistemic term, and UCB is nothing *but* its epistemic term — the gap bites hardest precisely here. Restoring it means putting uncertainty back at the level of the model parameters (ensembles of quantile regressors with randomized priors, or a Bayesian treatment of $\boldsymbol{\theta}$) and making optimism a statement about disagreement between plausible models rather than a fixed percentile of noise. That, along with the numerical evidence for everything claimed above, is a follow-up post.

## One CDF, three readouts

Step back far enough and the zoo collapses. PI fixes a threshold and reads out a probability: $\alpha_{\text{PI}}(\mathbf{x}) \triangleq 1 - F(\tau \mid \mathbf{x})$. UCB fixes a probability and reads out a threshold: $\alpha_{\text{UCB}}(\mathbf{x}) = F^{-1}(q \mid \mathbf{x})$. They are inverse readouts of the same conditional CDF, which suggests estimating the CDF once and reading it out however you please. A single threshold-conditioned classifier $C(\mathbf{x}, \tau) \approx P(y > \tau \mid \mathbf{x})$, trained on labels $\mathbb{1}(y_i > \tau)$ at sampled thresholds, contains every acquisition in this post: PI at every threshold by evaluation, EI by integration (since $\mathbb{E}[\max(y - \tau, 0)] = \int_\tau^\infty P(y > t) dt$) and UCB by inversion in $\tau$.

{{< figure src="figures/cdf-readouts.png" caption="One conditional CDF, three readouts. Fix a threshold τ and read off a probability (PI); fix a probability q and read off a threshold (UCB); integrate the gap above the CDF beyond τ (EI)." >}}

The pinball loss is secretly this same object. For any $q$,
$$
\rho_q(y - z) = \int \mathbb{1}\left( \min(z, y) < t < \max(z, y) \right) \left| q - \mathbb{1}(t \geq y) \right| dt,
$$
a threshold-integral of cost-weighted misclassification, with false negatives charged $q$ and false positives charged $1 - q$. Quantile regression stacks a continuum of the classification problems that BORE started with and reads the answer out sideways, in the units of $y$. UCB could not be one classification problem. It turns out to be all of them at once.

## Coda: the disguise, worn twice

Now for the sentence you were asked to remember. The conformal quantile method fits $m$ gradient-boosted quantile regressors with pinball losses at equally spaced levels $\alpha_j \triangleq j / (m + 1)$, with $m = 4$ in their experiments, and conformalizes the predictions.[^romano2019] Its acquisition step is *independent Thompson sampling*: draw a large uniform pool of candidates, hand each candidate one of its $m$ predicted quantiles uniformly at random as a pseudo-sample, and keep the best draw. The paper closes by suggesting UCB as a possible future extension.

Look at what the sampling step actually computes. The winning draw beats every other draw, including the draws of the candidates whose coin landed on the top level $\alpha_m$ (about a $1/m$ fraction of the pool, call it $\mathcal{D}_{\text{thin}}$), and each of those draws *is* a top-level quantile. Meanwhile the winner's own top-level quantile can only exceed whatever level it happened to draw, provided the fitted quantiles are pointwise monotone across levels. Chaining the three facts,
$$
\hat{q}_{\alpha_m}(\mathbf{x}_{\text{sel}})
\geq \text{winning draw}
\geq \max_{\mathbf{x} \in \mathcal{D}_{\text{thin}}} \hat{q}_{\alpha_m}(\mathbf{x}).
$$
Independent Thompson sampling over an $m$-point quantile grid is, at worst, fixed-level quantile maximization over a randomly thinned candidate pool. Whatever level the winner happened to draw, the sandwich holds, so the middle quantiles can lift the selection above thinned top-level maximization but never drop it below. With their pool of two thousand candidates, thinning to a quarter changes essentially nothing (the maximum over a random quarter of two thousand candidates is, in expectation, roughly the fourth-best of the full pool), and the sampled acquisition collapses onto its deterministic core: maximize $\hat{q}_{\alpha_m}$.

{{< figure src="figures/ts-collapse.png" caption="Independent Thompson sampling over four quantile curves: one dot per candidate, placed at its randomly drawn level. The winning draw sits on the top-level curve, at that curve's maximizer. The dash-dotted line is the analytic UCB at the grid-equivalent level, μ + √β_eff σ with β_eff = Φ⁻¹(0.8)² ≈ 0.71; under this toy's Gaussian predictive it coincides exactly with the top quantile curve. That coincidence is the corollary: the collapse itself needs no Gaussianity, but when the predictive is Gaussian, the sampler's deterministic core has a name and a β." >}}

That core has a name. Under a Gaussian predictive, the $\alpha_m$-quantile is $\mu + \Phi^{-1}\left( \tfrac{m}{m+1} \right) \sigma$: UCB with $\beta = \Phi^{-1}\left( \tfrac{m}{m+1} \right)^2$, which at $m = 4$ gives $\beta \approx 0.71$. The grid size, chosen as a resolution parameter, has been a $\beta$ dial all along, and the future work the paper asks for is the method it already runs. The extension amounts to deleting the sampling step, which also frees $\beta$ from the grid. Two honest boundaries: separately fitted quantiles can cross (a possibility their own footnote acknowledges), which adds slack to the monotonicity step; and in sparse pools the thinning genuinely bites, which is the high-dimensional regime their limitations section already flags as a concern for Thompson sampling. The disguise slips exactly where they sensed something slipping.

So the title holds in both directions. UCB is a quantile in disguise, and the leading quantile method is UCB in disguise. Elicitability makes the coincidence unremarkable. Scalar doors to scale-equivariant optimism are scarce (the quantile, and its least-squares cousin the expectile), and on a Gaussian predictive every such functional collapses to $\mu + c \sigma$ for some constant $c$. Whoever goes likelihood-free and wants optimism, by derivation or by engineering, ends up in the same room.[^doyle]

[^doyle]: The engineering arrivals are documented. ACHO ([Doyle, 2022](https://arxiv.org/abs/2207.03017)) guides conformal HPO by maximizing a conformalized upper quantile, the fixed-level rule adopted as a heuristic, and its follow-up ([Doyle, 2025](https://arxiv.org/abs/2509.17051)) benchmarks that rule alongside the Thompson sampling of Salinas et al. The derivation above upgrades the heuristic to the unique elicitable form of UCB, with the level pinned by $\beta$.

[^salinas2023cqr]: Salinas, D., Golebiowski, J., Klein, A., Seeger, M., & Archambeau, C. (2023). [Optimizing Hyperparameters with Conformal Quantile Regression](https://arxiv.org/abs/2305.03623). In *Proceedings of the 40th International Conference on Machine Learning (ICML)*.

[^tiao2021bore]: Tiao, L. C., Klein, A., Seeger, M. W., Bonilla, E. V., Archambeau, C., & Ramos, F. (2021). [BORE: Bayesian Optimization by Density-Ratio Estimation](https://proceedings.mlr.press/v139/tiao21a.html). In *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

[^song2022general]: Song, J., Yu, L., Neiswanger, W., & Ermon, S. (2022). [A General Recipe for Likelihood-Free Bayesian Optimization](https://proceedings.mlr.press/v162/song22b.html). In *Proceedings of the 39th International Conference on Machine Learning (ICML)*.

[^thesisexposition]: The derivation follows the presentation in Chapter 5 of [my PhD thesis](/publications/phd-thesis/), which rederives LFBO from the variational representation underlying $f$-divergence estimation.

[^wilson2018maximizing]: Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018). [Maximizing Acquisition Functions for Bayesian Optimization](https://arxiv.org/abs/1805.10196). In *Advances in Neural Information Processing Systems 31 (NeurIPS)*.

[^gneiting2011making]: Gneiting, T. (2011). [Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902). *Journal of the American Statistical Association*, 106(494), 746–762. The convex-level-sets condition is due to K. Osband's 1985 PhD thesis, *Providing Incentives for Better Cost Forecasting* (UC Berkeley).

[^srinivas2010]: Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2010). [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design](https://arxiv.org/abs/0912.3995). In *Proceedings of the 27th International Conference on Machine Learning (ICML)*.

[^bergstra2011]: Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html). In *Advances in Neural Information Processing Systems 24 (NeurIPS)*.

[^fissler2016]: Fissler, T., & Ziegel, J. F. (2016). [Higher Order Elicitability and Osband's Principle](https://arxiv.org/abs/1503.08123). *The Annals of Statistics*, 44(4), 1680–1707.

[^romano2019]: Romano, Y., Patterson, E., & Candès, E. (2019). [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222). In *Advances in Neural Information Processing Systems 32 (NeurIPS)*.
