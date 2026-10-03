---
title: "Attention Is Almost a Gaussian Process - Part III: What the Solve Buys"
subtitle: ""
summary: "We dissect what the Gram solve buys over softmax normalization — explaining away, pooled replicates, escape from the convex hull of the targets — and then quantify it: a closed-form per-query price of softmax, mapped across the queries of a trained attention head."
authors:
- me
tags:
- Transformers
- Gaussian Processes
- Variational Inference
- Probabilistic Models
- Machine Learning
- PyTorch
categories:
- technical
date: 2026-07-15T09:30:00-04:00
lastmod: 2026-07-15T09:30:00-04:00
featured: false
math: true
draft: true

image:
  caption: ""
  focal_point: ""
  preview_only: false
---

> [!NOTE]
> This is **Part III** of a three-part series, *Attention Is Almost a Gaussian Process*. Part I (*One Layer, Both Directions*) and Part II (*The Model Zoo*) are forthcoming as links.

<!-- TODO(beats): opening beat -->

## What the solve buys

**Softmax is the diagonal approximation.** Take a normalized kernel, $\kappa(\mathbf{x}, \mathbf{x}) = 1$, and split the Gram matrix into its diagonal and its off-diagonal part $\mathbf{E} = \mathbf{K} - \mathbf{I}$. When the training inputs are mutually distant, $\mathbf{E} \approx \mathbf{0}$ and the solve collapses:

$$
\mathbf{w}(\mathbf{x}_*) \;=\; (\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}_* \;\approx\; \frac{\mathbf{k}_*}{1+\sigma^2},
\qquad
\mathbf{k}_* = k(\mathbf{X}, \mathbf{x}_*),
$$

while softmax gives $\mathbf{k}_* / \sum_j k_{*j}$ — the same weights up to a per-query positive scalar. So softmax attention *is* the GP smoother precisely when the keys are uncorrelated, and the first Neumann correction says how it degrades when they are not:

$$
(\mathbf{K} + \sigma^2\mathbf{I})^{-1}
= \frac{1}{1+\sigma^2}\left(\mathbf{I} + \frac{\mathbf{E}}{1+\sigma^2}\right)^{-1}
\approx \frac{1}{1+\sigma^2}\left(\mathbf{I} - \frac{\mathbf{E}}{1+\sigma^2}\right),
$$

so that $(\mathbf{E}\mathbf{k}_*)_i = \sum_{j \neq i} \kappa(\mathbf{x}_i, \mathbf{x}_j)\, k_{*j}$ is subtracted from point $i$'s weight: each key is docked by the scores of the keys correlated with it. That subtraction is explaining away, and softmax has no mechanism for it.

**Replicates pool; repetition is not evidence.** Suppose the context contains $m$ exact copies of one observation $(\mathbf{x}_\circ, y_\circ)$, each with independent noise. As a function of $f(\mathbf{x}_\circ)$, the likelihood factorizes through the mean,

$$
\prod_{r=1}^{m} \mathcal{N}\!\left(y_r \mid f(\mathbf{x}_\circ), \sigma^2\right)
\;\propto\;
\mathcal{N}\!\left(\bar{y} \mid f(\mathbf{x}_\circ), \sigma^2/m\right),
$$

so to the GP, $m$ replicates are one observation with noise $\sigma^2/m$: the group's total weight is bounded by what a single noiseless measurement at $\mathbf{x}_\circ$ would receive, no matter how large $m$ grows. Under softmax each copy keeps its full score $\kappa_\circ = \kappa(\mathbf{x}_*, \mathbf{x}_\circ)$, and the group's share is

$$
\frac{m\,\kappa_\circ}{m\,\kappa_\circ + \sum_{\text{others}} \kappa_j} \;\longrightarrow\; 1
\quad \text{as } m \to \infty .
$$

Repeat a context token often enough and it wins outright — repetition masquerades as evidence. Retrieval-augmented pipelines hand attention near-duplicate passages every day.

**Simplex weights cannot leave the hull.** Weights that are positive and sum to one make every prediction a convex combination of the observed targets, so a softmax readout is trapped in $[\min_i y_i, \max_i y_i]$ for every query. Continuing a rising trend past the largest observed value, or swinging a seasonal signal through its next peak, requires $\sum_i w_i y_i > \max_i y_i$ — impossible on the simplex, routine for the solve, whose weights go negative to anti-correlate with observations half a period away. The Mauna Loa CO₂ kernel from [Rasmussen and Williams](https://gaussianprocess.org/gpml/) is the canonical case: its extrapolation exceeds every past observation each year, which no per-query-normalized attention over those observations can represent. The weights themselves are worth looking at, so let us look.

{{< figure src="weight-profile-rbf.png" caption="Nadaraya–Watson weights and GP smoother weights for the same query. RBF kernel, ℓ = 1, σ² = 0.05, fifty evenly spaced training points; the dashed line is zero." >}}

One query, both weight profiles. The softmax weights form a positive bell around the query and sum to one. The solve's weights spike at the query's nearest neighbors and dip negative on either flank — a shape the smoothing literature calls the [*equivalent kernel*](https://proceedings.neurips.cc/paper_files/paper/2004/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) — and those flanks are the previous section made visible: the explaining-away subtraction, and the mechanism by which predictions escape the hull. The companion figure script re-asserts the smoother identity before it draws anything.

<!-- TODO(beats): plug-in suboptimality gap, real-head diagnostic figure, temperature and ordinary-kriging footnotes -->

## Which way does the dictionary run

The dictionary is close to total. The rows fenced out of this series fill in the same way — Markovian GPs and linear attention's recurrence, Vecchia approximations and sparse causal masks, non-Gaussian likelihoods and reweighted attention — so "can this GP construction be expressed as attention" stops being the interesting question. The interesting one is directional: for each row, one side has usually already done the work the other side needs. The solve and its error bars wait on the GP side; the systems discipline and the trillion-token training runs wait on the other. GP predictions are attention. The rest is deciding which way to read each row.
