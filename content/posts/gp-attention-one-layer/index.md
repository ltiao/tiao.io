---
title: "Attention Is Almost a Gaussian Process - Part I: One Layer, Both Directions"
subtitle: ""
summary: "We read the Gaussian process posterior predictive as attention — softmax attention is Nadaraya–Watson smoothing, and exact GP regression differs by one line — then reverse the direction: given a trained attention layer, we derive the GP predictive law it implies, including the error bars it discards. Every identity is executable with PyTorch's own attention primitives."
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
date: 2026-07-14T21:30:00-04:00
lastmod: 2026-07-14T21:30:00-04:00
featured: false
math: true
draft: true

image:
  caption: ""
  focal_point: ""
  preview_only: false
---

> [!NOTE]
> This is **Part I** of a three-part series, *Attention Is Almost a Gaussian Process*. Part II (*The Model Zoo*) and Part III (*What the Solve Buys*) are forthcoming.

The posterior predictive mean of a Gaussian process is a weighted sum of the training targets:

$$
\mu(\mathbf{x}_*) \;=\; k(\mathbf{x}_*, \mathbf{X})\,(\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y} \;=\; \sum_{i=1}^{n} w_i(\mathbf{x}_*)\, y_i .
$$

Read it the way a transformer would. The test input is a query, the training inputs are keys, the targets are values, and $w(\mathbf{x}_*)$ is a row of attention weights — the whole predictor is cross-attention whose attention matrix is the linear smoother $\mathbf{K}_{*X}(\mathbf{K} + \sigma^2\mathbf{I})^{-1}$. Softmax attention computes the same object with one substitution: the Gram inverse replaced by per-row normalization, which makes a standard attention layer exactly a Nadaraya–Watson estimator — kernel evaluations, normalized per query. GP predictions are attention. Everything interesting in this post lives in the gap between those two normalizers.

The rest is a dictionary. Each row is an identity, and each gets paid off across this series — in equations and in runnable PyTorch.

| Attention | Gaussian process |
|---|---|
| queries | test inputs |
| keys | training inputs |
| values | training targets |
| attention matrix | linear smoother |
| softmax | diagonal approximation of the Gram inverse (Part III) |
| learned latent tokens | inducing points (Part II) |
| KV cache | variational posterior (Part II) |
| heads | latent processes (Part II) |
| output projection $\mathbf{W}_O$ | coregionalization mixing (Part II) |

## Stock attention is already a kernel regressor

That reading sounds like an analogy, so let us make it exact before building on it. The Nadaraya–Watson estimator with an RBF kernel predicts

$$
\hat{f}(x_*) \;=\; \sum_{i=1}^{n} \frac{\kappa(x_*, x_i)}{\sum_{j=1}^{n}\kappa(x_*, x_j)}\, y_i,
\qquad
\kappa(x_*, x_i) = \exp\!\left(-\frac{(x_* - x_i)^2}{2\ell^2}\right),
$$

which is a softmax over the scores $s_i = -(x_* - x_i)^2 / 2\ell^2$. To run it on PyTorch's attention kernel, the squared distance has to become a dot product. Expand it:

$$
-\frac{(x_* - x_i)^2}{2\ell^2}
\;=\; \frac{x_*}{\ell}\cdot\frac{x_i}{\ell} \;+\; 1 \cdot\left(-\frac{x_i^2}{2\ell^2}\right) \;-\; \frac{x_*^2}{2\ell^2}
\;=\; \big\langle\, q(x_*),\, k(x_i) \,\big\rangle \;-\; \frac{x_*^2}{2\ell^2},
$$

with two-dimensional queries and keys

$$
q(x) = \left(\frac{x}{\ell},\; 1\right),
\qquad
k(x) = \left(\frac{x}{\ell},\; -\frac{x^2}{2\ell^2}\right).
$$

The leftover $-x_*^2/2\ell^2$ does not depend on $i$, and softmax is invariant to per-row shifts, $\operatorname{softmax}(\mathbf{s} + c\,\mathbf{1}) = \operatorname{softmax}(\mathbf{s})$, so it vanishes from the weights. Stock `F.scaled_dot_product_attention` with these queries and keys computes Nadaraya–Watson exactly — `scale=1.0` because the lengthscale already plays the role of temperature, and the default $1/\sqrt{d}$ would distort it:

```python
q = torch.stack([x_star / ell, torch.ones_like(x_star)], dim=-1)   # [n*, 2]
k = torch.stack([x / ell, -x**2 / (2 * ell**2)], dim=-1)           # [n, 2]
out = F.scaled_dot_product_attention(q, k, y[:, None], scale=1.0)

log_k = -(x_star[:, None] - x)**2 / (2 * ell**2)
ref = torch.softmax(log_k, dim=-1) @ y
torch.testing.assert_close(out.squeeze(-1), ref)
```

The assert passes at float64 defaults ($\mathrm{rtol} = \mathrm{atol} = 10^{-7}$). Every identity in this post ships with one of these. The construction is ours; the reading of softmax attention as kernel smoothing goes back at least to [Tsai et al. (2019)](https://arxiv.org/abs/1908.11775).

## One line separates them

Now put the GP weights from the opening identity,

$$
\mathbf{w}(\mathbf{x}_*)^\top \;=\; k(\mathbf{x}_*, \mathbf{X})\,(\mathbf{K} + \sigma^2\mathbf{I})^{-1},
$$

next to the Nadaraya–Watson weights, in code. Shared scores, shared readout, one differing line:

```python
log_k = -(x_star[:, None] - x)**2 / (2 * ell**2)     # scores, both paths
K     = (-(x[:, None] - x)**2 / (2 * ell**2)).exp()  # the matrix softmax never builds

w_nw = torch.softmax(log_k, dim=-1)                           # normalize
w_gp = torch.linalg.solve(K + sigma_sq * I, log_k.exp().T).T  # solve

pred_nw = w_nw @ y
pred_gp = w_gp @ y   # the textbook posterior mean (asserted against a Cholesky reference)
```

Both are attention: kernel scores in, weighted sum of values out. The difference between a heuristic smoother and the posterior mean — the Bayes-optimal prediction under squared loss, given the prior — is which normalizer turns scores into weights. And note what the solve consumes that softmax never computes: $\mathbf{K}$, the key–key similarities. Softmax attention compares the query to each key; the solve also compares the keys to each other. Everything in the gap follows from that; Part III dissects it.

{{< figure src="fork.png" caption="One pipeline, one differing stage: both normalizers turn the same kernel scores into weights for the same readout, and only the solve consumes the Gram matrix." >}}

One more object before we reverse direction. The GP's error bars reuse the same attention weights:

$$
\sigma^2(\mathbf{x}_*) \;=\; k(\mathbf{x}_*, \mathbf{x}_*) \;-\; k(\mathbf{x}_*, \mathbf{X})\,(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\, k(\mathbf{X}, \mathbf{x}_*)
\;=\; k(\mathbf{x}_*, \mathbf{x}_*) \;-\; k(\mathbf{x}_*, \mathbf{X})\, \mathbf{w}(\mathbf{x}_*),
$$

one extra inner product between the scores and the weights already in hand. Attention, read as a GP, computes the posterior mean and throws away the error bars at the last step. The reverse direction below is going to need it.

<!-- TODO(beats): reverse-direction capstone — exponential-feature completion, the two objects the forward pass never computes, the layer's predictive mean and variance -->

## What these identities are not

They are statements about what one cross-attention readout computes — exact when queries, keys, and values carry $(\mathbf{x}, y)$ pairs: in-context regression, neural-process-style decoders, the constructions in the companion script. For self-attention inside a language model they describe the operator rather than the task — though the in-context-learning literature has shown transformers implementing exactly this kind of regression internally ([von Oswald et al., 2022](https://arxiv.org/abs/2212.07677)). And they are distinct from the infinite-width theorems ([Hron et al., 2020](https://proceedings.mlr.press/v119/hron20a.html)), where attention networks converge to GPs over random parameters: that is a statement about randomness over weights; this post is arithmetic about a single forward pass.
