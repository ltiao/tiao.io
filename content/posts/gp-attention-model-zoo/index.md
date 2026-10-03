---
title: "Attention Is Almost a Gaussian Process - Part II: The Model Zoo"
subtitle: ""
summary: "We read the sparse variational GP's predictive mean as cross-attention onto learned tokens — the KV cache is the variational posterior — and multi-head attention as the posterior mean of a mean-field linear model of coregionalization. Every identity is executable with PyTorch's own attention primitives."
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
date: 2026-07-15T09:00:00-04:00
lastmod: 2026-07-15T09:00:00-04:00
featured: false
math: true
draft: true

image:
  caption: ""
  focal_point: ""
  preview_only: false
---

> [!NOTE]
> This is **Part II** of a three-part series, *Attention Is Almost a Gaussian Process*. Part I (*One Layer, Both Directions*) and Part III (*What the Solve Buys*) are forthcoming as links.

<!-- TODO(beats): opening beat -->

## M learned tokens stand in for n points

Exact conditioning drags an $n \times n$ solve everywhere it goes. The [sparse variational construction](https://proceedings.mlr.press/v5/titsias09a.html) compresses: fix $M$ pseudo-inputs $\mathbf{Z} = \{\mathbf{z}_1, \dots, \mathbf{z}_M\}$, write $\mathbf{u} = f(\mathbf{Z})$ for the process values there, and approximate the posterior over $\mathbf{u}$ by a Gaussian $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$, chosen to minimize $\mathrm{KL}\!\left(q(\mathbf{u}) \,\|\, p(\mathbf{u} \mid \mathbf{y})\right)$ — equivalently, by [maximizing the evidence lower bound](https://arxiv.org/abs/1309.6835). The prior conditional mean given the pseudo-values is linear in them,

$$
\mathbb{E}\!\left[f(\mathbf{x}_*) \mid \mathbf{u}\right] \;=\; k(\mathbf{x}_*, \mathbf{Z})\,\mathbf{K}_{ZZ}^{-1}\,\mathbf{u},
$$

so the predictive mean under $q$ follows by taking the expectation:

$$
\mu(\mathbf{x}_*) \;=\; k(\mathbf{x}_*, \mathbf{Z})\,\mathbf{K}_{ZZ}^{-1}\,\mathbf{m}.
$$

Read as attention: the keys are the $M$ learned locations $\mathbf{Z}$, the values are $\mathbf{v} = \mathbf{K}_{ZZ}^{-1}\mathbf{m}$, and the prediction is scores times values — cross-attention onto $M$ learned tokens, unnormalized, with no softmax anywhere: the kernel weighting stands on its own. In code it is a plain matmul, and the companion script asserts it against the textbook form through a separate Cholesky chain. The $n$ training points appear nowhere in the formula. They entered once, through $(\mathbf{m}, \mathbf{S})$, and left.

{{< figure src="kv-digest.png" caption="The context is written once into a learned digest, and every query cross-attends the digest. In the SVGP the digest is (m, S); in a transformer it is the KV cache." >}}

That last sentence should sound familiar. A transformer serving a long context makes the same move: the context enters once, is digested into keys and values, and every later query attends to the digest. The KV cache is the variational posterior. The difference is that the SVGP knows what its digest is for: $(\mathbf{m}, \mathbf{S})$ is chosen by an explicit objective, the ELBO, which prices exactly how much predictive accuracy the compression costs. The transformer literature's cache-compression heuristics rediscover pieces of that objective one at a time — heavy-hitter eviction ([H2O](https://arxiv.org/abs/2306.14048)), sink-preserving sliding windows ([StreamingLLM](https://arxiv.org/abs/2309.17453)), low-rank latent compression of keys and values ([DeepSeek's multi-head latent attention](https://arxiv.org/abs/2405.04434)) — each a hand-designed answer to the question the ELBO answers by construction. The variational treatment even comes with rates: for sufficiently smooth kernels, $M$ growing polylogarithmically in $n$ suffices ([Burt et al., 2019](https://proceedings.mlr.press/v97/burt19a.html)). And $\mathbf{S}$, the half of the digest attention has no analogue for, enters only the predictive variance — the same error bars attention already discards. The architecture literature has brushed against this identity from the other side: the [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) named its induced set attention blocks after exactly this construction.

## Heads are latent processes

The dictionary's last rows pair heads with latent processes and the output projection with coregionalization mixing. The linear model of coregionalization builds $D$ correlated outputs from $H$ independent latent processes: group $h$ carries $d_h$ latent functions $\mathbf{g}_h$, each an independent draw from $\mathcal{GP}(0, k_h)$ with the group's own kernel, mixed by matrices $\mathbf{A}_h \in \mathbb{R}^{D \times d_h}$:

$$
\mathbf{f}(\mathbf{x}) \;=\; \sum_{h=1}^{H} \mathbf{A}_h\, \mathbf{g}_h(\mathbf{x}).
$$

Condition each latent group on its own evidence $\mathcal{D}_h$, independently of the others — the mean-field treatment — and the posterior mean is

$$
\mathbb{E}\!\left[\mathbf{f}(\mathbf{x}_*)\right] \;\approx\; \sum_{h=1}^{H} \mathbf{A}_h\, \mathbb{E}\!\left[\mathbf{g}_h(\mathbf{x}_*) \mid \mathcal{D}_h\right],
$$

each term a kernel smoother under its group's own kernel. Now write out multi-head attention. Head $h$ smooths the values under its own learned metric — the kernel induced by $\mathbf{W}_Q^h, \mathbf{W}_K^h$ — and concatenation followed by the output projection is a sum over per-head blocks:

$$
\mathrm{out}(\mathbf{x}_*)
\;=\; \sum_{h=1}^{H} \mathbf{W}_O^{h} \sum_{i} a^h_i(\mathbf{x}_*)\, \mathbf{W}_V^{h}\, \mathbf{e}(\mathbf{x}_i)
\;=\; \sum_{h=1}^{H} \mathbf{A}_h\, \hat{\mathbf{g}}_h(\mathbf{x}_*),
\qquad
\mathbf{A}_h \triangleq \mathbf{W}_O^{h} \mathbf{W}_V^{h},
$$

with $\operatorname{rank}(\mathbf{A}_h) \le d_{\text{head}}$. Multi-head attention is the mean-field LMC posterior mean: heads are the latent processes, and the output projection is the coregionalization mixing. The mean-field qualifier is doing honest work — exact joint conditioning would couple the latents through the shared observations, and per-group independence is precisely what parallel heads implement.

{{< figure src="heads-lmc.png" caption="Each head smooths under its own kernel; the output projection mixes the heads. This is the linear model of coregionalization, drawn as an architecture." >}}

The identity is executable, and the algebra forces two constraints that make the construction instructive. `nn.MultiheadAttention` with hand-set packed projections matches the hand-rolled mixture at $2 \times 10^{-16}$, provided the token features carry $(x, x^2, 1)$ — a linear projection cannot square its input, so the RBF key coordinate must ride in as a feature — and provided the query rows carry a factor $\sqrt{d_{\text{head}}}$ to cancel the module's built-in $1/\sqrt{d_{\text{head}}}$. The companion script does the surgery with labeled row ranges.

If heads are latent processes and $\mathbf{W}_O$ is the mixing, then head redundancy is coregionalization rank, and head pruning is rank estimation — the empirical head-pruning literature, read as model selection.

<!-- TODO(beats): closer -->
