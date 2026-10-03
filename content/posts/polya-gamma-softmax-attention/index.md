---
title: "A Primer on Pólya-gamma Random Variables - Part V: Taylor Attention and KV Cache Compression"
subtitle: ""
summary: "We connect Taylor expansions of LogSumExp and Softmax to the zero-tilt Pólya-Gamma quadratic, then show how the same truncation replaces a growing KV cache with fixed moment statistics and why adaptive curvature resists that compression."
authors:
- me
tags:
- Pólya-Gamma
- Variational Inference
- Bayesian Statistics
- Probabilistic Models
- Machine Learning
categories:
- technical
date: 2026-08-30
lastmod: 2026-08-30
featured: false
draft: true
math: true

image:
  caption: ""
  focal_point: Center
  preview_only: false

projects: []
---

> [!NOTE]
> This is **Part V** of a series on Pólya-Gamma random variables.
> See also:
> [Part I - Basic Relationships]({{< relref "/posts/polya-gamma-basic-relationships" >}}),
> [Part II - Bayesian Logistic Regression]({{< relref "/posts/polya-gamma-bayesian-logistic-regression" >}}),
> [Part III - Local Variational Methods]({{< relref "/posts/polya-gamma-sigmoid-local-variational-lower-bound" >}}),
> and Part IV, *Beyond Binary*.

{{< toc >}}

The Jaakkola-Jordan local variational bound has a parameter $\xi$ that
controls where it tells the exact truth. Its quadratic curvature is the mean
of a tilted Pólya-gamma variable,
$$
\lambda(\xi)=
\mathbb{E}_{\mathrm{PG}(1,\xi)}[\omega]=
\frac{1}{2\xi}\tanh\left(\frac{\xi}{2}\right).
$$

Set $\xi=0$ and the machinery collapses to an especially familiar
approximation:
$$
\log \sigma(\psi)=
-\log 2+\frac{\psi}{2}-\frac{\psi^2}{8}
+O(\!\psi^4).
$$

I had not given this zero-tilt member of the family much thought. Then I
encountered [a derivation of linear attention](https://kexue.fm/archives/11814)
obtained by expanding LogSumExp and Softmax around uniform attention. The same
quadratic appeared, now in many logits at once.

This post follows that coincidence. The Taylor approximation used in linear
attention is the multivariate cousin of the zero-centered quadratic from
[Part III]({{< relref "/posts/polya-gamma-sigmoid-local-variational-lower-bound" >}}).
Truncating it turns an ever-growing KV cache into a fixed collection of moment
statistics. But the Pólya-gamma view also exposes the price: the fixed
curvature that makes this compression possible is precisely what the local
variational method learned not to keep fixed.

## From one logit to many

The binary case is already a two-class Softmax. Write its centered logits as
$\boldsymbol{x}=(\psi/2,-\psi/2)$. The probability of the first class is
$\sigma(\psi)$, while its log-partition function is
$$
\begin{align*}
\operatorname{lse}(\boldsymbol{x})
&= \log\left(e^{\psi/2}+e^{-\psi/2}\right) \\
&= \log 2+\log\cosh\left(\frac{\psi}{2}\right) \\
&= \log 2+\frac{\psi^2}{8}-\frac{\psi^4}{192}+O(\!\psi^6).
\end{align*}
$$
Subtracting this from the first logit recovers the expansion in the opening,
including its quadratic coefficient $-1/8$.

There are now two routes to the same coefficient. Taylor expansion reads it
from the curvature of $\log\cosh(\psi/2)$ at the origin. Pólya-gamma
augmentation starts instead from the exact identity
$$
\frac{1}{\cosh(\psi/2)}=
\mathbb{E}_{\mathrm{PG}(1,0)}
\left[\exp\left(-\frac{\omega\psi^2}{2}\right)\right].
$$
Holding the variational tilt at zero replaces the random curvature $\omega$
by its expectation. Since
$$
\mathbb{E}_{\mathrm{PG}(1,0)}[\omega]=\frac14,
$$
the quadratic contribution is
$-\tfrac12\mathbb{E}[\omega]\psi^2=-\psi^2/8$. The augmentation is still an
exact mixture identity; the approximation enters when we retain only its
untilted quadratic surrogate. At the origin that surrogate is both the
$\xi=0$ Jaakkola-Jordan bound and the second-order Taylor expansion.

## LogSumExp as a cumulant generator

The same calculation scales to any number of logits once we choose the right
random variable. Let $I$ be uniformly distributed on
$\{1,\dotsc,n\}$, set $X=x_I$, and write
$$
\bar{x}=\mathbb{E}[X]=\frac{1}{n}\sum_{i=1}^n x_i,
\qquad
Y=X-\bar{x}.
$$
Then LogSumExp is the log moment-generating function of this empirical
distribution, up to the elementary terms $\log n$ and $\bar{x}$:
$$
\operatorname{lse}(\boldsymbol{x})=
\log n+\bar{x}+\log\mathbb{E}[e^Y].
$$

To make clear what the approximation regards as small, insert a scale
parameter $\varepsilon$ into the centered logits. If
$\boldsymbol{y}=\boldsymbol{x}-\bar{x}\boldsymbol{1}$ and
$\overline{\boldsymbol{y}^r}=n^{-1}\sum_i y_i^r$, then
$$
\begin{align*}
\operatorname{lse}(\bar{x}\boldsymbol{1}+\varepsilon\boldsymbol{y})
={}&\log n+\bar{x}+\log\mathbb{E}[e^{\varepsilon Y}] \\
={}&\log n+\bar{x}
+\frac{\varepsilon^2}{2}\overline{\boldsymbol{y}^2}
+\frac{\varepsilon^3}{6}\overline{\boldsymbol{y}^3} \\
&+\varepsilon^4\left(
\frac{\overline{\boldsymbol{y}^4}}{24}
-\frac{(\overline{\boldsymbol{y}^2})^2}{8}
\right)
+O(\varepsilon^5).
\end{align*}
$$
Equivalently, if $\kappa_r(Y)$ denotes the $r$th cumulant of $Y$,
$$
\operatorname{lse}(\bar{x}\boldsymbol{1}+\varepsilon\boldsymbol{y})=
\log n+\bar{x}
+\sum_{r=2}^{\infty}\frac{\kappa_r(Y)}{r!}\varepsilon^r.
$$
The first cumulant is absent because $Y$ is centered. The next three are its
variance, third central moment, and
$\overline{\boldsymbol{y}^4}-3(\overline{\boldsymbol{y}^2})^2$.

This is the general form of the binary calculation above. LogSumExp begins
with the score of a uniform collection, $\log n+\bar{x}$, then corrects it for
the spread, asymmetry, and higher-order shape of its logits. Taylor truncation
amounts to deciding how many of those empirical cumulants to remember. Setting
$\varepsilon=1$ gives the expansion of the original logits; its accuracy is
therefore governed by their centered spread, not by a common offset that
Softmax would ignore anyway.

## Two Taylor softmaxes

Softmax is the gradient of LogSumExp. Differentiating the cumulant expansion
therefore gives, component by component,
$$
\operatorname{softmax}(\varepsilon\boldsymbol{y})_i=
\frac{1}{n}\left[
1+\varepsilon y_i
+\frac{\varepsilon^2}{2}
\left(y_i^2-\overline{\boldsymbol{y}^2}\right)
\right]
+O(\varepsilon^3).
$$
At zeroth order the attention is uniform. The linear term raises the weight of
a logit above the mean and lowers the weight of one below it; the quadratic
term adjusts for squared deviation. Every correction is centered, so the
truncated weights still sum to one. They need not remain nonnegative: a
sufficiently large negative deviation sends the first-order weight below zero,
and higher-order truncation does not remove that general failure.

Most work called *Taylor Softmax* takes a neighboring route. Instead of
expanding the normalized probabilities, it truncates the exponential and then
normalizes the result. At second order this gives
$$
\begin{align*}
\widetilde{p}_i
&=\frac{1+\varepsilon y_i+\varepsilon^2y_i^2/2}
{\sum_j(1+\varepsilon y_j+\varepsilon^2y_j^2/2)} \\
&=\frac{1+\varepsilon y_i+\varepsilon^2y_i^2/2}
{n\left(1+\varepsilon^2\overline{\boldsymbol{y}^2}/2\right)}.
\end{align*}
$$
Expanding the denominator shows that $\widetilde{p}_i$ agrees with the direct
Softmax expansion through $O(\varepsilon^2)$. But using the displayed rational
function rather than expanding it further retains additional higher-order
terms. More importantly,
$$
1+z+\frac{z^2}{2}=\frac{(z+1)^2+1}{2}>0,
$$
so $\widetilde{\boldsymbol{p}}$ is an honest probability distribution for
every finite logit vector.

The two constructions thus have the same local curvature and different global
behavior. The direct expansion preserves normalization order by order but may
lose positivity. Truncating and renormalizing the exponential preserves both
by turning the polynomial approximation into a rational one. This second
construction is the one used by
[TaylorShift](https://link.springer.com/chapter/10.1007/978-3-031-78172-8_1)
and by the Taylor feature map in
[BASED](https://proceedings.mlr.press/v235/arora24a.html), although the
implementations make their own choices about scaling and centering the logits.

## What the cache remembers

Consider one causal attention head. A new query $\boldsymbol{q}$ scores the
keys from the preceding $n$ tokens and returns a weighted average of their
values,
$$
\boldsymbol{o}(\boldsymbol{q})=
\frac{\sum_{i=1}^n
\exp(\boldsymbol{q}^{\top}\boldsymbol{k}_i)\boldsymbol{v}_i}
{\sum_{i=1}^n\exp(\boldsymbol{q}^{\top}\boldsymbol{k}_i)}.
$$
I have absorbed the usual $1/\sqrt{d}$ scale into the queries and keys. At
the next decoding step the query changes, while all preceding keys and values
stay fixed. Exact attention keeps every pair
$(\boldsymbol{k}_i,\boldsymbol{v}_i)$ available because it does not know which
one the next query will select. This is the KV cache, and its size after $n$
tokens is $O(n(d_k+d_v))$ per layer and batch element.

The exponential dot-product kernel has an infinite-dimensional feature map.
Replacing it by a polynomial cuts that feature map off after finitely many
coordinates. Only then can the past be collected into a fixed-size state. The
compression is exact for the truncated kernel, not for the original Softmax.

## First-order compression

Let
$$
\bar{\boldsymbol{k}}=\frac1n\sum_i\boldsymbol{k}_i,
\qquad
\bar{\boldsymbol{v}}=\frac1n\sum_i\boldsymbol{v}_i.
$$
For this query, the centered logits are
$y_i=\boldsymbol{q}^{\top}(\boldsymbol{k}_i-\bar{\boldsymbol{k}})$. Keeping
only the first-order Softmax correction gives
$$
p_i(\boldsymbol{q})
\approx
\frac1n\left[
1+\boldsymbol{q}^{\top}
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})
\right].
$$
Substitution into the attention output yields
$$
\begin{align*}
\boldsymbol{o}_1(\boldsymbol{q})
&=\bar{\boldsymbol{v}}
+\boldsymbol{q}^{\top}\boldsymbol{C}_{kv}, \\
\boldsymbol{C}_{kv}
&\triangleq
\frac1n\sum_i
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})\boldsymbol{v}_i^{\top} \\
&=\frac1n\sum_i\boldsymbol{k}_i\boldsymbol{v}_i^{\top}
-\bar{\boldsymbol{k}}\bar{\boldsymbol{v}}^{\top}.
\end{align*}
$$
The individual tokens have disappeared. To evaluate this approximation for
every future query, it is enough to retain
$$
n,
\qquad
\sum_i\boldsymbol{k}_i,
\qquad
\sum_i\boldsymbol{v}_i,
\qquad
\sum_i\boldsymbol{k}_i\boldsymbol{v}_i^{\top}.
$$
Each statistic admits an online additive update. The resulting state costs
$O(d_kd_v)$ rather than $O(n(d_k+d_v))$ memory. Calling this a compressed KV
cache is reasonable, provided we remember what was compressed away: token
identity. The state can answer aggregate linear questions about its history,
but it cannot recover an arbitrary old value as if the original key were
still present.

## Quadratic features, cubic state

The second-order direct expansion remembers one moment more. Define the key
covariance and the value-weighted second moment
$$
\begin{align*}
\boldsymbol{\Sigma}_k
&\triangleq
\frac1n\sum_i
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})^{\top}, \\
\boldsymbol{T}_{kkv}
&\triangleq
\frac1n\sum_i
\operatorname{vec}\!\left(
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})
(\boldsymbol{k}_i-\bar{\boldsymbol{k}})^{\top}
\right)\boldsymbol{v}_i^{\top}.
\end{align*}
$$
The approximate output becomes
$$
\boldsymbol{o}_2(\boldsymbol{q})=
\boldsymbol{o}_1(\boldsymbol{q})
+\frac12\left[
(\boldsymbol{q}\otimes\boldsymbol{q})^{\top}\boldsymbol{T}_{kkv}
-(\boldsymbol{q}^{\top}\boldsymbol{\Sigma}_k\boldsymbol{q})
\bar{\boldsymbol{v}}
\right].
$$
Once again, raw sums up to second order suffice to maintain the centered
statistics online. The price is visible in the shape of
$\boldsymbol{T}_{kkv}$: it has $O(d_k^2d_v)$ entries.

The renormalized Taylor construction packages the same growth more neatly as
a feature map. Ignoring duplicate entries of the symmetric outer product for
the moment, and writing the map for raw rather than centered scores, define
$$
\phi_2(\boldsymbol{u})=
\left[
1,
\boldsymbol{u},
\frac{1}{\sqrt{2}}\operatorname{vec}
(\boldsymbol{u}\boldsymbol{u}^{\top})
\right].
$$
It satisfies
$$
\phi_2(\boldsymbol{q})^{\top}\phi_2(\boldsymbol{k})=
1+\boldsymbol{q}^{\top}\boldsymbol{k}
+\frac12(\boldsymbol{q}^{\top}\boldsymbol{k})^2.
$$
Using centered scores replaces $\boldsymbol{k}_i$ by
$\boldsymbol{k}_i-\bar{\boldsymbol{k}}$; the raw moment statistics above still
support that correction. The uncentered display keeps the recurrent algebra
legible and matches the usual presentation of Taylor linear attention.
For the approximate kernel, all preceding tokens can therefore be summarized
by two recurrent states,
$$
\boldsymbol{S}_n=
\sum_{i=1}^n\phi_2(\boldsymbol{k}_i)\boldsymbol{v}_i^{\top},
\qquad
\boldsymbol{z}_n=
\sum_{i=1}^n\phi_2(\boldsymbol{k}_i),
$$
with additive updates
$$
\boldsymbol{S}_{n+1}=
\boldsymbol{S}_n+\phi_2(\boldsymbol{k}_{n+1})\boldsymbol{v}_{n+1}^{\top},
\qquad
\boldsymbol{z}_{n+1}=
\boldsymbol{z}_n+\phi_2(\boldsymbol{k}_{n+1}).
$$
Decoding no longer reads the original cache:
$$
\widetilde{\boldsymbol{o}}_n(\boldsymbol{q})=
\frac{\phi_2(\boldsymbol{q})^{\top}\boldsymbol{S}_n}
{\phi_2(\boldsymbol{q})^{\top}\boldsymbol{z}_n}.
$$

This is linear attention in its recurrent form. The state is constant in
sequence length but large in feature dimension. With symmetry taken into
account, the quadratic map has
$1+d_k+d_k(d_k+1)/2$ coordinates, so the dominant state still scales as
$O(d_k^2d_v)$. BASED controls this cost by projecting queries and keys to a
smaller dimension $d'$ before forming the quadratic features; its experiments
use values such as $d'=16$. Increasing $d'$ increases both the state size and
the amount of associative recall the model can support.

## What the compressed state cannot do

An exact cache can implement content-addressed retrieval: a query sharply
matching one old key can copy the corresponding value even when the context is
long. A finite collection of moments cannot preserve this ability for every
possible history. Different token sequences inevitably produce the same
state, and no later query can separate them. BASED makes this trade explicit,
both experimentally and through a communication-complexity lower bound:
reliable associative recall requires the recurrent state to grow with the
amount of information that may need to be recovered.

This is why the construction above is better described as a replacement for
the KV cache than as a lossless compression of it. Applying the recurrent
updates exactly computes polynomial attention. It only approximates a model
that was trained to use Softmax attention, and a direct substitution can alter
the behavior badly when the original attention is sharp. Training with the
polynomial kernel, or at least fine-tuning or distilling after conversion,
lets the surrounding projections adapt to the state they will actually have.

BASED therefore combines a small exact sliding window with a global Taylor
state. Recent tokens retain precise local
interactions, while older context is available through a coarser recurrent
summary. A closely related 2026 preprint,
[Tensor Cache](https://arxiv.org/abs/2605.22884), feeds evicted KV pairs into an
outer-product memory and keeps a sliding-window cache in front of it. The
division of labor is sensible: exact storage handles the few tokens for which
identity matters most, and moment storage keeps a cheap trace of everything
else.

## What the Pólya-gamma view adds

The zero-centered Taylor series fixes one curvature for every logit. Part III
gave us a family instead. In our convention, the Jaakkola-Jordan bound is
$$
\log\sigma(\psi)
\geq
\log\sigma(\xi)
+\frac{\psi-\xi}{2}
-\frac{\lambda(\xi)}{2}(\psi^2-\xi^2),
$$
where
$$
\lambda(\xi)=
\mathbb{E}_{\mathrm{PG}(1,\xi)}[\omega]=
\frac{1}{2\xi}\tanh\left(\frac{\xi}{2}\right).
$$
At $\xi=0$, $\lambda(0)=1/4$ and the quadratic term is $-\psi^2/8$, exactly
as in the opening. Away from the origin, $\lambda(\xi)$ decreases with
$|\xi|$. The bound adapts its curvature to the region of the logit it needs to
cover. For a fixed $\psi$ it touches at $\xi=|\psi|$; under a variational
posterior, the update becomes $\xi^2=\mathbb{E}[\psi^2]$.

That adaptivity is the part a fixed Taylor cache throws away. Suppose we tried
to attach a local Pólya-gamma curvature to each binary comparison made by an
attention query. Its quadratic term would contain factors of the form
$$
\lambda(\xi_{qi})(\boldsymbol{q}^{\top}\boldsymbol{k}_i)^2,
$$
with $\xi_{qi}$ chosen from the magnitude or uncertainty of that particular
query-key interaction. The coefficient now depends jointly on the new query
and an old key. It cannot in general be moved inside a query-independent sum
computed when the key enters the cache.

The two requirements conflict. Linear attention works because its kernel
separates as
$$
K(\boldsymbol{q},\boldsymbol{k})=
\phi(\boldsymbol{q})^{\top}\phi(\boldsymbol{k}),
$$
which permits the associative reordering
$$
\sum_i K(\boldsymbol{q},\boldsymbol{k}_i)\boldsymbol{v}_i=
\phi(\boldsymbol{q})^{\top}
\sum_i\phi(\boldsymbol{k}_i)\boldsymbol{v}_i^{\top}.
$$
Local variational curvature is useful precisely because it responds to each
interaction. The response destroys this finite separation unless we
approximate it again.

There are intermediate possibilities, though I would regard them as design
proposals rather than consequences of the Pólya-gamma identity. One could
share $\xi$ across a head or a block, quantize it into a small number of
curvature buckets, or approximate $\lambda(\xi_{qi})$ with another finite
separable feature expansion. Each choice restores some adaptivity and spends
additional state to do it. The untilted Taylor cache is the cheapest endpoint:
one fixed curvature, no per-token latent state, and no claim to remain accurate
once the logits move far from uniformity.

## The binary bridge stops at the softmax

The connection so far is a connection of local curvature, not a new
Pólya-gamma augmentation of the full attention distribution. This distinction
matters. For any one Softmax component we can write
$$
\operatorname{softmax}(\boldsymbol{x})_i=
\sigma\left(
x_i-\log\sum_{j\neq i}e^{x_j}
\right).
$$
Conditioned on all the other logits, this is an ordinary binary logistic
term, and the Pólya-gamma identity applies. Polson, Scott, and Windle use this
observation to construct coordinate-wise updates for multinomial logistic
regression.

It does not give attention a single convenient latent state. The offset
$\log\sum_{j\neq i}e^{x_j}$ couples the chosen component to every competitor;
for attention, it also changes with the query. Updating one component at a
time is useful in a Gibbs sampler because the remaining components can be held
fixed. Autoregressive decoding asks for all attention weights at once, for a
new query, without revisiting the stored keys. The coordinate construction
does not meet that requirement.

Part IV of this series met the same obstruction from the variational side.
Bouchard's bound breaks LogSumExp into
binary softplus terms that can each receive a Pólya-gamma quadratic, but an
additional independence bound remains. Its slack is not the KL divergence of
any Pólya-gamma factor and cannot be tightened by adjusting $\xi$. The Taylor
series takes a different bargain: it expands the coupled LogSumExp directly,
so no independence slack appears, but the resulting polynomial is local and
is not a global variational bound.

The three quadratic objects are therefore related without being
interchangeable. The direct Softmax series preserves total mass but can assign
negative weights. The renormalized exponential polynomial preserves a valid
probability distribution but has no variational-bound guarantee. The
Jaakkola-Jordan construction supplies a global bound and an exact augmented
model in the binary case, but its useful adaptive curvature is hostile to a
query-independent cache. Agreement at second order tells us that they share a
tangent. It does not transfer the other guarantees from one construction to
another.

## How much of this is new?

The calculus is old. Writing LogSumExp as a cumulant-generating function,
differentiating it to obtain Softmax, and expanding the exponential kernel are
standard moves. Their use in efficient attention has also been published in
several forms:

| result | status | what it contributes here |
|---|---|---|
| [The Hedgehog & the Porcupine](https://arxiv.org/abs/2402.04347) | 2024 preprint | Studies Taylor-exponential feature maps and the Softmax properties that linear attention needs to imitate. |
| [BASED](https://proceedings.mlr.press/v235/arora24a.html) | ICML 2024 | Uses a quadratic Taylor feature map as a fixed recurrent state and measures the resulting memory-recall tradeoff. |
| [TaylorShift](https://link.springer.com/chapter/10.1007/978-3-031-78172-8_1) | ICPR 2024 | Renormalizes the quadratic exponential polynomial and rearranges full attention from quadratic to linear complexity. |
| [The Key to Going Linear](https://arxiv.org/abs/2607.07706) | 2026 preprint | Studies post-hoc linearization of frozen pretrained Transformers and the structural changes needed to preserve quality. |
| [LogSumExp and Softmax Taylor expansions](https://kexue.fm/archives/11814) | 2026 technical post | Presents the centered expansion cleanly and connects its mean and covariance terms to sparse and linear attention. |

The derivations in this post should not be read as a new attention mechanism.
The part I have not found stated in these sources is the Pólya-gamma reading:
the common second-order curvature is the zero-tilt member of the
Jaakkola-Jordan family, while the adaptive tilt that improves the variational
bound interferes with the factorization needed for a fixed recurrent state.
That is an interpretation and a design constraint, not a publication claim.

## Coda

Taylor expansion does turn a KV cache into moment statistics. First order
keeps means and a key-value cross moment. Second order adds a value-weighted
key covariance, or equivalently a quadratic feature state. The storage no
longer grows with context, but the model has surrendered exact token-level
retrieval and the state grows quickly with feature dimension instead.

The Pólya-gamma connection does not remove this trade. It explains one source
of it. Zero tilt gives a fixed quadratic curvature, and fixed curvature gives
a finite feature map that can be accumulated before the query arrives. Local
variational inference improves the quadratic by adapting its curvature to the
logit, but that curvature depends on the interaction between the future query
and each stored key. Recovering it requires either revisiting the cache or
approximating the adaptation with more state.

I would therefore start with a hybrid rather than replace a pretrained KV
cache wholesale: exact attention for a small recent or selected set, a
positive normalized polynomial state for the evicted remainder, and enough
fine-tuning for the model to learn which path it can trust. Pólya-gamma
curvature may suggest better head-level, block-level, or bucketed corrections.
It does not offer a free adaptive cache. Once the curvature depends jointly on
the query and an old key, the decoder must either revisit that key or
approximate the dependence with a richer state.
