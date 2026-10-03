---
title: "A Primer on Pólya-gamma Random Variables - Part IV: Beyond Binary"
subtitle: ""
summary: "We push the Pólya-Gamma machinery beyond binary labels. Binomial and negative-binomial likelihoods come along for free; the softmax does not. We split Bouchard's classical softmax bound into an augmentation part and an irreducible independence part, tour the three standard escapes and the price each pays, and race them on a three-class problem."
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
date: 2026-07-07T16:30:00-04:00
lastmod: 2026-07-07T16:30:00-04:00
featured: false
math: true
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: Center
  preview_only: false

projects: []
---

> [!NOTE]
> This is **Part IV** of a series on Pólya-Gamma random variables.
> See also:
> [Part I — Basic Relationships]({{< relref "/posts/polya-gamma-basic-relationships" >}}),
> [Part II — Bayesian Logistic Regression]({{< relref "/posts/polya-gamma-bayesian-logistic-regression" >}}),
> and [Part III — Local Variational Methods]({{< relref "/posts/polya-gamma-sigmoid-local-variational-lower-bound" >}}).

{{< toc >}}

[Part III]({{< relref "/posts/polya-gamma-sigmoid-local-variational-lower-bound" >}})
of this series closed with a piece of advice: when a local variational
bound appears in the literature, ask which augmented model it is secretly
the ELBO of. For the Jaakkola-Jordan bound on the logistic sigmoid the
answer was the Pólya-gamma augmented model, the correspondence was exact,
and the bound's every quirk had a probabilistic explanation.

This closing part asks how far that machinery travels. Two questions set
the itinerary. First, which other likelihoods admit the same *exact*
treatment? The answer is pleasantly broad: anything built from the
sigmoid's algebraic shape, including binomial and negative-binomial
observation models, comes along for free. Second, what happens with the
one likelihood everyone actually wants, the softmax over $K \geq 3$
classes? There the answer is more interesting: the classical quadratic
bound of Bouchard has a slack that splits cleanly into a part our
augmentation lens explains (a KL divergence, as in Part III) and a part
it cannot (the price of pretending classes are independent), and the
split is measurable, term by term, on data. Every known way around
the obstruction pays a stated toll, and we will meet all three.

Everything quantitative in this post is checked numerically by the
scripts in the source bundle; the identities below pass at tolerances of
$10^{-10}$ or better.

## The Free Extensions

Part II's Laplace-transform callout was stated with more generality than
binary classification needed:
$$
\begin{align*}
\frac{\left (e^{\psi} \right )^a}{\left (1 + e^{\psi} \right )^b} &= 
\frac{1}{2^b} \exp{(\kappa \psi)} \int_0^\infty \exp{\left ( - \frac{\psi^2}{2} \omega \right )} 
\mathrm{PG}(\omega | b, 0) \, \mathrm{d}\omega,
\end{align*}
$$
with $\kappa = a - \frac{b}{2}$. Any likelihood whose $\psi$-dependence
takes the left-hand form is therefore conditionally Gaussian given a
$\mathrm{PG}(b, \cdot)$ auxiliary variable, with everything from Parts II
and III carrying over after the substitution. Two standard cases:

| likelihood | form in $\psi$ | augmentation | $\kappa$ |
|---|---|---|---|
| $\mathrm{Binom}(y \, ; m, \sigma(\psi))$ | $a = y$, $b = m$ | $\omega \sim \mathrm{PG}(m, 0)$ | $y - m/2$ |
| $\mathrm{NB}(y \, ; r, \sigma(\psi))$ | $a = y$, $b = y + r$ | $\omega \sim \mathrm{PG}(y + r, 0)$ | $(y - r)/2$ |

For the binomial, $m$ trials at input $\mathbf{x}_n$ share one auxiliary
variable of shape $m$; for the negative binomial (with $r$ the failure
count and $\sigma(\psi)$ the success probability), the shape depends on
the observed $y$, which is harmless because $y$ is observed. The
conditional posteriors are exponential tiltings exactly as before,
$p(\omega | \psi) = \mathrm{PG}(b, \psi)$, the first moment is
$\mathbb{E}[\omega] = b \, \lambda(c)$ with the same
$\lambda(c) = \frac{1}{2c}\tanh{\left(\frac{c}{2}\right)}$, and the
mean-field story of Part III repeats verbatim: the optimal variational
factor is $\mathrm{PG}(b, \xi_n)$ with $\xi_n^2 = \mathbb{E}_q[\psi_n^2]$,
and the induced bound is the $b$-fold Jaakkola-Jordan bound. Both
augmentations are verified against their exact pmfs by Monte Carlo in
`verify_identities.py`, including non-integer $r$.

So far, no news: the trick extends to everything that is secretly a power
of a sigmoid. The trouble starts when the denominator stops being a power
of $1 + e^{\psi}$.

## The Softmax Problem

For $K$ classes with logits
$\boldsymbol{\psi} = (\psi_1, \dotsc, \psi_K)$, the categorical
likelihood is
$$
p(y | \boldsymbol{\psi}) = \frac{e^{\psi_y}}{\sum_{k=1}^K e^{\psi_k}},
\qquad
\log p(y | \boldsymbol{\psi}) = \psi_y - \mathrm{lse}(\boldsymbol{\psi}),
$$
where $\mathrm{lse}(\boldsymbol{\psi}) \triangleq \log \sum_k e^{\psi_k}$
is the log-partition function. The obstruction to a Gaussian posterior is
now the log-sum-exp rather than a lone softplus, and no
$\mathrm{PG}(b, \cdot)$ variable will linearize it: the Laplace-transform
identity produces powers of $1 + e^{\psi}$ in a single variable, and
$\sum_k e^{\psi_k}$ is not of that form. The classical workaround, due to
Bouchard[^bouchard2007efficient], is a *pair* of bounds stacked on top of
each other, and the stack is where things get interesting.

### The α-Bound, and Its Bernoulli Reading

The first bound introduces a scalar $\alpha$ and pulls the sum apart into
softpluses. Since $e^{x_1} + \dotsb + e^{x_K} \leq \prod_k (1 + e^{x_k})$
(expand the product; every term on the left appears on the right),
$$
\mathrm{lse}(\boldsymbol{\psi}) \leq 
\alpha + \sum_{k=1}^K \varsigma(\psi_k - \alpha)
$$
for every $\alpha$, where $\varsigma(x) = \log(1 + e^x)$ is the softplus.
Each summand is now exactly the object Parts I-III know how to handle.

Before tightening further, look at the slack. With
$p_k \triangleq \sigma(\psi_k - \alpha)$, a short computation (verified
numerically) gives
$$
\begin{align*}
& \alpha + \sum_k \varsigma(\psi_k - \alpha) - \mathrm{lse}(\boldsymbol{\psi}) \newline
& \quad = -\log \mathbb{P}(\text{exactly one of } b_1, \dotsc, b_K \text{ fires}),
\end{align*}
$$
where $b_k \sim \mathrm{Bern}(p_k)$ independently. The α-bound is what
you get by replacing the categorical variable, which fires *exactly one*
class by construction, with $K$ independent Bernoulli variables, and the
slack is precisely the log-probability that this independent surrogate
happens to behave categorically. Optimizing $\alpha$ tunes the $p_k$ to
make the exactly-one event as probable as possible, but no finite
$\alpha$ makes it certain: some slack survives for every $\alpha$ and
every finite $\boldsymbol{\psi}$, already at $K = 2$. This bound touches
nothing.

### Tightening with Jaakkola-Jordan

The second stage is familiar. Each softplus gets the quadratic upper
bound from Part III,
$$
\varsigma(x) \leq 
g(x, \xi) \triangleq \frac{x - \xi}{2} + \frac{\lambda(\xi)}{2}(x^2 - \xi^2) + \varsigma(\xi),
$$
one local parameter $\xi_k$ per class, making the overall bound quadratic
in $\boldsymbol{\psi}$ and hence conjugate to Gaussian priors. The
softplus-side slack identity mirrors the sigmoid-side one from Part III
exactly:
$$
g(x, \xi) - \varsigma(x) = 
\mathrm{KL}[\, \mathrm{PG}(1, \xi) \, \| \, \mathrm{PG}(1, x) \,].
$$
For fixed $\xi$, the optimal $\alpha$ is available in closed form,
$$
\alpha^* = \frac{K/2 - 1 + \sum_k \lambda(\xi_k) \, \psi_k}{\sum_k \lambda(\xi_k)},
$$
(the objective is convex in $\alpha$), and alternating the $\alpha$,
$\xi$, and Gaussian-posterior updates gives Bouchard's variational
algorithm, every step conjugate.

### The Decomposition

Stacking the two stages and collecting the slack terms yields the
equation this post exists for. For any $\alpha$ and any
$\boldsymbol{\xi}$,
$$
\begin{align*}
& \left[ \alpha + \sum_k g(\psi_k - \alpha, \xi_k) \right] - \mathrm{lse}(\boldsymbol{\psi}) \newline
& \quad = \sum_k \mathrm{KL}[\, \mathrm{PG}(1, \xi_k) \, \| \, \mathrm{PG}(1, \psi_k - \alpha) \,] \newline
& \qquad + \left( -\log \mathbb{P}(\text{exactly one fires}) \right).
\end{align*}
$$
The first term is Part III's story replayed $K$ times: a sum of KL
divergences from tilted Pólya-gamma factors to the exact conditionals
they approximate, one per class, vanishing whenever
$\xi_k = |\psi_k - \alpha|$. This part of the bound *is* an augmentation
in the precise sense of the series; if it were the whole story, Bouchard's
bound would be the exactly-tight ELBO of a $K$-fold PG-augmented model
and would touch the log-partition function on a set of parameter values,
the way the Jaakkola-Jordan bound touches the sigmoid at $\xi = \pm\psi$.

The second term is new, strictly positive everywhere, and belongs to no
augmentation: it is the price of the independence surrogate, and no
choice of any parameter in the construction reduces it to zero. This is
what "the softmax has no Pólya-gamma" means, made quantitative. It is not
that auxiliary variables for the softmax do not exist (they do, in
several forms, as the next section shows); it is that *this* bound, the
one that preserves the clean quadratic conjugacy, contains a component
that no exponential-tilting family will ever absorb. A bound that touches
is an augmentation's ELBO wearing different clothes. A bound that touches
nothing is telling you that somewhere along the way, the model itself was
approximated.

## Three Escapes

If the softmax resists, one can change the decomposition, change the
target, or change the likelihood. All three options are in active use,
and each pays its toll somewhere else.

### Stick-Breaking: Pay with Symmetry

Linderman, Johnson, and Adams[^linderman2015dependent] decompose the
categorical (more generally, multinomial) variable into a cascade of
$K - 1$ binary decisions: is the class $1$? Given that it is not, is it
$2$? And so on. With logits $\tilde\psi_1, \dotsc, \tilde\psi_{K-1}$ for
the successive decisions,
$$
p(y = k | \tilde{\boldsymbol{\psi}}) = 
\sigma(\tilde\psi_k) \prod_{j < k} \sigma(-\tilde\psi_j),
$$
a product of sigmoids in *different* variables (the construction itself
goes back to Ren et al. and to Khan et al.[^stickbreaking-priors]). Each
factor is exactly Pólya-gamma augmentable, so exact Gibbs sampling and
conjugate variational inference both go through with the machinery of
Parts II and III, per decision. The toll is symmetry: the parameterization
depends on the order in which classes are peeled off, so exchangeable
classes are modeled unexchangeably, priors mean different things for
early and late classes, and (as the demo below shows) the asymmetry is
visible in the fitted posteriors, particularly in extrapolation.

### One-vs-Each: Pay with the Likelihood Itself

Titsias[^titsias2016one] lower-bounds the softmax by the product of its
pairwise comparisons,
$$
p(y | \boldsymbol{\psi}) \geq \prod_{k \neq y} \sigma(\psi_y - \psi_k),
$$
and the derivation is a small delight: it is the *same* product
inequality as Bouchard's α-step, applied to the reciprocal of the
probability rather than to the partition function. The symmetry extends
to the slack. With competitor variables
$b_k \sim \mathrm{Bern}(\sigma(\psi_k - \psi_y))$ for $k \neq y$,
independently,
$$
\log p(y | \boldsymbol{\psi}) - \sum_{k \neq y} \log \sigma(\psi_y - \psi_k) = 
-\log \mathbb{P}(\text{at most one fires}),
$$
which is verified numerically alongside the others and comes with a
pleasant corollary: the exact softmax probability is
$$
p(y | \boldsymbol{\psi}) = 
\mathbb{P}(\text{none fire} \mid \text{at most one fires}).
$$
Two consequences drop out. For $K = 2$ there is a single competitor,
"at most one fires" is certain, and the bound collapses to the exact
binary logistic likelihood; one-vs-each is a genuine generalization of
the binary case, where Bouchard's bound is strictly loose already
at $K = 2$. And every factor $\sigma(\psi_y - \psi_k)$ is a sigmoid of a
linear function of the logits, so the whole pseudo-likelihood is exactly
Pólya-gamma augmentable, one auxiliary variable per comparison, which is
the route Snell and Zemel[^snell2020bayesian] take for Gaussian process
classification. The toll: the object being augmented is a bound rather
than the likelihood, so the resulting posterior is a pseudo-posterior.
You get exact conjugate inference, for a model that is not quite the one
you wrote down. (Titsias also shows the bound is one member of a
hierarchy: merging competitors tightens it, one merge at a time, until
the exact softmax is recovered.)

### Logistic-Softmax: Pay with the Link

Galy-Fajou, Wenzel, Donner, and Opper[^galyfajou2019multiclass] change
the likelihood to the *logistic-softmax*,
$$
p(y = k | \mathbf{f}) = \frac{\sigma(f_k)}{\sum_{c=1}^K \sigma(f_c)},
$$
and then augment in three stages: a Gamma-integral identity
$\frac{1}{z} = \int_0^\infty e^{-\lambda z} \mathrm{d}\lambda$ removes
the normalizer at the price of a positive variable $\lambda$; the
resulting $e^{-\lambda \sigma(f_c)}$ factors expand, through the Poisson
moment generating function, into products of $\sigma(-f_c)^{n_c}$ with
Poisson-distributed counts $n_c$; and every remaining sigmoid power is
then Pólya-gamma augmentable. Three nested augmentations later the model
is conditionally conjugate, exactly, with closed-form block updates. The
toll is the link function itself: this is a different likelihood with
different tail behavior, chosen because it augments. (Their related-work
discussion also sharpens a distinction this post has been dancing around:
Polson et al.'s original paper already contains a Pólya-gamma scheme for
the standard softmax that supports Gibbs *sampling*, but its ELBO is
intractable, so "an augmentation exists" and "conjugate variational
inference exists" are different claims. The obstruction quantified above
is about the latter.)

## Three Ways to Classify a Point

To make the trade-offs concrete we race the three tractable options on a
three-class cousin of the series' house problem: inputs on the real line,
class-conditional densities $\mathcal{N}(-2, 0.8^2)$,
$\mathcal{N}(0.5, 1^2)$, and $\mathcal{N}(2.5, 0.7^2)$ with equal priors
(so the true class posteriors $\pi_k(x)$ are available in closed form),
$N = 150$ training points, degree-3 polynomial features, and independent
$\mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$ priors on each class's
weights. The three contestants:

1. **Bouchard VB**: the double bound above, with per-class Gaussian
   variational posteriors and closed-form $\xi$, $\alpha$, and Gaussian
   updates (the expected bound increases monotonically through every
   sweep, which we assert programmatically);
2. **stick-breaking PG Gibbs**: two binary Pólya-gamma Gibbs chains in
   the cascade order $0, 1, 2$, each identical to Part II's sampler;
3. **one-vs-each PG Gibbs**: a single Gibbs chain over the stacked
   weight vector, with one Pólya-gamma variable per (point, competitor)
   pair, targeting the pseudo-posterior.

{{< figure src="figures/class_prob_compare_paper_2781x927.png" title="Posterior predictive class probabilities $\pi_k(x)$ for the three methods against the analytic truth (black), one panel per class. Rugs show the training inputs, colored by class. The stick-breaking posterior visibly underestimates the middle-class peak and bends away from the truth at the plot edges, where the order-dependence of the cascade has the most room to express itself." numbered="true" >}}

Averaged over the prediction grid and the three classes, the mean
absolute errors against the true $\pi_k(x)$ are:

| method | mean abs. error | wall clock |
|---|---|---|
| Bouchard VB | 0.020 | 0.05 s |
| one-vs-each PG Gibbs | 0.022 | 0.27 s |
| stick-breaking PG Gibbs | 0.039 | 0.26 s |

Two observations, one expected and one not. The expected one:
stick-breaking pays its symmetry toll where the data cannot arbitrate,
with the largest errors and visible artifacts in the extrapolation
regions. The unexpected one: **the method with the provably loosest
bound produces the best predictions.** There is no contradiction. The
slack in Bouchard's bound is a statement about how well the method
approximates the *evidence* $\log p(\mathbf{y})$; predictions only need
the posterior over weights to land in the right place, and a uniformly
loose bound can still have its maximum near the right spot. Looseness is
a property of the certificate, and only loosely a property of the
answer. (One honest footnote on convergence: the softmax's invariance
under a shared shift of all logits, broken only weakly by the prior,
gives Bouchard VB a nearly flat direction along which the objective
crawls; the predictive distribution converges long before the objective's
last decimal places do.)

## Where the Slack Lives

The decomposition is not just an accounting identity; it is measurable,
per training point, at the fitted variational posterior. Taking
expectations of both slack components under the converged
$q(\boldsymbol{\beta})$:

{{< figure src="figures/slack_decomposition_paper_1500x927.png" title="Bouchard's bound slack per training point at the converged variational posterior, split into its two sources. The independence slack (blue) dominates and peaks at the two class boundaries, exactly where several classes compete and the exactly-one-Bernoulli surrogate is weakest. The tilting slack (red) is the familiar augmentation KL from Part III; it is small, and concentrates in the data-sparse tails where posterior uncertainty inflates $\xi$." numbered="true" >}}

Summed over the training set, the tilting (KL) component accounts
for $2.8$ nats and the independence component for $48.0$ nats: about
$94\%$ of Bouchard's looseness on this problem is the α-bound's
independence surrogate, and essentially none of it is the
Jaakkola-Jordan quadratic that the bound is usually named for. The
geography matches the theory. The tilting slack is Part III's
$\mathrm{KL}[\mathrm{PG}(1, \xi) \, \| \, \mathrm{PG}(1, \cdot)]$ in
expectation, so it grows where the posterior over $\psi_k$ is uncertain,
which here means the tails. The independence slack is
$-\log \mathbb{P}(\text{exactly one fires})$, so it grows where the
fitted logits make several classes plausible at once, which is precisely
the two decision boundaries. If you wanted to improve the bound, this
figure tells you where to spend: not on better $\xi$ updates, but on any
device that lets the per-class Bernoullis talk to each other.

## Coda

The series' moral now comes in two halves. A local variational bound
that *touches* its target on some set of parameter values is, in every
case this series has met, an augmentation's ELBO in disguise, and the
touching set is where the variational factor equals an exact
conditional. A bound that touches *nowhere* is confessing that the model
class itself was approximated somewhere in its derivation, and the
irreducible part of its slack measures that approximation. When you meet
a new bound in the wild, compute its touching set before anything else:
it tells you whether you are looking at inference error, which better
variational parameters can remove, or model error, which they cannot.

The Pólya-gamma distribution took fifteen years to arrive after the
Jaakkola-Jordan bound; the sigmoid's augmentation was there all along,
waiting to be named. The softmax's independence slack is a proof that
some bounds are not waiting for their distribution. There is nothing to
name.

## Links and Further Readings

- Papers:
  * The softmax quadratic bound (Bouchard, 2007)[^bouchard2007efficient]
  * Stick-breaking multinomial augmentation (Linderman et al., 2015)[^linderman2015dependent]
  * The one-vs-each bound (Titsias, 2016)[^titsias2016one]
  * One-vs-each with Pólya-gamma GPs (Snell & Zemel, 2020)[^snell2020bayesian]
  * The conjugate logistic-softmax (Galy-Fajou et al., 2019)[^galyfajou2019multiclass]
  * The binary story this post extends: Polson et al. (2013)[^polson2013bayesian], Jaakkola & Jordan (2000)[^jaakkola2000bayesian], Durante & Rigon (2019)[^durante2019conditionally]
- Code:
  * All figures and every identity quoted above are reproduced by the
    self-contained scripts in this post's source bundle (`src/`), run
    with `uv run`.

---

Cite as:

```
@article{tiao2026polyagammabeyond,
  title   = "{A} {P}rimer on {P}ólya-gamma {R}andom {V}ariables - {P}art IV: {B}eyond {B}inary",
  author  = "Tiao, Louis C",
  journal = "tiao.io",
  year    = "2026",
  url     = "https://tiao.io/posts/polya-gamma-beyond-binary/"
}
```

To receive updates on more posts like this, follow me on [Twitter] and [GitHub]!

[Twitter]: https://twitter.com/louistiao
[GitHub]: https://github.com/ltiao

[^bouchard2007efficient]: Bouchard, G. (2007). Efficient Bounds for the Softmax Function, Applications to Inference in Hybrid Models. In Presentation at the Workshop for Approximate Bayesian Inference in Continuous/Hybrid Systems at NIPS2007.
[^linderman2015dependent]: Linderman, S. W., Johnson, M. J., & Adams, R. P. (2015). [Dependent Multinomial Models Made Easy: Stick-Breaking with the Pólya-Gamma Augmentation](https://arxiv.org/abs/1506.05843). In Advances in Neural Information Processing Systems 28.
[^stickbreaking-priors]: Ren, L., Du, L., Carin, L., & Dunson, D. B. (2011). Logistic Stick-Breaking Process. Journal of Machine Learning Research, 12; Khan, M. E., Mohamed, S., Marlin, B. M., & Murphy, K. P. (2012). A Stick-Breaking Likelihood for Categorical Data Analysis with Latent Gaussian Models. In Proceedings of AISTATS.
[^titsias2016one]: Titsias, M. K. (2016). [One-vs-Each Approximation to Softmax for Scalable Estimation of Probabilities](https://arxiv.org/abs/1609.07410). In Advances in Neural Information Processing Systems 29.
[^snell2020bayesian]: Snell, J., & Zemel, R. (2020). [Bayesian Few-Shot Classification with One-vs-Each Pólya-Gamma Augmented Gaussian Processes](https://arxiv.org/abs/2007.10417). arXiv preprint arXiv:2007.10417.
[^galyfajou2019multiclass]: Galy-Fajou, T., Wenzel, F., Donner, C., & Opper, M. (2019). [Multi-Class Gaussian Process Classification Made Conjugate: Efficient Inference via Data Augmentation](https://arxiv.org/abs/1905.09670). In Proceedings of UAI 2019.
[^polson2013bayesian]: Polson, N. G., Scott, J. G., & Windle, J. (2013). [Bayesian Inference for Logistic Models using Pólya–Gamma Latent Variables](https://arxiv.org/abs/1205.0310). Journal of the American Statistical Association, 108(504), 1339-1349.
[^jaakkola2000bayesian]: Jaakkola, T. S., & Jordan, M. I. (2000). [Bayesian Parameter Estimation via Variational Methods](https://link.springer.com/article/10.1023/A:1008932416310). Statistics and Computing, 10(1), 25-37.
[^durante2019conditionally]: Durante, D., & Rigon, T. (2019). [Conditionally Conjugate Mean-Field Variational Bayes for Logistic Models](https://arxiv.org/abs/1711.06999). Statistical Science, 34(3), 472-485.
