---
aliases:
  - /post/polya-gamma-sigmoid-local-variational-lower-bound/
  # Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "A Primer on Pólya-gamma Random Variables - Part III: Local Variational Methods"
subtitle: ""
summary: "One weird trick to make exact inference in Bayesian logistic regression tractable."
authors: 
- me
tags:
- Machine Learning
- Bayesian Statistics
- Probabilistic Models
- Pólya-Gamma
categories: []
date: 2021-05-11T17:20:53+01:00
lastmod: 2021-05-11T17:20:53+01:00
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

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

> [!NOTE]
> This is **Part III** of a three-part series on Pólya-Gamma random variables.
> See also: [Part I — Basic Relationships]({{< relref "/posts/polya-gamma-basic-relationships" >}})
> and [Part II — Bayesian Logistic Regression]({{< relref "/posts/polya-gamma-bayesian-logistic-regression" >}}).

## Lower bound on the logistic sigmoid function

(Jaakkola and Jordan, 2000[^jaakkola2000bayesian], Bouchard, 2007[^bouchard2007efficient])

In addition to being a popular activation function in deep neural networks, the 
softplus function appears frequently throughout the machine learning literature.
$$
\varsigma(\psi) \doteq \log{(1 + e^{\psi})}
$$
Its derivative is the logistic sigmoid function, the logarithm of which can also 
be related to the softplus.

Its multivariate generalization is the log-sum-exp and its derivate is the 
softmax function for categorical / multi-class classification.
$$
\varsigma(\psi) \leq 
\frac{\psi - \xi}{2} + \frac{\lambda(\xi)}{2} (\psi^2 - \xi^2) + \varsigma(\xi) \doteq
g(\psi, \xi)
$$

$$
\lambda(\xi) \doteq \frac{1}{2\xi} \tanh{\left(\frac{\xi}{2}\right)}
$$

{{< figure src="figures/softplus_paper_1500x927.png" title="Softplus function." numbered="true" >}}


$$
\sigma(\psi) \doteq \frac{1}{1 + e^{-\psi}}
$$

$$
\sigma(\psi) \geq \sigma(\xi) \exp{\left(\frac{\psi - \xi}{2} - \frac{\lambda(\xi)}{2} (\psi^2 - \xi^2)\right)} 
\doteq \ell(\psi, \xi)
$$

{{< figure src="figures/sigmoid_paper_1500x927.png" title="Sigmoid function." numbered="true" >}}

The lower bound $\ell(\psi, \xi)$ is easy to derive given the upper bound $g(\psi, \xi)$ on $\varsigma(\psi)$.
We simply note that $\log{\sigma(\psi)} = - \varsigma(- \psi)$ and 
reverse the direction of the inequality accordingly to obtain the lower bound
$$ 
\begin{align}
\log{\sigma(\psi)} = - \varsigma(- \psi) & \geq -g(-\psi, \xi) \newline & =
\frac{\psi + \xi}{2} - \frac{\lambda(\xi)}{2} (\psi^2 - \xi^2) - \varsigma(\xi) \newline & = 
\frac{\psi - \xi}{2} - \frac{\lambda(\xi)}{2} (\psi^2 - \xi^2) + \log{\sigma(\xi)}.
\end{align}
$$
In the last step above, we've used the fact 
that $\frac{\xi}{2} - \varsigma(\xi) = - \frac{\xi}{2} - \varsigma(-\xi)$.
Hence, by exponentiating both sides (and noting that the exponential function 
is strictly increasing), we have
$$
\sigma(\psi) \geq \ell(\psi, \xi).
$$

## Polya-gamma Augmented Bayesian Logistic Regression Model

The model joint factorizes as
$$
\begin{align}
p(\mathbf{y}, \boldsymbol{\omega}, \boldsymbol{\beta}) &= 
\prod_{n=1}^N p(y_n, \omega_n, \boldsymbol{\beta}) \newline &=
p(\boldsymbol{\beta}) \prod_{n=1}^N p(y_n | \omega_n, \boldsymbol{\beta}) p(\omega_n) 
\end{align}
$$

### Priors

The prior on weight parameters
$$
p(\boldsymbol{\beta}) = \mathcal{N}(\boldsymbol{\beta} | \mathbf{m}, \mathbf{S})
$$

The prior on auxiliary variables
$$
p(\omega_n) = \mathrm{PG}(\omega_n | 1, 0)
$$

### Likelihood

$$
p(y_n | \omega_n, \boldsymbol{\beta}) = 
\frac{1}{2} \exp{\left\{-\frac{\omega_n}{2}\left(\psi_n^2 - 2\psi_n\frac{\kappa_n}{\omega_n}\right)\right\}}
$$
where $\kappa_n = y_n - \frac{1}{2}$ and $\psi_n = \boldsymbol{\beta}^{\top} \boldsymbol{\phi}_n$
for $\boldsymbol{\phi}_n = \phi(\mathbf{x}_n)$.

> [!NOTE]
> Recall that, crucially, we recover the standard Bernoulli likelihood by 
> marginalizing out the auxiliary variables
> $$
> \begin{align}
> p(y | \boldsymbol{\beta}) & =
> \int p(y | \omega, \boldsymbol{\beta}) p(\omega) \mathrm{d}\omega \newline & =
> \frac{e^{y \psi}}{1 + e^{\psi}} = 
> \sigma(\psi)^{y} (1-\sigma(\psi))^{1-y} \doteq 
> \mathrm{Bernoulli}(y | \psi)
> \end{align}
> $$

## Variational Inference

Approximate posterior $p(\boldsymbol{\beta}, \boldsymbol{\omega} | \mathbf{y})$

### Variational distribution
$$
q(\boldsymbol{\beta}, \boldsymbol{\omega}; \boldsymbol{\xi}) = 
q(\boldsymbol{\beta}) \prod_{n=1}^N q(\omega_n; \xi_n)
$$

variational distribution with local variational parameter $\xi$
$$
q(\omega_n; \xi_n) \doteq \mathrm{PG}(\omega_n | 1, \xi_n)
\doteq \cosh{\left(\frac{\xi_n}{2}\right)} \exp{\left(-\frac{\xi_n^2}{2}\omega_n\right)} \mathrm{PG}(\omega_n | 1, 0) 
$$

> [!NOTE]
> Note the first moment of the Polya-gamma $\mathrm{PG}(\omega | b, c)$ 
> distribution is related to the function $\lambda$ defined above
> $$
> \mathbb{E}_{\mathrm{PG}(\omega | b, c)}[\omega] = \frac{b}{2c} \tanh{\left(\frac{c}{2}\right)} = b \cdot \lambda(c).
> $$
> In particular, note that
> $$
> \mathbb{E}_{\mathrm{PG}(\omega | 1, c)}[\omega] = \lambda(c).
> $$

Minimize $\mathrm{KL}[q(\boldsymbol{\beta}, \boldsymbol{\omega}; \boldsymbol{\xi}) || p(\boldsymbol{\beta}, \boldsymbol{\omega} | \mathbf{y})]$ through maximization of the evidence lower bound (ELBO).

$$
\mathcal{L}(\xi_n) \doteq 
\mathbb{E}_{q(\boldsymbol{\beta}, \omega_n; \xi_n)}[\log{p(y_n | \omega_n, \boldsymbol{\beta})}] - 
\mathrm{KL}[q(\boldsymbol{\beta}, \omega_n; \xi_n) || p(\boldsymbol{\beta}, \omega_n)]
$$

We can write 

$$
\mathcal{L}(\xi_n) = \mathbb{E}_{q(\boldsymbol{\beta})}[
  \log{h(y_n, \xi_n, \boldsymbol{\beta})} + \log{p(\boldsymbol{\beta})} - 
  \log{q(\boldsymbol{\beta})}]
$$
where
$$
h(y, \xi, \boldsymbol{\beta}) \doteq \exp{H(y, \xi, \boldsymbol{\beta})}
$$
and
$$
H(y, \xi, \boldsymbol{\beta}) \doteq 
\mathbb{E}_{q(\omega;\xi)}[\log{p(y | \omega, \boldsymbol{\beta})}] - \mathrm{KL}[q(\omega;\xi) || p(\omega)]
$$


$$
\begin{align}
  H(y, \xi, \boldsymbol{\beta})
  &= \mathbb{E}_{q(\omega;\xi)}[\log{p(y | \omega, \boldsymbol{\beta})} + \log{p(\omega)} - \log{q(\omega;\xi)}] \newline
  &= \mathbb{E}_{q(\omega;\xi)}
  \left [ - \frac{\omega}{2} \left( \psi^2 - 2\psi\frac{\kappa}{\omega}\right) - \log{2}
          + \log{\mathrm{PG}(1, 0)} - \log{\mathrm{PG}(1, 0)} + \frac{\xi^2}{2} \omega - \log{\cosh{\left(\frac{\xi}{2}\right)}} 
  \right] \newline
  &= \mathbb{E}_{q(\omega;\xi)}
  \left [ - \frac{\omega}{2} \left( \psi^2 - 2\psi\frac{\kappa}{\omega}\right) 
          + \frac{\xi^2}{2} \omega + \log{\sigma(\xi)} - \frac{\xi}{2}
  \right ] \newline
  &= \mathbb{E}_{q(\omega;\xi)}
  \left [ y \psi - \frac{\psi + \xi}{2} - \frac{\omega}{2} (\psi^2 - \xi^2) + \log{\sigma(\xi)} \right ] \newline
  &= y \psi - \frac{\psi + \xi}{2}
    -\frac{1}{2} (\psi^2 - \xi^2) \mathbb{E}_{q(\omega;\xi)}[\omega]  
    + \log{\sigma(\xi)} \newline
  &= y \psi - \frac{\psi + \xi}{2}
    -\frac{\lambda(\xi)}{2} (\psi^2 - \xi^2)
    + \log{\sigma(\xi)} \newline
\end{align}
$$

$$
\begin{align}
h(y, \xi, \boldsymbol{\beta}) & = 
\sigma(\xi) \exp{\left(y \psi - \frac{\psi + \xi}{2} - \frac{\lambda(\xi)}{2} (\psi^2 - \xi^2)\right)} \newline & = 
e^{y \psi} \ell(-\psi, \xi) \newline & \leq 
e^{y \psi} \sigma(-\psi) = 
\sigma(\psi)^{y} (1- \sigma(\psi))^{1-y} = p(y | \boldsymbol{\beta}) 
\end{align}
$$

$$
\begin{align}
H(y, \xi, \boldsymbol{\beta}) & = 
  y \psi - g(\psi, \xi) \newline & \leq 
  y \psi + \log{\sigma(-\psi)} = 
  \log{p(y | \boldsymbol{\beta})}
\end{align}
$$

$$
p(y , \boldsymbol{\beta}) = p(y | \boldsymbol{\beta}) p(\boldsymbol{\beta}) 
\geq
h(y, \xi, \boldsymbol{\beta}) p(\boldsymbol{\beta}) 
$$

Let
$$
\boldsymbol{\Lambda} = \mathrm{diag}(\lambda(\xi_1) \cdots \lambda(\xi_N))
$$

$$
\begin{align}
\log{[p(y | \boldsymbol{\beta}) p(\boldsymbol{\beta})]} & \geq 
\log{[h(y, \xi, \boldsymbol{\beta}) p(\boldsymbol{\beta})]} \newline & =
\log{p(\boldsymbol{\beta})} + \sum_{n=1}^N \{ y_n \psi_n - \frac{\psi_n + \xi_n}{2} - \frac{\lambda(\xi_n)}{2} (\psi_n^2 - \xi_n^2) + \log{\sigma(\xi_n)} \} \newline & =
\log{p(\boldsymbol{\beta})} + \sum_{n=1}^N \{ \psi_n \kappa_n - \frac{\lambda(\xi_n)}{2} \psi_n^2 \} + \mathrm{const} \newline & =
\log{p(\boldsymbol{\beta})} + \sum_{n=1}^N \{ \boldsymbol{\beta}^{\top} \boldsymbol{\phi}_n \kappa_n - 
\frac{\lambda(\xi_n)}{2} \boldsymbol{\beta}^{\top} \left(\boldsymbol{\phi}_n \boldsymbol{\phi}_n^{\top}\right) \boldsymbol{\beta} \} + \mathrm{const} \newline & = - 
\frac{1}{2} (\boldsymbol{\beta} - \mathbf{m})^{\top} \mathbf{S}^{-1} (\boldsymbol{\beta} - \mathbf{m}) + \boldsymbol{\beta}^{\top} \boldsymbol{\Phi}^{\top} \kappa - 
\frac{1}{2} \boldsymbol{\beta}^{\top} \left(\boldsymbol{\Phi}^{\top} \boldsymbol{\Lambda} \boldsymbol{\Phi}\right) \boldsymbol{\beta} + \mathrm{const} \newline & = - 
\frac{1}{2} \boldsymbol{\beta}^{\top} \left(\mathbf{S}^{-1} + \boldsymbol{\Phi}^{\top} \boldsymbol{\Lambda} \boldsymbol{\Phi}\right) \boldsymbol{\beta} +
\boldsymbol{\beta}^{\top} \left(\mathbf{S}^{-1} \mathbf{m} + \boldsymbol{\Phi}^{\top} \kappa \right) + \mathrm{const} \newline & = - 
\frac{1}{2} 
\left(\boldsymbol{\beta} - \boldsymbol{\mu} \right)^{\top} 
\boldsymbol{\Sigma}^{-1}
\left(\boldsymbol{\beta} - \boldsymbol{\mu}\right) + \mathrm{const} 
\end{align}
$$
where
$$
\boldsymbol{\mu} = \boldsymbol{\Sigma} \left ( \mathbf{S}^{-1} \mathbf{m} + \boldsymbol{\Phi}^{\top} \boldsymbol{\kappa} \right )
\quad
\text{and}
\quad
\boldsymbol{\Sigma}^{-1} = \left( \mathbf{S}^{-1} + \boldsymbol{\Phi}^{\top} \boldsymbol{\Lambda} \boldsymbol{\Phi} \right).
$$
Hence, we adopt the following Gaussian approximation to the posterior
$$
q(\boldsymbol{\beta}; \boldsymbol{\xi}) = 
\mathcal{N}(\boldsymbol{\beta} | \boldsymbol{\mu}, \boldsymbol{\Sigma}).
$$


$$
\begin{align}
  H(y, \xi, \boldsymbol{\beta}) & = - \frac{\lambda(\xi)}{2} \left ( \psi^2 - 2 \psi \frac{\kappa}{\lambda(\xi)} \right ) + 
     \frac{\lambda(\xi)}{2} \xi^2 - \frac{\xi}{2} + \log{\sigma(\xi)} \newline
  &= - \frac{\lambda(\xi)}{2} \left( \psi - \frac{\kappa}{\lambda(\xi)} \right)^2 + C(y, \xi)
\end{align}
$$

$$
h(y, \xi, \boldsymbol{\beta}) \propto 
\exp{\left\{-\frac{\lambda(\xi)}{2}\left(\psi - \frac{\kappa}{\lambda(\xi)}\right)^2\right\}}
$$


### Prior over auxiliary variables

Second, let us define a prior over auxiliary variables $\boldsymbol{\omega}$ that 
factorize as
$$
p(\boldsymbol{\omega}) = \prod_{n=1}^N p(\omega_n) 
$$
where each factor $p(\omega_n)$ is a Pólya-gamma density
$$
p(\omega_n) = \mathrm{PG}(\omega_n | 1, 0),
$$
defined as an infinite [convolution](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions#See_also) of gamma distributions :

> [!NOTE]
> #### Pólya-gamma density (Polson et al. 2013)
>
> A random variable $\omega$ has a Pólya-gamma distribution with parameters $b > 0$ 
> and $c \in \mathbb{R}$, denoted $\omega \sim \mathrm{PG}(b, c)$, if
> $$
> \mathrm{PG}(b, c) = \frac{1}{2 \pi^2} \sum_{k=1}^{\infty} 
> \frac{g_k}{\left (k - \frac{1}{2} \right )^2 + \left ( \frac{c}{2\pi} \right )^2}
> $$
> where the $g_k \sim \mathrm{Ga}(b, 1)$ are independent gamma random variables.

#### Property I: Recovering the original model

First we show that we can recover the original likelihood $p(y_n | f_n)$ 
by integrating out $\boldsymbol{\omega}$.
Before we proceed, note that the $p(y_n | f_n)$ can be expressed more 
succinctly as
$$
p(y_n | f_n) = \frac{e^{y_n f_n}}{1 + e^{f_n}}.
$$
Refer to [Appendix I]({{< relref "#i" >}}) for derivations.
Next, note the following property of Pólya-gamma variables:

> [!NOTE]
> #### Laplace transform of the Pólya-gamma density (Polson et al. 2013)
>
> Based on the [Laplace transform](https://mathworld.wolfram.com/LaplaceTransform.html) 
> of the Pólya-gamma density function, we can derive the following relationship:
> $$
> \frac{\left (e^{u} \right )^a}{\left (1 + e^{u} \right )^b} = 
> \frac{1}{2^b} \exp{(\kappa u)} \
> \int_0^\infty \exp{\left ( - \frac{u^2}{2} \omega \right )} 
> p(\omega) d\omega,
> $$
> where $\kappa = a - \frac{b}{2}$ and $p(\omega) = \mathrm{PG}(\omega | b, 0)$.

Therefore, by substituting $\kappa = \kappa_n, a = y_n, b = 1$ and $u = f_n$ 
we get
$$
\begin{align}
\int p(y_n, \omega_n | f_n) d\omega_n &=
\int p(y_n | f_n, \omega_n) p(\omega_n) d\omega_n \newline &= 
\frac{1}{2} \int \exp{\left \{ - \frac{\omega_n}{2} \left (f_n^2 - 
                             2 f_n \frac{\kappa_n}{\omega_n} \right ) \right \}} p(\omega_n) d\omega_n \newline &= 
\frac{1}{2} \exp{(\kappa_n f_n)} 
\int \exp{\left ( - \frac{f_n^2}{2} \omega_n \right )} p(\omega_n) d\omega_n \newline &= 
\frac{\left (e^{f_n} \right )^{y_n}}{1 + e^{f_n}} = p(y_n | f_n)
\end{align}
$$
as required.

#### Property II: Gaussian-Gaussian conjugacy

Let us define the diagonal matrix $\boldsymbol{\Omega} = \mathrm{diag}(\omega_1 \cdots \omega_n)$ and vector $\mathbf{z} = \boldsymbol{\Omega}^{-1} \boldsymbol{\kappa}$. 
More simply, $\mathbf{z}$ is the vector with $n$th element $z_n = {\kappa_n} / {\omega_n}$.
Hence, by [completing the square](https://mathworld.wolfram.com/CompletingtheSquare.html), 
the per-datapoint conditional likelihood $p(y_n | f_n, \omega_n)$ above can be written as
$$
\begin{align}
p(y_n | f_n, \omega_n) & \propto
\exp{\left \{ - \frac{\omega_n}{2} \left (f_n - \frac{\kappa_n}{\omega_n} \right )^2 \right \}} \newline & = \exp{\left \{ - \frac{\omega_n}{2} \left (f_n - z_n \right )^2 \right \}}
\end{align}
$$
Importantly, this implies that the conditional likelihood over all 
variables $p(\mathbf{y} | \mathbf{f}, \boldsymbol{\omega})$ is simply a 
multivariate Gaussian distribution up to a constant factor
$$
p(\mathbf{y} | \mathbf{f}, \boldsymbol{\omega}) \propto \mathcal{N}\left (\boldsymbol{\Omega}^{-1} \boldsymbol{\kappa} | \mathbf{f}, \boldsymbol{\Omega}^{-1} \right ).
$$
Refer to [Appendix II]({{< relref "#ii" >}}) for derivations.
Therefore, a Gaussian prior $p(\mathbf{f})$ is conjugate to the 
conditional likelihood $p(\mathbf{y} | \mathbf{f}, \boldsymbol{\omega})$, which 
leads to $p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega})$, the posterior 
over $\mathbf{f}$ conditioned on the auxiliary latent 
variables $\boldsymbol{\omega}$, also being a Gaussian---a property that will 
prove crucial to us in the next section.

### Inference (Gibbs sampling)

We wish to compute the posterior 
distribution $p(\mathbf{f}, \boldsymbol{\omega} | \mathbf{y})$, the 
distribution over the hidden variables $(\mathbf{f}, \boldsymbol{\omega})$ 
conditioned on the observed variables $\mathbf{y}$.
To produce samples from this distribution 
$$
(\mathbf{f}^{(t)}, \boldsymbol{\omega}^{(t)}) \sim p(\mathbf{f}, \boldsymbol{\omega} | \mathbf{y}),
$$
we can readily apply Gibbs sampling[^geman1984stochastic], an MCMC 
algorithm that can be seen as a special case of the Metropolis-Hastings algorithm.

Each step of the Gibbs sampling procedure involves replacing the value of one 
of the variables by a value drawn from the distribution of that variable 
conditioned on the values of the remaining variables.
Specifically, we proceed as follows. 
At step $t$, we have values $\mathbf{f}^{(t-1)}, \boldsymbol{\omega}^{(t-1)}$
sampled from the previous step. 

1. We first replace $\mathbf{f}^{(t-1)}$ by a new
value $\mathbf{f}^{(t)}$ by sampling from the conditional distribution $p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega}^{(t-1)})$,
$$
\mathbf{f}^{(t)} \sim p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega}^{(t-1)}).
$$
2. Then we replace $\boldsymbol{\omega}^{(t-1)}$ by $\boldsymbol{\omega}^{(t)}$ by sampling 
from the conditional distribution $p(\boldsymbol{\omega}| \mathbf{f}^{(t)})$,
$$
\boldsymbol{\omega}^{(t)} \sim p(\boldsymbol{\omega}| \mathbf{f}^{(t)}),
$$
where we've used $\mathbf{f}^{(t)}$, the new value for $\mathbf{f}$ from step 1, 
straight away in the current step. Note that we've dropped the conditioning 
on $\mathbf{y}$, since $\boldsymbol{\omega}$ does not *a posteriori* depend 
on this variable.

We then proceed in like manner, cycling between the two variables in turn until 
some convergence criterion is met.

Suffice it to say, this requires us to first compute the conditional 
posteriors $p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega})$ 
and $p(\boldsymbol{\omega}| \mathbf{f})$, the calculation of which will be the
subject of the next two subsections.

#### Posterior over latent function values

The posterior over the latent function values $\mathbf{f}$ conditioned on the 
auxiliary latent variables $\boldsymbol{\omega}$ is
$$
p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega}) = \mathcal{N}(\mathbf{f} | \boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$
where
$$
\boldsymbol{\mu} = \boldsymbol{\Sigma} \left ( \mathbf{S} \mathbf{m} + \boldsymbol{\kappa} \right )
\quad
\text{and}
\quad
\boldsymbol{\Sigma} = \left (\mathbf{S} + \boldsymbol{\Omega} \right )^{-1}.
$$

We readily obtain $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ by noting,
as alluded to earlier, that 
$$
p(\mathbf{f}) = \mathcal{N}(\mathbf{m}, \mathbf{S}^{-1}),
\qquad
\text{and}
\qquad
p(\mathbf{y} | \mathbf{f}, \boldsymbol{\omega}) \propto \mathcal{N}\left (\boldsymbol{\Omega}^{-1} \boldsymbol{\kappa} | \mathbf{f}, \boldsymbol{\Omega}^{-1} \right ).
$$
Thereafter, we can appeal to the following elementary properties of Gaussian 
conditioning and perform some pattern-matching substitutions:

> [!NOTE]
> #### Marginal and Conditional Gaussians (Bishop, Section 2.3.3, pg. 93)
>
> Given a marginal Gaussian distribution for $\mathbf{b}$ and a conditional Gaussian 
> distribution for $\mathbf{a}$ given $\mathbf{b}$ in the form
>
> $$
> \begin{align}
> p(\mathbf{b}) & = 
> \mathcal{N}(\mathbf{b} | \mathbf{m}, \mathbf{S}^{-1}) \newline
> p(\mathbf{a} | \mathbf{b}) & = 
> \mathcal{N}(\mathbf{a} | \mathbf{W} \mathbf{b}, \boldsymbol{\Psi}^{-1})
> \end{align}
> $$
> the marginal distribution of $\mathbf{a}$ and the conditional distribution 
> of $\mathbf{b}$ given $\mathbf{a}$ are given by
> \begin{align}
> p(\mathbf{a}) & = 
> \mathcal{N}(\mathbf{a} | \mathbf{W} \mathbf{m}, \boldsymbol{\Psi}^{-1} + \mathbf{W} \mathbf{S}^{-1} \mathbf{W}^{\top}) \newline
> p(\mathbf{b} | \mathbf{a}) & = 
> \mathcal{N}(\mathbf{b} | \boldsymbol{\mu}, \boldsymbol{\Sigma})
> \end{align}
> where
> $$
> \boldsymbol{\mu} = \boldsymbol{\Sigma} \left ( \mathbf{W}^{\top} \boldsymbol{\Psi} \mathbf{a} + \mathbf{S} \mathbf{m} \right ),
> \quad
> \text{and}
> \quad
> \boldsymbol{\Sigma} = \left (\mathbf{S} + \mathbf{W}^{\top} \boldsymbol{\Psi} \mathbf{W}\right )^{-1}.
> $$

Note that we also could have derived this directly without resorting to 
the formulae above by reducing the product of two exponential-quadratic 
functions in $p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega}) \propto p(\mathbf{y} | \mathbf{f}, \boldsymbol{\omega}) p(\mathbf{f})$ into a single exponential-quadratic function
up to a constant factor. 
It would, however, have been rather tedious and mundane.

> [!NOTE]
> #### Example: Gaussian process prior
>
> To make this more concrete, let us revisit the Gaussian process prior we 
> discussed earlier, namely,
> $$
> p(\mathbf{f} | \mathbf{X}) = \mathcal{N}(\mathbf{m}, \mathbf{K}_X).
> $$
> By substituting $\mathbf{S}^{-1} = \mathbf{K}_X$ from before, we obtain
> $$
> p(\mathbf{f} | \mathbf{y}, \boldsymbol{\omega}) = 
> \mathcal{N}(\mathbf{f} | \boldsymbol{\Sigma} \left ( \mathbf{K}_X^{-1} \mathbf{m} + \boldsymbol{\kappa} \right ), \boldsymbol{\Sigma}),
> $$
> where $\boldsymbol{\Sigma} = \left (\mathbf{K}_X^{-1} + \boldsymbol{\Omega} \right )^{-1}.$

#### Posterior over auxiliary variables

The posterior over the auxiliary latent variables $\boldsymbol{\omega}$ 
conditioned on the latent function values $\mathbf{f}$ factorizes as
$$
p(\boldsymbol{\omega}| \mathbf{f}) = \prod_{n=1}^{N} p(\omega_n | f_n),
$$
where each factor
$$
p(\omega_n | f_n) = 
\frac{p(f_n, \omega_n)}{\int p(f_n, \omega_n) d\omega_n} \propto 
p(f_n, \omega_n).
$$
Now, the joint factorizes as $p(f_n, \omega_n) = p(f_n | \omega_n) p(\omega_n)$ where
$$
p(f_n | \omega_n) = \exp{\left (-\frac{f_n^2}{2}\omega_n \right )},
\quad
\text{and}
\quad
p(\omega_n) = \mathrm{PG}(\omega_n | 1, 0).
$$
Finally, by the [exponential-tilting property](https://en.wikipedia.org/wiki/Exponential_tilting) 
of the Pólya-gamma distribution, we have
$$
p(f_n, \omega_n) = 
\mathrm{PG}(\omega_n | 1, 0) \times
\exp{\left (-\frac{f_n^2}{2}\omega_n \right )} = 
\mathrm{PG}(\omega_n | 1, f_n).
$$
Hence, all in all, we have
$$
p(\omega_n | f_n) \propto \mathrm{PG}(\omega_n | 1, f_n).
$$
We have omitted the normalizing constant $\int p(f_n, \omega_n) d\omega_n$
from our discussion for the sake of brevity since it is not required to carry 
out inference using Gibbs sampling.
If you're interested in calculating it, 
refer to [Appendix III]({{< relref "#iii" >}}).

## Implementation (Weight-space view)

Having presented the general form of an augmented model for Bayesian logistic
regression, we now derive a simple instance of this model to tackle a synthetic
one-dimensional classification problem.
In this particular implementation, we make the following choices: 
(a) we incorporate a basis function to project inputs into a higher-dimensional feature space, and 
(b) we consider an isotropic Gaussian prior on the weights.

### Synthetic one-dimensional classification problem 

First we synthesize a one-dimensional classification problem for which 
the *true* class-membership probability $p(y = 1 | x)$ is both known and easy 
to compute.
To this end, let us introduce the following one-dimensional Gaussians,
$$
p(x) = \mathcal{N}(1, 1^2),
\qquad
\text{and}
\qquad
q(x) = \mathcal{N}(0, 2^2).
$$

In code we can specify these as:

```python
from scipy.stats import norm

p = norm(loc=1.0, scale=1.0)
q = norm(loc=0.0, scale=2.0)
```

We evenly draw a total of $N$ samples from both distributions:

```python
>>> X_p, X_q = draw_samples(num_train, p, q, rate=0.5, random_state=random_state)
```

where the function `draw_samples` is defined as:

```python
def draw_samples(num_samples, p, q, rate=0.5, random_state=None):

    num_top = int(num_samples * rate)
    num_bot = num_samples - num_top

    X_top = p.rvs(size=num_top, random_state=random_state)
    X_bot = q.rvs(size=num_bot, random_state=random_state)

    return X_top, X_bot
```

The densities of both distributions and their and samples are shown in the 
figure below.

{{< figure src="figures/density_paper_1500x927.png" title="Densities of two Gaussians and samples drawn from each." numbered="true" >}}

From these samples, let us now construct a classification dataset by assigning 
label $y = 1$ to inputs $x \sim p(x)$, and $y = 0$ to inputs $x \sim q(x)$.

```python
>>> X_train, y_train = make_dataset(X_p, X_q)
```

where the function `make_dataset` is defined as:


```python
def make_dataset(X_pos, X_neg):

    X = np.expand_dim(np.hstack([X_pos, X_neg]), axis=-1)
    y = np.hstack([np.ones_like(X_pos), np.zeros_like(X_neg)])

    return X, y
```

Crucially, the true class-membership probability is given exactly by
$$
p(y = 1 | x) = \frac{p(x)}{p(x) + q(x)},
$$
thus providing a ground-truth yardstick by which to measure the quality of our
resulting predictions.

The class-membership probability $p(y = 1 | x)$ is shown in the figure below as
the black curve, along with the dataset $\mathcal{D}_N = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$
where positive instances are colored red and negative instances are colored blue.

{{< figure src="figures/class_prob_true_paper_1500x927.png" title="Classification dataset $\mathcal{D}_N = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ and the true class-posterior probability." numbered="true" >}}

### Prior 

To increase the flexibility of our model, we introduce a basis 
function $\phi: \mathbb{R}^{D} \to \mathbb{R}^{K}$ that projects 
$D$-dimensional input vectors into a $K$-dimensional vector space. 
Accordingly, we introduce matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times K}$ 
such that the $n$th column of $\boldsymbol{\Phi}^{\top}$ consists of the 
vector $\phi(\mathbf{x}_n)$.
Hence, we assume *a priori* that the latent function is of the form  
$$
f(\mathbf{x}) = \boldsymbol{\beta}^{\top} \phi(\mathbf{x}),
$$
and express vector of latent function values 
as $\mathbf{f} = \boldsymbol{\Phi} \boldsymbol{\beta}$.
In this example, we consider a simply polynomial basis function,
$$
\phi(x) = \begin{bmatrix} 1 & x & x^2 & \cdots & x^{K-1} \end{bmatrix}^{\top}.
$$

Therefore, we call:

```python
>>> Phi = basis_function(X_train, degree=degree)
```

where the function `basis_function` is defined as:

```python
def basis_function(x, degree=3):
    return np.power(x, np.arange(degree))
```

Let us define 
the prior over weights as a simple isotropic Gaussian with 
precision $\alpha > 0$,
$$
p(\boldsymbol{\beta}) = \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I}),
$$
and the prior over each local auxiliary latent variable as before,
$$
p(\omega_n) = \mathrm{PG}(\omega_n | 1, 0).
$$
Since we have analytic forms for the conditional posteriors, we don't need to
implement the priors explicitly. 
However, in order to initialize the Gibbs sampler, we may want to be able to 
sample from the prior.
Let us do this using the prior over weights:


```python
m = np.zeros(latent_dim)

alpha = 2.0  # prior precision
S_inv = np.eye(latent_dim) / alpha

# initialize `beta`
beta = random_state.multivariate_normal(mean=m, cov=S_inv)
```

or more simply:

```python
alpha = 2.0  # prior precision

# initialize `beta`
beta = random_state.normal(size=latent_dim, scale=1/np.sqrt(alpha))
```

### Conditional likelihood

The conditional likelihood is defined like before, except we instead 
condition on weights $\boldsymbol{\beta}$ and substitute occurrences 
of $\mathbf{f}$ with $\boldsymbol{\Phi} \boldsymbol{\beta}$,
$$
p(\mathbf{y} | \boldsymbol{\beta}, \boldsymbol{\omega}) \propto \mathcal{N}\left (\boldsymbol{\Omega}^{-1} \boldsymbol{\kappa} | \boldsymbol{\Phi} \boldsymbol{\beta}, \boldsymbol{\Omega}^{-1} \right ).
$$

### Inference and Prediction  

#### Posterior over latent function values

The posterior over the latent weights $\boldsymbol{\beta}$ conditioned on the 
auxiliary latent variables $\boldsymbol{\omega}$ is
$$
p(\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\omega}) = \mathcal{N}(\boldsymbol{\beta} | \boldsymbol{\Sigma} \boldsymbol{\Phi}^{\top} \boldsymbol{\kappa}, \boldsymbol{\Sigma}),
$$
where 
$$
\boldsymbol{\Sigma} = \left (\boldsymbol{\Phi}^{\top} \boldsymbol{\Omega} \boldsymbol{\Phi} + \alpha \mathbf{I} \right )^{-1}.
$$

Let us implement the function that computes the mean and covariance 
of $p(\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\omega})$:

```python
def conditional_posterior_weights(Phi, kappa, alpha, omega):

    latent_dim = Phi.shape[-1]

    Sigma_inv = (omega * Phi.T) @ Phi + alpha * np.eye(latent_dim)

    mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
    Sigma = np.linalg.solve(Sigma_inv, np.eye(latent_dim))

    return mu, Sigma
```

and a function to return samples from the multivariate Gaussian parameterized
by this mean and covariance:

```python
def gassian_sample(mean, cov, random_state=None):
    random_state = check_random_state(random_state)
    return random_state.multivariate_normal(mean=mean, cov=cov)
```

#### Posterior over auxiliary variables

The conditional posterior over the local auxiliary variable $\omega_n$ is 
defined as before, except we instead condition on weights $\boldsymbol{\beta}$ 
and substitute occurrences of $f_n$ with $\boldsymbol{\beta}^{\top} \phi(\mathbf{x}_n)$,
$$
p(\omega_n | \boldsymbol{\beta}) \propto 
\mathrm{PG}(\omega_n | 1, \boldsymbol{\beta}^{\top} \phi(\mathbf{x}_n)).
$$

Let us implement a function to compute the parameters of the posterior 
Polya-gamma distribution:

```python
def conditional_posterior_auxiliary(Phi, beta):
    c = Phi @ beta
    b = np.ones_like(c)
    return b, c
```

and accordingly a function to return samples from this distribution:

```python
def polya_gamma_sample(b, c, pg=PyPolyaGamma()):
    assert b.shape == c.shape, "shape mismatch"
    omega = np.empty_like(b)
    pg.pgdrawv(b, c, omega)
    return omega
```

where we have imported the `PyPolyaGamma` object from 
the [pypolyagamma](https://github.com/slinderman/pypolyagamma) package:

```
from pypolyagamma import PyPolyaGamma
```

The `pypolyagamma` package can be installed via `pip` as usual: 

```bash
$ pip install pypolyagamma
```

To provide some context, this package is a [Cython](https://cython.org/) 
port, created by S. Linderman, of the original 
R package [BayesLogit](https://github.com/jwindle/BayesLogit) authored by J. Windle
that implements the method described in their paper on the efficient sampling 
of Pólya-gamma variables[^windle2014sampling].

#### Gibbs sampling

With these functions defined, we can define the Gibbs sampling procedure by the
simple for-loop below:

```python
# preprocessing
kappa = y_train - 0.5
Phi = basis_function(X_train, degree=degree)

# initialize `beta`
latent_dim = Phi.shape[-1]
beta = random_state.normal(size=latent_dim, scale=1/np.sqrt(alpha))

for i in range(num_iterations):

    b, c = conditional_posterior_auxiliary(Phi, beta)
    omega = polya_gamma_sample(b, c, pg=pg)

    mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, omega)
    beta = gassian_sample(mu, Sigma, random_state=random_state)
```

We now visualize the samples $(\boldsymbol{\beta}^{(t)}, \boldsymbol{\omega}^{(t)})$ 
produced by this procedure. 
In the figures that follow, we set the hues to be proportional to the step 
counter $t$ along a perceptually uniform colormap.

First, we show the sampled weight vector $\boldsymbol{\beta}^{(t)} \in \mathbb{R}^K$ 
where we have set $K = 3$.
We plot the $i$th entry $\beta_i^{(t)}$ against the $j$th entry $\beta_j^{(t)}$ 
for all $i < j$ and $0 < j < K$.
{{< figure src="figures/beta_paper_600x600.png" title="Parameter $\boldsymbol{\beta}^{(t)}$ samples as Gibbs sampling iteration $t$ increases." numbered="true" >}}
We find a strong correlation between $\beta_1$ and $\beta_2$, the 
coefficients associated with the linear and quadratic terms of our augmented 
feature representation, respectively. 
Furthermore, we find $\beta_1$ to consistently have a relatively large 
magnitude.

Second, we show the sampled auxiliary latent variables $\boldsymbol{\omega}^{(t)}$ by 
plotting the pairs $(x_n, \omega_n^{(t)})$.

{{< figure src="figures/omega_paper_1500x927.png" title="Auxiliary variable $\omega_n^{(t)}$ samples as Gibbs sampling iteration $t$ increases. For visualization purposes, each $\omega_n^{(t)}$ is placed at its corresponding input location $x_n$ along the  horizontal axis." numbered="true" >}}

As expected, we find longer-tailed distributions in the variables $\omega_n$ 
that are associated with negative examples.

Finally, we plot the sampled class-membership probability predictions
$$
\pi^{(t)}(\mathbf{x}) = \sigma(f^{(t)}(\mathbf{x})),
\quad
\text{where}
\quad
f^{(t)}(\mathbf{x}) = {\boldsymbol{\beta}^{(t)}}^{\top} \phi(\mathbf{x}),
$$
in the figure below:

{{< figure src="figures/class_prob_pred_paper_1500x927.png" title="Predicted class-membership probability $\pi^{(t)}(\mathbf{x})$ as Gibbs sampling iteration $t$ increases." numbered="true" >}}

At least qualitatively, we find that the sampling procedure produces 
predictions that fit the true class-membership probability reasonably well.

### Code

The full code is reproduced below:

```python
import numpy as np

from scipy.stats import norm
from pypolyagamma import PyPolyaGamma

from .utils import (draw_samples, make_dataset, basis_function,
                    conditional_posterior_auxiliary, polya_gamma_sample,
                    conditional_posterior_weights, gassian_sample)

# constants
num_train = 128
num_iterations = 1000
degree = 3
alpha = 2.0  # prior precision

seed = 8888
random_state = np.random.RandomState(seed)
pg = PyPolyaGamma(seed=seed)

# generate dataset
p = norm(loc=1.0, scale=1.0)
q = norm(loc=0.0, scale=2.0)

X_p, X_q = draw_samples(num_train, p, q, rate=0.5, random_state=random_state)
X_train, y_train = make_dataset(X_p, X_q)

# preprocessing
kappa = y_train - 0.5
Phi = basis_function(X_train, degree=degree)

# initialize `beta`
latent_dim = Phi.shape[-1]
beta = random_state.normal(size=latent_dim, scale=1/np.sqrt(alpha))

for i in range(num_iterations):

    b, c = conditional_posterior_auxiliary(Phi, beta)
    omega = polya_gamma_sample(b, c, pg=pg)

    mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, omega)
    beta = gassian_sample(mu, Sigma, random_state=random_state)
```

where the module `utils.py` contains:

```python
import numpy as np
from sklearn.utils import check_random_state
from pypolyagamma import PyPolyaGamma


def draw_samples(num_samples, p, q, rate=0.5, random_state=None):
    num_top = int(num_samples * rate)
    num_bot = num_samples - num_top

    X_top = p.rvs(size=num_top, random_state=random_state)
    X_bot = q.rvs(size=num_bot, random_state=random_state)
    return X_top, X_bot


def make_dataset(X_pos, X_neg):
    X = np.expand_dims(np.hstack([X_pos, X_neg]), axis=-1)
    y = np.hstack([np.ones_like(X_pos), np.zeros_like(X_neg)])
    return X, y


def basis_function(x, degree=3):
    return np.power(x, np.arange(degree))


def polya_gamma_sample(b, c, pg=PyPolyaGamma()):
    assert b.shape == c.shape, "shape mismatch"
    omega = np.empty_like(b)
    pg.pgdrawv(b, c, omega)
    return omega


def gassian_sample(mean, cov, random_state=None):
    random_state = check_random_state(random_state)
    return random_state.multivariate_normal(mean=mean, cov=cov)


def conditional_posterior_weights(Phi, kappa, alpha, omega):
    latent_dim = Phi.shape[-1]
    eye = np.eye(latent_dim)

    Sigma_inv = (omega * Phi.T) @ Phi + alpha * eye

    mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
    Sigma = np.linalg.solve(Sigma_inv, eye)
    return mu, Sigma


def conditional_posterior_auxiliary(Phi, beta):
    c = Phi @ beta
    b = np.ones_like(c)
    return b, c
```

### Bonus: Gibbs sampling with mutual recursion and generator delegation

The Gibbs sampling procedure naturally lends itself to implementations based
on [mutual recursion](https://en.wikipedia.org/wiki/Mutual_recursion).
Combining this with the `yield from` expression for [generator delegation](https://docs.python.org/3/whatsnew/3.3.html#pep-380),
we can succinctly replace the for-loop with the following mutually recursive 
functions:
```python
def gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state):
    b, c = conditional_posterior_auxiliary(Phi, beta)
    omega = polya_gamma_sample(b, c, pg=pg)
    yield from gibbs_sampler_helper(omega, Phi, kappa, alpha, pg, random_state)


def gibbs_sampler_helper(omega, Phi, kappa, alpha, pg, random_state):
    mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, omega)
    beta = gassian_sample(mu, Sigma, random_state=random_state)
    yield beta, omega
    yield from gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state)
```

Now you can use `gibbs_sampler` as a [generator](https://wiki.python.org/moin/Generators), 
for example, to explicitly iterate over it in a for-loop:
```python
for beta, omega in gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state):

    if stop_predicate:
        break

    # do something
    pass
```
or by making use of [itertools](https://docs.python.org/3/library/itertools.html) 
and other [functional programming](https://docs.python.org/3/howto/functional.html) 
primitives:
```python
from itertools import islice

# example: collect beta and omega samples into respective lists
betas, omegas = zip(*islice(gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state), num_iterations))
```
There are a few obvious drawbacks to this implementation. 
First, while it may be a lot fun to write, it will probably not be as fun to 
read when you revisit it later on down the line.
Second, you may occasionally find yourself hitting the maximum recursion depth 
before you have reached a sufficient number of iterations for the warm-up 
or "burn-in" phase to have been completed.
It goes without saying, the latter can make this implementation a non-starter.

## Links and Further Readings

- Papers:
  * Original paper (Polson et al., 2013)[^polson2013bayesian]
  * Extended to GP classification (Wenzel et al., 2019)[^wenzel2019efficient]
  * Few-shot classification with GPs and the one-vs-each likelihood (Snell et al., 2020)[^snell2020bayesian]
- Blog posts: 
  * [Pólya-Gamma Augmentation](https://gregorygundersen.com/blog/2019/09/20/polya-gamma/) by G. Gundersen
- Code:
  * [pypolyagamma](https://github.com/slinderman/pypolyagamma): A Python package by S. Linderman
  * [BayesLogit](https://github.com/jwindle/BayesLogit): An R package by J. Windle

---

Cite as:

```
@article{tiao2021polyagamma,
  title   = "{A} {P}rimer on {P}ólya-gamma {R}andom {V}ariables - {P}art II: {B}ayesian {L}ogistic {R}egression",
  author  = "Tiao, Louis C",
  journal = "tiao.io",
  year    = "2021",
  url     = "https://tiao.io/post/polya-gamma-bayesian-logistic-regression/"
}
```

To receive updates on more posts like this, follow me on [Twitter] and [GitHub]!

## Appendix

### I

First, note that the logistic function can be written as
$$
\sigma(u) = \frac{1}{1+e^{-u}} = \frac{e^u}{1+e^u}
$$
Therefore, we have
$$
\begin{align}
p(y_n | f_n) &= 
\left ( \frac{e^{f_n}}{1+e^{f_n}} \right )^{y_n}
\left ( \frac{\left (1+e^{f_n} \right ) - e^{f_n}}{1+e^{f_n}} \right )^{1-y_n} \newline &= 
\left ( \frac{e^{f_n}}{1+e^{f_n}} \right )^{y_n}
\left ( \frac{1}{1+e^{f_n}} \right )^{1-y_n} \newline &= 
\left (e^{f_n} \right )^{y_n} \left ( \frac{1}{1+e^{f_n}} \right )^{y_n}
\left ( \frac{1}{1+e^{f_n}} \right )^{1-y_n} \newline &= 
\frac{e^{y_n f_n}}{1 + e^{f_n}}
\end{align}
$$

[Twitter]: https://twitter.com/louistiao
[GitHub]: https://github.com/ltiao

[^polson2013bayesian]: Polson, N. G., Scott, J. G., & Windle, J. (2013). [Bayesian Inference for Logistic Models using Pólya–Gamma Latent Variables](https://arxiv.org/abs/1205.0310). Journal of the American Statistical Association, 108(504), 1339-1349.
[^windle2014sampling]: Windle, J., Polson, N. G., & Scott, J. G. (2014). [Sampling Pólya-gamma Random Variates: Alternate and Approximate Techniques](https://arxiv.org/abs/1405.0506). arXiv preprint arXiv:1405.0506.
[^wenzel2019efficient]: Wenzel, F., Galy-Fajou, T., Donner, C., Kloft, M., & Opper, M. (2019, July). [Efficient Gaussian Process Classification using Pòlya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383). In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 5417-5424).
[^snell2020bayesian]: Snell, J., & Zemel, R. (2020). [Bayesian Few-Shot Classification with One-vs-Each Pólya-Gamma Augmented Gaussian Processes](https://arxiv.org/abs/2007.10417). arXiv preprint arXiv:2007.10417.

[^jaakkola2000bayesian]: Jaakkola, T. S., & Jordan, M. I. (2000). [Bayesian Parameter Estimation via Variational Methods](https://link.springer.com/article/10.1023/A:1008932416310). Statistics and Computing, 10(1), 25-37.
[^bouchard2007efficient]: Bouchard, G. (2007). Efficient Bounds for the Softmax Function, Applications to Inference in Hybrid Models. In Presentation at the Workshop for Approximate Bayesian Inference in Continuous/Hybrid Systems at NIPS2007.