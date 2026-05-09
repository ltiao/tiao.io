---
aliases:
  - /post/polya-gamma-basic-relationships/
  # Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "A Primer on Pólya-gamma Random Variables - Part I: Basic Relationships"
subtitle: ""
summary: ""
authors:
- me
tags:
- Machine Learning
- Bayesian Statistics
- Probabilistic Models
- Pólya-Gamma
categories: []
date: 2021-03-04T17:20:53+01:00
lastmod: 2021-03-04T17:20:53+01:00
featured: false
draft: true
math: true

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
> Draft — work in progress. This is **Part I** of a three-part series on
> Pólya-Gamma random variables. See also:
> [Part II — Bayesian Logistic Regression]({{< relref "/posts/polya-gamma-bayesian-logistic-regression" >}})
> and [Part III — Local Variational Methods]({{< relref "/posts/polya-gamma-sigmoid-local-variational-lower-bound" >}}).

A classical problem in Bayesian statistics is inference in Bayesian logistic 
regression, logit link is non-conjugate with a Gaussian prior. 
This is an analytical tractability we now seem to simply accept and take for 
granted.

References[^polson2013bayesian] [^wenzel2019efficient]

The ubiquitous logistic sigmoid function:
$$
\sigma(u) = \frac{1}{1 + \exp{(-u)}}
$$ 

{{< figure src="figures/sigmoid_paper_1500x927.png" title="Logistic sigmoid function." numbered="true" >}}

The hyperbolic cosine function
$$
\cosh(u) = \frac{e^{u} + e^{-u}}{2}
$$

{{< figure src="figures/cosh_paper_1500x927.png" title="Hyperbolic cosine function." numbered="true" >}}

### Identity 1

Express the logistic function in terms of the hyperbolic cosine function:
$$
\sigma(u) = \frac{e^{\frac{u}{2}}}{2\cosh(\frac{u}{2})}
$$ 
This might be immediately obvious for some, but it wasn't for me.

Here is the approach I took to derive it.

First, note that
$$
\frac{1}{2\cosh(\frac{u}{2})} = \frac{1}{e^{\frac{u}{2}} + e^{-\frac{u}{2}}} 
$$
Multiplying by $1 = \frac{e^{\frac{u}{2}}}{e^{\frac{u}{2}}}$ gives
$$
\sigma(u) = \frac{e^{\frac{u}{2}}}{e^{\frac{u}{2}}} \frac{1}{1 + e^{-u}} 
  \= \frac{e^{\frac{u}{2}}}{e^{\frac{u}{2}} + e^{-\frac{u}{2}}}
  \= \frac{e^{\frac{u}{2}}}{2\cosh(\frac{u}{2})}
$$ 

### Identity 2

Laplace transform
$$
\mathbb{E}\_{\mathrm{PG}(\omega | b, 0)}[\exp{(- \omega t)}] = \frac{1}{\cosh^b{\left (\sqrt {\frac{t}{2}} \right )}}
$$

Setting $b=1$ and
$$
t = \frac{u^2}{2}
\Leftrightarrow
\frac{t}{2} = \left ( \frac{u}{2} \right )^2
\Leftrightarrow
\sqrt{\frac{t}{2}} = \frac{u}{2} 
$$
gives
$$
\mathbb{E}\_{\mathrm{PG}(\omega | 1, 0)} 
\left [\exp{ \left (- \frac{u^2}{2} \omega \right )} \right ] = \frac{1}{\cosh{\left ( \frac{u}{2} \right )}}
$$
Hence, we have
$$
\sigma(u) = \frac{e^{\frac{u}{2}}}{2 \cosh{\left ( \frac{u}{2} \right )}} =
\frac{1}{2} \mathbb{E}\_{\mathrm{PG}(\omega | 1, 0)} 
\left [\exp{ \left (\frac{u}{2} - \frac{u^2}{2} \omega \right )} \right ] 
$$

$$
p(u, \omega) = p(u | \omega) p(\omega)
$$

$$
p(u | \omega) = \frac{1}{2} \exp{ \left (\frac{u}{2} - \frac{u^2}{2} \omega \right )},
\qquad
\text{and}
\qquad
p(\omega) = \mathrm{PG}(\omega | 1, 0).
$$

$$
\begin{align}
p(u) 
&= \int p(u, \omega) d\omega \newline
&= \frac{1}{2} \int \exp{ \left (\frac{u}{2} - \frac{u^2}{2} \omega \right )} p(\omega) d\omega \newline
&= \sigma(u)
\end{align}
$$

```python
from pypolyagamma import PyPolyaGamma

seed = 8888
b, c = 1, 0

pg = PyPolyaGamma(seed=seed)
omega = pg.pgdraw(b, c)
```

{{< figure src="figures/samples_paper_1500x927.png" title="Polya-Gamma variable samples ($M=32$)." numbered="true" >}}

```python
def conditional(u, omega):
    return 0.5 * np.exp(-0.5*u*(u*omega - 1.0))
```

{{< figure src="figures/grid_paper_1500x927.png" title="Hyperbolic cosine function." numbered="true" >}}

$$
\begin{align}
\sigma(u) 
&= \int p(u | \omega) p(\omega) d\omega \newline
&\approx \frac{1}{M} \sum\_{m=1}^M p(u | \omega^{(m)}),
\qquad
\omega^{(m)} \sim \mathrm{PG}(\omega | 1, 0)
\end{align}
$$

{{< figure src="figures/monte_carlo_paper_1500x927.png" title="Hyperbolic cosine function." numbered="true" >}}


$$
\begin{align}
p(u | \omega) 
&= \frac{1}{2} \exp{ \left (\frac{u}{2} - \frac{u^2}{2} \omega \right )} \newline
&\propto \exp{ \left \\{ - \frac{\omega}{2} \left ( u - \frac{\omega^{-1}}{2} \right )^2 \right \\} } \newline
&\propto \mathcal{N} \left ( u ; \frac{\omega^{-1}}{2}, \omega^{-1} \right )
\end{align}
$$

more specifically we have

$$
p(u | \omega) =
\sqrt{\frac{\pi \omega^{-1}}{2}} \exp{ \left ( \frac{\omega^{-1}}{2^3} \right ) } \times
\mathcal{N} \left ( u ; \frac{\omega^{-1}}{2}, \omega^{-1} \right )
$$
derivations[^fn1]


```python
from scipy.stats import norm


def conditional(u, omega):

    var = np.reciprocal(omega)

    loc = 0.5*var
    scale = np.sqrt(var)

    rv = norm(loc=loc, scale=scale)

    return np.sqrt(0.5*np.pi) * scale * np.exp(0.5**2*loc) * rv.pdf(u)
```


$y\_n \in \\{ -1, 1 \\}$

likelihood, or observation model
$$
p(y\_n | f(\mathbf{x}\_n)) = \sigma(y\_n f(\mathbf{x}\_n))
$$

$f\_n = f(\mathbf{x}\_n)$

deliberately non-specific about the form of function $f$ so as to avoid losing
generality and ensure applicability in both the weight-space view and 
function-space view.

$$
f(\mathbf{x}\_n) = \boldsymbol{\beta}^{\top} \mathbf{x}\_n
$$

augmented likelihood, which factorizes as
$$
p(y\_n, \omega\_n | f(\mathbf{x}\_n)) = p(y\_n | \omega\_n, f(\mathbf{x}\_n)) p(\omega\_n)
$$

$$
p(y\_n | \omega\_n, f(\mathbf{x}\_n)) = 
\frac{1}{2} 
\exp{ \left (\frac{y\_n f(\mathbf{x}\_n)}{2} - 
\frac{f(\mathbf{x}\_n)^2}{2} \omega\_n \right )}, 
\qquad 
\text{and}
\qquad 
p(\omega\_n) = \mathrm{PG}(\omega\_n | 1, 0)
$$


$$
\begin{align}
p(y\_n | f(\mathbf{x}\_n)) &=
\int p(y\_n, \omega\_n | f(\mathbf{x}\_n)) d\omega\_n \newline 
&= \int p(y\_n | \omega\_n, f(\mathbf{x}\_n)) p(\omega\_n) d\omega\_n \newline 
&= \frac{1}{2} \int \exp{ \left (\frac{y\_n f(\mathbf{x}\_n)}{2} - 
\frac{f(\mathbf{x}\_n)^2}{2} \omega\_n \right )} \mathrm{PG}(\omega\_n | 1, 0) d\omega\_n \newline
&= \sigma(y\_n f(\mathbf{x}\_n))
\end{align}
$$

---



## Links and Further Readings

- https://gregorygundersen.com/blog/2019/09/20/polya-gamma/
- Test[^snell2020bayesian]

---

Cite as:

```
@article{tiao2021knowledge,
  title   = "{A}n {I}llustrated {G}uide to the {K}nowledge {G}radient {A}cquisition {F}unction",
  author  = "Tiao, Louis C",
  journal = "tiao.io",
  year    = "2021",
  url     = "https://tiao.io/post/an-illustrated-guide-to-the-knowledge-gradient-acquisition-function/"
}
```

To receive updates on more posts like this, follow me on [Twitter] and [GitHub]!

[Twitter]: https://twitter.com/louistiao
[GitHub]: https://github.com/ltiao

[^polson2013bayesian]: Polson, N. G., Scott, J. G., & Windle, J. (2013). [Bayesian Inference for Logistic Models using Pólya–Gamma Latent Variables](https://arxiv.org/abs/1205.0310). Journal of the American Statistical Association, 108(504), 1339-1349.
[^wenzel2019efficient]: Wenzel, F., Galy-Fajou, T., Donner, C., Kloft, M., & Opper, M. (2019, July). [Efficient Gaussian Process Classification using Pòlya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383). In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 5417-5424).
[^snell2020bayesian]: Snell, J., & Zemel, R. (2020). [Bayesian Few-Shot Classification with One-vs-Each Pólya-Gamma Augmented Gaussian Processes](https://arxiv.org/abs/2007.10417). arXiv preprint arXiv:2007.10417.

[^fn1]: Test
$$
\begin{align}
p(u | \omega) 
&= \frac{1}{2} \exp{ \left (\frac{u}{2} - \frac{u^2}{2} \omega \right )} \newline
&= \frac{1}{2} \exp{ \left \\{ - \frac{\omega}{2} \left ( u^2 - u \omega^{-1} \right ) \right \\} } \newline
&= \frac{1}{2} \exp{ \left \\{ \frac{\omega}{2} \left ( \frac{\omega^{-1}}{2} \right )^2 - \frac{\omega}{2} \left ( u - \frac{\omega^{-1}}{2} \right )^2 \right \\} } \newline
&= \frac{1}{2} \exp{ \left ( \frac{\omega^{-1}}{2^3} \right ) } \times 
\exp{ \left \\{ - \frac{\omega}{2} \left ( u - \frac{\omega^{-1}}{2} \right )^2 \right \\} } \newline
&= \frac{1}{2} \exp{ \left ( \frac{\omega^{-1}}{2^3} \right ) } \times 
\frac{\sqrt{2\pi\omega^{-1}}}{\sqrt{2\pi\omega^{-1}}} \exp{ \left \\{ - \frac{\omega}{2} \left ( u - \frac{\omega^{-1}}{2} \right )^2 \right \\} } \newline
&= \sqrt{\frac{\pi \omega^{-1}}{2}} \exp{ \left ( \frac{\omega^{-1}}{2^3} \right ) } \times 
\mathcal{N} \left ( u ; \frac{\omega^{-1}}{2}, \omega^{-1} \right )
\end{align}
$$
