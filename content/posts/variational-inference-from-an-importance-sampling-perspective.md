---
aliases:
  - /post/variational-inference-from-an-importance-sampling-perspective/
title: 'Variational Inference from an Importance Sampling Perspective'
subtitle: 'Create a beautifully simple website in under 10 minutes :rocket:'
summary: Create a beautifully simple website in under 10 minutes.
authors:
- me
tags:
- Machine Learning
- Variational Inference
- Importance Sampling
date: "2018-08-14T00:00:00"
math: true
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  placement: 2
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/CpkOjOcXdUY)'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

In this short note, we motivate variational inference from an importance 
sampling perspective.

Let $\mathbf{x}$ be a set of observed variables and $\mathbf{z}$ a 
set of hidden variables whose joint distribution factorizes as
$p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})$.
We marginalize out the latent variables $\mathbf{z}$ to obtain the 
*evidence*

$$
\begin{align}
  p(\mathbf{x}) 
  & = \int p(\mathbf{x} , \mathbf{z}) d\mathbf{z} \newline
  & = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z} 
  = \mathbb{E}\_{p(\mathbf{z})} [p(\mathbf{x} | \mathbf{z})].
\end{align}
$$

Evaluating this integral using Monte Carlo sampling is problematic since

need to sample from more interesting and important regions

having oversampled the important region, we have to adjust our estimate to 
account for having sampled from this other distribution.

Consider multiplying the integrand in the evidence integral by 
$1 = \frac{p(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})}$:

$$
p(\mathbf{x}) 
= \int p(\mathbf{x}, \mathbf{z}) \frac{p(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z} 
= \mathbb{E}\_{p(\mathbf{z}|\mathbf{x})} 
\left [
  \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z}|\mathbf{x})}
\right ]
$$

define function $w$ to be the *density ratio*

$$
w(\mathbf{x}, \mathbf{z}) := \frac{p(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})}.
$$

we can rewrite as

$$
p(\mathbf{x}) 
= \mathbb{E}\_{p(\mathbf{z}|\mathbf{x})} 
[w(\mathbf{x}, \mathbf{z}) p(\mathbf{x} | \mathbf{z})]
$$

- $p(\mathbf{x} | \mathbf{z})$ is the nominal distribution
- $p(\mathbf{z}|\mathbf{x})$ is the importance distribution
- $w(\mathbf{x}, \mathbf{z})$ is the importance weight

Intuitively, this ratio indicates the adjustment factor required for the 
approximate posterior $q(\mathbf{z})$ to be equal to prior 
$p(\mathbf{z})$, since $r(\mathbf{z}) q(\mathbf{z}) = p(\mathbf{z})$.


Ideally, we would be able to sample from the exact posterior distribution 
$p(\mathbf{z} | \mathbf{x})$. However this is intractable and why we in 
the first place. We approximate this using a variational distribution 
$q(\mathbf{z}; \lambda) \approx p(\mathbf{z} | \mathbf{x})$. 

$$
p(\mathbf{x}) 
= \mathbb{E}\_{q(\mathbf{z}; \lambda)} 
\left [
  \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}; \lambda)}
\right ]
$$

We define function $r$ to be the *density ratio*

$$
r(\mathbf{z}) := \frac{p(\mathbf{z})}{q(\mathbf{z}; \lambda)}
$$

$$
p(\mathbf{x}) 
\simeq \mathbb{E}\_{q(\mathbf{z}; \lambda)} [r(\mathbf{z}) p(\mathbf{x} | \mathbf{z})]
$$

- $p(\mathbf{x} | \mathbf{z})$ is the nominal distribution
- $q(\mathbf{z}; \lambda)$ is the importance distribution
- $r(\mathbf{z})$ is the importance weight


Taking the logarithm of both sides and by applying Jensen's inequality, we have

$$
\begin{align}
  \log p(\mathbf{x}) 
  & \simeq \log \mathbb{E}\_{q(\mathbf{z}; \lambda)} 
  [r(\mathbf{z}) p(\mathbf{x} | \mathbf{z})] \newline
  & \geq \mathbb{E}\_{q(\mathbf{z}; \lambda)} 
  \log [r(\mathbf{z}) p(\mathbf{x} | \mathbf{z})] \newline
  & = \mathbb{E}\_{q(\mathbf{z}; \lambda)} 
  [ \log r(\mathbf{z}) + \log p(\mathbf{x} | \mathbf{z}) ] \newline
  & = \mathbb{E}\_{q(\mathbf{z}; \lambda)} 
  [ \log p(\mathbf{z}) - \log q(\mathbf{z}; \lambda) + \log p(\mathbf{x} | \mathbf{z}) ] \newline
  & := \mathcal{L}\_{\mathrm{ELBO}}(\lambda)
\end{align}
$$

Hence, we arrive at exactly the *evidence lower bound* (ELBO). Importantly, the
tightness of the bound is determined by KL divergence between the approximate
and exact posterior, and maximizing the former is equivalent to minimizing the
latter. We have equality exactly when the KL divergence is zero, which occurs
iff $q(\mathbf{z}) = p(\mathbf{z})$.

Importance Weighted ELBO
------------------------

[^burda2015importance]

$$
p(\mathbf{x}) 
= \mathbb{E}\_{\mathbf{z}\_1, \dotsc, \mathbf{z}\_K \sim q(\mathbf{z}; \lambda)} 
\left [
  \sum\_{k=1}^K \frac{p(\mathbf{x}, \mathbf{z}\_k)}{q(\mathbf{z}\_k; \lambda)}
\right ]
$$


$$
\begin{align}
\log p(\mathbf{x}) 
& = \log \mathbb{E}\_{\mathbf{z}\_1, \dotsc, \mathbf{z}\_K \sim q(\mathbf{z}; \lambda)} 
\left [
  \sum\_{k=1}^K \frac{p(\mathbf{x}, \mathbf{z}\_k)}{q(\mathbf{z}\_k; \lambda)}
\right ] \newline
& \geq \mathbb{E}\_{\mathbf{z}\_1, \dotsc, \mathbf{z}\_K \sim q(\mathbf{z}; \lambda)} 
\left [
  \log \sum\_{k=1}^K \frac{p(\mathbf{x}, \mathbf{z}\_k)}{q(\mathbf{z}\_k; \lambda)}
\right ] \newline
& := \mathcal{L}\_K(\lambda)
\end{align}
$$

$K$ is given by `num_samples`

```python
import tensorflow_probability as tfp
```

```python
prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim))

posterior = build_posterior(x_data, fn=inference_network)
z_posterior = posterior.sample(num_samples)

likelihood = build_likelihood(z_posterior, fn=generative_network)

llh = likelihood.log_prob(x_data)
kl = posterior.log_prob(z_posterior) - prior.log_prob(z_posterior)
```

```python
iw_elbo = tf.reduce_mean(tf.reduce_logsumexp(llh - kl, axis=0) -
                         tf.log(tf.to_float(num_samples)))
```


References
----------

* https://casmls.github.io/general/2017/04/24/iwae-aae.html
* http://dustintran.com/blog/importance-weighted-autoencoders
* http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html
* https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf
* http://blog.shakirm.com/2018/01/machine-learning-trick-of-the-day-7-density-ratio-trick/
* http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html

[^burda2015importance]: Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). [Importance weighted autoencoders](https://arxiv.org/abs/1509.00519). *arXiv preprint* arXiv:1509.00519.