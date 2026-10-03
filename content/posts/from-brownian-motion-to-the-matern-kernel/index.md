---
title: "From Brownian Motion to the Matérn-1/2 Kernel"
subtitle: ""
summary: "We derive the Matérn-1/2 (exponential) kernel from Brownian motion via a classical time change, and unpack what the identity explains about how Matérn-1/2 models behave in practice."
authors:
- me
tags:
- Gaussian Processes
- Probabilistic Models
- Machine Learning
categories:
- technical
date: 2026-07-14T12:00:00-04:00
lastmod: 2026-07-14T12:00:00-04:00
featured: false
draft: true
math: true

image:
  caption: ""
  focal_point: Center
  preview_only: false

projects: []
---

Brownian motion is the simplest random function there is. Start at zero, $f(0) = 0$; over any interval, move by a Gaussian amount with variance equal to the elapsed time; make moves over disjoint intervals independent. That is the whole definition, and it already pins down a Gaussian process. For $t' \le t$, split $f(t) = f(t') + (f(t) - f(t'))$; the increment is independent of $f(t')$, so

$$
\operatorname{Cov}(f(t), f(t')) = \operatorname{Var} f(t') = t' = \min(t, t').
$$

So Brownian motion is the GP with the $\min$ kernel — a perfectly serviceable kernel that rarely appears in the catalogs, because it is not stationary: it remembers where zero is.

Two things stop the $\min$ kernel from being stationary: the variance grows, $\operatorname{Var} f(t) = t$, and the origin is special. Each suggests its own repair. Growth you fix by damping — multiply by something that decays. A special origin you fix by running the clock so fast the process forgets it — sample $f$ at times that race off exponentially. The two repairs have to balance: if the damping is $e^{-x/\ell}$, the variance it must cancel is the clock itself, so the clock has to be $e^{2x/\ell}$. That pins the variance at $\sigma^2$ by construction. What it does not obviously fix is the rest of the covariance — there is no a priori reason a function of $\min$ should collapse to a function of $x - x'$. That it does is the small miracle of the next three lines.

So define

$$
g(x) = \sigma e^{-x/\ell}\, f\big(e^{2x/\ell}\big), \qquad \ell, \sigma > 0.
$$

Then

$$
\begin{aligned}
\operatorname{Cov}\big(g(x), g(x')\big) &= \sigma^2 e^{-(x+x')/\ell}\operatorname{Cov}\big(f(e^{2x/\ell}), f(e^{2x'/\ell})\big) \\
&= \sigma^2 e^{-(x+x')/\ell} \min\big\{e^{2x/\ell}, e^{2x'/\ell}\big\} \\
&= \sigma^2 e^{-\left(x+x'-2\min(x,x')\right)/\ell} = \sigma^2 e^{-|x-x'|/\ell},
\end{aligned}
$$

using $x + x' - 2\min(x,x') = |x - x'|$. The $\min$ has collapsed into a pure function of the gap. The right-hand side is the exponential kernel: damped, re-clocked Brownian motion is exactly the Matérn-1/2 GP.

The object $g$ has a name: it is the Ornstein–Uhlenbeck process. The transformation has one too — it is Doob's (1942) — and running it backwards, $f(t) = \sigma^{-1}\sqrt{t}\; g\big(\tfrac{\ell}{2}\ln t\big)$, is a case of the Lamperti transform, the general duality between self-similar processes and stationary ones. None of this is new. It is merely uncirculated in the corners of the literature where Matérn-1/2 is a dropdown option.

What did the surgery produce, dynamically? The process $g$ satisfies

$$
dg = -\frac{g}{\ell}\,dx + \sqrt{\frac{2\sigma^2}{\ell}}\,dB
$$

— a stochastic differential equation, but one that reads in plain English: over each instant, $g$ takes a small independent Gaussian step (the $dB$ term) while being pulled back toward zero in proportion to how far away it sits (the drift term). The pull is called mean reversion, and its rate is $1/\ell$: the lengthscale of your kernel is a forgetting rate. Now run the identity backwards through this lens. As $\ell \to \infty$ the pull vanishes and only the noise remains. Brownian motion is a Matérn-1/2 process that forgot to mean-revert.

The punchline deserves one line of verification. From the kernel, the increments of $g$ satisfy

$$
\operatorname{Var}\big(g(x+h) - g(x)\big) = 2\sigma^2\big(1 - e^{-h/\ell}\big),
$$

which is $\approx (2\sigma^2/\ell)\,h$ for $h \ll \ell$ and saturates at $2\sigma^2$ for $h \gg \ell$. Small steps of a Matérn-1/2 process are Brownian steps with diffusion coefficient $2\sigma^2/\ell$; only at the scale of $\ell$ does the process notice its leash. Set $\sigma^2 = c\ell/2$ and let $\ell \to \infty$: the saturation recedes to infinity and the increment variance converges to $c\,h$, the Brownian value. Exact transform in one direction, scaling limit in the other, and the loop closes.

Now the inheritance with the highest cash value. Observe a Matérn-1/2 GP densely on a fixed window and ask the marginal likelihood for $\sigma^2$ and $\ell$ separately. Locally the data look like Brownian motion with diffusion coefficient $2\sigma^2/\ell$, and that number the data nail: the realized quadratic variation — the sum of squared consecutive increments — recovers it exactly as the grid refines. The mean reversion, by contrast, is a drift, and a drift changes the law of a path gently (Girsanov's theorem: it reweights the probabilities of paths without changing which paths are possible), so a fixed window contains only a fixed, finite amount of evidence about it, and refining the grid adds nothing. One combination, $\sigma^2/\ell$, is learned consistently; the orthogonal direction saturates. This is Zhang's (2004) microergodicity result, and it is also something you have seen: the long, nearly flat ridge in the Matérn-1/2 likelihood surface runs along $\sigma^2/\ell = \text{const}$. The optimizer was never the problem.

Time changes and pointwise rescalings do not create memory. Brownian motion is Markov — given where it is now, where it was is irrelevant — because its increments are independent, and $g$ inherits the property untouched. On a regular grid $x_k = k\Delta$ this is worth money: the Matérn-1/2 GP observed on the grid is exactly the AR(1) recursion $g_{k+1} = \varphi\, g_k + \varepsilon_k$ with $\varphi = e^{-\Delta/\ell}$; Brownian motion is the unit root $\varphi = 1$; the precision matrix is tridiagonal; and GP regression costs $O(n)$ with no inducing points in sight.

One last way to say it, for readers who think in frequencies. By Bochner's theorem every stationary kernel is the Fourier transform of a spectral density, and the exponential kernel's is the Lorentzian $S(\omega) \propto (1 + \ell^2\omega^2)^{-1}$: white noise pushed through a leaky integrator — an RC filter with leak rate $1/\ell$. Brownian motion is the same circuit with the resistor removed, a pure integrator with spectrum $1/\omega^2$. Take the exponential kernel and unhook the leash, and you get Brownian motion: in the time domain, in the frequency domain, in the likelihood surface, and on the grid.
