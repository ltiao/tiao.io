# From Brownian motion to the Matérn-1/2 kernel

The draft's time change $g(x) = \sigma e^{-x/\ell} f(e^{2x/\ell})$ is Doob's transformation (Doob 1942). Its inverse, $f(t) = \sigma^{-1}\sqrt{t}\, g\big(\tfrac{\ell}{2}\ln t\big)$, is the Lamperti transform, the general duality between self-similar and stationary processes, seen here at Hurst index $H = 1/2$. Both names belong in the note as scholarly anchors.

---

Brownian motion is a Matérn-1/2 process that forgot to mean-revert.

---

The transform turns Brownian motion into OU exactly, and a scaling limit turns OU back into Brownian motion. Set $\sigma^2 = c\ell/2$ and let $\ell \to \infty$: the increment variance $2\sigma^2(1 - e^{-|h|/\ell}) \to c|h|$, which is the Brownian variogram. And on scales $h \ll \ell$, every Matérn-1/2 path is locally indistinguishable from Brownian motion with diffusion coefficient $2\sigma^2/\ell$. Exact transform one way, scaling limit the other, and the loop closes.

---

The OU process solves $dg = -(g/\ell)\,dt + \sqrt{2\sigma^2/\ell}\,dB$, which is Brownian motion plus a linear mean-reverting drift. The Matérn-1/2 GP is literally driven by Brownian motion.

---

Girsanov's theorem says adding drift gives an equivalent measure on any finite window, so densely sampled data on a fixed interval cannot distinguish OU from Brownian motion with matched diffusion coefficient. Under infill asymptotics the only identifiable (microergodic) parameter is $\sigma^2/\ell$ (Zhang 2004; for Matérn-$\nu$ it is $\sigma^2/\ell^{2\nu}$). The flat ridge every practitioner has seen in the Matérn-1/2 likelihood surface is exactly $\sigma^2/\ell = \text{const}$. As a bonus, the realized quadratic variation of the path estimates $2\sigma^2/\ell$ directly from high-frequency data.

---

Doob's theorem says OU is the only stationary Gaussian Markov process that is mean-square continuous. Uniqueness is the answer to why $e^{-|h|/\ell}$ is special.

---

The Markov property has cash value everywhere. At ordered inputs both the Brownian and exponential kernels have tridiagonal precision matrices, giving exact $O(n)$ GP regression with no inducing points. On a regular grid Matérn-1/2 is exactly AR(1) with $\varphi = e^{-\Delta/\ell}$, and Brownian motion is the unit root $\varphi = 1$. Paths can be sampled by forward recursion instead of Cholesky, and the posterior mean between two observations depends only on the two flanking points. This is the 1D seed of state-space GPs (Hartikainen & Särkkä 2010; Särkkä & Solin's SDE book) and of celerite (Foreman-Mackey et al. 2017).

---

By Bochner, the exponential kernel's spectral density is Lorentzian, $S(\omega) \propto (1 + \ell^2\omega^2)^{-1}$, so the exponential kernel is the characteristic function of a Cauchy distribution. The systems reading: OU is white noise through a leaky integrator (an RC filter) with leak rate $1/\ell$, and Brownian motion is white noise through a pure integrator ($1/\omega^2$ spectrum). The practical corollary is that random Fourier features for Matérn-1/2 must sample Cauchy-distributed frequencies, and those heavy tails are why RFF works badly for this kernel and the state-space route wins.

---

The Green's-function view: $\int_{-\infty}^{\min(s,t)} e^{-(s-u)/\ell} e^{-(t-u)/\ell}\,du = \tfrac{\ell}{2} e^{-|s-t|/\ell}$. OU is white noise convolved with a causal exponential, the Green's function of $d/dt + 1/\ell$, and Brownian motion is white noise convolved with a step, since $\min(s,t) = \int \mathbb{1}[u \le s]\mathbb{1}[u \le t]\,du$.

---

The spline ladder: step features give Brownian motion, whose posterior mean is the linear smoothing spline (Kimeldorf–Wahba); ReLU features give integrated Brownian motion, whose posterior mean is the cubic smoothing spline; OU gives an exponential (tension) spline. The deployable line: connect-the-dots is a GP posterior.

---

Brownian motion enters a second, independent way, via Schoenberg: $\exp(-\gamma(h))$ is a valid kernel for any variogram $\gamma$, and $|h|$ is precisely the variogram of Brownian motion, so the exponential kernel is the Schoenberg exponential of the Brownian variogram. RBF is the same construction applied to $|h|^2$, the variogram of a random line. The $\gamma$-exponential family $\exp(-|h|^{2H})$ is exactly the fBM variograms, $0 < H \le 1$. And $\min(s,t) = \tfrac{1}{2}(s + t - |s-t|)$, so the $-|h|$ exponent already sits inside the Brownian kernel. Brownian motion appears twice: as the process (Doob) and as the variogram (Schoenberg).

---

Uhlenbeck & Ornstein (1930) built the OU process as the *velocity* of a physical Brownian particle with friction (the Langevin equation), and idealized Brownian motion is the overdamped limit. So $e^{-|h|/\ell}$ is what Brownian motion actually looks like in a lab, as a velocity autocorrelation.

---

Finance runs the same pair. Vasicek's short-rate model is OU; Ho–Lee is Brownian motion. Mean reversion versus unit root.

---

The Markov property does not survive to $\mathbb{R}^2$, a common misconception. Rozanov's criterion requires the spectral density to be the reciprocal of a polynomial, and for the isotropic exponential kernel $S(\omega) \propto (1 + |\omega|^2)^{-(d+1)/2}$, which is polynomial only for odd $d$. So the process is Markov in $d = 1$ and $d = 3$, not $d = 2$, which is why the Lindgren–Rue–Lindström SPDE machinery for spatial statistics is approximate.

---

The Laplace kernel and the NTK of infinitely wide ReLU networks have the same RKHS on the sphere (Geifman et al. 2020; Chen & Xu 2021). Matérn-1/2 roughness is secretly the smoothness class of infinite-width ReLU nets.

---

BayesOpt practicalities. Matérn-1/2 posterior samples are nowhere differentiable, so gradient-based optimization of pathwise or Thompson samples is ill-posed. Extrapolation reverts to the prior mean at rate $\ell$, versus Brownian motion's flat martingale extrapolation with linearly growing variance. The RKHS is Sobolev $H^1$ and the sample paths are almost surely *not* in it, the classic pitfall. And worst-case information gain is the largest in the Matérn family.

---

Structure half-thought: the spine is the draft's Doob/Lamperti derivation; add the $\ell \to \infty$ limit to close the loop; the Girsanov ridge and the Markov/AR(1) consequences are the "why you should care" section; the Cauchy/Lorentzian spectral fact is a one-line coda. Everything else is margin notes or a future post.

---

Draft nits to fix: "Weiner" should be "Wiener"; the increment law $\mathcal{N}(0, t - t')$ needs $t > t'$ stated, or write $|t - t'|$; the $\min$-covariance derivation silently uses $f(0) = 0$ and independent increments, and it is the independence that collapses $\mathrm{Cov}(f(t), f(t'))$ to $\mathrm{Var}\,f(t') = t'$.

---

Leading-word candidates, a compact coinage the piece could hang on: "the Girsanov ridge" for the flat direction $\sigma^2/\ell$ in the likelihood; "forgot to mean-revert"; "leaky integrator".

---

The note exists because of an interview. The question was to derive the GP kernel for Brownian motion: independent increments, $f(0) = 0$, and $\mathrm{Cov}(f(t), f(t'))$ collapses to $\min(t, t')$. The interview ended; the debrief with myself didn't. Somewhere in that debrief, $\min(t, t')$ turned out to be one time-change away from the exponential kernel.

---

Calibration, after a sweep for prior art: the identity is classical. Doob (1942) is the transformation's name, the inverse is Lamperti (1962), the OU Wikipedia page lists the time-change representation, and stochastic-calculus courses assign it as an exercise. Yet no GP/ML treatment seems to derive the Matérn-1/2 kernel this way. GPML reaches OU through the SDE and the uniqueness theorem, never through the time change from $\min(t,t')$. The note claims exposition rather than novelty: moving a known wire between communities. The gap between documented and known is where expository notes live.

---

A stats course slide out there asserts "exponential kernel = Brownian motion in 1D" as a bare bullet, underived, and as stated it is sloppy: the honest versions are the local equivalence at scales $h \ll \ell$ or the exact time-change identity. Evidence that the connection is half-known, asserted in courses and written down properly nowhere.

---

Payoff decision: the reader I am writing for finishes with "so that is why my Matérn-1/2 fits behave this way." The identity is the hook; the practical consequences are the payoff. The probability literature never bothers with that half, because probabilists do not fit GPs for a living.
