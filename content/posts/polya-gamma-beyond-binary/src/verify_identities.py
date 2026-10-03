# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "polyagamma",
# ]
# ///
"""Numerical checks for the identities quoted in Part IV of the
Polya-Gamma series (beyond binary: softmax / multinomial, Bouchard's
bound, the one-vs-each / Titsias bound, and binomial / negative-binomial
augmentation).

Self-contained; run with:  uv run verify_identities.py

Notation
--------
    lambda(xi) = tanh(xi/2) / (2 xi)              ( = E_{PG(1,xi)}[omega] )
    softplus   s(x) = log(1 + e^x)                ( = np.logaddexp(0, x) )
    JJ softplus upper bound
        g(x, xi) = (x - xi)/2 + (lambda(xi)/2)(x^2 - xi^2) + s(xi)
    KL[PG(1, xi) || PG(1, c)]
        = logcosh(xi/2) - logcosh(c/2) + (c^2 - xi^2)/2 * lambda(xi)
    LSE(psi) = log sum_k exp(psi_k)

Checks
------
1. softplus-side JJ identity:  g(x, xi) - s(x) = KL[PG(1, xi) || PG(1, x)]
2. Bouchard step 1 (alpha-bound):  slack = -log P(exactly one Bernoulli fires) > 0
3. Bouchard full decomposition:  B - LSE = sum_k KL[.] + slack
4. optimal alpha stationarity:  closed form vs numerical argmin
5. one-vs-each (Titsias) bound:  slack = -log P(at most one fires)
6. binomial PG augmentation (Monte Carlo)
7. negative-binomial PG augmentation (Monte Carlo)
"""
import numpy as np
from numpy.testing import assert_allclose
from polyagamma import random_polyagamma
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln, log_expit, logsumexp


# --- primitives -----------------------------------------------------------

def lambd(xi):
    return 0.5 * np.tanh(0.5 * xi) / xi


def softplus(x):
    return np.logaddexp(0.0, x)


def log_cosh(x):
    # logcosh(x) = |x| + log1p(exp(-2|x|)) - log 2  (stable for large |x|)
    a = np.abs(x)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def g_bound(x, xi):
    return 0.5 * (x - xi) + 0.5 * lambd(xi) * (x**2 - xi**2) + softplus(xi)


def kl_pg(xi, c):
    # KL[PG(1, xi) || PG(1, c)]
    return log_cosh(0.5 * xi) - log_cosh(0.5 * c) + 0.5 * (c**2 - xi**2) * lambd(xi)


def slack_alpha(psi, alpha):
    # S_alpha(psi, alpha) = alpha + sum_k s(psi_k - alpha) - LSE(psi)
    return alpha + softplus(psi - alpha).sum(-1) - logsumexp(psi, axis=-1)


def log_p_exactly_one(psi, alpha):
    # log P(exactly one of K independent Bernoullis fires),
    # p_k = sigma(psi_k - alpha).  Computed stably in log-space:
    #   log( p_k * prod_{j!=k}(1 - p_j) ) = log p_k + sum_j log(1-p_j) - log(1-p_k)
    z = psi - alpha
    log_p = log_expit(z)          # log p_k
    log_1mp = log_expit(-z)       # log(1 - p_k)
    total = log_1mp.sum(-1, keepdims=True)
    terms = log_p + total - log_1mp
    return logsumexp(terms, axis=-1)


# --- checks ---------------------------------------------------------------

def check1_softplus_jj():
    x, xi = np.meshgrid(np.linspace(-10.0, 10.0, 401),
                        np.linspace(1e-3, 10.0, 400))
    lhs = g_bound(x, xi) - softplus(x)
    rhs = kl_pg(xi, x)
    # g is a genuine *upper* bound on softplus  <=>  KL >= 0
    assert np.all(lhs >= -1e-12), "g is not an upper bound on softplus?!"
    assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-13)
    print("PASS 1  softplus-side JJ:  g(x,xi) - softplus(x) = KL[PG(1,xi) || PG(1,x)]")


def check2_bouchard_alpha(rng):
    min_slack = np.inf
    for K in (2, 3, 5, 10):
        psi = rng.normal(0.0, 3.0, size=(400, K))
        for alpha in np.linspace(-8.0, 8.0, 41):
            S = slack_alpha(psi, alpha)
            neg_log_p1 = -log_p_exactly_one(psi, alpha)
            assert_allclose(S, neg_log_p1, rtol=1e-10, atol=1e-12)
            assert np.all(S > 0.0), (K, alpha, S.min())
            min_slack = min(min_slack, S.min())
    # ...also at the optimal alpha (argmin of the step-1 bound), never tight
    for K in (2, 3, 5, 10):
        for _ in range(50):
            psi = rng.normal(0.0, 3.0, size=K)
            res = minimize_scalar(lambda a: a + softplus(psi - a).sum(),
                                  method="bounded", bounds=(-50.0, 50.0),
                                  options={"xatol": 1e-11})
            S = slack_alpha(psi, res.x)
            assert S > 0.0, (K, res.x, S)
            assert_allclose(S, -log_p_exactly_one(psi, res.x),
                            rtol=1e-10, atol=1e-12)
            min_slack = min(min_slack, float(S))
    print("PASS 2  Bouchard alpha-bound:  slack = -log P(exactly one fires) > 0 "
          f"(incl. optimal alpha; min slack seen = {min_slack:.3e})")


def check3_bouchard_full(rng):
    for K in (2, 3, 5, 10):
        psi = rng.normal(0.0, 3.0, size=(300, K))
        xi = rng.uniform(0.05, 6.0, size=(300, K))
        for alpha in np.linspace(-6.0, 6.0, 25):
            B = alpha + g_bound(psi - alpha, xi).sum(-1)
            lhs = B - logsumexp(psi, axis=-1)
            rhs = kl_pg(xi, psi - alpha).sum(-1) + slack_alpha(psi, alpha)
            assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)
    print("PASS 3  Bouchard full:  B - LSE = "
          "sum_k KL[PG(1,xi_k) || PG(1,psi_k-alpha)] + slack")


def check4_optimal_alpha(rng):
    worst_alpha = 0.0
    worst_deriv = 0.0
    for _ in range(200):
        K = int(rng.choice([2, 3, 5, 10]))
        psi = rng.normal(0.0, 3.0, size=K)
        xi = rng.uniform(0.05, 6.0, size=K)
        lam = lambd(xi)
        # closed form
        a_star = (0.5 * K - 1.0 + (lam * psi).sum()) / lam.sum()
        # independent numerical argmin of  a -> a + sum_k g(psi_k - a, xi_k)
        f = lambda a: a + g_bound(psi - a, xi).sum()
        res = minimize_scalar(f, method="bounded", bounds=(-60.0, 60.0),
                              options={"xatol": 1e-11})
        worst_alpha = max(worst_alpha, abs(a_star - res.x))
        # derivative  1 - sum_k [ 1/2 + lambda_k (psi_k - alpha) ]  vanishes at a_star
        deriv = 1.0 - (0.5 + lam * (psi - a_star)).sum()
        worst_deriv = max(worst_deriv, abs(deriv))
    assert worst_alpha < 1e-6, worst_alpha
    assert worst_deriv < 1e-10, worst_deriv
    print(f"PASS 4  optimal alpha*: closed form vs argmin max|Δ| = {worst_alpha:.2e}; "
          f"f'(alpha*) = 1 - Σ[1/2 + λ_k(ψ_k-α*)] = 0 (max |f'| = {worst_deriv:.1e})")


def check5_one_vs_each(rng):
    min_slack = np.inf
    for K in (2, 3, 5, 10):
        for _ in range(300):
            psi = rng.normal(0.0, 3.0, size=K)
            y = int(rng.integers(K))
            others = np.delete(psi, y)
            log_p = psi[y] - logsumexp(psi)              # log softmax_y(psi)
            log_bound = log_expit(psi[y] - others).sum()  # sum log sigma(psi_y - psi_k)
            slack = log_p - log_bound
            assert slack >= -1e-12, ("one-vs-each is not a lower bound", slack)
            # combinatorial P(at most one fires), p_k = sigma(psi_k - psi_y), k != y
            p = expit(others - psi[y])
            P0 = np.prod(1.0 - p)                          # = bound
            P1 = np.sum(p * (P0 / (1.0 - p)))             # sum_k p_k prod_{j!=k}(1-p_j)
            neg_log_atmost1 = -np.log(P0 + P1)
            assert_allclose(slack, neg_log_atmost1, rtol=1e-9, atol=1e-12)
            min_slack = min(min_slack, float(slack))
    print("PASS 5  one-vs-each:  log p(y|psi) - sum_{k!=y} log sigma(psi_y-psi_k) "
          f"= -log P(at most one fires) >= 0 (min slack seen = {min_slack:.3e})")


def _mc_expect(h, psi, n, rng):
    # Monte Carlo estimate of E_{omega ~ PG(h, 0)}[ exp(-psi^2 omega / 2) ]
    omega = random_polyagamma(h, 0.0, size=n, random_state=rng)
    return float(np.mean(np.exp(-0.5 * psi**2 * omega)))


def check6_binomial(rng):
    # Binomial pmf via PG augmentation:  kappa = y - m/2, PG(m, 0)
    cases = [(1, 1, 0.7), (5, 3, -1.2), (10, 4, 1.5),
             (20, 13, 0.8), (20, 7, -2.0), (8, 0, 1.0)]
    worst = 0.0
    for m, y, psi in cases:
        log_comb = gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1)
        lhs = np.exp(log_comb + y * log_expit(psi) + (m - y) * log_expit(-psi))
        kappa = y - 0.5 * m
        mc = _mc_expect(m, psi, 2_000_000, rng)
        rhs = np.exp(log_comb - m * np.log(2.0) + kappa * psi) * mc
        rel = abs(rhs - lhs) / abs(lhs)
        assert rel < 1e-2, (m, y, psi, rel)
        worst = max(worst, rel)
    print(f"PASS 6  binomial PG augmentation matches pmf "
          f"(Monte Carlo, max rel err = {worst:.2e} < 1e-2)")


def check7_negbinom(rng):
    # NB pmf (r failures, success prob sigma(psi)) via PG:  kappa = (y-r)/2, PG(y+r, 0)
    cases = [(1.0, 2, 0.8), (5.0, 3, -1.0), (2.5, 4, 1.2),
             (7.3, 1, -1.5), (3.0, 6, 0.5)]
    worst = 0.0
    for r, y, psi in cases:
        b = y + r
        log_comb = gammaln(y + r) - gammaln(r) - gammaln(y + 1)  # log C(y+r-1, y)
        lhs = np.exp(log_comb + y * log_expit(psi) + r * log_expit(-psi))
        kappa = 0.5 * (y - r)
        mc = _mc_expect(b, psi, 2_000_000, rng)
        rhs = np.exp(log_comb - b * np.log(2.0) + kappa * psi) * mc
        rel = abs(rhs - lhs) / abs(lhs)
        assert rel < 1e-2, (r, y, psi, rel)
        worst = max(worst, rel)
    print(f"PASS 7  negative-binomial PG augmentation matches pmf "
          f"(Monte Carlo, incl. non-integer r; max rel err = {worst:.2e} < 1e-2)")


def main():
    rng = np.random.default_rng(8888)
    check1_softplus_jj()
    check2_bouchard_alpha(rng)
    check3_bouchard_full(rng)
    check4_optimal_alpha(rng)
    check5_one_vs_each(rng)
    check6_binomial(rng)
    check7_negbinom(rng)
    print("\nall checks passed.")


if __name__ == "__main__":
    main()
