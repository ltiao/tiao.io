# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "polyagamma",
# ]
# ///
"""Numerical checks for the identities quoted in Part III.

Self-contained; run with:  uv run verify_identities.py

1. The Jaakkola-Jordan bound written with lambda(xi) = E_{PG(1,xi)}[omega]:
       sigma(psi) >= ell(psi, xi)
                   = sigma(xi) * exp((psi - xi)/2 - lambda(xi)/2 * (psi^2 - xi^2))
2. The slack of the bound is an exponentiated Polya-Gamma KL divergence:
       sigma(psi) / ell(psi, xi) = exp( KL[PG(1, xi) || PG(1, psi)] )
   where
       KL[PG(1, xi) || PG(1, psi)]
           = log cosh(xi/2) - log cosh(psi/2) + (psi^2 - xi^2)/2 * lambda(xi)
3. Special case:  KL[PG(1, xi) || PG(1, 0)] = log cosh(xi/2) - xi^2/2 * lambda(xi)
4. lambda(xi) = E_{PG(1, xi)}[omega], checked by Monte Carlo.
"""
import numpy as np
from polyagamma import random_polyagamma
from scipy.special import log_expit


def lambd(xi):
    return 0.5 * np.tanh(0.5 * xi) / xi


def log_cosh(x):
    # log cosh(x) = |x| + log1p(exp(-2|x|)) - log 2, stable for large |x|
    a = np.abs(x)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def log_ell(psi, xi):
    return log_expit(xi) + 0.5 * (psi - xi) - 0.5 * lambd(xi) * (psi**2 - xi**2)


def kl_pg(xi, psi):
    return log_cosh(0.5 * xi) - log_cosh(0.5 * psi) \
        + 0.5 * (psi**2 - xi**2) * lambd(xi)


def main():
    psi, xi = np.meshgrid(np.linspace(-10.0, 10.0, 401),
                          np.linspace(1e-3, 10.0, 400))

    gap = log_expit(psi) - log_ell(psi, xi)   # log sigma - log ell
    kl = kl_pg(xi, psi)

    assert np.all(gap >= -1e-12), "ell is not a lower bound?!"
    np.testing.assert_allclose(gap, kl, rtol=1e-9, atol=1e-12)
    print("identity 2 OK: sigma/ell = exp(KL[PG(1,xi) || PG(1,psi)])")

    # tightness at xi = +/- psi
    d = np.linspace(1e-3, 10.0, 400)
    np.testing.assert_allclose(kl_pg(d, d), 0.0, atol=1e-12)
    np.testing.assert_allclose(kl_pg(d, -d), 0.0, atol=1e-12)
    print("identity 2 OK: KL vanishes iff xi = +/- psi (checked at xi = +/- psi)")

    np.testing.assert_allclose(kl_pg(d, np.zeros_like(d)),
                               log_cosh(0.5 * d) - 0.5 * d**2 * lambd(d),
                               rtol=1e-12)
    print("identity 3 OK: KL[PG(1,xi) || PG(1,0)] special case")

    rng = np.random.default_rng(8888)
    for c in (0.1, 0.5, 1.0, 2.5, 5.0):
        omega = random_polyagamma(1.0, c, size=2_000_000, random_state=rng)
        err = abs(omega.mean() - lambd(c)) / lambd(c)
        assert err < 5e-3, (c, omega.mean(), lambd(c))
    print("identity 4 OK: E_PG(1,c)[omega] = tanh(c/2)/(2c)  (Monte Carlo)")


if __name__ == "__main__":
    main()
