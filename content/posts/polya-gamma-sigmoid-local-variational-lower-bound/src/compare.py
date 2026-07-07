# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "polyagamma",
#     "typer",
# ]
# ///
"""Gibbs (Part II) vs CAVI (Part III) on the same synthetic dataset.

Reproduces the seed-8888 one-dimensional classification problem used
throughout the series, runs both inference algorithms in the same
Polya-Gamma augmented model, and produces the two comparison figures:

- class_prob_compare_*        posterior predictive class probability
- beta_posterior_compare_*    Gibbs sample cloud vs Gaussian q(beta)

Self-contained; run with:  uv run compare.py ../figures/
(Uses the maintained `polyagamma` package, not `pypolyagamma`, which
predates Python 3.10.)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer

from pathlib import Path
from typing import List, Optional
from scipy.stats import norm
from scipy.special import expit
from polyagamma import random_polyagamma

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


# -- dataset (identical to Parts II and III) --------------------------------

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


def class_probability(x, p, q):
    return expit(p.logpdf(x) - q.logpdf(x))


# -- the two inference algorithms -------------------------------------------

def lambd(xi):
    return 0.5 * np.tanh(0.5 * xi) / xi


def cavi(Phi, kappa, alpha, num_iterations=200, tol=1e-12):
    """Mean-field VI: returns mu, Sigma of q(beta) and the converged xi."""
    latent_dim = Phi.shape[-1]
    eye = np.eye(latent_dim)

    xi = 1e-1 * np.ones(len(kappa))
    for _ in range(num_iterations):
        # update q(beta) = N(mu, Sigma) given lambda(xi)
        Sigma_inv = (lambd(xi) * Phi.T) @ Phi + alpha * eye
        mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
        Sigma = np.linalg.solve(Sigma_inv, eye)

        # update q(omega_n) = PG(1, xi_n) with xi_n^2 = E[psi_n^2]
        xi_new = np.sqrt(np.einsum('ij,ij->i', Phi @ (Sigma + np.outer(mu, mu)), Phi))
        delta, xi = np.max(np.abs(xi_new - xi)), xi_new
        if delta < tol:
            break

    return mu, Sigma, xi


def gibbs(Phi, kappa, alpha, num_iterations=5000, num_burnin=1000, seed=8888):
    """Gibbs sampling: returns beta samples of shape (num_kept, latent_dim)."""
    latent_dim = Phi.shape[-1]
    eye = np.eye(latent_dim)
    rng = np.random.default_rng(seed)

    beta = rng.normal(size=latent_dim, scale=1 / np.sqrt(alpha))

    betas = []
    for _ in range(num_iterations):
        omega = random_polyagamma(1.0, Phi @ beta, random_state=rng)

        Sigma_inv = (omega * Phi.T) @ Phi + alpha * eye
        mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
        Sigma = np.linalg.solve(Sigma_inv, eye)
        beta = rng.multivariate_normal(mean=mu, cov=Sigma)

        betas.append(beta)

    return np.vstack(betas[num_burnin:])


def cov_ellipse(mean, cov, n_std, ax, **kwargs):
    """Draw the n_std contour of a 2D Gaussian as an ellipse."""
    from matplotlib.patches import Ellipse

    evals, evecs = np.linalg.eigh(cov)
    theta = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
    width, height = 2 * n_std * np.sqrt(evals[-1]), 2 * n_std * np.sqrt(evals[0])
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor="none", **kwargs)
    return ax.add_patch(ellipse)


def main(
    output_dir: Path = typer.Argument(Path("figures/")),
    transparent: bool = False,
    context: str = "paper",
    style: str = "ticks",
    palette: str = "muted",
    width: float = typer.Option(5.0, "--width", "-w"),
    aspect: float = typer.Option(GOLDEN_RATIO, "--aspect", "-a"),
    dpi: float = 300.0,
    extension: Optional[List[str]] = typer.Option(None, "--extension", "-e"),
):
    extension = extension or ["png"]

    height = width / aspect
    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": (width, height),
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_dir.mkdir(parents=True, exist_ok=True)

    # constants (identical to Parts II and III)
    num_index_points = 512
    input_dim = 1
    num_train = 128
    degree = 3
    alpha = 2.0
    seed = 8888

    random_state = np.random.RandomState(seed)

    X_grid = np.linspace(-8.0, 8.0, num_index_points).reshape(-1, input_dim)

    p = norm(loc=1.0, scale=1.0)
    q = norm(loc=0.0, scale=2.0)

    X_p, X_q = draw_samples(num_train, p, q, rate=0.5, random_state=random_state)
    X_train, y_train = make_dataset(X_p, X_q)

    kappa = y_train - 0.5
    Phi = basis_function(X_train, degree=degree)
    Phi_grid = basis_function(X_grid, degree=degree)

    # inference
    mu, Sigma, xi = cavi(Phi, kappa, alpha)
    beta_gibbs = gibbs(Phi, kappa, alpha, seed=seed)

    rng = np.random.default_rng(seed)
    beta_cavi = rng.multivariate_normal(mean=mu, cov=Sigma, size=len(beta_gibbs))

    # summary for the post prose
    print("Gibbs posterior mean:", beta_gibbs.mean(axis=0))
    print("CAVI  posterior mean:", mu)
    print("Gibbs posterior sd:  ", beta_gibbs.std(axis=0))
    print("CAVI  posterior sd:  ", np.sqrt(np.diag(Sigma)))
    print("sd ratio (CAVI/Gibbs):",
          np.sqrt(np.diag(Sigma)) / beta_gibbs.std(axis=0))

    # -- figure 1: posterior predictive class probability -------------------
    pi_gibbs = expit(beta_gibbs @ Phi_grid.T)   # (S, num_index_points)
    pi_cavi = expit(beta_cavi @ Phi_grid.T)

    fig, ax = plt.subplots()

    ax.plot(X_grid, class_probability(X_grid, p, q), c='k',
            label=r"$\frac{p(x)}{p(x) + q(x)}$")

    for pi, name, color, ls in [(pi_gibbs, "Gibbs", "tab:green", "solid"),
                                (pi_cavi, "CAVI", "tab:purple", "dashed")]:
        lo, hi = np.quantile(pi, q=[0.025, 0.975], axis=0)
        ax.plot(X_grid, pi.mean(axis=0), color=color, linestyle=ls, label=name)
        ax.fill_between(X_grid.squeeze(axis=-1), lo, hi,
                        color=color, alpha=0.2, linewidth=0)

    ax.scatter(X_p, np.ones_like(X_p), marker='s', edgecolors="none",
               alpha=0.7, zorder=2)
    ax.scatter(X_q, np.zeros_like(X_q), marker='s', edgecolors="none",
               alpha=0.7, zorder=2)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_dir / f"class_prob_compare_{context}_{suffix}.{ext}",
                    dpi=dpi, transparent=transparent)
    plt.close(fig)

    # -- figure 2: posterior over beta --------------------------------------
    latent_dim = Phi.shape[-1]
    pairs = [(i, j) for j in range(latent_dim) for i in range(j)]

    s = height  # square panels
    fig, axes = plt.subplots(1, len(pairs), figsize=(len(pairs) * s, s),
                             squeeze=False)

    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(beta_gibbs[:, i], beta_gibbs[:, j], s=2.0,
                   color="tab:green", alpha=0.15, edgecolors="none",
                   rasterized=True, label="Gibbs")
        for n_std, lw in [(1, 1.0), (2, 0.8), (3, 0.6)]:
            cov_ellipse(mu[[i, j]], Sigma[np.ix_([i, j], [i, j])], n_std, ax,
                        edgecolor="tab:purple", linewidth=lw, zorder=3)
        ax.scatter(*mu[[i, j]], marker='x', color="tab:purple", zorder=4,
                   label=r"CAVI $q(\beta)$")
        ax.set_xlabel(rf"$\beta_{i}$")
        ax.set_ylabel(rf"$\beta_{j}$")

    axes.flat[-1].legend(loc="upper right")

    sns.despine(fig=fig, offset=1, trim=True)
    plt.tight_layout()

    suffix_wide = f"{len(pairs)*s*dpi:.0f}x{s*dpi:.0f}"
    for ext in extension:
        fig.savefig(output_dir / f"beta_posterior_compare_{context}_{suffix_wide}.{ext}",
                    dpi=dpi, transparent=transparent)
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)  # pragma: no cover
