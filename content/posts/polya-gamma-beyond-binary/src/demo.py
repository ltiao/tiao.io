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
"""Three inference schemes for a 3-class logistic-type model (Part IV).

One synthetic 1-D, K=3 problem (seed 8888), three posterior approximations
in the same softmax-linear model with quadratic features:

- Bouchard variational Bayes   (softmax bound + Jaakkola-Jordan tightening)
- stick-breaking PG Gibbs       (Linderman et al.; two binary PG chains)
- one-vs-each PG Gibbs           (Titsias pseudo-likelihood, PG augmented)

Produces two figures plus a hero variant:

- class_prob_compare_*    per-class posterior class probability vs truth
- slack_decomposition_*   Bouchard bound slack split into its two sources
- featured.png            clean hero variant of the slack decomposition

Self-contained; run with:  uv run demo.py ../figures/
(Uses the maintained `polyagamma` package, not `pypolyagamma`.)
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer

from pathlib import Path
from typing import List, Optional
from scipy.stats import norm
from scipy.special import expit, softmax, logsumexp
from polyagamma import random_polyagamma

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))

SEED = 8888
NUM_CLASSES = 3
DEGREE = 3
ALPHA = 2.0  # prior precision on every weight vector

# class-conditional densities (equal priors)
CLASS_MEANS = [-2.0, 0.5, 2.5]
CLASS_SDS = [0.8, 1.0, 0.7]
PER_CLASS = 50

METHOD_STYLE = {
    "Bouchard VB": dict(color="tab:purple", ls="dashed"),
    "stick-breaking": dict(color="tab:green", ls="solid"),
    "one-vs-each": dict(color="tab:orange", ls="dashdot"),
}


# -- shared pieces ----------------------------------------------------------

def basis_function(x, degree=DEGREE):
    """Feature map phi(x) = [1, x, x^2, ...]; x is (M, 1) -> (M, degree)."""
    return np.power(x, np.arange(degree))


def lambd(xi):
    """lambda(xi) = tanh(xi/2) / (2 xi); -> 1/4 as xi -> 0."""
    return 0.5 * np.tanh(0.5 * xi) / xi


def softplus(x):
    """varsigma(x) = log(1 + e^x), stable."""
    return np.logaddexp(0.0, x)


def g_bound(x, xi):
    """Jaakkola-Jordan quadratic upper bound on softplus(x), tight at x=+/-xi."""
    return 0.5 * (x - xi) + 0.5 * lambd(xi) * (x**2 - xi**2) + softplus(xi)


def make_dataset(seed=SEED):
    """N=150 points, 50 per class, drawn class-by-class in order 0,1,2."""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for k, (loc, scale) in enumerate(zip(CLASS_MEANS, CLASS_SDS)):
        xs.append(rng.normal(loc, scale, size=PER_CLASS))
        ys.append(np.full(PER_CLASS, k))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return x.reshape(-1, 1), y


def true_posterior(x_grid):
    """Analytic pi_k(x) = p_k(x) / sum_j p_j(x), equal priors."""
    pdfs = np.stack(
        [norm(loc, scale).pdf(x_grid.squeeze(-1))
         for loc, scale in zip(CLASS_MEANS, CLASS_SDS)],
        axis=0,
    )  # (K, G)
    return pdfs / pdfs.sum(axis=0, keepdims=True)


# -- Method 1: Bouchard variational Bayes -----------------------------------

def bouchard_elbo(Phi, Y, mu, Sigma, xi, a, alpha_prior):
    """Expected Bouchard lower bound minus sum_k KL[q(beta_k) || prior]."""
    N, D = Phi.shape
    K = Y.shape[1]
    m = Phi @ mu.T  # (N, K) = E[psi]
    s2 = np.stack(
        [np.einsum("nd,de,ne->n", Phi, Sigma[k], Phi) for k in range(K)], axis=1
    )  # (N, K) = Var[psi]

    z = m - a[:, None]
    Eg = 0.5 * (z - xi) + 0.5 * lambd(xi) * (z**2 + s2 - xi**2) + softplus(xi)

    E_psi_y = np.sum(m * Y, axis=1)  # E[psi_{n, y_n}]
    ell = np.sum(E_psi_y - a - np.sum(Eg, axis=1))

    kl = 0.0
    for k in range(K):
        _, logdet = np.linalg.slogdet(Sigma[k])
        kl += 0.5 * (
            alpha_prior * np.trace(Sigma[k])
            + alpha_prior * mu[k] @ mu[k]
            - D - D * np.log(alpha_prior) - logdet
        )
    return ell - kl


def bouchard_vb(Phi, y, alpha_prior, K, max_sweeps=300, tol=1e-10):
    """Coordinate-ascent VB. Returns mu (K,D), Sigma (K,D,D), xi (N,K), a (N,)."""
    N, D = Phi.shape
    eye = np.eye(D)
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0

    xi = np.ones((N, K))          # init xi = 1
    a = np.zeros(N)               # init alpha = 0
    mu = np.zeros((K, D))
    Sigma = np.stack([eye / alpha_prior for _ in range(K)])

    elbos = []
    for sweep in range(max_sweeps):
        lam = lambd(xi)  # (N, K)

        # 1. q(beta_k) = N(mu_k, Sigma_k), independent per class
        for k in range(K):
            Lam_k = alpha_prior * eye + (Phi.T * lam[:, k]) @ Phi
            b_k = Phi.T @ (Y[:, k] - 0.5 + lam[:, k] * a)
            Sigma[k] = np.linalg.inv(Lam_k)
            mu[k] = Sigma[k] @ b_k

        # 2. xi_{nk}^2 = E[(psi_{nk} - a_n)^2]
        m = Phi @ mu.T
        s2 = np.stack(
            [np.einsum("nd,de,ne->n", Phi, Sigma[k], Phi) for k in range(K)], axis=1
        )
        xi = np.sqrt(np.maximum((m - a[:, None]) ** 2 + s2, 1e-12))

        # 3. a_n
        lam = lambd(xi)
        a = (0.5 * K - 1.0 + np.sum(lam * m, axis=1)) / np.sum(lam, axis=1)

        L = bouchard_elbo(Phi, Y, mu, Sigma, xi, a, alpha_prior)
        elbos.append(L)
        if sweep > 0:
            assert L >= elbos[-2] - 1e-8, (
                f"Bouchard ELBO decreased at sweep {sweep}: "
                f"{elbos[-2]:.10f} -> {L:.10f}"
            )
            if abs(L - elbos[-2]) < tol:
                break

    return mu, Sigma, xi, a, elbos


def bouchard_predict(mu, Sigma, Phi_grid, num_samples=2000, seed=SEED):
    """S draws beta_k ~ q, softmax per draw, averaged."""
    rng = np.random.default_rng(seed)
    K = mu.shape[0]
    B = np.stack(
        [rng.multivariate_normal(mu[k], Sigma[k], size=num_samples) for k in range(K)],
        axis=1,
    )  # (S, K, D)
    Psi = np.einsum("skd,gd->skg", B, Phi_grid)  # (S, K, G)
    return softmax(Psi, axis=1).mean(axis=0)     # (K, G)


# -- Method 2: stick-breaking PG Gibbs --------------------------------------

def gibbs_binary(Phi, z, alpha_prior, n_iter=4000, n_burn=1000, rng=None):
    """Binary PG Gibbs (Part II). Returns kept beta samples (n_kept, D)."""
    N, D = Phi.shape
    eye = np.eye(D)
    kappa = z - 0.5
    beta = np.zeros(D)

    samples = []
    for _ in range(n_iter):
        omega = random_polyagamma(1.0, Phi @ beta, random_state=rng)
        prec = alpha_prior * eye + (omega * Phi.T) @ Phi
        mean = np.linalg.solve(prec, Phi.T @ kappa)
        beta = rng.multivariate_normal(mean, np.linalg.inv(prec))
        samples.append(beta)
    return np.array(samples[n_burn:])


def stick_breaking_gibbs(Phi, y, alpha_prior, n_iter=4000, n_burn=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    # stick 0: is y == 0?  (all points)
    z0 = (y == 0).astype(float)
    beta0 = gibbs_binary(Phi, z0, alpha_prior, n_iter, n_burn, rng)
    # stick 1: given y != 0, is y == 1?  (points with y in {1, 2})
    mask = y != 0
    z1 = (y[mask] == 1).astype(float)
    beta1 = gibbs_binary(Phi[mask], z1, alpha_prior, n_iter, n_burn, rng)
    return beta0, beta1


def stick_breaking_predict(beta0, beta1, Phi_grid):
    """Average the stick-breaking class probabilities over kept samples."""
    s0 = expit(beta0 @ Phi_grid.T)  # (S, G) = sigma(eta_0)
    s1 = expit(beta1 @ Phi_grid.T)  # (S, G) = sigma(eta_1)
    pi0 = s0
    pi1 = (1.0 - s0) * s1
    pi2 = (1.0 - s0) * (1.0 - s1)
    return np.stack([pi0.mean(0), pi1.mean(0), pi2.mean(0)], axis=0)  # (K, G)


# -- Method 3: one-vs-each PG Gibbs -----------------------------------------

def build_ove_design(Phi, y, K):
    """Stack a_{nk} = (e_{y_n} - e_k) kron phi_n for every k != y_n. -> (M, K*D)."""
    eye_k = np.eye(K)
    rows = []
    for n in range(len(y)):
        for k in range(K):
            if k != y[n]:
                rows.append(np.kron(eye_k[y[n]] - eye_k[k], Phi[n]))
    return np.array(rows)  # (M, K*D)


def ove_gibbs(A, alpha_prior, n_iter=4000, n_burn=1000, seed=SEED):
    """One-vs-each pseudo-posterior via PG augmentation. Returns beta (n_kept, K*D)."""
    M, P = A.shape
    eye = np.eye(P)
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    rhs = 0.5 * A.sum(axis=0)  # sum_{n,k} kappa a_{nk}, kappa = 1/2

    samples = []
    for _ in range(n_iter):
        omega = random_polyagamma(1.0, A @ beta, random_state=rng)
        prec = alpha_prior * eye + (A.T * omega) @ A
        mean = np.linalg.solve(prec, rhs)
        beta = rng.multivariate_normal(mean, np.linalg.inv(prec))
        samples.append(beta)
    return np.array(samples[n_burn:])


def ove_predict(betas, Phi_grid, K):
    """Average softmax(psi(x)) over kept beta (pseudo-posterior)."""
    S = betas.shape[0]
    B = betas.reshape(S, K, -1)                    # row k = beta_k
    Psi = np.einsum("skd,gd->skg", B, Phi_grid)    # (S, K, G)
    return softmax(Psi, axis=1).mean(axis=0)       # (K, G)


# -- slack decomposition (Bouchard) -----------------------------------------

def slack_components(Phi, mu, Sigma, xi, a, num_samples=4000, seed=SEED):
    """Per training point, the two sources of Bouchard bound slack (MC).

    (a) tilting slack    = sum_k E_q[ g(psi_k - a, xi_k) - varsigma(psi_k - a) ]
    (b) independence slk = E_q[ a + sum_k varsigma(psi_k - a) - LSE(psi) ]
    Both are >= 0 for every draw, so their per-point means are >= 0.
    """
    rng = np.random.default_rng(seed + 1)
    K = mu.shape[0]
    B = np.stack(
        [rng.multivariate_normal(mu[k], Sigma[k], size=num_samples) for k in range(K)],
        axis=1,
    )  # (S, K, D)
    Psi = np.einsum("skd,nd->snk", B, Phi)  # (S, N, K)

    z = Psi - a[None, :, None]
    gv = g_bound(z, xi[None, :, :])
    sp = softplus(z)

    comp_a = np.sum(gv - sp, axis=2).mean(axis=0)                      # (N,)
    lse = logsumexp(Psi, axis=2)                                      # (S, N)
    comp_b = (a[None, :] + np.sum(sp, axis=2) - lse).mean(axis=0)     # (N,)

    assert comp_a.min() >= -1e-9, f"tilting slack went negative: {comp_a.min():.2e}"
    assert comp_b.min() >= -1e-9, f"independence slack went negative: {comp_b.min():.2e}"
    return comp_a, comp_b


def draw_slack(ax, x_train, comp_a, comp_b, colors, minimal=False):
    """Stacked-area slack decomposition onto ax (independence bottom, tilting top)."""
    xs = x_train.squeeze(-1)
    order = np.argsort(xs)
    xo, a_o, b_o = xs[order], comp_a[order], comp_b[order]
    c_ind, c_tilt = colors

    ax.fill_between(xo, 0.0, b_o, color=c_ind, alpha=0.75, linewidth=0,
                    label="independence slack")
    ax.fill_between(xo, b_o, b_o + a_o, color=c_tilt, alpha=0.75, linewidth=0,
                    label="tilting (KL) slack")
    ax.plot(xo, a_o + b_o, color="k", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"$x_n$")
    if not minimal:
        ax.set_ylabel(r"expected bound slack (nats)")
    ax.set_ylim(bottom=0.0)


# -- driver -----------------------------------------------------------------

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
    muted = sns.color_palette("muted")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- data & grid --------------------------------------------------------
    K = NUM_CLASSES
    X_train, y = make_dataset()
    Phi = basis_function(X_train)
    X_grid = np.linspace(-6.0, 6.0, 512).reshape(-1, 1)
    Phi_grid = basis_function(X_grid)
    pi_true = true_posterior(X_grid)  # (K, G)

    timings = {}

    # -- Method 1: Bouchard VB ---------------------------------------------
    t0 = time.perf_counter()
    mu, Sigma, xi, a, elbos = bouchard_vb(Phi, y, ALPHA, K)
    pi_bouchard = bouchard_predict(mu, Sigma, Phi_grid)
    timings["Bouchard VB"] = time.perf_counter() - t0
    print(f"Bouchard VB: {len(elbos)} sweeps, final ELBO = {elbos[-1]:.6f}")

    # -- Method 2: stick-breaking PG Gibbs ---------------------------------
    t0 = time.perf_counter()
    beta0, beta1 = stick_breaking_gibbs(Phi, y, ALPHA)
    pi_stick = stick_breaking_predict(beta0, beta1, Phi_grid)
    timings["stick-breaking"] = time.perf_counter() - t0

    # -- Method 3: one-vs-each PG Gibbs ------------------------------------
    t0 = time.perf_counter()
    A = build_ove_design(Phi, y, K)
    beta_ove = ove_gibbs(A, ALPHA)
    pi_ove = ove_predict(beta_ove, Phi_grid, K)
    timings["one-vs-each"] = time.perf_counter() - t0

    preds = {
        "Bouchard VB": pi_bouchard,
        "stick-breaking": pi_stick,
        "one-vs-each": pi_ove,
    }

    # -- report: mean absolute error over grid and classes -----------------
    print("\nmean |pi_hat - pi_true| (over grid and classes):")
    for name, pi in preds.items():
        mae = np.mean(np.abs(pi - pi_true))
        print(f"  {name:16s} {mae:.5f}   (wall-clock {timings[name]:.2f}s)")

    # -- slack decomposition ------------------------------------------------
    comp_a, comp_b = slack_components(Phi, mu, Sigma, xi, a)
    total_a, total_b = comp_a.sum(), comp_b.sum()
    print("\nBouchard bound slack (summed over training points):")
    print(f"  tilting (KL) slack   {total_a:.4f}   (mean/point {comp_a.mean():.4f})")
    print(f"  independence slack   {total_b:.4f}   (mean/point {comp_b.mean():.4f})")
    print(f"  total                {total_a + total_b:.4f}")

    # -- figure 1: per-class posterior class probability -------------------
    s = height
    fig, axes = plt.subplots(1, K, figsize=(K * s, s), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.plot(X_grid.squeeze(-1), pi_true[k], c="k", label="truth")
        for name, pi in preds.items():
            ax.plot(X_grid.squeeze(-1), pi[k], label=name, **METHOD_STYLE[name])
        for c in range(K):
            sns.rugplot(x=X_train[y == c].squeeze(-1), ax=ax, color=muted[c],
                        height=0.05, lw=1.0, alpha=0.7)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(rf"$\pi_{k}(x)$")
        ax.set_ylim(-0.03, 1.03)
    axes.flat[-1].legend(loc="upper left", fontsize="small")

    sns.despine(fig=fig, offset=1, trim=True)
    plt.tight_layout()

    suffix1 = f"{K*s*dpi:.0f}x{s*dpi:.0f}"
    for ext in extension:
        fig.savefig(output_dir / f"class_prob_compare_{context}_{suffix1}.{ext}",
                    dpi=dpi, transparent=transparent)
    plt.close(fig)

    # -- figure 2: slack decomposition -------------------------------------
    slack_colors = (muted[0], muted[3])  # blue = independence, red = tilting
    fig, ax = plt.subplots()
    draw_slack(ax, X_train, comp_a, comp_b, slack_colors)
    ax.legend(loc="upper right", fontsize="small")

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_dir / f"slack_decomposition_{context}_{suffix}.{ext}",
                    dpi=dpi, transparent=transparent)
    plt.close(fig)

    # -- figure 3: featured hero (clean variant of figure 2) ---------------
    hero_w = 8.0
    fig, ax = plt.subplots(figsize=(hero_w, hero_w / GOLDEN_RATIO))
    draw_slack(ax, X_train, comp_a, comp_b, slack_colors, minimal=True)
    ax.set_ylabel("")
    ax.legend(loc="upper right", frameon=False)
    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    plt.tight_layout()
    fig.savefig(output_dir.parent / "featured.png", dpi=dpi, transparent=transparent)
    plt.close(fig)

    print("\nwrote:")
    print(f"  {output_dir / f'class_prob_compare_{context}_{suffix1}.png'}")
    print(f"  {output_dir / f'slack_decomposition_{context}_{suffix}.png'}")
    print(f"  {output_dir.parent / 'featured.png'}")


if __name__ == "__main__":
    typer.run(main)  # pragma: no cover
