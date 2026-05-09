import sys
import click

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.special import expit
from sklearn.utils import check_random_state
from pypolyagamma import PyPolyaGamma

from itertools import islice
from pathlib import Path


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


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

    Sigma_inv = (omega * Phi.T) @ Phi + alpha * np.eye(latent_dim)

    mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
    Sigma = np.linalg.solve(Sigma_inv, np.eye(latent_dim))

    return mu, Sigma


def conditional_posterior_auxiliary(Phi, beta):

    c = Phi @ beta
    b = np.ones_like(c)

    return b, c


def gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state):
    b, c = conditional_posterior_auxiliary(Phi, beta)
    omega = polya_gamma_sample(b, c, pg=pg)
    yield from gibbs_sampler_helper(omega, Phi, kappa, alpha, pg, random_state)


def gibbs_sampler_helper(omega, Phi, kappa, alpha, pg, random_state):
    mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, omega)
    beta = gassian_sample(mu, Sigma, random_state=random_state)
    yield beta, omega
    yield from gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state)


def plot_func(func, X, ax=None):
    # c="tab:red",
    return ax.plot(X, func(X), linestyle="dashed")


def logit(x, p, q):

    return p.logpdf(x) - q.logpdf(x)


def density_ratio(x, p, q):

    return np.exp(logit(x, p, q))


def class_probability(x, p, q):

    return expit(logit(x, p, q))


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


@click.command()
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option("--num-samples", "-s", default=5)
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--palette-minor', default="crest")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
def main(output_dir, num_samples, transparent, context, style, palette,
         palette_minor, width, height, aspect, dpi, extension):

    # preamble
    if height is None:
        height = width / aspect
    # height *= num_iterations
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # constants
    num_index_points = 512
    input_dim = 1
    num_train = 128
    num_iterations = 1000
    degree = 3

    seed = 8888
    random_state = np.random.RandomState(seed)

    x_min, x_max = -5.0, 5.0

    X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, input_dim)

    p = norm(loc=1.0, scale=1.0)
    q = norm(loc=0.0, scale=2.0)

    X_p, X_q = draw_samples(num_train, p, q, rate=0.5, random_state=random_state)
    X_train, y_train = make_dataset(X_p, X_q)

    kappa = y_train - 0.5

    Phi_grid = basis_function(X_grid, degree=degree)
    Phi = basis_function(X_train, degree=degree)
    latent_dim = Phi.shape[-1]

    # m = np.zeros(latent_dim)

    alpha = 2.0  # prior precision
    # S_inv = np.eye(latent_dim) / alpha

    # A = random_state.rand(latent_dim, latent_dim)
    # B = np.linalg.inv(A)

    # S_inv = A.T @ A
    # S = B @ B.T

    # Sm = np.linalg.solve(S_inv, m)

    pg = PyPolyaGamma(seed=seed)

    beta = random_state.normal(size=latent_dim, scale=1/np.sqrt(alpha))
    # beta = random_state.multivariate_normal(mean=m, cov=S_inv)

    betas, omegas = zip(*islice(gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state), num_iterations))

    # betas = []
    # omegas = []
    # for i in range(num_iterations):

    #     b, c = conditional_posterior_auxiliary(Phi, beta)
    #     omega = polya_gamma_sample(b, c, pg=pg)

    #     mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, omega)
    #     beta = gassian_sample(mu, Sigma, random_state=random_state)

    #     omegas.append(omega)
    #     betas.append(beta)

    frame = pd.DataFrame(data=expit(Phi_grid @ np.vstack(betas).T),
                         index=X_grid.squeeze(axis=-1))
    frame.columns.name = "iteration"
    frame.index.name = "x"

    s = frame.stack()
    s.name = "p"

    data = s.reset_index()

    fig, ax = plt.subplots()

    ax.scatter(X_p, np.ones_like(X_p), marker='s', edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(X_q, np.zeros_like(X_q), marker='s', edgecolors="none", alpha=0.7, zorder=2)

    ax.plot(X_grid, class_probability(X_grid, p, q), c='k', label=r"$\frac{p(x)}{p(x) + q(x)}$")

    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"class_prob_true_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    sns.lineplot(x="x", y="p", hue="iteration", units="iteration", estimator=None,
                 palette=palette_minor, linewidth=0.1, alpha=0.6,
                 data=data.query("iteration % 2 == 0"), ax=ax)

    ax.scatter(X_p, np.ones_like(X_p), marker='s', edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(X_q, np.zeros_like(X_q), marker='s', edgecolors="none", alpha=0.7, zorder=2)

    ax.plot(X_grid, class_probability(X_grid, p, q), c='k', label=r"$\frac{p(x)}{p(x) + q(x)}$")

    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"class_prob_pred_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    sns.lineplot(x="x", y="p", hue="iteration", units="iteration", estimator=None,
                 palette=palette_minor, linewidth=0.1, alpha=0.6, legend=False,
                 data=data.query("iteration % 2 == 0"), ax=ax)

    ax.scatter(X_p, np.ones_like(X_p), marker='s', edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(X_q, np.zeros_like(X_q), marker='s', edgecolors="none", alpha=0.7, zorder=2)

    ax.plot(X_grid, class_probability(X_grid, p, q), c='k', label=r"$\frac{p(x)}{p(x) + q(x)}$")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    # ax.axis("off")
    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"header_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    data = pd.DataFrame(data=np.vstack(betas), columns=map(r"$\beta_{{{:d}}}$".format, range(latent_dim)))
    data.index.name = "iteration"
    data.reset_index(inplace=True)

    g = sns.PairGrid(data.query("iteration % 2 == 0"), hue="iteration",
                     palette=palette_minor, corner=True,
                     height=height, aspect=aspect)
    g = g.map_lower(plt.scatter, facecolor="none", alpha=0.3)

    for ext in extension:
        g.savefig(output_path.joinpath(f"beta_{context}_{suffix}.{ext}"),
                  dpi=dpi, transparent=transparent)

    fig, ax = plt.subplots()

    p_line, = ax.plot(X_grid, p.pdf(X_grid), label=r'$p(x)$')
    q_line, = ax.plot(X_grid, q.pdf(X_grid), label=r'$q(x)$')

    ax.set_ylim(-0.02, None)

    sns.rugplot(X_p, color=p_line.get_color(), alpha=0.6, ax=ax)
    sns.rugplot(X_q, color=q_line.get_color(), alpha=0.6, ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'density')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"density_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    frame = pd.DataFrame(data=np.vstack(omegas), columns=X_train.squeeze(axis=-1))
    frame.index.name = "iteration"
    frame.columns.name = "x"

    s = frame.stack()
    s.name = "omega"

    data = s.reset_index()

    fig, ax = plt.subplots()

    sns.scatterplot(x="x", y="omega", hue="iteration", palette=palette_minor,
                    facecolor="none", alpha=0.3, marker="_",
                    data=data.query("iteration % 2 == 0"), ax=ax)

    ax.set_ylim(-0.02, None)

    sns.rugplot(X_p, alpha=0.6, ax=ax)
    sns.rugplot(X_q, alpha=0.6, ax=ax)
    # sns.rugplot(X_p, color=p_line.get_color(), alpha=0.6, ax=ax)
    # sns.rugplot(X_q, color=q_line.get_color(), alpha=0.6, ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\omega$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"omega_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    ax.plot(X_grid, logit(X_grid, p, q), c='k', label=r"$f(x) = \log p(x) - \log q(x)$")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logit_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    ax.plot(X_grid, density_ratio(X_grid, p, q), c='k', label=r"$r(x) = \exp f(x)$")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$r(x)$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
