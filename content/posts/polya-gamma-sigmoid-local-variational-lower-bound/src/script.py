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

from pathlib import Path

from utils import (first_moment, softplus, softplus_upper_bound,
                   sigmoid_lower_bound)

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


def gassian_sample(mean, cov, random_state=None):
    random_state = check_random_state(random_state)
    return random_state.multivariate_normal(mean=mean, cov=cov)


def conditional_posterior_weights(Phi, kappa, alpha, lambd):

    latent_dim = Phi.shape[-1]

    Sigma_inv = (lambd * Phi.T) @ Phi + alpha * np.eye(latent_dim)

    mu = np.linalg.solve(Sigma_inv, Phi.T @ kappa)
    Sigma = np.linalg.solve(Sigma_inv, np.eye(latent_dim))

    return mu, Sigma

# def kappa(y):
#     return y - 0.5


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


def plot_func(func, X, ax=None):
    # c="tab:red",
    return ax.plot(X, func(X), linestyle="dashed")


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
    num_iterations = 2500
    degree = 3

    seed = 8888
    random_state = np.random.RandomState(seed)

    x_min, x_max = -8.0, 8.0

    X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, input_dim)

    p = norm(loc=1.0, scale=1.0)
    q = norm(loc=0.0, scale=2.0)

    X_p, X_q = draw_samples(num_train, p, q, rate=0.5, random_state=random_state)
    X_train, y_train = make_dataset(X_p, X_q)

    kappa = y_train - 0.5

    Phi_grid = basis_function(X_grid, degree=degree)
    Phi = basis_function(X_train, degree=degree)
    latent_dim = Phi.shape[-1]

    alpha = 2.0  # prior precision

    pg = PyPolyaGamma(seed=seed)

    # beta = random_state.normal(size=latent_dim, scale=1/np.sqrt(alpha))
    # beta = random_state.multivariate_normal(mean=m, cov=S_inv)
    # betas, omegas = zip(*islice(gibbs_sampler(beta, Phi, kappa, alpha, pg, random_state), num_iterations))

    print(np.allclose(np.einsum('ij,ij->i', Phi, Phi), np.diag(Phi @ Phi.T)))

    xi = 1e-1 * np.ones(num_train)
    lambd = first_moment(xi)

    mus = []
    xis = []
    for i in range(num_iterations):

        # E-step
        mu, Sigma = conditional_posterior_weights(Phi, kappa, alpha, lambd)

        # M-step
        xi_squared = np.einsum('ij,ij->i', Phi @ (Sigma + np.outer(mu, mu)), Phi)
        xi = np.sqrt(xi_squared)

        lambd = first_moment(xi)

        mus.append(mu)
        xis.append(xi)

    frame = pd.DataFrame(data=expit(Phi_grid @ np.vstack(mus).T),
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
                 data=data, ax=ax)

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

    data = pd.DataFrame(data=np.vstack(mus), columns=map(r"$\mu_{{{:d}}}$".format, range(latent_dim)))
    data.index.name = "iteration"
    data.reset_index(inplace=True)

    g = sns.PairGrid(data, hue="iteration",
                     palette=palette_minor, corner=True,
                     height=height, aspect=aspect)
    g = g.map_lower(plt.scatter, facecolor="none", alpha=0.3)

    for ext in extension:
        g.savefig(output_path.joinpath(f"mu_{context}_{suffix}.{ext}"),
                  dpi=dpi, transparent=transparent)

    frame = pd.DataFrame(data=np.vstack(xis),
                         columns=X_train.squeeze(axis=-1))
    frame.index.name = "iteration"
    frame.columns.name = "x"

    s = frame.stack()
    s.name = "xi"

    data = s.reset_index()

    fig, ax = plt.subplots()

    sns.scatterplot(x="x", y="xi", hue="iteration", palette=palette_minor,
                    facecolor="none", alpha=0.3, marker="_", data=data, ax=ax)

    # ax.set_ylim(-0.02, None)

    sns.rugplot(X_p, alpha=0.6, ax=ax)
    sns.rugplot(X_q, alpha=0.6, ax=ax)
    # sns.rugplot(X_p, color=p_line.get_color(), alpha=0.6, ax=ax)
    # sns.rugplot(X_q, color=q_line.get_color(), alpha=0.6, ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\xi$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"xi_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    frame = pd.DataFrame(data=first_moment(np.vstack(xis)),
                         columns=X_train.squeeze(axis=-1))
    frame.index.name = "iteration"
    frame.columns.name = "x"

    s = frame.stack()
    s.name = "lambda"

    data = s.reset_index()

    fig, ax = plt.subplots()

    sns.scatterplot(x="x", y="lambda", hue="iteration", palette=palette_minor,
                    facecolor="none", alpha=0.3, marker="_", data=data, ax=ax)

    # ax.set_ylim(-0.02, None)

    sns.rugplot(X_p, alpha=0.6, ax=ax)
    sns.rugplot(X_q, alpha=0.6, ax=ax)
    # sns.rugplot(X_p, color=p_line.get_color(), alpha=0.6, ax=ax)
    # sns.rugplot(X_q, color=q_line.get_color(), alpha=0.6, ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\lambda$')

    ax.legend()

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"lambda_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0

    xi_grid = np.linspace(1e-8, 6., 50)

    frame = pd.DataFrame(data=softplus_upper_bound(X_grid, xi_grid),
                         index=X_grid.squeeze(axis=-1),
                         columns=xi_grid)
    frame.columns.name = "xi"
    frame.index.name = "x"

    s = frame.stack()
    s.name = "y"

    data = s.reset_index().rename(columns=dict(x=r"$\psi$",
                                               y=r"$g(\psi, \xi)$",
                                               xi=r"$\xi$"))

    fig, ax = plt.subplots()

    # plot_func(softplus, X_grid, ax=ax)

    sns.lineplot(x=r"$\psi$", y=r"$g(\psi, \xi)$", hue=r"$\xi$",
                 linewidth=0.2, alpha=0.6, palette="crest", data=data, ax=ax)
    ax.plot(X_grid, softplus(X_grid), linestyle="dashed")

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"softplus_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    frame = pd.DataFrame(data=sigmoid_lower_bound(X_grid, xi_grid),
                         index=X_grid.squeeze(axis=-1),
                         columns=xi_grid)
    frame.columns.name = "xi"
    frame.index.name = "x"

    s = frame.stack()
    s.name = "y"

    data = s.reset_index().rename(columns=dict(x=r"$\psi$",
                                               y=r"$\ell(\psi, \xi)$",
                                               xi=r"$\xi$"))

    fig, ax = plt.subplots()

    # plot_func(softplus, X_grid, ax=ax)

    sns.lineplot(x=r"$\psi$", y=r"$\ell(\psi, \xi)$", hue=r"$\xi$",
                 linewidth=0.2, alpha=0.6, palette="crest", data=data, ax=ax)
    ax.plot(X_grid, expit(X_grid), linestyle="dashed")

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"sigmoid_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    fig, ax = plt.subplots()

    plot_func(first_moment, X_grid, ax=ax)

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"foo_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    rows = []
    for c in X_grid[::50].squeeze(axis=-1):
        pg = PyPolyaGamma(seed=seed)
        for i in range(num_samples):
            omega = pg.pgdraw(1.0, c)
            rows.append(dict(i=i, c=c, omega=omega))

    data = pd.DataFrame(rows).rename(columns=dict(c=r"$c$", omega=r"$\omega$"))

    fig, ax = plt.subplots()

    ax.set_title(r"Samples $\omega \sim \mathrm{PG}(\omega | 1, c)$")

    sns.scatterplot(x=r"$c$", y=r"$\omega$",  # hue=r"$b$",
                    # dodge=True, jitter=False,
                    alpha=0.6, palette="flare",
                    data=data, ax=ax)

    # plot_func(first_moment, X_grid, ax=ax)
    ax.plot(X_grid, first_moment(X_grid), linestyle="dashed")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"samples_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
