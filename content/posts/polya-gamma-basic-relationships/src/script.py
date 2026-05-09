import sys
import click

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import expit
from pypolyagamma import PyPolyaGamma

from pathlib import Path


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def func(u, omega):
    return 0.5 * np.exp(-0.5*u*(u*omega - 1.0))


def make_data(x, omegas):

    frame = pd.DataFrame(data=func(x, omegas),
                         index=x.squeeze(axis=-1),
                         columns=omegas)
    frame.columns.name = "omega"
    frame.index.name = "u"

    s = frame.stack()
    s.name = "y"

    data = s.reset_index()

    return data


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
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
def main(output_dir, num_samples, transparent, context, style, palette, width,
         height, aspect, dpi, extension):

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
    b, c = 1, 0

    num_index_points = 512
    num_features = 1
    num_samples = 32

    seed = 8889

    x_min, x_max = -8.0, 8.0

    X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

    bs = np.arange(0.25, 3., 0.75)
    cs = np.linspace(-3, 3, 5)

    rows = []
    for b in bs:
        for c in cs:
            pg = PyPolyaGamma(seed=seed)
            for i in range(num_samples):
                omega = pg.pgdraw(b, c)
                rows.append(dict(i=i, b=b, c=c, omega=omega))

    data = pd.DataFrame(rows).rename(columns=dict(b=r"$b$", c=r"$c$",
                                                  omega=r"$\omega$"))

    fig, ax = plt.subplots()

    ax.set_title(r"Samples $\omega \sim \mathrm{PG}(\omega | b, c)$")

    sns.stripplot(x=r"$c$", y=r"$\omega$", hue=r"$b$",
                  dodge=True, jitter=True,
                  alpha=0.6, palette="flare",
                  data=data, ax=ax)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"samples_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    plot_func(expit, X_grid, ax=ax)

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"sigmoid_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    plot_func(np.cosh, X_grid, ax=ax)

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"cosh_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    omegas = np.linspace(0.1, 1.6, 50)
    data = make_data(X_grid, omegas).rename(columns=dict(u=r"$u$", y=r"$y$",
                                                         omega=r"$\omega$"))

    fig, ax = plt.subplots()

    plot_func(expit, X_grid, ax=ax)

    sns.lineplot(x=r"$u$", y=r"$y$", hue=r"$\omega$",
                 linewidth=0.5, alpha=0.8,
                 palette="crest", data=data)

    # ax.set_ylim(-0.04, 1.04)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"grid_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    pg = PyPolyaGamma(seed=seed)
    omegas = np.asarray([pg.pgdraw(1, 0) for i in range(1024)])
    data = make_data(X_grid, omegas).rename(columns=dict(u=r"$u$", y=r"$y$",
                                                         omega=r"$\omega$"))

    fig, ax = plt.subplots()

    plot_func(expit, X_grid, ax=ax)

    sns.lineplot(x=r"$u$", y=r"$y$",
                 hue=r"$\omega$",
                 units=r"$\omega$", estimator=None,
                 linewidth=0.1, alpha=0.8,
                 palette="crest",
                 data=data, ax=ax)
    sns.lineplot(x=r"$u$", y=r"$y$", ci=None,
                 # color="tab:blue",
                 data=data, legend=False, alpha=0.75, ax=ax)

    ax.set_ylim(-0.1, 2.1)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"monte_carlo_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    plot_func(expit, X_grid, ax=ax)

    sns.lineplot(x=r"$u$", y=r"$y$",
                 hue=r"$\omega$",
                 units=r"$\omega$", estimator=None,
                 linewidth=0.1, alpha=0.8,
                 palette="crest",
                 data=data, legend=False, ax=ax)
    sns.lineplot(x=r"$u$", y=r"$y$", ci=None,
                 # color="tab:blue",
                 data=data, legend=False, alpha=0.75, ax=ax)

    ax.set_ylim(-0.1, 2.1)

    ax.axis("off")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"header_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
