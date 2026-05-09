import sys
import click

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import check_random_state
from scipy.optimize import minimize
from pathlib import Path
from tqdm import trange


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def unsqueeze(axis):

    def unsqueeze_dec(fn):

        def new_fn(input):
            return fn(tf.expand_dims(input, axis=axis))

        return new_fn

    return unsqueeze_dec


def atleast_2d(func):

    def new_func(x):

        return func(np.atleast_2d(x))

    return new_func


def value_and_gradient(value_fn):

    @tf.function
    def value_and_gradient_fn(x):

        # Equivalent to `tfp.math.value_and_gradient(value_fn, x)`, with the
        # only difference that the gradients preserve their `dtype` rather than
        # casting to `tf.float32`, which is problematic for scipy.optimize
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            val = value_fn(x)

        grad = tape.gradient(val, x)

        return val, grad

    return value_and_gradient_fn


def numpy_io(fn):

    def new_fn(*args):

        new_args = map(tf.convert_to_tensor, args)
        outputs = fn(*new_args)
        new_outputs = [output.numpy() for output in outputs]

        return new_outputs

    return new_fn


def forrester(x):
    """
    Forrester's.
    """
    # return (6.0*x-2.0)**2 * np.sin(12.0 * x - 4.0)
    return np.sin(3.0*x) + x**2 - 0.7*x


def convert1(func):

    @numpy_io
    @value_and_gradient
    @unsqueeze(axis=0)  # np.atleast_2d
    def new_func(x):
        return func(index_points=x)

    return new_func


def convert2(func):

    @numpy_io
    @value_and_gradient
    def new_func(x):
        return func(index_points=x)

    return new_func


def convert3(func, ind):

    @numpy_io
    @value_and_gradient
    @unsqueeze(axis=0)  # np.atleast_2d
    def new_func(x):
        ret = func(index_points=x)
        return ret[ind]

    return new_func


def plot_observations(X, y, zorder=1, ax=None):
    return ax.scatter(X, y,  marker="x", color="black",
                      label="observations", zorder=zorder)


def plot_latent_function(func, x, zorder=-1, ax=None):
    return ax.plot(x, func(x), color="tab:gray", label="latent function", zorder=zorder)


def plot_predictive_distribution(x, y_mean, y_stddev, m=2.0,
                                 subscript="n", superscript=None,
                                 mean_label="mean", std_dev_label="std dev",
                                 fill_between_alpha=0.2, zorder=2, ax=None):

    if ax is None:
        ax = plt.gca()

    suffix = rf"_{{{subscript}}}" if subscript is not None else r""
    suffix += rf"^{{{superscript}}}" if superscript is not None else r""

    print(suffix)

    line, = ax.plot(x, y_mean, label=rf"{mean_label} $\mu{suffix}(\mathbf{{x}})$", alpha=0.8,
                    zorder=zorder)
    ax.fill_between(x.squeeze(axis=-1),
                    y_mean - m * y_stddev,
                    y_mean + m * y_stddev,
                    alpha=fill_between_alpha,
                    label=rf"{std_dev_label} $\pm 2 \sigma{suffix}(\mathbf{{x}})$", zorder=zorder)
    return line


def plot_optimization_history(X_history, Y_history, zorder=3, ax=None):

    return ax.quiver(X_history[:-1], Y_history[:-1],
                     X_history[1:] - X_history[:-1],
                     Y_history[1:] - Y_history[:-1],
                     scale_units='xy', angles='xy', scale=1.0, width=3e-3,
                     color="k", linewidth=0.05, zorder=zorder+1)


def plot_optimum(x, y, ymin, color, subscript="n", superscript=None,
                 min_label="min", ax=None):

    suffix = rf"_{{{subscript}}}" if subscript is not None else ""
    suffix += rf"^{{{superscript}}}" if superscript is not None else ""

    ax.axhline(y=y, color=color, linestyle="dashed", linewidth=0.5, zorder=3)
    ax.scatter(x, y, color=color, marker='*',
               label=rf"{min_label} $\tau{suffix}$", zorder=3)
    ax.scatter(x, ymin, color=color, marker="^", alpha=0.8)


def plot_labels(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$")


def broadcast_hstack(a, b):
    """
    Array of shape (n,) and (m, k), broadcast and horizontally stack
    to get array of shape (m, n + k)
    """
    n = len(a)
    m = len(b)

    a_broad = np.broadcast_to(a, shape=(m, n))

    # c = np.append(a_broad, b, axis=axis)
    c = np.hstack([a_broad, b])

    return c


def standardize(frame):
    frame.rename(index=lambda m: rf"sim mean $\mu_{{n+1}}^{{({m+1:d})}}(\mathbf{{x}})$", inplace=True)
    frame.index.name = "sample"
    frame.reset_index(inplace=True)
    return frame


def multi_start(minimizer_fn=minimize):

    def new_minimizer(fn, bounds, num_starts, num_samples=None, random_state=None,
                      *args, **kwargs):

        random_state = check_random_state(random_state)

        assert "x0" not in kwargs, "`x0` should not be specified"
        assert "jac" not in kwargs or kwargs["jac"], "`jac` must be true"

        if num_samples is None:
            num_samples = num_starts
        else:
            assert num_samples >= num_starts

        low, high = zip(*bounds)
        dims = len(bounds)

        X_init = random_state.uniform(low=low, high=high, size=(num_samples, dims))

        values, _ = fn(X_init)
        ind = np.argsort(values)

        results = []
        for i in range(num_starts):
            x_init = X_init[ind[i]]
            result = minimizer_fn(fn, x0=x_init, bounds=bounds, *args, **kwargs)
            results.append(result)

        return results

    return new_minimizer


@click.command()
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--x-cand', default=0.25, type=float)
@click.option('--x-init', default=-0.85, type=float)
@click.option("--num-epochs", "-e", default=500)
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
def main(output_dir, x_cand, x_init, num_epochs, num_samples, transparent,
         context, style, palette, width, height, aspect, dpi, extension):

    x_cand_2d = np.atleast_2d([x_cand])
    x_init_1d = np.array([x_init])

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
    tfd = tfp.distributions
    kernels = tfp.math.psd_kernels
    kernel_cls = kernels.MaternFiveHalves

    num_index_points = 512
    num_features = 1
    num_observations = 10

    noise_variance = 0.05

    learning_rate = 0.05
    beta_1 = 0.5
    beta_2 = 0.99
    jitter = 1e-6

    seed = 8889
    random_state = np.random.RandomState(seed)

    bound = x_min, x_max = -1.0, 2.0
    bounds = [bound]

    X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

    eps = noise_variance * random_state.randn(num_observations, num_features)

    X = x_min + (x_max - x_min) * random_state.rand(num_observations, num_features)
    Y = forrester(X) + eps
    y = Y.squeeze(axis=-1)

    def predictive(x):
        return tfd.GaussianProcessRegressionModel(
                kernel=kernel, index_points=np.atleast_2d(x),
                observation_index_points=X, observations=y,
                observation_noise_variance=observation_noise_variance,
                jitter=jitter)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$y$")

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"observations_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    amplitude = tfp.util.TransformedVariable(
        1.0, bijector=tfp.bijectors.Softplus(), dtype="float64", name='amplitude')
    length_scale = tfp.util.TransformedVariable(
        1.0, bijector=tfp.bijectors.Softplus(), dtype="float64", name='length_scale')
    observation_noise_variance = tfp.util.TransformedVariable(
        1e-1, bijector=tfp.bijectors.Softplus(), dtype="float64",
        name='observation_noise_variance')

    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

    foo = predictive(X_grid)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    plot_predictive_distribution(x=X_grid,
                                 y_mean=foo.mean(),
                                 y_stddev=foo.stddev(), ax=ax)

    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"predictive_mean_before_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    gp = tfd.GaussianProcess(
        kernel=kernel, index_points=X,
        observation_noise_variance=observation_noise_variance)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=beta_1, beta_2=beta_2)

    for epoch in trange(num_epochs):

        with tf.GradientTape() as tape:
            nll = - gp.log_prob(y)

        gradients = tape.gradient(nll, gp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gp.trainable_variables))

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    plot_predictive_distribution(x=X_grid,
                                 y_mean=foo.mean(),
                                 y_stddev=foo.stddev(), ax=ax)
    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"predictive_mean_after_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    func = convert1(foo.mean)
    val, grad = func(x_init_1d)
    click.secho(f"value={val}, gradient={grad}", fg="green")

    x_history = [x_init_1d]
    res = minimize(func, jac=True, x0=x_init_1d, bounds=bounds,
                   method="L-BFGS-B", callback=x_history.append)

    X_history = np.vstack(x_history)
    Y_history = foo.mean(index_points=X_history)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    line = plot_predictive_distribution(x=X_grid,
                                        y_mean=foo.mean(),
                                        y_stddev=foo.stddev(), ax=ax)
    plot_optimization_history(X_history, Y_history, ax=ax)

    ymin, ymax = ax.get_ylim()
    plot_optimum(x=res.x, y=res.fun, ymin=ymin, color=line.get_color(), ax=ax)

    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"predictive_minimum_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    y_cand_sample = predictive(x=x_cand_2d).sample(seed=seed)

    X_next = np.vstack((X, x_cand_2d))
    y_next = np.hstack((y, y_cand_sample))

    bar = tfd.GaussianProcessRegressionModel(
        kernel=kernel, index_points=X_grid,
        observation_index_points=X_next, observations=y_next,
        observation_noise_variance=observation_noise_variance, jitter=jitter)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    line1 = plot_predictive_distribution(x=X_grid,
                                         y_mean=foo.mean(),
                                         y_stddev=foo.stddev(), ax=ax)
    line2 = plot_predictive_distribution(x=X_grid,
                                         y_mean=bar.mean(),
                                         y_stddev=bar.stddev(),
                                         subscript="n+1", superscript="(1)",
                                         mean_label="sim mean", std_dev_label="sim std dev",
                                         zorder=4, ax=ax)

    ax.axvline(x=x_cand_2d, c="tab:gray", linewidth=0.5, zorder=2)
    ax.scatter(x_cand_2d, y_cand_sample, color=line2.get_color(), marker=".",
               zorder=3, label="sim outcome $y_c^{(1)}$")

    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"simulated_predictive_mean_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    minimize_multi_start = multi_start(minimizer_fn=minimize)

    func = convert1(bar.mean)
    res_best = minimize(func, jac=True, x0=x_init_1d, bounds=bounds,
                        method="L-BFGS-B")

    # func = convert2(bar.mean)
    # results = minimize_multi_start(func, bounds,
    #                                num_starts=5, num_samples=10,
    #                                jac=True, method="L-BFGS-B",
    #                                random_state=random_state)
    # res_best = min(filter(lambda r: r.success, results), key=lambda r: r.fun)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    line1 = plot_predictive_distribution(x=X_grid,
                                         y_mean=foo.mean(),
                                         y_stddev=foo.stddev(), ax=ax)
    line2 = plot_predictive_distribution(x=X_grid,
                                         y_mean=bar.mean(),
                                         y_stddev=bar.stddev(),
                                         subscript="n+1", superscript="(1)",
                                         mean_label="sim mean", std_dev_label="sim std dev",
                                         zorder=4, ax=ax)

    ax.axvline(x=x_cand_2d, c="tab:gray", linewidth=0.5, zorder=2)
    ax.scatter(x_cand_2d, y_cand_sample, color=line2.get_color(), marker=".",
               zorder=3, label=r"sim outcome $y_c^{(1)}$")

    ymin, ymax = ax.get_ylim()
    plot_optimum(x=res.x, y=res.fun, ymin=ymin, color=line1.get_color(), ax=ax)
    plot_optimum(x=res_best.x, y=res_best.fun, ymin=ymin,
                 color=line2.get_color(),
                 subscript="n+1", superscript="(1)",
                 min_label="sim min", ax=ax)

    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"simulated_predictive_minimum_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    ###

    # TODO: Keep this as native TF Tensor
    y_cand_samples = predictive(x=x_cand_2d).sample(sample_shape=(num_samples,), seed=seed).numpy().reshape(-1, 1)

    X_next = np.expand_dims(np.vstack((X, x_cand_2d)), axis=0)
    y_next = broadcast_hstack(y, y_cand_samples)

    bar = tfd.GaussianProcessRegressionModel(
        kernel=kernel, index_points=X_grid,
        observation_index_points=X_next, observations=y_next,
        observation_noise_variance=observation_noise_variance, jitter=jitter)

    line_frame = pd.DataFrame(data=bar.mean().numpy(), columns=X_grid.squeeze(axis=-1))
    line_frame.rename(index=r"mean $\mu_{{n+1}}^{{({:d})}}(\mathbf{{x}})$".format, inplace=True)
    line_frame.index.name = "sample"
    line_frame.reset_index(inplace=True)

    line_data = line_frame.melt(id_vars="sample", var_name="x", value_name="y")

    scatter_data_simulated = pd.DataFrame(data=dict(x=x_cand,
                                                    y=y_cand_samples.squeeze(),
                                                    kind="sim outcome"))
    standardize(scatter_data_simulated)

    rows = []
    for s in range(num_samples):

        func = convert3(bar.mean, ind=s)
        results = minimize_multi_start(func, bounds,
                                       num_starts=5, num_samples=1000,
                                       jac=True, method="L-BFGS-B",
                                       random_state=random_state)
        res_best = min(filter(lambda r: r.success, results), key=lambda r: r.fun)
        row = dict(x=res_best.x.item(), y=res_best.fun, kind="sim min")
        rows.append(row)

    scatter_data_minima = pd.DataFrame(rows)
    standardize(scatter_data_minima)

    scatter_data = scatter_data_simulated.append(scatter_data_minima, sort=True,
                                                 ignore_index=True)

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    plot_predictive_distribution(x=X_grid,
                                 y_mean=foo.mean(),
                                 y_stddev=foo.stddev(), ax=ax)
    sns.scatterplot(x="x", y="y", hue="sample", palette="crest",
                    zorder=3, alpha=0.6, legend=False,
                    data=scatter_data_simulated, ax=ax)
    sns.lineplot(x="x", y="y", hue="sample", palette="crest", linewidth=0.5,
                 zorder=4, alpha=0.6, data=line_data, ax=ax)

    ax.axvline(x=x_cand_2d, c="tab:gray", linewidth=0.5, zorder=2)

    # ax.scatter(*np.broadcast_arrays(x_cand, y_cand_samples), c="tab:gray", marker=".", zorder=3)
    # plot_optimum(x=res.x, y=res.fun, color=line.get_color(), ax=ax)
    plot_labels(ax)

    ax.legend(loc="lower right")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"bar_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    fig, ax = plt.subplots()

    plot_observations(X, y, ax=ax)
    plot_latent_function(func=forrester, x=X_grid, ax=ax)
    line = plot_predictive_distribution(x=X_grid,
                                        y_mean=foo.mean(),
                                        y_stddev=foo.stddev(), ax=ax)

    ax.axvline(x=x_cand_2d, c="tab:gray", linewidth=0.5, zorder=2)

    # plot_optimum(x=res.x, y=res.fun, ymin=ymin, color=line.get_color(), ax=ax)
    ax.scatter(res.x, res.fun, color=line.get_color(), marker='*', zorder=3)

    sns.scatterplot(x="x", y="y", hue="sample", palette="crest", style="kind",
                    markers={"sim min": "*", "sim outcome": "o"},
                    zorder=3, alpha=0.6, legend=False, data=scatter_data, ax=ax)
    sns.lineplot(x="x", y="y", hue="sample", palette="crest", linewidth=0.5,
                 zorder=4, alpha=0.6, legend=False, data=line_data, ax=ax)

    ax.axis("off")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"header_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    def test(y, **kwargs):
        print(y, kwargs)
        kwargs.pop("label")
        return plt.axhline(y=y.to_numpy().item(), **kwargs)

    def loc(x, ymin, **kwargs):
        kwargs.pop("label")
        return plt.scatter(x, ymin, **kwargs)

    g = sns.FacetGrid(hue="sample", palette="crest",
                      data=scatter_data_minima,
                      height=height, aspect=aspect)

    plot_observations(X, y, ax=g.ax)
    plot_latent_function(func=forrester, x=X_grid, ax=g.ax)
    line = plot_predictive_distribution(x=X_grid,
                                        y_mean=foo.mean(),
                                        y_stddev=foo.stddev(), ax=g.ax)

    sns.scatterplot(x="x", y="y", hue="sample", palette="crest", style="kind",
                    markers={"sim min": "*", "sim outcome": "o"},
                    zorder=3, alpha=0.6, legend=False,
                    data=scatter_data, ax=g.ax)
    sns.lineplot(x="x", y="y", hue="sample", palette="crest", linewidth=0.5,
                 zorder=4, alpha=0.6, data=line_data, ax=g.ax)

    ymin, ymax = g.ax.get_ylim()

    plot_optimum(x=res.x, y=res.fun, ymin=ymin, color=line.get_color(), ax=g.ax)

    g.map(test, "y", linestyle="dashed", linewidth=0.5, zorder=3, alpha=0.6)
    g.map(loc, "x", ymin=ymin, marker="^", alpha=0.8)

    g.ax.axvline(x=x_cand_2d, c="tab:gray", linewidth=0.5, zorder=2)

    plot_labels(g.ax)
    g.ax.legend(loc="lower right")

    for ext in extension:
        g.savefig(output_path.joinpath(f"baz_{context}_{suffix}.{ext}"),
                  dpi=dpi, transparent=transparent)

    # fig, ax = plt.subplots()

    # plot_observations(X, y, ax=ax)
    # plot_latent_function(func=forrester, x=X_grid, ax=ax)
    # plot_predictive_distribution(x=X_grid,
    #                              y_mean=foo.mean(),
    #                              y_stddev=foo.stddev(),
    #                              ax=ax)

    # ax.scatter(*np.broadcast_arrays(x_cand, y_cand_samples), marker=".", zorder=6)
    # ax.axvline(x=x_cand, linewidth=1.0, linestyle="dashed", zorder=5)

    # plot_labels(ax)

    # ax.legend(loc="lower right")

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"predictive_samples_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
