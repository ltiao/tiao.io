import numpy as np

from scipy.special import expit


def first_moment(x):
    return .5 * np.tanh(.5*x) / x


def softplus(x):
    return np.log(1. + np.exp(x))


def softplus_upper_bound(x, xi):
    return .5 * (x + first_moment(xi) * (x**2 - xi**2)) + \
        np.log(np.exp(.5*xi) + np.exp(-.5*xi))


def sigmoid_lower_bound(x, xi):
    return expit(xi) * np.exp(.5*(x - xi - first_moment(xi) * (x**2 - xi**2)))
