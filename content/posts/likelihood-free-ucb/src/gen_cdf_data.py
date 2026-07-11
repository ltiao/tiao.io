"""Regenerate cdf_data.tex: exact Gaussian CDF coordinates for cdf-readouts.tex."""
import numpy as np
from math import erf, sqrt

ts = np.linspace(-2.2, 4.6, 171)
F = [0.5 * (1 + erf((t - 1) / sqrt(2))) for t in ts]
pts = " ".join(f"({t:.4f},{v:.5f})" for t, v in zip(ts, F))
sh = " ".join(
    f"({t:.4f},{0.5 * (1 + erf((t - 1) / sqrt(2))):.5f})" for t in ts if t >= 0
)
with open("cdf_data.tex", "w") as f:
    f.write("\\def\\cdfcurve{" + pts + "}\n")
    f.write("\\def\\cdfshade{" + sh + "}\n")
