"""Regenerate heat_data.tex: exact Gaussian-smoothed step curves for heat-smoothing.tex."""
import numpy as np
from math import erf, sqrt

mus = np.linspace(-4, 4, 201)
with open("heat_data.tex", "w") as f:
    for name, sig in (("heatA", 0.04), ("heatB", 0.35), ("heatC", 0.9), ("heatD", 1.8)):
        pts = " ".join(
            f"({m:.3f},{0.5 * (1 + erf(m / (sig * sqrt(2)))):.5f})" for m in mus
        )
        f.write("\\def\\" + name + "{" + pts + "}\n")
