import numpy as np

# Left panel: GP posterior on sparse data with a gap -> epistemic band balloons
rng = np.random.default_rng(4)
f = lambda x: 0.7*np.sin(2*np.pi*x)
Xtr = np.array([0.04, 0.11, 0.19, 0.27, 0.33, 0.78, 0.86, 0.94])
sn, ell, sf = 0.05, 0.16, 0.7
ytr = f(Xtr) + sn*rng.standard_normal(len(Xtr))
k = lambda a, b: sf**2*np.exp(-0.5*(a[:,None]-b[None,:])**2/ell**2)
L = np.linalg.cholesky(k(Xtr,Xtr) + sn**2*np.eye(len(Xtr)))
al = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
xs = np.linspace(0, 1, 141)
Ks = k(xs, Xtr)
mu = Ks @ al
v = np.linalg.solve(L, Ks.T)
s = np.sqrt(np.maximum(sf**2 - (v**2).sum(0), 1e-12))
def coords(x, y): return " ".join(f"({a:.4f},{b:.4f})" for a, b in zip(x, y))
print("% LEFT: gp mean / upper / lower / data")
print(f"\\def\\gpmean{{{coords(xs, mu)}}}")
print(f"\\def\\gpupper{{{coords(xs, mu+2*s)}}}")
print(f"\\def\\gplower{{{coords(xs, mu-2*s)}}}")
print(f"\\def\\gpdata{{{coords(Xtr, ytr)}}}")

# Right panel: heteroscedastic noise, same x-gap; aleatoric band = f +- 2 sigma(x)
sig = lambda x: 0.06 + 0.24*x
xl = rng.uniform(0.0, 0.36, 55)
xr = rng.uniform(0.76, 1.0, 35)
Xn = np.concatenate([xl, xr])
Yn = f(Xn) + sig(Xn)*rng.standard_normal(len(Xn))
print("% RIGHT: true fn / upper / lower / data")
print(f"\\def\\almean{{{coords(xs, f(xs))}}}")
print(f"\\def\\alupper{{{coords(xs, f(xs)+2*sig(xs))}}}")
print(f"\\def\\allower{{{coords(xs, f(xs)-2*sig(xs))}}}")
print(f"\\def\\aldata{{{coords(Xn, Yn)}}}")
