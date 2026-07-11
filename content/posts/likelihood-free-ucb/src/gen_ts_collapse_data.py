import numpy as np

mu = lambda x: 0.55*np.exp(-(x-0.3)**2/(2*0.12**2)) + 0.5*np.exp(-(x-0.75)**2/(2*0.12**2))
sg = lambda x: 0.03 + 0.45*np.exp(-(x-0.75)**2/(2*0.12**2))
zs = [-0.8416, -0.2533, 0.2533, 0.8416]
Q = lambda x, j: mu(x) + zs[j]*sg(x)

grid = np.linspace(0.02, 0.98, 400)
xstar = grid[np.argmax(Q(grid, 3))]

for seed in range(200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.03, 0.97, 40)
    J = rng.integers(0, 4, 40)
    Y = np.array([Q(x, j) for x, j in zip(X, J)])
    w = int(np.argmax(Y))
    # want: winner drew top level, sits near the top-curve argmax, dots not clumped
    if J[w] == 3 and abs(X[w]-xstar) < 0.025 and (J == 3).sum() >= 8:
        break

print(f"% seed={seed}  winner=({X[w]:.4f},{Y[w]:.4f})  top argmax x*={xstar:.4f}, Q*={Q(xstar,3):.4f}")
for j in range(4):
    pts = " ".join(f"({X[i]:.4f},{Y[i]:.4f})" for i in range(40) if J[i]==j and i!=w)
    print(f"% level {j+1} (alpha={0.2*(j+1):.1f})")
    print(f"\\addplot[only marks, mark=*, mark size=1.6pt, c{j+1}, opacity=0.55] coordinates {{ {pts} }};")
print(f"\\addplot[only marks, mark=star, mark size=4.5pt, very thick, winner] coordinates {{ ({X[w]:.4f},{Y[w]:.4f}) }};")
print(f"\\coordinate (xstar) at (axis cs:{xstar:.4f},{Q(xstar,3):.4f});")
