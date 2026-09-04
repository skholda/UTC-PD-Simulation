"""25 + 30 um, untruncated sweeps only: the 9 points and their linear fit."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
g = t[(t.D != 40) & t.ok].copy().sort_values('x')
x, y = g.x.values, g.y.values
A = np.vstack([x, np.ones_like(x)]).T
(sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
res = y - A @ [sl, b]
r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(A.T @ A); se = np.sqrt(cov[1, 1])
f = lambda v: np.sqrt(1000/v) if v > 0 else np.nan
fT, lo, hi = f(b), f(b + se), f(b - se)
print(f'N={len(x)} slope={sl:.3f} intercept={b:+.3f}+-{se:.3f} AdjR2={adj:.3f} '
      f'f_T={fT:.1f} ({lo:.1f}-{hi:.1f})')

MK = {25: 'o', 30: 's'}; CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, ax = plt.subplots(figsize=(6.2, 5.4))
for V in (-7, -5, -3):
    for D in (25, 30):
        q = g[(g.V == V) & (g.D == D)]
        if len(q):
            ax.scatter(q.x, q.y, s=88, marker=MK[D], facecolor='none',
                       edgecolor=CV[V], lw=1.9, zorder=5,
                       label=f'{D} $\\mu$m, {V} V  ($N$ = {len(q)})')
for _, r in g.iterrows():
    ax.annotate(r.lab, (r.x, r.y), textcoords='offset points', xytext=(8, 4),
                fontsize=8, color=CV[r.V])
xr = np.linspace(0, x.max()*1.1, 40)
ax.plot(xr, sl*xr + b, 'k-', lw=1.8, zorder=4, label='linear fit')
hs = f'{hi:.1f}' if np.isfinite(hi) else '$\\infty$'
ax.annotate(f'25 + 30 $\\mu$m, untruncated,  $N$ = {len(x)}\n'
            f'slope = {sl:.3f}\n'
            f'intercept = {b:.3f} $\\pm$ {se:.3f}  ($\\times10^{{-3}}$)\n'
            f'Adj. $R^2$ = {adj:.3f}\n'
            f'$f_T$ = {fT:.1f} GHz  ({lo:.1f}–{hs})',
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=10)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_xlim(0, x.max()*1.1); ax.set_ylim(0, y.max()*1.12)
ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig('ft_userbw_no40_clean.png', dpi=300)
print('wrote ft_userbw_no40_clean.png')
