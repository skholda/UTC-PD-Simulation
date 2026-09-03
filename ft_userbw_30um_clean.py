"""30 um, untruncated sweeps only: the 7 points and their linear fit."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
g = t[(t.D == 30) & t.ok].copy().sort_values('x')

x, y = g.x.values, g.y.values
A = np.vstack([x, np.ones_like(x)]).T
(sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
res = y - A @ [sl, b]
r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(A.T @ A)
b_se = np.sqrt(cov[1, 1])
fT, lo, hi = (np.sqrt(1000/v) for v in (b, b + b_se, b - b_se))
print(f'N={len(x)} slope={sl:.3f} intercept={b:+.3f}+-{b_se:.3f} '
      f'AdjR2={adj:.3f} f_T={fT:.1f} GHz ({lo:.1f}-{hi:.1f})')
print(g[['V', 'lab', 'f_RC', 'f3', 'fmax', 'x', 'y']].to_string(
      index=False, float_format=lambda v: f'{v:.2f}'))

CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, ax = plt.subplots(figsize=(6.2, 5.4))
for V in (-7, -5, -3):
    q = g[g.V == V]
    ax.scatter(q.x, q.y, s=90, marker='s', facecolor='none', edgecolor=CV[V],
               lw=1.9, zorder=5, label=f'{V} V  ($N$ = {len(q)})')
for _, r in g.iterrows():
    ax.annotate(r.lab, (r.x, r.y), textcoords='offset points', xytext=(8, 4),
                fontsize=8, color=CV[r.V])
xr = np.linspace(0, x.max()*1.1, 40)
ax.plot(xr, sl*xr + b, 'k-', lw=1.8, zorder=4, label='linear fit')
ax.annotate(f'30 $\\mu$m,  $N$ = {len(x)}\n'
            f'slope = {sl:.3f}\n'
            f'intercept = {b:.3f} $\\pm$ {b_se:.3f}  ($\\times10^{{-3}}$)\n'
            f'Adj. $R^2$ = {adj:.3f}\n'
            f'$f_T$ = {fT:.1f} GHz  ({lo:.1f}–{hi:.1f})',
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=10)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_xlim(0, x.max()*1.1); ax.set_ylim(0, y.max()*1.12)
ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=8.5, loc='lower right')
fig.tight_layout()
fig.savefig('ft_userbw_30um_clean.png', dpi=300)
print('wrote ft_userbw_30um_clean.png')
