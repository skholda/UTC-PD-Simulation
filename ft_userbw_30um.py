"""f_T fit restricted to the 30 um devices: untruncated (7) and all (15)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
t = t[(t.D == 30) & np.isfinite(t.x) & np.isfinite(t.y)].copy()

def fit(s):
    x, y = s.x.values, s.y.values
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [sl, b]
    r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(A.T @ A)
    b_se = np.sqrt(cov[1, 1])
    f = lambda v: np.sqrt(1000/v) if v > 0 else np.nan
    return dict(N=len(x), sl=sl, b=b, b_se=b_se, adj=adj,
                fT=f(b), lo=f(b + b_se), hi=f(b - b_se))

U = fit(t[t.ok]); A = fit(t)
for n, o in (('30 um untruncated', U), ('30 um all', A)):
    print(f"{n:<20s} N={o['N']:2d} slope={o['sl']:.3f} "
          f"intercept={o['b']:+.3f}+-{o['b_se']:.3f} AdjR2={o['adj']:.3f} "
          f"f_T={o['fT']:.1f} ({o['lo']:.1f}-{o['hi']:.1f})")

CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, ax = plt.subplots(figsize=(6.6, 5.7))
for V in (-7, -5, -3):
    q = t[t.V == V]; ok = q[q.ok]; tr = q[~q.ok]
    ax.scatter(ok.x, ok.y, s=84, marker='s', facecolor='none', edgecolor=CV[V],
               lw=1.8, zorder=5, label=f'{V} V  (fitted, $N$ = {len(ok)})')
    if len(tr):
        ax.scatter(tr.x, tr.y, s=84, marker='s', facecolor=CV[V],
                   edgecolor=CV[V], alpha=.4, lw=1.0, zorder=4,
                   label=f'{V} V  (truncated, $N$ = {len(tr)})')
for _, r in t.iterrows():
    ax.annotate(r.lab, (r.x, r.y), textcoords='offset points', xytext=(7, 4),
                fontsize=7, color=CV[r.V], alpha=.9)
xr = np.linspace(0, t.x.max()*1.1, 40)
ax.plot(xr, U['sl']*xr + U['b'], '-', color='k', lw=1.7, zorder=3,
        label=f"fit, untruncated:  $f_T$ = {U['fT']:.1f} GHz")
ax.plot(xr, A['sl']*xr + A['b'], '--', color='0.45', lw=1.5, zorder=3,
        label=f"fit, all 15:  $f_T$ = {A['fT']:.1f} GHz")
ax.plot(xr, xr, ':', color='0.65', lw=1.1, zorder=2, label='unit slope')

def block(o, tag):
    return (f'{tag}:  $N$ = {o["N"]}\n'
            f'   slope = {o["sl"]:.3f},  Adj. $R^2$ = {o["adj"]:.3f}\n'
            f'   intercept = {o["b"]:.3f} $\\pm$ {o["b_se"]:.3f}\n'
            f'   $f_T$ = {o["fT"]:.1f} GHz  ({o["lo"]:.1f}–'
            + (f'{o["hi"]:.1f})' if np.isfinite(o["hi"]) else '$\\infty$)'))
ax.annotate(block(U, 'untruncated') + '\n\n' + block(A, 'all 30 $\\mu$m'),
            xy=(0.035, 0.965), xycoords='axes fraction', va='top',
            fontsize=8.8, family='monospace')
ax.set_title(r'30 $\mu$m only', fontsize=11)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_xlim(0, t.x.max()*1.1); ax.set_ylim(0, t.y.max()*1.12)
ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=7.4, loc='lower right')
fig.tight_layout()
fig.savefig('ft_userbw_30um.png', dpi=300)
print('wrote ft_userbw_30um.png')
