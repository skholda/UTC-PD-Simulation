"""f_T fit with every point included (N=24) vs. untruncated only (N=14)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
t = t[np.isfinite(t.x) & np.isfinite(t.y)].copy()

def fit(s):
    x, y = s.x.values, s.y.values
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [sl, b]
    r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(A.T @ A)
    b_se = np.sqrt(cov[1, 1])
    fT = np.sqrt(1000/b) if b > 0 else np.nan
    lo = np.sqrt(1000/(b + b_se)) if b + b_se > 0 else np.nan
    hi = np.sqrt(1000/(b - b_se)) if b - b_se > 0 else np.nan
    return dict(N=len(x), sl=sl, b=b, b_se=b_se, adj=adj, fT=fT, lo=lo, hi=hi)

A24 = fit(t); A14 = fit(t[t.ok])
for n, o in (('all 24', A24), ('untruncated 14', A14)):
    print(f"{n:<16s} slope={o['sl']:.3f} intercept={o['b']:+.3f}+-{o['b_se']:.3f} "
          f"AdjR2={o['adj']:.3f} f_T={o['fT']:.1f} ({o['lo']:.1f}-{o['hi']:.1f})")

MK = {25: 'o', 30: 's', 40: '^'}
CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, ax = plt.subplots(figsize=(6.6, 5.7))
for V in (-7, -5, -3):
    for D in (25, 30, 40):
        q = t[(t.V == V) & (t.D == D)]
        if not len(q):
            continue
        ok = q[q.ok]; tr = q[~q.ok]
        if len(ok):
            ax.scatter(ok.x, ok.y, s=78, marker=MK[D], facecolor='none',
                       edgecolor=CV[V], lw=1.8, zorder=5,
                       label=f'{D} $\\mu$m, {V} V')
        if len(tr):
            ax.scatter(tr.x, tr.y, s=78, marker=MK[D], facecolor=CV[V],
                       edgecolor=CV[V], lw=1.0, alpha=.45, zorder=4)
ax.scatter([], [], s=78, marker='s', facecolor='0.5', edgecolor='0.5',
           alpha=.45, label='filled = truncated sweep')
xr = np.linspace(0, t.x.max()*1.1, 40)
ax.plot(xr, A24['sl']*xr + A24['b'], '-', color='k', lw=1.7, zorder=3,
        label=f"all 24:  $f_T$ = {A24['fT']:.1f} GHz")
ax.plot(xr, A14['sl']*xr + A14['b'], '--', color='0.4', lw=1.6, zorder=3,
        label=f"14 untruncated:  $f_T$ = {A14['fT']:.1f} GHz")

def block(o, tag):
    return (f'{tag}:  $N$ = {o["N"]}\n'
            f'   slope = {o["sl"]:.3f},  Adj. $R^2$ = {o["adj"]:.3f}\n'
            f'   intercept = {o["b"]:.3f} $\\pm$ {o["b_se"]:.3f}\n'
            f'   $f_T$ = {o["fT"]:.1f} GHz  ({o["lo"]:.1f}–{o["hi"]:.1f})')
ax.annotate(block(A24, 'all points') + '\n\n' + block(A14, 'untruncated only'),
            xy=(0.035, 0.965), xycoords='axes fraction', va='top',
            fontsize=8.8, family='monospace')
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_xlim(0, t.x.max()*1.1); ax.set_ylim(0, t.y.max()*1.12)
ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=7.4, loc='lower right', ncol=2)
fig.tight_layout()
fig.savefig('ft_userbw_all24.png', dpi=300)
print('wrote ft_userbw_all24.png')
