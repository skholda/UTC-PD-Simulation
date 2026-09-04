"""25 + 30 um only (40 um excluded): all points, with the untruncated subset
as a secondary fit for reference."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
t = t[(t.D != 40) & np.isfinite(t.x) & np.isfinite(t.y)].copy()

def fit(s):
    x, y = s.x.values, s.y.values
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [sl, b]
    r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(A.T @ A); se = np.sqrt(cov[1, 1])
    f = lambda v: np.sqrt(1000/v) if v > 0 else np.nan
    return dict(N=len(x), sl=sl, b=b, se=se, adj=adj, fT=f(b), lo=f(b + se), hi=f(b - se))

ALL = fit(t); UN = fit(t[t.ok])
for n, o in (('25+30 um, all', ALL), ('25+30 um, untruncated', UN)):
    hi = f"{o['hi']:.1f}" if np.isfinite(o['hi']) else 'inf'
    print(f"{n:<24s} N={o['N']:2d} slope={o['sl']:.3f} intercept={o['b']:+.3f}+-{o['se']:.3f} "
          f"AdjR2={o['adj']:.3f} f_T={o['fT']:.1f} ({o['lo']:.1f}-{hi})")

MK = {25: 'o', 30: 's'}; CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, ax = plt.subplots(figsize=(6.6, 5.7))
for V in (-7, -5, -3):
    for D in (25, 30):
        q = t[(t.V == V) & (t.D == D)]
        if not len(q): continue
        ok = q[q.ok]; tr = q[~q.ok]
        if len(ok):
            ax.scatter(ok.x, ok.y, s=84, marker=MK[D], facecolor='none',
                       edgecolor=CV[V], lw=1.8, zorder=5, label=f'{D} $\\mu$m, {V} V')
        if len(tr):
            ax.scatter(tr.x, tr.y, s=84, marker=MK[D], facecolor=CV[V],
                       edgecolor=CV[V], alpha=.4, lw=1.0, zorder=4)
ax.scatter([], [], s=84, marker='s', facecolor='0.5', edgecolor='0.5', alpha=.4,
           label='filled = truncated sweep')
xr = np.linspace(0, t.x.max()*1.1, 40)
ax.plot(xr, ALL['sl']*xr + ALL['b'], 'k-', lw=1.8, zorder=3,
        label=f"all {ALL['N']}:  $f_T$ = {ALL['fT']:.1f} GHz")
ax.plot(xr, UN['sl']*xr + UN['b'], '--', color='0.45', lw=1.5, zorder=3,
        label=f"untruncated {UN['N']}:  $f_T$ = {UN['fT']:.1f} GHz")

def block(o, tag):
    hi = f"{o['hi']:.1f}" if np.isfinite(o['hi']) else '$\\infty$'
    return (f'{tag}:  $N$ = {o["N"]}\n   slope = {o["sl"]:.3f},  Adj. $R^2$ = {o["adj"]:.3f}\n'
            f'   intercept = {o["b"]:.3f} $\\pm$ {o["se"]:.3f}\n'
            f'   $f_T$ = {o["fT"]:.1f} GHz  ({o["lo"]:.1f}–{hi})')
ax.annotate(block(ALL, 'all points') + '\n\n' + block(UN, 'untruncated only'),
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=8.8,
            family='monospace')
ax.set_title(r'25 + 30 $\mu$m  (40 $\mu$m excluded)', fontsize=11)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)', fontsize=11)
ax.set_xlim(0, t.x.max()*1.1); ax.set_ylim(0, t.y.max()*1.12)
ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7.4, loc='lower right')
fig.tight_layout(); fig.savefig('ft_userbw_no40.png', dpi=300)
print('wrote ft_userbw_no40.png')
