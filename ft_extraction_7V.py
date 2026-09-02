"""f_T extraction plot, -7 V (2-L ladder baseline, D = 25/30/40 um).

The quadrature relation 1/f_3dB^2 = 1/f_RC^2 + 1/f_T^2 has unit slope by
construction, so the slope is a test of the f_RC model, not a fit parameter --
and locking it to 1 would reduce the "extraction" to mean(y - x). Only the
free-slope fit is shown; the unit-slope line is drawn as a reference, not a fit.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('ft_extraction_multiD.csv')
d['flag'] = d['flag'].fillna('')
g = d[(d.use) & (d.V == -7)].copy()
g['x'] = 1000/g.f_RC**2
g['y'] = 1000/g.f3**2
g['dv'] = g.y - g.x                      # implied 1000/f_T^2, per device
g['fT'] = np.where(g.dv > 0, np.sqrt(1000/g.dv.where(g.dv > 0)), np.nan)

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan)

SUB = [('all 12',              g,                       'k',       'o'),
       ('30 $\\mu$m campaign A', g[g.lab.isin(['200', '33/36', '55', 'WO(A)'])],
                                                        '#c0392b', 's'),
       ('30 $\\mu$m campaign B', g[g.lab.isin(['120', '32/40', '71/76', 'WO(B)'])],
                                                        '#e67e22', 'D'),
       ('40 $\\mu$m',           g[g.D == 40],            '#2e8b57', '^')]
print(f"{'subset':<22s}{'N':>3s}{'slope':>8s}{'intercept':>11s}{'Adj.R2':>8s}{'f_T':>10s}")
for name, s, _, _ in SUB:
    if len(s) < 3:
        continue
    sl, b, adj, ft = freefit(s.x.values, s.y.values)
    print(f"{name:<22s}{len(s):3d}{sl:8.3f}{b:+11.3f}{adj:8.3f}"
          + (f"{ft:10.1f}" if np.isfinite(ft) else f"{'none':>10s}"))

MK = {25: 'o', 30: 's', 40: '^'}
CL = {25: '#1f77b4', 30: '#c0392b', 40: '#2e8b57'}
sl, b, adj, ft = freefit(g.x.values, g.y.values)

fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.1))

# ── (a) the extraction plot ───────────────────────────────────────────────
ax = axs[0]
for D in (25, 30, 40):
    s = g[g.D == D]
    ax.scatter(s.x, s.y, s=70, marker=MK[D], facecolor='none',
               edgecolor=CL[D], linewidth=1.7, zorder=4,
               label=f'{D} $\\mu$m  ($N$ = {len(s)})')
xr = np.linspace(0, g.x.max()*1.18, 50)
ax.plot(xr, sl*xr + b, 'k-', lw=1.6, zorder=3,
        label=f'free fit: slope {sl:.2f}')
ax.plot(xr, xr, ':', color='0.5', lw=1.4, zorder=2,
        label='unit slope (required)')
ax.annotate(f'$-7$ V,  $N$ = {len(g)}\n'
            f'slope = {sl:.3f}   (must be 1)\n'
            f'intercept = {b:+.3f}   (must be $>0$)\n'
            f'Adj. $R^2$ = {adj:.3f}\n'
            r'$\Rightarrow$ $f_T$ not extractable',
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=9)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')
ax.set_xlim(0, g.x.max()*1.18); ax.set_ylim(0, g.y.max()*1.20)
ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=8, loc='lower right')
ax.set_title('(a) 2-L ladder baseline', fontsize=10)

# ── (b) per-device implied f_T ────────────────────────────────────────────
ax = axs[1]
o = g.sort_values(['D', 'f_RC'])
pos = np.arange(len(o))
for k, (_, r) in enumerate(o.iterrows()):
    ax.scatter(k, r.fT, s=80, marker=MK[r.D], facecolor='none',
               edgecolor=CL[r.D], linewidth=1.8, zorder=4)
ax.axhline(32.6, color='0.35', ls='--', lw=1.3,
           label=r'$f_{tr}$ = 32.6 GHz from the $H_{ph}$ model')
ax.set_xticks(pos)
ax.set_xticklabels([f'{r.D:.0f} $\\mu$m {r.lab}' for _, r in o.iterrows()],
                   rotation=55, ha='right', fontsize=7.5)
ax.set_ylabel(r'implied $f_T = (1/f_{3dB}^2 - 1/f_{RC}^2)^{-1/2}$   (GHz)')
ax.set_ylim(0, 125)
ax.grid(alpha=.3, ls=':', axis='y')
ax.legend(fontsize=8, loc='upper left')
ax.set_title('(b) $f_T$ implied device by device — should be one number',
             fontsize=10)

fig.tight_layout()
fig.savefig('ft_extraction_7V.png', dpi=300)
print('\nwrote ft_extraction_7V.png')
