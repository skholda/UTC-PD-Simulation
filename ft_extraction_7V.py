"""f_T extraction plot, -7 V only (2-L ladder baseline, D = 25/30/40 um)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('ft_extraction_multiD.csv')
d['flag'] = d['flag'].fillna('')
g = d[(d.use) & (d.V == -7)].copy()
g['x'] = 1000/g.f_RC**2
g['y'] = 1000/g.f3**2

x, y = g.x.values, g.y.values
A = np.vstack([x, np.ones_like(x)]).T
(slope, icept), *_ = np.linalg.lstsq(A, y, rcond=None)
r2 = 1 - ((y - A @ [slope, icept])**2).sum()/((y - y.mean())**2).sum()
adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
b1 = np.mean(y - x); se = np.std(y - x, ddof=1)/np.sqrt(len(x))
fT1 = np.sqrt(1000/b1); lo = np.sqrt(1000/(b1 + se)); hi = np.sqrt(1000/(b1 - se))

print(f'n={len(x)}  free slope={slope:.3f}  intercept={icept:+.3f}  '
      f'Adj.R2={adj:.3f}')
print(f'slope-locked intercept={b1:+.3f}+-{se:.3f}  '
      f'f_T={fT1:.1f} GHz ({lo:.1f}-{hi:.1f})')

MK = {25: 'o', 30: 's', 40: '^'}
CL = {25: '#1f77b4', 30: '#c0392b', 40: '#2e8b57'}

fig, ax = plt.subplots(figsize=(6.2, 5.4))
for D in (25, 30, 40):
    s = g[g.D == D]
    if not len(s):
        continue
    ax.scatter(s.x, s.y, s=70, marker=MK[D], facecolor='none',
               edgecolor=CL[D], linewidth=1.7, zorder=4,
               label=f'{D} $\\mu$m  ($N$ = {len(s)})')

ax.set_xlim(0, x.max()*1.18); ax.set_ylim(0, y.max()*1.22)

xr = np.linspace(0, x.max()*1.18, 50)
ax.plot(xr, slope*xr + icept, '-',  color='k',    lw=1.5, zorder=3,
        label=f'free slope = {slope:.2f}')
ax.plot(xr, xr + b1,          '--', color='0.45', lw=1.5, zorder=3,
        label='slope locked to 1')

ax.annotate(f'$-7$ V,  $N$ = {len(x)}\n'
            f'free-slope fit:  Adj. $R^2$ = {adj:.3f}\n'
            f'locked-slope intercept = {b1:.3f} $\\pm$ {se:.3f}\n'
            f'$f_T$ = {fT1:.1f} GHz  ({lo:.1f}–{hi:.1f})',
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', ha='left',
            fontsize=9)
ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')

ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=8, loc='lower right', frameon=True)
fig.tight_layout()

# device labels: greedy placement in display space against the real text boxes
fig.canvas.draw()
rend = fig.canvas.get_renderer()
MARK_R = 7.0                                   # marker half-size, points
taken = [ax.transData.transform((r.x, r.y)) for _, r in g.iterrows()]
boxes = [(px - MARK_R, py - MARK_R, px + MARK_R, py + MARK_R)
         for px, py in taken]
CAND = [(10, -3), (10, 5), (-10, -3), (-10, 5), (0, 10), (0, -14),
        (16, -12), (-16, -12), (16, 10), (-16, 10)]
for _, r in g.sort_values('x').iterrows():
    best = None
    for dx, dy in CAND:
        t = ax.annotate(r.lab, (r.x, r.y), textcoords='offset points',
                        xytext=(dx, dy), fontsize=7.0, color=CL[r.D],
                        ha='left' if dx > 0 else 'right' if dx < 0 else 'center',
                        va='bottom' if dy > 0 else 'top')
        bb = t.get_window_extent(renderer=rend)
        b = (bb.x0 - 1.5, bb.y0 - 1.5, bb.x1 + 1.5, bb.y1 + 1.5)
        if all(b[0] > o[2] or b[2] < o[0] or b[1] > o[3] or b[3] < o[1]
               for o in boxes):
            best = b
            break
        t.remove()
    if best is None:                            # nothing clear: keep the first
        dx, dy = CAND[0]
        t = ax.annotate(r.lab, (r.x, r.y), textcoords='offset points',
                        xytext=(dx, dy), fontsize=7.0, color=CL[r.D],
                        ha='left', va='top')
        bb = t.get_window_extent(renderer=rend)
        best = (bb.x0, bb.y0, bb.x1, bb.y1)
    boxes.append(best)

fig.savefig('ft_extraction_7V.png', dpi=300)
print('wrote ft_extraction_7V.png')
