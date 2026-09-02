"""Why does the f_T plot come out with slope != 1?

The quadrature relation forces unit slope, so a free-slope fit that returns
something else is evidence that f_RC is wrong -- not something to be fixed by
locking the slope. This script:

  1. lists the per-device f_T implied by each (f_RC, f_3dB) pair,
  2. sweeps circuit variants for f_RC and reports the free-slope fit of each,
     using slope -> 1 with a positive intercept as the selection criterion.

The S11 fit itself is untouched; only which elements enter the photocurrent
path H_ckt is varied.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

exec(open('ft_extraction_multiD.py').read().split('# ── main loop')[0])

d = pd.read_csv('ft_extraction_multiD.csv')
d['flag'] = d['flag'].fillna('')
g = d[(d.use) & (d.V == -7)].copy()

fg = np.linspace(1e6, 160e9, 32001); wg = 2*np.pi*fg

def frc(r, a1, a2, am):
    Rm = np.inf if r['open'] else r['Rm']
    return f3dB_of(fg, H_ckt(wg, r['Cpd']*1e-15, Rm, r['Lc1']*1e-12*a1,
                             r['Lm']*1e-12*am, r['Lc2']*1e-12*a2))/1e9

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan)

# ── 1. per-device f_T implied by the present model ────────────────────────
print('Per-device f_T from the as-fitted 2-L ladder (-7 V):')
print(f"{'device':>14s} {'f_RC':>7s} {'f_3dB':>7s} {'1/f3^2-1/fRC^2':>15s} {'f_T':>8s}")
y = 1000/g.f3.values**2; x = 1000/g.f_RC.values**2
for (_, r), xi, yi in zip(g.iterrows(), x, y):
    dv = yi - xi
    ft = f'{np.sqrt(1000/dv):8.1f}' if dv > 0 else '     n/a'
    print(f"{str(r.D)+'um '+r.lab:>14s} {r.f_RC:7.2f} {r.f3:7.2f} {dv:15.3f} {ft}")
s, b, adj, ft = freefit(x, y)
print(f'  free-slope fit: slope={s:.3f}  intercept={b:+.3f}  Adj.R2={adj:.3f}'
      f'  -> f_T {"= %.1f GHz" % ft if np.isfinite(ft) else "not extractable"}\n')

# ── 2. which elements in the photocurrent path give unit slope? ───────────
VAR = [('full ladder (as fitted)',            1.0, 1.0, 1.0),
       ('L_CPW2 removed',                     1.0, 0.0, 1.0),
       ('L_CPW1 removed',                     0.0, 1.0, 1.0),
       ('L_m removed',                        1.0, 1.0, 0.0),
       ('only L_CPW1 (L_CPW2, L_m removed)',  1.0, 0.0, 0.0),
       ('all inductances removed',            0.0, 0.0, 0.0)]
print('Free-slope fit for each photocurrent-path variant (-7 V, N=%d):' % len(g))
print(f"{'variant':>36s} {'slope':>7s} {'intercept':>10s} {'Adj.R2':>7s} {'f_T (GHz)':>10s}")
for name, a1, a2, am in VAR:
    xv = np.array([1000/frc(r, a1, a2, am)**2 for _, r in g.iterrows()])
    s, b, adj, ft = freefit(xv, y)
    print(f'{name:>36s} {s:7.3f} {b:+10.3f} {adj:7.3f} '
          f'{ft:10.1f}' if np.isfinite(ft) else
          f'{name:>36s} {s:7.3f} {b:+10.3f} {adj:7.3f} {"none":>10s}')

# ── 3. continuous sweep: scale every fitted inductance by alpha ───────────
al = np.linspace(0.0, 1.0, 41)
sl, ic, ft = [], [], []
for a in al:
    xv = np.array([1000/frc(r, a, a, a)**2 for _, r in g.iterrows()])
    s, b, _, f = freefit(xv, y)
    sl.append(s); ic.append(b); ft.append(f)
sl, ic, ft = map(np.array, (sl, ic, ft))
i = np.argmin(np.abs(sl - 1.0))
a1 = np.interp(1.0, sl, al)
b1 = np.interp(a1, al, ic)
print(f'\nUnit slope is reached at alpha = {a1:.3f} '
      f'(fitted inductances scaled by this factor in the photocurrent path)')
print(f'  intercept there = {b1:+.4f}  ->  '
      + (f'f_T = {np.sqrt(1000/b1):.1f} GHz' if b1 > 0 else
         'still <= 0, no f_T'))

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax[0].plot(al, sl, 'o-', ms=3, color='#c0392b')
ax[0].axhline(1.0, color='0.4', ls='--', lw=1.2)
ax[0].axvline(a1, color='0.4', ls=':', lw=1.2)
ax[0].set_xlabel(r'inductance scale $\alpha$ in the photocurrent path')
ax[0].set_ylabel('free-fit slope')
ax[0].set_title(r'(a) unit slope at $\alpha$ = %.2f' % a1, fontsize=10)
ax[0].grid(alpha=.3, ls=':')

ax[1].plot(al, ic, 'o-', ms=3, color='#2471a3')
ax[1].axhline(0.0, color='0.4', ls='--', lw=1.2)
ax[1].axvline(a1, color='0.4', ls=':', lw=1.2)
ax[1].set_xlabel(r'inductance scale $\alpha$ in the photocurrent path')
ax[1].set_ylabel(r'intercept  $1000/f_T^2$')
ax[1].set_title('(b) intercept crosses zero near the same point', fontsize=10)
ax[1].grid(alpha=.3, ls=':')
fig.tight_layout()
fig.savefig('ft_slope_diagnosis.png', dpi=300)
print('\nwrote ft_slope_diagnosis.png')
