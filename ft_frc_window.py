"""Does the f_T extraction stabilise on the devices with the lowest f_RC?

Sweeps a window on f_RC and refits the free-slope line inside it.

Reading of the free slope: if the modelled f_RC is off by a uniform
multiplicative factor k, i.e. f_RC(model) = k * f_RC(true), then
    y = 1/f_3dB^2 = 1/f_RC(true)^2 + 1/f_T^2 = k^2 * x_model + 1/f_T^2,
so the free fit returns slope = k^2 while the INTERCEPT is still the correct
1/f_T^2. A slope that is stable across a subset therefore does not invalidate
that subset's intercept -- it measures the f_RC scale error. What does
invalidate the extraction is a slope (i.e. a k) that drifts between subsets,
because then no single scale factor describes the model error.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = pd.read_csv('ft_userbw.csv')
t = t[np.isfinite(t.x) & np.isfinite(t.y)].copy()

def freefit(s):
    x, y = s.x.values, s.y.values
    A = np.vstack([x, np.ones_like(x)]).T
    (sl, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [sl, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2) if len(x) > 2 else np.nan
    return dict(N=len(x), slope=sl, k=np.sqrt(sl) if sl > 0 else np.nan,
                icept=b, adj=adj,
                fT=np.sqrt(1000/b) if b > 0 else np.nan)

SETS = {'-7 V': t[t.V == -7], 'all biases': t}

for name, base in SETS.items():
    print(f'\n=== {name}:  keep only f_RC <= ceiling '
          f'(f_RC spans {base.f_RC.min():.1f}-{base.f_RC.max():.1f} GHz)')
    print(f"{'ceiling':>8s}{'N':>4s}{'slope':>8s}{'k':>7s}"
          f"{'intercept':>11s}{'Adj.R2':>8s}{'f_T':>9s}   devices")
    for ceil in [40, 35, 32, 28, 25, 22, 20]:
        s = base[base.f_RC <= ceil]
        if len(s) < 3:
            continue
        o = freefit(s)
        dev = ', '.join(f'{int(r.D)}/{r.lab}' for _, r in
                        s.sort_values('f_RC').iterrows())
        ft = f"{o['fT']:9.1f}" if np.isfinite(o['fT']) else f"{'none':>9s}"
        print(f"{ceil:8.0f}{o['N']:4d}{o['slope']:8.3f}{o['k']:7.3f}"
              f"{o['icept']:+11.3f}{o['adj']:8.3f}{ft}   {dev[:70]}")

    print(f"  --- and the complement: keep only f_RC >= floor")
    for flo in [18, 20, 22, 25, 28, 30]:
        s = base[base.f_RC >= flo]
        if len(s) < 3:
            continue
        o = freefit(s)
        ft = f"{o['fT']:9.1f}" if np.isfinite(o['fT']) else f"{'none':>9s}"
        print(f"{flo:8.0f}{o['N']:4d}{o['slope']:8.3f}{o['k']:7.3f}"
              f"{o['icept']:+11.3f}{o['adj']:8.3f}{ft}")

# ── continuous sweep for the plot ──────────────────────────────────────────
def sweep(base, ceilings):
    out = []
    for c in ceilings:
        s = base[base.f_RC <= c]
        if len(s) < 3:
            out.append((c, np.nan, np.nan, np.nan, len(s)))
            continue
        o = freefit(s)
        out.append((c, o['slope'], o['icept'], o['fT'], o['N']))
    return np.array(out, dtype=float)

ce = np.arange(19.0, 42.1, 0.5)
sw7 = sweep(t[t.V == -7], ce)
swa = sweep(t, ce)

fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.6))
CL = {'-7 V': '#c0392b', 'all biases': '#2471a3'}
for tag, sw in (('-7 V', sw7), ('all biases', swa)):
    axs[0].plot(sw[:, 0], sw[:, 1], '-o', ms=3, color=CL[tag], label=tag)
    axs[1].plot(sw[:, 0], sw[:, 2], '-o', ms=3, color=CL[tag], label=tag)
    axs[2].plot(sw[:, 0], sw[:, 3], '-o', ms=3, color=CL[tag], label=tag)
axs[0].axhline(1.0, color='0.45', ls='--', lw=1.2)
axs[0].set_ylabel('free-fit slope  ($=k^2$)')
axs[0].set_title('(a) slope vs. the $f_{RC}$ ceiling', fontsize=10)
axs[1].axhline(0.0, color='0.45', ls='--', lw=1.2)
axs[1].set_ylabel(r'intercept  $1000/f_T^2$')
axs[1].set_title('(b) intercept', fontsize=10)
axs[2].axhline(32.6, color='0.35', ls='--', lw=1.2,
               label=r'$f_{tr}$ = 32.6 GHz ($H_{ph}$ model)')
axs[2].set_ylabel(r'extracted $f_T$  (GHz)')
axs[2].set_ylim(0, 140)
axs[2].set_title('(c) extracted $f_T$', fontsize=10)
for a in axs:
    a.set_xlabel(r'keep devices with $f_{RC}\leq$ ceiling  (GHz)')
    a.grid(alpha=.3, ls=':'); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig('ft_frc_window.png', dpi=300)
print('\nwrote ft_frc_window.png')
