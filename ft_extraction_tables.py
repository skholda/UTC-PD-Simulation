"""Word-ready table image for the multi-diameter f_T extraction."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('ft_extraction_multiD.csv')
d['flag'] = d['flag'].fillna('')

# ── Table I : per-device inputs, circuit fit and extracted frequencies ─────
hdrI = ['$D$\n($\\mu$m)', 'Bias\n(V)', 'Folder', '$R_m$\n($\\Omega$)',
        '$C_{PD}$\n(fF)', '$L_{CPW1}$\n(pH)', '$L_m$\n(pH)', '$L_{CPW2}$\n(pH)',
        'S11 fit\nRMS', '$f_{RC}$\n(GHz)', '$f_{3dB}$\n(GHz)', 'Used']
rowsI = []
for _, r in d.iterrows():
    rm = 'open' if r['open'] else f"{r.Rm:.1f}"
    f3 = '—' if not np.isfinite(r.f3) else f"{r.f3:.2f}"
    rowsI.append([f"{r.D:.0f}", f"{r.V:.0f}", r.lab, rm, f"{r.Cpd:.1f}",
                  f"{r.Lc1:.1f}", '—' if r['open'] else f"{r.Lm:.1f}",
                  f"{r.Lc2:.1f}", f"{r.rms:.3f}", f"{r.f_RC:.2f}", f3,
                  'yes' if r.use else 'no'])
    if not r.use:
        rowsI[-1][-1] = {'poor S11 fit': 'no$^{a}$',
                         'f3dB > fRC': 'no$^{b}$',
                         'no -3dB in sweep': 'no$^{c}$'}[r.flag]

# ── Table II : f_T extraction summary ─────────────────────────────────────
g = d[d.use]
def summ(sub):
    x = 1000/sub.f_RC.values**2; y = 1000/sub.f3.values**2
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ np.array([s, b])
    r2 = 1 - ((y - yh)**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    b1 = np.mean(y - x); se = np.std(y - x, ddof=1)/np.sqrt(len(x))
    return dict(n=len(x), s=s, b=b, adj=adj,
                fT=np.sqrt(1000/b) if b > 0 else np.nan, b1=b1, se=se,
                fT1=np.sqrt(1000/b1) if b1 > 0 else np.nan,
                lo=np.sqrt(1000/(b1 + se)) if b1 + se > 0 else np.nan,
                hi=np.sqrt(1000/(b1 - se)) if b1 - se > 0 else np.nan)

hdrII = ['Data set', '$N$', 'Free slope', 'Free intercept\n($\\times10^{-3}$)',
         'Adj. $R^2$', '$f_T$ free\n(GHz)',
         'Locked intercept\n($\\times10^{-3}$)', '$f_T$ locked\n(GHz)']
rowsII = []
for tag, sub in (('$-7$ V', g[g.V == -7]), ('$-5$ V', g[g.V == -5]),
                 ('both biases', g)):
    o = summ(sub)
    rowsII.append([tag, f"{o['n']}", f"{o['s']:.3f}", f"{o['b']:+.3f}",
                   f"{o['adj']:.3f}",
                   '—' if not np.isfinite(o['fT']) else f"{o['fT']:.1f}",
                   f"{o['b1']:+.3f} $\\pm$ {o['se']:.3f}",
                   f"{o['fT1']:.1f}  ({o['lo']:.1f}–{o['hi']:.1f})"])

# ── Table III : fixed model constants ─────────────────────────────────────
hdrIII = ['Quantity', 'Symbol', 'Value', 'Source']
rowsIII = [
 ['Series resistance',        '$R_s$',     '8.92 $\\Omega$',  'project baseline (locked)'],
 ['CPW pad capacitance',      '$C_{CPW}$', '46.53 fF',        'baseline; 25 $\\mu$m pad-only S11 gives 42–50 fF'],
 ['Load resistance',          '$R_L$',     '50 $\\Omega$',    'network analyser / power meter'],
 ['Junction capacitance',     '$C_{PD}$',  '100.4 / 133.5 / 227.6 fF at $-7$ V',
                                                              'measured C–V, $D$ = 25/30/40 $\\mu$m'],
 ['',                         '',          '119.9 / 161.1 / 274.6 fF at $-5$ V', ''],
 ['Shunt resistance',         '$R_m$',     'from $S_{11}$ at 10 MHz',
                                                              'measured per device and bias'],
 ['Free parameters',          '$L_{CPW1}$, $L_{CPW2}$, $L_m$', 'fitted to each $S_{11}$',
                                                              '$L_m$ fitted (no FEM value exists for the new devices)'],
]

def draw(ax, hdr, rows, title, widths=None, fs=7.6):
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=hdr, cellLoc='center',
                 loc='upper center', colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.55)
    ncol = len(hdr)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('0.6'); c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e8eef4'); c.set_text_props(weight='bold')
            c.set_height(c.get_height()*1.9)
        elif i % 2 == 0:
            c.set_facecolor('#f7f7f7')
    ax.set_title(title, fontsize=9.5, weight='bold', loc='left', pad=14)

fig = plt.figure(figsize=(13.4, 16.0))
gs = fig.add_gridspec(3, 1, height_ratios=[len(rowsI)+3.5, len(rowsII)+4.0,
                                           len(rowsIII)+3.6], hspace=0.10)
draw(fig.add_subplot(gs[0]), hdrI, rowsI,
     'TABLE I.  Measured devices, 2-L ladder $S_{11}$ fit and extracted '
     'RC-limited / measured 3-dB bandwidths')
draw(fig.add_subplot(gs[1]), hdrII, rowsII,
     'TABLE II.  $f_T$ extraction from $1/f_{3dB}^{2} = 1/f_{RC}^{2} + '
     '1/f_{T}^{2}$  (free slope vs. slope locked to unity)', fs=8.2)
draw(fig.add_subplot(gs[2]), hdrIII, rowsIII,
     'TABLE III.  Fixed circuit constants and model inputs',
     widths=[0.20, 0.13, 0.30, 0.37], fs=8.0)
fig.text(0.135, 0.502,
         '$^{a}$ $S_{11}$ residual RMS > 0.20 (degenerate inductance fit).\n'
         '$^{b}$ measured $f_{3dB}$ exceeds the modelled $f_{RC}$: the response '
         'stays flat to 29 GHz, inconsistent with a 40 $\\mu$m open device.\n'
         '$^{c}$ the sweep stops before $-3$ dB is reached.',
         fontsize=8.0, va='top', linespacing=1.6)
fig.savefig('ft_extraction_tables.png', dpi=300, bbox_inches='tight',
            facecolor='white')
print('wrote ft_extraction_tables.png')
