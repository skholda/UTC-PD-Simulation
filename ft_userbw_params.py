"""RC-circuit parameters behind the f_T extraction (ft_userbw.py), as a
Word-ready table image, plus a self-consistency check: f_RC is recomputed
from the tabulated (rounded) values and compared with the value the fit used.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

exec(open('ft_extraction_multiD.py').read().split('# ── main loop')[0])

t = pd.read_csv('ft_userbw.csv')
t['note'] = t['note'].fillna('')
t = t.sort_values(['V', 'D', 'Rm'], ascending=[False, True, True])

# ── self-consistency: rebuild f_RC from the rounded table values ──────────
fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
chk = []
for _, r in t.iterrows():
    Rm = np.inf if r['open'] else round(r.Rm, 1)
    f = f3dB_of(fg, H_ckt(wg, round(r.Cpd, 1)*1e-15, Rm,
                          round(r.L1, 1)*1e-12, round(r.Lm, 1)*1e-12,
                          round(r.L2, 1)*1e-12))/1e9
    chk.append(f)
t['f_RC_chk'] = chk
print('max |f_RC(table) - f_RC(rebuilt from rounded values)| = '
      f'{(t.f_RC - t.f_RC_chk).abs().max():.3f} GHz')

# ── Table I : topology and fixed constants ────────────────────────────────
hdrI = ['Element', 'Symbol', 'Value', 'Status', 'Source']
rowsI = [
 ['Photocurrent source',      '$I_{ph}$',  '1 mA (all sheets)',       'fixed',
  'file names; the level cancels in the normalised response'],
 ['Junction capacitance',     '$C_{PD}$',  '25 $\\mu$m: 100.4 / 119.9 / 166.7 fF',
  'fixed', 'measured C–V at $-7$ / $-5$ / $-3$ V'],
 ['',                         '',          '30 $\\mu$m: 133.5 / 161.1 / 230.8 fF', '', ''],
 ['',                         '',          '40 $\\mu$m: 227.6 / 274.6 / 390.4 fF', '', ''],
 ['Series resistance',        '$R_s$',     '8.92 $\\Omega$',           'fixed (locked)',
  'project baseline; same value for every diameter and bias'],
 ['CPW pad capacitance',      '$C_{CPW}$', '46.53 fF',                'fixed (locked)',
  'baseline; 25 $\\mu$m pad-only S11 reads 42–50 fF'],
 ['Shunt (matching) resistor','$R_m$',     'per device, see Table II', 'fixed',
  'Re($Z_{in}$) of that device\'s own S11 at 10 MHz'],
 ['Resistor inductance',      '$L_m$',     'per device',              'free',
  'fitted to S11 (no FEM value for the new devices)'],
 ['CPW inductance, PD side',  '$L_{CPW1}$','per device',              'free', 'fitted to S11'],
 ['CPW inductance, port side','$L_{CPW2}$','per device',              'free', 'fitted to S11'],
 ['Load',                     '$R_L$',     '50 $\\Omega$',             'fixed', 'VNA / power meter'],
]

# ── Table II : per-device values ──────────────────────────────────────────
hdrII = ['$D$\n($\\mu$m)', 'Bias\n(V)', 'Device', '$R_m$\n($\\Omega$)',
         '$C_{PD}$\n(fF)', '$L_{CPW1}$\n(pH)', '$L_m$\n(pH)', '$L_{CPW2}$\n(pH)',
         'S11 fit\nRMS', '$f_{RC}$\n(GHz)', '$f_{3dB}$\n(GHz)',
         'Sweep\nend (GHz)', 'In fit']
rowsII = []
for _, r in t.iterrows():
    rm = 'open' if r['open'] else f'{r.Rm:.1f}'
    lm = '—' if r['open'] else f'{r.Lm:.1f}'
    f3 = f'{r.f3:.2f}' if np.isfinite(r.f3) else '—'
    used = 'yes' if r.ok else ('no$^{a}$' if r.frac > 0.85 else 'no')
    rowsII.append([f'{r.D:.0f}', f'{r.V:.0f}', r.lab if r.lab == 'WO' else f'{r.lab} $\\Omega$',
                   rm, f'{r.Cpd:.1f}', f'{r.L1:.1f}', lm, f'{r.L2:.1f}',
                   f'{r.rms:.3f}', f'{r.f_RC:.2f}', f3, f'{r.fmax:.1f}', used])

def draw(ax, hdr, rows, title, widths=None, fs=7.6, scale=1.5):
    ax.axis('off')
    global _last_tb
    _last_tb = tb = ax.table(cellText=rows, colLabels=hdr, cellLoc='center',
                  loc='upper center', colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.scale(1, scale)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor('0.6'); c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e8eef4'); c.set_text_props(weight='bold')
            c.set_height(c.get_height()*1.9)
        elif i % 2 == 0:
            c.set_facecolor('#f7f7f7')
    ax.set_title(title, fontsize=9.5, weight='bold', loc='left', pad=14)

fig = plt.figure(figsize=(14.0, 13.6))
gs = fig.add_gridspec(2, 1, height_ratios=[len(rowsI)+3.2, len(rowsII)+4.2],
                      hspace=0.08)
draw(fig.add_subplot(gs[0]), hdrI, rowsI,
     'TABLE I.  2-L ladder used for $f_{RC}$:  '
     '$I_{ph}\\,\\|\\,C_{PD}$ — $R_s$ — [$C_{CPW}$] — $L_{CPW1}$ — '
     '[$R_m$ + $L_m$] — $L_{CPW2}$ — $R_L$   (brackets: shunt to ground)',
     widths=[0.19, 0.08, 0.27, 0.12, 0.34], fs=7.8, scale=1.45)
draw(fig.add_subplot(gs[1]), hdrII, rowsII,
     'TABLE II.  Per-device values.  $f_{RC}$: $-3$ dB of $|V_{RL}/I_{ph}|$ '
     'from the fitted ladder.  $f_{3dB}$: measured, 3rd-order polynomial.',
     fs=7.4, scale=1.42)
fig.canvas.draw()
_bb = _last_tb.get_window_extent(fig.canvas.get_renderer()).transformed(
    fig.transFigure.inverted())
fig.text(0.075, _bb.y0 - 0.012,
         '$^{a}$ dropped from the $f_T$ fit: measured $f_{3dB}$ lies above 85% '
         'of that sheet\'s sweep span, where the polynomial estimate is '
         'biased high (30 $\\mu$m open device: 17.25 GHz from a 19.5 GHz '
         'sweep vs. 14.58 GHz from a 30 GHz sweep).\n'
         'Fit weights: unweighted complex-S11 least squares over the full '
         'sweep (0.01–40/50 GHz), parameters in pH with bounds 0–600 pH.  '
         f'Rebuilding $f_{{RC}}$ from the rounded values above reproduces the '
         f'fit to within {(t.f_RC - t.f_RC_chk).abs().max():.3f} GHz.',
         fontsize=7.8, va='top', linespacing=1.6)
fig.savefig('ft_userbw_params.png', dpi=300, bbox_inches='tight',
            facecolor='white')

# console copy
pd.set_option('display.width', 220)
print(t[['D', 'V', 'lab', 'Rm', 'Cpd', 'L1', 'Lm', 'L2', 'rms', 'f_RC',
         'f3', 'fmax', 'ok']].to_string(index=False,
      float_format=lambda v: f'{v:.2f}'))
print('\nwrote ft_userbw_params.png')
