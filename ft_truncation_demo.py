"""What "truncated sweep" means, shown on the data.

(a) the same 30 um open device at -5 V measured with a 19.5 GHz sweep and a
    30 GHz sweep: the cubic fitted to the short sweep puts f_3dB 18% higher.
(b)-(i) the eight 25/30 um sheets dropped from the f_T fit: in each, the
    measured -3 dB crossing sits within the last 3-13% of the sweep, so the
    cubic has no data beyond the roll-off to constrain its curvature.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)          # PAIR, sheet_path, FRAC_MAX

def load(sheet):
    df = pd.read_excel(sheet_path(sheet), header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[k], p[k]

def poly_f3(f, p):
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    f3 = np.nan
    if len(i) and i[0] > 0:
        j = i[0]; f3 = float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]]))
    return c, ref, f3

t = pd.read_csv('ft_userbw.csv')
drop = t[(t.D != 40) & (~t.ok) & np.isfinite(t.frac)].sort_values('frac',
                                                                    ascending=False)
lookup = {(D, V, lab, camp): sheet for D, V, lab, camp, s11, sheet, _ in PAIR}

CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.2))
axs = axs.ravel()

# ── (a) the demonstration pair ────────────────────────────────────────────
ax = axs[0]
A = 'data_PD0008_1/Bandwidth/30um/Figure_03_27_2026/WO'
for sheet, ls, lab in (('data_bw_user/Bias_5V_Iph_1mA_30um_WO_1.xlsx', '--',
                        '19.5 GHz sweep'),
                       (f'{A}/Bias_-5V_Iph_1mA_30GHz.xlsx', '-', '30 GHz sweep')):
    f, p = load(sheet); c, ref, f3 = poly_f3(f, p)
    col = '#e67e22' if ls == '--' else '#2471a3'
    ax.plot(f, p - ref, 'o', ms=3.3, mfc='none', color=col, alpha=.7)
    ff = np.linspace(0, f[-1], 600)
    ax.plot(ff, np.polyval(c, ff) - ref, ls, color=col, lw=2.0,
            label=f'{lab}:  $f_{{3dB}}$ = {f3:.2f} GHz')
    ax.axvline(f3, color=col, ls=':', lw=1.2)
    ax.axvline(f[-1], color=col, lw=0.8, alpha=.5)
ax.axhline(-3, color='0.5', ls='--', lw=1.0)
ax.set_ylim(-9, 1.5); ax.set_xlim(0, 31)
ax.set_title('(a)  same device, same bias: 30 $\\mu$m open, $-5$ V', fontsize=9.5)
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalised response (dB)')
ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=.3, ls=':')
ax.annotate('short sweep ends 2.3 GHz\nafter its own $-3$ dB point\n'
            '$\\rightarrow$ cubic reads 18% high',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=8.2, color='#e67e22')

# ── (b)-(i) the dropped sheets ────────────────────────────────────────────
for ax, (_, r), tag in zip(axs[1:], drop.iterrows(), 'bcdefghi'):
    sheet = lookup[(r.D, r.V, r.lab, r.camp)]
    f, p = load(sheet); c, ref, f3 = poly_f3(f, p)
    col = CV[r.V]
    ax.plot(f, p - ref, 'o', ms=3.3, mfc='none', color=col, alpha=.75,
            label='measured')
    ff = np.linspace(0, f[-1], 600)
    ax.plot(ff, np.polyval(c, ff) - ref, '-', color='k', lw=1.7,
            label='3rd-order fit')
    ax.axhline(-3, color='0.5', ls='--', lw=1.0)
    ax.axvline(f3, color=col, ls=':', lw=1.4)
    ax.axvspan(f3, f[-1], color='0.85', alpha=.6, zorder=0)
    ax.axvline(f[-1], color='0.35', lw=1.0)
    ax.set_xlim(0, 40); ax.set_ylim(-9, 1.5)
    dev = 'open' if r['open'] else f'{r.Rm:.0f} $\\Omega$'
    ax.set_title(f'({tag})  30 $\\mu$m, {dev}, {r.V:.0f} V, {r.camp}',
                 fontsize=9.5)
    ax.annotate(f'$f_{{3dB}}$ = {f3:.1f} GHz\nsweep ends {f[-1]:.1f} GHz\n'
                f'$f_{{3dB}}$ / end = {r.frac:.0%}',
                xy=(0.04, 0.06), xycoords='axes fraction', va='bottom',
                fontsize=8.2)
    ax.grid(alpha=.3, ls=':')
    if tag == 'b':
        ax.legend(fontsize=7.5, loc='upper right')
for k, ax in enumerate(axs):
    if k % 3 == 0: ax.set_ylabel('Normalised response (dB)')
    if k >= 6: ax.set_xlabel('Frequency (GHz)')
fig.suptitle('"Truncated sweep": the measurement stops right after the $-3$ dB '
             'crossing (grey band = data the cubic never sees), so $f_{3dB}$ is '
             f'biased high.  Dropped when $f_{{3dB}}$ > {FRAC_MAX:.0%} of the sweep end.',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig('ft_truncation_demo.png', dpi=200, facecolor='white')
print('wrote ft_truncation_demo.png')
print(drop[['D', 'V', 'lab', 'camp', 'f3', 'fmax', 'frac']].to_string(
      index=False, float_format=lambda v: f'{v:.2f}'))
