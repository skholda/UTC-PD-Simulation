"""Truncation bias tested on single sweeps (no cross-run comparison).

(a) the two -5 V open-device runs of 03/27 in raw dBm, un-normalised: they
    agree to a few tenths of a dB where they overlap, so the difference in
    the earlier normalised plot came from each run's own cubic setting its
    own f = 0 reference, not from the data.
(b)-(f) self-truncation: ONE long sweep is cut at decreasing f_max and the
    cubic is refitted each time. f_3dB is plotted against f_3dB/f_max, the
    ratio used for the 85% cut. The bias appears from the data alone.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)          # sheet_path, FRAC_MAX

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
    if len(i) and i[0] > 0:
        j = i[0]
        return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), c, ref
    return np.nan, c, ref

def self_truncate(f, p, nmin=12):
    """Refit after cutting the sweep at every measured point from the end."""
    out = []
    for k in range(len(f), nmin - 1, -1):
        fk, pk = f[:k], p[:k]
        f3, _, _ = poly_f3(fk, pk)
        out.append((fk[-1], f3))
    return np.array(out)

A = 'data_PD0008_1/Bandwidth/30um/Figure_03_27_2026/WO'
LONG = [   # long sweeps to self-truncate: (label, sheet, colour)
 ('30 $\\mu$m open, $-5$ V  (30 GHz run)',   f'{A}/Bias_-5V_Iph_1mA_30GHz.xlsx',        '#2471a3'),
 ('30 $\\mu$m open, $-7$ V  (30 GHz run)',   'data_bw_user/Bias_7V_Iph_1mA_30GHz_30um_WO_2.xlsx', '#c0392b'),
 ('25 $\\mu$m open, $-7$ V  (31 GHz)',       'data_bw_user/Bias_7V_Iph_1mA_diff_probe_upto_30GHz_25um_WO.xlsx', '#1f77b4'),
 ('30 $\\mu$m 38 $\\Omega$, $-7$ V  (38 GHz)','data_bw_user/Bias_7V_Iph_1mA_38ohm_30um_1.xlsx', '#8e44ad'),
 ('30 $\\mu$m 200 $\\Omega$, $-7$ V  (26 GHz)','data_bw_user/Bias_7V_Iph_1mA_30um_200ohm_1.xlsx', '#2e8b57'),
]

fig, axs = plt.subplots(2, 3, figsize=(14.4, 8.6))
axs = axs.ravel()

# ── (a) raw overlay of the two -5 V runs ──────────────────────────────────
ax = axs[0]
for sheet, col, lab in (('data_bw_user/Bias_5V_Iph_1mA_30um_WO_1.xlsx', '#e67e22',
                         '12:42 run, to 19.5 GHz'),
                        (f'{A}/Bias_-5V_Iph_1mA_30GHz.xlsx', '#2471a3',
                         '13:01 run, to 30 GHz')):
    f, p = load(sheet)
    ax.plot(f, p, 'o-', ms=3.4, lw=0.9, mfc='none', color=col, label=lab)
ax.set_xlim(0, 31); ax.set_ylim(-26, -15)
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Cal RF POW (dBm), raw')
ax.set_title('(a)  the two $-5$ V runs, raw, no normalisation', fontsize=9.5)
ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=.3, ls=':')
f1, p1 = load('data_bw_user/Bias_5V_Iph_1mA_30um_WO_1.xlsx')
f2, p2 = load(f'{A}/Bias_-5V_Iph_1mA_30GHz.xlsx')
d = np.abs(np.interp(f1, f2, p2) - p1)
ax.annotate(f'overlap 1–19.5 GHz:\nmean |diff| = {d[(f1>=1)].mean():.2f} dB\n'
            f'max  |diff| = {d[(f1>=1)].max():.2f} dB',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=8.5)

# ── (b)-(f) self-truncation ───────────────────────────────────────────────
summary = []
for ax, (lab, sheet, col), tag in zip(axs[1:], LONG, 'bcdef'):
    f, p = load(sheet)
    st = self_truncate(f, p)
    full = st[0, 1]
    ok = np.isfinite(st[:, 1])
    ratio = st[ok, 1]/st[ok, 0]
    ax.plot(ratio, st[ok, 1], 'o-', ms=3.6, lw=1.0, color=col)
    ax.axhline(full, color='0.4', ls='--', lw=1.1,
               label=f'full sweep: {full:.2f} GHz')
    ax.axvline(FRAC_MAX, color='0.6', ls=':', lw=1.3, label=f'cut = {FRAC_MAX:.0%}')
    # where does the estimate leave a ±5% band around the full-sweep value?
    band = np.abs(st[ok, 1] - full)/full <= 0.05
    if (~band).any():
        r_bad = ratio[~band].min()
        ax.axvspan(r_bad, 1.0, color='#f4b6a0', alpha=.35, zorder=0)
    else:
        r_bad = np.nan
    ax.set_xlim(0.45, 1.0)
    lo, hi = np.nanmin(st[ok, 1]), np.nanmax(st[ok, 1])
    ax.set_ylim(lo - 0.12*(hi - lo) - 0.3, hi + 0.12*(hi - lo) + 0.3)
    ax.set_xlabel(r'$f_{3dB}$ / (truncated sweep end)')
    ax.set_ylabel(r'$f_{3dB}$ from cubic (GHz)')
    ax.set_title(f'({tag})  {lab}', fontsize=9.5)
    ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7.6, loc='upper left')
    # value at the ratio closest to 0.90 and 0.95
    def at(r):
        i = np.argmin(np.abs(ratio - r)); return st[ok, 1][i], ratio[i]
    v90, r90 = at(0.90); v95, r95 = at(0.95)
    ax.annotate(f'at {r90:.0%}: {v90:.2f} GHz ({(v90/full-1)*100:+.0f}%)\n'
                f'at {r95:.0%}: {v95:.2f} GHz ({(v95/full-1)*100:+.0f}%)',
                xy=(0.97, 0.06), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=8.2)
    summary.append((lab, full, v90, v95, r_bad))

fig.suptitle('Truncation bias from single sweeps: cut one sweep progressively '
             'shorter and refit the cubic each time.  Shaded: estimate off by '
             '> 5% from the full-sweep value.', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig('ft_truncation_self.png', dpi=200, facecolor='white')

print(f"{'sweep':<42s}{'full':>7s}{'@90%':>14s}{'@95%':>14s}{'>5% off from':>13s}")
for lab, full, v90, v95, r_bad in summary:
    lab = lab.replace('$\\mu$', 'u').replace('$\\Omega$', 'ohm').replace('$-', '-').replace('$', '')
    rb = f'{r_bad:.0%}' if np.isfinite(r_bad) else 'never'
    print(f'{lab:<42s}{full:7.2f}{v90:8.2f} ({(v90/full-1)*100:+3.0f}%)'
          f'{v95:8.2f} ({(v95/full-1)*100:+3.0f}%){rb:>13s}')
print('\nwrote ft_truncation_self.png')
