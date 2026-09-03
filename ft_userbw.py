"""f_T extraction using the user-supplied bandwidth sheets.

Pairings are settled by measurement date, not by folder or file name:

  30 um  January  campaign : BW 01/21/2026  <->  S11 01/26/2026
  30 um  March    campaign : BW 03/27-29    <->  S11 03/30/2026
  40 um  V1                : BW 02/04       <->  S11 02/26/2026
  40 um  V2                : BW 02/25       <->  S11 02/26/2026
  25 um                    : BW 02/16       <->  S11 02/13/2026

f_3dB comes from the user's sheets ('Cal RF POW (dBm)', 3rd-order polynomial,
DC reference extrapolated to f = 0). f_RC comes from the 2-L ladder fitted to
that device's own S11, with R_s and C_CPW locked at the project baseline,
C_PD from the measured C-V, and R_m read off S11 at 10 MHz.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

exec(open('ft_extraction_multiD.py').read().split('# ── main loop')[0])

C_PD_CV = {(25, -3): 166.7, (25, -5): 119.9, (25, -7): 100.4,
           (30, -3): 230.8, (30, -5): 161.1, (30, -7): 133.5,
           (40, -3): 390.4, (40, -5): 274.6, (40, -7): 227.6}

M = '30um/main_figure_03_30_2026'      # relative to S11D
# (D, bias, label, campaign, S11 file, bandwidth sheet, note)
PAIR = [
 (25, -7, 'WO',    '25um',   '25um/WO_-7V.s2p',
  'Bias_7V_Iph_1mA_diff_probe_upto_30GHz_25um_WO.xlsx', ''),
 (25, -5, 'WO',    '25um',   '25um/WO_-5V.s2p',
  'Bias_5V_Iph_1mA_diff_probe_upto_30GHz_25um_WO.xlsx', ''),

 (30, -7, '38',    'Mar',    f'{M}/33ohm/Bias_-7V_33ohm.s1p',
  'Bias_7V_Iph_1mA_30um_38ohm_2.xlsx', ''),
 (30, -7, '60',    'Mar',    f'{M}/55ohm/Bias_-7V_55ohm.s1p',
  'Bias_7V_Iph_1mA_30um_60ohm_2.xlsx', ''),
 (30, -7, '200',   'Mar',    f'{M}/200 ohm-1/Bias_-7V_200ohm-1.s1p',
  'Bias_7V_Iph_1mA_30um_200ohm_1.xlsx', ''),
 (30, -7, 'WO',    'Mar',    f'{M}/WO/Bias_-7V_WO.s1p',
  'Bias_7V_Iph_1mA_30GHz_30um_WO_2.xlsx', ''),
 (30, -5, '38',    'Mar',    f'{M}/33ohm/Bias_-5V_38ohm.s1p',
  'Bias_5V_Iph_1mA_30um_38ohm_1.xlsx', ''),
 (30, -5, '60',    'Mar',    f'{M}/55ohm/Bias_-5V_60ohm.s1p',
  'Bias_5V_Iph_1mA_30um_60ohm_1.xlsx', ''),
 (30, -5, '200',   'Mar',    f'{M}/200 ohm-1/Bias_-5V_200ohm-1.s1p',
  'Bias_5V_Iph_1mA_30um_200ohm_1.xlsx', ''),
 (30, -5, 'WO',    'Mar',    f'{M}/WO/Bias_-5V_WO.s1p',
  'data_PD0008_1/Bandwidth/30um/Figure_03_27_2026/WO/'
  'Bias_-5V_Iph_1mA_30GHz.xlsx',
  '30 GHz sweep; the 19.5 GHz sheet reads 17.25 GHz, biased high by truncation'),
 (30, -3, '60',    'Mar',    f'{M}/55ohm/Bias_-3V_55ohm.s1p',
  'Bias_3V_Iph_1mA_30um_60ohm_1.xlsx', ''),
 (30, -3, '200',   'Mar',    f'{M}/200 ohm-1/Bias_-3V_200ohm-1.s1p',
  'Bias_3V_Iph_1mA_30um_200ohm_1.xlsx', ''),
 (30, -3, 'WO',    'Mar',    f'{M}/WO/Bias_-3V_WO.s1p',
  'Bias_3V_Iph_1mA_30um_WO_1.xlsx', ''),

 (30, -7, '38',    'Jan',    '30um/32 ohm/-7V.s1p',
  'Bias_7V_Iph_1mA_38ohm_30um_1.xlsx', ''),
 (30, -7, '76',    'Jan',    '30um/71 ohm/-7V.s1p',
  'Bias_7V_Iph_1mA_30um_76ohm_1.xlsx', ''),
 (30, -7, '120',   'Jan',    '30um/120 ohm/-7V.s1p',
  'Bias_7V_Iph_1mA_30um_120ohm_1.xlsx', ''),
 (30, -7, 'WO',    'Jan',    '30um/WO/-7V.s1p',
  'Bias_7V_Iph_1mA_30um_WO_1.xlsx', ''),

 (40, -7, '38',    '40V1',   '40um/40ohm_V1/S11_-7V_36ohm.s1p',
  'Bias_7V_Iph_1mA_40um_38ohm_1.xlsx', ''),
 (40, -7, '100',   '40V1',   '40um/100ohm_V1/S11_-7V.s1p',
  'Bias_7V_Iph_1mA_40um_100ohm_1.xlsx', ''),
 (40, -7, '140',   '40V1',   '40um/140ohm_V1/S11_-7V_120ohm.s1p',
  'Bias_7V_Iph_1mA_120ohm_40um_140ohm_1.xlsx', ''),
 (40, -7, 'WO',    '40V1',   '40um/WO_V2/S11_-7V_without.s1p',
  'Bias_7V_Iph_1mA_40um_WO_3.xlsx', ''),
 (40, -5, 'WO',    '40V2',   '40um/WO_V2/S11_-5V.s1p',
  'Bias_5V_Iph_1mA_40um_WO_2.xlsx', ''),
 # the sheets named "40um_38ohm_2" are the 80 ohm V2 device (user confirmed)
 (40, -7, '80',    '40V2',   '40um/80ohm_V2/S11_-7V.s1p',
  'Bias_7V_Iph_1mA_40um_38ohm_2.xlsx', 'sheet labelled 38ohm_2'),
 (40, -5, '80',    '40V2',   '40um/80ohm_V2/S11_-5V.s1p',
  'Bias_5V_Iph_1mA_40um_38ohm_2.xlsx', 'sheet labelled 38ohm_2'),
]
# excluded, with the reason
EXCL = [
 ('30 um 60 ohm, Jan run 1 (36.54 GHz)',
  'no S11 for this device in the January set (that set has 120/38/76/WO)'),
 ('40 um WO, run 2 at -7 V (27.29 GHz)',
  'same-day run 3 gives 10.17 GHz and -5 V gives 10.03 GHz; 27.29 GHz is '
  'also above the modelled f_RC, so run 2 is the outlier'),
]

BWD = 'data_bw_user'

# a -3 dB point sitting this far into a device's own sweep is unreliable:
# the 30 um open device reads 17.25 GHz from a 19.5 GHz sweep against
# 14.58 GHz from a 30 GHz sweep of the same device and bias.
FRAC_MAX = 0.85

def sheet_path(sheet):
    q = os.path.join(BWD, sheet)
    return q if os.path.exists(q) else sheet

def user_f3(sheet):
    df = pd.read_excel(sheet_path(sheet), header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6]); f, p = f[k], p[k]
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]
        return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), 'poly3', f[-1]
    # the cubic can miss a drop confined to the last points; fall back to the
    # raw crossing when the measured data itself goes below -3 dB
    rel = p - ref
    k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]
        return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]])), 'raw', f[-1]
    return np.nan, 'none', f[-1]

fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
rows = []
for D, V, lab, camp, s11, sheet, note in PAIR:
    sp = os.path.join(S11D, s11)
    if not os.path.exists(sp):
        print('MISSING S11', sp); continue
    f, Sm = read_s11(sp); w = 2*np.pi*f
    Rm_meas = (R_L*(1 + Sm[0])/(1 - Sm[0])).real
    op = lab == 'WO'
    Rm = np.inf if op else Rm_meas
    Cpd = C_PD_CV[(D, V)]*1e-15

    keys = ['L1', 'L2'] if op else ['L1', 'Lm', 'L2']
    p0 = {'L1': 60.0, 'Lm': 70.0, 'L2': 130.0}
    def unpack(p):
        v = {'L1': 0.0, 'Lm': 0.0, 'L2': 0.0}
        for k, val in zip(keys, p):
            v[k] = val*1e-12
        return v['L1'], v['Lm'], v['L2']
    def resid(p):
        L1, Lm, L2 = unpack(p)
        e = S11_model(w, Cpd, Rm, L1, Lm, L2) - Sm
        return np.concatenate([e.real, e.imag])
    r = least_squares(resid, [p0[k] for k in keys],
                      bounds=([0.0]*len(keys), [600.0]*len(keys)),
                      x_scale=[50.0]*len(keys))
    L1, Lm, L2 = unpack(r.x)
    rms = np.sqrt(np.mean(np.abs(S11_model(w, Cpd, Rm, L1, Lm, L2) - Sm)**2))
    fRC = f3dB_of(fg, H_ckt(wg, Cpd, Rm, L1, Lm, L2))/1e9
    f3, f3src, fmax = user_f3(sheet)
    rows.append(dict(D=D, V=V, lab=lab, camp=camp, Rm=Rm_meas, open=op,
                     Cpd=Cpd*1e15, L1=L1*1e12, Lm=Lm*1e12, L2=L2*1e12,
                     rms=rms, f_RC=fRC, f3=f3, f3src=f3src, fmax=fmax,
                     note=note))

t = pd.DataFrame(rows).sort_values(['D', 'V', 'Rm'])
t['x'] = 1000/t.f_RC**2
t['y'] = 1000/t.f3**2
t['frac'] = t.f3/t.fmax          # where -3 dB sits within the device's sweep
t['ok'] = np.isfinite(t.x) & np.isfinite(t.y) & (t.frac <= FRAC_MAX)
t['fT_dev'] = np.where(t.y > t.x, np.sqrt(1000/(t.y - t.x).where(t.y > t.x)),
                       np.nan)
pd.set_option('display.width', 250)
print(t[['D', 'V', 'lab', 'camp', 'Rm', 'Cpd', 'rms', 'f_RC', 'f3', 'fmax',
         'frac', 'ok', 'x', 'y', 'fT_dev', 'f3src']].to_string(
      index=False, float_format=lambda v: f'{v:.2f}'))
t.to_csv('ft_userbw.csv', index=False)
print('\nexcluded outright:')
for a, b in EXCL:
    print(f'  {a}\n      {b}')
print(f'\ndropped as truncated (f_3dB above {FRAC_MAX:.0%} of the sweep span):')
for _, r in t[~t.ok & np.isfinite(t.frac)].sort_values('frac').iterrows():
    print(f'  {int(r.D)} um {r.lab:>3s} {int(r.V)} V   '
          f'f_3dB {r.f3:5.2f} of {r.fmax:5.2f} GHz  ({r.frac:.0%})')

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan)

g = t[t.ok]
print(f"\n{'subset':<38s}{'N':>3s}{'slope':>8s}{'k':>7s}{'intercept':>11s}"
      f"{'Adj.R2':>8s}{'f_T':>9s}")
SUB = [('untruncated, all biases',            g),
       ('untruncated, -7 V',                  g[g.V == -7]),
       ('untruncated, 30 um',                 g[g.D == 30]),
       ('untruncated, 40 um',                 g[g.D == 40]),
       ('everything incl. truncated',         t[np.isfinite(t.x) & np.isfinite(t.y)]),
       ('truncated only',                     t[~t.ok & np.isfinite(t.frac)])]
res = {}
for name, s in SUB:
    s = s[np.isfinite(s.x) & np.isfinite(s.y)]
    if len(s) < 3:
        continue
    sl, b, adj, ft = freefit(s.x.values, s.y.values)
    res[name] = (sl, b, adj, ft, s)
    print(f"{name:<38s}{len(s):3d}{sl:8.3f}{np.sqrt(sl):7.3f}{b:+11.3f}"
          f"{adj:8.3f}" + (f"{ft:9.1f}" if np.isfinite(ft) else f"{'none':>9s}"))

# ── plot ───────────────────────────────────────────────────────────────────
MK = {25: 'o', 30: 's', 40: '^'}
CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.3))

ax = axs[0]
sl, b, adj, ft = freefit(g.x.values, g.y.values)
tr = t[~t.ok & np.isfinite(t.frac)]
ax.scatter(tr.x, tr.y, s=58, marker='x', color='0.62', lw=1.4, zorder=3,
           label=f'truncated sweep, dropped ($N$ = {len(tr)})')
for V in (-3, -5, -7):
    for D in (25, 30, 40):
        q = g[(g.V == V) & (g.D == D)]
        if not len(q):
            continue
        ax.scatter(q.x, q.y, s=74, marker=MK[D], facecolor='none',
                   edgecolor=CV[V], lw=1.7, zorder=5,
                   label=f'{D} $\\mu$m, {V} V')
xr = np.linspace(0, g.x.max()*1.12, 40)
ax.plot(xr, sl*xr + b, 'k-', lw=1.6, zorder=4,
        label=f'fit to untruncated, slope {sl:.2f}')
ax.plot(xr, xr, ':', color='0.5', lw=1.4, zorder=2, label='unit slope')
ax.annotate(f'untruncated only,  $N$ = {len(g)}\n'
            f'slope = {sl:.3f}  ($k$ = {np.sqrt(sl):.3f})\n'
            f'intercept = {b:+.3f}\nAdj. $R^2$ = {adj:.3f}\n'
            + (f'$f_T$ = {ft:.1f} GHz' if np.isfinite(ft)
               else '$f_T$ not extractable'),
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=9)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_title('(a) $f_T$ from the untruncated sweeps', fontsize=10)
ax.legend(fontsize=7, loc='lower right')

# (b) sensitivity of the answer to where the truncation line is drawn
ax2 = axs[1]
th = np.arange(0.70, 1.005, 0.01)
FT, SL, NN = [], [], []
for x0 in th:
    s = t[np.isfinite(t.x) & np.isfinite(t.y) & (t.frac <= x0)]
    if len(s) < 3:
        FT.append(np.nan); SL.append(np.nan); NN.append(len(s)); continue
    a, c, _, f_ = freefit(s.x.values, s.y.values)
    FT.append(f_); SL.append(a); NN.append(len(s))
ax2.plot(th, FT, '-o', ms=3.4, color='#c0392b', label=r'extracted $f_T$')
ax2.axhline(32.6, color='0.35', ls='--', lw=1.2,
            label=r'$f_{tr}$ = 32.6 GHz ($H_{ph}$ model)')
ax2.axvline(FRAC_MAX, color='0.6', ls=':', lw=1.3)
ax2.set_xlabel(r'truncation cut: keep points with $f_{3dB}/f_{sweep,max}\leq$')
ax2.set_ylabel(r'extracted $f_T$  (GHz)')
ax2.set_ylim(0, 90)
axb = ax2.twinx()
axb.plot(th, SL, '-s', ms=3.0, color='#2471a3', alpha=.85)
axb.set_ylabel('free-fit slope', color='#2471a3')
axb.tick_params(axis='y', colors='#2471a3')
axb.axhline(1.0, color='#2471a3', ls=':', lw=1.0)
ax2.set_title('(b) how the answer moves with the truncation cut', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')

for a in (ax,):
    a.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
    a.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')
for a in (ax, ax2):
    a.grid(alpha=.3, ls=':')
fig.tight_layout()
fig.savefig('ft_userbw.png', dpi=300)
print('\nwrote ft_userbw.png, ft_userbw.csv')
