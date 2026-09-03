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
  'Bias_5V_Iph_1mA_30um_WO_1.xlsx', ''),
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

def user_f3(sheet):
    df = pd.read_excel(os.path.join(BWD, sheet), header=14)
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
        return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), 'poly3'
    # the cubic can miss a drop confined to the last points; fall back to the
    # raw crossing when the measured data itself goes below -3 dB
    rel = p - ref
    k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]
        return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]])), 'raw'
    return np.nan, 'none'

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
    f3, f3src = user_f3(sheet)
    rows.append(dict(D=D, V=V, lab=lab, camp=camp, Rm=Rm_meas, open=op,
                     Cpd=Cpd*1e15, L1=L1*1e12, Lm=Lm*1e12, L2=L2*1e12,
                     rms=rms, f_RC=fRC, f3=f3, f3src=f3src, note=note))

t = pd.DataFrame(rows).sort_values(['D', 'V', 'Rm'])
t['x'] = 1000/t.f_RC**2
t['y'] = 1000/t.f3**2
t['fT_dev'] = np.where(t.y > t.x, np.sqrt(1000/(t.y - t.x).where(t.y > t.x)),
                       np.nan)
pd.set_option('display.width', 240)
print(t.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
t.to_csv('ft_userbw.csv', index=False)
print('\nexcluded:')
for a, b in EXCL:
    print(f'  {a}\n      {b}')

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan)

print(f"\n{'subset':<30s}{'N':>3s}{'slope':>8s}{'intercept':>11s}"
      f"{'Adj.R2':>8s}{'f_T':>10s}")
SUB = [('-7 V, all', t[t.V == -7]),
       ('-5 V, all', t[t.V == -5]),
       ('-3 V, all', t[t.V == -3]),
       ('all biases', t),
       ('-7 V, 30 um Mar', t[(t.V == -7) & (t.camp == 'Mar')]),
       ('-7 V, 30 um Jan', t[(t.V == -7) & (t.camp == 'Jan')]),
       ('-7 V, 30 um both', t[(t.V == -7) & t.camp.isin(['Jan', 'Mar'])]),
       ('-7 V, 40 um', t[(t.V == -7) & (t.D == 40)]),
       ('30 um Mar, all biases', t[t.camp == 'Mar'])]
res = {}
for name, s in SUB:
    s = s[np.isfinite(s.x) & np.isfinite(s.y)]
    if len(s) < 3:
        continue
    sl, b, adj, ft = freefit(s.x.values, s.y.values)
    res[name] = (sl, b, adj, ft, s)
    print(f"{name:<30s}{len(s):3d}{sl:8.3f}{b:+11.3f}{adj:8.3f}"
          + (f"{ft:10.1f}" if np.isfinite(ft) else f"{'none':>10s}"))

# ── plot ───────────────────────────────────────────────────────────────────
MK = {25: 'o', 30: 's', 40: '^'}
CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.2))

ax = axs[0]
s7 = t[(t.V == -7) & np.isfinite(t.x) & np.isfinite(t.y)]
sl, b, adj, ft = freefit(s7.x.values, s7.y.values)
for D in (25, 30, 40):
    q = s7[s7.D == D]
    if not len(q):
        continue
    ax.scatter(q.x, q.y, s=72, marker=MK[D], facecolor='none',
               edgecolor=CV[-7], lw=1.7, zorder=4, label=f'{D} $\\mu$m')
xr = np.linspace(0, s7.x.max()*1.15, 40)
ax.plot(xr, sl*xr + b, 'k-', lw=1.6, zorder=3, label=f'free fit, slope {sl:.2f}')
ax.plot(xr, xr, ':', color='0.5', lw=1.4, zorder=2, label='unit slope (required)')
ax.annotate(f'$-7$ V,  $N$ = {len(s7)}\nslope = {sl:.3f}\n'
            f'intercept = {b:+.3f}\nAdj. $R^2$ = {adj:.3f}\n'
            + (f'$f_T$ = {ft:.1f} GHz' if np.isfinite(ft)
               else '$f_T$ not extractable'),
            xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=9)
ax.set_title('(a) $-7$ V, user-supplied $f_{3dB}$', fontsize=10)

ax2 = axs[1]
for V in (-3, -5, -7):
    q = t[(t.V == V) & np.isfinite(t.x) & np.isfinite(t.y)]
    for D in (25, 30, 40):
        u = q[q.D == D]
        if not len(u):
            continue
        ax2.scatter(u.x, u.y, s=72, marker=MK[D], facecolor='none',
                    edgecolor=CV[V], lw=1.7, zorder=4,
                    label=f'{D} $\\mu$m, {V} V')
qa = t[np.isfinite(t.x) & np.isfinite(t.y)]
sl2, b2, adj2, ft2 = freefit(qa.x.values, qa.y.values)
xr2 = np.linspace(0, qa.x.max()*1.12, 40)
ax2.plot(xr2, sl2*xr2 + b2, 'k-', lw=1.6, zorder=3)
ax2.plot(xr2, xr2, ':', color='0.5', lw=1.4, zorder=2)
ax2.annotate(f'all biases,  $N$ = {len(qa)}\nslope = {sl2:.3f}\n'
             f'intercept = {b2:+.3f}\nAdj. $R^2$ = {adj2:.3f}\n'
             + (f'$f_T$ = {ft2:.1f} GHz' if np.isfinite(ft2)
                else '$f_T$ not extractable'),
             xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=9)
ax2.set_title('(b) $-3$, $-5$ and $-7$ V together', fontsize=10)

for a in (ax, ax2):
    a.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
    a.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')
    a.set_xlim(left=0); a.set_ylim(bottom=0)
    a.grid(alpha=.3, ls=':'); a.legend(fontsize=7.5, loc='lower right')
fig.tight_layout()
fig.savefig('ft_userbw.png', dpi=300)
print('\nwrote ft_userbw.png, ft_userbw.csv')
