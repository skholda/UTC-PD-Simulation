"""Review every 30 um open-device (WO) bandwidth sweep.

The -7 V and -5 V values used so far are nearly identical (17.46 vs 17.25 GHz)
even though C_PD grows 21% between those biases, which is not physical. The
two sheets differ in sweep span (30 GHz vs 19.5 GHz) and the archive also
holds a 30 GHz -5 V sweep and a 'new cal' re-calibration that were not in the
set used. This compares them all under one method, and tests how sensitive
f_3dB is to truncating the fit range.
"""
import os, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = 'data_PD0008_1/Bandwidth/30um'
SHEETS = [
 ('Jan  -7 V  (user)',      -7, 'data_bw_user/Bias_7V_Iph_1mA_30um_WO_1.xlsx'),
 ('Mar  -7 V  30GHz (user)',-7, 'data_bw_user/Bias_7V_Iph_1mA_30GHz_30um_WO_2.xlsx'),
 ('Mar  -7 V  30GHz new cal', -7,
  f'{A}/Figure_03_27_2026/WO/new cal/Bias_-7V_Iph_1mA_30GHz_WO.xlsx'),
 ('Mar  -5 V  19.5GHz (user)', -5, 'data_bw_user/Bias_5V_Iph_1mA_30um_WO_1.xlsx'),
 ('Mar  -5 V  30GHz',       -5, f'{A}/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA_30GHz.xlsx'),
 ('Mar  -5 V  19.5GHz new cal', -5,
  f'{A}/Figure_03_27_2026/WO/new cal/Bias_-5V_Iph_1mA.xlsx'),
 ('Mar  -5 V  30GHz new cal', -5,
  f'{A}/Figure_03_27_2026/WO/new cal/Bias_-5V_Iph_1mA_30GHz.xlsx'),
 ('Mar  -3 V  (user)',      -3, 'data_bw_user/Bias_3V_Iph_1mA_30um_WO_1.xlsx'),
 ('Mar  -3 V  new cal',     -3, f'{A}/Figure_03_27_2026/WO/new cal/Bias_-3V_Iph_1mA.xlsx'),
]

def load(path):
    df = pd.read_excel(path, header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[k], p[k]

def f3_of(f, p, fmax=None):
    if fmax is not None:
        m = f <= fmax
        if m.sum() < 8:
            return np.nan
        f, p = f[m], p[m]
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]
        return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]]))
    rel = p - ref
    k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]
        return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]]))
    return np.nan

recs = []
print(f"{'sheet':<30s}{'V':>3s}{'N':>4s}{'span (GHz)':>13s}{'f_3dB':>8s}"
      f"{'trunc 20':>10s}{'trunc 25':>10s}{'P(0) dBm':>10s}")
for name, V, path in SHEETS:
    if not os.path.exists(path):
        print(f'{name:<30s}  MISSING  {path}'); continue
    f, p = load(path)
    c = np.polyfit(f, p, 3)
    a = f3_of(f, p)
    b20 = f3_of(f, p, 20.0)
    b25 = f3_of(f, p, 25.0)
    print(f'{name:<30s}{V:3d}{len(f):4d}{f[0]:6.2f}-{f[-1]:<6.2f}'
          f'{a:8.2f}{b20:10.2f}{b25:10.2f}{np.polyval(c, 0.0):10.2f}')
    recs.append(dict(name=name, V=V, f=f, p=p, c=c, f3=a))

# ── overlay ───────────────────────────────────────────────────────────────
CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
LS = ['-', '--', ':', '-.']
fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.0))

seen = {}
for r in recs:
    k = seen.get(r['V'], 0); seen[r['V']] = k + 1
    ref = np.polyval(r['c'], 0.0)
    axs[0].plot(r['f'], r['p'] - ref, 'o', ms=3, mfc='none', alpha=.55,
                color=CV[r['V']])
    ff = np.linspace(0, r['f'][-1], 600)
    axs[0].plot(ff, np.polyval(r['c'], ff) - ref, LS[k % 4], lw=1.5,
                color=CV[r['V']], label=f"{r['name']}  ({r['f3']:.1f} GHz)")
axs[0].axhline(-3, color='0.5', ls='--', lw=1.1)
axs[0].set_xlabel('Frequency (GHz)')
axs[0].set_ylabel('Normalised response (dB)')
axs[0].set_ylim(-9, 2); axs[0].set_xlim(0, 32)
axs[0].grid(alpha=.3, ls=':')
axs[0].legend(fontsize=7, loc='lower left')
axs[0].set_title('(a) all 30 $\\mu$m open-device sweeps, normalised',
                 fontsize=10)

# absolute level: an open device's DC response should scale with nothing but
# I_ph, so P(0) ought to agree between biases
for r in recs:
    axs[1].plot(r['f'], r['p'], 'o-', ms=2.6, lw=1.0, color=CV[r['V']],
                alpha=.8, label=f"{r['name']}")
axs[1].set_xlabel('Frequency (GHz)')
axs[1].set_ylabel('Cal RF POW (dBm), absolute')
axs[1].set_xlim(0, 32); axs[1].grid(alpha=.3, ls=':')
axs[1].legend(fontsize=7, loc='lower left')
axs[1].set_title('(b) same sweeps without normalising', fontsize=10)
fig.tight_layout()
fig.savefig('wo_review.png', dpi=300)
print('\nwrote wo_review.png')
