"""
f_T extraction from 1/f_3dB^2 vs 1/f_RC^2  (multi-diameter, multi-resistor)
===========================================================================
1/f_3dB^2 = 1/f_RC^2 + 1/f_T^2   ->  linear fit, intercept = 1/f_T^2

  f_RC   : from the 2-L ladder circuit model (baseline topology), fitted to the
           measured S11 of each individual device.
  f_3dB  : from the measured RF response (Cal RF POW, dBm), 3rd-order
           polynomial fit.

Devices: PD0008-1, D = 25 / 30 / 40 um, several shunt resistors, -7 V and -5 V.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import least_squares, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = 'data_PD0008_1'
S11D = f'{ROOT}/S11'
BWD  = f'{ROOT}/Bandwidth'

# ── locked circuit constants (unchanged from project baseline) ──────────────
Rs_FIX    = 8.92        # ohm   series resistance
C_CPW_FIX = 46.53e-15   # F     CPW pad capacitance (confirmed by 25um pad-only S11)
R_L       = 50.0        # ohm

# ── C_PD from measured C-V (data_CV/CV_summary.txt) ─────────────────────────
C_PD_CV = {   # fF, keyed by (D_um, bias_V)
    (25, -5): 119.9, (25, -7): 100.4,
    (30, -5): 161.1, (30, -7): 133.5,
    (40, -5): 274.6, (40, -7): 227.6,
}

# ── device list: (D, bias, label, S11 file, bandwidth xlsx) ─────────────────
DEV = [
 # ---- -7 V -------------------------------------------------------------
 (25,-7,'WO',     '25um/WO_-7V.s2p',
                  '25 um/Bias_-7V_Iph_1mA_diff_probe_upto_30GHz.xlsx'),
 (30,-7,'200',    '30um/main_figure_03_30_2026/200 ohm-1/Bias_-7V_200ohm-1.s1p',
                  '30um/Figure_03_27_2026/200 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (30,-7,'33/36',  '30um/main_figure_03_30_2026/33ohm/Bias_-7V_33ohm.s1p',
                  '30um/Figure_03_27_2026/36 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (30,-7,'55',     '30um/main_figure_03_30_2026/55ohm/Bias_-7V_55ohm.s1p',
                  '30um/Figure_03_27_2026/55 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (30,-7,'WO(A)',  '30um/main_figure_03_30_2026/WO/Bias_-7V_WO.s1p',
                  '30um/Figure_03_27_2026/WO/Bias_-7V_Iph_1mA_30GHz.xlsx'),
 (30,-7,'120',    '30um/120 ohm/-7V.s1p',
                  '30um/120 ohm/Bias_-7V_Iph_1.034mA.xlsx'),
 (30,-7,'32/40',  '30um/32 ohm/-7V.s1p',
                  '30um/40 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (30,-7,'71/76',  '30um/71 ohm/-7V.s1p',
                  '30um/76 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (30,-7,'WO(B)',  '30um/WO/-7V.s1p',
                  '30um/WO/Bias_-7V_Iph_1mA.xlsx'),
 (40,-7,'100',    '40um/100ohm_V1/S11_-7V.s1p',
                  '40um/100ohm/Bias_-7V_Iph_1mA.xlsx'),
 (40,-7,'140',    '40um/140ohm_V1/S11_-7V_120ohm.s1p',
                  '40um/140hom/Bias_-7V_Iph_1mA_120ohm.xlsx'),
 (40,-7,'40',     '40um/40ohm_V1/S11_-7V_36ohm.s1p',
                  '40um/40ohm/Bias_-7V_Iph_1mA_36ohm.xlsx'),
 (40,-7,'80(V2)', '40um/80ohm_V2/S11_-7V.s1p',
                  '40um/V2/40 ohm/Bias_-7V_Iph_1mA.xlsx'),
 (40,-7,'WO(V2)', '40um/WO_V2/S11_-7V_without.s1p',
                  '40um/V2/WO/Bias_-7V_Iph_1mA.xlsx'),
 # ---- -5 V -------------------------------------------------------------
 (25,-5,'WO',     '25um/WO_-5V.s2p',
                  '25 um/Bias_-5V_Iph_1mA_diff_probe_upto_30GHz.xlsx'),
 (30,-5,'200',    '30um/main_figure_03_30_2026/200 ohm-1/Bias_-5V_200ohm-1.s1p',
                  '30um/Figure_03_27_2026/200 ohm/Bias_-5V_Iph_1mA.xlsx'),
 (30,-5,'33/36',  '30um/main_figure_03_30_2026/33ohm/Bias_-5V_38ohm.s1p',
                  '30um/Figure_03_27_2026/36 ohm/Bias_-5V_Iph_1mA.xlsx'),
 (30,-5,'55',     '30um/main_figure_03_30_2026/55ohm/Bias_-5V_60ohm.s1p',
                  '30um/Figure_03_27_2026/55 ohm/Bias_-5V_Iph_1mA.xlsx'),
 (30,-5,'WO(A)',  '30um/main_figure_03_30_2026/WO/Bias_-5V_WO.s1p',
                  '30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA_30GHz.xlsx'),
 (40,-5,'80(V2)', '40um/80ohm_V2/S11_-5V.s1p',
                  '40um/V2/40 ohm/Bias_-5V_Iph_1mA.xlsx'),
 (40,-5,'WO(V2)', '40um/WO_V2/S11_-5V.s1p',
                  '40um/V2/WO/Bias_-5V_Iph_1mA.xlsx'),
]

# ── touchstone reader ───────────────────────────────────────────────────────
def read_s11(path):
    f, s = [], []
    for ln in open(path, errors='ignore'):
        ln = ln.strip()
        if not ln or ln[0] in '!#':
            continue
        try:
            v = [float(x) for x in ln.split()]
        except ValueError:
            continue
        f.append(v[0]); s.append(v[1:3])          # S11 dB, S11 deg
    f = np.asarray(f); s = np.asarray(s)
    return f, 10**(s[:, 0]/20)*np.exp(1j*np.deg2rad(s[:, 1]))

# ── 2-L ladder ─────────────────────────────────────────────────────────────
#  Iph||C_PD -[Rs]- o -[C_CPW to gnd]- -[L_CPW1]- o -[R_m+L_m to gnd]- -[L_CPW2]- port
def _abcd(w, Cpd, Rm, Lc1, Lm, Lc2, include_Cpd=True):
    A = np.ones_like(w, dtype=complex); B = np.zeros_like(w, dtype=complex)
    C = np.zeros_like(w, dtype=complex); D = np.ones_like(w, dtype=complex)
    def ser(Z):
        nonlocal A, B, C, D
        B, D = A*Z + B, C*Z + D
    def sh(Y):
        nonlocal A, B, C, D
        A, C = A + B*Y, C + D*Y
    if include_Cpd:
        sh(1j*w*Cpd)
    ser(Rs_FIX + 0j*w)
    sh(1j*w*C_CPW_FIX)
    ser(1j*w*Lc1)
    if np.isfinite(Rm):
        sh(1/(Rm + 1j*w*Lm))
    ser(1j*w*Lc2)
    return A, B, C, D

def Zin(w, Cpd, Rm, Lc1, Lm, Lc2):
    """Impedance looking into the port (probe side)."""
    Z1 = Rs_FIX + 1/(1j*w*Cpd)
    Y1 = 1j*w*C_CPW_FIX + 1/Z1
    Z2 = 1j*w*Lc1 + 1/Y1
    Y2 = (1/(Rm + 1j*w*Lm) if np.isfinite(Rm) else 0.0) + 1/Z2
    return 1j*w*Lc2 + 1/Y2

def S11_model(w, Cpd, Rm, Lc1, Lm, Lc2):
    Z = Zin(w, Cpd, Rm, Lc1, Lm, Lc2)
    return (Z - R_L)/(Z + R_L)

def H_ckt(w, Cpd, Rm, Lc1, Lm, Lc2):
    """Transimpedance V_RL / I_ph."""
    A, B, C, D = _abcd(w, Cpd, Rm, Lc1, Lm, Lc2)
    return R_L/(C*R_L + D)

# ── helpers ────────────────────────────────────────────────────────────────
def f3dB_of(fgrid, mag):
    """-3 dB (electrical) point of |H| normalised to its lowest-frequency value."""
    dB = 20*np.log10(np.abs(mag)/np.abs(mag[0]))
    below = np.where(dB <= -3.0)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return np.nan
    g = lambda x: np.interp(x, fgrid, dB) + 3.0
    return brentq(g, fgrid[i-1], fgrid[i])

def load_response(xlsx):
    """Measured Cal RF POW (dBm) vs beat frequency, cleaned.

    The raw sheets occasionally carry a stray first/last row (a re-measured
    point appended out of order, or a saturated +30 dBm reading); those are
    dropped and the sweep is sorted / de-duplicated in frequency.
    """
    df = pd.read_excel(xlsx, header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values   # Cal RF POW (dBm)
    m = np.isfinite(f) & np.isfinite(p) & (p < 0.0) & (f > 0.0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    keep = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[keep], p[keep]

def measured_f3dB(xlsx):
    """-3 dB point from a 3rd-order polynomial fit of the measured response."""
    f, p = load_response(xlsx)
    c = np.polyfit(f, p, 3)
    ref = np.polyval(c, 0.0)                       # DC reference
    ff = np.linspace(0.0, f[-1], 40001)
    pp = np.polyval(c, ff) - ref
    below = np.where(pp <= -3.0)[0]
    f3 = np.nan
    if len(below) and below[0] > 0:
        i = below[0]
        f3 = np.interp(-3.0, [pp[i], pp[i-1]], [ff[i], ff[i-1]])
    reached = np.any(p - ref <= -3.0)              # data actually crosses -3 dB
    if not reached:
        f3 = np.nan
    return f3, f, p, c, ref

# ── main loop ──────────────────────────────────────────────────────────────
fg   = np.linspace(1e6, 120e9, 24001)      # dense grid for f_RC
wg   = 2*np.pi*fg
rows = []

for D, V, lab, s11rel, bwrel in DEV:
    sp = os.path.join(S11D, s11rel)
    bp = os.path.join(BWD,  bwrel)
    if not (os.path.exists(sp) and os.path.exists(bp)):
        print('MISSING', sp if not os.path.exists(sp) else bp); continue

    f, Sm = read_s11(sp)
    w = 2*np.pi*f
    Zm = R_L*(1 + Sm)/(1 - Sm)
    Rm_meas = Zm[0].real                       # DC shunt resistance from S11
    open_dev = (Rm_meas > 1000) or ('WO' in lab)
    Rm = np.inf if open_dev else Rm_meas
    Cpd = C_PD_CV[(D, V)]*1e-15

    # ---- fit L_CPW1, L_CPW2 (+ L_m when a resistor is present); units: pH --
    if open_dev:
        p0, lo, hi = [80.0, 120.0], [0.0, 0.0], [600.0, 600.0]
        pack = lambda p: (p[0]*1e-12, 0.0, p[1]*1e-12)
    else:
        p0 = [80.0, 70.0, 120.0]
        lo = [0.0, 0.0, 0.0]; hi = [600.0, 400.0, 600.0]
        pack = lambda p: (p[0]*1e-12, p[1]*1e-12, p[2]*1e-12)

    def resid(p):
        Lc1, Lm, Lc2 = pack(p)
        e = S11_model(w, Cpd, Rm, Lc1, Lm, Lc2) - Sm
        return np.concatenate([e.real, e.imag])

    r = least_squares(resid, p0, bounds=(lo, hi), x_scale=[50.0]*len(p0))
    Lc1, Lm, Lc2 = pack(r.x)
    rms = np.sqrt(np.mean(np.abs(S11_model(w, Cpd, Rm, Lc1, Lm, Lc2) - Sm)**2))

    # ---- f_RC from the fitted circuit -----------------------------------
    f_RC   = f3dB_of(fg, H_ckt(wg, Cpd, Rm, Lc1, Lm, Lc2))/1e9      # full 2-L
    f_RC0  = f3dB_of(fg, H_ckt(wg, Cpd, Rm, 0.0, 0.0, 0.0))/1e9     # L removed

    # ---- measured f_3dB --------------------------------------------------
    f3, fmeas, pmeas, cpoly, ref = measured_f3dB(bp)

    # ---- screening ------------------------------------------------------
    if not np.isfinite(f3):
        flag = 'no -3dB in sweep'
    elif f3 >= f_RC:
        flag = 'f3dB > fRC'
    elif rms > 0.20:
        flag = 'poor S11 fit'
    else:
        flag = ''

    rows.append(dict(D=D, V=V, lab=lab, Rm=Rm_meas, open=open_dev,
                     Cpd=Cpd*1e15, Lc1=Lc1*1e12, Lm=Lm*1e12, Lc2=Lc2*1e12,
                     rms=rms, f_RC=f_RC, f_RC0=f_RC0, f3=f3, flag=flag,
                     fmeas=fmeas, pmeas=pmeas, cpoly=cpoly, ref=ref,
                     fs=f, Sm=Sm))

df = pd.DataFrame([{k: v for k, v in r.items()
                    if k not in ('fmeas', 'pmeas', 'cpoly', 'ref', 'fs', 'Sm')}
                   for r in rows])
df['use'] = df['flag'] == ''
pd.set_option('display.width', 220)
print(df.to_string(index=False,
      formatters={'Rm': '{:8.1f}'.format, 'Cpd': '{:6.1f}'.format,
                  'Lc1': '{:6.1f}'.format, 'Lm': '{:6.1f}'.format,
                  'Lc2': '{:6.1f}'.format, 'rms': '{:.4f}'.format,
                  'f_RC': '{:6.2f}'.format, 'f3': '{:6.2f}'.format}))

df.to_csv('ft_extraction_multiD.csv', index=False)
df = df[df.use].copy()

# ── f_T extraction ─────────────────────────────────────────────────────────
def fit_ft(sub, col='f_RC'):
    """Unconstrained line  y = a*x + b   and  slope-locked  y = x + b.

    The quadrature relation fixes the slope at 1, so the slope-locked
    intercept  b = <y - x>  is the physically meaningful estimator; the free
    slope is kept as a consistency check on the f_RC model.
    """
    x = 1000/sub[col].values**2
    y = 1000/sub['f3'].values**2
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ np.array([slope, icept])
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum((y - yh)**2)/ss_tot
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    b1 = np.mean(y - x)                       # slope locked to 1
    sd = np.std(y - x, ddof=1)/np.sqrt(len(x))
    return dict(slope=slope, icept=icept, r2=r2, adj=adj, n=len(x), x=x, y=y,
                fT=np.sqrt(1000/icept) if icept > 0 else np.nan,
                b1=b1, b1_se=sd,
                fT1=np.sqrt(1000/b1) if b1 > 0 else np.nan,
                fT1_lo=np.sqrt(1000/(b1 + sd)) if b1 + sd > 0 else np.nan,
                fT1_hi=np.sqrt(1000/(b1 - sd)) if b1 - sd > 0 else np.nan)

def report(tag, o):
    if not o:
        print(f'{tag}: too few points'); return
    fT1 = f"{o['fT1']:6.2f}" if np.isfinite(o['fT1']) else '   n/a'
    print(f"{tag:26s} n={o['n']:2d} | free slope={o['slope']:6.3f} "
          f"b={o['icept']:+7.4f} f_T={o['fT']:6.2f} GHz AdjR2={o['adj']:.4f}"
          f" | slope=1: b={o['b1']:+7.4f}+-{o['b1_se']:.4f} f_T={fT1} GHz")

print()
res = {}
for col, name in (('f_RC', '2-L ladder (fitted L)'), ('f_RC0', 'ladder, L removed')):
    for V, sub in (('-7 V', df[df.V == -7]), ('-5 V', df[df.V == -5]), ('all', df)):
        o = fit_ft(sub, col)
        report(f'{name} / {V}', o)
        res[(col, V)] = o
    print()

# ── plots ──────────────────────────────────────────────────────────────────
MK = {25: 'o', 30: 's', 40: '^'}
CL = {-7: '#c0392b', -5: '#2471a3'}

def panel(ax, col, title):
    for V in (-7, -5):
        sub = df[df.V == V]
        for D in (25, 30, 40):
            s = sub[sub.D == D]
            if not len(s):
                continue
            ax.scatter(1000/s[col]**2, 1000/s['f3']**2, s=62, marker=MK[D],
                       facecolor='none', edgecolor=CL[V], linewidth=1.6,
                       label=f'{D} $\\mu$m, {V} V', zorder=4)
    o = res[(col, 'all')]
    xr = np.linspace(0, o['x'].max()*1.12, 50)
    ax.plot(xr, o['slope']*xr + o['icept'], '-', color='k', lw=1.5, zorder=3,
            label='free-slope fit')
    ax.plot(xr, xr + o['b1'], '--', color='0.45', lw=1.5, zorder=3,
            label='slope locked to 1')
    fT1 = f"{o['fT1']:.1f} GHz" if np.isfinite(o['fT1']) else 'not resolvable'
    ax.annotate(f"free slope = {o['slope']:.3f},  Adj. $R^2$ = {o['adj']:.3f}\n"
                f"locked-slope intercept = {o['b1']:+.3f}\n"
                f"$f_T$ = {fT1}",
                xy=(0.035, 0.965), xycoords='axes fraction', va='top',
                ha='left', fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
    ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(alpha=.3, ls=':')
    ax.legend(fontsize=7.5, loc='lower right', frameon=True)

fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.0))
panel(axs[0], 'f_RC',  r'(a) $f_{RC}$ from the fitted 2-L ladder')
panel(axs[1], 'f_RC0', r'(b) $f_{RC}$ with the series inductances removed')
fig.tight_layout()
fig.savefig('ft_extraction_multiD.png', dpi=300)

# single-panel version of the baseline (2-L ladder) for direct use
fig2, ax2 = plt.subplots(figsize=(6.2, 5.2))
panel(ax2, 'f_RC', r'$f_T$ extraction — 2-L ladder baseline, $D$ = 25/30/40 $\mu$m')
fig2.tight_layout()
fig2.savefig('ft_extraction_2Lladder.png', dpi=300)

# Origin-ready export
with open('ft_extraction_points.txt', 'w') as fo:
    fo.write('D_um\tBias_V\tR_m_ohm\tC_PD_fF\tf_RC_2L_GHz\tf_RC_noL_GHz\t'
             'f_3dB_GHz\tx_2L\tx_noL\ty\n')
    for _, r in df.iterrows():
        fo.write(f"{r.D}\t{r.V}\t{r.Rm:.1f}\t{r.Cpd:.1f}\t{r.f_RC:.3f}\t"
                 f"{r.f_RC0:.3f}\t{r.f3:.3f}\t{1000/r.f_RC**2:.4f}\t"
                 f"{1000/r.f_RC0**2:.4f}\t{1000/r.f3**2:.4f}\n")

print('wrote ft_extraction_multiD.png, ft_extraction_2Lladder.png,'
      ' ft_extraction_multiD.csv, ft_extraction_points.txt')
