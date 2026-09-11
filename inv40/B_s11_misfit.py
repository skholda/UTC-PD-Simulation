"""Dimension B - what exactly is wrong with the 40 um S11 fits.

Everything here is diagnostic / exploratory.  The project baseline
(R_s = 8.92 ohm, C_CPW = 46.53 fF, C_PD from C-V, 2-L ladder, L's free 0-600 pH)
is NOT changed; the refits below reproduce it and then probe it.

Outputs (all under inv40/B_*):
  B_fits.csv            baseline refit per device/bias, multistart, bound flags,
                        band-resolved residuals                       (tasks 1, 5)
  B_fmax_refit.csv      refits restricted to f <= 40 / 30 / 20 GHz      (task 2)
  B_Ctotal.csv          C_total from open-device S11 at 1/2/3/5 GHz     (task 3)
  B_RpCp.csv            exploratory R_s + (R_p || C_p) series element   (task 4)
  B_resid_*.png, B_overlay_*.png, B_Ctotal.png, B_RpCp_resid.png
"""
import os, glob, numpy as np, pandas as pd
os.chdir('/home/user/UTC-PD-Simulation')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)          # read_s11, S11_model, H_ckt, f3dB_of, Rs_FIX, C_CPW_FIX, R_L,
                   # S11D, C_PD_CV, PAIR

OUT = 'inv40'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
       '#4a3aa7', '#e34948']            # fixed categorical order
fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg

# ── device inventory ────────────────────────────────────────────────────────
M = f'{S11D}/30um/main_figure_03_30_2026'
DEVS = []
for folder, lab in [('40ohm_V1', '38'), ('80ohm_V2', '80'), ('100ohm_V1', '100'),
                    ('140ohm_V1', '140'), ('WO_V2', 'WO')]:
    for V in (-7, -5, -3):
        g = glob.glob(f'{S11D}/40um/{folder}/S11_{V}V*.s1p')
        assert len(g) == 1, g
        DEVS.append(dict(D=40, V=V, lab=lab, camp='40', path=g[0]))
for folder, lab in [('33ohm', '38'), ('55ohm', '60'), ('200 ohm-1', '200'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        g = glob.glob(f'{M}/{folder}/Bias_{V}V_*.s1p')
        assert len(g) == 1, g
        DEVS.append(dict(D=30, V=V, lab=lab, camp='Mar', path=g[0]))
for folder, lab in [('32 ohm', '38'), ('71 ohm', '76'), ('120 ohm', '120'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(dict(D=30, V=V, lab=lab, camp='Jan',
                         path=f'{S11D}/30um/{folder}/{V}V.s1p'))
for d in DEVS:
    f, S = read_s11(d['path'])
    d['f'], d['S'], d['w'] = f, S, 2*np.pi*f
    Z0 = R_L*(1 + S[0])/(1 - S[0])
    d['Rm_meas'] = Z0.real
    d['open'] = d['lab'] == 'WO'
    d['Rm'] = np.inf if d['open'] else Z0.real
    d['Cpd'] = C_PD_CV[(d['D'], d['V'])]*1e-15
    d['key'] = f"{d['D']}um {d['lab']:>3s} {d['camp']} {d['V']}V"

def rms_c(e):
    return float(np.sqrt(np.mean(np.abs(e)**2)))

# ── generic ladder with a pluggable series element ──────────────────────────
def Zin_gen(w, Cpd, Rm, L1, Lm, L2, Zser):
    Z1 = Zser + 1/(1j*w*Cpd)
    Y1 = 1j*w*C_CPW_FIX + 1/Z1
    Z2 = 1j*w*L1 + 1/Y1
    Y2 = (1/(Rm + 1j*w*Lm) if np.isfinite(Rm) else 0.0) + 1/Z2
    return 1j*w*L2 + 1/Y2

def H_gen(w, Cpd, Rm, L1, Lm, L2, Zser):
    A = np.ones_like(w, dtype=complex); B = np.zeros_like(w, dtype=complex)
    C = np.zeros_like(w, dtype=complex); D = np.ones_like(w, dtype=complex)
    def ser(Z):
        nonlocal B, D; B, D = A*Z + B, C*Z + D
    def sh(Y):
        nonlocal A, C; A, C = A + B*Y, C + D*Y
    sh(1j*w*Cpd); ser(Zser + 0j*w); sh(1j*w*C_CPW_FIX); ser(1j*w*L1)
    if np.isfinite(Rm):
        sh(1/(Rm + 1j*w*Lm))
    ser(1j*w*L2)
    return R_L/(C*R_L + D)

# ── baseline ladder fit with multistart ────────────────────────────────────
L_GRID = [(L1, Lm, L2) for L1 in (0, 60, 150, 250) for Lm in (0, 70, 200)
          for L2 in (60, 130, 200)]
P0_BASE = (60.0, 70.0, 130.0)                  # ft_userbw.py start

def fit_ladder(d, fmax_GHz=None, starts=None, hi=600.0):
    m = np.ones(len(d['f']), bool) if fmax_GHz is None else d['f'] <= fmax_GHz*1e9 + 1
    w, S = d['w'][m], d['S'][m]
    op = d['open']
    def unpack(p):
        return (p[0]*1e-12, 0.0, p[1]*1e-12) if op else tuple(x*1e-12 for x in p)
    def resid(p):
        e = S11_model(w, d['Cpd'], d['Rm'], *unpack(p)) - S
        return np.concatenate([e.real, e.imag])
    n = 2 if op else 3
    sols = []
    for p0 in (starts or [P0_BASE]):
        p0 = (p0[0], p0[2]) if op else p0
        r = least_squares(resid, p0, bounds=([0.0]*n, [hi]*n), x_scale=[50.0]*n)
        L = unpack(r.x)
        e = S11_model(w, d['Cpd'], d['Rm'], *L) - S
        sols.append((rms_c(e), L, r.x))
    sols.sort(key=lambda s: s[0])
    return sols, m

def bound_flags(L1, Lm, L2, op, hi=600.0, tol=0.05):
    fl = []
    for name, v in (('L1', L1*1e12), ('Lm', Lm*1e12), ('L2', L2*1e12)):
        if name == 'Lm' and op:
            continue
        if v <= tol: fl.append(f'{name}@0')
        elif v >= hi - tol: fl.append(f'{name}@{hi:.0f}')
    return ','.join(fl)

def distinct_minima(sols, tol_rms=1e-4, tol_L=2.0):
    """cluster local minima by (rms, L) closeness; return list of (rms, L pH)."""
    out = []
    for rms, L, _ in sols:
        Lp = np.array(L)*1e12
        if not any(abs(rms - r0) < tol_rms and np.all(abs(Lp - L0) < tol_L) for r0, L0 in out):
            out.append((rms, Lp))
    return out

BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 + 5 : baseline refit, residual anatomy, bound flags
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for d in DEVS:
    base_sols, m = fit_ladder(d)                     # ft_userbw start only
    ms_sols, _ = fit_ladder(d, starts=[P0_BASE] + L_GRID)
    rms_b, L_b, _ = base_sols[0]
    rms_m, L_m, _ = ms_sols[0]
    minima = distinct_minima(ms_sols)
    # residual anatomy of the multistart-best (= baseline when they agree)
    L = L_m
    Sf = S11_model(d['w'], d['Cpd'], d['Rm'], *L)
    e = Sf - d['S']
    fG = d['f']/1e9
    tot = np.sum(np.abs(e)**2)
    cum = np.cumsum(np.abs(e)**2)/tot
    f50 = float(np.interp(0.5, cum, fG)); f80 = float(np.interp(0.8, cum, fG))
    band = {}
    for lo, hi_ in BANDS:
        sel = (fG > lo) & (fG <= hi_)
        band[f'rms_{lo}_{hi_}'] = rms_c(e[sel]) if sel.any() else np.nan
        band[f'frac_{lo}_{hi_}'] = float(np.sum(np.abs(e[sel])**2)/tot) if sel.any() else np.nan
    dmag = np.abs(Sf) - np.abs(d['S'])
    dphi = np.rad2deg(np.angle(Sf*np.conj(d['S'])))
    okph = np.abs(d['S']) > 0.1
    fRC = f3dB_of(fg, H_ckt(wg, d['Cpd'], d['Rm'], *L))/1e9
    rows.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'],
                     Rm=d['Rm_meas'], Cpd_fF=d['Cpd']*1e15, fmax_GHz=fG[-1],
                     L1=L_b[0]*1e12, Lm=L_b[1]*1e12, L2=L_b[2]*1e12, rms=rms_b,
                     flags=bound_flags(*L_b, d['open']),
                     ms_L1=L_m[0]*1e12, ms_Lm=L_m[1]*1e12, ms_L2=L_m[2]*1e12,
                     ms_rms=rms_m, ms_flags=bound_flags(*L_m, d['open']),
                     n_minima=len(minima),
                     minima=' | '.join(f'{r:.4f}:({a:.0f},{b:.0f},{c:.0f})'
                                       for r, (a, b, c) in minima[:4]),
                     f_RC=fRC,
                     f_peak=fG[np.argmax(np.abs(e))], e_peak=float(np.abs(e).max()),
                     f50=f50, f80=f80,
                     rms_mag=float(np.sqrt(np.mean(dmag**2))),
                     rms_phase_deg=float(np.sqrt(np.mean(dphi[okph]**2))),
                     rms_le5=rms_c(e[fG <= 5]), rms_le20=rms_c(e[fG <= 20]),
                     **band))
    d['fit'] = dict(L=L, Sf=Sf, e=e, dmag=dmag, dphi=dphi, okph=okph, fRC=fRC)
T1 = pd.DataFrame(rows)
T1.to_csv(f'{OUT}/B_fits.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
FF = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
print('=== TASK 1/5: baseline refit (ft_userbw start) vs multistart best ===')
print(T1[['D', 'V', 'lab', 'camp', 'Rm', 'Cpd_fF', 'L1', 'Lm', 'L2', 'rms', 'flags',
          'ms_L1', 'ms_Lm', 'ms_L2', 'ms_rms', 'ms_flags', 'n_minima', 'f_RC']]
      .to_string(index=False, float_format=lambda v: f'{v:.2f}'))
print('\n--- local minima (rms:(L1,Lm,L2) pH), up to 4 ---')
for _, r in T1.iterrows():
    print(f"{r.D}um {r.lab:>3s} {r.camp} {r.V}V : {r.minima}")
print('\n=== TASK 1: residual anatomy (multistart-best solution) ===')
cols = ['D', 'V', 'lab', 'camp', 'ms_rms', 'rms_mag', 'rms_phase_deg', 'f_peak', 'e_peak',
        'f50', 'f80', 'rms_le5', 'rms_le20'] + [f'rms_{a}_{b}' for a, b in BANDS] \
       + [f'frac_{a}_{b}' for a, b in BANDS]
print(T1[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ── residual figures ────────────────────────────────────────────────────────
def resid_fig(sel, title, fname):
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 9.0), sharex=True)
    for i, d in enumerate(sel):
        c = PAL[i % len(PAL)]; ls = '--' if d['camp'] == 'Jan' else '-'
        fG = d['f']/1e9; ft = d['fit']
        lbl = (f"{d['lab']} ohm" if not d['open'] else 'open') + \
              (f" ({d['camp']})" if d['D'] == 30 else '') + \
              f"  rms {rms_c(ft['e']):.3f}"
        axs[0].plot(fG, np.abs(ft['e']), ls, color=c, lw=1.6, label=lbl)
        axs[1].plot(fG, ft['dmag'], ls, color=c, lw=1.6)
        ph = np.where(ft['okph'], ft['dphi'], np.nan)
        axs[2].plot(fG, ph, ls, color=c, lw=1.6)
    axs[0].set_ylabel(r'$|S_{11}^{fit}-S_{11}^{meas}|$')
    axs[1].set_ylabel(r'$|S_{11}^{fit}|-|S_{11}^{meas}|$')
    axs[2].set_ylabel(r'phase error  (deg)'); axs[2].set_xlabel('frequency (GHz)')
    axs[2].annotate('phase shown only where |S11,meas| > 0.1', xy=(0.02, 0.04),
                    xycoords='axes fraction', fontsize=7.5, color='0.35')
    for a in axs:
        a.grid(alpha=.3, ls=':'); a.axhline(0, color='0.5', lw=0.8)
    for x in (10, 20, 30, 40):
        for a in axs: a.axvline(x, color='0.85', lw=0.8, zorder=0)
    axs[0].set_title(title, fontsize=10); axs[0].legend(fontsize=7.5, ncol=2)
    axs[0].set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)

sel40_7 = [d for d in DEVS if d['D'] == 40 and d['V'] == -7]
sel40_5 = [d for d in DEVS if d['D'] == 40 and d['V'] == -5]
sel30_7 = [d for d in DEVS if d['D'] == 30 and d['V'] == -7]
resid_fig(sel40_7, '40 um, -7 V: baseline 2-L ladder residual vs frequency', 'B_resid_40um_m7V.png')
resid_fig(sel40_5, '40 um, -5 V: baseline 2-L ladder residual vs frequency', 'B_resid_40um_m5V.png')
resid_fig(sel30_7, '30 um, -7 V (control): baseline 2-L ladder residual vs frequency', 'B_resid_30um_m7V.png')

def overlay_fig(sel, title, fname):
    n = len(sel)
    fig, axs = plt.subplots(n, 2, figsize=(10.5, 2.3*n), sharex=True, squeeze=False)
    for i, d in enumerate(sel):
        fG = d['f']/1e9; Sf = d['fit']['Sf']; Sm = d['S']
        c = PAL[i % len(PAL)]
        for j, (part, nm) in enumerate(((np.real, 'Re'), (np.imag, 'Im'))):
            ax = axs[i, j]
            ax.plot(fG, part(Sm), 'o', ms=2.6, color=c, mfc='none', label='measured')
            ax.plot(fG, part(Sf), '-', color='k', lw=1.3, label='2-L ladder fit')
            ax.set_ylabel(f'{nm} $S_{{11}}$'); ax.grid(alpha=.3, ls=':')
            if j == 0:
                ax.annotate((f"{d['lab']} ohm" if not d['open'] else 'open') +
                            f"  L1/Lm/L2 = {d['fit']['L'][0]*1e12:.0f}/{d['fit']['L'][1]*1e12:.0f}/"
                            f"{d['fit']['L'][2]*1e12:.0f} pH  rms {rms_c(d['fit']['e']):.3f}",
                            xy=(0.02, 0.9), xycoords='axes fraction', fontsize=8, va='top')
        if i == 0:
            axs[0, 1].legend(fontsize=7.5, loc='lower right')
    for a in axs[-1]: a.set_xlabel('frequency (GHz)')
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}', dpi=200); plt.close(fig)
overlay_fig(sel40_7, '40 um, -7 V: measured vs fitted S11', 'B_overlay_40um_m7V.png')
overlay_fig([d for d in sel30_7 if d['camp'] == 'Mar'], '30 um, -7 V (March): measured vs fitted S11',
            'B_overlay_30um_m7V.png')

# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 : restrict the 40 um fits to f <= 40 / 30 / 20 GHz
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for d in DEVS:
    if d['D'] != 40 or d['V'] == -3:
        continue
    full = d['fit']
    for fmax in (50, 40, 30, 20):
        sols, m = fit_ladder(d, fmax_GHz=fmax, starts=[P0_BASE] + L_GRID)
        rms_in, L, _ = sols[0]
        Sf = S11_model(d['w'], d['Cpd'], d['Rm'], *L); e = Sf - d['S']
        fRC = f3dB_of(fg, H_ckt(wg, d['Cpd'], d['Rm'], *L))/1e9
        rows.append(dict(V=d['V'], lab=d['lab'], fmax=fmax, L1=L[0]*1e12, Lm=L[1]*1e12,
                         L2=L[2]*1e12, flags=bound_flags(*L, d['open']),
                         rms_inband=rms_in, rms_full50=rms_c(e),
                         rms_full_fit_inband=rms_c(full['e'][m]),   # 50-GHz fit scored in band
                         f_RC=fRC, f_RC_full=full['fRC']))
T2 = pd.DataFrame(rows)
T2.to_csv(f'{OUT}/B_fmax_refit.csv', index=False)
print('\n=== TASK 2: 40 um refits restricted to f <= fmax (multistart best) ===')
print('  rms_inband: refit scored on f<=fmax | rms_full50: same L scored on 10 MHz-50 GHz |')
print('  rms_full_fit_inband: the 50-GHz baseline L scored on f<=fmax only')
print(T2.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 : C_total from open-device S11 at 1, 2, 3, 5 GHz
# ═══════════════════════════════════════════════════════════════════════════
FPROBE = [1e9, 2e9, 3e9, 5e9]
def Zat(f, S, f0):
    Z = R_L*(1 + S)/(1 - S)
    return np.interp(f0, f, Z.real) + 1j*np.interp(f0, f, Z.imag)
rows = []
for d in DEVS:
    if not d['open']:
        continue
    L = d['fit']['L']
    for f0 in FPROBE:
        Zm = Zat(d['f'], d['S'], f0); w0 = 2*np.pi*f0
        Zmod = Zin_gen(np.array([w0]), d['Cpd'], d['Rm'], *L, Rs_FIX)[0]
        Zmod0 = Zin_gen(np.array([w0]), d['Cpd'], d['Rm'], 0.0, 0.0, 0.0, Rs_FIX)[0]
        Cm = -1/(w0*Zm.imag); Cmod = -1/(w0*Zmod.imag); Cmod0 = -1/(w0*Zmod0.imag)
        Cexp = d['Cpd'] + C_CPW_FIX
        rows.append(dict(D=d['D'], V=d['V'], camp=d['camp'], f_GHz=f0/1e9,
                         ReZ_meas=Zm.real, ReZ_model=Zmod.real, ImZ_meas=Zm.imag,
                         C_meas_fF=Cm*1e15, C_CV_plus_CPW=Cexp*1e15,
                         C_model_fittedL=Cmod*1e15, C_model_noL=Cmod0*1e15,
                         C_meas_minus_CPW=(Cm - C_CPW_FIX)*1e15, Cpd_CV=d['Cpd']*1e15,
                         excess_pct_vs_CV=(Cm - Cexp)/Cexp*100,
                         excess_pct_vs_model=(Cm - Cmod)/Cmod*100))
T3 = pd.DataFrame(rows)
# pad-only structures
PADS = [('25um pad', f'{S11D}/25um/pad/CPW_-7V.s2p', -7),
        ('25um pad', f'{S11D}/25um/pad/CPW_-5V.s2p', -5),
        ('25um pad', f'{S11D}/25um/pad/CPW_-3V.s2p', -3),
        ('30um pad', f'{M}/pad_30um/Bias_-7V_pad.s1p', -7),
        ('30um pad', f'{M}/pad_30um/Bias_-5V_pad.s1p', -5),
        ('30um pad', f'{M}/pad_30um/Bias_-3V_pad.s1p', -3)]
prow = []
for nm, p, V in PADS:
    f, S = read_s11(p)
    for f0 in FPROBE + [10e9]:
        Zm = Zat(f, S, f0); w0 = 2*np.pi*f0
        prow.append(dict(structure=nm, V=V, f_GHz=f0/1e9, ReZ=Zm.real, ImZ=Zm.imag,
                         C_fF=-1/(w0*Zm.imag)*1e15))
TP = pd.DataFrame(prow)
T3.to_csv(f'{OUT}/B_Ctotal.csv', index=False)
TP.to_csv(f'{OUT}/B_Cpad.csv', index=False)
print('\n=== TASK 3: C_total = -1/(w Im Z) of the open devices ===')
print(T3.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
print('\n--- pad-only structures ---')
print(TP.pivot_table(index=['structure', 'V'], columns='f_GHz', values='C_fF')
        .to_string(float_format=lambda v: f'{v:.2f}'))
print('\n--- summary at 1-2 GHz (mean of the two): measured C - 46.53 vs C_PD(C-V) ---')
S3 = (T3[T3.f_GHz <= 2].groupby(['D', 'camp', 'V'])
      [['C_meas_fF', 'C_CV_plus_CPW', 'C_model_fittedL', 'C_meas_minus_CPW', 'Cpd_CV',
        'excess_pct_vs_CV', 'excess_pct_vs_model', 'ReZ_meas', 'ReZ_model']].mean())
print(S3.to_string(float_format=lambda v: f'{v:.2f}'))
S3.to_csv(f'{OUT}/B_Ctotal_summary.csv')

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, D in zip(axs, (40, 30)):
    k = 0
    for d in DEVS:
        if not d['open'] or d['D'] != D:
            continue
        fG = d['f']/1e9; sel = (fG >= 0.4) & (fG <= 12)
        Z = R_L*(1 + d['S'])/(1 - d['S'])
        Capp = -1/(d['w']*Z.imag)*1e15
        Zmod = Zin_gen(d['w'], d['Cpd'], d['Rm'], *d['fit']['L'], Rs_FIX)
        Cmod = -1/(d['w']*Zmod.imag)*1e15
        c = PAL[k % len(PAL)]; k += 1
        ls = '--' if d['camp'] == 'Jan' else '-'
        ax.plot(fG[sel], Capp[sel], 'o', ms=3, color=c, mfc='none',
                label=f"{d['V']} V {d['camp'] if D == 30 else ''} meas")
        ax.plot(fG[sel], Cmod[sel], ls, color=c, lw=1.2, label=f"{d['V']} V ladder fit")
        ax.axhline((d['Cpd'] + C_CPW_FIX)*1e15, color=c, lw=0.8, ls=':')
    ax.set_title(f'{D} um open device: apparent C = -1/(w Im Z)\n'
                 'dotted = C_PD(C-V) + 46.53 fF', fontsize=9.5)
    ax.set_xlabel('frequency (GHz)'); ax.set_ylabel('apparent capacitance (fF)')
    ax.set_xscale('log'); ax.grid(alpha=.3, ls=':', which='both'); ax.legend(fontsize=6.5, ncol=2)
fig.tight_layout(); fig.savefig(f'{OUT}/B_Ctotal.png', dpi=200); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4 : EXPLORATORY  R_s + (R_p || C_p) series element
# ═══════════════════════════════════════════════════════════════════════════
def fit_RpCp(d, hi=600.0):
    w, S = d['w'], d['S']; op = d['open']
    keys = ['Rp', 'lgCp', 'L1', 'L2'] + ([] if op else ['Lm'])
    lo = dict(Rp=0.0, lgCp=-1.0, L1=0.0, L2=0.0, Lm=0.0)       # Cp in fF, log10
    hb = dict(Rp=300.0, lgCp=4.0, L1=hi, L2=hi, Lm=hi)
    xs = dict(Rp=10.0, lgCp=0.5, L1=50.0, L2=50.0, Lm=50.0)
    def unpack(p):
        v = dict(zip(keys, p)); v.setdefault('Lm', 0.0)
        return v['Rp'], 10**v['lgCp']*1e-15, v['L1']*1e-12, v['Lm']*1e-12, v['L2']*1e-12
    def model(p, w_):
        Rp, Cp, L1, Lm, L2 = unpack(p)
        Zs = Rs_FIX + Rp/(1 + 1j*w_*Rp*Cp)
        Z = Zin_gen(w_, d['Cpd'], d['Rm'], L1, Lm, L2, Zs)
        return (Z - R_L)/(Z + R_L)
    def resid(p):
        e = model(p, w) - S; return np.concatenate([e.real, e.imag])
    Lb = d['fit']['L']
    starts = []
    for Rp0 in (5.0, 20.0, 60.0, 150.0):
        for lg0 in (1.5, 2.5, 3.3):
            for Ls in ((Lb[0]*1e12, Lb[1]*1e12, Lb[2]*1e12), (60, 70, 130), (180, 0, 90)):
                v = dict(Rp=Rp0, lgCp=lg0, L1=Ls[0], Lm=Ls[1], L2=Ls[2])
                starts.append([v[k] for k in keys])
    best = None
    for p0 in starts:
        r = least_squares(resid, p0, bounds=([lo[k] for k in keys], [hb[k] for k in keys]),
                          x_scale=[xs[k] for k in keys])
        e = model(r.x, w) - S; rr = rms_c(e)
        if best is None or rr < best[0]:
            best = (rr, r.x, e)
    rr, p, e = best
    Rp, Cp, L1, Lm, L2 = unpack(p)
    Zs = lambda w_: Rs_FIX + Rp/(1 + 1j*w_*Rp*Cp)
    fRC = f3dB_of(fg, H_gen(wg, d['Cpd'], d['Rm'], L1, Lm, L2, Zs(wg)))/1e9
    return dict(Rp=Rp, Cp_fF=Cp*1e15, tau_ps=Rp*Cp*1e12, f_p_GHz=1/(2*np.pi*Rp*Cp)/1e9,
                L1=L1*1e12, Lm=Lm*1e12, L2=L2*1e12, rms=rr, f_RC=fRC, e=e,
                flags=bound_flags(L1, Lm, L2, op) +
                      (',Rp@300' if Rp >= 299.9 else '') + (',Rp@0' if Rp <= 0.05 else '') +
                      (',Cp@lo' if Cp <= 0.11e-15 else '') + (',Cp@hi' if Cp >= 9.9e-12 else ''))

rows = []
for d in DEVS:
    if d['V'] == -3:
        continue
    r = fit_RpCp(d)
    d['rpcp'] = r
    A = np.pi*(d['D']/2)**2
    rows.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'], Rm=d['Rm_meas'],
                     rms_base=rms_c(d['fit']['e']), f_RC_base=d['fit']['fRC'],
                     rms_RpCp=r['rms'], f_RC_RpCp=r['f_RC'], Rp=r['Rp'], Cp_fF=r['Cp_fF'],
                     tau_ps=r['tau_ps'], f_p_GHz=r['f_p_GHz'], Rp_x_area=r['Rp']*A,
                     Cp_per_area=r['Cp_fF']/A, L1=r['L1'], Lm=r['Lm'], L2=r['L2'],
                     flags=r['flags']))
T4 = pd.DataFrame(rows)
T4.to_csv(f'{OUT}/B_RpCp.csv', index=False)
print('\n=== TASK 4 (EXPLORATORY): R_s(8.92, locked) + R_p||C_p series element ===')
print('  Rp_x_area in ohm*um^2 (contact-like scaling => constant), Cp_per_area in fF/um^2')
print(T4.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, sel, ttl in ((axs[0], sel40_7, '40 um, -7 V'),
                     (axs[1], [d for d in sel30_7 if d['camp'] == 'Mar'], '30 um, -7 V (Mar)')):
    for i, d in enumerate(sel):
        c = PAL[i % len(PAL)]; fG = d['f']/1e9
        nm = f"{d['lab']} ohm" if not d['open'] else 'open'
        ax.plot(fG, np.abs(d['fit']['e']), '-', color=c, lw=1.5,
                label=f"{nm} baseline rms {rms_c(d['fit']['e']):.3f}")
        ax.plot(fG, np.abs(d['rpcp']['e']), '--', color=c, lw=1.3,
                label=f"{nm} +Rp||Cp rms {d['rpcp']['rms']:.3f}")
    ax.set_title(f'{ttl}: |residual| baseline (solid) vs exploratory Rp||Cp (dashed)', fontsize=9)
    ax.set_xlabel('frequency (GHz)'); ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=6.5, ncol=2)
axs[0].set_ylabel(r'$|S_{11}^{fit}-S_{11}^{meas}|$'); axs[0].set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(f'{OUT}/B_RpCp_resid.png', dpi=200); plt.close(fig)
print('\nwrote inv40/B_fits.csv B_fmax_refit.csv B_Ctotal.csv B_Cpad.csv B_Ctotal_summary.csv '
      'B_RpCp.csv and B_*.png')
