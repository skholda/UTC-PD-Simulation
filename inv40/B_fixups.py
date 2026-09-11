"""Dimension B, part 3 - corrections and one carry-through.  All exploratory.

 (a') model-free open-device C_tot / L_tot, properly scaled regression
      (w*Im Z)/1e12 = a0 + a1*(w/1e10)^2  ->  C_tot = -1e-12/a0,  L_tot = a1*1e-8
 (b') resistor-loaded devices: apparent C at 1-2 GHz with the correct low-f
      correction  C_app ~ C_tot - (L_m + L_2)/R_m^2
 (f)  carry the EXPLORATORY port-side-delay ladder through the f_T extraction
      for every device in ft_userbw.py's PAIR (f_3dB unchanged, from ft_userbw.csv)
Outputs: inv40/B_open_CL.csv (overwritten), B_Capp_resistor.csv (overwritten),
         B_delay_ft.csv, B_delay_ft.png, B_resid_delay_40um_m7V.png
"""
import os, glob, numpy as np, pandas as pd
os.chdir('/home/user/UTC-PD-Simulation')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)
OUT = 'inv40'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
M = f'{S11D}/30um/main_figure_03_30_2026'
base = pd.read_csv(f'{OUT}/B_fits.csv')
rms_c = lambda e: float(np.sqrt(np.mean(np.abs(e)**2)))

def load(path, D, V, lab, camp):
    f, S = read_s11(path)
    Z = R_L*(1 + S)/(1 - S)
    op = lab == 'WO'
    return dict(D=D, V=V, lab=lab, camp=camp, f=f, w=2*np.pi*f, S=S, Z=Z, open=op,
                Rm_meas=Z[0].real, Rm=np.inf if op else Z[0].real, Cpd=C_PD_CV[(D, V)]*1e-15)

DEVS = []
for folder, lab in [('40ohm_V1', '38'), ('80ohm_V2', '80'), ('100ohm_V1', '100'),
                    ('140ohm_V1', '140'), ('WO_V2', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(load(glob.glob(f'{S11D}/40um/{folder}/S11_{V}V*.s1p')[0], 40, V, lab, '40'))
for folder, lab in [('33ohm', '38'), ('55ohm', '60'), ('200 ohm-1', '200'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(load(glob.glob(f'{M}/{folder}/Bias_{V}V_*.s1p')[0], 30, V, lab, 'Mar'))
for folder, lab in [('32 ohm', '38'), ('71 ohm', '76'), ('120 ohm', '120'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(load(f'{S11D}/30um/{folder}/{V}V.s1p', 30, V, lab, 'Jan'))
def base_row(d):
    r = base[(base.D == d['D']) & (base.V == d['V']) & (base.lab.astype(str) == d['lab']) & (base.camp == d['camp'])]
    return r.iloc[0]
def Zin_gen(w, Cpd, Rm, L1, Lm, L2, Zser):
    Z1 = Zser + 1/(1j*w*Cpd); Y1 = 1j*w*C_CPW_FIX + 1/Z1; Z2 = 1j*w*L1 + 1/Y1
    Y2 = (1/(Rm + 1j*w*Lm) if np.isfinite(Rm) else 0.0) + 1/Z2
    return 1j*w*L2 + 1/Y2
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

# ── (a') ───────────────────────────────────────────────────────────────────
def CL_fit(w, Zi, fmin=0.25e9, fmax=8e9):
    sel = (w >= 2*np.pi*fmin) & (w <= 2*np.pi*fmax)
    x = (w[sel]/1e10)**2; y = w[sel]*Zi[sel]/1e12
    A = np.vstack([np.ones_like(x), x]).T
    (a0, a1), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [a0, a1]
    return -1e-12/a0, a1*1e-8, float(np.sqrt(np.mean(res**2)))*1e12/np.mean(w[sel])
rows = []
for d in DEVS:
    if not d['open']:
        continue
    C, L, rr = CL_fit(d['w'], d['Z'].imag)
    br = base_row(d)
    Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
    Cm, Lm_, _ = CL_fit(d['w'], Zm.imag)
    Zm0 = Zin_gen(d['w'], d['Cpd'], d['Rm'], 0, 0, 0, Rs_FIX)
    Cm0, Lm0, _ = CL_fit(d['w'], Zm0.imag)
    rows.append(dict(D=d['D'], V=d['V'], camp=d['camp'], C_tot_fF=C*1e15,
                     C_CV_plus_CPW=(d['Cpd'] + C_CPW_FIX)*1e15,
                     excess_fF=(C - d['Cpd'] - C_CPW_FIX)*1e15,
                     excess_pct_of_CPD=(C - d['Cpd'] - C_CPW_FIX)/d['Cpd']*100,
                     L_tot_pH=L*1e12, ladder_L1pL2_pH=br.ms_L1 + br.ms_L2,
                     ladder_sameFit_C_fF=Cm*1e15, ladder_sameFit_L_pH=Lm_*1e12,
                     noL_ladder_C_fF=Cm0*1e15, noL_ladder_L_pH=Lm0*1e12,
                     resid_ImZ_ohm=rr))
TA = pd.DataFrame(rows); TA.to_csv(f'{OUT}/B_open_CL.csv', index=False)
print("=== (a') open devices, model-free: w*ImZ = -1/C_tot + w^2 L_tot, 0.25-8 GHz ===")
print("  ladder_sameFit_*: the same regression applied to the baseline ladder with its fitted L's")
print("  noL_ladder_*: same regression on the ladder with L1=Lm=L2=0 (checks the R_s/C_CPW bias of the method)")
print(TA.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ── (b') ───────────────────────────────────────────────────────────────────
rows = []
for d in DEVS:
    if d['open']:
        continue
    br = base_row(d); Y = 1/d['Z']
    Capp = np.mean([np.interp(f0, d['f'], Y.imag)/(2*np.pi*f0) for f0 in (1e9, 2e9)])
    Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
    Ym = 1/Zm
    Cmod = np.mean([np.interp(f0, d['f'], Ym.imag)/(2*np.pi*f0) for f0 in (1e9, 2e9)])
    corr = (br.ms_Lm + br.ms_L2)*1e-12/d['Rm_meas']**2
    rows.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'], Rm=d['Rm_meas'],
                     Capp_meas_fF=Capp*1e15, Capp_ladder_fF=Cmod*1e15,
                     C_CV_plus_CPW=(d['Cpd'] + C_CPW_FIX)*1e15,
                     LmL2_over_Rm2_fF=corr*1e15,
                     Capp_meas_corr_fF=(Capp + corr)*1e15,
                     Capp_ladder_corr_fF=(Cmod + corr)*1e15,
                     meas_minus_ladder_fF=(Capp - Cmod)*1e15))
TB = pd.DataFrame(rows); TB.to_csv(f'{OUT}/B_Capp_resistor.csv', index=False)
print("\n=== (b') resistor-loaded devices: Im(Y)/w at 1-2 GHz; low-f expansion C_app ~ C_tot - (Lm+L2)/Rm^2 ===")
print("  reliable only where LmL2_over_Rm2 is small (high R_m); meas_minus_ladder is model-independent-ish")
print(TB.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ── (f) delay-ladder carried through the f_T extraction ─────────────────────
L_GRID = [(L1, Lm, L2) for L1 in (0, 60, 150, 250) for Lm in (0, 70, 200) for L2 in (60, 130, 200)]
def fit_delay(w, S, Cpd, Rm, op, hi=600.0):
    def unpack(p):
        return ((p[0]*1e-12, 0.0, p[1]*1e-12) if op else (p[0]*1e-12, p[1]*1e-12, p[2]*1e-12)), p[-1]*1e-12
    def model(p, w_):
        L, tau = unpack(p); return S11_model(w_, Cpd, Rm, *L)*np.exp(-2j*w_*tau)
    def resid(p):
        e = model(p, w) - S; return np.concatenate([e.real, e.imag])
    n = 2 if op else 3; best = None
    for L0 in L_GRID:
        for t0 in (0.0, 1.5, 3.0):
            p0 = ([L0[0], L0[2]] if op else list(L0)) + [t0]
            r = least_squares(resid, p0, bounds=([0.0]*n + [-6.0], [hi]*n + [6.0]), x_scale=[50.0]*n + [0.5])
            e = model(r.x, w) - S; rr = rms_c(e)
            if best is None or rr < best[0]:
                best = (rr, r.x, e)
    rr, p, e = best; L, tau = unpack(p)
    return rr, tau*1e12, L, e

ub = pd.read_csv('ft_userbw.csv'); ub['lab'] = ub['lab'].astype(str)
rows = []; keep = {}
for D, V, lab, camp, s11, sheet, note in PAIR:
    f, S = read_s11(os.path.join(S11D, s11)); w = 2*np.pi*f
    op = lab == 'WO'; Rm_meas = (R_L*(1 + S[0])/(1 - S[0])).real
    Rm = np.inf if op else Rm_meas; Cpd = C_PD_CV[(D, V)]*1e-15
    rr, tau, L, e = fit_delay(w, S, Cpd, Rm, op)
    fRC = f3dB_of(fg, H_ckt(wg, Cpd, Rm, *L))/1e9
    u = ub[(ub.D == D) & (ub.V == V) & (ub.lab == lab) & (ub.camp == camp)].iloc[0]
    x = 1000/fRC**2; y = u.y
    rows.append(dict(D=D, V=V, lab=lab, camp=camp, Rm=Rm_meas, rms_base=u.rms, f_RC_base=u.f_RC,
                     rms_delay=rr, tau_ps=tau, L1=L[0]*1e12, Lm=L[1]*1e12, L2=L[2]*1e12,
                     f_RC_delay=fRC, f3=u.f3, ok=u.ok, x_base=u.x, x_delay=x, y=y,
                     fT_dev_base=u.fT_dev,
                     fT_dev_delay=np.sqrt(1000/(y - x)) if y > x else np.nan))
    if D == 40 and V == -7:
        keep[lab] = (f, e, u)
TF = pd.DataFrame(rows); TF.to_csv(f'{OUT}/B_delay_ft.csv', index=False)
print('\n=== (f) EXPLORATORY: port-side-delay ladder, every PAIR device, f_3dB unchanged ===')
print(TF[['D', 'V', 'lab', 'camp', 'Rm', 'rms_base', 'rms_delay', 'tau_ps', 'L1', 'Lm', 'L2',
          'f_RC_base', 'f_RC_delay', 'f3', 'ok', 'fT_dev_base', 'fT_dev_delay']]
      .to_string(index=False, float_format=lambda v: f'{v:.2f}'))

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    b1 = np.mean(y - x)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan), b1, (np.sqrt(1000/b1) if b1 > 0 else np.nan)
g = TF[TF.ok]
print(f"\n{'subset':<34s}{'model':<10s}{'N':>3s}{'slope':>8s}{'icpt':>8s}{'AdjR2':>8s}{'f_T':>7s}"
      f"{'| slope=1 icpt':>16s}{'f_T':>7s}")
summ = []
for name, s in (('untruncated, all D', g), ('untruncated, 30 um', g[g.D == 30]),
                ('untruncated, 40 um', g[g.D == 40]), ('untruncated, 30+40 um -7 V', g[(g.V == -7) & (g.D != 25)])):
    for mdl, col in (('baseline', 'x_base'), ('delay', 'x_delay')):
        sl, b, adj, ft, b1, ft1 = freefit(s[col].values, s.y.values)
        print(f"{name:<34s}{mdl:<10s}{len(s):3d}{sl:8.3f}{b:+8.3f}{adj:8.3f}{ft:7.1f}{b1:+16.3f}{ft1:7.1f}")
        summ.append(dict(subset=name, model=mdl, N=len(s), slope=sl, icpt=b, adjR2=adj, fT=ft, icpt_slope1=b1, fT_slope1=ft1))
pd.DataFrame(summ).to_csv(f'{OUT}/B_delay_ft_summary.csv', index=False)

MK = {25: 'o', 30: 's', 40: '^'}; CV = {-3: '#4a3aa7', -5: '#2a78d6', -7: '#e34948'}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, col, ttl in ((axs[0], 'x_base', 'baseline 2-L ladder (locked)'), (axs[1], 'x_delay', 'EXPLORATORY: ladder + port delay')):
    for V in (-7, -5, -3):
        for D in (25, 30, 40):
            q = g[(g.V == V) & (g.D == D)]
            if len(q):
                ax.scatter(q[col], q.y, s=70, marker=MK[D], facecolor='none', edgecolor=CV[V], lw=1.6,
                           label=f'{D} um, {V} V', zorder=5)
    sl, b, adj, ft, b1, ft1 = freefit(g[col].values, g.y.values)
    xr = np.linspace(0, g[col].max()*1.1, 30)
    ax.plot(xr, sl*xr + b, 'k-', lw=1.4, label=f'free fit: slope {sl:.2f}, f_T {ft:.1f} GHz')
    ax.plot(xr, xr + b1, 'k--', lw=1.0, label=f'slope 1: f_T {ft1:.1f} GHz')
    ax.set_title(ttl, fontsize=10); ax.set_xlabel(r'$1000/f_{RC}^2$  (GHz$^{-2}$)')
    ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7, loc='lower right'); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
axs[0].set_ylabel(r'$1000/f_{3dB}^2$  (GHz$^{-2}$)')
fig.tight_layout(); fig.savefig(f'{OUT}/B_delay_ft.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.2))
for i, (lab, (f, e, u)) in enumerate(keep.items()):
    d = [q for q in DEVS if q['D'] == 40 and q['V'] == -7 and q['lab'] == lab][0]
    br = base_row(d)
    Sf = S11_model(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12)
    nm = f'{lab} ohm' if lab != 'WO' else 'open'
    ax.plot(f/1e9, np.abs(Sf - d['S']), '-', color=PAL[i], lw=1.5, label=f'{nm} baseline rms {rms_c(Sf - d["S"]):.3f}')
    ax.plot(f/1e9, np.abs(e), '--', color=PAL[i], lw=1.3, label=f'{nm} +delay rms {rms_c(e):.3f}')
ax.set_title('40 um, -7 V: |residual| baseline (solid) vs EXPLORATORY port-delay ladder (dashed)', fontsize=9)
ax.set_xlabel('frequency (GHz)'); ax.set_ylabel(r'$|S_{11}^{fit}-S_{11}^{meas}|$'); ax.grid(alpha=.3, ls=':')
ax.legend(fontsize=6.5, ncol=2); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(f'{OUT}/B_resid_delay_40um_m7V.png', dpi=200); plt.close(fig)
print('\nwrote inv40/B_open_CL.csv B_Capp_resistor.csv B_delay_ft.csv B_delay_ft_summary.csv B_delay_ft.png B_resid_delay_40um_m7V.png')
