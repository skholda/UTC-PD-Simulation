"""Dimension B, part 2 - follow-ups to B_s11_misfit.py.  All exploratory.

 (a) model-free open-device extraction: w*Im Z = -1/C_tot + w^2 L_tot fitted over
     0.25-8 GHz -> C_tot, L_tot per device/bias; open-device Re Z(f) vs model.
 (b) low-frequency apparent C = Im(Y)/w of the resistor-loaded devices at 1-2 GHz
     (contaminated by L_m/R_m^2, reported with that caveat).
 (c) exploratory alt-2: port-side delay tau (reference-plane shift / CPW as a
     line) in front of the baseline ladder.
 (d) exploratory alt-3: CPW L1/L2 LOCKED at the 30 um March -7 V values, free
     L_m + a mesa series inductance L0 (between C_PD and R_s).
 (e) f_RC that each 40 um device would need for f_T = 35.5 GHz, vs the spread of
     f_RC across the band-restricted / alternative fits.
Outputs: inv40/B_open_CL.csv, B_open_ReZ.png, B_Capp_resistor.csv, B_alt_delay.csv,
         B_alt_lockedCPW.csv, B_fRC_needed.csv
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

# device inventory (same as B_s11_misfit.py)
DEVS = []
for folder, lab in [('40ohm_V1', '38'), ('80ohm_V2', '80'), ('100ohm_V1', '100'),
                    ('140ohm_V1', '140'), ('WO_V2', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(dict(D=40, V=V, lab=lab, camp='40',
                         path=glob.glob(f'{S11D}/40um/{folder}/S11_{V}V*.s1p')[0]))
for folder, lab in [('33ohm', '38'), ('55ohm', '60'), ('200 ohm-1', '200'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(dict(D=30, V=V, lab=lab, camp='Mar',
                         path=glob.glob(f'{M}/{folder}/Bias_{V}V_*.s1p')[0]))
for folder, lab in [('32 ohm', '38'), ('71 ohm', '76'), ('120 ohm', '120'), ('WO', 'WO')]:
    for V in (-7, -5, -3):
        DEVS.append(dict(D=30, V=V, lab=lab, camp='Jan', path=f'{S11D}/30um/{folder}/{V}V.s1p'))
for d in DEVS:
    f, S = read_s11(d['path'])
    d['f'], d['S'], d['w'] = f, S, 2*np.pi*f
    d['Z'] = R_L*(1 + S)/(1 - S)
    d['Rm_meas'] = d['Z'][0].real
    d['open'] = d['lab'] == 'WO'
    d['Rm'] = np.inf if d['open'] else d['Rm_meas']
    d['Cpd'] = C_PD_CV[(d['D'], d['V'])]*1e-15
base = pd.read_csv(f'{OUT}/B_fits.csv')
def base_row(d):
    r = base[(base.D == d['D']) & (base.V == d['V']) & (base.lab == d['lab']) & (base.camp == d['camp'])]
    return r.iloc[0]
rms_c = lambda e: float(np.sqrt(np.mean(np.abs(e)**2)))

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
    if np.isfinite(Rm): sh(1/(Rm + 1j*w*Lm))
    ser(1j*w*L2)
    return R_L/(C*R_L + D)

# ═══════════════════════════════════════════════════════════════════════════
# (a) model-free C_tot, L_tot of the open devices
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for d in DEVS:
    if not d['open']:
        continue
    fG = d['f']/1e9; sel = (fG >= 0.25) & (fG <= 8.0)
    w = d['w'][sel]; y = w*d['Z'][sel].imag                 # = -1/C + w^2 L
    A = np.vstack([np.ones_like(w), w**2]).T
    (a0, a1), *_ = np.linalg.lstsq(A, y, rcond=None)
    Ctot, Ltot = -1/a0, a1
    resid = y - A @ [a0, a1]
    # same fit applied to the baseline ladder model (fitted L's) for reference
    br = base_row(d)
    Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
    ym = w*Zm[sel].imag
    (b0, b1), *_ = np.linalg.lstsq(A, ym, rcond=None)
    rows.append(dict(D=d['D'], V=d['V'], camp=d['camp'], C_tot_fF=Ctot*1e15,
                     C_CV_plus_CPW=(d['Cpd'] + C_CPW_FIX)*1e15,
                     excess_fF=(Ctot - d['Cpd'] - C_CPW_FIX)*1e15,
                     excess_pct_of_CPD=(Ctot - d['Cpd'] - C_CPW_FIX)/d['Cpd']*100,
                     L_tot_pH=Ltot*1e12, L1pL2_ladder_pH=br.ms_L1 + br.ms_L2,
                     C_model_fit=-1/b0*1e15, L_model_fit_pH=b1*1e12,
                     fit_rms_ohm_x_w=float(np.sqrt(np.mean(resid**2)))))
TA = pd.DataFrame(rows)
TA.to_csv(f'{OUT}/B_open_CL.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('=== (a) open devices: w*ImZ = -1/C + w^2 L over 0.25-8 GHz (model-free) ===')
print(TA.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# open-device Re Z(f) vs model
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, D in zip(axs, (40, 30)):
    k = 0
    for d in DEVS:
        if not d['open'] or d['D'] != D or d['V'] != -7:
            continue
        br = base_row(d)
        Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
        c = PAL[k]; k += 1
        sel = d['f'] >= 0.5e9
        ax.plot(d['f'][sel]/1e9, d['Z'][sel].real, 'o', ms=2.6, mfc='none', color=c,
                label=f"{d['camp']} -7 V measured")
        ax.plot(d['f'][sel]/1e9, Zm[sel].real, '-', color=c, lw=1.3, label=f"{d['camp']} ladder (R_s = 8.92)")
    ax.set_title(f'{D} um open device, -7 V: Re Z_in vs frequency (f >= 0.5 GHz)', fontsize=9.5)
    ax.set_xlabel('frequency (GHz)'); ax.set_ylabel('Re Z (ohm)'); ax.grid(alpha=.3, ls=':')
    ax.set_ylim(-5, 40); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{OUT}/B_open_ReZ.png', dpi=200); plt.close(fig)
# tabulate Re Z at a few frequencies
rows = []
for d in DEVS:
    if not d['open'] or d['V'] != -7:
        continue
    br = base_row(d)
    Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
    for f0 in (2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50):
        if f0*1e9 > d['f'][-1] + 1: continue
        rows.append(dict(D=d['D'], camp=d['camp'], f_GHz=f0,
                         ReZ_meas=np.interp(f0*1e9, d['f'], d['Z'].real),
                         ReZ_model=np.interp(f0*1e9, d['f'], Zm.real),
                         ImZ_meas=np.interp(f0*1e9, d['f'], d['Z'].imag),
                         ImZ_model=np.interp(f0*1e9, d['f'], Zm.imag)))
TR = pd.DataFrame(rows)
TR.to_csv(f'{OUT}/B_open_ReZ.csv', index=False)
print('\n--- open devices -7 V: Re/Im Z measured vs baseline ladder ---')
print(TR.pivot_table(index='f_GHz', columns=['D', 'camp'], values=['ReZ_meas', 'ReZ_model'])
        .to_string(float_format=lambda v: f'{v:.1f}'))

# ═══════════════════════════════════════════════════════════════════════════
# (b) low-frequency apparent C of the resistor-loaded devices
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for d in DEVS:
    if d['open']:
        continue
    Y = 1/d['Z']
    br = base_row(d)
    vals = {}
    for f0 in (1e9, 2e9):
        Yi = np.interp(f0, d['f'], Y.imag); w0 = 2*np.pi*f0
        vals[f0] = Yi/w0
    Capp = np.mean(list(vals.values()))
    # ladder prediction of the same quantity (with the fitted L's) and the Lm/Rm^2 term
    Zm = Zin_gen(d['w'], d['Cpd'], d['Rm'], br.ms_L1*1e-12, br.ms_Lm*1e-12, br.ms_L2*1e-12, Rs_FIX)
    Ym = 1/Zm
    Cmod = np.mean([np.interp(f0, d['f'], Ym.imag)/(2*np.pi*f0) for f0 in (1e9, 2e9)])
    rows.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'], Rm=d['Rm_meas'],
                     Capp_meas_fF=Capp*1e15, Capp_ladder_fF=Cmod*1e15,
                     C_CV_plus_CPW=(d['Cpd'] + C_CPW_FIX)*1e15,
                     Lm_over_Rm2_fF=br.ms_Lm*1e-12/d['Rm_meas']**2*1e15,
                     Capp_plus_LmRm2=(Capp + br.ms_Lm*1e-12/d['Rm_meas']**2)*1e15))
TB = pd.DataFrame(rows)
TB.to_csv(f'{OUT}/B_Capp_resistor.csv', index=False)
print('\n=== (b) resistor-loaded devices: Im(Y)/w at 1-2 GHz (Lm/Rm^2 subtracts from it) ===')
print(TB.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ═══════════════════════════════════════════════════════════════════════════
# (c) alt-2: port-side delay tau (reference-plane shift), L's free
# ═══════════════════════════════════════════════════════════════════════════
L_GRID = [(L1, Lm, L2) for L1 in (0, 60, 150, 250) for Lm in (0, 70, 200) for L2 in (60, 130, 200)]
def fit_delay(d, hi=600.0):
    w, S = d['w'], d['S']; op = d['open']
    def unpack(p):
        tau = p[-1]*1e-12
        L = (p[0]*1e-12, 0.0, p[1]*1e-12) if op else (p[0]*1e-12, p[1]*1e-12, p[2]*1e-12)
        return L, tau
    def model(p, w_):
        L, tau = unpack(p)
        return S11_model(w_, d['Cpd'], d['Rm'], *L)*np.exp(-2j*w_*tau)
    def resid(p):
        e = model(p, w) - S; return np.concatenate([e.real, e.imag])
    n = 2 if op else 3
    best = None
    for L0 in L_GRID:
        for t0 in (-1.0, 0.0, 1.0, 2.0):
            p0 = ([L0[0], L0[2]] if op else list(L0)) + [t0]
            r = least_squares(resid, p0, bounds=([0.0]*n + [-6.0], [hi]*n + [6.0]),
                              x_scale=[50.0]*n + [0.5])
            e = model(r.x, w) - S; rr = rms_c(e)
            if best is None or rr < best[0]:
                best = (rr, r.x, e)
    rr, p, e = best
    L, tau = unpack(p)
    fRC = f3dB_of(fg, H_ckt(wg, d['Cpd'], d['Rm'], *L))/1e9     # tau is a pure phase, no |H| change
    return dict(rms=rr, tau_ps=tau*1e12, L1=L[0]*1e12, Lm=L[1]*1e12, L2=L[2]*1e12, f_RC=fRC, e=e)

# ═══════════════════════════════════════════════════════════════════════════
# (d) alt-3: CPW L1/L2 locked at 30 um March -7 V values; free Lm + mesa L0
# ═══════════════════════════════════════════════════════════════════════════
ref = base[(base.D == 30) & (base.camp == 'Mar') & (base.V == -7) & base.lab.isin(['38', '60'])]
L1_LOCK, L2_LOCK = ref.ms_L1.mean(), ref.ms_L2.mean()
refo = base[(base.D == 30) & (base.camp == 'Mar') & (base.V == -7) & (base.lab == 'WO')].iloc[0]
LSUM_LOCK_OPEN = refo.ms_L1 + refo.ms_L2      # open device: only L1+L2 is identifiable
def fit_lockedCPW(d, with_L0=True, hi=600.0):
    w, S = d['w'], d['S']; op = d['open']
    keys = ([] if op else ['Lm']) + (['L0'] if with_L0 else [])
    if not keys:
        L = (LSUM_LOCK_OPEN*0.5e-12, 0.0, LSUM_LOCK_OPEN*0.5e-12)
        e = S11_model(w, d['Cpd'], d['Rm'], *L) - S
        return dict(rms=rms_c(e), Lm=0.0, L0=0.0, f_RC=f3dB_of(fg, H_ckt(wg, d['Cpd'], d['Rm'], *L))/1e9)
    def unpack(p):
        v = dict(zip(keys, p)); Lm = v.get('Lm', 0.0)*1e-12; L0 = v.get('L0', 0.0)*1e-12
        if op:
            L1, L2 = LSUM_LOCK_OPEN*0.5e-12, LSUM_LOCK_OPEN*0.5e-12
        else:
            L1, L2 = L1_LOCK*1e-12, L2_LOCK*1e-12
        return L1, Lm, L2, L0
    def model(p, w_):
        L1, Lm, L2, L0 = unpack(p)
        Z = Zin_gen(w_, d['Cpd'], d['Rm'], L1, Lm, L2, Rs_FIX + 1j*w_*L0)
        return (Z - R_L)/(Z + R_L)
    def resid(p):
        e = model(p, w) - S; return np.concatenate([e.real, e.imag])
    best = None
    for Lm0 in (0.0, 60.0, 200.0):
        for L00 in (0.0, 50.0, 150.0):
            p0 = [x for k, x in (('Lm', Lm0), ('L0', L00)) if k in keys]
            r = least_squares(resid, p0, bounds=([0.0]*len(keys), [hi]*len(keys)), x_scale=[50.0]*len(keys))
            e = model(r.x, w) - S; rr = rms_c(e)
            if best is None or rr < best[0]:
                best = (rr, r.x, e)
    rr, p, e = best
    L1, Lm, L2, L0 = unpack(p)
    fRC = f3dB_of(fg, H_gen(wg, d['Cpd'], d['Rm'], L1, Lm, L2, Rs_FIX + 1j*wg*L0))/1e9
    return dict(rms=rr, Lm=Lm*1e12, L0=L0*1e12, f_RC=fRC, e=e)

rowsC, rowsD = [], []
for d in DEVS:
    if d['V'] == -3:
        continue
    br = base_row(d)
    rc = fit_delay(d)
    rowsC.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'], rms_base=br.ms_rms,
                      f_RC_base=br.f_RC, rms_delay=rc['rms'], tau_ps=rc['tau_ps'],
                      L1=rc['L1'], Lm=rc['Lm'], L2=rc['L2'], f_RC_delay=rc['f_RC']))
    r0 = fit_lockedCPW(d, with_L0=False); r1 = fit_lockedCPW(d, with_L0=True)
    rowsD.append(dict(D=d['D'], V=d['V'], lab=d['lab'], camp=d['camp'], rms_base=br.ms_rms,
                      f_RC_base=br.f_RC, rms_lock_LmOnly=r0['rms'], Lm_lockOnly=r0['Lm'],
                      f_RC_lockOnly=r0['f_RC'], rms_lock_L0=r1['rms'], Lm_lockL0=r1['Lm'],
                      L0_pH=r1['L0'], f_RC_lockL0=r1['f_RC']))
TC = pd.DataFrame(rowsC); TD = pd.DataFrame(rowsD)
TC.to_csv(f'{OUT}/B_alt_delay.csv', index=False); TD.to_csv(f'{OUT}/B_alt_lockedCPW.csv', index=False)
print('\n=== (c) EXPLORATORY alt-2: port-side delay tau in front of the baseline ladder ===')
print(TC.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print(f'\n=== (d) EXPLORATORY alt-3: L1/L2 locked at 30um Mar -7V ({L1_LOCK:.1f}/{L2_LOCK:.1f} pH; '
      f'open: L1+L2 = {LSUM_LOCK_OPEN:.1f} pH split evenly), free Lm (+ mesa L0) ===')
print(TD.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ═══════════════════════════════════════════════════════════════════════════
# (e) f_RC needed for f_T = 35.5 GHz vs what any of the fits deliver
# ═══════════════════════════════════════════════════════════════════════════
ub = pd.read_csv('ft_userbw.csv'); ub = ub[ub.D == 40]
fm = pd.read_csv(f'{OUT}/B_fmax_refit.csv'); rp = pd.read_csv(f'{OUT}/B_RpCp.csv')
rows = []
for _, r in ub.iterrows():
    f3 = r.f3
    need = (1/f3**2 - 1/35.5**2)**-0.5
    need_326 = (1/f3**2 - 1/32.6**2)**-0.5
    q = fm[(fm.V == r.V) & (fm.lab.astype(str) == str(r.lab))]
    p = rp[(rp.D == 40) & (rp.V == r.V) & (rp.lab.astype(str) == str(r.lab))]
    rows.append(dict(V=r.V, lab=r.lab, f3dB=f3, fT_dev=r.fT_dev, f_RC_base=r.f_RC,
                     f_RC_needed_fT35p5=need, f_RC_needed_fT32p6=need_326,
                     pct_change_needed=(need/r.f_RC - 1)*100,
                     f_RC_fmax20=q[q.fmax == 20].f_RC.iloc[0], f_RC_fmax40=q[q.fmax == 40].f_RC.iloc[0],
                     f_RC_RpCp=p.f_RC_RpCp.iloc[0] if len(p) else np.nan,
                     f_RC_delay=TC[(TC.D == 40) & (TC.V == r.V) & (TC.lab.astype(str) == str(r.lab))].f_RC_delay.iloc[0],
                     f_RC_lockL0=TD[(TD.D == 40) & (TD.V == r.V) & (TD.lab.astype(str) == str(r.lab))].f_RC_lockL0.iloc[0]))
TE = pd.DataFrame(rows)
TE.to_csv(f'{OUT}/B_fRC_needed.csv', index=False)
print('\n=== (e) 40 um: f_RC that would give f_T = 35.5 / 32.6 GHz with the measured f_3dB, vs fits ===')
print(TE.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
print('\nwrote inv40/B_open_CL.csv B_open_ReZ.csv B_open_ReZ.png B_Capp_resistor.csv B_alt_delay.csv '
      'B_alt_lockedCPW.csv B_fRC_needed.csv')
