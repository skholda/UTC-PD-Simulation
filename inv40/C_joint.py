"""inv40 / dimension C, part 2: joint fits and shape discrimination.

  - f_tr of H_ph vs the common scale k (maps k -> equivalent transit bandwidth)
  - joint (shared) tau_x / k / f_g over the five 40 um -7 V devices and over all
    seven 40 um sheets; same for the 30 um controls
  - how different the fitted pole / k / gaussian curves are over the measured
    band (can the data tell the shapes apart?)
  - double identical pole 1/(1+jw tau)^2 as a steeper alternative
  - S11 consequence of the C_PD that the RF curve wants with H_ph kept (C_ph)
  - physical equivalents of tau_x
All exploratory.  Output: inv40/C_joint.log, inv40/C_joint.csv
"""
import os, io, numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open('/home/user/UTC-PD-Simulation/inv40/C_fullcurve.py').read().split("T = pd.DataFrame(rows)")[0]
     .replace("say('\\n=== (1)", "pass  # '"))   # re-use loaders, H_ph, curves, rows

LOG2 = io.StringIO()
def say2(*a, **k):
    print(*a, **k); print(*a, **k, file=LOG2)

T = pd.DataFrame(rows)
K = 20*np.log10(np.e)

say2('=== f_tr of H_ph alone vs common scale k (this is the staircase baseline, tau_A = 1.989 ps) ===')
say2('   k     f_tr(GHz)')
for k in (0.5, 0.8, 1.0, 1.2, 1.29, 1.5, 1.75, 2.0, 2.1, 2.2, 2.25, 2.4, 2.5, 3.0):
    say2(f'  {k:4.2f}   {f3dB_of(fg, H_ph(wg, k))/1e9:6.2f}')
# previous tau set (case3_prevtau: tau_A = 3.530 ps) for the 32.6 GHz number
def H_ph_prev(w):
    s = lambda x: np.sinc(x/np.pi)
    tA = 3.530e-12
    return (W_A/(1+1j*w*tA) + W_C/(1+1j*w*tA)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            + W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            + W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm
say2(f'  previous tau set (tau_A = 3.530 ps, all else equal): f_tr = {f3dB_of(fg, H_ph_prev(wg))/1e9:.2f} GHz')

# ── joint fits ──────────────────────────────────────────────────────────────
def joint(keys, label):
    cs = [curves[k] for k in keys]
    def res_pole(p):
        return np.concatenate([c['pm'] - (c['M0'] - 10*np.log10(1 + (2*np.pi*c['fm']*1e9*p[0]*1e-12)**2)) for c in cs])
    def res_k(p):
        out = []
        for c in cs:
            w = 2*np.pi*c['fm']*1e9
            out.append(c['pm'] - (dB(H_ph(w, p[0])) - dB(H_ph(2*np.pi*1e6, p[0])) + c['Mckt']))
        return np.concatenate(out)
    def res_g(p):
        return np.concatenate([c['pm'] - (c['M0'] - K*(c['fm']/p[0])**2) for c in cs])
    def res_2p(p):
        return np.concatenate([c['pm'] - (c['M0'] - 20*np.log10(1 + (2*np.pi*c['fm']*1e9*p[0]*1e-12)**2)) for c in cs])
    def res_pole_c(p):    # per-device DC offsets + shared tau
        out = []
        for j, c in enumerate(cs):
            out.append(c['pm'] - (c['M0'] - 10*np.log10(1 + (2*np.pi*c['fm']*1e9*p[0]*1e-12)**2) + p[1+j]))
        return np.concatenate(out)
    n = sum(len(c['pm']) for c in cs)
    rms = lambda r: float(np.sqrt(np.mean(r**2)))
    base = rms(np.concatenate([c['r0'] for c in cs]))
    fp = least_squares(res_pole, [6.0], bounds=([0.0], [200.0]))
    fk = least_squares(res_k, [1.5], bounds=([0.0], [20.0]))
    fgs = least_squares(res_g, [40.0], bounds=([5.0], [5000.0]))
    f2 = least_squares(res_2p, [3.0], bounds=([0.0], [200.0]))
    fpc = least_squares(res_pole_c, [6.0] + [0.0]*len(cs), bounds=([0.0] + [-3.0]*len(cs), [200.0] + [3.0]*len(cs)))
    # per-device rms sum-of-squares with individual params, for comparison
    ind_pole = rms(np.concatenate([c['r_pole'] for c in cs]))
    ind_k = rms(np.concatenate([c['r_k'] for c in cs]))
    ind_g = rms(np.concatenate([c['r_g'] for c in cs]))
    say2(f'\n--- joint fit: {label}  (N_dev = {len(cs)}, N_pts = {n}) ---')
    say2(f'  baseline rms                 {base:.3f} dB')
    say2(f'  shared pole   tau_x = {fp.x[0]:6.2f} ps  (f_x = {1e3/(2*np.pi*fp.x[0]) if fp.x[0] > 0 else np.inf:6.2f} GHz)'
         f'   rms {rms(fp.fun):.3f}   (per-device tau: {ind_pole:.3f})')
    say2(f'  shared pole + per-dev offset tau_x = {fpc.x[0]:6.2f} ps   rms {rms(fpc.fun):.3f}   offsets '
         + ' '.join(f'{v:+.2f}' for v in fpc.x[1:]))
    say2(f'  shared k      k     = {fk.x[0]:6.3f}  (f_tr(k) = {f3dB_of(fg, H_ph(wg, fk.x[0]))/1e9:6.2f} GHz)'
         f'   rms {rms(fk.fun):.3f}   (per-device k:   {ind_k:.3f})')
    say2(f'  shared gauss  f_g   = {fgs.x[0]:6.2f} GHz                     rms {rms(fgs.fun):.3f}   (per-device f_g: {ind_g:.3f})')
    say2(f'  shared double pole (1+jw tau)^-2  tau = {f2.x[0]:6.2f} ps  rms {rms(f2.fun):.3f}')
    return dict(label=label, base=base, tau=fp.x[0], rms_pole=rms(fp.fun), tau_c=fpc.x[0], rms_pole_c=rms(fpc.fun),
                k=fk.x[0], ftr_k=f3dB_of(fg, H_ph(wg, fk.x[0]))/1e9, rms_k=rms(fk.fun),
                f_g=fgs.x[0], rms_g=rms(fgs.fun), tau2=f2.x[0], rms_2p=rms(f2.fun))

keys40_7 = [k for k in curves if k[0] == 40 and k[1] == -7]
keys40 = [k for k in curves if k[0] == 40]
keys40_V1 = [k for k in curves if k[0] == 40 and k[3] == '40V1']
keys40_V2 = [k for k in curves if k[0] == 40 and k[3] == '40V2']
keys30_7 = [k for k in curves if k[0] == 30 and k[1] == -7]
keys30 = [k for k in curves if k[0] == 30]
J = [joint(keys40_7, '40 um, -7 V (5 devices)'), joint(keys40, '40 um, all 7 sheets'),
     joint(keys40_V1, '40 um V1 only (38/100/140/WO -7 V)'), joint(keys40_V2, '40 um V2 only (80 -7/-5, WO -5)'),
     joint(keys30_7, '30 um, -7 V (8 devices)'), joint(keys30, '30 um, -7 and -5 V (12)')]
pd.DataFrame(J).to_csv(f'{OUT}/C_joint.csv', index=False)

# ── per-device double pole and shape discrimination ────────────────────────
say2('\n=== per device: double identical pole (1+jw tau2)^-2 ; max |curve difference| between the fitted '
     'pole / gauss / k models over the measured band ===')
say2(' D  V lab camp  tau2_ps  rms_2p | rms_pole rms_k rms_g | max|pole-gauss| max|pole-k| max|k-gauss| (dB, over band)  excess@10/15/20GHz (meas-model)')
rows2 = []
for q in T.itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    w = 2*np.pi*c['fm']*1e9
    f2 = least_squares(lambda p: c['pm'] - (c['M0'] - 20*np.log10(1 + (w*p[0]*1e-12)**2)), [3.0], bounds=([0.0], [200.0]))
    r2 = float(np.sqrt(np.mean(f2.fun**2)))
    dpg = float(np.max(np.abs(c['Mpole'] - c['Mg']))); dpk = float(np.max(np.abs(c['Mpole'] - c['Mk'])))
    dkg = float(np.max(np.abs(c['Mk'] - c['Mg'])))
    ex = [float(np.interp(fq, c['fm'], c['r0'])) if fq <= c['fm'][-1] else np.nan for fq in (10, 15, 20)]
    say2(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}  {f2.x[0]:6.2f}  {r2:.3f} |  {q.rms_pole:.3f}  {q.rms_k:.3f} {q.rms_g:.3f} |'
         f'   {dpg:.2f}          {dpk:.2f}        {dkg:.2f}          {ex[0]:+.2f} / {ex[1]:+.2f} / {ex[2]:+.2f}')
    rows2.append(dict(D=q.D, V=q.V, lab=q.lab, camp=q.camp, tau2_ps=f2.x[0], rms_2p=r2, dpg=dpg, dpk=dpk, dkg=dkg))
pd.DataFrame(rows2).to_csv(f'{OUT}/C_shape.csv', index=False)

# ── the pole shape itself: expected excess of a 7 ps pole and a 35 GHz gaussian ─
say2('\n=== reference shapes: excess attenuation (dB) at 10 / 15 / 20 GHz ===')
for tau in (5.0, 6.0, 7.0, 8.0):
    say2(f'  single pole tau = {tau:.0f} ps: ' + ' / '.join(f'{-10*np.log10(1 + (2*np.pi*fq*1e9*tau*1e-12)**2):+.2f}' for fq in (10, 15, 20)))
for fgv in (31.0, 35.0, 40.0):
    say2(f'  gaussian f_g = {fgv:.0f} GHz:    ' + ' / '.join(f'{-K*(fq/fgv)**2:+.2f}' for fq in (10, 15, 20)))
for k in (2.0, 2.2, 2.5):
    say2(f'  H_ph(k={k}) / H_ph(1):     ' + ' / '.join(f'{dB(H_ph(2*np.pi*fq*1e9, k)) - dB(H_ph(2*np.pi*fq*1e9)):+.2f}' for fq in (10, 15, 20)))

# ── S11 consequence of C_ph (RF-wanted C_PD with H_ph kept) ───────────────
say2('\n=== S11 consequence of the RF-wanted C_PD with H_ph kept (C_ph): rms with L fixed / L refitted (exploratory) ===')
say2(' D  V lab camp   C_CV   C_ph  ratio | s11 rms: base  C_ph,Lfix  C_ph,Lrefit | f_RC base -> with C_ph (L refit)')
for q, r in zip(T.itertuples(), sel.itertuples()):
    key = (q.D, q.V, str(q.lab), str(q.camp)); s11rel, sheet = lookup[key]
    fs, Sm = read_s11(os.path.join(S11D, s11rel)); ws = 2*np.pi*fs
    Rm = np.inf if r.open else r.Rm
    L1, Lm, L2 = r.L1*1e-12, r.Lm*1e-12, r.L2*1e-12
    def s11rms(C, a, b, cc):
        return float(np.sqrt(np.mean(np.abs(S11_model(ws, C, Rm, a, b, cc) - Sm)**2)))
    keys = ['L1', 'L2'] if r.open else ['L1', 'Lm', 'L2']
    def unpack(p):
        v = {'L1': 0.0, 'Lm': 0.0, 'L2': 0.0}
        for kk, val in zip(keys, p): v[kk] = val*1e-12
        return v['L1'], v['Lm'], v['L2']
    C = q.C_ph*1e-15
    def resid(p):
        e = S11_model(ws, C, Rm, *unpack(p)) - Sm
        return np.concatenate([e.real, e.imag])
    rr = least_squares(resid, [60.0, 70.0, 130.0][:len(keys)] if not r.open else [60.0, 130.0],
                       bounds=([0.0]*len(keys), [600.0]*len(keys)), x_scale=[50.0]*len(keys))
    Ls = unpack(rr.x)
    say2(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}  {q.Cpd_CV:6.1f} {q.C_ph:6.1f} {q.C_ph_ratio:5.2f} |'
         f'          {q.s11rms_base:.3f}    {s11rms(C, L1, Lm, L2):.3f}      {s11rms(C, *Ls):.3f}     |'
         f'  {q.f_RC:6.2f} -> {f3dB_of(fg, H_ckt(wg, C, Rm, *Ls))/1e9:6.2f}')

# ── physical equivalents of tau_x ─────────────────────────────────────────
say2('\n=== what a tau_x of the fitted size would be, if it were an RC or a transit term ===')
for tau in (6.5, 7.0):
    say2(f'  tau_x = {tau} ps: f_x = {1e3/(2*np.pi*tau):.1f} GHz;  C for R = 50 ohm: {tau*1e-12/50*1e15:.0f} fF;  '
         f'C for R = 58.92 ohm (R_L+R_s): {tau*1e-12/58.92*1e15:.0f} fF;  C for R = 8.92 ohm: {tau*1e-12/8.92*1e15:.0f} fF;'
         f'  quadrature f_T with f_tr = 42.2 GHz: {(1/42.2**2 + (2*np.pi*tau*1e-3)**2)**-0.5:.1f} GHz;'
         f'  with f_tr = 32.6: {(1/32.6**2 + (2*np.pi*tau*1e-3)**2)**-0.5:.1f} GHz')

open(f'{OUT}/C_joint.log', 'w').write(LOG2.getvalue())
say2('\nwrote inv40/C_joint.log, inv40/C_joint.csv, inv40/C_shape.csv')
