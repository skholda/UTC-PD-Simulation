"""inv40 / dimension C, part 3: three sensitivity checks.
 (a) tau_x / k refit with the previous tau set (tau_A = 3.530 ps, f_tr = 32.6 GHz)
     instead of the staircase baseline (tau_A = 1.989 ps, f_tr = 42.2 GHz).
 (b) largest single-step drop in each measured sheet (step discontinuities).
 (c) RF-wanted C_PD (H_ph kept) vs the S11-wanted C_PD from the established
     free-R_s/C_PD S11 fits (ft_40um_free_RsCpd_perdevice.csv).
Output: inv40/C_extra.log
"""
import os, io, numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open('/home/user/UTC-PD-Simulation/inv40/C_fullcurve.py').read().split("T = pd.DataFrame(rows)")[0])
LOG3 = io.StringIO()
def say3(*a, **k):
    print(*a, **k); print(*a, **k, file=LOG3)
T = pd.DataFrame(rows)

def H_ph_prev(w):
    s = lambda x: np.sinc(x/np.pi)
    tA = 3.530e-12
    return (W_A/(1+1j*w*tA) + W_C/(1+1j*w*tA)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            + W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            + W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm

say3('=== (a) pole fit with the PREVIOUS tau set (f_tr = 32.6 GHz) instead of the baseline (42.2 GHz) ===')
say3(' D  V lab camp   rms0_prev  tau_x_prev  f_x_prev  rms_pole_prev | baseline: rms0  tau_x  rms_pole')
for q in T.itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    w = 2*np.pi*c['fm']*1e9; w0 = 2*np.pi*1e6
    M0p = dB(H_ph_prev(w)) - dB(H_ph_prev(w0)) + c['Mckt']
    r0p = c['pm'] - M0p
    fp = least_squares(lambda p: c['pm'] - (M0p - 10*np.log10(1 + (w*p[0]*1e-12)**2)), [5.0], bounds=([0.0], [200.0]))
    rp = float(np.sqrt(np.mean(fp.fun**2)))
    say3(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}   {np.sqrt(np.mean(r0p**2)):.3f}      {fp.x[0]:5.2f}     '
         f'{1e3/(2*np.pi*fp.x[0]) if fp.x[0] > 0 else np.inf:6.1f}    {rp:.3f}       |          {q.rms0:.3f}  {q.tau_x_ps:5.2f}  {q.rms_pole:.3f}')

say3('\n=== (b) largest single-step drop between consecutive measured points, per sheet (Cal RF POW, dB) ===')
say3(' D  V lab camp  sheet                                   n   f_lo -> f_hi (GHz)   step(dB)  median|step|  2nd-largest drop @ f')
for q in T.itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    fm, pm = c['fm'], c['pm']
    d = np.diff(pm); o = np.argsort(d)
    i, j = o[0], o[1]
    say3(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}  {q.sheet:<40s} {len(fm):3d}  {fm[i]:5.2f} -> {fm[i+1]:5.2f}      {d[i]:+.2f}     '
         f'{np.median(np.abs(d)):.2f}         {d[j]:+.2f} @ {fm[j]:5.2f}->{fm[j+1]:5.2f}')

say3('\n=== (c) RF-wanted C_PD (H_ph kept, C_ph) vs S11-wanted C_PD (free R_s + C_PD S11 fit, established) — 40 um ===')
F = pd.read_csv('ft_40um_free_RsCpd_perdevice.csv')
say3('  V lab   C_CV   C_S11(free Rs,C)  Rs_S11   C_RF(H_ph kept)  C_RF(f_T->inf)   C_RF/C_CV  C_S11/C_CV  C_RF/C_S11')
for q in T[T.D == 40].itertuples():
    m = F[(F.V == q.V) & (F.lab.astype(str) == str(q.lab))]
    if not len(m):
        say3(f'  {q.V} {q.lab}: no row in ft_40um_free_RsCpd_perdevice.csv'); continue
    m = m.iloc[0]
    say3(f'{q.V:3d} {q.lab:>3s}  {q.Cpd_CV:6.1f}     {m.Cpd:6.1f}        {m.Rs:5.1f}      {q.C_ph:6.1f}          {q.C_inf:6.1f}'
         f'        {q.C_ph_ratio:.2f}       {m.Cpd/q.Cpd_CV:.2f}        {q.C_ph/m.Cpd:.2f}')

say3('\n=== measured f_3dB vs model f_3dB (baseline, and after the fitted pole) ===')
say3(' D  V lab camp   f3_meas  f3_model_base  ratio  f3_model_pole  f3_model_k  f3_model_g')
for q in T.itertuples():
    say3(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}   {q.f3_meas:6.2f}    {q.f3_model_base:6.2f}      {q.f3_meas/q.f3_model_base:.2f}     '
         f'{q.f3_model_pole:6.2f}       {q.f3_model_k:6.2f}      {q.f3_model_g:6.2f}')

open(f'{OUT}/C_extra.log', 'w').write(LOG3.getvalue())
say3('\nwrote inv40/C_extra.log')
