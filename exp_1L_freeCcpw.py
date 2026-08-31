"""
Experiment: 1-L ladder with C_CPW as a FREE parameter (38 / 60 ohm only)
========================================================================
Topology: Iph || C_PD -[Rs]- { C_CPW || (R_m + L_m) } -[L_CPW]- port
Fit per device: (L_CPW, C_CPW).   Locked: C_PD = 131 fF (C-V), Rs, L_m (FEM).
Global search = coarse 2-D grid scan + multi-start Nelder-Mead.

Result (-7 V):
   38 ohm : L_CPW = 159.96 pH, C_CPW = 92.38 fF -> RMS_S11 = 0.05984
   60 ohm : L_CPW = 166.43 pH, C_CPW = 73.48 fF -> RMS_S11 = 0.06639
Baseline is NOT modified by this script.
"""
import re, numpy as np
from scipy.optimize import minimize

Rs = 8.92; Cpd = 131e-15                      # locked

def s11_1L(w, Ccpw, Rp, L, Lm):
    Z1 = Rs + 1/(1j*w*Cpd)
    Yrm = 0.0 if np.isinf(Rp) else 1/(Rp + 1j*w*Lm)
    Zin = 1j*w*L + 1/(1j*w*Ccpw + Yrm + 1/Z1)
    return (Zin - 50)/(Zin + 50)

_main = open('main.py').read(); _ns = {'np': np}
for v in ['_s1p_33', '_s1p_55']:
    m = re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)', _main, re.M | re.S)
    exec(m.group(0), _ns)

DEV = [('38ohm', 38.0, '_s1p_33', 65.6e-12, 158.5),
       ('60ohm', 60.0, '_s1p_55', 71.8e-12, 172.5)]

for lbl, Rp, key, Lm, L_old in DEV:
    a = _ns[key]; ws = 2*np.pi*a[:, 0]
    S11m = 10**(a[:, 1]/20)*np.exp(1j*np.deg2rad(a[:, 2]))
    cost = lambda L, C: np.sqrt(np.mean(
        np.abs(s11_1L(ws, C*1e-15, Rp, L*1e-12, Lm) - S11m)**2))
    best = (1e9, None, None)
    for C in np.arange(5, 220, 2.0):
        vals = np.array([cost(L, C) for L in np.arange(60, 320, 2.0)])
        i = int(np.argmin(vals))
        if vals[i] < best[0]:
            best = (vals[i], np.arange(60, 320, 2.0)[i], C)
    ref = None
    for g in [(best[1], best[2]), (L_old, 46.53), (150, 30), (200, 80), (120, 120)]:
        r = minimize(lambda p: cost(*p) if (p[0] > 0 and p[1] > 0) else 1e3, g,
                     method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-10, maxiter=8000))
        if ref is None or r.fun < ref.fun:
            ref = r
    print(f'{lbl}: L_CPW={ref.x[0]:7.2f} pH  C_CPW={ref.x[1]:6.2f} fF  '
          f'RMS={ref.fun:.5f}   (fixed-C reference RMS={cost(L_old, 46.53):.5f})')
