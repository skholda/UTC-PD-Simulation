"""
Material-resolved layer-average transit times  τ = Σ W_sub / v_sub(E_avg,sub)
============================================================================
Each sublayer uses ITS OWN material v(E) curve and its own average field:
  E_avg,sub = mean |E(z)| over the sublayer  (device field, Lumerical CHARGE
              -7 V, Iph=0.5 mA, lateral 10 um)
  v_sub     = drift velocity at E_avg,sub from that material's v(E) curve
  τ_sub     = W_sub / v_sub

Sublayers (depletion edge at z=0.98 um):
  dep InGaAs absorber : z=0.82-0.98 (160 nm)  InGaAs   -> W_Ad, τ_eD, τ_h
  grading Q1.4        : z=0.805-0.82 (15 nm)  InGaAsP  ┐
  grading Q1.1        : z=0.79-0.805 (15 nm)  InGaAsP  ├ collector-side stack
  cliff               : z=0.74-0.79  (50 nm)  InP      │  -> part of W_C, τ_C
  collector           : z=0.00-0.74 (740 nm)  InP      ┘

Geometry: W_A=480 (undep InGaAs), W_Ad=160 (dep InGaAs), W_C=820 (grading+cliff
+collector).  τ_A (undep-absorber diffusion) is NOT computed here.
"""
import os, numpy as np
from scipy.interpolate import interp1d

DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_efield')
FIELD = os.path.join(DDIR, 'Efield_7V_0p5mA_lat10um.txt')

def load_field(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('z('):
            continue
        p = line.split(',')
        if len(p) >= 2:
            try: rows.append([float(p[0]), float(p[1])])
            except ValueError: pass
    d = np.array(rows); z, E = d[:, 0], np.abs(d[:, 1])
    o = np.argsort(z); z, E = z[o], E[o]
    zu, i = np.unique(z, return_index=True)
    return interp1d(zu, E[i], fill_value='extrapolate')

def load_vE(path, unit):                       # v in m/s, E in kV/cm
    a = np.loadtxt(path, delimiter=','); o = np.argsort(a[:, 0])
    Ek, v = a[o, 0], a[o, 1]
    zu, i = np.unique(Ek, return_index=True)
    return interp1d(zu, v[i]*unit, bounds_error=False,
                    fill_value=(v[i][0]*unit, v[i][-1]*unit))

Ef = load_field(FIELD)
vInGaAs = load_vE(os.path.join(DDIR, 'velocity_InGaAs.csv'), 1e4)      # 1e6 cm/s col
vInP    = load_vE(os.path.join(DDIR, 'velocity_InP.csv'),    1e5)      # 1e7 cm/s col
vQ11    = load_vE(os.path.join(DDIR, 'velocity_InGaAsP_Q1p1.csv'), 1e4)
vQ14    = load_vE(os.path.join(DDIR, 'velocity_InGaAsP_Q1p4.csv'), 1e4)

def E_avg(z0, z1, N=8000):
    zz = np.linspace(z0, z1, N)
    return np.mean(np.abs(Ef(zz))) / 1e5       # kV/cm

# sublayers: (name, z0, z1 [um], v(E) curve, material, group)
SUB = [
    ('dep InGaAs abs', 0.820, 0.980, vInGaAs, 'InGaAs', 'W_Ad'),
    ('grading Q1.4',   0.805, 0.820, vQ14,    'Q1.4',   'W_C'),
    ('grading Q1.1',   0.790, 0.805, vQ11,    'Q1.1',   'W_C'),
    ('cliff',          0.740, 0.790, vInP,    'InP',    'W_C'),
    ('collector',      0.000, 0.740, vInP,    'InP',    'W_C'),
]

print('='*74)
print('Material-resolved layer-average transit times (new field, 0.5 mA)')
print('='*74)
print(f'{"sublayer":16s}|{"W(nm)":>6}|{"E_avg":>9}|{"v(E_avg)":>13}|{"tau(ps)":>8}| material')
print('-'*74)
tau_eD = 0.0; tau_C = 0.0
for name, z0, z1, vf, mat, grp in SUB:
    W = (z1-z0)*1e-6; Ea = E_avg(z0*1e-6, z1*1e-6); vv = float(vf(Ea)); t = W/vv
    print(f'{name:16s}|{W*1e9:6.0f}|{Ea:6.1f}kV|{vv/1e5:8.2f}e5 m/s|{t*1e12:7.3f} | {mat}')
    if grp == 'W_Ad': tau_eD += t
    else:             tau_C  += t
print('-'*74)
v_h = 0.48e7 * 0.01     # m/s  InGaAs hole saturation (literature)
tau_h = 160e-9 / v_h
print(f'  τ_eD (W_Ad=160, InGaAs)          = {tau_eD*1e12:.3f} ps')
print(f'  τ_C  (W_C=820, grad+cliff+coll)  = {tau_C*1e12:.3f} ps')
print(f'  τ_h  (W_Ad=160 / v_h,sat)        = {tau_h*1e12:.3f} ps')
print(f'  (τ_A = 3.530 ps kept; W_norm = 480+820+2*160 = 1620 nm)')
