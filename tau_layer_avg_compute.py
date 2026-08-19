"""
Layer-average transit times  τ = W / v(E_avg)
=============================================
For each drift layer:
  1. E_avg = mean |E(z)| over the layer window   <- from the DEVICE E-field
             (Lumerical CHARGE: -7 V, Iph=0.5 mA, lateral 10 um)
  2. v(E_avg) from the MATERIAL v(E) curve (InGaAs / InP)
  3. τ = W / v(E_avg)

Layer windows (model geometry, depletion edge at z=0.98 um):
  dep-abs  W_D = 240 nm  ->  z = 0.74 - 0.98 um  (InGaAs)
  collector W_C = 740 nm ->  z = 0.00 - 0.74 um  (InP)

τ_A (undep-absorber diffusion) is NOT computed here; τ_h uses hole saturation.
Run:  python tau_layer_avg_compute.py
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

def load_vE(path, unit):                       # returns v(E) in m/s, E in kV/cm
    a = np.loadtxt(path, delimiter=','); o = np.argsort(a[:, 0])
    Ek, v = a[o, 0], a[o, 1]
    zu, i = np.unique(Ek, return_index=True)
    return interp1d(zu, v[i]*unit, bounds_error=False,
                    fill_value=(v[i][0]*unit, v[i][-1]*unit))

Ef = load_field(FIELD)
vInGaAs = load_vE(os.path.join(DDIR, 'velocity_InGaAs.csv'), 1e4)   # col in 1e6 cm/s
vInP    = load_vE(os.path.join(DDIR, 'velocity_InP.csv'),    1e5)   # col in 1e7 cm/s

def E_avg(z0, z1, N=20000):
    zz = np.linspace(z0, z1, N)
    return np.mean(np.abs(Ef(zz))) / 1e5       # V/m -> kV/cm

LAYERS = [
    ('dep-abs  W_D', 0.74e-6, 0.98e-6, 240e-9, vInGaAs, 'InGaAs'),
    ('collector W_C', 0.00e-6, 0.74e-6, 740e-9, vInP,    'InP'),
]

print('='*70)
print('Layer-average transit times  τ = W / v(E_avg)   (new field, 0.5mA)')
print('='*70)
print(f'{"layer":15s} | {"E_avg":>10s} | {"v(E_avg)":>13s} | {"τ = W/v":>9s}')
print('-'*70)
tau = {}
for name, z0, z1, W, vf, mat in LAYERS:
    Ea = E_avg(z0, z1); vv = float(vf(Ea)); t = W/vv
    tau[name] = t
    print(f'{name:15s} | {Ea:6.1f} kV/cm | {vv/1e5:6.2f}e5 m/s | {t*1e12:6.3f} ps')
print('-'*70)
print(f'  tau_eD = {tau["dep-abs  W_D"]*1e12:.3f} ps   tau_C = {tau["collector W_C"]*1e12:.3f} ps')
print(f'  (tau_A = 3.530 ps kept; tau_h = 5.000 ps hole saturation)')
