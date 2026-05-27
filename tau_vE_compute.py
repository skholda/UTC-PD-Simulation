"""
Compute drift-layer transit times from simulated E-field + v(E)
================================================================
Uses Efield_dark.csv (Lumerical) and v(E) curves for InGaAs, InP, Q1.1.

Layer assignment in z-coordinate (deduced from cliff peak position):
   E-field peak (cliff, 50 nm wide) is at z ≈ 0.80 μm in the new data.
   Old main.py convention had cliff at z = -0.10 to -0.05 μm.
   The +0.88 μm offset is applied to all layer boundaries.

   New boundaries (z in μm):
     Depleted absorber : 0.59 to 0.75   (160 nm)  InGaAs
     Grading           : 0.75 to 0.78   ( 30 nm)  Q1.1/Q1.4 (use Q11)
     Cliff             : 0.78 to 0.83   ( 50 nm)  InP
     Collector         : 0.83 to 1.57   (740 nm)  InP

Integrate τ = ∫ dz / v(E(z))  for each layer using the v(E) curves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_efield')

# ── Load E-field ────────────────────────────────────────────────────
_ef = np.loadtxt(os.path.join(DDIR, 'Efield_dark.csv'), delimiter=',', skiprows=1)
_z, _E = _ef[:, 0], np.abs(_ef[:, 1])
# Sort & dedupe
order = np.argsort(_z); _z, _E = _z[order], _E[order]
_z_uniq, idx = np.unique(_z, return_index=True); _E = _E[idx]; _z = _z_uniq
Ef_func = interp1d(_z, _E, kind='linear', fill_value='extrapolate')

# ── Load v(E) curves ────────────────────────────────────────────────
def _load_vE(path):
    d = np.loadtxt(path, delimiter=',')
    order = np.argsort(d[:, 0])
    return d[order, 0], d[order, 1]

E_InP,    v_InP    = _load_vE(os.path.join(DDIR, 'velocity_InP.csv'))
E_InGaAs, v_InGaAs = _load_vE(os.path.join(DDIR, 'velocity_InGaAs.csv'))
E_Q11,    v_Q11    = _load_vE(os.path.join(DDIR, 'velocity_InGaAsP_Q1p1.csv'))

# File units (matching original main.py interpretation):
#   E: kV/cm
#   v_InP:    1e7 cm/s  → multiply by 1e5 to get m/s
#   v_InGaAs: 1e6 cm/s  → multiply by 1e4 to get m/s
#   v_Q11:    1e6 cm/s  → multiply by 1e4 to get m/s
vf_InP    = interp1d(E_InP,    v_InP    * 1e5, bounds_error=False,
                     fill_value=(v_InP[0]    * 1e5, v_InP[-1]    * 1e5))
vf_InGaAs = interp1d(E_InGaAs, v_InGaAs * 1e4, bounds_error=False,
                     fill_value=(v_InGaAs[0] * 1e4, v_InGaAs[-1] * 1e4))
vf_Q11    = interp1d(E_Q11,    v_Q11    * 1e4, bounds_error=False,
                     fill_value=(v_Q11[0]    * 1e4, v_Q11[-1]    * 1e4))

# ── Layer boundaries (μm) ───────────────────────────────────────────
# z direction: n+ (substrate side, z=0) → p+ (top contact, z=+1.7)
# Cliff peak at z ≈ 0.80 μm in data; cliff is 50 nm wide between
# grading (toward absorber, larger z) and collector (toward n+, smaller z).
#   Undep absorber : z = 1.00 to 1.48 μm  (480 nm)  — diffusion, not used here
#   Depleted absorber : z = 0.84 to 1.00   (160 nm)  InGaAs
#   Grading        : z = 0.81 to 0.84      ( 30 nm)  Q1.1/Q1.4
#   Cliff          : z = 0.76 to 0.81      ( 50 nm)  InP (peak E)
#   Collector      : z = 0.02 to 0.76      (740 nm)  InP (decreasing E)
LAYERS = [
    ('Depleted absorber',  0.84e-6, 1.00e-6, vf_InGaAs, 'InGaAs', 2.1e5),
    ('Grading Q1.4+Q1.1',  0.81e-6, 0.84e-6, vf_Q11,    'Q1.1',   2.1e5),
    ('Cliff (InP)',        0.76e-6, 0.81e-6, vf_InP,    'InP',    4.0e5),
    ('Collector (InP)',    0.02e-6, 0.76e-6, vf_InP,    'InP',    4.0e5),
]

def integrate_tau(z0, z1, vfunc, N=10000):
    zz = np.linspace(z0, z1, N)
    dz = zz[1] - zz[0]
    Ek = np.abs(Ef_func(zz)) / 1e5            # V/m → kV/cm
    vv = np.maximum(vfunc(Ek), 1e3)           # floor at 1 km/s
    return float(np.sum(dz / vv)), zz, Ek, vv

# ── Compute and print ───────────────────────────────────────────────
print('='*98)
print(f'{"Layer":>22} | {"z range (μm)":>15} | {"W (nm)":>7} | '
      f'{"τ_vE (ps)":>10} | {"τ_vos (ps)":>11} | {"Ratio":>6}')
print('='*98)
results = {}
for name, z0, z1, vfunc, mat, v_os in LAYERS:
    tau_vE, zz, Ek, vv = integrate_tau(z0, z1, vfunc)
    W = z1 - z0
    tau_vos = W / v_os
    ratio = tau_vE / tau_vos if tau_vos > 0 else np.inf
    Ek_avg = np.mean(Ek); vv_avg = np.mean(vv)
    print(f'{name:>22} | {z0*1e6:>6.3f}–{z1*1e6:<6.3f} | {W*1e9:>6.0f}  | '
          f'{tau_vE*1e12:>10.3f} | {tau_vos*1e12:>11.3f} | {ratio:>5.2f}×')
    results[name] = dict(tau_vE=tau_vE, tau_vos=tau_vos, zz=zz, Ek=Ek, vv=vv,
                         z0=z0, z1=z1, mat=mat, W=W, Ek_avg=Ek_avg, vv_avg=vv_avg)
print('-'*98)
total_vE  = sum(r['tau_vE']  for r in results.values())
total_vos = sum(r['tau_vos'] for r in results.values())
print(f'{"Total drift":>22} | {"":>15} | {"980":>6}  | '
      f'{total_vE*1e12:>10.3f} | {total_vos*1e12:>11.3f} | {total_vE/total_vos:>5.2f}×')

# ── Plot E-field + v(E) integration ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Drift transit time from simulated E-field + v(E)\n'
             '(blue: E-field, red: v(E)-derived drift velocity)',
             fontsize=12, fontweight='bold')

# (0,0) E-field profile with layer boundaries
ax = axes[0, 0]
ax.semilogy(_z*1e6, _E/1e5, '-', color='#2196F3', lw=1.0)
ax.set_xlabel('z (μm)'); ax.set_ylabel('|E| (kV/cm)')
ax.set_title('E-field profile (dark, −7 V)', fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
colors = ['#1B998B', '#9C27B0', '#FF9800', '#E91E8C']
for (name, z0, z1, *_), c in zip(LAYERS, colors):
    ax.axvspan(z0*1e6, z1*1e6, color=c, alpha=0.18, label=name)
ax.set_xlim(0.5, 1.7); ax.legend(fontsize=8, loc='upper right')

# (0,1) v(E) curves
ax = axes[0, 1]
ax.plot(E_InP,    v_InP*1e5/1e5,    '-', color='#E63946', lw=1.5, label='InP (×10⁵ m/s)')
ax.plot(E_InGaAs, v_InGaAs*1e4/1e5, '-', color='#1B998B', lw=1.5, label='InGaAs (×10⁵ m/s)')
ax.plot(E_Q11,    v_Q11*1e4/1e5,    '-', color='#9C27B0', lw=1.5, label='InGaAsP Q1.1 (×10⁵ m/s)')
ax.set_xlabel('E (kV/cm)'); ax.set_ylabel('v (×10⁵ m/s)')
ax.set_title('v(E) curves', fontweight='bold')
ax.grid(True, alpha=0.3); ax.set_xlim(0, 50); ax.legend(fontsize=9)

# (1,0) E(z) per layer (linear)
ax = axes[1, 0]
for (name, z0, z1, *_), c in zip(LAYERS, colors):
    r = results[name]
    ax.plot(r['zz']*1e6, r['Ek'], color=c, lw=1.5, label=f'{name}: ⟨E⟩={r["Ek_avg"]:.1f} kV/cm')
ax.set_xlabel('z (μm)'); ax.set_ylabel('E (kV/cm)')
ax.set_title('E-field sampled in each drift layer', fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

# (1,1) v(z) per layer
ax = axes[1, 1]
for (name, z0, z1, *_), c in zip(LAYERS, colors):
    r = results[name]
    ax.plot(r['zz']*1e6, r['vv']/1e5, color=c, lw=1.5,
            label=f'{name}: τ={r["tau_vE"]*1e12:.2f} ps (vs vos={r["tau_vos"]*1e12:.2f})')
ax.set_xlabel('z (μm)'); ax.set_ylabel('v (×10⁵ m/s)')
ax.set_title('Drift velocity v(E(z)) in each layer', fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('tau_vE_layers.png', dpi=150, bbox_inches='tight')
print('\nSaved: tau_vE_layers.png')

# Save τ values for use in main fitting
with open('tau_vE_values.txt', 'w') as f:
    f.write('# Transit times from E-field + v(E) integration (ps)\n')
    f.write('# Layer\tz0_um\tz1_um\tW_nm\ttau_vE_ps\ttau_vos_ps\n')
    for name, *_ in LAYERS:
        r = results[name]
        f.write(f'{name}\t{r["z0"]*1e6:.3f}\t{r["z1"]*1e6:.3f}\t'
                f'{r["W"]*1e9:.0f}\t{r["tau_vE"]*1e12:.4f}\t{r["tau_vos"]*1e12:.4f}\n')
print('Saved: tau_vE_values.txt')
