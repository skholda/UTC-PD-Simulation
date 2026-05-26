"""
Rs sweep with/without shunt resistance Rp
=========================================
Sweep Rs from 1 to 50 Ohm (10 Ohm step).
Compare: Rp = infinity (no shunt) vs Rp = 50 Ohm.

Circuit:
  I_ph || C_PD -- Rs -- L_CPW2 -- NodeA -- L_CPW -- R_L(50Ω)
                                     |
                              C_CPW   Rp + jωL_Rp
                                |        |
                               GND      GND
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Fixed circuit parameters (from CLAUDE.md / main script) ─────────
C_CPW   = 28.93e-15      # F
L_CPW   = 178.9e-12      # H  (total CPW inductance)
L_CPW2  = 0.0            # H  (PD-side CPW segment; 0 for clean comparison)
L_Rp    = 0.0            # H  (parasitic L of Rp; 0 for clean comparison)
C_PD    = 132.7e-15      # F
R_L     = 50.0           # Ω

# ── UTC-PD epilayer + transit times (constant v_os model) ───────────
W_A_undep = 480e-9
W_A_dep   = 160e-9
W_cliff   = 50e-9
W_C       = 740e-9
W_tot     = W_A_undep + W_A_dep + W_cliff + W_C

tau_A      = 7.862e-12
v_os_InP   = 4.0e5       # m/s
v_os_InGaAs = 2.1e5      # m/s
v_h_InGaAs = 4.5e4       # m/s (hole)

tau_Ad = W_A_dep / v_os_InGaAs
tau_cl = W_cliff / v_os_InP
tau_C  = W_C     / v_os_InP
tau_h  = W_A_dep / v_h_InGaAs

def H_ph(w):
    sinc = lambda x: np.sinc(x / np.pi)
    HA   = 1 / (1 + 1j*w*tau_A)
    HAd  = sinc(w*tau_Ad/2) * np.exp(-1j*(w*tau_A + w*tau_Ad/2))
    Hcl  = sinc(w*tau_cl/2) * np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl/2))
    Hco  = sinc(w*tau_C/2)  * np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl + w*tau_C/2))
    HAd_h = sinc(w*tau_h/2) * np.exp(-1j*(w*tau_A + w*tau_h/2))
    return (W_A_undep*HA + W_A_dep*HAd + W_A_dep*HAd_h
            + W_cliff*Hcl + W_C*Hco) / (W_tot + W_A_dep)

def _Y_Rp(w, Rp):
    if np.isinf(Rp):
        return 0.0
    return 1.0 / (Rp + 1j*w*L_Rp)

def H_ckt(w, Rs, Rp):
    Zs  = Rs + 1j*w*L_CPW2
    Y_A = 1j*w*C_CPW + _Y_Rp(w, Rp) + 1/(1j*w*L_CPW + R_L)
    return (R_L / (1j*w*L_CPW + R_L)) / (1j*w*C_PD + Y_A*(1 + 1j*w*C_PD*Zs))

def H_total(w, Rs, Rp):
    return H_ph(w) * H_ckt(w, Rs, Rp)

def find_3dB(f, Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Sweep configuration ─────────────────────────────────────────────
Rs_list = [1, 10, 20, 30, 40, 50]           # Ω
Rp_cases = [(np.inf, 'No shunt (Open)'),
            (50.0,   r'$R_p$ = 50 Ω')]

f = np.linspace(0.1e9, 60e9, 4000)
w = 2*np.pi*f

# ── Compute ─────────────────────────────────────────────────────────
results = {}      # results[Rp][Rs] = (H_dBohm, f3dB_norm)
for Rp, _ in Rp_cases:
    results[Rp] = {}
    for Rs in Rs_list:
        Ht = H_total(w, Rs, Rp)
        H_abs_dB = 20*np.log10(np.abs(Ht))            # absolute |H| in dBΩ
        Hd_norm  = 20*np.log10(np.abs(Ht)/np.abs(Ht[0]))
        f3 = find_3dB(f, Hd_norm)
        results[Rp][Rs] = (H_abs_dB, f3)

# ── Plot (single overlay) ──────────────────────────────────────────
cmap = plt.cm.viridis
colors = [cmap(i/(len(Rs_list)-1)) for i in range(len(Rs_list))]
ls_map = {np.inf: '-', 50.0: '--'}     # solid = no shunt, dashed = Rp=50Ω

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.suptitle(
    r'$R_s$ sweep — Transimpedance $|H_{tot}|$ (absolute)' '\n'
    rf'$C_j$={C_PD*1e15:.1f} fF,  $L_{{CPW}}$={L_CPW*1e12:.1f} pH,  '
    rf'$C_{{CPW}}$={C_CPW*1e15:.2f} fF',
    fontsize=12, fontweight='bold')

for Rp, _ in Rp_cases:
    for ci, Rs in enumerate(Rs_list):
        Hd, f3 = results[Rp][Rs]
        rp_lbl = r'$R_p{=}\infty$' if np.isinf(Rp) else r'$R_p{=}50\,\Omega$'
        ax.plot(f/1e9, Hd, color=colors[ci], lw=1.7, ls=ls_map[Rp],
                label=rf'$R_s$={Rs:>2} Ω, {rp_lbl}  |  $H(0)$={Hd[0]:.2f} dBΩ')

ax.set_xlabel('Frequency (GHz)', fontsize=11)
ax.set_ylabel(r'$|H_{tot}|$ (dBΩ)', fontsize=11)
ax.set_xlim(0, 60)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9, ncol=2)

fig.tight_layout()
out_path = 'Rs_sweep_Rp_compare.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

# ── Bandwidth summary bar plot ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
x = np.arange(len(Rs_list))
bw = 0.38
bw1 = [results[np.inf][Rs][1] for Rs in Rs_list]
bw2 = [results[50.0][Rs][1]  for Rs in Rs_list]

bars1 = ax2.bar(x - bw/2, bw1, bw, color='#1B998B',
                edgecolor='k', lw=0.8, label='No shunt (Open)')
bars2 = ax2.bar(x + bw/2, bw2, bw, color='#FF8C00',
                edgecolor='k', lw=0.8, label=r'$R_p$ = 50 Ω')
for b, v in zip(bars1, bw1):
    if not np.isnan(v):
        ax2.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}',
                 ha='center', fontsize=8.5, fontweight='bold')
for b, v in zip(bars2, bw2):
    if not np.isnan(v):
        ax2.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}',
                 ha='center', fontsize=8.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels([rf'{Rs} Ω' for Rs in Rs_list])
ax2.set_xlabel(r'$R_s$', fontsize=12)
ax2.set_ylabel(r'$f_{3dB}$ (GHz)', fontsize=12)
ax2.set_title(r'3 dB Bandwidth vs $R_s$', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.35)
fig2.tight_layout()
out_path2 = 'Rs_sweep_BW_summary.png'
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path2}')

# ── Print summary table ─────────────────────────────────────────────
print()
print(f'{"Rs (Ω)":>8} | {"BW @ Rp=∞ (GHz)":>18} | {"BW @ Rp=50Ω (GHz)":>20} | {"ΔBW":>8}')
print('-' * 65)
for Rs in Rs_list:
    f3_inf = results[np.inf][Rs][1]
    f3_50  = results[50.0][Rs][1]
    s1 = f'{f3_inf:.2f}' if not np.isnan(f3_inf) else '>60'
    s2 = f'{f3_50:.2f}'  if not np.isnan(f3_50)  else '>60'
    d  = f'{f3_50 - f3_inf:+.2f}' if (not np.isnan(f3_inf) and not np.isnan(f3_50)) else 'N/A'
    print(f'{Rs:>8} | {s1:>18} | {s2:>20} | {d:>8}')
