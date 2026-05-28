"""
Compare paper single-sinc (locked baseline) vs multi-layer sinc decomposition
=============================================================================
Both use v_os transit times. Only difference: whether collector region is
treated as a single sinc(ωτ_C/2) or as a sum of per-layer sinc(ωτ_i/2)
terms with cumulative phase delays.

τ_A (absorber 1-pole) and τ_R (dielectric relaxation) are identical in
both — only the drift (W_C) part differs.
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt

# ── Paper H_ph (locked baseline) ────────────────────────────────────
W_A     = 480e-9
tau_A   = 3.530e-12
tau_R   = 0.070e-12

# Drift layers (v_os values)
W_dep    = 160e-9; tau_dep = 0.762e-12       # InGaAs dep absorber
W_gr     = 30e-9;  tau_gr  = 0.143e-12       # InGaAsP grading (Q1.1+Q1.4)
W_cl     = 50e-9;  tau_cl  = 0.125e-12       # InP cliff
W_col    = 740e-9; tau_col = 1.850e-12       # InP collector
W_C      = W_dep + W_gr + W_cl + W_col       # 980 nm
tau_C    = tau_dep + tau_gr + tau_cl + tau_col   # 2.880 ps

W_tot    = W_A + W_C

# Cumulative delays (entry into each layer)
tcum_dep = 0.0
tcum_gr  = tau_dep
tcum_cl  = tau_dep + tau_gr
tcum_col = tau_dep + tau_gr + tau_cl

def absorber_term(w):
    return W_A * (2.0 + 1j*w*tau_R) / (2.0*(1.0 + 1j*w*tau_R))

def drift_paper(w):
    """Paper single-sinc over W_C with τ_C = sum of τ_i."""
    sinc = lambda x: np.sinc(x/np.pi)
    return W_C * sinc(w*tau_C/2) * np.exp(-1j*w*tau_C/2)

def drift_multi(w):
    """Multi-layer sinc decomposition with cumulative phase."""
    sinc = lambda x: np.sinc(x/np.pi)
    return ( W_dep * sinc(w*tau_dep/2) * np.exp(-1j*w*(tcum_dep + tau_dep/2))
           + W_gr  * sinc(w*tau_gr /2) * np.exp(-1j*w*(tcum_gr  + tau_gr /2))
           + W_cl  * sinc(w*tau_cl /2) * np.exp(-1j*w*(tcum_cl  + tau_cl /2))
           + W_col * sinc(w*tau_col/2) * np.exp(-1j*w*(tcum_col + tau_col/2)) )

def H_ph(w, drift_func):
    return (absorber_term(w) + drift_func(w)) / (W_tot * (1.0 + 1j*w*tau_A))

# ── Sweep and compare ──────────────────────────────────────────────
f = np.linspace(0.1e9, 60e9, 6000)
w = 2*np.pi*f
Hp = H_ph(w, drift_paper)
Hm = H_ph(w, drift_multi)

mag_paper_dB = 20*np.log10(np.abs(Hp)/np.abs(Hp[0]))
mag_multi_dB = 20*np.log10(np.abs(Hm)/np.abs(Hm[0]))
diff_dB      = mag_multi_dB - mag_paper_dB
phase_paper  = np.unwrap(np.angle(Hp))*180/np.pi
phase_multi  = np.unwrap(np.angle(Hm))*180/np.pi
diff_phase   = phase_multi - phase_paper

# Find -3 dB points
def f3dB(Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan
f3_paper = f3dB(mag_paper_dB)
f3_multi = f3dB(mag_multi_dB)

# Print table at key frequencies
print('='*80)
print(f'Paper single-sinc  τ_C = {tau_C*1e12:.3f} ps   (τ_eff = {tau_C/2*1e12:.3f} ps)')
print(f'Multi-layer        τ_i = {tau_dep*1e12:.3f}, {tau_gr*1e12:.3f}, '
      f'{tau_cl*1e12:.3f}, {tau_col*1e12:.3f} ps')
print(f'                   τ_eff (thickness-weighted center) = '
      f'{(W_dep*(tcum_dep+tau_dep/2) + W_gr*(tcum_gr+tau_gr/2) + W_cl*(tcum_cl+tau_cl/2) + W_col*(tcum_col+tau_col/2))/W_C*1e12:.3f} ps')
print('-'*80)
print(f'{"f (GHz)":>8} | {"|H| paper (dB)":>14} | {"|H| multi (dB)":>14} | {"ΔdB":>8} | '
      f'{"Δphase (°)":>12}')
print('-'*80)
for ft in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
    i = np.argmin(np.abs(f - ft*1e9))
    print(f'{ft:>8} | {mag_paper_dB[i]:>14.4f} | {mag_multi_dB[i]:>14.4f} | '
          f'{diff_dB[i]:>+8.4f} | {diff_phase[i]:>+12.3f}')
print('-'*80)
print(f'  f_3dB:  paper = {f3_paper:.2f} GHz,  multi = {f3_multi:.2f} GHz')

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    r'$H_{ph}$ comparison: Paper single-sinc  vs  Multi-layer decomposition' '\n'
    rf'(both use $v_{{os}}$; $\tau_C^{{paper}}$={tau_C*1e12:.2f} ps; '
    rf'multi: $\tau_{{dep,gr,cl,col}}$ = {tau_dep*1e12:.2f}/{tau_gr*1e12:.2f}/'
    rf'{tau_cl*1e12:.2f}/{tau_col*1e12:.2f} ps)',
    fontsize=11, fontweight='bold')

ax = axes[0, 0]
ax.plot(f/1e9, mag_paper_dB, '-', color='#E63946', lw=2.0, label='Paper (single sinc)')
ax.plot(f/1e9, mag_multi_dB, '--', color='#1B998B', lw=1.6, label='Multi-layer')
ax.axhline(-3, color='gray', ls=':', lw=0.8)
ax.axvline(f3_paper, color='#E63946', ls=':', lw=0.7, alpha=0.6)
ax.axvline(f3_multi, color='#1B998B', ls=':', lw=0.7, alpha=0.6)
ax.text(45, -2.5, '-3 dB', fontsize=9, color='gray')
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized $|H_{ph}|$ (dB)')
ax.set_title(rf'$|H_{{ph}}|$ — $f_{{3dB}}$ paper={f3_paper:.1f} GHz, multi={f3_multi:.1f} GHz',
             fontsize=10, fontweight='bold')
ax.set_xlim(0, 60); ax.set_ylim(-15, 1)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(f/1e9, diff_dB, '-', color='#FF8C00', lw=1.8)
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel(r'$\Delta|H_{ph}|$ (dB)  [multi − paper]')
ax.set_title('Magnitude difference', fontsize=10, fontweight='bold')
ax.set_xlim(0, 60); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(f/1e9, phase_paper, '-', color='#E63946', lw=2.0, label='Paper')
ax.plot(f/1e9, phase_multi, '--', color='#1B998B', lw=1.6, label='Multi-layer')
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Phase (°)')
ax.set_title(r'$\angle H_{ph}$', fontsize=10, fontweight='bold')
ax.set_xlim(0, 60); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(f/1e9, diff_phase, '-', color='#FF8C00', lw=1.8)
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel(r'$\Delta$Phase (°)  [multi − paper]')
ax.set_title('Phase difference', fontsize=10, fontweight='bold')
ax.set_xlim(0, 60); ax.grid(True, alpha=0.3)

fig.tight_layout()
out = 'Hph_paper_vs_multi.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
