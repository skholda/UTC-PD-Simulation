"""
UTC-PD 30 μm Final Simulation  (BASELINE — locked configuration)
=================================================================
Locked framework:
  H_ph(ω) = exact paper MUTC transfer function H_MUTC(ω), two generation groups:
    Bracket 1 (undep-abs generated, weight η_U/[W_T(1+jωτ_A)]):
      W_U·(2+jωτ_R)/(2(1+jωτ_R)) + W_D·D(τ_eD) + W_C·sinc(ωτ_C/2)·exp(-jω(τ_eD+τ_C/2))
    Bracket 2 (dep-abs generated in-situ, weight η_D/W_T, NO τ_A pole):
      W_U·D(τ_h) + (W_D/jωτ_eD)·{1-D(τ_eD)} + (W_D/jωτ_h)·{1-D(τ_h)}
      + W_C·sinc(ωτ_eD/2)·sinc(ωτ_C/2)·exp(-jω(τ_eD+τ_C/2))
    with D(τ)=sinc(ωτ/2)·exp(-jωτ/2), sinc(x)=sin(x)/x.
    Key: τ_A (diffusion pole) multiplies ONLY the undep-absorber group. Dep-abs
    carriers are born in-situ -> no τ_A, uniform-generation triangular transit.
    Each region a carrier crosses gets its own term (undep e AND dep e both
    cross the collector -> W_C appears in both brackets).

  Region split (each width counted once; W_T = W_U+W_D+W_C):
    W_U = 480 nm  (undep absorber, p+ InGaAs, diff + quasi-field)
    W_D = 240 nm  (depleted InGaAs absorber, in-situ e/h generation)
    W_C = 740 nm  (electron-only collector, InP)
    W_T = 1460 nm

  Generation fractions (uniform optical generation over absorbers):
    η_U = W_U/(W_U+W_D) = 0.6667   η_D = W_D/(W_U+W_D) = 0.3333

  Transit times (LAYER-AVERAGE:  τ = W / v(E_avg); τ_A NOT recomputed):
    E_avg from device field (Lumerical -7 V, 0.5 mA, lat 10 um); v(E_avg) from v(E) curve.
    τ_A   = 3.530 ps   (undep-abs diffusion:  W_U^2 / [D_e (3 + ln(p_max/p_min))])
    τ_eD  = 3.039 ps   (dep-abs e, E_avg=179 kV/cm -> InGaAs v=0.79e7 cm/s)
    τ_C   = 6.994 ps   (collector e, E_avg=27 kV/cm -> InP v=1.06e7 cm/s)
    τ_h   = 5.000 ps   (dep-abs hole, W_D/v_h,sat, v_h,sat=0.48e7 cm/s InGaAs, lit.)
    τ_R   = neglected in bandwidth calculation

  Transit-time-limited f_tr = 30.74 GHz  (|H_ph| = -3 dB)

  Circuit (S11-fitted):
    Cj    = 131.0 fF   (common, S11-only fit)
    Rs    = 8.92 Ω
    C_CPW = 46.53 fF   (pad-fitted)
    L_total = 197.9 pH (common, S11-only fit over 40 GHz, Option B)
    L_CPW, L_CPW2 per device (L_CPW + L_CPW2 = L_total):
        200 Ω: L_CPW=197.9, L_CPW2=0
        38 Ω : L_CPW=141.6, L_CPW2=56.3
        60 Ω : L_CPW=149.9, L_CPW2=48.0
        Open : L_CPW=197.9, L_CPW2=0
    L_Rp  : per-device, MATLAB FEM values (NOT fitted)
            200Ω: 153.7 pH, 38Ω: 65.6 pH, 60Ω: 71.8 pH, Open: 0 pH

Outputs:
  Combined fit figure, publication Smith, freq-response overlay,
  Origin Pro export tree.
"""
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. H_ph(ω)  —  exact paper MUTC transfer function  H_MUTC(ω)  (LOCKED)
# ═══════════════════════════════════════════════════════════════════
#   Two generation groups (by absorption location):
#     Bracket 1 (undep-absorber generated, weight η_U/[W_T(1+jωτ_A)]):
#         W_U·(2+jωτ_R)/(2(1+jωτ_R))
#       + W_D·D(τ_eD)                                   crossing dep absorber
#       + W_C·sinc(ωτ_C/2)·exp(-jω(τ_eD+τ_C/2))         crossing collector
#     Bracket 2 (dep-absorber generated in-situ, weight η_D/W_T, NO τ_A pole):
#         W_U·D(τ_h)                                     Ramo complement (hole→p+)
#       + (W_D/jωτ_eD)·{1-D(τ_eD)}                       electron, uniform-gen triangular
#       + (W_D/jωτ_h )·{1-D(τ_h )}                       hole, uniform-gen triangular
#       + W_C·sinc(ωτ_eD/2)·sinc(ωτ_C/2)·exp(-jω(τ_eD+τ_C/2))   e→collector
#   with D(τ)=sinc(ωτ/2)·exp(-jωτ/2),  sinc(x)=sin(x)/x.
W_U = 480e-9             # undepleted p-InGaAs absorber   (diffusion, quasi-field)
W_D = 240e-9             # depleted InGaAs absorber        (in-situ e/h generation)
W_C = 740e-9             # electron-only collector (InP)
W_T = W_U + W_D + W_C    # 1460 nm  total transport thickness

# Electron drift transit times by the LAYER-AVERAGE method: τ = W / v(E_avg),
# where E_avg = mean |E| over the layer from the DEVICE field (Lumerical CHARGE,
# -7 V, Iph=0.5 mA, lateral 10 um) and v(E_avg) from the material v(E) curve.
# Averaging the field first is dip-robust (a thin low-field patch cannot dominate).
# See tau_layer_avg_compute.py for the reproducible calculation. (τ_A NOT recomputed.)
#   dep-abs  W_D (z=0.74-0.98): E_avg=179.1 kV/cm, InGaAs v=0.79e5 m/s -> 3.039 ps
#   collector W_C (z=0-0.74):   E_avg= 27.2 kV/cm, InP    v=1.06e5 m/s -> 6.994 ps
v_h_InGaAs = 4.8e4       # m/s  InGaAs hole  saturation      (0.48e7 cm/s, literature)
tau_A  = 3.530e-12       # undep-absorber effective electron transit (diff+quasi-field) — KEPT
tau_R  = 0.0            # dielectric relaxation — NEGLECTED in bandwidth calc
tau_eD = 3.039e-12       # dep-absorber electron   (layer-average v(E_avg), new field)
tau_C  = 6.994e-12       # collector electron      (layer-average v(E_avg), new field)
tau_h  = W_D / v_h_InGaAs   # 5.00 ps  depleted-absorber hole (saturation)

# Generation fractions — uniform optical generation over the absorbers:
eta_U = W_U / (W_U + W_D)     # 0.6667
eta_D = W_D / (W_U + W_D)     # 0.3333

# ── back-compat aliases for reporting/summary code ─────────────────
W_A_paper, W_C_paper, W_Adep, W_norm = W_U, W_C, W_D, W_T

def H_ph(w):
    """Exact paper MUTC photocurrent transfer function H_MUTC(ω).

    τ_A (diffusion pole) multiplies ONLY the undep-absorber-generated group
    (bracket 1). Depleted-absorber carriers are generated in-situ (bracket 2,
    no τ_A) with uniform-generation *triangular* transit transfer functions.
    """
    sinc = lambda x: np.sinc(x/np.pi)
    D = lambda tau: sinc(w*tau/2) * np.exp(-1j*w*tau/2)          # rectangular transit
    def Tri(tau):                                                # uniform-gen triangular
        x = 1j*w*tau
        with np.errstate(invalid='ignore', divide='ignore'):
            val = (1.0 - D(tau))/x
        return np.where(np.abs(x) < 1e-12, 0.5 + 0j, val)
    casc = sinc(w*tau_C/2) * np.exp(-1j*w*(tau_eD + tau_C/2))    # collector w/ dep-abs delay
    b1 = ( W_U*(2.0 + 1j*w*tau_R)/(2.0*(1.0 + 1j*w*tau_R))
         + W_D*D(tau_eD)
         + W_C*casc )
    b1 = b1 * eta_U / (W_T*(1.0 + 1j*w*tau_A))
    b2 = ( W_U*D(tau_h)
         + W_D*Tri(tau_eD)
         + W_D*Tri(tau_h)
         + W_C*sinc(w*tau_eD/2)*casc )
    b2 = b2 * eta_D / W_T
    return b1 + b2

# ═══════════════════════════════════════════════════════════════════
# 2. CIRCUIT (LOCKED)
# ═══════════════════════════════════════════════════════════════════
C_CPW = 46.53e-15
R_L   = 50.0
Cj    = 131.0e-15
Rs    = 8.92

def _Y_Rp(w, Rp, Lrp):
    if np.isinf(Rp):
        return 0.0
    return 1.0 / (Rp + 1j*w*Lrp)

def H_ckt(w, Rs, Cpd, Rp, Lcpw, Lrp=0.0, Lcpw2=0.0):
    Zs  = Rs + 1j*w*Lcpw2
    Y_A = 1j*w*C_CPW + _Y_Rp(w, Rp, Lrp) + 1/(1j*w*Lcpw + R_L)
    return (R_L/(1j*w*Lcpw + R_L)) / (1j*w*Cpd + Y_A*(1 + 1j*w*Cpd*Zs))

def sim_S11(w, Rs, Cpd, Rp, Lcpw, Lrp=0.0, Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Z_dev = Zs + 1/(1j*w*Cpd)
    Y_n   = 1j*w*C_CPW + _Y_Rp(w, Rp, Lrp) + 1/Z_dev
    Z_in  = 1j*w*Lcpw + 1/Y_n
    return (Z_in - 50)/(Z_in + 50)

def get_bw(f, Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ═══════════════════════════════════════════════════════════════════
# 3. LOAD MEASUREMENT DATA (from main.py)
# ═══════════════════════════════════════════════════════════════════
_main = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz','ref_loss_dB',
            '_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
            '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)', _main, re.MULTILINE|re.DOTALL)
    if m is None: sys.exit(f'Missing {var} in main.py')
    exec(m.group(0), _ns)
ref_f_GHz, ref_loss_dB = _ns['ref_f_GHz'], _ns['ref_loss_dB']

# ═══════════════════════════════════════════════════════════════════
# 4. DEVICE CONFIGURATIONS (LOCKED)
# ═══════════════════════════════════════════════════════════════════
configs = [
    dict(lbl='Rp=200Ω', Rp=200.0,   col='#888888', mk='D',
         s1p=_ns['_s1p_200'], freq=_ns['_freq_200'],
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),    # L_tot=197.9, FEM L_Rp
    dict(lbl='Rp=38Ω',  Rp=38.0,    col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'],  freq=_ns['_freq_33'],
         Lcpw=141.6e-12, Lcpw2=56.3e-12, Lrp=65.6e-12),    # L_tot=197.9, FEM L_Rp
    dict(lbl='Rp=60Ω',  Rp=60.0,    col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'],  freq=_ns['_freq_55'],
         Lcpw=149.9e-12, Lcpw2=48.0e-12, Lrp=71.8e-12),    # L_tot=197.9, FEM L_Rp
    dict(lbl='Open',    Rp=np.inf,  col='#E91E8C', mk='^',
         s1p=_ns['_s1p_WO'],  freq=_ns['_freq_WO'],
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=0.0),
]

def gs1p(arr, f_max):
    fr = arr[:,0]; S = 10**(arr[:,1]/20)*np.exp(1j*np.deg2rad(arr[:,2]))
    m = fr <= f_max; return fr[m], S[m]
def gfr(arr):
    f_ghz = arr[:,0]; cal = arr[:,1] + np.interp(f_ghz, ref_f_GHz, ref_loss_dB)
    return f_ghz*1e9, cal - cal[0]

# ═══════════════════════════════════════════════════════════════════
# 5. SIMULATE & REPORT
# ═══════════════════════════════════════════════════════════════════
f_plot = np.linspace(0.1e9, 50e9, 5000); w_plot = 2*np.pi*f_plot

print('='*100)
print('UTC-PD 30 μm FINAL  (locked baseline)')
print('-'*100)
print(f'  H_ph: exact paper H_MUTC  W_U={W_U*1e9:.0f} nm, '
      f'W_D={W_D*1e9:.0f} nm, W_C={W_C*1e9:.0f} nm  (η_U={eta_U:.4f}, η_D={eta_D:.4f}, uniform gen.)')
print(f'        τ_A={tau_A*1e12:.3f} ps  (undep-abs diffusion pole)')
print(f'        τ_eD={tau_eD*1e12:.3f} ps  (dep-abs electron, layer-avg v(E_avg))')
print(f'        τ_C={tau_C*1e12:.3f} ps  (collector electron, layer-avg v(E_avg))')
print(f'        τ_h={tau_h*1e12:.3f} ps  (dep-abs hole, W_D/v_h,sat)')
print(f'        τ_R neglected in bandwidth calc')
_wtr = 2*np.pi*np.linspace(1e9,200e9,400000)
_mtr = np.abs(H_ph(_wtr))/np.abs(H_ph(1e-3*2*np.pi))
_itr = np.where(_mtr <= 1/np.sqrt(2))[0]
print(f'  transit-limited f_tr = {_wtr[_itr[0]]/2/np.pi/1e9:.2f} GHz  (|H_ph|=-3 dB)')
print(f'  Circuit:  Cj={Cj*1e15:.1f} fF, Rs={Rs} Ω, C_CPW={C_CPW*1e15:.2f} fF')
print('='*100)
print(f'{"Device":>10} | {"L_CPW":>7} | {"L_CPW2":>7} | {"L_Rp":>7} | '
      f'{"RMS_S11":>9} | {"BW":>7} | {"RMS_H":>7}')
print('-'*100)

for cfg in configs:
    fm, pm     = gfr(cfg['freq']); wm = 2*np.pi*fm
    # S11: use FULL measured range (up to 40 GHz), not truncated by freq response
    fs11, S11m = gs1p(cfg['s1p'], cfg['s1p'][:,0].max()); ws = 2*np.pi*fs11

    S11s = sim_S11(ws, Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S11s - S11m)**2))

    Hckt_m = H_ckt(wm,     Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot, Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Hd_m = 20*np.log10(np.abs(H_ph(wm)*Hckt_m)/np.abs(H_ph(wm[0])*Hckt_m[0]))
    Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m - pm)**2))
    bw    = get_bw(f_plot, Hd_p)
    bw_s  = f'{bw:.1f}' if not np.isnan(bw) else '>50'

    cfg.update(fs11=fs11, S11m=S11m, S11s=S11s, rms_s11=rms_s11,
               fm=fm, pm=pm, Hd_m=Hd_m, Hd_p=Hd_p, bw=bw, bw_s=bw_s, rms_H=rms_H)

    print(f'{cfg["lbl"]:>10} | {cfg["Lcpw"]*1e12:>6.1f}pH | {cfg["Lcpw2"]*1e12:>6.1f}pH | '
          f'{cfg["Lrp"]*1e12:>6.1f}pH | {rms_s11:>9.5f} | {bw_s:>7} | {rms_H:>7.3f}')

# ═══════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════
def draw_smith(ax, lw_grid=0.6):
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0), 1, fill=False, color='#888', lw=lw_grid+0.3))
    ax.axhline(0, color='#888', lw=lw_grid, zorder=0)
    for r in [0.2, 0.5, 1, 2, 5]:
        cx, rad = r/(r+1), 1/(r+1)
        ax.add_patch(plt.Circle((cx,0), rad, fill=False, color='#aaa',
                                lw=lw_grid, ls=':', zorder=0))
    theta = np.linspace(0, np.pi, 400)
    for x in [0.2, 0.5, 1, 2, 5]:
        for sgn in [1, -1]:
            cx, cy, rad = 1, sgn/x, 1/x
            xx = cx + rad*np.cos(theta); yy = cy + rad*np.sin(theta)*sgn
            m  = (xx**2 + yy**2 <= 1.002)
            ax.plot(xx[m], yy[m], color='#aaa', lw=lw_grid, ls=':', zorder=0)

# ── Combined 3-row figure ──────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    rf'UTC-PD 30 μm — FINAL  (paper $H_{{ph}}$ + $v_{{os}}$ drift)' '\n'
    rf'$W_A$={W_A_paper*1e9:.0f}/$W_C$={W_C_paper*1e9:.0f} nm,  '
    rf'$\tau_A$={tau_A*1e12:.2f} / $\tau_C$={tau_C*1e12:.2f} ps,  '
    rf'$C_j$={Cj*1e15:.1f} fF (S11-fit common)' '\n'
    r'Row 1: Smith  |  Row 2: $|S_{11}|$ dB  |  Row 3: Frequency response',
    fontsize=11, fontweight='bold')

for ci, cfg in enumerate(configs):
    col = cfg['col']; mk = cfg['mk']

    ax = axes[0, ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S11s'].real, cfg['S11s'].imag, '--', color='k', lw=1.5, zorder=5, label='Sim')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|={cfg["rms_s11"]:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    ax = axes[1, ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-',  color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11s'])), '--', color='k',  lw=1.4, label='Sim')
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)

    ax = axes[2, ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, cfg['Hd_p'], '-', color=col, lw=2.0,
            label=f'Sim  BW={cfg["bw_s"]} GHz  RMS={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('UTC_PD_final_fit.png', dpi=150, bbox_inches='tight')
print('\nSaved: UTC_PD_final_fit.png')

# ── Publication Smith ──────────────────────────────────────────────
_pub_lbl = {'Rp=200Ω':'200 Ω', 'Rp=38Ω':'38 Ω', 'Rp=60Ω':'60 Ω', 'Open':'WO'}
fig_sc, axes_sc = plt.subplots(2, 2, figsize=(10, 10))
fig_sc.patch.set_facecolor('white')
for ci, cfg in enumerate(configs):
    ax = axes_sc[ci//2][ci%2]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=12, color=cfg['col'], zorder=6, label='Meas.')
    ax.plot(cfg['S11s'].real, cfg['S11s'].imag, '--', color='k', lw=1.5, zorder=5, label='Fit')
    ax.text(-0.95, 0.92, 'Diameter: 30 μm', fontsize=13, va='top', ha='left')
    ax.text(-0.95, 0.72, f'Resistance: {_pub_lbl[cfg["lbl"]]}',
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.text(-0.95, 0.50, 'Bias: −7 V', fontsize=13, va='top', ha='left')
    ax.legend(fontsize=13, loc='lower left', framealpha=0.85,
              edgecolor='none', handlelength=1.5, markerscale=1.8)
fig_sc.tight_layout(pad=1.5)
fig_sc.savefig('UTC_PD_final_smith.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: UTC_PD_final_smith.png')

# ── Freq-response overlay ──────────────────────────────────────────
fig_fr, ax_fr = plt.subplots(figsize=(10, 6))
for cfg in configs:
    ax_fr.plot(cfg['fm']/1e9, cfg['pm'], cfg['mk'], color=cfg['col'], ms=4, alpha=0.6)
    ax_fr.plot(f_plot/1e9, cfg['Hd_p'], '-', color=cfg['col'], lw=1.5,
               label=f'{cfg["lbl"]}  BW={cfg["bw_s"]} GHz')
ax_fr.axhline(-3, color='gray', ls='--', lw=0.8, label='-3 dB')
ax_fr.set_xlabel('Frequency (GHz)', fontsize=12)
ax_fr.set_ylabel('Normalized Response (dB)', fontsize=12)
ax_fr.set_xlim(0, 45); ax_fr.set_ylim(-12, 3)
ax_fr.set_title('Frequency Response — Final (paper $H_{ph}$ + $v_{os}$)', fontsize=12)
ax_fr.legend(fontsize=10, loc='lower left'); ax_fr.grid(True, alpha=0.3)
fig_fr.tight_layout()
fig_fr.savefig('UTC_PD_final_freqresp.png', dpi=150, bbox_inches='tight')
print('Saved: UTC_PD_final_freqresp.png')

# ═══════════════════════════════════════════════════════════════════
# 7. ORIGIN PRO EXPORT
# ═══════════════════════════════════════════════════════════════════
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'origin_data_final')
os.makedirs(_out, exist_ok=True)

def _write(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(f'{v:.8g}' for v in r) + '\n')
def _tag(lbl):
    return lbl.replace('Ω','ohm').replace('=','_').replace(' ','_')

for cfg in configs:
    t  = _tag(cfg['lbl'])
    fs = cfg['fs11']
    S11m, S11s = cfg['S11m'], cfg['S11s']
    _write(os.path.join(_out, f'smith_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg',
            'Re_S11_meas','Im_S11_meas','Re_S11_fit','Im_S11_fit'],
           list(zip(fs/1e9,
                    20*np.log10(np.abs(S11m)+1e-30), np.angle(S11m, deg=True),
                    20*np.log10(np.abs(S11s)+1e-30), np.angle(S11s, deg=True),
                    S11m.real, S11m.imag, S11s.real, S11s.imag)))
    _write(os.path.join(_out, f's11_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg'],
           list(zip(fs/1e9,
                    20*np.log10(np.abs(S11m)), np.angle(S11m, deg=True),
                    20*np.log10(np.abs(S11s)), np.angle(S11s, deg=True))))
    _write(os.path.join(_out, f'freqresp_meas_{t}.txt'),
           ['Freq_GHz','Norm_dB'], list(zip(cfg['fm']/1e9, cfg['pm'])))
    _write(os.path.join(_out, f'freqresp_sim_{t}.txt'),
           ['Freq_GHz','Norm_dB'], list(zip(f_plot/1e9, cfg['Hd_p'])))

# Summary
with open(os.path.join(_out, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f'# Final baseline (locked)\n')
    f.write(f'# H_ph: exact paper H_MUTC (eta-weighted, uniform-gen triangular dep-abs)  '
            f'W_U={W_U*1e9:.0f} nm  W_D={W_D*1e9:.0f} nm  W_C={W_C*1e9:.0f} nm  W_T={W_T*1e9:.0f} nm  '
            f'eta_U={eta_U:.4f} eta_D={eta_D:.4f} (uniform generation)\n')
    f.write(f'#   tau_A={tau_A*1e12:.3f} ps  tau_eD={tau_eD*1e12:.3f} ps  tau_C={tau_C*1e12:.3f} ps  '
            f'tau_h={tau_h*1e12:.3f} ps  (layer-avg v(E); tau_R neglected)  |  f_tr=30.74 GHz\n')
    f.write(f'# Circuit:  Cj={Cj*1e15:.2f} fF  Rs={Rs} ohm  C_CPW={C_CPW*1e15:.2f} fF\n')
    f.write('Device\tL_CPW_pH\tL_CPW2_pH\tL_Rp_pH\tCj_fF\tRs_ohm\t'
            'RMS_S11\tBW_GHz\tRMS_H_dB\n')
    for cfg in configs:
        bw_v = 0 if np.isnan(cfg['bw']) else cfg['bw']
        f.write(f'{cfg["lbl"]}\t{cfg["Lcpw"]*1e12:.4f}\t{cfg["Lcpw2"]*1e12:.4f}\t'
                f'{cfg["Lrp"]*1e12:.4f}\t{Cj*1e15:.4f}\t{Rs:.4f}\t'
                f'{cfg["rms_s11"]:.5f}\t{bw_v:.4f}\t{cfg["rms_H"]:.4f}\n')

print(f'\nOrigin export to: {_out}/')
print('  smith_*.txt          — Smith chart (freq+dB+phase+Re/Im, meas & fit)')
print('  s11_*.txt            — |S11| + phase vs freq')
print('  freqresp_meas_*.txt  — Measured normalized response')
print('  freqresp_sim_*.txt   — Simulated normalized response')
print('  summary.txt          — Per-device parameter & metric summary')
