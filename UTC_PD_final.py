"""
UTC-PD 30 μm Final Simulation
=============================
Incorporates all modifications from the analysis session:
  * Cj = 147 fF  (fixed for all devices)
  * L_Rp (38 Ω)  = 55.2 pH  (S11-fitted, vs MATLAB FEM 65.6 pH)
  * L_Rp other devices = MATLAB FEM
  * C_CPW = 46.53 fF  (pad-fitted)
  * tau_A = 7.862 ps  (code default — gives best freq-response match)
  * Other circuit params (Rs, L_CPW, L_CPW2) from original S11 fit
  * Origin Pro export included

Topology:
  I_ph || C_PD ── Rs ── L_CPW2 ── NodeA ── L_CPW ── R_L
                                   │
                            C_CPW   Rp + jωL_Rp
                              │         │
                             GND       GND
"""
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ═══════════════════════════════════════════════════════════════════
# 1. UTC-PD LAYER STRUCTURE
# ═══════════════════════════════════════════════════════════════════
W_A_undep = 480e-9
W_A_dep   = 160e-9
W_cliff   = 50e-9
W_C       = 740e-9
W_tot     = W_A_undep + W_A_dep + W_cliff + W_C

# ═══════════════════════════════════════════════════════════════════
# 2. CARRIER VELOCITIES (Ishibashi 2000 §6.5)
# ═══════════════════════════════════════════════════════════════════
v_os_InP    = 4.0e5      # m/s  electron overshoot, InP
v_os_InGaAs = 2.1e5      # m/s  electron overshoot, InGaAs
v_h_InGaAs  = 4.5e4      # m/s  hole saturation, InGaAs

# ═══════════════════════════════════════════════════════════════════
# 3. TRANSIT TIMES
# ═══════════════════════════════════════════════════════════════════
tau_A  = 7.862e-12        # absorber (diffusion + quasi-field, effective)
tau_Ad = W_A_dep / v_os_InGaAs       # depleted absorber (electron)
tau_cl = W_cliff / v_os_InP          # cliff
tau_C  = W_C     / v_os_InP          # collector
tau_h  = W_A_dep / v_h_InGaAs        # depleted absorber (hole)

def H_ph(w):
    """Transit-time photoresponse (Ishibashi + hole correction)."""
    sinc = lambda x: np.sinc(x / np.pi)
    HA   = 1 / (1 + 1j*w*tau_A)
    HAd  = sinc(w*tau_Ad/2) * np.exp(-1j*(w*tau_A + w*tau_Ad/2))
    Hcl  = sinc(w*tau_cl/2) * np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl/2))
    Hco  = sinc(w*tau_C/2)  * np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl + w*tau_C/2))
    HAd_h = sinc(w*tau_h/2) * np.exp(-1j*(w*tau_A + w*tau_h/2))
    return (W_A_undep*HA + W_A_dep*HAd + W_A_dep*HAd_h
            + W_cliff*Hcl + W_C*Hco) / (W_tot + W_A_dep)

# ═══════════════════════════════════════════════════════════════════
# 4. CIRCUIT
# ═══════════════════════════════════════════════════════════════════
C_CPW   = 46.53e-15       # F (pad-fitted, overrides ADS 28.93)
R_L     = 50.0            # Ω
Cj      = 147.0e-15       # F (FIXED, all devices)
Rs      = 8.92            # Ω

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
# 5. LOAD MEASUREMENT DATA (from main.py inline arrays)
# ═══════════════════════════════════════════════════════════════════
_main = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz','ref_loss_dB',
            '_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
            '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)', _main, re.MULTILINE|re.DOTALL)
    if m is None:
        sys.exit(f'Missing array {var} in main.py')
    exec(m.group(0), _ns)

ref_f_GHz   = _ns['ref_f_GHz']
ref_loss_dB = _ns['ref_loss_dB']

# ═══════════════════════════════════════════════════════════════════
# 6. DEVICE CONFIGURATIONS  (Cj=147 fF, L_Rp(38Ω)=55.2 pH)
# ═══════════════════════════════════════════════════════════════════
configs = [
    dict(lbl='Rp=200Ω', Rp=200.0,   col='#888888', mk='D',
         s1p=_ns['_s1p_200'], freq=_ns['_freq_200'],
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=153.0e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,    col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'],  freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=55.2e-12),   # ← fitted
    dict(lbl='Rp=60Ω',  Rp=60.0,    col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'],  freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf,  col='#E91E8C', mk='^',
         s1p=_ns['_s1p_WO'],  freq=_ns['_freq_WO'],
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=0.0),
]

# ═══════════════════════════════════════════════════════════════════
# 7. DATA HELPERS
# ═══════════════════════════════════════════════════════════════════
def get_s1p(arr, f_max):
    fr  = arr[:,0]
    S11 = 10**(arr[:,1]/20) * np.exp(1j*np.deg2rad(arr[:,2]))
    m   = fr <= f_max
    return fr[m], S11[m]

def get_freq_response(arr):
    f_ghz = arr[:,0]
    cal   = arr[:,1] + np.interp(f_ghz, ref_f_GHz, ref_loss_dB)
    return f_ghz*1e9, cal - cal[0]

# ═══════════════════════════════════════════════════════════════════
# 8. RUN SIMULATION (no fitting — all params fixed)
# ═══════════════════════════════════════════════════════════════════
f_plot = np.linspace(0.1e9, 50e9, 5000)
w_plot = 2*np.pi*f_plot

print('='*98)
print(f'UTC-PD 30 μm Final  |  Cj = {Cj*1e15:.0f} fF (fixed),  Rs = {Rs} Ω,  '
      f'C_CPW = {C_CPW*1e15:.2f} fF,  τ_A = {tau_A*1e12:.3f} ps')
print('='*98)
print(f'{"Device":>10} | {"L_CPW":>7} | {"L_CPW2":>7} | {"L_Rp":>7} | '
      f'{"RMS_S11":>9} | {"BW":>7} | {"RMS_H":>7}')
print('-'*98)

for cfg in configs:
    fm, pm     = get_freq_response(cfg['freq']);   wm = 2*np.pi*fm
    fs11, S11m = get_s1p(cfg['s1p'], fm.max());    ws = 2*np.pi*fs11

    # S11
    S11s = sim_S11(ws, Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S11s - S11m)**2))

    # Frequency response (at measurement points + smooth curve)
    Hckt_m = H_ckt(wm,     Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot, Rs, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Ht_m = H_ph(wm) * Hckt_m
    Ht_p = H_ph(w_plot) * Hckt_p
    Hd_m = 20*np.log10(np.abs(Ht_m)/np.abs(Ht_m[0]))
    Hd_p = 20*np.log10(np.abs(Ht_p)/np.abs(Ht_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m - pm)**2))
    bw    = get_bw(f_plot, Hd_p)
    bw_s  = f'{bw:.1f}' if not np.isnan(bw) else '>50'

    cfg.update(dict(fs11=fs11, S11m=S11m, S11s=S11s, rms_s11=rms_s11,
                    fm=fm, pm=pm, Hd_m=Hd_m, Hd_p=Hd_p,
                    bw=bw, bw_s=bw_s, rms_H=rms_H))

    print(f'{cfg["lbl"]:>10} | {cfg["Lcpw"]*1e12:>6.1f}pH | {cfg["Lcpw2"]*1e12:>6.1f}pH | '
          f'{cfg["Lrp"]*1e12:>6.1f}pH | {rms_s11:>9.5f} | {bw_s:>7} | {rms_H:>7.3f}')

# ═══════════════════════════════════════════════════════════════════
# 9. PLOTTING
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

# ── Combined figure (Smith + |S11| dB + freq resp) per device ──────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    rf'UTC-PD 30 μm — Final  |  $C_j$={Cj*1e15:.0f} fF (fixed),  $L_{{R_p}}$(38Ω)=55.2 pH (fitted),  '
    rf'$\tau_A$={tau_A*1e12:.2f} ps' '\n'
    r'Row 1: Smith chart   |   Row 2: $|S_{11}|$ (dB)   |   Row 3: Frequency response',
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
    ax.set_xlabel('Frequency (GHz)', fontsize=9)
    ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)

    ax = axes[2, ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, cfg['Hd_p'], '-', color=col, lw=2.0,
            label=f'Sim  BW={cfg["bw_s"]} GHz  RMS={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9)
    ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('UTC_PD_final_fit.png', dpi=150, bbox_inches='tight')
print('\nSaved: UTC_PD_final_fit.png')

# ── Publication-style Smith chart 2×2 ──────────────────────────────
_pub_lbl = {'Rp=200Ω':'200 Ω', 'Rp=38Ω':'38 Ω', 'Rp=60Ω':'60 Ω', 'Open':'WO'}
_FS = 14
fig_sc, axes_sc = plt.subplots(2, 2, figsize=(10, 10))
fig_sc.patch.set_facecolor('white')
for ci, cfg in enumerate(configs):
    ax = axes_sc[ci//2][ci%2]; draw_smith(ax)
    col = cfg['col']
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=12, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S11s'].real, cfg['S11s'].imag, '--', color='k', lw=1.5, zorder=5, label='Fit')
    ax.text(-0.95, 0.92, 'Diameter: 30 μm', fontsize=_FS-1, va='top', ha='left')
    ax.text(-0.95, 0.72, f'Resistance: {_pub_lbl[cfg["lbl"]]}',
            fontsize=_FS, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.text(-0.95, 0.50, 'Bias: −7 V', fontsize=_FS-1, va='top', ha='left')
    ax.legend(fontsize=_FS-1, loc='lower left', framealpha=0.85,
              edgecolor='none', handlelength=1.5, markerscale=1.8)
fig_sc.tight_layout(pad=1.5)
fig_sc.savefig('UTC_PD_final_smith.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: UTC_PD_final_smith.png')

# ── Frequency response overlay ─────────────────────────────────────
fig_fr, ax_fr = plt.subplots(figsize=(10, 6))
for cfg in configs:
    col = cfg['col']; mk = cfg['mk']
    ax_fr.plot(cfg['fm']/1e9, cfg['pm'], mk, color=col, ms=4, alpha=0.6)
    ax_fr.plot(f_plot/1e9, cfg['Hd_p'], '-', color=col, lw=1.5,
               label=f'{cfg["lbl"]}  BW={cfg["bw_s"]} GHz')
ax_fr.axhline(-3, color='gray', ls='--', lw=0.8, label='-3 dB')
ax_fr.set_xlabel('Frequency (GHz)', fontsize=12)
ax_fr.set_ylabel('Normalized Response (dB)', fontsize=12)
ax_fr.set_xlim(0, 45); ax_fr.set_ylim(-12, 3)
ax_fr.set_title('Frequency Response: Measured vs Simulation (final params)', fontsize=12)
ax_fr.legend(fontsize=10, loc='lower left')
ax_fr.grid(True, alpha=0.3)
fig_fr.tight_layout()
fig_fr.savefig('UTC_PD_final_freqresp.png', dpi=150, bbox_inches='tight')
print('Saved: UTC_PD_final_freqresp.png')

# ═══════════════════════════════════════════════════════════════════
# 10. ORIGIN PRO EXPORT
# ═══════════════════════════════════════════════════════════════════
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'origin_data_final')
os.makedirs(_out, exist_ok=True)

def _write(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(f'{v:.8g}' for v in r) + '\n')

def _tag(lbl):
    return (lbl.replace('Ω','ohm').replace('=','_').replace(' ','_'))

for cfg in configs:
    t = _tag(cfg['lbl'])
    fs = cfg['fs11']
    S11m, S11s = cfg['S11m'], cfg['S11s']
    # Smith chart (freq + |S11|dB + phase + Re/Im)
    rows = list(zip(fs/1e9,
                    20*np.log10(np.abs(S11m)+1e-30), np.angle(S11m, deg=True),
                    20*np.log10(np.abs(S11s)+1e-30), np.angle(S11s, deg=True),
                    S11m.real, S11m.imag, S11s.real, S11s.imag))
    _write(os.path.join(_out, f'smith_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg',
            'Re_S11_meas','Im_S11_meas','Re_S11_fit','Im_S11_fit'], rows)
    # |S11| dB + phase (same content, separate file for convenience)
    _write(os.path.join(_out, f's11_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg'],
           list(zip(fs/1e9,
                    20*np.log10(np.abs(S11m)), np.angle(S11m, deg=True),
                    20*np.log10(np.abs(S11s)), np.angle(S11s, deg=True))))
    # Frequency response
    _write(os.path.join(_out, f'freqresp_meas_{t}.txt'),
           ['Freq_GHz','Norm_dB'], list(zip(cfg['fm']/1e9, cfg['pm'])))
    _write(os.path.join(_out, f'freqresp_sim_{t}.txt'),
           ['Freq_GHz','Norm_dB'], list(zip(f_plot/1e9, cfg['Hd_p'])))

# Summary table
sum_rows = []
for cfg in configs:
    sum_rows.append([cfg['lbl'],
                     cfg['Lcpw']*1e12, cfg['Lcpw2']*1e12, cfg['Lrp']*1e12,
                     Cj*1e15, Rs,
                     cfg['rms_s11'],
                     0 if np.isnan(cfg['bw']) else cfg['bw'],
                     cfg['rms_H']])
with open(os.path.join(_out, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write('Device\tL_CPW_pH\tL_CPW2_pH\tL_Rp_pH\tCj_fF\tRs_ohm\tRMS_S11\tBW_GHz\tRMS_H_dB\n')
    for r in sum_rows:
        f.write(f'{r[0]}\t' + '\t'.join(f'{v:.4f}' for v in r[1:]) + '\n')

print(f'\nOrigin data exported to: {_out}')
print('  smith_*.txt          — Smith chart (freq + dB + phase + Re/Im, meas & fit)')
print('  s11_*.txt            — |S11| dB + phase vs frequency')
print('  freqresp_meas_*.txt  — Measured normalized response')
print('  freqresp_sim_*.txt   — Simulated normalized response')
print('  summary.txt          — Per-device parameter & metric summary')
