"""
UTC-PD 30 μm  —  -5 V / 5 mA simulation  (rigorous H_ph baseline)
================================================================
Same locked framework as UTC_PD_final.py (-7 V), with the ONLY bias-dependent
change being the junction capacitance (C-V):
    C_PD(-7 V) = 131 fF   ->   C_PD(-5 V) = 161 fF   (C-V values, LOCKED)
Inductances are geometric (bias-independent): 1-L L_CPW and FEM L_m reused
from the -7 V fit AS-IS. Rs, C_CPW bias-independent.

H_ph: identical rigorous Ramo J_tot integral as the -7 V baseline
      (undep-abs electron carries τ_A; dep-abs in-situ e/h carry none).

Data:
    S11            : data_5V_5mA/S11_*.s1p    (measured at -5 V, to 40 GHz)
    Freq response  : data_5V_5mA/*.xlsx       (measured at -5 V / 5 mA, Cal RF col)
Outputs: S11/Smith/freq-response figures + origin_data_5V_5mA/ export tree.
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. H_ph(ω) — 4-term transit (SAME as -7 V baseline, bias-independent)
# ═══════════════════════════════════════════════════════════════════
W_A = 480e-9; W_Ad = 160e-9; W_C = 820e-9
W_norm = W_A + W_C + 2*W_Ad                       # 1620 nm  (DC -> |H_ph(0)|=1)
tau_A = 1.989e-12; tau_R = 0.0                    # staircase tau_A; tau_R neglected
# material-resolved layer-average v(E_avg), device field (-7V,0.5mA,lat10um); τ_A kept:
tau_eD = 2.026e-12                                # dep InGaAs abs 160 nm (InGaAs)
tau_C  = 7.794e-12                                # grading(InGaAsP)+cliff(InP)+collector(InP)
tau_h  = W_Ad / 4.8e4                             # 3.333 ps (InGaAs h sat 0.48e7 cm/s, lit.)

def H_ph(w):
    # 4-term transit: τ_A on undep-absorber terms only; in-situ dep-abs e/h no τ_A.
    sinc = lambda x: np.sinc(x/np.pi)
    t1 = W_A /(1.0+1j*w*tau_A) * (2.0+1j*w*tau_R)/(2.0*(1.0+1j*w*tau_R))
    t2 = W_C /(1.0+1j*w*tau_A) * sinc(w*tau_C/2)  * np.exp(-1j*w*tau_C/2)
    t3 = W_Ad * sinc(w*tau_eD/2) * np.exp(-1j*w*tau_eD/2)
    t4 = W_Ad * sinc(w*tau_h/2)  * np.exp(-1j*w*tau_h/2)
    return (t1 + t2 + t3 + t4) / W_norm

# ═══════════════════════════════════════════════════════════════════
# 2. CIRCUIT — 1-L LADDER (single CPW inductance; L_CPW1=0); bias-independent
#    params locked; only C_PD changes with bias: 131 fF @ -7 V -> 161 fF @ -5 V
#   Iph ∥ C_PD ─[R_S]─ node1{ C_CPW ∥ (R_m+L_m) } ─[L_CPW]─ port
# ═══════════════════════════════════════════════════════════════════
C_CPW = 46.53e-15; R_L = 50.0; Rs = 8.92
Cj    = 161.0e-15                                 # C_PD from C-V @ -5 V (LOCKED)

def sim_S11(w, Rp, Lcpw1, Lrp, Lcpw2):
    Z1  = Rs + 1/(1j*w*Cj)
    Y1  = 1j*w*C_CPW + 1/Z1
    Z2  = 1j*w*Lcpw1 + 1/Y1
    Yrm = 0.0 if np.isinf(Rp) else 1/(Rp + 1j*w*Lrp)
    Y2  = Yrm + 1/Z2
    Z_in = 1j*w*Lcpw2 + 1/Y2
    return (Z_in - 50)/(Z_in + 50)
def H_ckt(w, Rp, Lcpw1, Lrp, Lcpw2):
    """Transimpedance V_RL/I_ph of the ladder via ABCD cascade."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    A = np.ones_like(w, dtype=complex); B = np.zeros_like(w, dtype=complex)
    C = np.zeros_like(w, dtype=complex); D = np.ones_like(w, dtype=complex)
    def _series(Z):
        nonlocal A, B, C, D
        A, B, C, D = A, A*Z + B, C, C*Z + D
    def _shunt(Y):
        nonlocal A, B, C, D
        A, B, C, D = A + B*Y, B, C + D*Y, D
    _shunt(1j*w*Cj)
    _series(Rs + 0j*w)
    _shunt(1j*w*C_CPW)
    _series(1j*w*Lcpw1)
    if not np.isinf(Rp):
        _shunt(1/(Rp + 1j*w*Lrp))
    _series(1j*w*Lcpw2)
    return R_L/(C*R_L + D)
def get_bw(f, Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ═══════════════════════════════════════════════════════════════════
# 3. DEVICE CONFIGS — ladder L values reused from -7 V fit (locked,
#    bias-independent);  L_m = FEM values (not fitted)
# ═══════════════════════════════════════════════════════════════════
configs = [
    dict(lbl='Rp=200Ω', Rp=200.0,  col='#888888', mk='D',
         s1p='data_5V_5mA/S11_200ohm.s1p', fr='data_5V_5mA/200ohm.xlsx',
         Lcpw=0.0, Lcpw2=200.4e-12, Lrp=153.7e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,   col='#1B998B', mk='o',
         s1p='data_5V_5mA/S11_38ohm.s1p',  fr='data_5V_5mA/38ohm.xlsx',
         Lcpw=0.0, Lcpw2=158.5e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60.0,   col='#FF8C00', mk='s',
         s1p='data_5V_5mA/S11_60ohm.s1p',  fr='data_5V_5mA/60ohm.xlsx',
         Lcpw=0.0, Lcpw2=172.5e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf, col='#E91E8C', mk='^',
         s1p='data_5V_5mA/S11_WO.s1p',     fr='data_5V_5mA/WO.xlsx',
         Lcpw=0.0, Lcpw2=190.0e-12, Lrp=0.0),
]

def load_s1p(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('!', '#')): continue
            p = line.split()
            if len(p) >= 3: rows.append([float(p[0]), float(p[1]), float(p[2])])
    a = np.array(rows)
    return a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))   # Hz, complex
def load_fr(path):
    df = pd.read_excel(path, header=14)
    f = pd.to_numeric(df.iloc[:,0], errors='coerce')
    cal = pd.to_numeric(df.iloc[:,6], errors='coerce')            # Cal RF (dBm)
    m = f.notna() & cal.notna()
    f, cal = f[m].values*1e9, cal[m].values
    return f, cal - cal[0]

f_plot = np.linspace(0.1e9, 50e9, 5000); w_plot = 2*np.pi*f_plot

print('='*100)
print('UTC-PD 30 μm  —  -5 V / 5 mA  (ladder circuit, C_PD=161 fF C-V, L locked from -7 V)')
print('-'*100)
_wtr = 2*np.pi*np.linspace(1e9,200e9,400000)
_mtr = np.abs(H_ph(_wtr))/np.abs(H_ph(1e-3*2*np.pi))
_itr = np.where(_mtr <= 1/np.sqrt(2))[0]
print(f'  transit-limited f_tr = {_wtr[_itr[0]]/2/np.pi/1e9:.2f} GHz  (bias-independent H_ph)')
print(f'  Cj={Cj*1e15:.1f} fF   Rs={Rs} Ω   C_CPW={C_CPW*1e15:.2f} fF')
print('='*100)
print(f'{"Device":>10} | {"RMS_S11":>9} | {"BW model":>9} | {"RMS_H (dB)":>10}')
print('-'*100)

for cfg in configs:
    fs11, S11m = load_s1p(cfg['s1p']); ws = 2*np.pi*fs11
    S11s = sim_S11(ws, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S11s - S11m)**2))

    fm, pm = load_fr(cfg['fr']); wm = 2*np.pi*fm
    Hckt_m = H_ckt(wm,     cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    Hd_m = 20*np.log10(np.abs(H_ph(wm)*Hckt_m)/np.abs(H_ph(wm[0])*Hckt_m[0]))
    Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m - pm)**2))
    bw = get_bw(f_plot, Hd_p); bw_s = f'{bw:.1f}' if not np.isnan(bw) else '>50'

    cfg.update(fs11=fs11, S11m=S11m, S11s=S11s, rms_s11=rms_s11,
               fm=fm, pm=pm, Hd_m=Hd_m, Hd_p=Hd_p, bw=bw, bw_s=bw_s, rms_H=rms_H)
    print(f'{cfg["lbl"]:>10} | {rms_s11:>9.5f} | {bw_s:>7} G | {rms_H:>10.3f}')

# ═══════════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════════
def draw_smith(ax, lw=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw+0.3)); ax.axhline(0,color='#888',lw=lw,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1); ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#aaa',lw=lw,ls=':',zorder=0))
    th=np.linspace(0,np.pi,400)
    for x in [0.2,0.5,1,2,5]:
        for s in [1,-1]:
            xx=1+(1/x)*np.cos(th); yy=s/x+(1/x)*np.sin(th)*s; m=(xx**2+yy**2<=1.002)
            ax.plot(xx[m],yy[m],color='#aaa',lw=lw,ls=':',zorder=0)

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('UTC-PD 30 μm  —  Bias = -5 V, I_ph = 5 mA  (ladder circuit, $C_{PD}$=172 fF, L locked from -7 V)\n'
             'Row 1: Smith  |  Row 2: $|S_{11}|$ dB  |  Row 3: Frequency response',
             fontsize=11, fontweight='bold')
for ci, cfg in enumerate(configs):
    col, mk = cfg['col'], cfg['mk']
    ax = axes[0, ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S11s'].real, cfg['S11s'].imag, '--', color='k', lw=1.5, zorder=5, label='Sim')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|={cfg["rms_s11"]:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85, edgecolor='none')
    ax = axes[1, ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-', color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11s'])), '--', color='k', lw=1.4, label='Sim')
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_xlim(0, cfg['fs11'].max()/1e9); ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)
    ax = axes[2, ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22, edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, cfg['Hd_p'], '-', color=col, lw=2.0, label=f'Sim  BW={cfg["bw_s"]} GHz  RMS={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3); ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('UTC_PD_5V_fit.png', dpi=150, bbox_inches='tight')
print('\nSaved: UTC_PD_5V_fit.png')

# freq-response overlay
fig_fr, ax_fr = plt.subplots(figsize=(10, 6))
for cfg in configs:
    ax_fr.plot(cfg['fm']/1e9, cfg['pm'], cfg['mk'], color=cfg['col'], ms=4, alpha=0.6)
    ax_fr.plot(f_plot/1e9, cfg['Hd_p'], '-', color=cfg['col'], lw=1.5, label=f'{cfg["lbl"]}  BW={cfg["bw_s"]} GHz')
ax_fr.axhline(-3, color='gray', ls='--', lw=0.8, label='-3 dB')
ax_fr.set_xlabel('Frequency (GHz)', fontsize=12); ax_fr.set_ylabel('Normalized Response (dB)', fontsize=12)
ax_fr.set_xlim(0, 45); ax_fr.set_ylim(-12, 3)
ax_fr.set_title('Frequency Response @ -5 V / 5 mA  (rigorous $H_{ph}$)', fontsize=12)
ax_fr.legend(fontsize=10, loc='lower left'); ax_fr.grid(True, alpha=0.3)
fig_fr.tight_layout(); fig_fr.savefig('UTC_PD_5V_freqresp.png', dpi=150, bbox_inches='tight')
print('Saved: UTC_PD_5V_freqresp.png')

# ═══════════════════════════════════════════════════════════════════
# 5. ORIGIN EXPORT
# ═══════════════════════════════════════════════════════════════════
_out = 'origin_data_5V_5mA'; os.makedirs(_out, exist_ok=True)
def _write(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows: f.write('\t'.join(f'{v:.8g}' for v in r) + '\n')
def _tag(l): return l.replace('Ω','ohm').replace('=','_').replace(' ','_')
for cfg in configs:
    t = _tag(cfg['lbl']); fs = cfg['fs11']; S11m, S11s = cfg['S11m'], cfg['S11s']
    _write(os.path.join(_out, f'smith_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg',
            'Re_S11_meas','Im_S11_meas','Re_S11_fit','Im_S11_fit'],
           list(zip(fs/1e9, 20*np.log10(np.abs(S11m)+1e-30), np.angle(S11m,deg=True),
                    20*np.log10(np.abs(S11s)+1e-30), np.angle(S11s,deg=True),
                    S11m.real, S11m.imag, S11s.real, S11s.imag)))
    _write(os.path.join(_out, f's11_{t}.txt'),
           ['Freq_GHz','S11_meas_dB','S11_meas_deg','S11_fit_dB','S11_fit_deg'],
           list(zip(fs/1e9, 20*np.log10(np.abs(S11m)), np.angle(S11m,deg=True),
                    20*np.log10(np.abs(S11s)), np.angle(S11s,deg=True))))
    _write(os.path.join(_out, f'freqresp_meas_{t}.txt'), ['Freq_GHz','Norm_dB'], list(zip(cfg['fm']/1e9, cfg['pm'])))
    _write(os.path.join(_out, f'freqresp_sim_{t}.txt'),  ['Freq_GHz','Norm_dB'], list(zip(f_plot/1e9, cfg['Hd_p'])))
with open(os.path.join(_out, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write('# -5 V / 5 mA  (ladder circuit, C_PD=161 fF C-V, L locked from -7 V)\n')
    f.write(f'# transit-limited f_tr = {_wtr[_itr[0]]/2/np.pi/1e9:.2f} GHz (bias-independent)\n')
    f.write('Device\tCj_fF\tL_CPW_pH\tL_CPW2_pH\tL_Rp_pH\tRMS_S11\tBW_GHz\tRMS_H_dB\n')
    for cfg in configs:
        bw_v = 0 if np.isnan(cfg['bw']) else cfg['bw']
        f.write(f'{cfg["lbl"]}\t{Cj*1e15:.2f}\t{cfg["Lcpw"]*1e12:.4f}\t{cfg["Lcpw2"]*1e12:.4f}\t'
                f'{cfg["Lrp"]*1e12:.4f}\t{cfg["rms_s11"]:.5f}\t{bw_v:.4f}\t{cfg["rms_H"]:.4f}\n')
print(f'\nOrigin export to: {_out}/')
