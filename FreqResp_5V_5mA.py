import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ── Locked Option B baseline (fitted at -7 V) ──────────────────────
W_A=480e-9; W_C=980e-9; W=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t=W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t=W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    return (abs_t+col_t)/(W*(1+1j*w*tau_A))

C_CPW=46.53e-15; R_L=50.0; Rs=8.92; Cj=131.0e-15
def _Y_Rp(w,Rp,Lrp):
    return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def H_ckt(w,Cpd,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cpd+Y_A*(1+1j*w*Cpd*Zs))
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

configs=[
    dict(lbl='Rp=200Ω', Rp=200.0, file='data_5V_5mA/200ohm.xlsx', col='#888888', mk='D',
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,  file='data_5V_5mA/38ohm.xlsx',  col='#1B998B', mk='o',
         Lcpw=141.6e-12, Lcpw2=56.3e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60.0,  file='data_5V_5mA/60ohm.xlsx',  col='#FF8C00', mk='s',
         Lcpw=149.9e-12, Lcpw2=48.0e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf,file='data_5V_5mA/WO.xlsx',     col='#E91E8C', mk='^',
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=0.0),
]

def load_5V(path):
    df = pd.read_excel(path, header=14)
    f  = pd.to_numeric(df.iloc[:,0], errors='coerce')
    cal= pd.to_numeric(df.iloc[:,6], errors='coerce')   # Cal RF POW (dBm)
    m  = f.notna() & cal.notna()
    return f[m].values*1e9, cal[m].values

f_plot=np.linspace(0.1e9,45e9,4000); w_plot=2*np.pi*f_plot

print('='*95)
print('5 V / 5 mA measured freq response  vs  locked baseline model (fitted at -7 V S11)')
print('='*95)
print(f'{"Device":>10} | {"N pts":>5} | {"f range":>13} | {"BW model":>9} | {"RMS_H (dB)":>10}')
print('-'*70)

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
fig.suptitle('Frequency Response @ Bias = -5 V, I_ph = 5 mA\n'
             'scatter: measured (Cal RF, normalized) | line: locked baseline model (-7 V S11 fit)',
             fontsize=12, fontweight='bold')

overlay_fig, oax = plt.subplots(figsize=(10, 6.5))
export = {}

for ci, cfg in enumerate(configs):
    fm, cal = load_5V(cfg['file'])
    pm = cal - cal[0]                      # normalize to first point
    wm = 2*np.pi*fm

    Hckt_m = H_ckt(wm,    Cj,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot,Cj,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hd_m = 20*np.log10(np.abs(H_ph(wm)*Hckt_m)/np.abs(H_ph(wm[0])*Hckt_m[0]))
    Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m-pm)**2))
    bw = get_bw(f_plot,Hd_p)
    print(f'{cfg["lbl"]:>10} | {len(fm):>5} | {fm[0]/1e9:>4.1f}-{fm[-1]/1e9:>5.1f} GHz | '
          f'{bw:>7.1f}G | {rms_H:>10.3f}')

    ax = axes[ci//2][ci%2]
    ax.scatter(fm/1e9, pm, color='k', marker=cfg['mk'], s=24,
               edgecolors=cfg['col'], linewidths=1.1, zorder=5, label='Meas (-5 V, 5 mA)')
    ax.plot(f_plot/1e9, Hd_p, '-', color=cfg['col'], lw=2.0,
            label=f'Model (-7 V fit)  BW={bw:.1f} GHz, RMS={rms_H:.2f} dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized Response (dB)')
    ax.set_title(cfg['lbl'], fontsize=11, fontweight='bold')
    ax.set_xlim(0, 45); ax.set_ylim(-15, 3)
    ax.legend(fontsize=9, loc='lower left'); ax.grid(True, alpha=0.3)

    oax.plot(fm/1e9, pm, cfg['mk'], color=cfg['col'], ms=4.5, alpha=0.65)
    oax.plot(f_plot/1e9, Hd_p, '-', color=cfg['col'], lw=1.6,
             label=f'{cfg["lbl"]}  BW={bw:.1f} GHz, RMS={rms_H:.2f} dB')
    export[cfg['lbl']] = (fm, pm, Hd_m, bw, rms_H)

fig.tight_layout()
fig.savefig('FreqResp_5V_5mA_vs_model.png', dpi=150, bbox_inches='tight')
print('\nSaved: FreqResp_5V_5mA_vs_model.png')

oax.axhline(-3, color='gray', ls='--', lw=0.8, label='-3 dB')
oax.set_xlabel('Frequency (GHz)', fontsize=12)
oax.set_ylabel('Normalized Response (dB)', fontsize=12)
oax.set_title('Freq Response @ -5 V / 5 mA vs locked model (-7 V S11 fit)', fontsize=12)
oax.set_xlim(0, 45); oax.set_ylim(-15, 3)
oax.legend(fontsize=9.5, loc='lower left'); oax.grid(True, alpha=0.3)
overlay_fig.tight_layout()
overlay_fig.savefig('FreqResp_5V_5mA_overlay.png', dpi=150, bbox_inches='tight')
print('Saved: FreqResp_5V_5mA_overlay.png')

# export data
import os
os.makedirs('origin_data_5V_5mA', exist_ok=True)
for lbl,(fm,pm,Hd_m,bw,rms) in export.items():
    t = lbl.replace('Ω','ohm').replace('=','_')
    with open(f'origin_data_5V_5mA/freqresp_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tNorm_dB_meas\tNorm_dB_model\n')
        for i in range(len(fm)):
            fp.write(f'{fm[i]/1e9:.6g}\t{pm[i]:.6g}\t{Hd_m[i]:.6g}\n')
print('Saved: origin_data_5V_5mA/freqresp_*.txt')
