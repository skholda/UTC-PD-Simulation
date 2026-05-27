"""
Simulation with Cj = 147 fF fixed for ALL devices
=================================================
Keep Rs, L_CPW, L_CPW2, L_Rp at original S11-fit values per device.
Cj = 147 fF fixed (no refit).
Show S11 + freq response for both tau_A values (7.86 / 3.53 ps).
"""
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt

# ── Load arrays from main.py ─────────────────────────────────────
_main = open(os.path.join(os.path.dirname(__file__), 'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz', 'ref_loss_dB',
            '_s1p_200', '_s1p_33', '_s1p_55', '_s1p_WO',
            '_freq_200', '_freq_33', '_freq_55', '_freq_WO']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)', _main, re.MULTILINE | re.DOTALL)
    exec(m.group(0), _ns)
ref_f_GHz, ref_loss_dB = _ns['ref_f_GHz'], _ns['ref_loss_dB']
S1P  = {'Rp=200Ω':_ns['_s1p_200'], 'Rp=38Ω':_ns['_s1p_33'],
        'Rp=60Ω': _ns['_s1p_55'],  'Open':  _ns['_s1p_WO']}
FREQ = {'Rp=200Ω':_ns['_freq_200'], 'Rp=38Ω':_ns['_freq_33'],
        'Rp=60Ω': _ns['_freq_55'],  'Open':  _ns['_freq_WO']}

# ── Transit-time model ───────────────────────────────────────────
W_A_undep=480e-9; W_A_dep=160e-9; W_cliff=50e-9; W_C=740e-9
W_tot=W_A_undep+W_A_dep+W_cliff+W_C
v_os_InP=4.0e5; v_os_InGaAs=2.1e5; v_h_InGaAs=4.5e4
tau_Ad=W_A_dep/v_os_InGaAs; tau_cl=W_cliff/v_os_InP
tau_C=W_C/v_os_InP; tau_h=W_A_dep/v_h_InGaAs

def H_ph_factory(tau_A):
    def H_ph(w):
        sinc=lambda x: np.sinc(x/np.pi)
        HA=1/(1+1j*w*tau_A)
        HAd=sinc(w*tau_Ad/2)*np.exp(-1j*(w*tau_A+w*tau_Ad/2))
        Hcl=sinc(w*tau_cl/2)*np.exp(-1j*(w*tau_A+w*tau_Ad+w*tau_cl/2))
        Hco=sinc(w*tau_C/2) *np.exp(-1j*(w*tau_A+w*tau_Ad+w*tau_cl+w*tau_C/2))
        HAd_h=sinc(w*tau_h/2)*np.exp(-1j*(w*tau_A+w*tau_h/2))
        return (W_A_undep*HA+W_A_dep*HAd+W_A_dep*HAd_h
                +W_cliff*Hcl+W_C*Hco)/(W_tot+W_A_dep)
    return H_ph

# ── Circuit ──────────────────────────────────────────────────────
C_CPW = 46.53e-15; R_L = 50.0
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp + 1j*w*Lrp)
def H_ckt(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Y_A = 1j*w*C_CPW + _Y_Rp(w,Rp,Lrp) + 1/(1j*w*Lcpw + R_L)
    return (R_L/(1j*w*Lcpw + R_L)) / (1j*w*Cpd + Y_A*(1 + 1j*w*Cpd*Zs))
def sim_S11(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Z_dev = Zs + 1/(1j*w*Cpd)
    Y_n = 1j*w*C_CPW + _Y_Rp(w,Rp,Lrp) + 1/Z_dev
    Z_in = 1j*w*Lcpw + 1/Y_n
    return (Z_in - 50)/(Z_in + 50)
def get_bw(f,Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Configs (fixed) ───────────────────────────────────────────────
Cj_FIXED = 147.0e-15
configs = [
    dict(lbl='Rp=200Ω', Rp=200,    col='#888888', mk='D',
         Rs=8.92, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=153e-12),
    dict(lbl='Rp=38Ω',  Rp=38,     col='#1B998B', mk='o',
         Rs=8.92, Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60,     col='#FF8C00', mk='s',
         Rs=8.92, Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf, col='#E91E8C', mk='^',
         Rs=8.92, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=0.0),
]

def gfr(arr):
    f_ghz=arr[:,0]
    cal=arr[:,1]+np.interp(f_ghz,ref_f_GHz,ref_loss_dB)
    return f_ghz*1e9, cal-cal[0]
def gs1p(arr,f_max):
    fr=arr[:,0]
    S11=10**(arr[:,1]/20)*np.exp(1j*np.deg2rad(arr[:,2]))
    mask=fr<=f_max
    return fr[mask],S11[mask]

TAU = [('τ_A=7.86 ps (orig)', 7.862e-12, '-'),
       ('τ_A=3.53 ps (paper)', 3.530e-12, '--')]

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot

print(f'\n{"="*98}')
print(f'  Cj = {Cj_FIXED*1e15:.0f} fF (FIXED for all devices)   |   '
      f'Rs = 8.92 Ω,  C_CPW = {C_CPW*1e15:.2f} fF')
print('='*98)
print(f'{"Device":>10} | {"L_CPW":>7} | {"L_CPW2":>7} | {"L_Rp":>6} | '
      f'{"RMS_S11":>9} | {"BW(7.86)":>9} | {"BW(3.53)":>9} | '
      f'{"RMS_H(7.86)":>11} | {"RMS_H(3.53)":>11}')
print('-'*98)

plot_data = []
for cfg in configs:
    fm,pm = gfr(FREQ[cfg['lbl']])
    f_max=fm.max()
    fs11,S11m = gs1p(S1P[cfg['lbl']],f_max)
    ws=2*np.pi*fs11; wm=2*np.pi*fm

    # S11 (Cj-independent of tau_A)
    S11s = sim_S11(ws,cfg['Rs'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],
                   Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S11s-S11m)**2))

    # Freq response for both tau_A
    Hckt_p = H_ckt(w_plot,cfg['Rs'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],
                   Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
    Hckt_m = H_ckt(wm,cfg['Rs'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],
                   Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
    rows={}
    for tag,tA,_ in TAU:
        Hph=H_ph_factory(tA)
        Hd_p = 20*np.log10(np.abs(Hph(w_plot)*Hckt_p)/np.abs(Hph(0)*Hckt_p[0]))
        Hd_m = 20*np.log10(np.abs(Hph(wm)*Hckt_m)/np.abs(Hph(wm[0])*Hckt_m[0]))
        bw=get_bw(f_plot,Hd_p)
        rms_H=np.sqrt(np.mean((Hd_m-pm)**2))
        rows[tag]=(bw,rms_H,Hd_p)

    bw1,rh1,_=rows['τ_A=7.86 ps (orig)']
    bw2,rh2,_=rows['τ_A=3.53 ps (paper)']
    s1=f'{bw1:.1f}' if not np.isnan(bw1) else '>50'
    s2=f'{bw2:.1f}' if not np.isnan(bw2) else '>50'
    print(f'{cfg["lbl"]:>10} | {cfg["Lcpw"]*1e12:>6.1f}pH | {cfg["Lcpw2"]*1e12:>6.1f}pH | '
          f'{cfg["Lrp"]*1e12:>5.1f}pH | {rms_s11:>9.5f} | {s1:>9} | {s2:>9} | '
          f'{rh1:>11.3f} | {rh2:>11.3f}')
    plot_data.append((cfg,fs11,S11m,S11s,rms_s11,fm,pm,rows))

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle(
    rf'Simulation with $C_j$ = {Cj_FIXED*1e15:.0f} fF fixed (all devices), others kept from original S11 fit' '\n'
    r'Row 1: S11  |  Row 2: Frequency response — solid: $\tau_A$=7.86 ps   dashed: $\tau_A$=3.53 ps',
    fontsize=11, fontweight='bold')

for ci,(cfg,fs11,S11m,S11s,rms_s11,fm,pm,rows) in enumerate(plot_data):
    col=cfg['col']; mk=cfg['mk']
    # S11
    ax=axes[0,ci]
    ax.plot(fs11/1e9,20*np.log10(np.abs(S11m)),'-',color=col,lw=1.0,label='Meas')
    ax.plot(fs11/1e9,20*np.log10(np.abs(S11s)),'--',color='navy',lw=1.5,
            label=f'Sim (Cj=147fF)  RMS={rms_s11:.4f}')
    ax.set_xlabel('Frequency (GHz)',fontsize=9)
    ax.set_ylabel('|S11| (dB)',fontsize=9)
    ax.set_title(cfg['lbl'],fontsize=10,fontweight='bold')
    ax.set_xlim(0,fs11.max()/1e9)
    ax.legend(fontsize=8.5,loc='lower right')
    ax.grid(True,alpha=0.3)
    # Freq response
    ax=axes[1,ci]
    ax.scatter(fm/1e9,pm,color='k',marker=mk,s=20,edgecolors=col,linewidths=1.0,
               zorder=5,label='Meas')
    for tag,tA,ls in TAU:
        bw,rh,Hd_p = rows[tag]
        bw_s=f'{bw:.1f}' if not np.isnan(bw) else '>50'
        clr = col if ls=='-' else 'navy'
        ax.plot(f_plot/1e9, Hd_p, ls, color=clr, lw=1.8,
                label=f'{tag}  BW={bw_s}, RMS={rh:.2f}dB')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)',fontsize=9)
    ax.set_ylabel('Normalized H (dB)',fontsize=9)
    ax.set_xlim(0,50); ax.set_ylim(-12,3)
    ax.legend(fontsize=8,loc='lower left')
    ax.grid(True,alpha=0.3)

fig.tight_layout()
out='Cj_147fF_fixed_sim.png'
fig.savefig(out,dpi=150,bbox_inches='tight')
print(f'\nSaved: {out}')
