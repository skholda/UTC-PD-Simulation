"""
Per-device (Cj, L_Rp) joint refit — 38 Ω and 60 Ω devices
==========================================================
S11-only cost.  Other params fixed at baseline:
  Rs = 8.92 Ω
  C_CPW = 46.53 fF
  L_CPW, L_CPW2 (per device, from earlier fit)
  Paper H_ph + v_os transit (τ_A=3.53, τ_C=2.88, τ_R=0.07 ps)

Free per device:
  Cj   (40-300 fF)
  L_Rp (0-300 pH)
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# ── Paper H_ph (locked) ─────────────────────────────────────────────
W_A=480e-9; W_C=980e-9; W_tot=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t=W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t=W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    return (abs_t+col_t)/(W_tot*(1+1j*w*tau_A))

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp+1j*w*Lrp)
def sim_S11(w,Cpd,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Z_dev=Zs+1/(1j*w*Cpd)
    Y_n=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Z_dev
    Z_in=1j*w*Lcpw+1/Y_n
    return (Z_in-50)/(Z_in+50)
def H_ckt(w,Cpd,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cpd+Y_A*(1+1j*w*Cpd*Zs))
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Data ────────────────────────────────────────────────────────────
_main=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'main.py')).read()
_ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_s1p_33','_s1p_55','_freq_33','_freq_55']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz,ref_loss_dB=_ns['ref_f_GHz'],_ns['ref_loss_dB']

configs=[
    dict(lbl='Rp=38Ω', Rp=38.0, col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'], freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12,
         Cj_base=141.1e-15, Lrp_base=43.4e-12),
    dict(lbl='Rp=60Ω', Rp=60.0, col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'], freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12,
         Cj_base=141.1e-15, Lrp_base=44.8e-12),
]
def gs1p(a,fm): fr=a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); m=fr<=fm; return fr[m],S[m]
def gfr(a):    f=a[:,0]; c=a[:,1]+np.interp(f,ref_f_GHz,ref_loss_dB); return f*1e9,c-c[0]
for cfg in configs:
    fm,pm=gfr(cfg['freq']); fs11,S11m=gs1p(cfg['s1p'],fm.max())
    cfg.update(fm=fm,pm=pm,wm=2*np.pi*fm,fs11=fs11,S11m=S11m,ws=2*np.pi*fs11)

# ── Joint (Cj, L_Rp) fit per device ─────────────────────────────────
print('='*100)
print(f'Joint (Cj, L_Rp) fit, S11-only  |  others fixed (paper H_ph + v_os)')
print('='*100)
print(f'{"Device":>10} | {"Cj (fF)":>9} | {"L_Rp (pH)":>10} | {"RMS_S11":>9} | '
      f'{"baseline RMS":>12} | {"BW (GHz)":>9} | {"RMS_H (dB)":>11}')
print('-'*100)

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot
for cfg in configs:
    def cost(p):
        Cj,Lrp=p
        S=sim_S11(cfg['ws'],Cj,cfg['Rp'],cfg['Lcpw'],Lrp,cfg['Lcpw2'])
        return np.mean(np.abs(S-cfg['S11m'])**2)

    bounds=[(40e-15, 300e-15),(0, 300e-12)]
    res = differential_evolution(cost, bounds, seed=42, maxiter=3000,
                                 tol=1e-16, polish=False)
    pol = minimize(cost, res.x, method='L-BFGS-B', bounds=bounds,
                   options={'ftol':1e-20,'gtol':1e-16,'maxiter':50000})
    Cj_fit, Lrp_fit = pol.x
    rms_fit = np.sqrt(pol.fun)

    # Baseline comparison
    S_base = sim_S11(cfg['ws'],cfg['Cj_base'],cfg['Rp'],cfg['Lcpw'],
                     cfg['Lrp_base'],cfg['Lcpw2'])
    rms_base = np.sqrt(np.mean(np.abs(S_base-cfg['S11m'])**2))

    S_fit = sim_S11(cfg['ws'],Cj_fit,cfg['Rp'],cfg['Lcpw'],Lrp_fit,cfg['Lcpw2'])
    Hckt_m=H_ckt(cfg['wm'],Cj_fit,cfg['Rp'],cfg['Lcpw'],Lrp_fit,cfg['Lcpw2'])
    Hckt_p=H_ckt(w_plot,   Cj_fit,cfg['Rp'],cfg['Lcpw'],Lrp_fit,cfg['Lcpw2'])
    Hd_m=20*np.log10(np.abs(H_ph(cfg['wm'])*Hckt_m)/np.abs(H_ph(cfg['wm'][0])*Hckt_m[0]))
    Hd_p=20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m-cfg['pm'])**2))
    bw    = get_bw(f_plot,Hd_p)

    Hckt_m_b=H_ckt(cfg['wm'],cfg['Cj_base'],cfg['Rp'],cfg['Lcpw'],cfg['Lrp_base'],cfg['Lcpw2'])
    Hckt_p_b=H_ckt(w_plot,   cfg['Cj_base'],cfg['Rp'],cfg['Lcpw'],cfg['Lrp_base'],cfg['Lcpw2'])
    Hd_p_b=20*np.log10(np.abs(H_ph(w_plot)*Hckt_p_b)/np.abs(H_ph(0)*Hckt_p_b[0]))
    bw_b = get_bw(f_plot,Hd_p_b)

    cfg.update(Cj_fit=Cj_fit, Lrp_fit=Lrp_fit, rms_fit=rms_fit,
               S_fit=S_fit, S_base=S_base, rms_base=rms_base,
               Hd_p=Hd_p, Hd_p_b=Hd_p_b, bw=bw, bw_b=bw_b, rms_H=rms_H)

    print(f'{cfg["lbl"]:>10} | {Cj_fit*1e15:>8.2f} | {Lrp_fit*1e12:>9.2f} | '
          f'{rms_fit:>9.5f} | {rms_base:>12.5f} | {bw:>9.2f} | {rms_H:>11.3f}')

# ── Plot ────────────────────────────────────────────────────────────
def draw_smith(ax,lw=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw+0.3))
    ax.axhline(0,color='#888',lw=lw,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#aaa',lw=lw,ls=':',zorder=0))
    theta=np.linspace(0,np.pi,400)
    for x in [0.2,0.5,1,2,5]:
        for sgn in [1,-1]:
            cx,cy,rad=1,sgn/x,1/x
            xx=cx+rad*np.cos(theta); yy=cy+rad*np.sin(theta)*sgn
            mm=(xx**2+yy**2<=1.002)
            ax.plot(xx[mm],yy[mm],color='#aaa',lw=lw,ls=':',zorder=0)

fig, axes = plt.subplots(3, 2, figsize=(13, 14))
fig.suptitle(
    r'Joint (Cj, L_Rp) refit — 38 Ω and 60 Ω, S11-only' '\n'
    r'gray dotted: baseline (Cj=141.1 fF) | red dashed: fit'  '\n'
    r'Row 1: Smith   |   Row 2: $|S_{11}|$ dB   |   Row 3: Frequency response',
    fontsize=11, fontweight='bold')

for ci, cfg in enumerate(configs):
    col=cfg['col']; mk=cfg['mk']

    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S_base'].real, cfg['S_base'].imag, ':', color='gray', lw=1.3, zorder=4,
            label=f'baseline RMS={cfg["rms_base"]:.5f}')
    ax.plot(cfg['S_fit'].real, cfg['S_fit'].imag, '--', color='red', lw=1.5, zorder=5,
            label=f'fit Cj={cfg["Cj_fit"]*1e15:.1f}fF L_Rp={cfg["Lrp_fit"]*1e12:.1f}pH\nRMS={cfg["rms_fit"]:.5f}')
    ax.set_title(f'{cfg["lbl"]}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    ax=axes[1,ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-', color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S_base'])), ':', color='gray', lw=1.3, label='baseline')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S_fit'])), '--', color='red', lw=1.5, label='fit')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)

    ax=axes[2,ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, cfg['Hd_p_b'], ':', color='gray', lw=1.3,
            label=f'baseline BW={cfg["bw_b"]:.1f}GHz')
    ax.plot(f_plot/1e9, cfg['Hd_p'], '--', color='red', lw=1.5,
            label=f'fit BW={cfg["bw"]:.1f}GHz, RMS_H={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized H (dB)')
    ax.set_xlim(0,50); ax.set_ylim(-12,3)
    ax.legend(fontsize=9, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
out='CjLRp_refit_38_60.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
