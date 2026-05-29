"""
Evaluate with FEM L_Rp values (no fitting)
==========================================
All parameters fixed from the user table:
  C_PD  = 131.0 fF (common)
  R_s   = 8.92 ohm
  C_CPW = 46.53 fF
  L_CPW : per-device
  L_CPW2: per-device
  L_Rp  : MATLAB FEM values (not fitted)

Paper H_ph baseline locked: τ_A = 3.53, τ_C = 2.88, τ_R = 0.07 ps
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt

# ── Paper H_ph (locked) ─────────────────────────────────────────────
W_A=480e-9; W_C=980e-9; W_tot=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t=W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t=W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    return (abs_t+col_t)/(W_tot*(1+1j*w*tau_A))

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92; Cj=131.0e-15
def _Y_Rp(w,Rp,Lrp):
    return 1.0/(Rp+1j*w*Lrp) if not np.isinf(Rp) else 0.0
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
for v in ['ref_f_GHz','ref_loss_dB','_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
          '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz,ref_loss_dB=_ns['ref_f_GHz'],_ns['ref_loss_dB']

# ── Configs from user table ─────────────────────────────────────────
configs=[
    dict(lbl='Rp=200Ω', Rp=200.0,   col='#888888', mk='D',
         s1p=_ns['_s1p_200'], freq=_ns['_freq_200'],
         Lcpw=178.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),   # FEM
    dict(lbl='Rp=38Ω',  Rp=38.0,    col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'],  freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=65.6e-12),    # FEM
    dict(lbl='Rp=60Ω',  Rp=60.0,    col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'],  freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),    # FEM
    dict(lbl='Open',    Rp=np.inf,  col='#E91E8C', mk='^',
         s1p=_ns['_s1p_WO'],  freq=_ns['_freq_WO'],
         Lcpw=178.9e-12, Lcpw2=0.0,     Lrp=0.0),
]
def gs1p(a,fm): fr=a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); m=fr<=fm; return fr[m],S[m]
def gfr(a):    f=a[:,0]; c=a[:,1]+np.interp(f,ref_f_GHz,ref_loss_dB); return f*1e9,c-c[0]

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot

print('='*110)
print('Evaluation with FEM L_Rp (no fitting, all params from user table)')
print('Cj=131.0 fF, Rs=8.92 Ω, C_CPW=46.53 fF  |  τ_A=3.53, τ_C=2.88, τ_R=0.07 ps')
print('='*110)
print(f'{"Device":>10} | {"L_CPW":>7} | {"L_CPW2":>7} | {"L_Rp":>8} | '
      f'{"RMS_S11":>9} | {"BW (GHz)":>9} | {"RMS_H":>7}')
print('-'*110)

for cfg in configs:
    fm,pm=gfr(cfg['freq']); wm=2*np.pi*fm
    fs11,S11m=gs1p(cfg['s1p'],fm.max()); ws=2*np.pi*fs11

    S11s = sim_S11(ws,Cj,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S11s - S11m)**2))

    Hckt_m=H_ckt(wm,Cj,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hckt_p=H_ckt(w_plot,Cj,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hd_m=20*np.log10(np.abs(H_ph(wm)*Hckt_m)/np.abs(H_ph(wm[0])*Hckt_m[0]))
    Hd_p=20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H=np.sqrt(np.mean((Hd_m-pm)**2))
    bw=get_bw(f_plot,Hd_p)
    bw_s=f'{bw:.1f}' if not np.isnan(bw) else '>50'

    cfg.update(fs11=fs11,S11m=S11m,S11s=S11s,rms_s11=rms_s11,
               fm=fm,pm=pm,Hd_p=Hd_p,bw=bw,bw_s=bw_s,rms_H=rms_H)

    print(f'{cfg["lbl"]:>10} | {cfg["Lcpw"]*1e12:>6.1f}pH | {cfg["Lcpw2"]*1e12:>6.1f}pH | '
          f'{cfg["Lrp"]*1e12:>7.1f}pH | {rms_s11:>9.5f} | {bw_s:>9} | {rms_H:>7.3f}')

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

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    r'Evaluation with FEM $L_{R_p}$ values (no fitting)' '\n'
    r'$C_j$=131 fF, $R_s$=8.92 Ω, $C_{CPW}$=46.53 fF | $L_{R_p}$ from MATLAB FEM table' '\n'
    r'Row 1: Smith   Row 2: $|S_{11}|$ dB   Row 3: Freq response',
    fontsize=11, fontweight='bold')

for ci, cfg in enumerate(configs):
    col=cfg['col']; mk=cfg['mk']
    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real,cfg['S11m'].imag,s=10,color=col,zorder=6,label='Meas.')
    ax.plot(cfg['S11s'].real,cfg['S11s'].imag,'--',color='k',lw=1.5,zorder=5,label='Sim')
    ax.set_title(f'{cfg["lbl"]}\nL_Rp={cfg["Lrp"]*1e12:.1f}pH (FEM)\nRMS|ΔΓ|={cfg["rms_s11"]:.4f}',
                 fontsize=10,fontweight='bold')
    ax.legend(fontsize=8.5,loc='lower left',framealpha=0.85,edgecolor='none',
              handlelength=1.5,markerscale=1.5)

    ax=axes[1,ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-', color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11s'])), '--', color='k',  lw=1.4, label='Sim')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)

    ax=axes[2,ci]
    ax.scatter(cfg['fm']/1e9,cfg['pm'],color='k',marker=mk,s=22,
               edgecolors=col,linewidths=1.0,zorder=5,label='Meas.')
    ax.plot(f_plot/1e9,cfg['Hd_p'],'-',color=col,lw=2.0,
            label=f'Sim BW={cfg["bw_s"]} GHz, RMS={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized H (dB)')
    ax.set_xlim(0,50); ax.set_ylim(-12,3)
    ax.legend(fontsize=8.5,loc='lower left'); ax.grid(True,alpha=0.3)

fig.tight_layout()
out='eval_FEM_LRp.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
