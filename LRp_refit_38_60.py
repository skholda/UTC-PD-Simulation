"""
Aggressive S11-only L_Rp refit — 38 Ω and 60 Ω devices
=======================================================
Compares three approaches:
  (1) bounded minimize_scalar (current baseline)
  (2) wider bounds (0 to 500 pH)
  (3) global differential_evolution + L-BFGS-B polish

Also plots RMS_S11 vs L_Rp landscape to confirm we're at the global minimum.
All other circuit + transit-time params fixed.
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, differential_evolution, minimize

# ── Paper H_ph (locked baseline) ────────────────────────────────────
W_A_paper=480e-9; W_C_paper=980e-9; W_paper=W_A_paper+W_C_paper
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t = W_A_paper*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t = W_C_paper*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    return (abs_t+col_t)/(W_paper*(1+1j*w*tau_A))

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92
Cj_FIXED=141.1e-15
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp+1j*w*Lrp)
def sim_S11(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
    Zs=Rs+1j*w*Lcpw2
    Z_dev=Zs+1/(1j*w*Cpd)
    Y_n=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Z_dev
    Z_in=1j*w*Lcpw+1/Y_n
    return (Z_in-50)/(Z_in+50)
def H_ckt(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
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
         Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp_baseline=43.4e-12),
    dict(lbl='Rp=60Ω', Rp=60.0, col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'], freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp_baseline=44.8e-12),
]
def gs1p(a,fm): fr=a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); m=fr<=fm; return fr[m],S[m]
def gfr(a):    f=a[:,0]; c=a[:,1]+np.interp(f,ref_f_GHz,ref_loss_dB); return f*1e9,c-c[0]

for cfg in configs:
    fm,pm=gfr(cfg['freq']); fs11,S11m=gs1p(cfg['s1p'],fm.max())
    cfg.update(fm=fm,pm=pm,wm=2*np.pi*fm,fs11=fs11,S11m=S11m,ws=2*np.pi*fs11)

def cost(Lrp,cfg):
    S=sim_S11(cfg['ws'],Rs,Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp=Lrp,Lcpw2=cfg['Lcpw2'])
    return np.mean(np.abs(S-cfg['S11m'])**2)

# ── Three fitting approaches ───────────────────────────────────────
f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot
print('='*100)
print(f'S11-only L_Rp refit  |  Cj={Cj_FIXED*1e15:.2f} fF, Rs={Rs} Ω fixed')
print('='*100)

LRp_scan=np.linspace(0, 200e-12, 4001)

for cfg in configs:
    print(f'\n────────  {cfg["lbl"]}  ────────')

    # (1) baseline bounds
    res1 = minimize_scalar(lambda L: cost(L,cfg), bounds=(0,800e-12),
                           method='bounded', options={'xatol':1e-18})
    Lrp1 = res1.x; rms1 = np.sqrt(res1.fun)

    # (2) wider bounds + differential_evolution
    res2 = differential_evolution(lambda L: cost(L[0],cfg), [(0, 500e-12)],
                                  seed=42, maxiter=2000, tol=1e-16, polish=False)
    Lrp2 = res2.x[0]; rms2 = np.sqrt(res2.fun)

    # (3) DE + L-BFGS-B polish
    pol = minimize(lambda L: cost(L[0],cfg), [Lrp2], method='L-BFGS-B',
                   bounds=[(0, 500e-12)],
                   options={'ftol':1e-20,'gtol':1e-16,'maxiter':50000})
    Lrp3 = pol.x[0]; rms3 = np.sqrt(pol.fun)

    # Scan
    scan = np.array([cost(L,cfg) for L in LRp_scan])
    scan_rms = np.sqrt(scan)
    i_min = np.argmin(scan)
    Lrp_scan_best = LRp_scan[i_min]; rms_scan_best = scan_rms[i_min]

    print(f'  Baseline   bounded scalar  : L_Rp = {Lrp1*1e12:.3f} pH   RMS = {rms1:.6f}')
    print(f'  Wide bounds + DE           : L_Rp = {Lrp2*1e12:.3f} pH   RMS = {rms2:.6f}')
    print(f'  DE + L-BFGS-B polish       : L_Rp = {Lrp3*1e12:.3f} pH   RMS = {rms3:.6f}')
    print(f'  Fine grid scan (4001 pts)  : L_Rp = {Lrp_scan_best*1e12:.3f} pH   RMS = {rms_scan_best:.6f}')

    # Use the polished value as the final
    cfg['Lrp_fit'] = Lrp3
    cfg['rms_fit'] = rms3
    cfg['S_fit']   = sim_S11(cfg['ws'],Rs,Cj_FIXED,cfg['Rp'],cfg['Lcpw'],
                             Lrp=Lrp3,Lcpw2=cfg['Lcpw2'])

    # Baseline comparison
    S_base = sim_S11(cfg['ws'],Rs,Cj_FIXED,cfg['Rp'],cfg['Lcpw'],
                     Lrp=cfg['Lrp_baseline'],Lcpw2=cfg['Lcpw2'])
    rms_base = np.sqrt(np.mean(np.abs(S_base-cfg['S11m'])**2))
    cfg['S_base'] = S_base; cfg['rms_base'] = rms_base

    # Freq response with new Lrp
    Hckt_m = H_ckt(cfg['wm'],Rs,Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp=Lrp3,Lcpw2=cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot,   Rs,Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp=Lrp3,Lcpw2=cfg['Lcpw2'])
    Hd_m = 20*np.log10(np.abs(H_ph(cfg['wm'])*Hckt_m)/np.abs(H_ph(cfg['wm'][0])*Hckt_m[0]))
    Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    cfg['Hd_p'] = Hd_p
    cfg['bw']   = get_bw(f_plot,Hd_p)
    cfg['rms_H']= np.sqrt(np.mean((Hd_m-cfg['pm'])**2))

    cfg['scan_L'] = LRp_scan
    cfg['scan_rms'] = scan_rms

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
    rf'S11-only $L_{{R_p}}$ refit  —  38 Ω and 60 Ω  (Cj={Cj_FIXED*1e15:.1f} fF, all else fixed)' '\n'
    r'Row 1: RMS$_{S11}$ landscape   |   Row 2: Smith chart   |   Row 3: $|S_{11}|$ dB',
    fontsize=11, fontweight='bold')

for ci, cfg in enumerate(configs):
    col=cfg['col']

    # Row 0: Cost landscape
    ax=axes[0,ci]
    ax.plot(cfg['scan_L']*1e12, cfg['scan_rms'], '-', color=col, lw=1.5)
    ax.axvline(cfg['Lrp_baseline']*1e12, color='gray', ls=':', lw=1.0,
               label=f'baseline {cfg["Lrp_baseline"]*1e12:.1f} pH')
    ax.axvline(cfg['Lrp_fit']*1e12, color='red', ls='--', lw=1.0,
               label=f'best {cfg["Lrp_fit"]*1e12:.1f} pH')
    ax.scatter([cfg['Lrp_fit']*1e12], [cfg['rms_fit']], color='red', s=40, zorder=5)
    ax.set_xlabel(r'$L_{R_p}$ (pH)'); ax.set_ylabel(r'RMS $|\Delta\Gamma|$')
    ax.set_title(f'{cfg["lbl"]}  cost landscape', fontsize=10, fontweight='bold')
    ax.set_xlim(0,200); ax.grid(True,alpha=0.3); ax.legend(fontsize=9)

    # Row 1: Smith
    ax=axes[1,ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S_base'].real, cfg['S_base'].imag, ':', color='gray', lw=1.3, zorder=4,
            label=f'baseline  RMS={cfg["rms_base"]:.5f}')
    ax.plot(cfg['S_fit'].real, cfg['S_fit'].imag, '--', color='red', lw=1.5, zorder=5,
            label=f'best       RMS={cfg["rms_fit"]:.5f}')
    ax.set_title(f'{cfg["lbl"]}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    # Row 2: |S11| dB
    ax=axes[2,ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])),'-',color=col,lw=1.0,label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S_base'])),':',color='gray',lw=1.3,
            label=f'baseline ({cfg["Lrp_baseline"]*1e12:.1f} pH)')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S_fit'])),'--',color='red',lw=1.5,
            label=f'best ({cfg["Lrp_fit"]*1e12:.1f} pH)')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)

fig.tight_layout()
out='LRp_refit_38_60.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
