"""
Cj = 131 fF fixed, refit L_Rp for 38 Ω and 60 Ω only
=====================================================
S11-only cost. Multiple optimizers for convergence verification.
Paper H_ph + v_os baseline (τ_C = 2.88 ps).
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, differential_evolution, minimize

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
Cj_FIXED = 131.0e-15

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
for v in ['ref_f_GHz','ref_loss_dB','_s1p_33','_s1p_55','_freq_33','_freq_55']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz,ref_loss_dB=_ns['ref_f_GHz'],_ns['ref_loss_dB']

configs=[
    dict(lbl='Rp=38Ω', Rp=38.0, col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'], freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12),
    dict(lbl='Rp=60Ω', Rp=60.0, col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'], freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12),
]
def gs1p(a,fm): fr=a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); m=fr<=fm; return fr[m],S[m]
def gfr(a):    f=a[:,0]; c=a[:,1]+np.interp(f,ref_f_GHz,ref_loss_dB); return f*1e9,c-c[0]

for cfg in configs:
    fm,pm=gfr(cfg['freq']); fs11,S11m=gs1p(cfg['s1p'],fm.max())
    cfg.update(fm=fm,pm=pm,wm=2*np.pi*fm,fs11=fs11,S11m=S11m,ws=2*np.pi*fs11)

def cost(Lrp,cfg):
    S=sim_S11(cfg['ws'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp,cfg['Lcpw2'])
    return np.mean(np.abs(S-cfg['S11m'])**2)

# ── 4 optimizers per device ─────────────────────────────────────────
f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot
print('='*100)
print(f'Cj = {Cj_FIXED*1e15:.1f} fF fixed  |  L_Rp refit (38 Ω, 60 Ω)  |  τ_C = {tau_C*1e12:.2f} ps')
print('='*100)
LRp_scan=np.linspace(0, 150e-12, 4001)

for cfg in configs:
    print(f'\n──────  {cfg["lbl"]}  ──────')
    # (1) bounded scalar
    res1 = minimize_scalar(lambda L: cost(L,cfg), bounds=(0,500e-12),
                           method='bounded', options={'xatol':1e-18})
    # (2) DE
    res2 = differential_evolution(lambda L: cost(L[0],cfg), [(0,300e-12)],
                                  seed=42, maxiter=3000, tol=1e-16, polish=False)
    # (3) DE + L-BFGS-B
    pol = minimize(lambda L: cost(L[0],cfg), res2.x, method='L-BFGS-B',
                   bounds=[(0,300e-12)],
                   options={'ftol':1e-20,'gtol':1e-16,'maxiter':50000})
    # (4) grid scan
    scan = np.array([cost(L,cfg) for L in LRp_scan])
    i_min = np.argmin(scan); Lrp_scan_best = LRp_scan[i_min]

    print(f'  bounded scalar   : L_Rp = {res1.x*1e12:.4f} pH  RMS = {np.sqrt(res1.fun):.6f}')
    print(f'  DE                : L_Rp = {res2.x[0]*1e12:.4f} pH  RMS = {np.sqrt(res2.fun):.6f}')
    print(f'  DE + L-BFGS-B    : L_Rp = {pol.x[0]*1e12:.4f} pH  RMS = {np.sqrt(pol.fun):.6f}')
    print(f'  grid (4001 pts)  : L_Rp = {Lrp_scan_best*1e12:.4f} pH  RMS = {np.sqrt(scan[i_min]):.6f}')

    Lrp_final = pol.x[0]
    S_fit = sim_S11(cfg['ws'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp_final,cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S_fit-cfg['S11m'])**2))

    Hckt_m=H_ckt(cfg['wm'],Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp_final,cfg['Lcpw2'])
    Hckt_p=H_ckt(w_plot,   Cj_FIXED,cfg['Rp'],cfg['Lcpw'],Lrp_final,cfg['Lcpw2'])
    Hd_m=20*np.log10(np.abs(H_ph(cfg['wm'])*Hckt_m)/np.abs(H_ph(cfg['wm'][0])*Hckt_m[0]))
    Hd_p=20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    cfg.update(Lrp=Lrp_final, S_fit=S_fit, rms_s11=rms_s11,
               Hd_p=Hd_p, rms_H=np.sqrt(np.mean((Hd_m-cfg['pm'])**2)),
               bw=get_bw(f_plot,Hd_p), scan_L=LRp_scan, scan_rms=np.sqrt(scan))

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

fig, axes = plt.subplots(4, 2, figsize=(13, 18))
fig.suptitle(
    rf'$C_j$={Cj_FIXED*1e15:.0f} fF fixed  |  $L_{{R_p}}$ refit (38 Ω, 60 Ω)' '\n'
    r'Row 1: cost landscape   Row 2: Smith   Row 3: $|S_{11}|$ dB   Row 4: Freq response',
    fontsize=11, fontweight='bold')

for ci, cfg in enumerate(configs):
    col=cfg['col']; mk=cfg['mk']

    ax=axes[0,ci]
    ax.plot(cfg['scan_L']*1e12, cfg['scan_rms'], '-', color=col, lw=1.5)
    ax.axvline(cfg['Lrp']*1e12, color='red', ls='--', lw=1.0,
               label=f'best {cfg["Lrp"]*1e12:.2f} pH')
    ax.scatter([cfg['Lrp']*1e12],[cfg['rms_s11']], color='red', s=40, zorder=5)
    ax.set_xlabel(r'$L_{R_p}$ (pH)'); ax.set_ylabel(r'RMS $|\Delta\Gamma|$')
    ax.set_title(f'{cfg["lbl"]}  cost landscape',fontsize=10,fontweight='bold')
    ax.set_xlim(0,150); ax.grid(True,alpha=0.3); ax.legend(fontsize=9)

    ax=axes[1,ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(cfg['S_fit'].real, cfg['S_fit'].imag, '--', color='k', lw=1.5, zorder=5,
            label=f'L_Rp={cfg["Lrp"]*1e12:.2f}pH\nRMS={cfg["rms_s11"]:.4f}')
    ax.set_title(f'{cfg["lbl"]}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    ax=axes[2,ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-', color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S_fit'])), '--', color='k', lw=1.4, label='Sim')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)

    ax=axes[3,ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    bw_s = f'{cfg["bw"]:.1f}' if not np.isnan(cfg['bw']) else '>50'
    ax.plot(f_plot/1e9, cfg['Hd_p'], '-', color=col, lw=1.8,
            label=f'Sim BW={bw_s} GHz, RMS={cfg["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized H (dB)')
    ax.set_xlim(0,50); ax.set_ylim(-12,3)
    ax.legend(fontsize=9, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
out='Cj131_LRp_refit_38_60.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')

# Summary
print()
print('Summary  (Cj = 131 fF fixed)')
print('-'*60)
for cfg in configs:
    bw_s = f'{cfg["bw"]:.2f}' if not np.isnan(cfg['bw']) else '>50'
    print(f'  {cfg["lbl"]:>8}: L_Rp={cfg["Lrp"]*1e12:6.2f} pH  '
          f'RMS_S11={cfg["rms_s11"]:.5f}  BW={bw_s} GHz  RMS_H={cfg["rms_H"]:.3f} dB')
