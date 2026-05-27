"""
Per-device L_Rp refit with PAPER-based H_ph(omega)
===================================================
H_ph follows the equation:
  J_tot = (1/W) * [integral over absorber (J_e+J_h) + integral over collector J_e]
        = J_e(W_A)/W * [ W_A * (2 + jωτ_R)/(2(1+jωτ_R))
                       + W_C * sinc(ωτ_C/2) * exp(-jωτ_C/2) ]
  J_e(W_A) = J_DC / (1 + jωτ_A)

Two regions:
  W_A = 480 nm  (undepleted absorber, p+ InGaAs)
  W_C = 980 nm  (dep abs + grading + cliff + collector — drift region)
  W   = 1460 nm
  τ_A = 3.53 ps  (paper: diffusion + quasi-field)
  τ_C = 10.56 ps (v(E) integrated over full W_C)
  τ_R = 0.07 ps  (dielectric relaxation, p~1e18, μ_h~100)

Circuit fit:
  - Cj = 141.1 fF fixed (from prior common S11-only fit)
  - Per-device L_Rp refit via S11-only cost
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Paper H_ph parameters ───────────────────────────────────────────
W_A_paper = 480e-9         # undep absorber
W_C_paper = 980e-9         # depleted region (dep abs + grading + cliff + collector)
W_paper   = W_A_paper + W_C_paper
tau_A     = 3.530e-12      # paper formula: W_A^2 / [D_e (3+ln R)]
tau_C     = 10.557e-12     # v(E) integrated over 980 nm
tau_R     = 0.07e-12       # dielectric relaxation (small)

def H_ph(w):
    """Paper-based H_ph(ω) — 2-region model."""
    sinc = lambda x: np.sinc(x/np.pi)
    # absorber factor (with hole dielectric relaxation)
    absorber_term = W_A_paper * (2.0 + 1j*w*tau_R) / (2.0*(1.0 + 1j*w*tau_R))
    # collector drift (sinc + delay)
    collector_term = W_C_paper * sinc(w*tau_C/2) * np.exp(-1j*w*tau_C/2)
    # J_e(W_A) = J_DC / (1 + jωτ_A)
    return (absorber_term + collector_term) / (W_paper * (1.0 + 1j*w*tau_A))

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92
Cj_FIXED = 141.1e-15
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp + 1j*w*Lrp)
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

# ── Load data ───────────────────────────────────────────────────────
_main = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz','ref_loss_dB',
            '_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
            '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz, ref_loss_dB = _ns['ref_f_GHz'], _ns['ref_loss_dB']

configs = [
    dict(lbl='Rp=200Ω', Rp=200.0,   col='#888888', mk='D',
         s1p=_ns['_s1p_200'], freq=_ns['_freq_200'],
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp_FEM=153.0e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,    col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'],  freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp_FEM=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60.0,    col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'],  freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp_FEM=71.8e-12),
    dict(lbl='Open',    Rp=np.inf,  col='#E91E8C', mk='^',
         s1p=_ns['_s1p_WO'],  freq=_ns['_freq_WO'],
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp_FEM=0.0),
]
def gs1p(arr,f_max):
    fr=arr[:,0]; S=10**(arr[:,1]/20)*np.exp(1j*np.deg2rad(arr[:,2]))
    m=fr<=f_max; return fr[m], S[m]
def gfr(arr):
    f_ghz=arr[:,0]; cal=arr[:,1]+np.interp(f_ghz,ref_f_GHz,ref_loss_dB)
    return f_ghz*1e9, cal-cal[0]
for cfg in configs:
    fm,pm = gfr(cfg['freq'])
    fs11,S11m = gs1p(cfg['s1p'], fm.max())
    cfg.update(fm=fm,pm=pm,wm=2*np.pi*fm, fs11=fs11,S11m=S11m,ws=2*np.pi*fs11)

print('='*100)
print(f'PAPER H_ph:  W_A={W_A_paper*1e9:.0f} nm, W_C={W_C_paper*1e9:.0f} nm, '
      f'W={W_paper*1e9:.0f} nm')
print(f'  τ_A={tau_A*1e12:.3f} ps (paper), τ_C={tau_C*1e12:.3f} ps (v(E) over W_C), '
      f'τ_R={tau_R*1e12:.3f} ps')
print(f'Circuit:  Cj={Cj_FIXED*1e15:.2f} fF (fixed), Rs={Rs}, C_CPW={C_CPW*1e15:.2f} fF')
print(f'Fit:  per-device L_Rp via S11-only cost')
print('='*100)
print(f'{"Device":>10} | {"L_Rp_FEM":>9} | {"L_Rp_fit":>9} | '
      f'{"RMS_S11":>9} | {"BW (GHz)":>9} | {"RMS_H":>7}')
print('-'*100)

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot
results=[]
for cfg in configs:
    if np.isinf(cfg['Rp']):
        Lrp_fit = 0.0
    else:
        def cost(Lrp):
            S = sim_S11(cfg['ws'], Rs, Cj_FIXED, cfg['Rp'], cfg['Lcpw'],
                        Lrp=Lrp, Lcpw2=cfg['Lcpw2'])
            return np.mean(np.abs(S - cfg['S11m'])**2)
        res = minimize_scalar(cost, bounds=(0.0, 800e-12), method='bounded',
                              options={'xatol':1e-18})
        Lrp_fit = res.x
    S_fit = sim_S11(cfg['ws'], Rs, Cj_FIXED, cfg['Rp'], cfg['Lcpw'],
                    Lrp=Lrp_fit, Lcpw2=cfg['Lcpw2'])
    rms_s11 = np.sqrt(np.mean(np.abs(S_fit - cfg['S11m'])**2))

    Hckt_m = H_ckt(cfg['wm'], Rs, Cj_FIXED, cfg['Rp'], cfg['Lcpw'],
                   Lrp=Lrp_fit, Lcpw2=cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot,    Rs, Cj_FIXED, cfg['Rp'], cfg['Lcpw'],
                   Lrp=Lrp_fit, Lcpw2=cfg['Lcpw2'])
    Hd_m = 20*np.log10(np.abs(H_ph(cfg['wm'])*Hckt_m)/np.abs(H_ph(cfg['wm'][0])*Hckt_m[0]))
    Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
    rms_H = np.sqrt(np.mean((Hd_m - cfg['pm'])**2))
    bw    = get_bw(f_plot, Hd_p)
    bw_s  = f'{bw:.1f}' if not np.isnan(bw) else '>50'

    print(f'{cfg["lbl"]:>10} | {cfg["Lrp_FEM"]*1e12:>8.1f}pH | {Lrp_fit*1e12:>8.1f}pH | '
          f'{rms_s11:>9.5f} | {bw_s:>9} | {rms_H:>7.3f}')
    results.append(dict(cfg=cfg, Lrp_fit=Lrp_fit, S_fit=S_fit, rms_s11=rms_s11,
                        Hd_p=Hd_p, bw=bw, bw_s=bw_s, rms_H=rms_H))

# ── Plot ────────────────────────────────────────────────────────────
def draw_smith(ax,lw_grid=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw_grid+0.3))
    ax.axhline(0,color='#888',lw=lw_grid,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#aaa',lw=lw_grid,ls=':',zorder=0))
    theta=np.linspace(0,np.pi,400)
    for x in [0.2,0.5,1,2,5]:
        for sgn in [1,-1]:
            cx,cy,rad=1,sgn/x,1/x
            xx=cx+rad*np.cos(theta); yy=cy+rad*np.sin(theta)*sgn
            mm=(xx**2+yy**2<=1.002)
            ax.plot(xx[mm],yy[mm],color='#aaa',lw=lw_grid,ls=':',zorder=0)

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    rf'Paper $H_{{ph}}$  |  $W_A$={W_A_paper*1e9:.0f}nm,  $W_C$={W_C_paper*1e9:.0f}nm,  '
    rf'$\tau_A$={tau_A*1e12:.2f}ps,  $\tau_C$={tau_C*1e12:.2f}ps,  $\tau_R$={tau_R*1e12:.2f}ps' '\n'
    rf'$C_j$={Cj_FIXED*1e15:.1f} fF (fixed)  |  per-device $L_{{R_p}}$ refit via S11-only',
    fontsize=10.5, fontweight='bold')

for ci, r in enumerate(results):
    cfg = r['cfg']; col = cfg['col']; mk = cfg['mk']
    ax = axes[0, ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(r['S_fit'].real, r['S_fit'].imag, '--', color='k', lw=1.5, zorder=5,
            label=f'Sim  L_Rp={r["Lrp_fit"]*1e12:.1f}pH')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|={r["rms_s11"]:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    ax = axes[1, ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-',  color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(r['S_fit'])), '--', color='k', lw=1.4, label='Sim')
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right'); ax.grid(True, alpha=0.3)

    ax = axes[2, ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, r['Hd_p'], '-', color=col, lw=2.0,
            label=f'Sim (paper $H_{{ph}}$)  BW={r["bw_s"]} GHz  RMS={r["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
out = 'fit_LRp_paper_Hph.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
