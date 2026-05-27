"""
Fit common Cj — baseline: v(E) drift + paper tau_A = 3.53 ps
=============================================================
All other parameters fixed:
  - Rs = 8.92 Ω
  - L_CPW, L_CPW2, L_Rp per device (from prior S11 fits)
  - C_CPW = 46.53 fF
  - L_Rp(38Ω) = 55.2 pH (fitted earlier)

Free parameter:
  - Cj  (single common value, all 4 devices)

Two cost functions tested:
  (A) S11-only:  minimize  Σ_devices  mean|S11_sim − S11_meas|²
  (B) Combined:  minimize  Σ_devices  [|ΔΓ|² + (1/400)·(ΔH_dB)²]
"""
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Transit times (v(E) drift + paper tau_A = 3.53 ps) ──────────────
W_A_undep=480e-9; W_A_dep=160e-9; W_C=740e-9
W_tot=W_A_undep+W_A_dep+50e-9+W_C
tau_A = 3.530e-12               # paper formula
tau_Ad = 2.026e-12              # v(E) integrated
tau_cl_eff = 0.272e-12 + 0.527e-12  # v(E): grading + cliff lumped
tau_C  = 7.731e-12              # v(E) integrated
tau_h  = W_A_dep / 4.5e4
W_cl_eff = 80e-9                # grading + cliff

def H_ph(w):
    sinc = lambda x: np.sinc(x/np.pi)
    HA    = 1/(1+1j*w*tau_A)
    HAd   = sinc(w*tau_Ad/2)*np.exp(-1j*(w*tau_A + w*tau_Ad/2))
    Hcl   = sinc(w*tau_cl_eff/2)*np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl_eff/2))
    Hco   = sinc(w*tau_C/2)*np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl_eff + w*tau_C/2))
    HAd_h = sinc(w*tau_h/2)*np.exp(-1j*(w*tau_A + w*tau_h/2))
    return (W_A_undep*HA + W_A_dep*HAd + W_A_dep*HAd_h
            + W_cl_eff*Hcl + W_C*Hco)/(W_tot + W_A_dep)

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp + 1j*w*Lrp)
def H_ckt(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
    Zs=Rs+1j*w*Lcpw2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cpd+Y_A*(1+1j*w*Cpd*Zs))
def sim_S11(w,Rs,Cpd,Rp,Lcpw,Lrp=0.0,Lcpw2=0.0):
    Zs=Rs+1j*w*Lcpw2
    Z_dev=Zs+1/(1j*w*Cpd)
    Y_n=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Z_dev
    Z_in=1j*w*Lcpw+1/Y_n
    return (Z_in-50)/(Z_in+50)
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Load measurement data from main.py ──────────────────────────────
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
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=153.0e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,    col='#1B998B', mk='o',
         s1p=_ns['_s1p_33'],  freq=_ns['_freq_33'],
         Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=55.2e-12),
    dict(lbl='Rp=60Ω',  Rp=60.0,    col='#FF8C00', mk='s',
         s1p=_ns['_s1p_55'],  freq=_ns['_freq_55'],
         Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf,  col='#E91E8C', mk='^',
         s1p=_ns['_s1p_WO'],  freq=_ns['_freq_WO'],
         Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=0.0),
]
def gs1p(arr,f_max):
    fr=arr[:,0]; S=10**(arr[:,1]/20)*np.exp(1j*np.deg2rad(arr[:,2]))
    m=fr<=f_max; return fr[m], S[m]
def gfr(arr):
    f_ghz=arr[:,0]; cal=arr[:,1]+np.interp(f_ghz,ref_f_GHz,ref_loss_dB)
    return f_ghz*1e9, cal-cal[0]

# Cache measurement data
for cfg in configs:
    fm,pm = gfr(cfg['freq'])
    fs11,S11m = gs1p(cfg['s1p'], fm.max())
    cfg.update(fm=fm,pm=pm,wm=2*np.pi*fm, fs11=fs11,S11m=S11m,ws=2*np.pi*fs11)

# ── Cost functions over all devices ─────────────────────────────────
def cost_s11(Cj):
    tot=0.0
    for cfg in configs:
        S=sim_S11(cfg['ws'],Rs,Cj,cfg['Rp'],cfg['Lcpw'],
                  Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        tot += np.mean(np.abs(S-cfg['S11m'])**2)
    return tot

def cost_combined(Cj):
    tot=0.0
    for cfg in configs:
        S=sim_S11(cfg['ws'],Rs,Cj,cfg['Rp'],cfg['Lcpw'],
                  Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        tot += np.mean(np.abs(S-cfg['S11m'])**2)
        Hckt = H_ckt(cfg['wm'],Rs,Cj,cfg['Rp'],cfg['Lcpw'],
                     Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        Ht = H_ph(cfg['wm']) * Hckt
        Hd = 20*np.log10(np.abs(Ht)/np.abs(Ht[0]))
        tot += np.mean((Hd - cfg['pm'])**2) / 400.0
    return tot

print('='*100)
print(f'Baseline: v(E) drift + paper τ_A = 3.53 ps')
print(f'  drift τ: τ_Ad={tau_Ad*1e12:.3f}, τ_cl_eff={tau_cl_eff*1e12:.3f}, '
      f'τ_C={tau_C*1e12:.3f} ps   (W_cl_eff={W_cl_eff*1e9:.0f} nm)')
print(f'  Rs={Rs}, C_CPW={C_CPW*1e15:.2f} fF  |  per-device L_CPW, L_CPW2, L_Rp from prior fits')
print(f'  Fitting: SINGLE common Cj across all 4 devices')
print('='*100)

res_a = minimize_scalar(cost_s11,      bounds=(50e-15, 300e-15), method='bounded',
                        options={'xatol':1e-18})
res_b = minimize_scalar(cost_combined, bounds=(50e-15, 300e-15), method='bounded',
                        options={'xatol':1e-18})
Cj_A = res_a.x; Cj_B = res_b.x

print(f'\n(A) S11-only  common Cj = {Cj_A*1e15:.2f} fF')
print(f'(B) Combined  common Cj = {Cj_B*1e15:.2f} fF')

# ── Evaluate both Cj values ─────────────────────────────────────────
f_plot = np.linspace(0.1e9, 50e9, 5000); w_plot = 2*np.pi*f_plot

def evaluate(Cj_val, label):
    print(f'\n--- Cj = {Cj_val*1e15:.2f} fF  ({label}) ---')
    print(f'{"Device":>10} | {"RMS_S11":>9} | {"BW (GHz)":>9} | {"RMS_H (dB)":>11}')
    print('-'*50)
    results=[]
    for cfg in configs:
        S = sim_S11(cfg['ws'],Rs,Cj_val,cfg['Rp'],cfg['Lcpw'],
                    Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        rms_s11 = np.sqrt(np.mean(np.abs(S - cfg['S11m'])**2))
        Hckt_m = H_ckt(cfg['wm'], Rs,Cj_val,cfg['Rp'],cfg['Lcpw'],
                       Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        Hckt_p = H_ckt(w_plot, Rs,Cj_val,cfg['Rp'],cfg['Lcpw'],
                       Lrp=cfg['Lrp'],Lcpw2=cfg['Lcpw2'])
        Hd_m = 20*np.log10(np.abs(H_ph(cfg['wm'])*Hckt_m)/np.abs(H_ph(cfg['wm'][0])*Hckt_m[0]))
        Hd_p = 20*np.log10(np.abs(H_ph(w_plot)*Hckt_p)/np.abs(H_ph(0)*Hckt_p[0]))
        rms_H = np.sqrt(np.mean((Hd_m - cfg['pm'])**2))
        bw    = get_bw(f_plot, Hd_p)
        bw_s  = f'{bw:.1f}' if not np.isnan(bw) else '>50'
        print(f'{cfg["lbl"]:>10} | {rms_s11:>9.5f} | {bw_s:>9} | {rms_H:>11.3f}')
        results.append(dict(cfg=cfg,S=S,rms_s11=rms_s11,Hd_p=Hd_p,bw=bw,bw_s=bw_s,rms_H=rms_H))
    return results

res_eval_A = evaluate(Cj_A, 'S11-only fit')
res_eval_B = evaluate(Cj_B, 'Combined fit')

# Decide which to use as "primary" — combined fit (balances both)
PRIMARY_Cj = Cj_B; PRIMARY = res_eval_B; PRIMARY_LBL = 'combined'

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
    rf'Common $C_j$ fit  |  $v(E)$ drift + paper $\tau_A$=3.53 ps' '\n'
    rf'$C_j^{{S11-only}}$={Cj_A*1e15:.1f} fF (gray dashed)   |   '
    rf'$C_j^{{combined}}$={Cj_B*1e15:.1f} fF (navy dashed, primary)' '\n'
    r'Row 1: Smith   |   Row 2: $|S_{11}|$ dB   |   Row 3: Freq response',
    fontsize=11, fontweight='bold')

for ci, r_b in enumerate(res_eval_B):
    cfg = r_b['cfg']; col = cfg['col']; mk = cfg['mk']
    r_a = res_eval_A[ci]

    # Smith
    ax = axes[0, ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real, cfg['S11m'].imag, s=10, color=col, zorder=6, label='Meas.')
    ax.plot(r_a['S'].real, r_a['S'].imag, ':', color='gray', lw=1.4, zorder=4,
            label=f'Cj={Cj_A*1e15:.1f}')
    ax.plot(r_b['S'].real, r_b['S'].imag, '--', color='navy', lw=1.5, zorder=5,
            label=f'Cj={Cj_B*1e15:.1f}')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|: A={r_a["rms_s11"]:.4f}  B={r_b["rms_s11"]:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.85, edgecolor='none',
              handlelength=1.5, markerscale=1.5)

    # |S11| dB
    ax = axes[1, ci]
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(cfg['S11m'])), '-',  color=col, lw=1.0, label='Meas.')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(r_a['S'])), ':',
            color='gray', lw=1.3, label=f'Cj={Cj_A*1e15:.1f}')
    ax.plot(cfg['fs11']/1e9, 20*np.log10(np.abs(r_b['S'])), '--',
            color='navy', lw=1.4, label=f'Cj={Cj_B*1e15:.1f}')
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_xlim(0, cfg['fs11'].max()/1e9)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

    # Freq response
    ax = axes[2, ci]
    ax.scatter(cfg['fm']/1e9, cfg['pm'], color='k', marker=mk, s=22,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas.')
    ax.plot(f_plot/1e9, r_a['Hd_p'], ':', color='gray', lw=1.3,
            label=f'Cj={Cj_A*1e15:.1f}  BW={r_a["bw_s"]}, RMS={r_a["rms_H"]:.2f}dB')
    ax.plot(f_plot/1e9, r_b['Hd_p'], '--', color='navy', lw=1.5,
            label=f'Cj={Cj_B*1e15:.1f}  BW={r_b["bw_s"]}, RMS={r_b["rms_H"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9); ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3)
    ax.legend(fontsize=7.5, loc='lower left'); ax.grid(True, alpha=0.3)

fig.tight_layout()
out = 'Cj_common_vE_tauA3p53.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
