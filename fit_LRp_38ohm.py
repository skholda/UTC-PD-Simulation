"""
Refit L_Rp for 38-Ohm device with Cj = 147 fF fixed
====================================================
Cj = 147 fF (all devices), Rs/L_CPW/L_CPW2 from original S11 fit.
For Rp=38Ω: vary L_Rp as free parameter, S11-only cost.
Compare against original (Cj=132.7, L_Rp=65.6 pH).
"""
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Load arrays ──────────────────────────────────────────────────
_main = open(os.path.join(os.path.dirname(__file__),'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz','ref_loss_dB','_s1p_33','_freq_33']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz, ref_loss_dB = _ns['ref_f_GHz'], _ns['ref_loss_dB']
S1P_38 = _ns['_s1p_33']; FREQ_38 = _ns['_freq_33']

# ── Transit-time ─────────────────────────────────────────────────
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
C_CPW=46.53e-15; R_L=50.0
def _Y_Rp(w,Rp,Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp+1j*w*Lrp)
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
def gfr(arr):
    f_ghz=arr[:,0]; cal=arr[:,1]+np.interp(f_ghz,ref_f_GHz,ref_loss_dB)
    return f_ghz*1e9, cal-cal[0]
def gs1p(arr,f_max):
    fr=arr[:,0]
    S11=10**(arr[:,1]/20)*np.exp(1j*np.deg2rad(arr[:,2]))
    m=fr<=f_max
    return fr[m], S11[m]

# ── Device-fixed params (38Ω) ────────────────────────────────────
Rp=38.0; Rs=8.92; Lcpw=135.9e-12; Lcpw2=43.0e-12
Cj_FIXED=147.0e-15
LRp_ORIG=65.6e-12          # MATLAB FEM

fm,pm=gfr(FREQ_38); f_max=fm.max()
fs11,S11m=gs1p(S1P_38,f_max)
ws=2*np.pi*fs11; wm=2*np.pi*fm

# ── 1-parameter fit: only L_Rp ──────────────────────────────────
def cost(Lrp):
    S=sim_S11(ws,Rs,Cj_FIXED,Rp,Lcpw,Lrp=Lrp,Lcpw2=Lcpw2)
    return np.mean(np.abs(S-S11m)**2)

res=minimize_scalar(cost, bounds=(0.0, 1000e-12), method='bounded',
                    options={'xatol':1e-18})
LRp_NEW=res.x

# ── Three cases ─────────────────────────────────────────────────
cases=[
    ('Original (Cj=132.7fF, L_Rp=65.6pH)', 132.7e-15, LRp_ORIG, 'gray', ':'),
    ('Cj=147fF, L_Rp=65.6pH (FEM)',         Cj_FIXED,  LRp_ORIG, 'navy', '--'),
    (f'Cj=147fF, L_Rp={LRp_NEW*1e12:.1f}pH (fitted)', Cj_FIXED, LRp_NEW, 'red',  '-'),
]

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot

print('='*100)
print(f'  38-Ohm device: refit L_Rp only (Cj={Cj_FIXED*1e15:.0f} fF, '
      f'Rs={Rs} Ω, L_CPW={Lcpw*1e12:.1f} pH, L_CPW2={Lcpw2*1e12:.1f} pH fixed)')
print('='*100)
print(f'{"Case":>50} | {"RMS_S11":>9} | {"BW(7.86)":>9} | {"BW(3.53)":>9} | '
      f'{"RMS_H(7.86)":>11} | {"RMS_H(3.53)":>11}')
print('-'*100)

case_data=[]
for lbl,Cj,Lrp,clr,ls in cases:
    Ss = sim_S11(ws, Rs, Cj, Rp, Lcpw, Lrp=Lrp, Lcpw2=Lcpw2)
    rms_s11 = np.sqrt(np.mean(np.abs(Ss-S11m)**2))
    Hckt_p = H_ckt(w_plot, Rs, Cj, Rp, Lcpw, Lrp=Lrp, Lcpw2=Lcpw2)
    Hckt_m = H_ckt(wm,     Rs, Cj, Rp, Lcpw, Lrp=Lrp, Lcpw2=Lcpw2)
    res_tau={}
    for tag,tA in [('7.86',7.862e-12),('3.53',3.530e-12)]:
        Hph=H_ph_factory(tA)
        Hd_p=20*np.log10(np.abs(Hph(w_plot)*Hckt_p)/np.abs(Hph(0)*Hckt_p[0]))
        Hd_m=20*np.log10(np.abs(Hph(wm)    *Hckt_m)/np.abs(Hph(wm[0])*Hckt_m[0]))
        bw=get_bw(f_plot,Hd_p)
        rms_H=np.sqrt(np.mean((Hd_m-pm)**2))
        res_tau[tag]=dict(Hd_p=Hd_p,bw=bw,rms_H=rms_H)
    case_data.append(dict(lbl=lbl,Cj=Cj,Lrp=Lrp,clr=clr,ls=ls,
                          Ss=Ss,rms_s11=rms_s11,res=res_tau))
    bw1=res_tau['7.86']['bw']; bw2=res_tau['3.53']['bw']
    s1=f'{bw1:.1f}' if not np.isnan(bw1) else '>50'
    s2=f'{bw2:.1f}' if not np.isnan(bw2) else '>50'
    print(f'{lbl:>50} | {rms_s11:>9.5f} | {s1:>9} | {s2:>9} | '
          f'{res_tau["7.86"]["rms_H"]:>11.3f} | {res_tau["3.53"]["rms_H"]:>11.3f}')

# ── Plot ────────────────────────────────────────────────────────
def draw_smith(ax,lw_grid=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw_grid+0.3))
    ax.axhline(0,color='#888',lw=lw_grid,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#aaa',
                                lw=lw_grid,ls=':',zorder=0))
    theta=np.linspace(0,np.pi,400)
    for x in [0.2,0.5,1,2,5]:
        for sgn in [1,-1]:
            cx,cy,rad=1,sgn/x,1/x
            xx=cx+rad*np.cos(theta); yy=cy+rad*np.sin(theta)*sgn
            m=(xx**2+yy**2<=1.002)
            ax.plot(xx[m],yy[m],color='#aaa',lw=lw_grid,ls=':',zorder=0)

fig,axes=plt.subplots(2,2,figsize=(14,11))
fig.suptitle(
    rf'38-Ω device: $L_{{R_p}}$ refit (Cj=147 fF, others fixed)' '\n'
    rf'Fitted: $L_{{R_p}}$={LRp_NEW*1e12:.2f} pH  vs  MATLAB FEM: {LRp_ORIG*1e12:.1f} pH',
    fontsize=12, fontweight='bold')

# Smith chart
ax=axes[0,0]; draw_smith(ax)
ax.scatter(S11m.real,S11m.imag,s=12,color='#1B998B',zorder=6,label='Meas.')
for cd in case_data:
    ax.plot(cd['Ss'].real, cd['Ss'].imag, cd['ls'], color=cd['clr'], lw=1.6,
            label=f'{cd["lbl"].split(" (")[0]}  RMS={cd["rms_s11"]:.4f}')
ax.set_title('Smith chart',fontsize=11,fontweight='bold')
ax.legend(fontsize=8.5,loc='lower left',framealpha=0.9,edgecolor='none',
          handlelength=1.8,markerscale=1.4)

# |S11| dB
ax=axes[0,1]
ax.plot(fs11/1e9,20*np.log10(np.abs(S11m)),'-',color='#1B998B',lw=1.2,label='Meas.')
for cd in case_data:
    ax.plot(fs11/1e9, 20*np.log10(np.abs(cd['Ss'])),
            cd['ls'], color=cd['clr'], lw=1.6,
            label=f'{cd["lbl"].split(" (")[0]}  RMS={cd["rms_s11"]:.4f}')
ax.set_xlabel('Frequency (GHz)',fontsize=10)
ax.set_ylabel('|S11| (dB)',fontsize=10)
ax.set_title('|S11| magnitude',fontsize=11,fontweight='bold')
ax.set_xlim(0,fs11.max()/1e9)
ax.legend(fontsize=8.5,loc='lower right')
ax.grid(True,alpha=0.3)

# Frequency response (tau_A = 7.86)
for ai,tag in enumerate(['7.86','3.53']):
    ax=axes[1,ai]
    ax.scatter(fm/1e9,pm,color='k',marker='o',s=22,edgecolors='#1B998B',
               linewidths=1.0,zorder=5,label='Meas.')
    for cd in case_data:
        d=cd['res'][tag]
        bw_s=f'{d["bw"]:.1f}' if not np.isnan(d['bw']) else '>50'
        ax.plot(f_plot/1e9, d['Hd_p'], cd['ls'], color=cd['clr'], lw=1.6,
                label=f'{cd["lbl"].split(" (")[0]}  BW={bw_s} GHz  RMS={d["rms_H"]:.2f}dB')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)',fontsize=10)
    ax.set_ylabel('Normalized H (dB)',fontsize=10)
    ax.set_title(rf'Frequency Response  ($\tau_A$ = {tag} ps)',
                 fontsize=11,fontweight='bold')
    ax.set_xlim(0,50); ax.set_ylim(-12,3)
    ax.legend(fontsize=8.5,loc='lower left')
    ax.grid(True,alpha=0.3)

fig.tight_layout()
out='fit_LRp_38ohm.png'
fig.savefig(out,dpi=150,bbox_inches='tight')
print(f'\nSaved: {out}')
