"""
-7 V baseline re-run with depleted-absorber HOLE term included in H_ph
======================================================================
H_ph(ω) = 1/(1+jωτ_A) · 1/(W+W_Adep) · [ W_A·(2+jωτ_R)/(2(1+jωτ_R))
                                        + W_C·sinc(ωτ_C/2)·exp(-jωτ_C/2)
                                        + W_Adep·sinc(ωτ_h/2)·exp(-jω(τ_A+τ_h/2)) ]
  τ_h = W_Adep/v_h = 160nm / 4.5e4 m/s = 3.56 ps  (depleted-absorber hole, backward drift)

All circuit params: locked Option B baseline (-7 V), Cj=131 fF.
S11 unchanged (H_ph-independent). Only frequency response / BW differs.
"""
import numpy as np, pandas as pd, re, os
import matplotlib.pyplot as plt

# ── Transit times ───────────────────────────────────────────────────
W_A=480e-9; W_C=980e-9; W_Adep=160e-9; W=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
v_h=4.5e4
tau_h=W_Adep/v_h                       # 3.56 ps

def H_ph_nohole(w):
    sinc=lambda x: np.sinc(x/np.pi)
    return (W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))+W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2))/(W*(1+1j*w*tau_A))

def H_ph_hole(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t = W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t = W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    hole_t= W_Adep*sinc(w*tau_h/2)*np.exp(-1j*(w*tau_A+w*tau_h/2))
    return (abs_t+col_t+hole_t)/((W+W_Adep)*(1+1j*w*tau_A))

# ── Circuit (-7 V Option B locked) ─────────────────────────────────
C_CPW=46.53e-15; R_L=50.0; Rs=8.92; Cj=131.0e-15
def _Y_Rp(w,Rp,Lrp): return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def H_ckt(w,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cj+Y_A*(1+1j*w*Cj*Zs))
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]; return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Load measured freq response from main.py ────────────────────────
_main=open('main.py').read(); _ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.MULTILINE|re.DOTALL)
    exec(m.group(0),_ns)
ref_f_GHz,ref_loss_dB=_ns['ref_f_GHz'],_ns['ref_loss_dB']
def gfr(a): f=a[:,0]; c=a[:,1]+np.interp(f,ref_f_GHz,ref_loss_dB); return f*1e9,c-c[0]

configs=[
    dict(lbl='Rp=200Ω', Rp=200.0, col='#888888', mk='D', freq=_ns['_freq_200'],
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),
    dict(lbl='Rp=38Ω',  Rp=38.0,  col='#1B998B', mk='o', freq=_ns['_freq_33'],
         Lcpw=141.6e-12, Lcpw2=56.3e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60.0,  col='#FF8C00', mk='s', freq=_ns['_freq_55'],
         Lcpw=149.9e-12, Lcpw2=48.0e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf,col='#E91E8C', mk='^', freq=_ns['_freq_WO'],
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=0.0),
]
f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot

# transit-time-limited BW
from scipy.optimize import brentq
ftr_no=brentq(lambda f:20*np.log10(np.abs(H_ph_nohole(2*np.pi*f*1e9))/np.abs(H_ph_nohole(2*np.pi*1e6)))+3,1,200)
ftr_ho=brentq(lambda f:20*np.log10(np.abs(H_ph_hole(2*np.pi*f*1e9))/np.abs(H_ph_hole(2*np.pi*1e6)))+3,1,200)

print('='*90)
print(f'H_ph WITH depleted-absorber hole term (tau_h = {tau_h*1e12:.2f} ps)')
print(f'  Transit-time-limited f_tr:  no-hole = {ftr_no:.2f} GHz  |  with-hole = {ftr_ho:.2f} GHz')
print('='*90)
print(f'{"Device":>10} | {"BW no-hole":>10} | {"BW hole":>8} | {"RMS_H no-hole":>13} | {"RMS_H hole":>10}')
print('-'*75)

fig,axes=plt.subplots(2,2,figsize=(14,9.5))
fig.suptitle(f'Frequency Response — depleted-absorber hole term included '
             f'($\\tau_h$={tau_h*1e12:.2f} ps)\n'
             f'$f_{{tr}}$: {ftr_no:.1f} GHz (no hole) → {ftr_ho:.1f} GHz (with hole)  |  '
             f'-7 V, Cj=131 fF',fontsize=12,fontweight='bold')
export={}
for ci,cfg in enumerate(configs):
    fm,pm=gfr(cfg['freq']); wm=2*np.pi*fm
    Hckt_m=H_ckt(wm,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hckt_p=H_ckt(w_plot,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    res={}
    for tag,Hph in [('no-hole',H_ph_nohole),('hole',H_ph_hole)]:
        Hd_m=20*np.log10(np.abs(Hph(wm)*Hckt_m)/np.abs(Hph(wm[0])*Hckt_m[0]))
        Hd_p=20*np.log10(np.abs(Hph(w_plot)*Hckt_p)/np.abs(Hph(1e6)*Hckt_p[0]))
        res[tag]=dict(rms=np.sqrt(np.mean((Hd_m-pm)**2)),bw=get_bw(f_plot,Hd_p),Hd_p=Hd_p,Hd_m=Hd_m)
    print(f'{cfg["lbl"]:>10} | {res["no-hole"]["bw"]:>9.1f}G | {res["hole"]["bw"]:>7.1f}G | '
          f'{res["no-hole"]["rms"]:>13.3f} | {res["hole"]["rms"]:>10.3f}')
    ax=axes[ci//2][ci%2]
    ax.scatter(fm/1e9,pm,color='k',marker=cfg['mk'],s=24,edgecolors=cfg['col'],
               linewidths=1.1,zorder=5,label='Meas')
    ax.plot(f_plot/1e9,res['no-hole']['Hd_p'],':',color='gray',lw=1.5,
            label=f'no hole  BW={res["no-hole"]["bw"]:.1f}G, RMS={res["no-hole"]["rms"]:.2f}')
    ax.plot(f_plot/1e9,res['hole']['Hd_p'],'-',color=cfg['col'],lw=2.0,
            label=f'with hole  BW={res["hole"]["bw"]:.1f}G, RMS={res["hole"]["rms"]:.2f}')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalized H (dB)')
    ax.set_title(cfg['lbl'],fontsize=11,fontweight='bold')
    ax.set_xlim(0,50); ax.set_ylim(-15,3)
    ax.legend(fontsize=9,loc='lower left'); ax.grid(True,alpha=0.3)
    export[cfg['lbl']]=(f_plot,res['hole']['Hd_p'],fm,pm,res['hole']['Hd_m'])
fig.tight_layout()
fig.savefig('FreqResp_hole_included.png',dpi=150,bbox_inches='tight')
print('\nSaved: FreqResp_hole_included.png')

out='origin_data_hole'; os.makedirs(out,exist_ok=True)
for lbl,(fp,Hdp,fm,pm,Hdm) in export.items():
    t=lbl.replace('Ω','ohm').replace('=','_')
    with open(f'{out}/model_40GHz_{t}.txt','w') as f:
        f.write('Freq_GHz\tNorm_dB_model_hole\n')
        for i in range(len(fp)):
            if fp[i]<=40e9: f.write(f'{fp[i]/1e9:.6g}\t{Hdp[i]:.6g}\n')
    with open(f'{out}/freqresp_{t}.txt','w') as f:
        f.write('Freq_GHz\tNorm_dB_meas\tNorm_dB_model_hole\n')
        for i in range(len(fm)):
            f.write(f'{fm[i]/1e9:.6g}\t{pm[i]:.6g}\t{Hdm[i]:.6g}\n')
print(f'Saved: {out}/ (model_40GHz + freqresp per device)')
