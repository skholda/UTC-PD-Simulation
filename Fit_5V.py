"""
-5 V S11 fitting (Option B procedure) + freq response verification
===================================================================
Cj = 161 fF (C-V @ -5 V, fixed). Rs, C_CPW, FEM L_Rp: bias-independent, locked.
Fit: L_total (common) + L_CPW2 (38, 60) over full 40 GHz S11.
Consistency check vs -7 V fit (L should be bias-independent).
"""
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# ── Paper H_ph (locked) ─────────────────────────────────────────────
W_A=480e-9; W_C=980e-9; W=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    return (W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))+W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2))/(W*(1+1j*w*tau_A))

C_CPW=46.53e-15; R_L=50.0; Rs=8.92
Cj = 161.0e-15    # C-V @ -5 V (user)

def _Y_Rp(w,Rp,Lrp):
    return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def sim_S11(w,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Z_dev=Zs+1/(1j*w*Cj)
    Y_n=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Z_dev
    Z_in=1j*w*Lcpw+1/Y_n
    return (Z_in-50)/(Z_in+50)
def H_ckt(w,Rp,Lcpw,Lrp,Lcpw2):
    Zs=Rs+1j*w*Lcpw2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cj+Y_A*(1+1j*w*Cj*Zs))
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Load -5 V S1P ───────────────────────────────────────────────────
def load_s1p(path):
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith(('!','#')): continue
            p=line.split()
            if len(p)>=3: rows.append([float(p[0]),float(p[1]),float(p[2])])
    a=np.array(rows)
    S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))
    return a[:,0], S

configs=[
    dict(lbl='Rp=200Ω', Rp=200.0, s1p='data_5V_5mA/S11_200ohm.s1p', fr='data_5V_5mA/200ohm.xlsx',
         col='#888888', mk='D', Lrp=153.7e-12, has_L2=False),
    dict(lbl='Rp=38Ω',  Rp=38.0,  s1p='data_5V_5mA/S11_38ohm.s1p',  fr='data_5V_5mA/38ohm.xlsx',
         col='#1B998B', mk='o', Lrp=65.6e-12,  has_L2=True),
    dict(lbl='Rp=60Ω',  Rp=60.0,  s1p='data_5V_5mA/S11_60ohm.s1p',  fr='data_5V_5mA/60ohm.xlsx',
         col='#FF8C00', mk='s', Lrp=71.8e-12,  has_L2=True),
    dict(lbl='Open',    Rp=np.inf,s1p='data_5V_5mA/S11_WO.s1p',     fr='data_5V_5mA/WO.xlsx',
         col='#E91E8C', mk='^', Lrp=0.0,       has_L2=False),
]
for cfg in configs:
    fr,S=load_s1p(cfg['s1p'])
    cfg['fs11']=fr; cfg['S11m']=S; cfg['ws']=2*np.pi*fr

def load_fr(path):
    df=pd.read_excel(path,header=14)
    f=pd.to_numeric(df.iloc[:,0],errors='coerce'); cal=pd.to_numeric(df.iloc[:,6],errors='coerce')
    m=f.notna()&cal.notna(); f,cal=f[m].values*1e9,cal[m].values
    pm=cal-cal[0]; keep=pm<2.0
    return f[keep],pm[keep]

# ── Option B fit at -5 V: [L_total, L2_38, L2_60] ───────────────────
def cost(p):
    L_tot,L2_38,L2_60=p
    if L2_38<0 or L2_60<0 or L2_38>L_tot or L2_60>L_tot: return 1e6
    tot=0.0
    for cfg in configs:
        L2 = L2_38 if cfg['lbl']=='Rp=38Ω' else (L2_60 if cfg['lbl']=='Rp=60Ω' else 0.0)
        S=sim_S11(cfg['ws'],cfg['Rp'],L_tot-L2,cfg['Lrp'],L2)
        tot+=np.mean(np.abs(S-cfg['S11m'])**2)
    return tot

bounds=[(120e-12,300e-12),(0,120e-12),(0,120e-12)]
res=differential_evolution(cost,bounds,seed=42,maxiter=3000,tol=1e-16,polish=False)
pol=minimize(cost,res.x,method='L-BFGS-B',bounds=bounds,
             options={'ftol':1e-20,'gtol':1e-16,'maxiter':50000})
L_tot,L2_38,L2_60=pol.x

print('='*100)
print(f'-5 V S11 fit (Option B)  |  Cj={Cj*1e15:.0f} fF fixed, FEM L_Rp, Rs={Rs}, C_CPW={C_CPW*1e15:.2f} fF')
print('='*100)
print(f'{"Param":>14} | {"-5 V fit":>10} | {"-7 V fit":>10} | consistency')
print('-'*60)
print(f'{"L_total":>14} | {L_tot*1e12:>8.1f}pH | {197.9:>8.1f}pH | Δ={L_tot*1e12-197.9:+.1f} pH')
print(f'{"L_CPW2 (38Ω)":>14} | {L2_38*1e12:>8.1f}pH | {56.3:>8.1f}pH | Δ={L2_38*1e12-56.3:+.1f} pH')
print(f'{"L_CPW2 (60Ω)":>14} | {L2_60*1e12:>8.1f}pH | {48.0:>8.1f}pH | Δ={L2_60*1e12-48.0:+.1f} pH')

f_plot=np.linspace(0.1e9,45e9,4000); w_plot=2*np.pi*f_plot
print()
print(f'{"Device":>10} | {"L_CPW":>7} | {"L_CPW2":>7} | {"RMS_S11":>8} | {"BW model":>8} | {"BW meas":>8} | {"RMS_H":>6}')
print('-'*80)

def draw_smith(ax,lw=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw+0.3))
    ax.axhline(0,color='#888',lw=lw,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#aaa',lw=lw,ls=':',zorder=0))
    th=np.linspace(0,np.pi,400)
    for x in [0.2,0.5,1,2,5]:
        for sg in [1,-1]:
            xx=1+ (1/x)*np.cos(th); yy=sg/x+(1/x)*np.sin(th)*sg
            m=(xx**2+yy**2<=1.002); ax.plot(xx[m],yy[m],color='#aaa',lw=lw,ls=':',zorder=0)

fig,axes=plt.subplots(3,4,figsize=(20,14))
fig.suptitle(f'-5 V S11 fit (Option B, Cj=161 fF)  |  L_total={L_tot*1e12:.1f} pH  '
             f'(vs 197.9 pH @ -7 V)\nRow1: Smith  Row2: |S11| dB  Row3: Freq response (-5 V, 5 mA)',
             fontsize=11,fontweight='bold')

os.makedirs('origin_data_5V_5mA',exist_ok=True)
Z0=50.0
for ci,cfg in enumerate(configs):
    L2 = L2_38 if cfg['lbl']=='Rp=38Ω' else (L2_60 if cfg['lbl']=='Rp=60Ω' else 0.0)
    Lc = L_tot-L2
    S=sim_S11(cfg['ws'],cfg['Rp'],Lc,cfg['Lrp'],L2)
    rms=np.sqrt(np.mean(np.abs(S-cfg['S11m'])**2))
    fm,pm=load_fr(cfg['fr']); wm=2*np.pi*fm
    Hm=H_ph(wm)*H_ckt(wm,cfg['Rp'],Lc,cfg['Lrp'],L2)
    Hp=H_ph(w_plot)*H_ckt(w_plot,cfg['Rp'],Lc,cfg['Lrp'],L2)
    Hd_m=20*np.log10(np.abs(Hm)/np.abs(Hm[0]))
    Hd_p=20*np.log10(np.abs(Hp)/np.abs(Hp[0]))
    rms_H=np.sqrt(np.mean((Hd_m-pm)**2)); bw=get_bw(f_plot,Hd_p)
    i3=np.where(pm<=-3)[0]; bwm=fm[i3[0]]/1e9 if len(i3) else np.nan
    bwm_s=f'{bwm:.1f}' if not np.isnan(bwm) else f'>{fm[-1]/1e9:.0f}'
    print(f'{cfg["lbl"]:>10} | {Lc*1e12:>6.1f}pH | {L2*1e12:>6.1f}pH | {rms:>8.5f} | '
          f'{bw:>7.1f}G | {bwm_s:>7}G | {rms_H:>6.2f}')

    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(cfg['S11m'].real,cfg['S11m'].imag,s=10,color=cfg['col'],zorder=6,label='Meas -5V')
    ax.plot(S.real,S.imag,'--',color='k',lw=1.5,zorder=5,label='Fit')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|={rms:.4f}',fontsize=10,fontweight='bold')
    ax.legend(fontsize=8.5,loc='lower left',framealpha=0.85,edgecolor='none',
              handlelength=1.5,markerscale=1.5)

    ax=axes[1,ci]
    ax.plot(cfg['fs11']/1e9,20*np.log10(np.abs(cfg['S11m'])),'-',color=cfg['col'],lw=1.0,label='Meas')
    ax.plot(cfg['fs11']/1e9,20*np.log10(np.abs(S)),'--',color='k',lw=1.4,label='Fit')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0,40); ax.legend(fontsize=8.5,loc='lower right'); ax.grid(True,alpha=0.3)

    ax=axes[2,ci]
    ax.scatter(fm/1e9,pm,color='k',marker=cfg['mk'],s=22,edgecolors=cfg['col'],
               linewidths=1.0,zorder=5,label='Meas (5 mA)')
    ax.plot(f_plot/1e9,Hd_p,'-',color=cfg['col'],lw=2.0,
            label=f'Model BW={bw:.1f}G, RMS={rms_H:.2f}dB')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Norm. H (dB)')
    ax.set_xlim(0,45); ax.set_ylim(-15,3)
    ax.legend(fontsize=8.5,loc='lower left'); ax.grid(True,alpha=0.3)

    # Origin export
    t=cfg['lbl'].replace('Ω','ohm').replace('=','_')
    zm=(1+cfg['S11m'])/(1-cfg['S11m']); zf=(1+S)/(1-S)
    with open(f'origin_data_5V_5mA/s11_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tS11_meas_dB\tS11_meas_deg\tS11_fit_dB\tS11_fit_deg\n')
        for i in range(len(cfg['fs11'])):
            fp.write(f'{cfg["fs11"][i]/1e9:.6g}\t{20*np.log10(abs(cfg["S11m"][i])):.6g}\t'
                     f'{np.angle(cfg["S11m"][i],deg=True):.6g}\t'
                     f'{20*np.log10(abs(S[i])):.6g}\t{np.angle(S[i],deg=True):.6g}\n')
    with open(f'origin_data_5V_5mA/impedance_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tr_meas\tx_meas\tr_fit\tx_fit\tR_meas_ohm\tX_meas_ohm\tR_fit_ohm\tX_fit_ohm\n')
        for i in range(len(cfg['fs11'])):
            fp.write(f'{cfg["fs11"][i]/1e9:.6g}\t{zm.real[i]:.6g}\t{zm.imag[i]:.6g}\t'
                     f'{zf.real[i]:.6g}\t{zf.imag[i]:.6g}\t{zm.real[i]*Z0:.6g}\t{zm.imag[i]*Z0:.6g}\t'
                     f'{zf.real[i]*Z0:.6g}\t{zf.imag[i]*Z0:.6g}\n')
    with open(f'origin_data_5V_5mA/freqresp_fit_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tNorm_dB_meas\tNorm_dB_model\n')
        for i in range(len(fm)):
            fp.write(f'{fm[i]/1e9:.6g}\t{pm[i]:.6g}\t{Hd_m[i]:.6g}\n')

fig.tight_layout()
fig.savefig('Fit_5V_result.png',dpi=150,bbox_inches='tight')
print('\nSaved: Fit_5V_result.png + origin_data_5V_5mA/{s11,impedance,freqresp_fit}_*.txt')
