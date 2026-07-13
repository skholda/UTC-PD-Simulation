"""
-5 V full result with Cj = 166.44 fF (S11 common refit)
========================================================
All L values locked at -7 V fit (bias-independent), FEM L_Rp.
Only Cj changed via -5V S11 common fit.
Outputs: 3-row figure + origin data (s11, impedance, freqresp meas+model,
model dense 40 GHz, absolute dBm).
"""
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt

W_A=480e-9; W_C=980e-9; W=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    return (W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))+W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2))/(W*(1+1j*w*tau_A))
C_CPW=46.53e-15; R_L=50.0; Rs=8.92
Cj=166.44e-15                       # S11 common refit @ -5 V
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
def load_s1p(path):
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith(('!','#')): continue
            p=line.split()
            if len(p)>=3: rows.append([float(p[0]),float(p[1]),float(p[2])])
    a=np.array(rows)
    return a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))
def load_fr(path):
    df=pd.read_excel(path,header=14)
    f=pd.to_numeric(df.iloc[:,0],errors='coerce'); cal=pd.to_numeric(df.iloc[:,6],errors='coerce')
    m=f.notna()&cal.notna(); f,cal=f[m].values*1e9,cal[m].values
    pm=cal-cal[0]; keep=pm<2.0
    return f[keep],pm[keep],cal[keep]

configs=[
    dict(lbl='Rp=200Ω', tag='Rp_200ohm', Rp=200.0, s1p='data_5V_5mA/S11_200ohm.s1p',
         fr='data_5V_5mA/200ohm.xlsx', col='#888888', mk='D',
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),
    dict(lbl='Rp=38Ω',  tag='Rp_38ohm',  Rp=38.0,  s1p='data_5V_5mA/S11_38ohm.s1p',
         fr='data_5V_5mA/38ohm.xlsx',  col='#1B998B', mk='o',
         Lcpw=141.6e-12, Lcpw2=56.3e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  tag='Rp_60ohm',  Rp=60.0,  s1p='data_5V_5mA/S11_60ohm.s1p',
         fr='data_5V_5mA/60ohm.xlsx',  col='#FF8C00', mk='s',
         Lcpw=149.9e-12, Lcpw2=48.0e-12, Lrp=71.8e-12),
    dict(lbl='Open',    tag='Open',      Rp=np.inf,s1p='data_5V_5mA/S11_WO.s1p',
         fr='data_5V_5mA/WO.xlsx',     col='#E91E8C', mk='^',
         Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=0.0),
]

f_plot=np.linspace(0.1e9,40e9,4000); w_plot=2*np.pi*f_plot
Iph=5.0e-3
out='origin_data_5V_5mA_Cj166'; os.makedirs(out,exist_ok=True)
Z0=50.0

print('='*95)
print(f'-5 V result  |  Cj = {Cj*1e15:.2f} fF (S11 common fit)  |  -7 V L values, FEM L_Rp')
print('='*95)
print(f'{"Device":>10} | {"RMS_S11":>8} | {"BW model":>8} | {"BW meas":>8} | {"RMS_H":>6}')
print('-'*60)

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
            xx=1+(1/x)*np.cos(th); yy=sg/x+(1/x)*np.sin(th)*sg
            m=(xx**2+yy**2<=1.002); ax.plot(xx[m],yy[m],color='#aaa',lw=lw,ls=':',zorder=0)

fig,axes=plt.subplots(3,4,figsize=(20,14))
fig.suptitle(f'-5 V / 5 mA  |  Cj = {Cj*1e15:.1f} fF (S11 refit; C-V: 161 fF)  |  '
             f'-7 V L values (bias-independent), FEM L_Rp\n'
             'Row1: Smith  Row2: |S11| dB  Row3: Freq response',
             fontsize=11,fontweight='bold')

for ci,cfg in enumerate(configs):
    fs11,S11m=load_s1p(cfg['s1p']); ws=2*np.pi*fs11
    S=sim_S11(ws,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    rms=np.sqrt(np.mean(np.abs(S-S11m)**2))
    fm,pm,cal=load_fr(cfg['fr']); wm=2*np.pi*fm
    Hm=H_ph(wm)*H_ckt(wm,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hp=H_ph(w_plot)*H_ckt(w_plot,cfg['Rp'],cfg['Lcpw'],cfg['Lrp'],cfg['Lcpw2'])
    Hd_m=20*np.log10(np.abs(Hm)/np.abs(Hm[0]))
    Hd_p=20*np.log10(np.abs(Hp)/np.abs(Hp[0]))
    P_abs=10*np.log10((np.abs(Iph*Hp)**2/(2*R_L))/1e-3)
    P_abs_m=10*np.log10((np.abs(Iph*Hm)**2/(2*R_L))/1e-3)
    rms_H=np.sqrt(np.mean((Hd_m-pm)**2)); bw=get_bw(f_plot,Hd_p)
    i3=np.where(pm<=-3)[0]; bwm=fm[i3[0]]/1e9 if len(i3) else np.nan
    bwm_s=f'{bwm:.1f}' if not np.isnan(bwm) else f'>{fm[-1]/1e9:.0f}'
    print(f'{cfg["lbl"]:>10} | {rms:>8.5f} | {bw:>7.1f}G | {bwm_s:>7}G | {rms_H:>6.2f}')

    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(S11m.real,S11m.imag,s=10,color=cfg['col'],zorder=6,label='Meas -5V')
    ax.plot(S.real,S.imag,'--',color='k',lw=1.5,zorder=5,label=f'Model Cj={Cj*1e15:.1f}fF')
    ax.set_title(f'{cfg["lbl"]}\nRMS|ΔΓ|={rms:.4f}',fontsize=10,fontweight='bold')
    ax.legend(fontsize=8.5,loc='lower left',framealpha=0.85,edgecolor='none',
              handlelength=1.5,markerscale=1.5)

    ax=axes[1,ci]
    ax.plot(fs11/1e9,20*np.log10(np.abs(S11m)),'-',color=cfg['col'],lw=1.0,label='Meas')
    ax.plot(fs11/1e9,20*np.log10(np.abs(S)),'--',color='k',lw=1.4,label='Model')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S11| (dB)')
    ax.set_xlim(0,40); ax.legend(fontsize=8.5,loc='lower right'); ax.grid(True,alpha=0.3)

    ax=axes[2,ci]
    ax.scatter(fm/1e9,pm,color='k',marker=cfg['mk'],s=22,edgecolors=cfg['col'],
               linewidths=1.0,zorder=5,label='Meas (5 mA)')
    ax.plot(f_plot/1e9,Hd_p,'-',color=cfg['col'],lw=2.0,
            label=f'Model BW={bw:.1f}G, RMS={rms_H:.2f}dB')
    ax.axhline(-3,color='gray',ls=':',lw=0.7,alpha=0.7)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Norm. H (dB)')
    ax.set_xlim(0,40); ax.set_ylim(-15,3)
    ax.legend(fontsize=8.5,loc='lower left'); ax.grid(True,alpha=0.3)

    # Origin exports
    t=cfg['tag']
    zm=(1+S11m)/(1-S11m); zf=(1+S)/(1-S)
    with open(f'{out}/s11_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tS11_meas_dB\tS11_meas_deg\tS11_model_dB\tS11_model_deg\n')
        for i in range(len(fs11)):
            fp.write(f'{fs11[i]/1e9:.6g}\t{20*np.log10(abs(S11m[i])):.6g}\t'
                     f'{np.angle(S11m[i],deg=True):.6g}\t'
                     f'{20*np.log10(abs(S[i])):.6g}\t{np.angle(S[i],deg=True):.6g}\n')
    with open(f'{out}/smith_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tRe_S11_meas\tIm_S11_meas\tRe_S11_model\tIm_S11_model\n')
        for i in range(len(fs11)):
            fp.write(f'{fs11[i]/1e9:.6g}\t{S11m.real[i]:.6g}\t{S11m.imag[i]:.6g}\t'
                     f'{S.real[i]:.6g}\t{S.imag[i]:.6g}\n')
    with open(f'{out}/impedance_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tr_meas\tx_meas\tr_model\tx_model\tR_meas_ohm\tX_meas_ohm\tR_model_ohm\tX_model_ohm\n')
        for i in range(len(fs11)):
            fp.write(f'{fs11[i]/1e9:.6g}\t{zm.real[i]:.6g}\t{zm.imag[i]:.6g}\t'
                     f'{zf.real[i]:.6g}\t{zf.imag[i]:.6g}\t{zm.real[i]*Z0:.6g}\t{zm.imag[i]*Z0:.6g}\t'
                     f'{zf.real[i]*Z0:.6g}\t{zf.imag[i]*Z0:.6g}\n')
    with open(f'{out}/freqresp_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tNorm_dB_meas\tNorm_dB_model\tCalRF_dBm_meas\tModel_dBm_5mA\n')
        for i in range(len(fm)):
            fp.write(f'{fm[i]/1e9:.6g}\t{pm[i]:.6g}\t{Hd_m[i]:.6g}\t{cal[i]:.6g}\t{P_abs_m[i]:.6g}\n')
    with open(f'{out}/model_40GHz_{t}.txt','w') as fp:
        fp.write('Freq_GHz\tModel_Norm_dB\tModel_dBm_5mA\n')
        for i in range(len(f_plot)):
            fp.write(f'{f_plot[i]/1e9:.6g}\t{Hd_p[i]:.6g}\t{P_abs[i]:.6g}\n')

fig.tight_layout()
fig.savefig('Eval_5V_Cj166.png',dpi=150,bbox_inches='tight')
print(f'\nSaved: Eval_5V_Cj166.png + {out}/ (5 file types x 4 devices)')
