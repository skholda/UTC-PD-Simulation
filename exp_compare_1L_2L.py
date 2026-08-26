"""
Side-by-side simulations, L_m fixed at FEM in BOTH cases:
  (A) 2-L ladder : Iph∥C_PD -Rs- [C_CPW] -L_CPW1- [Rm+Lm_FEM] -L_CPW2- port
      C_PD=137 fF, per-device (L_CPW1, L_CPW2) from ladder fit
  (B) single-L   : Iph∥C_PD -Rs- [C_CPW ∥ (Rm+Lm_FEM)] -L_CPW- port
      C_PD=148 fF, per-device L_CPW from single-L fit
Each produces its own full figure: Smith | |S11| dB | freq response.
"""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

Rs=8.92; C_CPW=46.53e-15; R_L=50.0

# ── H_ph (locked baseline) ─────────────────────────────────────────
W_A,W_Ad,W_C=480e-9,160e-9,820e-9; W_norm=W_A+W_C+2*W_Ad
tau_A,tau_R=3.530e-12,0.0; tau_eD,tau_C=2.026e-12,7.794e-12; tau_h=W_Ad/4.8e4
def H_ph(w):
    s=lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            +W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            +W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            +W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm

# ── generic ladder S11 / transimpedance via ABCD ───────────────────
def abcd(w, stages):
    w=np.atleast_1d(np.asarray(w,dtype=float))
    A=np.ones_like(w,dtype=complex); B=np.zeros_like(w,dtype=complex)
    C=np.zeros_like(w,dtype=complex); D=np.ones_like(w,dtype=complex)
    for kind,val in stages:
        Zy=val(w)
        if kind=='se': A,B,C,D = A, A*Zy+B, C, C*Zy+D
        else:          A,B,C,D = A+B*Zy, B, C+D*Zy, D
    return A,B,C,D
def stages_2L(Cpd,Rp,L1,L2,Lm):
    st=[('sh',lambda w: 1j*w*Cpd),('se',lambda w: Rs+0j*w),('sh',lambda w: 1j*w*C_CPW),
        ('se',lambda w: 1j*w*L1)]
    if not np.isinf(Rp): st.append(('sh',lambda w: 1/(Rp+1j*w*Lm)))
    st.append(('se',lambda w: 1j*w*L2))
    return st
def stages_1L(Cpd,Rp,L,Lm):
    st=[('sh',lambda w: 1j*w*Cpd),('se',lambda w: Rs+0j*w),('sh',lambda w: 1j*w*C_CPW)]
    if not np.isinf(Rp): st.append(('sh',lambda w: 1/(Rp+1j*w*Lm)))
    st.append(('se',lambda w: 1j*w*L))
    return st
def s11_from(stages, w):
    # input impedance seen from the port: reverse cascade == impedance looking left
    # build Zin directly by nodal reduction instead: traverse reversed
    # easier: Zin = (A*Zsrc+B)/(C*Zsrc+D) with source=open? No: port side.
    # For S11 at port we reduce left-to-right:
    Zy=None
    # nodal reduction: start with photodiode branch
    return None
def H_from(stages, w):
    A,B,C,D=abcd(w,stages)
    return R_L/(C*R_L+D)

# S11 via direct nodal reduction (works for both topologies)
def s11_2L(w,Cpd,Rp,L1,L2,Lm):
    Z1=Rs+1/(1j*w*Cpd); Y1=1j*w*C_CPW+1/Z1
    Z2=1j*w*L1+1/Y1
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y2=Yrm+1/Z2
    Zin=1j*w*L2+1/Y2
    return (Zin-50)/(Zin+50)
def s11_1L(w,Cpd,Rp,L,Lm):
    Z1=Rs+1/(1j*w*Cpd)
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y1=1j*w*C_CPW+Yrm+1/Z1
    Zin=1j*w*L+1/Y1
    return (Zin-50)/(Zin+50)

# ── measured data ──────────────────────────────────────────────────
_main=open('main.py').read(); _ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
          '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
ref_f,ref_loss=_ns['ref_f_GHz'],_ns['ref_loss_dB']

DEV=[('200Ω',200.0,'_s1p_200','_freq_200',153.7e-12,'#888888','D'),
     ('38Ω',38.0,'_s1p_33','_freq_33',65.6e-12,'#1B998B','o'),
     ('60Ω',60.0,'_s1p_55','_freq_55',71.8e-12,'#FF8C00','s'),
     ('Open',np.inf,'_s1p_WO','_freq_WO',0.0,'#E91E8C','^')]

# fitted parameter sets (L_m = FEM in both)
P2L={'lbl':'2-L ladder  (L_CPW1 + L_CPW2)','Cpd':137e-15,
     '200Ω':(5.2e-12,190.3e-12),'38Ω':(37.5e-12,146.0e-12),
     '60Ω':(32.2e-12,150.9e-12),'Open':(129.4e-12,54.6e-12)}
P1L={'lbl':'single L_CPW','Cpd':148e-15,
     '200Ω':186.0e-12,'38Ω':155.7e-12,'60Ω':165.0e-12,'Open':173.7e-12}

def draw_smith(ax,lw=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw+0.3)); ax.axhline(0,color='#888',lw=lw,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1); ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#ccc',lw=lw,ls=':',zorder=0))
    th=np.linspace(0,np.pi,300)
    for x in [0.2,0.5,1,2,5]:
        for s in [1,-1]:
            xx=1+(1/x)*np.cos(th); yy=s/x+(1/x)*np.sin(th)*s; m2=(xx**2+yy**2<=1.002)
            ax.plot(xx[m2],yy[m2],color='#ccc',lw=lw,ls=':',zorder=0)

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot

def run_case(name, tag, s11fun, Hstages, subtitle, fname):
    fig,axes=plt.subplots(3,4,figsize=(20,14))
    fig.suptitle(subtitle,fontsize=12,fontweight='bold')
    print('\n'+'='*84); print(name); print('='*84)
    print(f'{"Device":>7} | {"RMS_S11":>9} | {"BW":>7} | {"RMS_H":>7}')
    for ci,(lbl,Rp,s1pk,frk,Lm,col,mk) in enumerate(DEV):
        a=_ns[s1pk]; fs=a[:,0]; S11m=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); ws=2*np.pi*fs
        S=s11fun(ws,lbl,Rp,Lm); rms=np.sqrt(np.mean(np.abs(S-S11m)**2))
        af=_ns[frk]; fm=af[:,0]*1e9; pm=af[:,1]+np.interp(af[:,0],ref_f,ref_loss); pm=pm-pm[0]
        wm=2*np.pi*fm
        Hfun=lambda w,lbl=lbl,Rp=Rp,Lm=Lm: H_from(Hstages(lbl,Rp,Lm),w)
        Hm=H_ph(wm)*Hfun(wm); H0=H_ph(1e-3)*Hfun(np.array([1e-3]))
        Hdm=20*np.log10(np.abs(Hm)/np.abs(H0))
        Hp=H_ph(w_plot)*Hfun(w_plot); Hdp=20*np.log10(np.abs(Hp)/np.abs(H0))
        rmsH=np.sqrt(np.mean((Hdm-pm)**2))
        i=np.where(Hdp<=-3)[0]; bw=f_plot[i[0]]/1e9 if len(i) else np.nan
        print(f'{lbl:>7} | {rms:9.5f} | {bw:6.1f}G | {rmsH:7.3f}')
        ax=axes[0,ci]; draw_smith(ax)
        ax.scatter(S11m.real,S11m.imag,s=10,color=col,zorder=6,label='Meas.')
        ax.plot(S.real,S.imag,'--',color='k',lw=1.5,zorder=5,label='Fit')
        ax.set_title(f'{lbl}  RMS={rms:.4f}',fontsize=10,fontweight='bold')
        ax.legend(fontsize=8,loc='lower left',framealpha=.85,edgecolor='none')
        ax=axes[1,ci]
        ax.plot(fs,20*np.log10(np.abs(S11m)),'-',color=col,lw=1.0,label='Meas.')
        ax.plot(fs,20*np.log10(np.abs(S)),'--',color='k',lw=1.4,label='Fit')
        ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('|S11| dB'); ax.set_xlim(0,fs.max())
        ax.grid(True,alpha=0.3); ax.legend(fontsize=8,loc='lower right')
        ax=axes[2,ci]
        ax.scatter(fm/1e9,pm,color='k',marker=mk,s=22,edgecolors=col,linewidths=1.0,zorder=5,label='Meas.')
        ax.plot(f_plot/1e9,Hdp,'-',color=col,lw=2.0,label=f'Sim BW={bw:.1f}G RMS={rmsH:.2f}dB')
        ax.axhline(-3,color='gray',ls=':',lw=0.7)
        ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('Norm H (dB)'); ax.set_xlim(0,50); ax.set_ylim(-12,3)
        ax.grid(True,alpha=0.3); ax.legend(fontsize=8,loc='lower left')
    fig.tight_layout(); fig.savefig(fname,dpi=150,bbox_inches='tight')
    print(f'saved {fname}')

# (A) 2-L ladder
run_case('(A) 2-L ladder  |  C_PD=137 fF, L_m=FEM','2L',
    lambda ws,lbl,Rp,Lm: s11_2L(ws,P2L['Cpd'],Rp,P2L[lbl][0],P2L[lbl][1],Lm),
    lambda lbl,Rp,Lm: stages_2L(P2L['Cpd'],Rp,P2L[lbl][0],P2L[lbl][1],Lm),
    'Topology A — 2-L ladder ($L_{CPW1}$ + $L_{CPW2}$),  $C_{PD}$=137 fF,  $L_m$=FEM\n'
    'Row1: Smith | Row2: |S11| dB | Row3: Freq response',
    'compare_2L_full.png')

# (B) single L
run_case('(B) single L_CPW  |  C_PD=148 fF, L_m=FEM','1L',
    lambda ws,lbl,Rp,Lm: s11_1L(ws,P1L['Cpd'],Rp,P1L[lbl],Lm),
    lambda lbl,Rp,Lm: stages_1L(P1L['Cpd'],Rp,P1L[lbl],Lm),
    'Topology B — single $L_{CPW}$,  $C_{PD}$=148 fF,  $L_m$=FEM\n'
    'Row1: Smith | Row2: |S11| dB | Row3: Freq response',
    'compare_1L_full.png')
