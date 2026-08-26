"""
Experiment: single-L_CPW schematic
  Iph ∥ C_PD ─[R_S]─ node1{ C_CPW ∥ (R_m+L_m) } ─[L_CPW]─ port (R_L)
(= previous ladder with L_CPW1 = 0; both shunts share one node.)
Fit: common C_PD + per-device L_CPW.  Fixed: R_S, C_CPW, FEM L_m.
Compare S11 RMS with the 2-L ladder baseline.
"""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

Rs=8.92; C_CPW=46.53e-15; R_L=50.0
def S11_1L(w,Cpd,Rp,Lcpw,Lm):
    Z1=Rs+1/(1j*w*Cpd)
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y1=1j*w*C_CPW+Yrm+1/Z1
    Zin=1j*w*Lcpw+1/Y1
    return (Zin-50)/(Zin+50)

def S11_2L(w,Cpd,Rp,L1,L2,Lm):   # baseline ladder for reference
    Z1=Rs+1/(1j*w*Cpd); Y1=1j*w*C_CPW+1/Z1
    Z2=1j*w*L1+1/Y1
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y2=Yrm+1/Z2
    return (1j*w*L2+1/Y2-50)/(1j*w*L2+1/Y2+50)

_main=open('main.py').read(); _ns={'np':np}
for v in ['_s1p_200','_s1p_33','_s1p_55','_s1p_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
def gs1p(a): return a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))

DEV=[('200Ω',200.0,'_s1p_200',153.7e-12,(5.2e-12,190.3e-12),'#888888'),
     ('38Ω',38.0,'_s1p_33',65.6e-12,(37.5e-12,146.0e-12),'#1B998B'),
     ('60Ω',60.0,'_s1p_55',71.8e-12,(32.2e-12,150.9e-12),'#FF8C00'),
     ('Open',np.inf,'_s1p_WO',0.0,(129.4e-12,54.6e-12),'#E91E8C')]
data={lbl:gs1p(_ns[k]) for lbl,_,k,_1,_2,_3 in DEV}

def fit_L(Cpd,Rp,ws,S11m,Lm):
    cost=lambda Lp: np.sqrt(np.mean(np.abs(S11_1L(ws,Cpd,Rp,Lp*1e-12,Lm)-S11m)**2))
    r=minimize_scalar(cost,bounds=(20,400),method='bounded')
    return r.x, r.fun

# scan common C_PD
best=None
for Cf in np.arange(100,220,1):
    tot=0; out={}
    for lbl,Rp,k,Lm,_,_c in DEV:
        fs,S11m=data[lbl]; ws=2*np.pi*fs
        L,rms=fit_L(Cf*1e-15,Rp,ws,S11m,Lm); out[lbl]=(L,rms); tot+=rms
    if best is None or tot<best[0]: best=(tot,Cf,out)
tot,Cf,out=best

print('='*84)
print(f'Single-L_CPW schematic fit  |  common C_PD = {Cf:.0f} fF')
print('='*84)
print(f'{"Device":>7} | {"L_CPW":>8} | {"RMS_S11":>9} || {"ladder RMS (2-L)":>16}')
print('-'*84)
LAD={'200Ω':0.06331,'38Ω':0.05648,'60Ω':0.05978,'Open':0.07393}
for lbl,Rp,k,Lm,l12,c in DEV:
    L,rms=out[lbl]
    print(f'{lbl:>7} | {L:6.1f}pH | {rms:9.5f} || {LAD[lbl]:16.5f}')
print('-'*84)
print(f'total RMS = {tot:.5f}   (2-L ladder total = 0.25350)')

# plot
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

fig,axes=plt.subplots(2,4,figsize=(20,10))
fig.suptitle(f'Single-$L_{{CPW}}$ schematic fit (red, $C_{{PD}}$={Cf:.0f} fF) vs 2-L ladder baseline (black dashed)\n'
             'Row1: Smith | Row2: |S11| dB',fontsize=12,fontweight='bold')
for ci,(lbl,Rp,k,Lm,(L1b,L2b),col) in enumerate(DEV):
    fs,S11m=data[lbl]; ws=2*np.pi*fs
    L,rms=out[lbl]
    S1=S11_1L(ws,Cf*1e-15,Rp,L*1e-12,Lm)
    S2=S11_2L(ws,137e-15,Rp,L1b,L2b,Lm)
    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(S11m.real,S11m.imag,s=10,color=col,zorder=6,label='Meas.')
    ax.plot(S1.real,S1.imag,'-',color='red',lw=1.6,zorder=5,label=f'1-L (RMS {rms:.4f})')
    ax.plot(S2.real,S2.imag,'--',color='k',lw=1.4,zorder=5,label=f'2-L ladder ({LAD[lbl]:.4f})')
    ax.set_title(f'{lbl}   $L_{{CPW}}$={L:.0f}pH',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8,loc='lower left',framealpha=.85,edgecolor='none')
    ax=axes[1,ci]
    ax.plot(fs,20*np.log10(np.abs(S11m)),'-',color=col,lw=1.0,label='Meas.')
    ax.plot(fs,20*np.log10(np.abs(S1)),'-',color='red',lw=1.4,label='1-L')
    ax.plot(fs,20*np.log10(np.abs(S2)),'--',color='k',lw=1.3,label='2-L ladder')
    ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('|S11| dB'); ax.set_xlim(0,fs.max())
    ax.grid(True,alpha=0.3); ax.legend(fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig('exp_single_L_ladder.png',dpi=150,bbox_inches='tight')
print('saved exp_single_L_ladder.png')
