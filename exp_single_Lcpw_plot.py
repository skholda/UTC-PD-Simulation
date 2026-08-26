"""Plot: single-L_CPW fit vs locked Option-B S11 (Smith + |S11| dB)."""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

C_CPW,R_L,Cj,Rs = 46.53e-15,50.0,131.0e-15,8.92
def _Y_Rp(w,Rp,Lrp): return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def s11_single(w,Rp,Lcpw,Lrp):
    Zdev=Rs+1/(1j*w*Cj); Yn=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Zdev
    Zin=1j*w*Lcpw+1/Yn; return (Zin-50)/(Zin+50)
def s11_optB(w,Rp,Lcpw,Lcpw2,Lrp):
    Zs=Rs+1j*w*Lcpw2; Zdev=Zs+1/(1j*w*Cj); Yn=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Zdev
    Zin=1j*w*Lcpw+1/Yn; return (Zin-50)/(Zin+50)

_main=open('main.py').read(); _ns={'np':np}
for v in ['_s1p_200','_s1p_33','_s1p_55','_s1p_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
def gs1p(a): return a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))

DEV=[('200 Ω',200.0,'_s1p_200',153.7e-12,197.9e-12,0.0,'#888888'),
     ('38 Ω',38.0,'_s1p_33',65.6e-12,141.6e-12,56.3e-12,'#1B998B'),
     ('60 Ω',60.0,'_s1p_55',71.8e-12,149.9e-12,48.0e-12,'#FF8C00'),
     ('WO',np.inf,'_s1p_WO',0.0,197.9e-12,0.0,'#E91E8C')]

def draw_smith(ax,lw=0.6):
    ax.set_xlim(-1.08,1.08); ax.set_ylim(-1.08,1.08); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#888',lw=lw+0.3)); ax.axhline(0,color='#888',lw=lw,zorder=0)
    for r in [0.2,0.5,1,2,5]:
        cx,rad=r/(r+1),1/(r+1); ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='#ccc',lw=lw,ls=':',zorder=0))
    th=np.linspace(0,np.pi,300)
    for x in [0.2,0.5,1,2,5]:
        for s in [1,-1]:
            xx=1+(1/x)*np.cos(th); yy=s/x+(1/x)*np.sin(th)*s; m=(xx**2+yy**2<=1.002)
            ax.plot(xx[m],yy[m],color='#ccc',lw=lw,ls=':',zorder=0)

fig,axes=plt.subplots(2,4,figsize=(20,10))
fig.suptitle('S11 fit:  single $L_{CPW}$ (free)  vs  locked Option-B ($L_{CPW}+L_{CPW2}$)\n'
             'Row 1: Smith  |  Row 2: |S11| dB', fontsize=13, fontweight='bold')
for ci,(lbl,Rp,k,Lrp,LcpwB,Lcpw2B,col) in enumerate(DEV):
    fs,S11m=gs1p(_ns[k]); ws=2*np.pi*fs
    cost=lambda Lp: np.sqrt(np.mean(np.abs(s11_single(ws,Rp,Lp*1e-12,Lrp)-S11m)**2))
    r=minimize_scalar(cost,bounds=(50,400),method='bounded'); Lfit=r.x; rms1=r.fun
    S1=s11_single(ws,Rp,Lfit*1e-12,Lrp)
    SB=s11_optB(ws,Rp,LcpwB,Lcpw2B,Lrp); rmsB=np.sqrt(np.mean(np.abs(SB-S11m)**2))
    ax=axes[0,ci]; draw_smith(ax)
    ax.scatter(S11m.real,S11m.imag,s=10,color=col,zorder=6,label='Meas.')
    ax.plot(S1.real,S1.imag,'-',color='red',lw=1.6,zorder=5,label=f'single L (RMS {rms1:.3f})')
    ax.plot(SB.real,SB.imag,'--',color='k',lw=1.4,zorder=5,label=f'Opt-B (RMS {rmsB:.3f})')
    ax.set_title(f'{lbl}   $L_{{CPW}}$fit={Lfit:.0f}pH',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8,loc='lower left',framealpha=.85,edgecolor='none')
    ax=axes[1,ci]
    ax.plot(fs,20*np.log10(np.abs(S11m)),'-',color=col,lw=1.0,label='Meas.')
    ax.plot(fs,20*np.log10(np.abs(S1)),'-',color='red',lw=1.4,label='single L')
    ax.plot(fs,20*np.log10(np.abs(SB)),'--',color='k',lw=1.3,label='Opt-B')
    ax.set_xlabel('Freq (GHz)'); ax.set_ylabel('|S11| (dB)'); ax.set_xlim(0,fs.max())
    ax.grid(True,alpha=0.3); ax.legend(fontsize=8,loc='lower right')
fig.tight_layout()
fig.savefig('exp_single_Lcpw_fit.png',dpi=150,bbox_inches='tight')
print('saved exp_single_Lcpw_fit.png')
