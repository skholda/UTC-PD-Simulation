"""
CASE 8 — L_CPW between the shunt tap and the pad capacitance:
   Iph || C_PD -[Rs]- node1{ L_m + R_m } -[L_CPW]- node2{ C_CPW } - port
Free parameter: L_CPW only (per device).
Locked: C_PD=131 fF (C-V), Rs=8.92, C_CPW=46.53 fF, L_m=FEM, R_m design.
Transit time: latest staircase set (f_tr = 42.25 GHz).
"""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

Rs=8.92; Cpd=131e-15; Ccpw=46.53e-15; R_L=50.0
W_A,W_Ad,W_C=480e-9,160e-9,820e-9; W_norm=W_A+W_C+2*W_Ad
tau_A,tau_R=1.989e-12,0.0; tau_eD,tau_C=2.026e-12,7.794e-12; tau_h=W_Ad/4.8e4
def H_ph(w):
    s=lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            +W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            +W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            +W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm
def s11f(w,Rp,L,Lm):
    Z1=Rs+1/(1j*w*Cpd)
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Z2=1j*w*L+1/(Yrm+1/Z1)
    Zin=1/(1j*w*Ccpw+1/Z2)
    return (Zin-50)/(Zin+50)
def H_ckt(w,Rp,L,Lm):
    w=np.atleast_1d(np.asarray(w,dtype=float))
    A=np.ones_like(w,dtype=complex); B=np.zeros_like(w,dtype=complex)
    Cm=np.zeros_like(w,dtype=complex); D=np.ones_like(w,dtype=complex)
    def se(Z):
        nonlocal A,B,Cm,D; A,B,Cm,D=A,A*Z+B,Cm,Cm*Z+D
    def sh(Y):
        nonlocal A,B,Cm,D; A,B,Cm,D=A+B*Y,B,Cm+D*Y,D
    sh(1j*w*Cpd); se(Rs+0j*w)
    if not np.isinf(Rp): sh(1/(Rp+1j*w*Lm))
    se(1j*w*L); sh(1j*w*Ccpw)
    return R_L/(Cm*R_L+D)

_main=open('main.py').read(); _ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
          '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
ref_f,ref_loss=_ns['ref_f_GHz'],_ns['ref_loss_dB']

DEV=[('WO', np.inf,'_s1p_WO','_freq_WO',0.0,      '#E91E8C',30),
     ('200 ohm',200.0,'_s1p_200','_freq_200',153.7e-12,'#888888',30),
     ('60 ohm', 60.0,'_s1p_55','_freq_55',71.8e-12,'#FF8C00',35),
     ('38 ohm', 38.0,'_s1p_33','_freq_33',65.6e-12,'#1B998B',35)]

fit={}
print('CASE 8 fit (free: L_CPW only; C_CPW on the port side)')
for lbl,Rp,s1pk,frk,Lm,col,xmax in DEV:
    a=_ns[s1pk]; ws=2*np.pi*a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))
    cost=lambda L: np.sqrt(np.mean(np.abs(s11f(ws,Rp,L*1e-12,Lm)-S)**2))
    r=minimize_scalar(cost,bounds=(5,500),method='bounded')
    fit[lbl]=(r.x,r.fun)
    print(f'  {lbl:>8}: L_CPW={r.x:6.1f} pH   RMS={r.fun:.5f}')
print(f'  total RMS = {sum(v[1] for v in fit.values()):.5f}')

def draw_smith_labeled(ax):
    ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='k',lw=1.6)); ax.plot([-1,1],[0,0],color='k',lw=1.0)
    rs=[0.2,0.5,1.0,2.0,5.0]; RL={0.2:'0.2',0.5:'0.5',1.0:'1.0',2.0:'2.0',5.0:'5.0'}
    XL={0.2:'0.2j',0.5:'0.5j',1.0:'1.0j',2.0:'2.0j',5.0:'5.0j'}
    for r in rs:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='0.55',lw=0.6,ls=(0,(2,2))))
        ax.text((r-1)/(r+1),-0.06,RL[r],fontsize=11,ha='center',va='top')
    th=np.linspace(0,np.pi,400)
    for x in rs:
        for sgn in [1,-1]:
            xx=1.0+(1.0/x)*np.cos(th); yy=sgn/x+(1.0/x)*np.sin(th)*sgn
            msk=(xx**2+yy**2<=1.0005); ax.plot(xx[msk],yy[msk],color='0.55',lw=0.6,ls=(0,(2,2)))
            g=(1j*sgn*x-1)/(1j*sgn*x+1); ang=np.angle(g)
            ax.text(1.13*np.cos(ang),1.13*np.sin(ang),('-' if sgn<0 else '')+XL[x],fontsize=11,ha='center',va='center')

fig,axes=plt.subplots(2,2,figsize=(12.5,12))
for i,(lbl,Rp,s1pk,frk,Lm,col,xmax) in enumerate(DEV):
    ax=axes[i//2,i%2]; draw_smith_labeled(ax)
    a=_ns[s1pk]; fs=a[:,0]; S11m=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); ws=2*np.pi*fs
    ax.scatter(S11m.real,S11m.imag,s=26,facecolors='none',edgecolors=col,marker='s',linewidths=1.2,zorder=5)
    Sf=s11f(ws,Rp,fit[lbl][0]*1e-12,Lm)
    ax.plot(Sf.real,Sf.imag,'-',color=col,lw=2.6,zorder=6)
    ax.text(0.03,0.97,f'({chr(97+i)})',transform=ax.transAxes,fontsize=22,va='top')
    ax.text(0.30,0.87,f'Bias: -7 V\n{lbl}\nScatter: Measured\nSolid: Simulated',
            transform=ax.transAxes,fontsize=13.5,va='top',ha='left')
fig.tight_layout(); fig.savefig('case8_smith_7V.png',dpi=170,bbox_inches='tight',facecolor='white')
print('saved case8_smith_7V.png')

bw={}
fig,axes=plt.subplots(2,2,figsize=(13,10))
for i,(lbl,Rp,s1pk,frk,Lm,col,xmax) in enumerate(DEV):
    ax=axes[i//2,i%2]; L=fit[lbl][0]*1e-12
    af=_ns[frk]; fmG=af[:,0]; pm=af[:,1]+np.interp(fmG,ref_f,ref_loss); pm=pm-pm[0]
    f_plot=np.linspace(0.1e9,xmax*1e9,3000); w_plot=2*np.pi*f_plot
    H0=H_ph(1e-3)*H_ckt(np.array([1e-3]),Rp,L,Lm)
    Hdp=20*np.log10(np.abs(H_ph(w_plot)*H_ckt(w_plot,Rp,L,Lm))/np.abs(H0))
    fw=np.linspace(0.05e9,80e9,16000)
    Hw=20*np.log10(np.abs(H_ph(2*np.pi*fw)*H_ckt(2*np.pi*fw,Rp,L,Lm))/np.abs(H0))
    idx=np.where(Hw<=-3)[0]; bw[lbl]=fw[idx[0]]/1e9 if len(idx) else float('nan')
    ax.scatter(fmG,pm,s=34,facecolors='none',edgecolors=col,marker='s',linewidths=1.3,zorder=5)
    ax.plot(f_plot/1e9,Hdp,'-',color=col,lw=2.2,zorder=4)
    ax.set_xlim(0,xmax); ax.set_ylim(-15,10); ax.set_yticks([-15,-10,-5,0,5,10])
    ax.set_xlabel('Frequency [GHz]',fontsize=17)
    ax.set_ylabel(r'Normalized P$_{\mathrm{out}}$ [dB]',fontsize=17)
    ax.tick_params(labelsize=14,width=1.4,length=5)
    for sp in ax.spines.values(): sp.set_linewidth(1.6)
    ax.text(0.03,0.97,f'Diameter: 30 $\\mu$m\nResistance: {lbl}',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(0.03,0.16,'I$_{ph}$: 1 mA\nBias: -7  V',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(0.52,0.16,'Scatter: Measured\nSolid: Simulated',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(-0.16,1.06,f'({chr(97+i)})',transform=ax.transAxes,fontsize=22,va='top')
fig.tight_layout(); fig.savefig('case8_pout_7V.png',dpi=170,bbox_inches='tight',facecolor='white')
print('saved case8_pout_7V.png'); print('BW:',{k:round(v,1) for k,v in bw.items()})

def mk_table(ax,title,cols,rows,cw):
    ax.axis('off')
    if title: ax.set_title(title,fontsize=13,fontweight='bold',loc='left',pad=6)
    tb=ax.table(cellText=rows,colLabels=cols,loc='upper left',cellLoc='center',colLoc='center',colWidths=cw)
    tb.auto_set_font_size(False); tb.set_fontsize(11.5); tb.scale(1,1.5)
    for (r,c),cell in tb.get_celld().items():
        cell.set_edgecolor('0.25'); cell.set_linewidth(0.8)
        if r==0: cell.set_facecolor('#E8E8E8'); cell.set_text_props(fontweight='bold')
def mk_note(ax,t):
    ax.axis('off'); ax.text(0.02,0.95,t,transform=ax.transAxes,fontsize=10,style='italic',va='top')

fig=plt.figure(figsize=(10.5,11.6))
gs=fig.add_gridspec(7,1,height_ratios=[1.30,0.98,0.24,0.42,0.34,0.42,0.10],
                    hspace=0.75,top=0.925,bottom=0.02,left=0.03,right=0.985)
fig.suptitle('Circuit-model parameters  (free parameter: $L_{CPW}$)',fontsize=15,fontweight='bold',y=0.975)
mk_table(fig.add_subplot(gs[0]),'Table I.  Fixed parameters',
    ['Parameter','Value','Source'],
    [['C_PD','131 fF (−7 V)','C–V measurement (locked)'],
     ['R_S','8.92 Ω','Fixed'],
     ['C_CPW','46.53 fF','CPW pad (fixed)'],
     ['R_L','50 Ω','Load'],
     ['R_m','∞ / 200 / 60 / 38 Ω','Design values'],
     ['L_m','0 / 153.7 / 71.8 / 65.6 pH','FEM calculation (not fitted)']],
    [0.2,0.36,0.44])
mk_table(fig.add_subplot(gs[1]),'Table II.  Fitted parameter (S11 fit, −7 V)',
    ['Device','L_CPW (pH)','RMS of S11'],
    [[l.replace(' ohm',' Ω'),f'{fit[l][0]:.1f}',f'{fit[l][1]:.4f}'] for l,_,_,_,_,_,_ in DEV],
    [0.3,0.35,0.35])
mk_note(fig.add_subplot(gs[2]),
    'C_CPW sits on the port side of L_CPW, so the pad capacitance shunts the measurement plane\n'
    'directly. Total S11 RMS is 0.7922 versus 0.3378 when C_CPW shares the node with R_m + L_m.')
mk_table(fig.add_subplot(gs[3]),'Table III.  Transit-time parameters (no fitting)',
    ['τ_A (ps)','τ_eD (ps)','τ_C (ps)','τ_h (ps)','f_tr (GHz)'],
    [['1.989','2.026','7.794','3.333','42.25']],[0.2]*5)
mk_note(fig.add_subplot(gs[4]),
    'τ_A: staircase-doped undepleted absorber (5×10¹⁸ / 1.3×10¹⁸ / 4×10¹⁷ cm⁻³, Dₑ = 118 cm²/s);\n'
    'τ_eD, τ_C: layer-averaged v(E) on the simulated field;  τ_h: hole saturation velocity (0.48×10⁷ cm/s).')
mk_table(fig.add_subplot(gs[5]),'Table IV.  Simulated 3-dB bandwidth (−7 V)',
    ['Device','WO','200 Ω','60 Ω','38 Ω'],
    [['f_3dB (GHz)',f"{bw['WO']:.1f}",f"{bw['200 ohm']:.1f}",f"{bw['60 ohm']:.1f}",f"{bw['38 ohm']:.1f}"]],
    [0.24,0.19,0.19,0.19,0.19])
fig.add_subplot(gs[6]).axis('off')
fig.savefig('case8_tables.png',dpi=200,bbox_inches='tight',facecolor='white')
print('saved case8_tables.png')
