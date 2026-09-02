"""
CASE 3 — 2-L ladder, free params = L_CPW1, L_CPW2 only,
         with the PREVIOUS transit-time set (continuous-grading tau_A = 3.530 ps).
Publication-style figures matching the user's example images:
  case3_smith_7V.png : 2x2 Smith  (a) WO (b) 200 (c) 60 (d) 38
  case3_pout_7V.png  : 2x2 Normalized P_out [dB]
Locked: C_PD=131 fF (C-V), Rs=8.92, C_CPW=46.53 fF, L_m=FEM, R_m design.
H_ph: staircase tau_A baseline (f_tr=42.25 GHz).
"""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

Rs=8.92; Cpd=131e-15; Ccpw=46.53e-15; R_L=50.0
W_A,W_Ad,W_C=480e-9,160e-9,820e-9; W_norm=W_A+W_C+2*W_Ad
tau_A,tau_R=3.530e-12,0.0; tau_eD,tau_C=2.026e-12,7.794e-12; tau_h=W_Ad/4.8e4
def H_ph(w):
    s=lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            +W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            +W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            +W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm
def s11_2L(w,Rp,L1,L2,Lm):
    Z1=Rs+1/(1j*w*Cpd); Y1=1j*w*Ccpw+1/Z1; Z2=1j*w*L1+1/Y1
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Zin=1j*w*L2+1/(Yrm+1/Z2); return (Zin-50)/(Zin+50)
def H_2L(w,Rp,L1,L2,Lm):
    w=np.atleast_1d(np.asarray(w,dtype=float))
    A=np.ones_like(w,dtype=complex); B=np.zeros_like(w,dtype=complex)
    C=np.zeros_like(w,dtype=complex); D=np.ones_like(w,dtype=complex)
    def se(Z):
        nonlocal A,B,C,D; A,B,C,D=A,A*Z+B,C,C*Z+D
    def sh(Y):
        nonlocal A,B,C,D; A,B,C,D=A+B*Y,B,C+D*Y,D
    sh(1j*w*Cpd); se(Rs+0j*w); sh(1j*w*Ccpw); se(1j*w*L1)
    if not np.isinf(Rp): sh(1/(Rp+1j*w*Lm))
    se(1j*w*L2); return R_L/(C*R_L+D)

_main=open('main.py').read(); _ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
          '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
ref_f,ref_loss=_ns['ref_f_GHz'],_ns['ref_loss_dB']

# panel order/colors/fit values (L1,L2 in pH) per user's example
DEV=[('WO', np.inf,'_s1p_WO','_freq_WO',0.0,      (0.0,190.0),'#E91E8C',30),
     ('200 ohm',200.0,'_s1p_200','_freq_200',153.7e-12,(9.4,191.7),'#888888',30),
     ('60 ohm', 60.0,'_s1p_55','_freq_55',71.8e-12,(37.5,150.9),'#FF8C00',35),
     ('38 ohm', 38.0,'_s1p_33','_freq_33',65.6e-12,(42.4,145.8),'#1B998B',35)]

def draw_smith_labeled(ax):
    ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='k',lw=1.6))
    ax.plot([-1,1],[0,0],color='k',lw=1.0)
    rs=[0.2,0.5,1.0,2.0,5.0]
    for r in rs:
        cx,rad=r/(r+1),1/(r+1)
        ax.add_patch(plt.Circle((cx,0),rad,fill=False,color='0.55',lw=0.6,ls=(0,(2,2))))
        gx=(r-1)/(r+1)
        RL={0.2:'0.2',0.5:'0.5',1.0:'1.0',2.0:'2.0',5.0:'5.0'}
        ax.text(gx,-0.06,RL[r],fontsize=11,ha='center',va='top',color='k')
    th=np.linspace(0,np.pi,400)
    for x in rs:
        for sgn in [1,-1]:
            cx,cy,rad=1.0,sgn/x,1.0/x
            xx=cx+rad*np.cos(th); yy=cy+rad*np.sin(th)*sgn
            msk=(xx**2+yy**2<=1.0005)
            ax.plot(xx[msk],yy[msk],color='0.55',lw=0.6,ls=(0,(2,2)))
            g=(1j*sgn*x-1)/(1j*sgn*x+1); ang=np.angle(g)
            XL={0.2:'0.2j',0.5:'0.5j',1.0:'1.0j',2.0:'2.0j',5.0:'5.0j'}
            ax.text(1.13*np.cos(ang),1.13*np.sin(ang),('-' if sgn<0 else '')+XL[x],
                    fontsize=11,ha='center',va='center')

# ── Figure 1: Smith ────────────────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(12.5,12))
for i,(lbl,Rp,s1pk,frk,Lm,(L1,L2),col,xmax) in enumerate(DEV):
    ax=axes[i//2,i%2]; draw_smith_labeled(ax)
    a=_ns[s1pk]; fs=a[:,0]; S11m=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); ws=2*np.pi*fs
    ax.scatter(S11m.real,S11m.imag,s=26,facecolors='none',edgecolors=col,marker='s',linewidths=1.2,zorder=5)
    Sf=s11_2L(ws,Rp,L1*1e-12,L2*1e-12,Lm)
    ax.plot(Sf.real,Sf.imag,'-',color=col,lw=2.6,zorder=6)
    ax.text(-0.15,0.150,f'({chr(97+i)})',transform=ax.transAxes,fontsize=22,fontweight='bold',va='top') if False else None
    ax.text(0.03,0.97,f'({chr(97+i)})',transform=ax.transAxes,fontsize=22,va='top')
    tx=0.28 if i<2 else 0.34
    ax.text(0.30,0.87,f'Bias: -7 V\n{lbl}\nScatter: Measured\nSolid: Simulated',
            transform=ax.transAxes,fontsize=13.5,va='top',ha='left')
fig.tight_layout()
fig.savefig('case3_smith_7V.png',dpi=170,bbox_inches='tight',facecolor='white')
print('saved case3_smith_7V.png')

# ── Figure 2: Normalized P_out ─────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(13,10))
for i,(lbl,Rp,s1pk,frk,Lm,(L1,L2),col,xmax) in enumerate(DEV):
    ax=axes[i//2,i%2]
    af=_ns[frk]; fmG=af[:,0]; pm=af[:,1]+np.interp(fmG,ref_f,ref_loss); pm=pm-pm[0]
    f_plot=np.linspace(0.1e9,xmax*1e9,3000); w_plot=2*np.pi*f_plot
    H0=H_ph(1e-3)*H_2L(np.array([1e-3]),Rp,L1*1e-12,L2*1e-12,Lm)
    Hdp=20*np.log10(np.abs(H_ph(w_plot)*H_2L(w_plot,Rp,L1*1e-12,L2*1e-12,Lm))/np.abs(H0))
    ax.scatter(fmG,pm,s=34,facecolors='none',edgecolors=col,marker='s',linewidths=1.3,zorder=5)
    ax.plot(f_plot/1e9,Hdp,'-',color=col,lw=2.2,zorder=4)
    ax.set_xlim(0,xmax); ax.set_ylim(-15,10)
    ax.set_yticks([-15,-10,-5,0,5,10])
    ax.set_xlabel('Frequency [GHz]',fontsize=17)
    ax.set_ylabel(r'Normalized P$_{\mathrm{out}}$ [dB]',fontsize=17)
    ax.tick_params(labelsize=14,width=1.4,length=5)
    for sp in ax.spines.values(): sp.set_linewidth(1.6)
    ax.text(0.03,0.97,f'Diameter: 30 $\\mu$m\nResistance: {lbl}',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(0.03,0.16,'I$_{ph}$: 1 mA\nBias: -7  V',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(0.52,0.16,'Scatter: Measured\nSolid: Simulated',transform=ax.transAxes,fontsize=13.5,va='top')
    ax.text(-0.16,1.06,f'({chr(97+i)})',transform=ax.transAxes,fontsize=22,va='top')
fig.tight_layout()
fig.savefig('case3_pout_7V.png',dpi=170,bbox_inches='tight',facecolor='white')
print('saved case3_pout_7V.png')

# ── Parameter tables rendered as an image (English, for Word) ──────
# See case_2L_tables.png; generated by the table block kept in git history.


# ── bandwidth + tables ────────────────────────────────────────────
bw={}
for lbl,Rp,s1pk,frk,Lm,(L1,L2),col,xmax in DEV:
    args=(Rp,L1*1e-12,L2*1e-12,Lm)
    H0=H_ph(1e-3)*H_2L(np.array([1e-3]),*args)
    fw=np.linspace(0.05e9,80e9,16000)
    Hw=20*np.log10(np.abs(H_ph(2*np.pi*fw)*H_2L(2*np.pi*fw,*args))/np.abs(H0))
    i=np.where(Hw<=-3)[0]; bw[lbl]=fw[i[0]]/1e9 if len(i) else float('nan')
print('BW:',{k:round(v,1) for k,v in bw.items()})

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
gs=fig.add_gridspec(7,1,height_ratios=[1.30,0.98,0.16,0.42,0.40,0.42,0.10],
                    hspace=0.75,top=0.925,bottom=0.02,left=0.03,right=0.985)
fig.suptitle('Circuit-model parameters  (free parameters: $L_{CPW1}$, $L_{CPW2}$)',
             fontsize=15,fontweight='bold',y=0.975)
mk_table(fig.add_subplot(gs[0]),'Table I.  Fixed parameters',
    ['Parameter','Value','Source'],
    [['C_PD','131 fF (\u22127 V)','C\u2013V measurement (locked)'],
     ['R_S','8.92 \u03a9','Fixed'],
     ['C_CPW','46.53 fF','CPW pad (fixed)'],
     ['R_L','50 \u03a9','Load'],
     ['R_m','\u221e / 200 / 60 / 38 \u03a9','Design values'],
     ['L_m','0 / 153.7 / 71.8 / 65.6 pH','FEM calculation (not fitted)']],
    [0.2,0.36,0.44])
mk_table(fig.add_subplot(gs[1]),'Table II.  Fitted parameters (S11 fit, \u22127 V)',
    ['Device','L_CPW1 (pH)','L_CPW2 (pH)','Sum (pH)','RMS of S11'],
    [['WO','0.0 *','190.0','190.0','0.0681'],
     ['200 \u03a9','9.4','191.7','201.1','0.0712'],
     ['60 \u03a9','37.5','150.9','188.4','0.0603'],
     ['38 \u03a9','42.4','145.8','188.2','0.0581']],
    [0.16,0.21,0.21,0.18,0.24])
mk_note(fig.add_subplot(gs[2]),
    '* No shunt branch (WO): only the sum L_CPW1 + L_CPW2 is identifiable; L_CPW1 set to 0.')
mk_table(fig.add_subplot(gs[3]),'Table III.  Transit-time parameters (no fitting)',
    ['\u03c4_A (ps)','\u03c4_eD (ps)','\u03c4_C (ps)','\u03c4_h (ps)','f_tr (GHz)'],
    [['3.530','2.026','7.794','3.333','32.62']],[0.2]*5)
mk_note(fig.add_subplot(gs[4]),
    '\u03c4_A: continuous-grading formula W_A\u00b2 / [D\u2091 (3 + ln(p_max/p_min))] with p_max/p_min = 12.5,\n'
    'D\u2091 = 118 cm\u00b2/s.  \u03c4_eD, \u03c4_C: layer-averaged v(E) on the simulated field;\n'
    '\u03c4_h: hole saturation velocity (0.48\u00d710\u2077 cm/s).')
mk_table(fig.add_subplot(gs[5]),'Table IV.  Simulated 3-dB bandwidth (\u22127 V)',
    ['Device','WO','200 \u03a9','60 \u03a9','38 \u03a9'],
    [['f_3dB (GHz)',f"{bw['WO']:.1f}",f"{bw['200 ohm']:.1f}",f"{bw['60 ohm']:.1f}",f"{bw['38 ohm']:.1f}"]],
    [0.24,0.19,0.19,0.19,0.19])
fig.add_subplot(gs[6]).axis('off')
fig.savefig('case3_tables.png',dpi=200,bbox_inches='tight',facecolor='white')
print('saved case3_tables.png')
