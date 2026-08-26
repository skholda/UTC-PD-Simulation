"""
Experiment: single CPW inductance L_CPW (drop L_CPW2), free-fit per device
==========================================================================
Topology change: device branch Z_dev = Rs + 1/(jwCj)   (no L_CPW2)
  Y_n  = jwC_CPW + 1/(Rp+jwLrp) + 1/Z_dev
  Z_in = jw*L_CPW + 1/Y_n
Fit L_CPW per device to the measured S11 (full 40 GHz). Cj, Rs, C_CPW, L_Rp fixed.
Reports fitted L_CPW, S11 RMS, and freq-response BW; compares to the locked
Option-B (L_CPW + L_CPW2) baseline.
"""
import os, re, sys, numpy as np
from scipy.optimize import minimize_scalar

# ── H_ph (current baseline: 4-term, W_Ad=160, material-resolved) ───
W_A,W_Ad,W_C = 480e-9,160e-9,820e-9; W_norm=W_A+W_C+2*W_Ad
tau_A,tau_R=3.530e-12,0.0; tau_eD,tau_C=2.026e-12,7.794e-12; tau_h=W_Ad/4.8e4
def H_ph(w):
    s=lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            +W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            +W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            +W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm

# ── circuit (single L_CPW, no L_CPW2) ──────────────────────────────
C_CPW,R_L,Cj,Rs = 46.53e-15,50.0,131.0e-15,8.92
def _Y_Rp(w,Rp,Lrp): return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def sim_S11(w,Rp,Lcpw,Lrp):
    Z_dev = Rs + 1/(1j*w*Cj)
    Y_n   = 1j*w*C_CPW + _Y_Rp(w,Rp,Lrp) + 1/Z_dev
    Z_in  = 1j*w*Lcpw + 1/Y_n
    return (Z_in-50)/(Z_in+50)
def H_ckt(w,Rp,Lcpw,Lrp):
    Y_A = 1j*w*C_CPW + _Y_Rp(w,Rp,Lrp) + 1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cj + Y_A*(1+1j*w*Cj*Rs))
def get_bw(f,Hd):
    idx=np.where(Hd<=-3)[0]; return f[idx[0]]/1e9 if len(idx) else np.nan

# ── measured data (from main.py) ───────────────────────────────────
_main=open('main.py').read(); _ns={'np':np}
for v in ['ref_f_GHz','ref_loss_dB','_s1p_200','_s1p_33','_s1p_55','_s1p_WO',
          '_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
ref_f,ref_loss=_ns['ref_f_GHz'],_ns['ref_loss_dB']
def gs1p(a):
    fr=a[:,0]; S=10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2])); return fr,S
def gfr(a):
    fg=a[:,0]; cal=a[:,1]+np.interp(fg,ref_f,ref_loss); return fg*1e9,cal-cal[0]

DEV=[('Rp=200Ω',200.0,'_s1p_200','_freq_200',153.7e-12,197.9,0.0),
     ('Rp=38Ω', 38.0, '_s1p_33', '_freq_33', 65.6e-12, 141.6,56.3),
     ('Rp=60Ω', 60.0, '_s1p_55', '_freq_55', 71.8e-12, 149.9,48.0),
     ('Open',   np.inf,'_s1p_WO','_freq_WO', 0.0,       197.9,0.0)]

f_plot=np.linspace(0.1e9,50e9,5000); w_plot=2*np.pi*f_plot
print('='*94)
print('Single L_CPW (no L_CPW2) — free-fit per device  vs  locked Option-B (L_CPW+L_CPW2)')
print('='*94)
print(f'{"Device":>9} | {"L_CPW fit":>10} | {"RMS_S11":>9} | {"BW":>7} || '
      f'{"OptB L_tot":>10} | {"OptB RMS":>9} | {"OptB BW":>8}')
print('-'*94)
for lbl,Rp,s1pk,frk,Lrp,LcpwB,Lcpw2B in DEV:
    fs,S11m=gs1p(_ns[s1pk]); ws=2*np.pi*fs
    # fit single L_CPW
    def cost(Lp):
        return np.sqrt(np.mean(np.abs(sim_S11(ws,Rp,Lp*1e-12,Lrp)-S11m)**2))
    res=minimize_scalar(cost,bounds=(50,400),method='bounded')
    Lfit=res.x; rms=res.fun
    # freq response BW with fitted single L_CPW
    Hp=H_ph(w_plot)*H_ckt(w_plot,Rp,Lfit*1e-12,Lrp)
    Hd=20*np.log10(np.abs(Hp)/np.abs(H_ph(0)*H_ckt(1e-3,Rp,Lfit*1e-12,Lrp)))
    bw=get_bw(f_plot,Hd)
    # Option-B reference (locked): Z_dev has Lcpw2
    def s11B(w):
        Zs=Rs+1j*w*Lcpw2B*1e-12; Zdev=Zs+1/(1j*w*Cj)
        Yn=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/Zdev; return (1j*w*LcpwB*1e-12+1/Yn-50)/(1j*w*LcpwB*1e-12+1/Yn+50)
    rmsB=np.sqrt(np.mean(np.abs(s11B(ws)-S11m)**2))
    def HckB(w):
        Zs=Rs+1j*w*Lcpw2B*1e-12; YA=1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*LcpwB*1e-12+R_L)
        return (R_L/(1j*w*LcpwB*1e-12+R_L))/(1j*w*Cj+YA*(1+1j*w*Cj*Zs))
    HdB=20*np.log10(np.abs(H_ph(w_plot)*HckB(w_plot))/np.abs(H_ph(0)*HckB(1e-3)))
    bwB=get_bw(f_plot,HdB)
    bws=f'{bw:.1f}' if not np.isnan(bw) else '>50'; bwBs=f'{bwB:.1f}' if not np.isnan(bwB) else '>50'
    print(f'{lbl:>9} | {Lfit:8.1f}pH | {rms:9.5f} | {bws:>6}G || '
          f'{LcpwB+Lcpw2B:8.1f}pH | {rmsB:9.5f} | {bwBs:>7}G')
print('-'*94)
print('Fixed: Cj=131 fF, Rs=8.92, C_CPW=46.53 fF, L_Rp=FEM (per device).')
