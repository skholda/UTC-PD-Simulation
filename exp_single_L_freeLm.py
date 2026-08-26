"""
Single-L_CPW schematic, now with L_m ALSO free (per device):
  Iph ∥ C_PD ─[R_S]─ node1{ C_CPW ∥ (R_m+L_m) } ─[L_CPW]─ port
Fit: common C_PD + per-device (L_CPW, L_m).  Fixed: R_S, C_CPW.
Compare vs (a) single-L with FEM L_m, (b) 2-L ladder baseline.
"""
import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

Rs=8.92; C_CPW=46.53e-15
def S11_1L(w,Cpd,Rp,Lcpw,Lm):
    Z1=Rs+1/(1j*w*Cpd)
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y1=1j*w*C_CPW+Yrm+1/Z1
    Zin=1j*w*Lcpw+1/Y1
    return (Zin-50)/(Zin+50)

_main=open('main.py').read(); _ns={'np':np}
for v in ['_s1p_200','_s1p_33','_s1p_55','_s1p_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
def gs1p(a): return a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))

DEV=[('200Ω',200.0,'_s1p_200',153.7e-12,'#888888'),
     ('38Ω',38.0,'_s1p_33',65.6e-12,'#1B998B'),
     ('60Ω',60.0,'_s1p_55',71.8e-12,'#FF8C00'),
     ('Open',np.inf,'_s1p_WO',0.0,'#E91E8C')]
data={lbl:gs1p(_ns[k]) for lbl,_,k,_1,_2 in DEV}

def fit_dev(Cpd,Rp,ws,S11m,Lm_fem):
    if np.isinf(Rp):   # no shunt: only L_CPW matters
        cost=lambda Lp: np.sqrt(np.mean(np.abs(S11_1L(ws,Cpd,Rp,Lp*1e-12,0.0)-S11m)**2))
        r=minimize_scalar(cost,bounds=(20,400),method='bounded')
        return (r.x, np.nan), r.fun
    def cost(p):
        return np.sqrt(np.mean(np.abs(S11_1L(ws,Cpd,Rp,p[0]*1e-12,p[1]*1e-12)-S11m)**2))
    best=None
    for g in [(160,Lm_fem*1e12),(160,20),(160,120),(200,60),(120,200),(180,300)]:
        r=minimize(cost,g,method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-7,maxiter=4000))
        if best is None or r.fun<best.fun: best=r
    return (best.x[0],best.x[1]), best.fun

best=None
for Cf in np.arange(100,220,2):
    tot=0; out={}
    for lbl,Rp,k,Lm_fem,_c in DEV:
        fs,S11m=data[lbl]; ws=2*np.pi*fs
        p,rms=fit_dev(Cf*1e-15,Rp,ws,S11m,Lm_fem); out[lbl]=(p,rms); tot+=rms
    if best is None or tot<best[0]: best=(tot,Cf,out)
for Cf in np.arange(best[1]-2,best[1]+2,1):
    tot=0; out={}
    for lbl,Rp,k,Lm_fem,_c in DEV:
        fs,S11m=data[lbl]; ws=2*np.pi*fs
        p,rms=fit_dev(Cf*1e-15,Rp,ws,S11m,Lm_fem); out[lbl]=(p,rms); tot+=rms
    if tot<best[0]: best=(tot,Cf,out)
tot,Cf,out=best

print('='*94)
print(f'Single-L_CPW + FREE L_m fit  |  common C_PD = {Cf:.0f} fF')
print('='*94)
print(f'{"Device":>7} | {"L_CPW":>8} | {"L_m fit":>8} | {"L_m FEM":>8} | {"RMS":>8} || {"1L FEM-Lm":>9} | {"2L ladder":>9}')
print('-'*94)
REF1={'200Ω':0.05568,'38Ω':0.08207,'60Ω':0.07594,'Open':0.09283}
REF2={'200Ω':0.06331,'38Ω':0.05648,'60Ω':0.05978,'Open':0.07393}
for lbl,Rp,k,Lm_fem,_c in DEV:
    (L,Lm),rms=out[lbl]
    lms='   —  ' if np.isnan(Lm) else f'{Lm:6.1f}pH'
    print(f'{lbl:>7} | {L:6.1f}pH | {lms:>8} | {Lm_fem*1e12:6.1f}pH | {rms:8.5f} || {REF1[lbl]:9.5f} | {REF2[lbl]:9.5f}')
print('-'*94)
print(f'total = {tot:.5f}   (1L FEM-Lm: 0.30653, 2-L ladder: 0.25350)')
