"""
Re-fit S11 with the CORRECT ladder topology (matching the device schematic):
  Z1  = R_S + 1/(jw C_PD)            photodiode branch
  Y1  = jw C_CPW + 1/Z1             node1: C_CPW shunt
  Z2  = jw L_CPW1 + 1/Y1            L_CPW1 series
  Y2  = 1/(R_m + jw L_m) + 1/Z2      node2: S-curve resistor shunt
  Zin = jw L_CPW2 + 1/Y2            L_CPW2 series -> port
Fixed: R_S=8.92, C_CPW=46.53 fF, L_m=FEM per device, R_m=Rp per device.
Fit:   C_PD (common, S11) + L_CPW1, L_CPW2 per device.
"""
import re, numpy as np
from scipy.optimize import minimize

Rs=8.92; C_CPW=46.53e-15
def S11(w,Cpd,Rp,Lcpw1,Lcpw2,Lm):
    Z1=Rs+1/(1j*w*Cpd)
    Y1=1j*w*C_CPW+1/Z1
    Z2=1j*w*Lcpw1+1/Y1
    Yrm=0.0 if np.isinf(Rp) else 1/(Rp+1j*w*Lm)
    Y2=Yrm+1/Z2
    Zin=1j*w*Lcpw2+1/Y2
    return (Zin-50)/(Zin+50)

_main=open('main.py').read(); _ns={'np':np}
for v in ['_s1p_200','_s1p_33','_s1p_55','_s1p_WO']:
    m=re.search(rf'^{v}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S); exec(m.group(0),_ns)
def gs1p(a): return 2*np.pi*a[:,0], 10**(a[:,1]/20)*np.exp(1j*np.deg2rad(a[:,2]))

# (label, Rp, s1pkey, L_m FEM)
DEV=[('200Ω',200.0,'_s1p_200',153.7e-12),
     ('38Ω', 38.0, '_s1p_33', 65.6e-12),
     ('60Ω', 60.0, '_s1p_55', 71.8e-12),
     ('Open',np.inf,'_s1p_WO',0.0)]
data={lbl:gs1p(_ns[k]) for lbl,_,k,_2 in [(d[0],d[1],d[2],d[3]) for d in DEV]}

def fit_dev(Cpd,Rp,ws,S11m,Lm):
    def cost(p): return np.sqrt(np.mean(np.abs(S11(ws,Cpd,Rp,p[0]*1e-12,p[1]*1e-12,Lm)-S11m)**2))
    best=None
    for g in [(150,50),(120,60),(180,20),(100,100),(200,80)]:
        r=minimize(cost,g,method='Nelder-Mead',
                   options=dict(xatol=1e-3,fatol=1e-7,maxiter=4000))
        if best is None or r.fun<best.fun: best=r
    return best.x, best.fun

def total_rms(Cpd):
    tot=0; out={}
    for lbl,Rp,k,Lm in DEV:
        ws,S11m=data[lbl]; p,rms=fit_dev(Cpd,Rp,ws,S11m,Lm)
        out[lbl]=(p[0],p[1],rms); tot+=rms
    return tot,out

# scan common Cpd
best=None
for Cf in np.arange(90,220,5):
    tot,out=total_rms(Cf*1e-15)
    if best is None or tot<best[0]: best=(tot,Cf,out)
# refine around best
for Cf in np.arange(best[1]-4,best[1]+4,1):
    tot,out=total_rms(Cf*1e-15)
    if tot<best[0]: best=(tot,Cf,out)

tot,Cf,out=best
print('='*78)
print(f'CORRECT ladder re-fit   |   common C_PD = {Cf:.0f} fF   (was 131 fF, old topology)')
print('='*78)
print(f'{"Device":>7} | {"L_CPW1":>8} | {"L_CPW2":>8} | {"L_m(FEM)":>8} | {"RMS_S11":>9}')
print('-'*78)
for lbl,Rp,k,Lm in DEV:
    l1,l2,rms=out[lbl]
    print(f'{lbl:>7} | {l1:6.1f}pH | {l2:6.1f}pH | {Lm*1e12:6.1f}pH | {rms:9.5f}')
print('-'*78)
print(f'total RMS = {tot:.5f}   (old wrong-topology per-device RMS ~0.06-0.076)')
