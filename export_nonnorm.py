"""
Non-normalized (absolute) frequency-response export for Origin
==============================================================
Writes, per device and bias, the RAW measured response and the absolute model
transimpedance |H_ph*H_ckt| in dB-ohm (NO 0-dB-at-DC normalization).

  -7 V measured : arr[:,1] + cal-loss  [dB]   (from main.py _freq_* arrays)
  -5 V measured : Cal RF POW           [dBm]  (from data_5V_5mA/*.xlsx col 6)
  model         : 20*log10(|H_ph(w)*H_ckt(w)|)  [dB-ohm]   (shape matches meas
                  power up to an additive constant, since P_RF ~ |H_ph*H_ckt|^2)
"""
import os, re, sys, numpy as np, pandas as pd

# ── 4-term H_ph (current baseline) ─────────────────────────────────
W_A, W_Ad, W_C = 480e-9, 240e-9, 740e-9
W_norm = W_A + W_C + 2*W_Ad
tau_A, tau_R = 3.530e-12, 0.0
tau_eD, tau_C = 3.039e-12, 6.994e-12
tau_h = W_Ad/4.8e4
def H_ph(w):
    sinc = lambda x: np.sinc(x/np.pi)
    t1 = W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    t2 = W_C/(1+1j*w*tau_A)*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    t3 = W_Ad*sinc(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
    t4 = W_Ad*sinc(w*tau_h/2)*np.exp(-1j*w*tau_h/2)
    return (t1+t2+t3+t4)/W_norm

# ── circuit ────────────────────────────────────────────────────────
C_CPW, R_L, Rs = 46.53e-15, 50.0, 8.92
def _Y_Rp(w, Rp, Lrp):
    return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*Lrp)
def H_ckt(w, Cj, Rp, Lcpw, Lrp, Lcpw2):
    Zs = Rs+1j*w*Lcpw2
    Y_A = 1j*w*C_CPW+_Y_Rp(w,Rp,Lrp)+1/(1j*w*Lcpw+R_L)
    return (R_L/(1j*w*Lcpw+R_L))/(1j*w*Cj+Y_A*(1+1j*w*Cj*Zs))

# device L config (common to both biases; bias-independent)
DEV = {
 'Rp_200ohm': dict(Rp=200.0, Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=153.7e-12),
 'Rp_38ohm' : dict(Rp=38.0,  Lcpw=141.6e-12, Lcpw2=56.3e-12,Lrp=65.6e-12),
 'Rp_60ohm' : dict(Rp=60.0,  Lcpw=149.9e-12, Lcpw2=48.0e-12,Lrp=71.8e-12),
 'Open'     : dict(Rp=np.inf,Lcpw=197.9e-12, Lcpw2=0.0,     Lrp=0.0),
}
f_plot = np.linspace(0.1e9, 45e9, 4500); w_plot = 2*np.pi*f_plot

def model_dBm(cfg, Cj, Iph):
    # RF power delivered to the load:  P_RF = |Iph*H_ph*H_ckt|^2 / (2 R_L)
    H = H_ph(w_plot)*H_ckt(w_plot, Cj, cfg['Rp'], cfg['Lcpw'], cfg['Lrp'], cfg['Lcpw2'])
    P = np.abs(Iph*H)**2/(2*R_L)           # W
    return 10*np.log10(P/1e-3)             # dBm (absolute, no normalization)

def write(path, header, cols):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header)+'\n')
        for row in zip(*cols):
            f.write('\t'.join(f'{v:.8g}' for v in row)+'\n')

# ── -7 V measured (main.py) ────────────────────────────────────────
_main = open('main.py').read(); _ns={'np':np}
for var in ['ref_f_GHz','ref_loss_dB','_freq_200','_freq_33','_freq_55','_freq_WO']:
    m=re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)',_main,re.M|re.S)
    exec(m.group(0),_ns)
ref_f,ref_loss = _ns['ref_f_GHz'],_ns['ref_loss_dB']
map7={'Rp_200ohm':'_freq_200','Rp_38ohm':'_freq_33','Rp_60ohm':'_freq_55','Open':'_freq_WO'}

os.makedirs('origin_export/minus7V', exist_ok=True)
os.makedirs('origin_export/minus5V_5mA', exist_ok=True)

for tag,cfg in DEV.items():
    arr=_ns[map7[tag]]; fm=arr[:,0]; meas_abs=arr[:,1]+np.interp(fm,ref_f,ref_loss)  # dB
    write(f'origin_export/minus7V/freqresp_nonnorm_meas_{tag}.txt',
          ['Freq_GHz','Meas_dBm'], [fm, meas_abs])
    write(f'origin_export/minus7V/freqresp_nonnorm_sim_{tag}.txt',
          ['Freq_GHz','Model_dBm'], [f_plot/1e9, model_dBm(cfg, 131.0e-15, 1.0e-3)])  # Iph=1 mA

# ── -5 V measured (xlsx, Cal RF dBm) ───────────────────────────────
map5={'Rp_200ohm':'200ohm','Rp_38ohm':'38ohm','Rp_60ohm':'60ohm','Open':'WO'}
for tag,cfg in DEV.items():
    df=pd.read_excel(f'data_5V_5mA/{map5[tag]}.xlsx',header=14)
    fm=pd.to_numeric(df.iloc[:,0],errors='coerce')
    cal=pd.to_numeric(df.iloc[:,6],errors='coerce')       # Cal RF POW dBm
    m=fm.notna()&cal.notna(); fm,cal=fm[m].values,cal[m].values
    write(f'origin_export/minus5V_5mA/freqresp_nonnorm_meas_{tag}.txt',
          ['Freq_GHz','Meas_CalRF_dBm'], [fm, cal])
    write(f'origin_export/minus5V_5mA/freqresp_nonnorm_sim_{tag}.txt',
          ['Freq_GHz','Model_dBm'], [f_plot/1e9, model_dBm(cfg, 161.0e-15, 5.0e-3)])  # Iph=5 mA

print('Wrote non-normalized freqresp files (measured & model both in dBm):')
print('  origin_export/minus7V/freqresp_nonnorm_{meas,sim}_*.txt   (Iph=1 mA)')
print('  origin_export/minus5V_5mA/freqresp_nonnorm_{meas,sim}_*.txt (Iph=5 mA)')
