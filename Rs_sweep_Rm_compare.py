import os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---- locked baseline params ----
C_PD   = 131.0e-15
C_CPW  = 46.53e-15
L_CPW  = 197.9e-12     # total CPW inductance
L_CPW2 = 0.0
L_Rp   = 0.0
R_L    = 50.0
I_ph   = 1.0e-3        # 1 mA (from DC-level match: Open -16 dBm)

# paper H_ph
W_A=480e-9; W_C=980e-9; W=W_A+W_C
tau_A=3.530e-12; tau_C=2.880e-12; tau_R=0.070e-12
def H_ph(w):
    sinc=lambda x: np.sinc(x/np.pi)
    abs_t=W_A*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
    col_t=W_C*sinc(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
    return (abs_t+col_t)/(W*(1+1j*w*tau_A))

def _Y_Rp(w,Rp):
    return 0.0 if np.isinf(Rp) else 1.0/(Rp+1j*w*L_Rp)
def H_ckt(w,Rs,Rp):
    Zs=Rs+1j*w*L_CPW2
    Y_A=1j*w*C_CPW+_Y_Rp(w,Rp)+1/(1j*w*L_CPW+R_L)
    return (R_L/(1j*w*L_CPW+R_L))/(1j*w*C_PD+Y_A*(1+1j*w*C_PD*Zs))
def P_dBm(w,Rs,Rp):
    Ht=H_ph(w)*H_ckt(w,Rs,Rp)
    P=np.abs(I_ph*Ht)**2/(2*R_L)
    return 10*np.log10(P/1e-3)

f=np.linspace(0.1e9,40e9,2000); w=2*np.pi*f
Rs_list=[5,15,25,35,45]
n=len(Rs_list)
blues=cm.Blues(np.linspace(0.35,0.95,n))
reds =cm.Reds (np.linspace(0.35,0.95,n))

fig,ax=plt.subplots(figsize=(9,7))
# interleave order to match the original legend layout
out_dir='origin_data_RsRm_sweep'; os.makedirs(out_dir,exist_ok=True)
hdr=['Freq_GHz']; cols=[f/1e9]
for i,Rs in enumerate(Rs_list):
    P60 =P_dBm(w,Rs,60.0)
    Pwo =P_dBm(w,Rs,np.inf)
    ax.plot(f/1e9,P60,'-', color=blues[i],lw=1.8,label=f'$R_m$=60, $R_s$={Rs}')
    ax.plot(f/1e9,Pwo,'--',color=reds[i], lw=1.8,label=f'Open, $R_s$={Rs}')
    hdr += [f'Rm60_Rs{Rs}_dBm', f'Open_Rs{Rs}_dBm']; cols += [P60,Pwo]

ax.set_xlabel('Frequency [GHz]',fontsize=13)
ax.set_ylabel('Output Power [dBm]',fontsize=13)
ax.set_xlim(0,40); ax.set_ylim(-30,-11.5)
ax.grid(True,alpha=0.3)
ax.legend(fontsize=9,ncol=2,loc='lower left')
fig.tight_layout()
fig.savefig('Rs_sweep_Rm_compare.png',dpi=150,bbox_inches='tight')
print('Saved: Rs_sweep_Rm_compare.png')

# export data
data=np.column_stack(cols)
with open(os.path.join(out_dir,'Pout_RsRm_sweep.txt'),'w') as fp:
    fp.write('\t'.join(hdr)+'\n')
    for row in data:
        fp.write('\t'.join(f'{v:.6g}' for v in row)+'\n')
print('Saved: origin_data_RsRm_sweep/Pout_RsRm_sweep.txt')
print(f'\nDC levels check:  Open(Rs=5)={P_dBm(np.array([2*np.pi*0.1e9]),5,np.inf)[0]:.2f} dBm, '
      f'Rm60(Rs=5)={P_dBm(np.array([2*np.pi*0.1e9]),5,60)[0]:.2f} dBm')
