"""
Re-run with paper-derived tau_A = 3.53 ps vs original 7.862 ps
==============================================================
S11 fitting is independent of tau_A (H_ph not in S11), so circuit
parameters are unchanged. Only frequency response prediction changes.

Compares:
  - Original: tau_A = 7.862 ps (code value, includes extra physics)
  - Paper:    tau_A = 3.53 ps  (D_e-based, graded absorber, paper mu_e)
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Transit-time layers ─────────────────────────────────────────────
W_A_undep = 480e-9; W_A_dep = 160e-9; W_cliff = 50e-9; W_C = 740e-9
W_tot = W_A_undep + W_A_dep + W_cliff + W_C

v_os_InP   = 4.0e5
v_os_InGaAs = 2.1e5
v_h_InGaAs = 4.5e4

tau_Ad = W_A_dep / v_os_InGaAs
tau_cl = W_cliff / v_os_InP
tau_C  = W_C / v_os_InP
tau_h  = W_A_dep / v_h_InGaAs

def H_ph_factory(tau_A):
    def H_ph(w):
        sinc = lambda x: np.sinc(x/np.pi)
        HA   = 1/(1 + 1j*w*tau_A)
        HAd  = sinc(w*tau_Ad/2)*np.exp(-1j*(w*tau_A + w*tau_Ad/2))
        Hcl  = sinc(w*tau_cl/2)*np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl/2))
        Hco  = sinc(w*tau_C/2) *np.exp(-1j*(w*tau_A + w*tau_Ad + w*tau_cl + w*tau_C/2))
        HAd_h = sinc(w*tau_h/2)*np.exp(-1j*(w*tau_A + w*tau_h/2))
        return (W_A_undep*HA + W_A_dep*HAd + W_A_dep*HAd_h
                + W_cliff*Hcl + W_C*Hco)/(W_tot + W_A_dep)
    return H_ph

# ── Circuit ─────────────────────────────────────────────────────────
C_CPW = 28.93e-15
R_L = 50.0

def _Y_Rp(w, Rp, Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp + 1j*w*Lrp)

def H_ckt(w, Rs, Cpd, Rp, Lcpw, Lrp=0.0, Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Y_A = 1j*w*C_CPW + _Y_Rp(w, Rp, Lrp) + 1/(1j*w*Lcpw + R_L)
    return (R_L/(1j*w*Lcpw + R_L)) / (1j*w*Cpd + Y_A*(1 + 1j*w*Cpd*Zs))

def get_bw(f, Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Fitted circuit parameters (from CLAUDE.md / original fit) ──────
# These are S11-fitted, INDEPENDENT of tau_A
configs = [
    dict(lbl='Rp=200Ω', Rp=200,    col='#888888', mk='D',
         Rs=8.92, Cj=132.7e-15, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=153e-12),
    dict(lbl='Rp=38Ω',  Rp=38,     col='#1B998B', mk='o',
         Rs=8.92, Cj=132.7e-15, Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60,     col='#FF8C00', mk='s',
         Rs=8.92, Cj=132.7e-15, Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf, col='#E91E8C', mk='^',
         Rs=8.92, Cj=132.7e-15, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=0.0),
]

# ── Reference loss correction (from main script) ───────────────────
ref_f_GHz = np.array([
    0.5,1.17,1.5,2.0,2.67,3.08,3.75,4.25,4.75,5.08,5.75,6.5,7.0,7.5,7.83,
    8.5,9.0,9.5,10.17,10.83,11.33,11.83,12.33,12.67,13.5,13.83,14.5,15.0,
    15.5,16.0,16.5,17.0,17.67,18.33,18.67,19.33,19.67,20.33,20.83,21.33,
    22.0,22.5,23.17,23.67,24.17,24.67,24.83,25.17,25.33,25.83,26.33,26.83,
    27.08,27.58,27.75,27.92,28.25,28.58,29.08,29.58,30.08,30.58,31.08,31.83,
    32.33,32.83,33.33,33.83,34.33,34.83,35.33,35.83,36.5,37.67,41.83,47.0])
ref_loss_dB = np.array([
    0.12,0.25,0.35,0.52,0.36,0.36,0.35,0.34,0.32,0.32,0.34,0.36,0.38,0.4,
    0.41,0.42,0.43,0.45,0.48,0.53,0.56,0.59,0.62,0.64,0.72,0.75,0.81,0.87,
    0.89,0.94,0.97,0.94,0.95,0.96,0.98,1.03,1.06,1.09,1.1,1.06,1.06,1.08,
    1.09,1.12,1.16,1.19,1.2,1.2,1.21,1.16,1.06,1.01,0.97,0.92,0.93,0.92,
    0.91,0.93,0.97,1.04,1.14,1.1,1.29,1.46,1.62,1.81,1.77,1.72,1.6,1.43,
    1.49,1.66,1.85,1.96,1.17,2.22])

# ── Measured frequency response data (from main script) ────────────
_freq_200 = np.array([[0.33,-17.54],[0.83,-17.68],[1.33,-18.06],[1.83,-18.32],[2.50,-18.63],[2.92,-18.43],[3.42,-18.38],[3.92,-18.46],[4.42,-18.32],[4.92,-18.71],[5.58,-18.55],[6.08,-18.92],[6.67,-18.96],[7.17,-19.03],[7.67,-18.82],[8.17,-18.86],[8.67,-19.10],[9.17,-19.40],[9.83,-19.33],[10.33,-19.58],[11.00,-19.49],[11.33,-19.53],[11.83,-19.20],[12.50,-19.72],[13.00,-20.17],[13.67,-20.80],[14.17,-20.25],[14.67,-20.38],[15.17,-20.32],[15.67,-20.74],[16.17,-20.40],[16.67,-21.30],[17.33,-21.52],[17.83,-21.37],[18.33,-20.94],[18.83,-20.87],[19.33,-21.09],[19.83,-22.01],[20.33,-21.55],[21.00,-22.36],[21.50,-22.06],[17.17,-21.43],[17.67,-21.38],[18.17,-20.95],[18.67,-20.99],[19.00,-21.02],[19.67,-21.42],[20.33,-21.96],[20.67,-22.80],[21.17,-21.83],[21.67,-22.09],[22.17,-21.64],[22.67,-22.10],[23.33,-22.38],[24.00,-22.96],[24.33,-23.19],[24.83,-23.40],[25.33,-23.30],[25.83,-22.97],[26.33,-22.88]])
_freq_38 = np.array([[0.50,-21.85],[1.00,-22.08],[1.67,-22.38],[2.17,-22.43],[2.67,-22.31],[3.25,-22.37],[3.75,-22.41],[4.08,-22.42],[4.75,-22.50],[5.25,-22.43],[5.75,-22.55],[6.08,-22.49],[6.83,-22.51],[7.33,-22.57],[7.67,-22.56],[8.17,-22.65],[8.67,-22.67],[9.17,-22.78],[9.50,-22.67],[10.17,-22.75],[10.67,-22.72],[11.17,-22.78],[11.50,-22.78],[12.17,-23.06],[12.67,-23.13],[13.17,-23.11],[13.67,-23.17],[14.17,-23.15],[14.67,-23.02],[15.17,-23.10],[15.67,-23.41],[16.33,-23.72],[16.83,-23.41],[17.17,-23.62],[17.83,-23.39],[18.33,-23.27],[18.83,-23.27],[19.50,-23.43],[20.00,-24.10],[20.50,-23.66],[21.00,-24.09],[21.33,-23.59],[22.00,-23.61],[22.33,-23.35],[23.00,-23.55],[23.50,-24.09],[24.00,-24.29],[24.50,-24.27],[25.00,-24.34],[25.50,-24.03],[26.00,-24.37],[26.50,-23.88],[27.25,-24.57],[27.58,-24.69],[28.08,-24.99],[28.75,-24.53],[29.25,-24.42],[29.75,-24.49],[30.25,-25.71],[31.08,-25.85],[31.67,-26.44],[32.17,-26.22],[32.67,-26.79],[33.17,-25.85],[33.83,-26.05],[34.67,-27.56],[35.00,-27.60],[35.67,-28.72],[36.17,-27.33],[36.67,-27.44]])
_freq_59 = np.array([[0.83,-21.43],[1.50,-21.57],[2.00,-21.75],[2.50,-21.89],[2.92,-21.82],[3.58,-21.77],[3.92,-21.83],[4.58,-21.82],[5.08,-21.97],[5.58,-21.94],[6.08,-21.98],[6.67,-22.01],[7.17,-22.06],[7.67,-22.00],[8.17,-22.01],[8.67,-22.11],[9.33,-22.29],[9.83,-22.16],[10.33,-22.32],[10.67,-22.24],[11.17,-22.22],[11.67,-22.20],[12.33,-22.50],[12.83,-22.67],[13.25,-22.74],[13.83,-22.67],[14.33,-22.85],[14.83,-22.59],[15.67,-22.72],[16.00,-22.69],[16.50,-23.50],[17.00,-23.11],[17.50,-23.15],[18.00,-22.92],[18.67,-22.90],[19.17,-22.93],[19.67,-23.24],[20.17,-23.54],[20.67,-24.01],[21.17,-23.36],[21.67,-23.44],[22.17,-23.23],[22.67,-23.33],[23.17,-23.36],[23.67,-23.98],[24.33,-24.21],[24.83,-24.29],[25.33,-24.25],[25.83,-23.77],[26.33,-24.15],[26.83,-24.73],[27.42,-24.54],[27.92,-24.99],[28.42,-24.92],[29.08,-24.53],[29.42,-24.41],[29.92,-24.33],[30.58,-26.08],[31.33,-27.00]])
_freq_WO = np.array([[0.33,-16.38],[0.83,-16.52],[1.50,-16.71],[2.00,-17.13],[2.50,-17.54],[2.92,-17.31],[3.58,-16.99],[4.08,-17.21],[4.42,-16.95],[4.92,-17.37],[5.42,-17.12],[6.08,-17.94],[6.42,-17.77],[7.00,-17.58],[7.50,-17.76],[8.00,-17.46],[8.33,-17.60],[8.83,-17.77],[9.33,-18.45],[9.83,-18.01],[10.33,-18.71],[10.83,-18.16],[11.17,-17.82],[11.67,-17.94],[12.00,-18.50],[12.67,-18.68],[13.33,-18.74],[13.67,-19.76],[14.17,-19.17],[14.67,-18.99],[15.33,-18.91],[15.67,-19.51],[16.50,-20.67],[17.00,-19.70],[17.50,-20.34],[18.00,-20.01],[18.50,-19.84],[19.00,-19.62],[19.67,-19.61],[20.17,-21.00],[20.67,-21.75],[21.17,-20.87],[21.67,-21.17],[22.17,-20.77],[22.67,-20.84],[23.17,-20.61],[23.67,-21.83],[24.33,-22.32],[24.67,-22.43],[25.17,-22.63],[25.67,-21.93],[26.00,-22.79],[26.50,-21.84],[27.08,-23.15],[27.75,-23.35],[28.25,-23.98],[28.58,-23.64],[29.08,-23.44],[29.58,-23.32],[30.08,-23.02]])

freq_map = {'Rp=200Ω':_freq_200, 'Rp=38Ω':_freq_38, 'Rp=60Ω':_freq_59, 'Open':_freq_WO}

def get_freq_response(arr):
    f_ghz = arr[:,0]
    cal = arr[:,1] + np.interp(f_ghz, ref_f_GHz, ref_loss_dB)
    return f_ghz*1e9, cal - cal[0]

# ── Two scenarios ───────────────────────────────────────────────────
TAU_A_CASES = [
    ('Original',  7.862e-12, '-',  2.5),
    ('Paper τ_A', 3.530e-12, '--', 2.0),
]

f_plot = np.linspace(0.1e9, 50e9, 5000)
w_plot = 2*np.pi*f_plot

print(f'\n{"="*82}')
print(f'{"Device":>10} | {"Case":>10} | {"τ_A (ps)":>9} | {"BW (GHz)":>9} | {"RMS_H (dB)":>11}')
print('='*82)

results_table = []
for cfg in configs:
    row = {'lbl': cfg['lbl']}
    fm, pm = get_freq_response(freq_map[cfg['lbl']])
    wm = 2*np.pi*fm
    Hckt_m = H_ckt(wm, cfg['Rs'], cfg['Cj'], cfg['Rp'], cfg['Lcpw'],
                   Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    Hckt_p = H_ckt(w_plot, cfg['Rs'], cfg['Cj'], cfg['Rp'], cfg['Lcpw'],
                   Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    for tag, tauA, _, _ in TAU_A_CASES:
        H_ph = H_ph_factory(tauA)
        Ht_m = H_ph(wm) * Hckt_m
        Hd_m = 20*np.log10(np.abs(Ht_m)/np.abs(Ht_m[0]))
        rms_H = np.sqrt(np.mean((Hd_m - pm)**2))
        Ht_p = H_ph(w_plot) * Hckt_p
        Hd_p = 20*np.log10(np.abs(Ht_p)/np.abs(Ht_p[0]))
        bw = get_bw(f_plot, Hd_p)
        bw_s = f'{bw:.1f}' if not np.isnan(bw) else '>50'
        print(f'{cfg["lbl"]:>10} | {tag:>10} | {tauA*1e12:>9.3f} | {bw_s:>9} | {rms_H:>11.3f}')
        row[tag] = dict(Hd_p=Hd_p, Hd_m_curve=Hd_m, fm=fm, pm=pm, bw=bw, rms_H=rms_H)
    results_table.append(row)
    print('-'*82)

# ── Plot comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    r'Frequency Response: Original $\tau_A$=7.862 ps  vs  Paper-derived $\tau_A$=3.53 ps' '\n'
    r'(Circuit params fixed from S11 fit — $H_{ph}$ only changes)',
    fontsize=11, fontweight='bold')

for ci, row in enumerate(results_table):
    ax = axes[ci//2][ci%2]
    cfg = configs[ci]
    col = cfg['col']; mk = cfg['mk']

    # measured
    fm = row['Original']['fm']; pm = row['Original']['pm']
    ax.scatter(fm/1e9, pm, color='k', marker=mk, s=22, edgecolors=col,
               linewidths=1.0, zorder=5, label='Measured')

    # both predictions
    for tag, tauA, ls, lw in TAU_A_CASES:
        Hd_p = row[tag]['Hd_p']
        bw = row[tag]['bw']
        rms = row[tag]['rms_H']
        bw_s = f'{bw:.1f}' if not np.isnan(bw) else '>50'
        clr = col if tag == 'Original' else 'navy'
        ax.plot(f_plot/1e9, Hd_p, color=clr, lw=lw, ls=ls,
                label=rf'{tag} (τ_A={tauA*1e12:.2f}ps)  BW={bw_s} GHz, RMS={rms:.2f}dB')

    ax.axhline(-3, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.set_title(cfg['lbl'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (GHz)', fontsize=10)
    ax.set_ylabel('Normalized Response (dB)', fontsize=10)
    ax.set_xlim(0, 50)
    ax.set_ylim(-12, 3)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = 'tauA_compare_3p53_vs_7p86.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
