"""
Refit Cj only (others fixed) with tau_A = 3.53 ps
=================================================
Keep Rs, L_CPW, L_CPW2, L_Rp at original S11-fit values.
Vary only Cj per device to refit S11.
Then evaluate frequency-response prediction with tau_A = 3.53 ps and
compare against tau_A = 7.862 ps + original Cj.
"""
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Load data arrays from main.py without running its full pipeline ─
_main = open(os.path.join(os.path.dirname(__file__), 'main.py')).read()
_ns = {'np': np}
for var in ['ref_f_GHz', 'ref_loss_dB',
            '_s1p_200', '_s1p_33', '_s1p_55', '_s1p_WO',
            '_freq_200', '_freq_33', '_freq_55', '_freq_WO']:
    m = re.search(rf'^{var}\s*=\s*np\.array\(\[.*?\n\]\)', _main, re.MULTILINE | re.DOTALL)
    if not m:
        sys.exit(f'Could not find {var} in main.py')
    exec(m.group(0), _ns)

ref_f_GHz   = _ns['ref_f_GHz'];   ref_loss_dB = _ns['ref_loss_dB']
S1P = {'Rp=200Ω': _ns['_s1p_200'], 'Rp=38Ω': _ns['_s1p_33'],
       'Rp=60Ω':  _ns['_s1p_55'],  'Open':   _ns['_s1p_WO']}
FREQ = {'Rp=200Ω': _ns['_freq_200'], 'Rp=38Ω': _ns['_freq_33'],
        'Rp=60Ω':  _ns['_freq_55'],  'Open':   _ns['_freq_WO']}

# ── Transit-time model ──────────────────────────────────────────────
W_A_undep = 480e-9; W_A_dep = 160e-9; W_cliff = 50e-9; W_C = 740e-9
W_tot = W_A_undep + W_A_dep + W_cliff + W_C
v_os_InP = 4.0e5; v_os_InGaAs = 2.1e5; v_h_InGaAs = 4.5e4
tau_Ad = W_A_dep/v_os_InGaAs; tau_cl = W_cliff/v_os_InP
tau_C  = W_C/v_os_InP;        tau_h  = W_A_dep/v_h_InGaAs

def H_ph_factory(tau_A):
    def H_ph(w):
        sinc = lambda x: np.sinc(x/np.pi)
        HA   = 1/(1+1j*w*tau_A)
        HAd  = sinc(w*tau_Ad/2)*np.exp(-1j*(w*tau_A+w*tau_Ad/2))
        Hcl  = sinc(w*tau_cl/2)*np.exp(-1j*(w*tau_A+w*tau_Ad+w*tau_cl/2))
        Hco  = sinc(w*tau_C/2) *np.exp(-1j*(w*tau_A+w*tau_Ad+w*tau_cl+w*tau_C/2))
        HAd_h = sinc(w*tau_h/2)*np.exp(-1j*(w*tau_A+w*tau_h/2))
        return (W_A_undep*HA + W_A_dep*HAd + W_A_dep*HAd_h
                + W_cliff*Hcl + W_C*Hco)/(W_tot + W_A_dep)
    return H_ph

# ── Circuit ─────────────────────────────────────────────────────────
# C_CPW: pad-fitted value (overrides ADS default 28.93 fF in original code)
C_CPW = 46.53e-15; R_L = 50.0
def _Y_Rp(w, Rp, Lrp):
    if np.isinf(Rp): return 0.0
    return 1.0/(Rp + 1j*w*Lrp)
def H_ckt(w, Rs, Cpd, Rp, Lcpw, Lrp=0.0, Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Y_A = 1j*w*C_CPW + _Y_Rp(w, Rp, Lrp) + 1/(1j*w*Lcpw + R_L)
    return (R_L/(1j*w*Lcpw + R_L)) / (1j*w*Cpd + Y_A*(1 + 1j*w*Cpd*Zs))
def sim_S11(w, Rs, Cpd, Rp, Lcpw, Lrp=0.0, Lcpw2=0.0):
    Zs = Rs + 1j*w*Lcpw2
    Z_dev = Zs + 1/(1j*w*Cpd)
    Y_nodeA = 1j*w*C_CPW + _Y_Rp(w, Rp, Lrp) + 1/Z_dev
    Z_in = 1j*w*Lcpw + 1/Y_nodeA
    return (Z_in - 50)/(Z_in + 50)
def get_bw(f, Hd):
    idx = np.where(Hd <= -3)[0]
    return f[idx[0]]/1e9 if len(idx) else np.nan

# ── Reference / configs (Rs, L_CPW, L_CPW2, L_Rp fixed) ─────────────
configs = [
    dict(lbl='Rp=200Ω', Rp=200,    col='#888888', mk='D',
         Rs=8.92, Cj_orig=132.7e-15, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=153e-12),
    dict(lbl='Rp=38Ω',  Rp=38,     col='#1B998B', mk='o',
         Rs=8.92, Cj_orig=132.7e-15, Lcpw=135.9e-12, Lcpw2=43.0e-12, Lrp=65.6e-12),
    dict(lbl='Rp=60Ω',  Rp=60,     col='#FF8C00', mk='s',
         Rs=8.92, Cj_orig=132.7e-15, Lcpw=133.2e-12, Lcpw2=45.7e-12, Lrp=71.8e-12),
    dict(lbl='Open',    Rp=np.inf, col='#E91E8C', mk='^',
         Rs=8.92, Cj_orig=132.7e-15, Lcpw=178.9e-12, Lcpw2=0.0,    Lrp=0.0),
]

def get_freq_response(arr):
    f_ghz = arr[:,0]
    cal = arr[:,1] + np.interp(f_ghz, ref_f_GHz, ref_loss_dB)
    return f_ghz*1e9, cal - cal[0]
def get_s1p(arr, f_max):
    fr = arr[:,0]
    S11 = 10**(arr[:,1]/20) * np.exp(1j*np.deg2rad(arr[:,2]))
    mask = fr <= f_max
    return fr[mask], S11[mask]

TAU_A_NEW = 3.530e-12
TAU_A_OLD = 7.862e-12

f_plot = np.linspace(0.1e9, 50e9, 5000); w_plot = 2*np.pi*f_plot

# ── 1-parameter S11 refit: only Cj ──────────────────────────────────
print(f'\n{"="*108}')
print(f'{"Device":>10} | {"Rp":>5} | {"Cj_orig":>9} | {"Cj_fit":>9} | '
      f'{"RMS_S11 orig":>13} | {"RMS_S11 new":>12} | '
      f'{"BW(orig τ_A)":>12} | {"BW(new τ_A)":>12} | {"RMS_H new":>10}')
print('='*108)

results = []
for cfg in configs:
    fm_all, pm = get_freq_response(FREQ[cfg['lbl']])
    f_max = fm_all.max()
    fs11, S11m = get_s1p(S1P[cfg['lbl']], f_max)
    ws11 = 2*np.pi*fs11
    wm   = 2*np.pi*fm_all

    # Cost: |Sim - Meas|^2 (S11 only)
    def cost(Cj):
        S11s = sim_S11(ws11, cfg['Rs'], Cj, cfg['Rp'], cfg['Lcpw'],
                       Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
        return np.mean(np.abs(S11s - S11m)**2)

    res = minimize_scalar(cost, bounds=(50e-15, 300e-15), method='bounded',
                          options={'xatol': 1e-18})
    Cj_new = res.x

    # RMS S11 for both
    S11_orig = sim_S11(ws11, cfg['Rs'], cfg['Cj_orig'], cfg['Rp'], cfg['Lcpw'],
                       Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    S11_new  = sim_S11(ws11, cfg['Rs'], Cj_new, cfg['Rp'], cfg['Lcpw'],
                       Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    rms_orig = np.sqrt(np.mean(np.abs(S11_orig - S11m)**2))
    rms_new  = np.sqrt(np.mean(np.abs(S11_new  - S11m)**2))

    # Freq response: original Cj + tau_A_old   vs   new Cj + tau_A_new
    H_old = H_ph_factory(TAU_A_OLD); H_new = H_ph_factory(TAU_A_NEW)
    Hckt_orig = H_ckt(w_plot, cfg['Rs'], cfg['Cj_orig'], cfg['Rp'], cfg['Lcpw'],
                      Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    Hckt_new  = H_ckt(w_plot, cfg['Rs'], Cj_new,         cfg['Rp'], cfg['Lcpw'],
                      Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    Hd_orig = 20*np.log10(np.abs(H_old(w_plot)*Hckt_orig) / np.abs(H_old(0)*Hckt_orig[0]))
    Hd_new  = 20*np.log10(np.abs(H_new(w_plot)*Hckt_new ) / np.abs(H_new(0) *Hckt_new[0] ))
    bw_orig = get_bw(f_plot, Hd_orig)
    bw_new  = get_bw(f_plot, Hd_new)

    # RMS_H new at measurement points
    Hckt_m_new = H_ckt(wm, cfg['Rs'], Cj_new, cfg['Rp'], cfg['Lcpw'],
                       Lrp=cfg['Lrp'], Lcpw2=cfg['Lcpw2'])
    Ht_m_new = H_new(wm) * Hckt_m_new
    Hd_m_new = 20*np.log10(np.abs(Ht_m_new)/np.abs(Ht_m_new[0]))
    rms_H_new = np.sqrt(np.mean((Hd_m_new - pm)**2))

    bw_o_s = f'{bw_orig:.1f}' if not np.isnan(bw_orig) else '>50'
    bw_n_s = f'{bw_new:.1f}'  if not np.isnan(bw_new)  else '>50'
    print(f'{cfg["lbl"]:>10} | {cfg["Rp"]!s:>5} | {cfg["Cj_orig"]*1e15:>8.1f}fF | '
          f'{Cj_new*1e15:>8.1f}fF | {rms_orig:>13.5f} | {rms_new:>12.5f} | '
          f'{bw_o_s:>12} | {bw_n_s:>12} | {rms_H_new:>10.3f}')

    results.append(dict(cfg=cfg, Cj_new=Cj_new, S11m=S11m, fs11=fs11,
                        S11_orig=S11_orig, S11_new=S11_new,
                        fm=fm_all, pm=pm, Hd_orig=Hd_orig, Hd_new=Hd_new,
                        bw_orig=bw_orig, bw_new=bw_new, rms_H_new=rms_H_new))

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle(
    rf'Refit $C_j$ only (others fixed) — $\tau_A$=3.53 ps' '\n'
    r'Row 1: S11 (Meas vs Original Cj fit vs New Cj fit)   |   '
    r'Row 2: Frequency response (Original $\tau_A$+Cj vs New $\tau_A$+Cj)',
    fontsize=11, fontweight='bold')

for ci, r in enumerate(results):
    cfg = r['cfg']; col = cfg['col']; mk = cfg['mk']

    # Row 0: S11 magnitude
    ax = axes[0, ci]
    ax.plot(r['fs11']/1e9, 20*np.log10(np.abs(r['S11m'])),
            '-', color=col, lw=1.0, label='Meas')
    ax.plot(r['fs11']/1e9, 20*np.log10(np.abs(r['S11_orig'])),
            ':', color='gray', lw=1.5,
            label=f'Cj orig ({cfg["Cj_orig"]*1e15:.1f}fF)')
    ax.plot(r['fs11']/1e9, 20*np.log10(np.abs(r['S11_new'])),
            '--', color='navy', lw=1.5,
            label=f'Cj new ({r["Cj_new"]*1e15:.1f}fF)')
    ax.set_xlabel('Frequency (GHz)', fontsize=9)
    ax.set_ylabel('|S11| (dB)', fontsize=9)
    ax.set_title(cfg['lbl'], fontsize=10, fontweight='bold')
    ax.set_xlim(0, r['fs11'].max()/1e9)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Row 1: frequency response
    ax = axes[1, ci]
    ax.scatter(r['fm']/1e9, r['pm'], color='k', marker=mk, s=20,
               edgecolors=col, linewidths=1.0, zorder=5, label='Meas')
    bw_o_s = f'{r["bw_orig"]:.1f}' if not np.isnan(r['bw_orig']) else '>50'
    bw_n_s = f'{r["bw_new"]:.1f}'  if not np.isnan(r['bw_new'])  else '>50'
    ax.plot(f_plot/1e9, r['Hd_orig'], '-', color=col, lw=2.0,
            label=rf'$\tau_A$=7.86ps, Cj orig  BW={bw_o_s}GHz')
    ax.plot(f_plot/1e9, r['Hd_new'], '--', color='navy', lw=1.8,
            label=rf'$\tau_A$=3.53ps, Cj new  BW={bw_n_s}GHz, RMS={r["rms_H_new"]:.2f}dB')
    ax.axhline(-3, color='gray', ls=':', lw=0.7, alpha=0.7)
    ax.set_xlabel('Frequency (GHz)', fontsize=9)
    ax.set_ylabel('Normalized H (dB)', fontsize=9)
    ax.set_xlim(0, 50); ax.set_ylim(-12, 3)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = 'tauA_refit_Cj.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
