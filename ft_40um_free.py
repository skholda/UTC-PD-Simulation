"""40 um: free R_s and C_CPW in the S11 fit (user-authorised).

Two variants:
  per-device : R_s, C_CPW, L_CPW1, L_m, L_CPW2 all free for each device/bias
  joint      : one R_s and one C_CPW shared by every 40 um device/bias
               (same mesa, same pad), per-device L's

For each, f_RC is recomputed from the refitted ladder and the f_T extraction
(all diameters, untruncated) is redone with the new 40 um points. 25/30 um
keep the locked baseline values.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)      # PAIR, C_PD_CV, read_s11, S11D, R_L, f3dB_of, Rs_FIX, C_CPW_FIX

# ── ladder with R_s and C_CPW as arguments ─────────────────────────────────
def Zin_g(w, Rs, Cc, Cpd, Rm, L1, Lm, L2):
    Z1 = Rs + 1/(1j*w*Cpd)
    Y1 = 1j*w*Cc + 1/Z1
    Z2 = 1j*w*L1 + 1/Y1
    Y2 = (1/(Rm + 1j*w*Lm) if np.isfinite(Rm) else 0.0) + 1/Z2
    return 1j*w*L2 + 1/Y2

def S11_g(w, *a):
    Z = Zin_g(w, *a); return (Z - R_L)/(Z + R_L)

def H_g(w, Rs, Cc, Cpd, Rm, L1, Lm, L2):
    A = np.ones_like(w, dtype=complex); B = np.zeros_like(w, dtype=complex)
    C = np.zeros_like(w, dtype=complex); D = np.ones_like(w, dtype=complex)
    def ser(Z):
        nonlocal B, D; B, D = A*Z + B, C*Z + D
    def sh(Y):
        nonlocal A, C; A, C = A + B*Y, C + D*Y
    sh(1j*w*Cpd); ser(Rs + 0j*w); sh(1j*w*Cc); ser(1j*w*L1)
    if np.isfinite(Rm): sh(1/(Rm + 1j*w*Lm))
    ser(1j*w*L2)
    return R_L/(C*R_L + D)

fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg

# ── load the 40 um S11 set ─────────────────────────────────────────────────
dev = []
for D, V, lab, camp, s11, sheet, note in PAIR:
    if D != 40: continue
    f, Sm = read_s11(os.path.join(S11D, s11))
    Rm_meas = (R_L*(1 + Sm[0])/(1 - Sm[0])).real
    op = lab == 'WO'
    dev.append(dict(V=V, lab=lab, f=f, w=2*np.pi*f, Sm=Sm, open=op,
                    Rm=np.inf if op else Rm_meas, Rm_meas=Rm_meas,
                    Cpd=C_PD_CV[(40, V)]*1e-15))

base = pd.read_csv('ft_userbw.csv')

def rms_of(d, Rs, Cc, L1, Lm, L2):
    return np.sqrt(np.mean(np.abs(S11_g(d['w'], Rs, Cc, d['Cpd'], d['Rm'],
                                        L1, Lm, L2) - d['Sm'])**2))

# ── variant A: per-device, 5 free ──────────────────────────────────────────
rowsA = []
for d in dev:
    keys = ['Rs', 'Cc', 'L1', 'L2'] + ([] if d['open'] else ['Lm'])
    p0 = dict(Rs=8.92, Cc=46.5, L1=80.0, L2=120.0, Lm=50.0)
    lo = dict(Rs=0.0, Cc=5.0, L1=0.0, L2=0.0, Lm=0.0)
    hi = dict(Rs=60.0, Cc=300.0, L1=600.0, L2=600.0, Lm=400.0)
    sc = dict(Rs=5.0, Cc=20.0, L1=50.0, L2=50.0, Lm=50.0)
    def unpack(p):
        v = dict(zip(keys, p)); v.setdefault('Lm', 0.0)
        return v['Rs'], v['Cc']*1e-15, v['L1']*1e-12, v['Lm']*1e-12, v['L2']*1e-12
    def resid(p):
        Rs, Cc, L1, Lm, L2 = unpack(p)
        e = S11_g(d['w'], Rs, Cc, d['Cpd'], d['Rm'], L1, Lm, L2) - d['Sm']
        return np.concatenate([e.real, e.imag])
    r = least_squares(resid, [p0[k] for k in keys],
                      bounds=([lo[k] for k in keys], [hi[k] for k in keys]),
                      x_scale=[sc[k] for k in keys])
    Rs, Cc, L1, Lm, L2 = unpack(r.x)
    rms = rms_of(d, Rs, Cc, L1, Lm, L2)
    rms0 = rms_of(d, Rs_FIX, C_CPW_FIX,
                  *[base[(base.D == 40) & (base.V == d['V']) & (base.lab == d['lab'])]
                    .iloc[0][k]*1e-12 for k in ('L1', 'Lm', 'L2')])
    fRC = f3dB_of(fg, H_g(wg, Rs, Cc, d['Cpd'], d['Rm'], L1, Lm, L2))/1e9
    rowsA.append(dict(V=d['V'], lab=d['lab'], Rm=d['Rm_meas'], Rs=Rs, Cc=Cc*1e15,
                      L1=L1*1e12, Lm=Lm*1e12, L2=L2*1e12, rms=rms, rms_locked=rms0,
                      f_RC=fRC))
A = pd.DataFrame(rowsA)
pd.set_option('display.width', 220)
print('=== A. per-device free R_s, C_CPW (40 um) ===')
print(A.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ── variant B: joint, shared R_s and C_CPW ─────────────────────────────────
nd = len(dev)
def unpackB(p):
    Rs, Cc = p[0], p[1]*1e-15
    Ls = []; k = 2
    for d in dev:
        if d['open']:
            Ls.append((p[k]*1e-12, 0.0, p[k+1]*1e-12)); k += 2
        else:
            Ls.append((p[k]*1e-12, p[k+1]*1e-12, p[k+2]*1e-12)); k += 3
    return Rs, Cc, Ls
def residB(p):
    Rs, Cc, Ls = unpackB(p); out = []
    for d, (L1, Lm, L2) in zip(dev, Ls):
        e = S11_g(d['w'], Rs, Cc, d['Cpd'], d['Rm'], L1, Lm, L2) - d['Sm']
        out += [e.real, e.imag]
    return np.concatenate(out)
p0, lo, hi, sc = [8.92, 46.5], [0.0, 5.0], [60.0, 300.0], [5.0, 20.0]
for d in dev:
    if d['open']:
        p0 += [80, 120]; lo += [0, 0]; hi += [600, 600]; sc += [50, 50]
    else:
        p0 += [80, 50, 120]; lo += [0, 0, 0]; hi += [600, 400, 600]; sc += [50, 50, 50]
r = least_squares(residB, p0, bounds=(lo, hi), x_scale=sc)
RsB, CcB, LsB = unpackB(r.x)
rowsB = []
for d, (L1, Lm, L2) in zip(dev, LsB):
    fRC = f3dB_of(fg, H_g(wg, RsB, CcB, d['Cpd'], d['Rm'], L1, Lm, L2))/1e9
    rowsB.append(dict(V=d['V'], lab=d['lab'], Rm=d['Rm_meas'], L1=L1*1e12,
                      Lm=Lm*1e12, L2=L2*1e12, rms=rms_of(d, RsB, CcB, L1, Lm, L2),
                      f_RC=fRC))
B = pd.DataFrame(rowsB)
print(f'\n=== B. joint fit, shared R_s = {RsB:.2f} ohm, C_CPW = {CcB*1e15:.1f} fF '
      f'(locked baseline: {Rs_FIX} ohm, {C_CPW_FIX*1e15:.2f} fF) ===')
print(B.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ── redo the f_T extraction with the new 40 um f_RC ────────────────────────
def freefit(s):
    x, y = s.x.values, s.y.values
    M = np.vstack([x, np.ones_like(x)]).T
    (sl, b), *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ [sl, b]
    r2 = 1 - (res**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    cov = (res**2).sum()/(len(x) - 2)*np.linalg.inv(M.T @ M); se = np.sqrt(cov[1, 1])
    f = lambda v: np.sqrt(1000/v) if v > 0 else np.nan
    return dict(N=len(x), sl=sl, b=b, se=se, adj=adj, fT=f(b), lo=f(b + se), hi=f(b - se))

def swap(tab):
    t = base.copy()
    for _, r in tab.iterrows():
        m = (t.D == 40) & (t.V == r.V) & (t.lab == r.lab)
        t.loc[m, 'f_RC'] = r.f_RC
    t['x'] = 1000/t.f_RC**2
    return t

variants = {'locked (baseline)': base, 'per-device free': swap(A), 'joint free': swap(B)}
print(f"\n{'variant':<20s}{'subset':<22s}{'N':>3s}{'slope':>8s}{'intercept':>16s}"
      f"{'Adj.R2':>8s}{'f_T':>20s}")
res = {}
for vn, t in variants.items():
    for sn, s in (('all D, untruncated', t[t.ok]),
                  ('40 um, untruncated', t[t.ok & (t.D == 40)])):
        o = freefit(s); res[(vn, sn)] = (o, s)
        hi = f"{o['hi']:.1f}" if np.isfinite(o['hi']) else 'inf'
        print(f"{vn:<20s}{sn:<22s}{o['N']:3d}{o['sl']:8.3f}"
              f"{o['b']:+8.3f} ± {o['se']:.3f}{o['adj']:8.3f}"
              f"{o['fT']:9.1f} ({o['lo']:.1f}–{hi})")

# ── figure ─────────────────────────────────────────────────────────────────
MK = {25: 'o', 30: 's', 40: '^'}; CV = {-3: '#7d3c98', -5: '#2471a3', -7: '#c0392b'}
fig, axs = plt.subplots(1, 3, figsize=(16.2, 5.3))
for ax, vn in zip(axs, variants):
    o, s = res[(vn, 'all D, untruncated')]
    for V in (-7, -5, -3):
        for D in (25, 30, 40):
            q = s[(s.V == V) & (s.D == D)]
            if len(q):
                ax.scatter(q.x, q.y, s=74, marker=MK[D], facecolor='none',
                           edgecolor=CV[V], lw=1.7, zorder=5,
                           label=f'{D} $\\mu$m, {V} V')
    xr = np.linspace(0, s.x.max()*1.1, 40)
    ax.plot(xr, o['sl']*xr + o['b'], 'k-', lw=1.6, zorder=4)
    o40, _ = res[(vn, '40 um, untruncated')]
    hi = f"{o['hi']:.1f}" if np.isfinite(o['hi']) else '∞'
    ax.annotate(f"{vn}\n$N$ = {o['N']},  slope = {o['sl']:.3f}\n"
                f"intercept = {o['b']:.3f} $\\pm$ {o['se']:.3f}\n"
                f"Adj. $R^2$ = {o['adj']:.3f}\n"
                f"$f_T$ = {o['fT']:.1f} GHz  ({o['lo']:.1f}–{hi})\n"
                f"40 $\\mu$m alone: $f_T$ = {o40['fT']:.1f} GHz, slope {o40['sl']:.2f}",
                xy=(0.035, 0.965), xycoords='axes fraction', va='top', fontsize=8.6)
    ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}$)')
    ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}$)')
    ax.set_xlim(0, 9.2); ax.set_ylim(0, 11.5)
    ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7, loc='lower right')
axs[1].set_title(f'per-device: $R_s$ {A.Rs.min():.1f}–{A.Rs.max():.1f} $\\Omega$, '
                 f'$C_{{CPW}}$ {A.Cc.min():.0f}–{A.Cc.max():.0f} fF', fontsize=10)
axs[2].set_title(f'joint: $R_s$ = {RsB:.2f} $\\Omega$, $C_{{CPW}}$ = {CcB*1e15:.1f} fF',
                 fontsize=10)
axs[0].set_title(f'locked: $R_s$ = {Rs_FIX} $\\Omega$, $C_{{CPW}}$ = {C_CPW_FIX*1e15:.2f} fF',
                 fontsize=10)
fig.tight_layout()
fig.savefig('ft_40um_free.png', dpi=300)
A.to_csv('ft_40um_free_perdevice.csv', index=False)
B.to_csv('ft_40um_free_joint.csv', index=False)
print('\nwrote ft_40um_free.png')
