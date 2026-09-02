"""Unit slope as a model-selection test for the f_RC circuit.

For each candidate topology the S11 of every device is re-fitted with that
topology, f_RC is taken from the SAME topology's H_ckt, and the free-slope fit
of 1/f_3dB^2 vs 1/f_RC^2 is reported. The quadrature relation forces unit
slope, so a topology that does not return slope ~ 1 with a positive intercept
is disqualified as an f_RC model -- this is the test the earlier locked-slope
fit was hiding.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ft_extraction_multiD.py').read()
exec(src.split('# ── main loop')[0])          # constants, readers, DEV, f3dB_of

# ── topologies ─────────────────────────────────────────────────────────────
# Each returns the ABCD chain from the photocurrent node to the port.
# order of shunt/series elements is what distinguishes them.
def chain(w, Cpd, Rm, L1, Lm, L2, topo):
    A = np.ones_like(w, dtype=complex); B = np.zeros_like(w, dtype=complex)
    C = np.zeros_like(w, dtype=complex); D = np.ones_like(w, dtype=complex)
    def ser(Z):
        nonlocal B, D
        B, D = A*Z + B, C*Z + D
    def sh(Y):
        nonlocal A, C
        A, C = A + B*Y, C + D*Y
    Ym = (1/(Rm + 1j*w*Lm)) if np.isfinite(Rm) else 0.0
    sh(1j*w*Cpd); ser(Rs_FIX + 0j*w)
    if topo == '2L':          # C_CPW - L1 - Rm - L2      (project baseline)
        sh(1j*w*C_CPW_FIX); ser(1j*w*L1); sh(Ym); ser(1j*w*L2)
    elif topo == '1L':        # C_CPW - Rm - L2           (single bridge)
        sh(1j*w*C_CPW_FIX); sh(Ym); ser(1j*w*L2)
    elif topo == 'CafterL':   # L1 - C_CPW - Rm - L2
        ser(1j*w*L1); sh(1j*w*C_CPW_FIX); sh(Ym); ser(1j*w*L2)
    elif topo == 'RmFirst':   # Rm - L1 - C_CPW - L2
        sh(Ym); ser(1j*w*L1); sh(1j*w*C_CPW_FIX); ser(1j*w*L2)
    else:
        raise ValueError(topo)
    return A, B, C, D

def Zin_t(w, Cpd, Rm, L1, Lm, L2, topo):
    A, B, C, D = chain(w, Cpd, Rm, L1, Lm, L2, topo)
    return D/C          # looking in from the port, source side left open

def S11_t(w, *a):
    Z = Zin_t(w, *a)
    return (Z - R_L)/(Z + R_L)

def H_t(w, *a):
    A, B, C, D = chain(w, *a)
    return R_L/(C*R_L + D)

TOPO = {'2L':      ('2-L ladder  C_CPW-L1-Rm-L2   (baseline)', ('L1', 'Lm', 'L2')),
        '1L':      ('1-L ladder  C_CPW-Rm-L2',                 ('Lm', 'L2')),
        'CafterL': ('L1-C_CPW-Rm-L2',                          ('L1', 'Lm', 'L2')),
        'RmFirst': ('Rm-L1-C_CPW-L2',                          ('L1', 'Lm', 'L2'))}

fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
base = pd.read_csv('ft_extraction_multiD.csv')
base['flag'] = base['flag'].fillna('')

def fit_device(f, Sm, Cpd, Rm, topo, free):
    w = 2*np.pi*f
    p0 = {'L1': 60.0, 'Lm': 70.0, 'L2': 130.0}
    keys = [k for k in free if not (k == 'Lm' and not np.isfinite(Rm))]
    def unpack(p):
        v = {'L1': 0.0, 'Lm': 0.0, 'L2': 0.0}
        for k, val in zip(keys, p):
            v[k] = val*1e-12
        return v['L1'], v['Lm'], v['L2']
    def resid(p):
        L1, Lm, L2 = unpack(p)
        e = S11_t(w, Cpd, Rm, L1, Lm, L2, topo) - Sm
        return np.concatenate([e.real, e.imag])
    r = least_squares(resid, [p0[k] for k in keys],
                      bounds=([0.0]*len(keys), [600.0]*len(keys)),
                      x_scale=[50.0]*len(keys))
    L1, Lm, L2 = unpack(r.x)
    rms = np.sqrt(np.mean(np.abs(S11_t(w, Cpd, Rm, L1, Lm, L2, topo) - Sm)**2))
    return L1, Lm, L2, rms

def freefit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [s, b])**2).sum()/((y - y.mean())**2).sum()
    adj = 1 - (1 - r2)*(len(x) - 1)/(len(x) - 2)
    return s, b, adj, (np.sqrt(1000/b) if b > 0 else np.nan)

# devices: the -7 V rows that survived screening
rows7 = [r for r in DEV if r[1] == -7]
out = {}
for key, (name, free) in TOPO.items():
    recs = []
    for D, V, lab, s11rel, bwrel in rows7:
        m = base[(base.D == D) & (base.V == V) & (base.lab == lab)]
        if not len(m) or not bool(m.iloc[0].use):
            continue
        m = m.iloc[0]
        f, Sm = read_s11(os.path.join(S11D, s11rel))
        Rm = np.inf if m['open'] else m['Rm']
        Cpd = m['Cpd']*1e-15
        L1, Lm, L2, rms = fit_device(f, Sm, Cpd, Rm, key, free)
        fRC = f3dB_of(fg, H_t(wg, Cpd, Rm, L1, Lm, L2, key))/1e9
        recs.append(dict(D=D, lab=lab, L1=L1*1e12, Lm=Lm*1e12, L2=L2*1e12,
                         rms=rms, fRC=fRC, f3=m['f3']))
    t = pd.DataFrame(recs)
    x = 1000/t.fRC.values**2; y = 1000/t.f3.values**2
    s, b, adj, ft = freefit(x, y)
    out[key] = dict(name=name, t=t, s=s, b=b, adj=adj, ft=ft,
                    rms=t.rms.mean(), x=x, y=y)

print(f"{'topology':<40s} {'<S11 rms>':>9s} {'slope':>7s} {'intercept':>10s} "
      f"{'Adj.R2':>7s} {'f_T (GHz)':>10s}")
for k, o in out.items():
    ft = f"{o['ft']:10.1f}" if np.isfinite(o['ft']) else f"{'none':>10s}"
    print(f"{o['name']:<40s} {o['rms']:9.3f} {o['s']:7.3f} {o['b']:+10.3f} "
          f"{o['adj']:7.3f} {ft}")

# ── plot every topology ────────────────────────────────────────────────────
MK = {25: 'o', 30: 's', 40: '^'}
CLR = {25: '#1f77b4', 30: '#c0392b', 40: '#2e8b57'}
fig, axs = plt.subplots(2, 2, figsize=(10.6, 9.2))
for ax, (k, o) in zip(axs.ravel(), out.items()):
    t = o['t']
    for D in (25, 30, 40):
        s = t[t.D == D]
        if not len(s):
            continue
        ax.scatter(1000/s.fRC**2, 1000/s.f3**2, s=58, marker=MK[D],
                   facecolor='none', edgecolor=CLR[D], lw=1.6, zorder=4,
                   label=f'{D} $\\mu$m')
    xr = np.linspace(0, o['x'].max()*1.15, 40)
    ax.plot(xr, o['s']*xr + o['b'], 'k-', lw=1.5, zorder=3)
    ax.plot(xr, xr, ':', color='0.55', lw=1.2, zorder=2, label='unit slope')
    ft = f"$f_T$ = {o['ft']:.1f} GHz" if np.isfinite(o['ft']) else \
         '$f_T$ not extractable'
    ax.annotate(f"slope = {o['s']:.3f}\nintercept = {o['b']:+.3f}\n"
                f"Adj. $R^2$ = {o['adj']:.3f}\n{ft}\n"
                f"$\\langle S_{{11}}$ rms$\\rangle$ = {o['rms']:.3f}",
                xy=(0.035, 0.96), xycoords='axes fraction', va='top',
                fontsize=8.5)
    ax.set_title(o['name'], fontsize=9)
    ax.set_xlabel(r'$1000/f_{RC}^{2}$   (GHz$^{-2}\times10^{3}$)')
    ax.set_ylabel(r'$1000/f_{3dB}^{2}$   (GHz$^{-2}\times10^{3}$)')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7.5, loc='lower right')
fig.suptitle(r'$-7$ V, $N$ = %d — unit slope as the model test for $f_{RC}$'
             % len(out['2L']['t']), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig('ft_topology_test.png', dpi=300)
print('\nwrote ft_topology_test.png')
