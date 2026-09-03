"""S11 fits and frequency responses behind the f_T extraction (ft_userbw.py).

For every device/bias in the extraction: (a) Smith chart, measured S11 as open
squares and the fitted 2-L ladder as a solid line; (b) normalised response,
measured Cal RF POW (scatter, 3rd-order-polynomial DC reference) against the
model |H_ph * H_ckt| (solid), with the modelled f_RC and the measured f_3dB
marked. One Smith figure and one response figure per bias.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)                       # PAIR, C_PD_CV, sheet_path, read_s11, H_ckt, ...

# ── H_ph: staircase-tau_A baseline (unchanged) ─────────────────────────────
W_A, W_Ad, W_C = 480e-9, 160e-9, 820e-9
W_norm = W_A + W_C + 2*W_Ad
tau_A, tau_R, tau_eD, tau_C = 1.989e-12, 0.0, 2.026e-12, 7.794e-12
tau_h = W_Ad/4.8e4
def H_ph(w):
    s = lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            + W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            + W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            + W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm

t = pd.read_csv('ft_userbw.csv')
t['note'] = t['note'].fillna('')

def draw_smith(ax):
    ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='k', lw=1.3))
    ax.plot([-1, 1], [0, 0], color='k', lw=0.8)
    rs = [0.2, 0.5, 1.0, 2.0, 5.0]
    for r in rs:
        ax.add_patch(plt.Circle((r/(r+1), 0), 1/(r+1), fill=False,
                                color='0.6', lw=0.5, ls=(0, (2, 2))))
        ax.text((r-1)/(r+1), -0.05, f'{r:g}', fontsize=7, ha='center',
                va='top', color='k')
    th = np.linspace(0, np.pi, 300)
    for x in rs:
        for sg in (1, -1):
            xx = 1 + np.cos(th)/x; yy = sg/x + sg*np.sin(th)/x
            m = xx**2 + yy**2 <= 1.0005
            ax.plot(xx[m], yy[m], color='0.6', lw=0.5, ls=(0, (2, 2)))
            g = (1j*sg*x-1)/(1j*sg*x+1); a = np.angle(g)
            ax.text(1.11*np.cos(a), 1.11*np.sin(a),
                    ('-' if sg < 0 else '') + f'{x:g}j', fontsize=7,
                    ha='center', va='center')

def meas_response(sheet):
    df = pd.read_excel(sheet_path(sheet), header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6]); f, p = f[k], p[k]
    c = np.polyfit(f, p, 3)
    return f, p - np.polyval(c, 0.0)

CL = {25: '#1f77b4', 30: '#c0392b', 40: '#2e8b57'}
lookup = {(D, V, lab, camp): (s11, sheet) for D, V, lab, camp, s11, sheet, _ in PAIR}

for V in (-7, -5, -3):
    sub = t[t.V == V].sort_values(['D', 'Rm']).reset_index(drop=True)
    n = len(sub); ncol = 4 if n > 4 else n; nrow = int(np.ceil(n/ncol))

    # ── Smith ─────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3*ncol, 3.4*nrow),
                            squeeze=False)
    for i, r in sub.iterrows():
        ax = axs[i//ncol, i % ncol]; draw_smith(ax)
        s11rel, sheet = lookup[(r.D, r.V, r.lab, r.camp)]
        f, Sm = read_s11(os.path.join(S11D, s11rel)); w = 2*np.pi*f
        Rm = np.inf if r['open'] else r.Rm
        Sf = S11_model(w, r.Cpd*1e-15, Rm, r.L1*1e-12, r.Lm*1e-12, r.L2*1e-12)
        col = CL[r.D]
        ax.scatter(Sm.real, Sm.imag, s=13, facecolors='none', edgecolors=col,
                   marker='s', linewidths=0.9, zorder=5)
        ax.plot(Sf.real, Sf.imag, '-', color=col, lw=1.8, zorder=6)
        dev = 'open' if r['open'] else f'{r.Rm:.1f} $\\Omega$'
        ax.text(0.02, 0.98, f'({chr(97+i)})', transform=ax.transAxes,
                fontsize=11, va='top')
        ax.text(0.30, 0.95, f'{r.D:.0f} $\\mu$m, {dev}\n{V} V, {r.camp}\n'
                f'rms {r.rms:.3f}', transform=ax.transAxes, fontsize=8,
                va='top')
    for k in range(n, nrow*ncol):
        axs[k//ncol, k % ncol].axis('off')
    fig.suptitle(f'$S_{{11}}$: measured (squares) vs fitted 2-L ladder (line), '
                 f'{V} V', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'ft_userbw_smith_{abs(V)}V.png', dpi=200, facecolor='white')

    # ── frequency response ────────────────────────────────────────────────
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9*ncol, 3.1*nrow),
                            squeeze=False)
    for i, r in sub.iterrows():
        ax = axs[i//ncol, i % ncol]
        s11rel, sheet = lookup[(r.D, r.V, r.lab, r.camp)]
        fm, pm = meas_response(sheet)
        xmax = max(40.0, np.ceil(fm[-1]/5)*5)
        fp = np.linspace(0.05e9, xmax*1e9, 2500); wp = 2*np.pi*fp
        Rm = np.inf if r['open'] else r.Rm
        H = H_ph(wp)*H_ckt(wp, r.Cpd*1e-15, Rm, r.L1*1e-12, r.Lm*1e-12,
                           r.L2*1e-12)
        Hc = H_ckt(wp, r.Cpd*1e-15, Rm, r.L1*1e-12, r.Lm*1e-12, r.L2*1e-12)
        col = CL[r.D]
        ax.scatter(fm, pm, s=16, facecolors='none', edgecolors=col, marker='s',
                   linewidths=0.9, zorder=5, label='measured')
        ax.plot(fp/1e9, 20*np.log10(np.abs(H)/np.abs(H[0])), '-', color=col,
                lw=1.8, zorder=4, label=r'model $H_{ph}H_{ckt}$')
        ax.plot(fp/1e9, 20*np.log10(np.abs(Hc)/np.abs(Hc[0])), '--',
                color='0.45', lw=1.1, zorder=3, label=r'$H_{ckt}$ only')
        ax.axhline(-3, color='0.6', lw=0.8, ls=':')
        ax.axvline(r.f_RC, color='0.45', lw=1.0, ls='--')
        if np.isfinite(r.f3):
            ax.axvline(r.f3, color=col, lw=1.0, ls=':')
        ax.set_xlim(0, xmax); ax.set_ylim(-15, 5)
        ax.set_yticks([-15, -10, -5, 0, 5])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, ls=':')
        dev = 'open' if r['open'] else f'{r.Rm:.1f} $\\Omega$'
        used = '' if r.ok else '   [dropped: truncated sweep]'
        ax.text(0.03, 0.96, f'({chr(97+i)})  {r.D:.0f} $\\mu$m, {dev}, {V} V, '
                f'{r.camp}{used}', transform=ax.transAxes, fontsize=8,
                va='top')
        f3s = f'{r.f3:.1f}' if np.isfinite(r.f3) else '—'
        ax.text(0.03, 0.12, f'$f_{{RC}}$ = {r.f_RC:.1f}   '
                f'$f_{{3dB}}$ = {f3s} GHz', transform=ax.transAxes,
                fontsize=8, va='bottom')
        if i % ncol == 0:
            ax.set_ylabel('Normalised response (dB)', fontsize=8.5)
        if i//ncol == nrow-1 or i + ncol >= n:
            ax.set_xlabel('Frequency (GHz)', fontsize=8.5)
        if i == 0:
            ax.legend(fontsize=6.5, loc='center right')
    for k in range(n, nrow*ncol):
        axs[k//ncol, k % ncol].axis('off')
    fig.suptitle(f'Frequency response, {V} V:  dashed grey = modelled '
                 f'$f_{{RC}}$,  dotted colour = measured $f_{{3dB}}$',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'ft_userbw_resp_{abs(V)}V.png', dpi=200, facecolor='white')
    print(f'{V} V: {n} panels -> ft_userbw_smith_{abs(V)}V.png, '
          f'ft_userbw_resp_{abs(V)}V.png')
