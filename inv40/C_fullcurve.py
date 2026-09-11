"""inv40 / dimension C: full-curve fits of the measured RF responses.

For every 40 um device/bias in the f_T extraction (ft_userbw.csv, D == 40) and,
as control, the 30 um devices at -7 V and -5 V:

  (1) overlay measured normalised Cal RF POW (cubic f=0 reference, exactly as
      the pipeline does) on the model |H_ph * H_ckt| with the device's own
      fitted ladder (L1/Lm/L2 from ft_userbw.csv, R_s/C_CPW locked, C_PD from
      C-V); residual in dB over the measured band.
  (2) model x ONE extra single pole 1/(1 + j w tau_x), tau_x >= 0 free.
  (3) model with all transit times in H_ph scaled by a common factor k.
  (4) model x exp(-(f/f_g)^2)  (Gaussian in amplitude, quadratic in dB);
      also a linear-in-dB term exp(-f/f_e) for comparison.
  (5) ladder alone (H_ph -> 1, i.e. f_T -> infinity) with C_PD free; and the
      baseline H_ph with C_PD free.  Then: what does that C_PD do to the S11
      fit (L's fixed, and L's refitted)?

Everything here is EXPLORATORY; the project baseline is not touched.
Outputs: inv40/C_fullcurve.csv, inv40/C_fullcurve.log, inv40/C_*.png
"""
import os, sys, io, numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/home/user/UTC-PD-Simulation')
src = open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0]
exec(src)          # PAIR, C_PD_CV, sheet_path, read_s11, H_ckt, S11_model, f3dB_of, ...

OUT = 'inv40'
LOG = io.StringIO()
def say(*a, **k):
    print(*a, **k); print(*a, **k, file=LOG)

# ── H_ph, transit-time baseline from ft_userbw_plots.py, with a scale k ────
W_A, W_Ad, W_C = 480e-9, 160e-9, 820e-9
W_norm = W_A + W_C + 2*W_Ad
tau_A, tau_R, tau_eD, tau_C = 1.989e-12, 0.0, 2.026e-12, 7.794e-12
tau_h = W_Ad/4.8e4
def H_ph(w, k=1.0):
    s = lambda x: np.sinc(x/np.pi)
    tA, tR, tE, tC, th = k*tau_A, k*tau_R, k*tau_eD, k*tau_C, k*tau_h
    return (W_A/(1+1j*w*tA)*(2+1j*w*tR)/(2*(1+1j*w*tR))
            + W_C/(1+1j*w*tA)*s(w*tC/2)*np.exp(-1j*w*tC/2)
            + W_Ad*s(w*tE/2)*np.exp(-1j*w*tE/2)
            + W_Ad*s(w*th/2)*np.exp(-1j*w*th/2))/W_norm

fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
f_tr = f3dB_of(fg, H_ph(wg))/1e9
say(f'H_ph alone: f_tr = {f_tr:.2f} GHz  (expected 32.6)')

def meas_response(sheet):
    """Measured Cal RF POW in dB relative to the cubic's f=0 value (pipeline)."""
    df = pd.read_excel(sheet_path(sheet), header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6]); f, p = f[k], p[k]
    c = np.polyfit(f, p, 3)
    return f, p - np.polyval(c, 0.0), p, c

dB = lambda H: 20*np.log10(np.abs(H))

t = pd.read_csv('ft_userbw.csv'); t['note'] = t['note'].fillna('')
sel = t[(t.D == 40) | ((t.D == 30) & t.V.isin([-7, -5]))].copy()
sel = sel.sort_values(['D', 'V', 'Rm'], ascending=[False, False, True]).reset_index(drop=True)
lookup = {(D, V, lab, camp): (s11, sheet) for D, V, lab, camp, s11, sheet, _ in PAIR}

rows = []; curves = {}
for i, r in sel.iterrows():
    key = (int(r.D), int(r.V), str(r.lab), str(r.camp))
    s11rel, sheet = lookup[key]
    fm, pm, praw, cpoly = meas_response(sheet)          # fm in GHz, pm in dB
    w = 2*np.pi*fm*1e9
    Rm = np.inf if r['open'] else r.Rm
    Cpd, L1, Lm, L2 = r.Cpd*1e-15, r.L1*1e-12, r.Lm*1e-12, r.L2*1e-12
    w0 = 2*np.pi*1e6                                    # DC reference

    ckt = lambda ww, C=Cpd: H_ckt(ww, C, Rm, L1, Lm, L2)
    Hc0 = ckt(w0)
    Mckt = dB(ckt(w)) - dB(Hc0)                          # ladder alone, DC-normalised
    Mph  = lambda k=1.0: dB(H_ph(w, k)) - dB(H_ph(w0, k))
    M0   = Mph() + Mckt                                  # baseline model
    r0   = pm - M0
    n = len(fm)
    def rms(x): return float(np.sqrt(np.mean(x**2)))
    lo, hi = fm <= 15.0, fm > 15.0
    at = lambda fq: float(np.interp(fq, fm, r0)) if fq <= fm[-1] else np.nan

    # (2) extra single pole, tau_x >= 0 (ps) ; strict (no offset) and free offset
    pole = lambda tau_ps: -10*np.log10(1 + (w*tau_ps*1e-12)**2)
    fitp = least_squares(lambda p: pm - (M0 + pole(p[0])), [5.0], bounds=([0.0], [200.0]))
    tau_x = fitp.x[0]; r_pole = pm - (M0 + pole(tau_x))
    fitpo = least_squares(lambda p: pm - (M0 + pole(p[0]) + p[1]), [5.0, 0.0],
                          bounds=([0.0, -3.0], [200.0, 3.0]))
    tau_xo, c0 = fitpo.x; r_poleo = pm - (M0 + pole(tau_xo) + c0)
    # pole fitted only on f <= 15 GHz, then evaluated on the full band
    fitp15 = least_squares(lambda p: (pm - (M0 + pole(p[0])))[lo], [5.0], bounds=([0.0], [200.0]))
    tau_x15 = fitp15.x[0]

    # (3) common scale k on all transit times
    fitk = least_squares(lambda p: pm - (Mph(p[0]) + Mckt), [1.0], bounds=([0.0], [20.0]))
    kfac = fitk.x[0]; r_k = pm - (Mph(kfac) + Mckt)
    fitko = least_squares(lambda p: pm - (Mph(p[0]) + Mckt + p[1]), [1.0, 0.0],
                          bounds=([0.0, -3.0], [20.0, 3.0]))
    kfaco, c0k = fitko.x; r_ko = pm - (Mph(kfaco) + Mckt + c0k)

    # (4) Gaussian  exp(-(f/f_g)^2): dB term = -8.686*(f/f_g)^2 = -8.686*a*f^2, a>=0 (linear LSQ)
    K = 20*np.log10(np.e)
    a = max(0.0, -np.sum(r0*fm**2)/(K*np.sum(fm**4)))
    f_g = 1/np.sqrt(a) if a > 0 else np.inf
    r_g = r0 + K*a*fm**2
    # linear-in-dB  exp(-f/f_e)
    b = max(0.0, -np.sum(r0*fm)/(K*np.sum(fm**2)))
    f_e = 1/b if b > 0 else np.inf
    r_e = r0 + K*b*fm

    # (5) ladder alone (H_ph=1), C_PD free ; and baseline H_ph with C_PD free
    def Mc(C): return dB(ckt(w, C)) - dB(ckt(w0, C))
    fitC = least_squares(lambda p: pm - Mc(p[0]*1e-15), [r.Cpd], bounds=([10.0], [5000.0]))
    C_inf = fitC.x[0]; r_Cinf = pm - Mc(C_inf*1e-15)
    fitC2 = least_squares(lambda p: pm - (Mph() + Mc(p[0]*1e-15)), [r.Cpd], bounds=([10.0], [5000.0]))
    C_ph = fitC2.x[0]; r_Cph = pm - (Mph() + Mc(C_ph*1e-15))

    # what those C_PD values do to S11 (exploratory)
    fs, Sm = read_s11(os.path.join(S11D, s11rel)); ws = 2*np.pi*fs
    def s11rms(C, L1_, Lm_, L2_):
        return float(np.sqrt(np.mean(np.abs(S11_model(ws, C, Rm, L1_, Lm_, L2_) - Sm)**2)))
    rms_base = s11rms(Cpd, L1, Lm, L2)
    rms_Cinf_fixL = s11rms(C_inf*1e-15, L1, Lm, L2)
    keys = ['L1', 'L2'] if r['open'] else ['L1', 'Lm', 'L2']
    def refitL(C):
        def unpack(p):
            v = {'L1': 0.0, 'Lm': 0.0, 'L2': 0.0}
            for kk, val in zip(keys, p): v[kk] = val*1e-12
            return v['L1'], v['Lm'], v['L2']
        def resid(p):
            e = S11_model(ws, C, Rm, *unpack(p)) - Sm
            return np.concatenate([e.real, e.imag])
        rr = least_squares(resid, [60.0, 70.0, 130.0][:len(keys)] if not r['open'] else [60.0, 130.0],
                           bounds=([0.0]*len(keys), [600.0]*len(keys)), x_scale=[50.0]*len(keys))
        Ls = unpack(rr.x)
        return s11rms(C, *Ls), Ls
    rms_Cinf_refit, Ls_inf = refitL(C_inf*1e-15)
    fRC_Cinf = f3dB_of(fg, H_ckt(wg, C_inf*1e-15, Rm, L1, Lm, L2))/1e9
    fRC_Cinf_refit = f3dB_of(fg, H_ckt(wg, C_inf*1e-15, Rm, *Ls_inf))/1e9

    # model f_3dB of each fitted curve (on a dense grid), for comparison with measured f3
    def f3_of_model(fun):
        return f3dB_of(fg, fun)/1e9
    Hb = H_ph(wg)*H_ckt(wg, Cpd, Rm, L1, Lm, L2)
    f3_base = f3_of_model(Hb)
    f3_pole = f3_of_model(Hb/(1 + 1j*wg*tau_x*1e-12))
    f3_k    = f3_of_model(H_ph(wg, kfac)*H_ckt(wg, Cpd, Rm, L1, Lm, L2))
    f3_g    = f3_of_model(Hb*np.exp(-(fg/1e9/f_g)**2)) if np.isfinite(f_g) else f3_base
    f3_Cinf = f3_of_model(H_ckt(wg, C_inf*1e-15, Rm, L1, Lm, L2))

    dev = 'open' if r['open'] else f'{r.Rm:.1f}'
    rows.append(dict(D=int(r.D), V=int(r.V), lab=r.lab, camp=r.camp, Rm=dev, sheet=os.path.basename(sheet),
                     n=n, fmax=fm[-1], Cpd_CV=r.Cpd, f_RC=r.f_RC, f3_meas=r.f3, f3src=r.f3src,
                     rms0=rms(r0), mean0=float(r0.mean()), min0=float(r0.min()),
                     rms0_le15=rms(r0[lo]) if lo.any() else np.nan,
                     rms0_gt15=rms(r0[hi]) if hi.any() else np.nan,
                     r0_5=at(5.0), r0_10=at(10.0), r0_15=at(15.0), r0_20=at(20.0), r0_end=float(r0[-1]),
                     tau_x_ps=tau_x, f_x=1e3/(2*np.pi*tau_x) if tau_x > 0 else np.inf, rms_pole=rms(r_pole),
                     tau_xo_ps=tau_xo, c0_pole=c0, rms_poleo=rms(r_poleo),
                     tau_x15_ps=tau_x15,
                     k=kfac, rms_k=rms(r_k), ko=kfaco, c0_k=c0k, rms_ko=rms(r_ko),
                     f_g=f_g, rms_g=rms(r_g), f_e=f_e, rms_e=rms(r_e),
                     C_inf=C_inf, rms_Cinf=rms(r_Cinf), C_inf_ratio=C_inf/r.Cpd,
                     C_ph=C_ph, rms_Cph=rms(r_Cph), C_ph_ratio=C_ph/r.Cpd,
                     s11rms_base=rms_base, s11rms_Cinf_fixL=rms_Cinf_fixL, s11rms_Cinf_refitL=rms_Cinf_refit,
                     L1_inf=Ls_inf[0]*1e12, Lm_inf=Ls_inf[1]*1e12, L2_inf=Ls_inf[2]*1e12,
                     fRC_Cinf=fRC_Cinf, fRC_Cinf_refit=fRC_Cinf_refit,
                     f3_model_base=f3_base, f3_model_pole=f3_pole, f3_model_k=f3_k, f3_model_g=f3_g,
                     f3_model_Cinf=f3_Cinf))
    curves[key] = dict(fm=fm, pm=pm, M0=M0, Mckt=Mckt, r0=r0,
                       Mpole=M0 + pole(tau_x), Mk=Mph(kfac) + Mckt, Mg=M0 - K*a*fm**2,
                       Mcinf=Mc(C_inf*1e-15), r_pole=r_pole, r_k=r_k, r_g=r_g, r_Cinf=r_Cinf)

T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/C_fullcurve.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60)
F = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)

say('\n=== (1) baseline model |H_ph*H_ckt| vs measured: residual (meas - model, dB) over the measured band ===')
say(T[['D', 'V', 'lab', 'camp', 'Rm', 'n', 'fmax', 'f_RC', 'f3_meas', 'rms0', 'mean0', 'min0',
       'rms0_le15', 'rms0_gt15', 'r0_5', 'r0_10', 'r0_15', 'r0_20', 'r0_end', 'f3_model_base']]
    .to_string(index=False, float_format=lambda v: f'{v:.2f}'))

say('\n=== (2) extra single pole  1/(1+jw tau_x)  [strict: no DC offset]  and  [offset c0 free] ===')
say(T[['D', 'V', 'lab', 'camp', 'Rm', 'rms0', 'tau_x_ps', 'f_x', 'rms_pole', 'f3_model_pole', 'f3_meas',
       'tau_xo_ps', 'c0_pole', 'rms_poleo', 'tau_x15_ps']]
    .to_string(index=False, float_format=lambda v: f'{v:.2f}'))

say('\n=== (3) common scale k on all transit times in H_ph   [strict] and [offset free] ===')
say(T[['D', 'V', 'lab', 'camp', 'Rm', 'rms0', 'k', 'rms_k', 'f3_model_k', 'ko', 'c0_k', 'rms_ko']]
    .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

say('\n=== (4) Gaussian exp(-(f/f_g)^2) [f_g in GHz] and linear-in-dB exp(-f/f_e) [f_e in GHz] ===')
say(T[['D', 'V', 'lab', 'camp', 'Rm', 'rms0', 'rms_pole', 'rms_k', 'f_g', 'rms_g', 'f3_model_g', 'f_e', 'rms_e']]
    .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

say('\n=== (5) ladder alone (H_ph=1) with C_PD free; baseline H_ph with C_PD free; S11 consequence ===')
say(T[['D', 'V', 'lab', 'camp', 'Rm', 'Cpd_CV', 'C_inf', 'C_inf_ratio', 'rms_Cinf', 'f3_model_Cinf',
       'C_ph', 'C_ph_ratio', 'rms_Cph', 's11rms_base', 's11rms_Cinf_fixL', 's11rms_Cinf_refitL',
       'L1_inf', 'Lm_inf', 'L2_inf', 'f_RC', 'fRC_Cinf', 'fRC_Cinf_refit']]
    .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

say('\n=== summary by diameter (mean +- sd over devices; N) ===')
for D in (40, 30):
    s = T[T.D == D]
    say(f'D = {D} um, N = {len(s)}')
    for col in ('rms0', 'tau_x_ps', 'f_x', 'rms_pole', 'k', 'rms_k', 'f_g', 'rms_g', 'f_e', 'rms_e',
                'C_inf_ratio', 'C_ph_ratio', 'rms_Cinf', 'tau_xo_ps', 'c0_pole', 'rms_poleo', 'ko', 'rms_ko'):
        v = s[col].replace(np.inf, np.nan).astype(float)
        say(f'   {col:12s} mean {v.mean():8.3f}  sd {v.std(ddof=1) if len(v) > 1 else 0:7.3f}  '
            f'min {v.min():8.3f}  max {v.max():8.3f}   n_finite {v.notna().sum()}')
say('\n40 um, -7 V only (the five devices of the extraction):')
s = T[(T.D == 40) & (T.V == -7)]
for col in ('tau_x_ps', 'f_x', 'k', 'f_g', 'rms0', 'rms_pole', 'rms_k', 'rms_g', 'C_inf_ratio'):
    v = s[col].replace(np.inf, np.nan).astype(float)
    say(f'   {col:12s} mean {v.mean():8.3f}  sd {v.std(ddof=1):7.3f}  min {v.min():8.3f}  max {v.max():8.3f}')

say('\n=== which one-parameter model wins per device (lowest rms among pole / k / gauss / lin) ===')
for _, q in T.iterrows():
    cand = {'pole': q.rms_pole, 'k': q.rms_k, 'gauss': q.rms_g, 'lin': q.rms_e}
    best = min(cand, key=cand.get)
    say(f"  {q.D} um {q.V:>3d} V {q.lab:>3s} {q.camp:>4s}  base {q.rms0:.3f} | pole {q.rms_pole:.3f}  k {q.rms_k:.3f}  "
        f"gauss {q.rms_g:.3f}  lin {q.rms_e:.3f}  -> {best}")

# ── figures ─────────────────────────────────────────────────────────────────
CL = {30: '#c0392b', 40: '#2e8b57'}
def overlay(D, fname, title):
    s = T[T.D == D].reset_index(drop=True); n = len(s)
    ncol = 4 if n > 4 else n; nrow = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 3.2*nrow), squeeze=False)
    for i, q in s.iterrows():
        ax = axs[i//ncol, i % ncol]
        c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
        fm = c['fm']
        ax.scatter(fm, c['pm'], s=16, facecolors='none', edgecolors='k', marker='s', lw=0.9, zorder=6, label='measured')
        ax.plot(fm, c['M0'], '-', color=CL[D], lw=1.8, zorder=5, label='baseline $H_{ph}H_{ckt}$')
        ax.plot(fm, c['Mckt'], '--', color='0.5', lw=1.0, zorder=3, label='$H_{ckt}$ only')
        ax.plot(fm, c['Mpole'], '-', color='#2471a3', lw=1.3, zorder=5, label=f'+pole $\\tau_x$={q.tau_x_ps:.1f} ps')
        ax.plot(fm, c['Mk'], ':', color='#7d3c98', lw=1.6, zorder=5, label=f'$k$={q.k:.2f}')
        ax.plot(fm, c['Mg'], '-.', color='#d68910', lw=1.3, zorder=5, label=f'gauss $f_g$={q.f_g:.1f}')
        ax.plot(fm, c['Mcinf'], '-', color='0.3', lw=0.9, zorder=4, label=f'ladder only, $C$={q.C_inf:.0f} fF')
        ax.axhline(-3, color='0.6', lw=0.8, ls=':')
        ax.set_title(f'{q.D} um, {q.Rm} $\\Omega$, {q.V} V, {q.camp}   rms0 {q.rms0:.2f}', fontsize=9)
        ax.set_xlabel('f (GHz)'); ax.set_ylabel('dB'); ax.grid(alpha=.3, ls=':')
        ax.set_ylim(min(-12, c['pm'].min() - 1), 2)
        ax.legend(fontsize=6, loc='lower left')
    for k in range(n, nrow*ncol): axs[k//ncol, k % ncol].axis('off')
    fig.suptitle(title, fontsize=11); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{OUT}/{fname}', dpi=170, facecolor='white'); plt.close(fig)

overlay(40, 'C_overlay_40um.png', '40 um: measured (squares) vs baseline model and one-parameter corrections')
overlay(30, 'C_overlay_30um.png', '30 um controls (-7 V, -5 V): measured vs baseline model and one-parameter corrections')

def resid(D, fname, title):
    s = T[T.D == D].reset_index(drop=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, q in s.iterrows():
        c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
        lab = f'{q.Rm} Ω {q.V} V {q.camp}'
        axs[0].plot(c['fm'], c['r0'], '-o', ms=2.6, lw=1.0, label=lab)
        axs[1].plot(c['fm'], c['r_pole'], '-o', ms=2.6, lw=1.0, label=lab)
    for ax, tt in zip(axs, ['baseline residual  (meas - model, dB)', 'residual after extra pole']):
        ax.axhline(0, color='k', lw=0.8); ax.grid(alpha=.3, ls=':'); ax.set_xlabel('f (GHz)'); ax.set_ylabel('dB')
        ax.set_title(tt, fontsize=10); ax.legend(fontsize=6.5)
    fig.suptitle(title, fontsize=11); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/{fname}', dpi=170, facecolor='white'); plt.close(fig)

resid(40, 'C_resid_40um.png', '40 um: residuals')
resid(30, 'C_resid_30um.png', '30 um controls: residuals')

# parameter summary figure
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
for j, (col, yl) in enumerate([('tau_x_ps', r'$\tau_x$ (ps)'), ('k', 'k'), ('f_g', r'$f_g$ (GHz)'), ('C_inf_ratio', r'$C_{PD}$(RF, $f_T\to\infty$) / $C_{CV}$')]):
    ax = axs[j]
    for D, xoff in ((40, 0), (30, 1)):
        s = T[T.D == D]
        v = s[col].replace(np.inf, np.nan).astype(float).values
        ax.scatter(np.full(len(v), xoff) + np.linspace(-0.15, 0.15, len(v)), v, color=CL[D], s=30,
                   marker=['^', 's'][D == 30], label=f'{D} um')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['40 um', '30 um']); ax.set_ylabel(yl); ax.grid(alpha=.3, ls=':')
    if col == 'f_g': ax.set_ylim(0, 80)
axs[0].legend(fontsize=8)
fig.suptitle('one-parameter fit results per device (exploratory)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f'{OUT}/C_params.png', dpi=170, facecolor='white'); plt.close(fig)

open(f'{OUT}/C_fullcurve.log', 'w').write(LOG.getvalue())
say('\nwrote inv40/C_fullcurve.csv, inv40/C_fullcurve.log, inv40/C_overlay_40um.png, inv40/C_overlay_30um.png, '
    'inv40/C_resid_40um.png, inv40/C_resid_30um.png, inv40/C_params.png')
