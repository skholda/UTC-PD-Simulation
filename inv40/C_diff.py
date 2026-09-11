"""inv40 / dimension C, part 4: differential analysis against the 30 um controls.

 (a) 30 um residual signature S(f) = mean over the 30 um control sheets of
     (measured - baseline model), on a common grid; separately for the March
     campaign (8 sheets, -7/-5 V) and for all 12.
 (b) 40 um residual minus S(f): what is 40 um-specific?  Refit the single pole /
     gaussian / k on the corrected curve.
 (c) de-trended step at the 16.0 -> 16.5 GHz boundary in every sheet, in the
     calibrated power, the raw power and the loss column.
Output: inv40/C_diff.log, inv40/C_diff.csv, inv40/C_diff.png, inv40/C_step16.csv
"""
import os, io, numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open('/home/user/UTC-PD-Simulation/inv40/C_fullcurve.py').read().split("T = pd.DataFrame(rows)")[0])
LOG4 = io.StringIO()
def say4(*a, **k):
    print(*a, **k); print(*a, **k, file=LOG4)
T = pd.DataFrame(rows)
K = 20*np.log10(np.e)

grid = np.arange(0.5, 37.01, 0.25)
def signature(keys):
    M = []
    for k in keys:
        c = curves[k]
        v = np.interp(grid, c['fm'], c['r0'], left=np.nan, right=np.nan)
        M.append(v)
    M = np.array(M)
    n = np.sum(np.isfinite(M), axis=0)
    mean = np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=0)
    sd = np.nanstd(np.where(np.isfinite(M), M, np.nan), axis=0, ddof=1)
    return mean, sd, n

k30mar = [k for k in curves if k[0] == 30 and k[3] == 'Mar']
k30all = [k for k in curves if k[0] == 30]
k30jan = [k for k in curves if k[0] == 30 and k[3] == 'Jan']
k40 = [k for k in curves if k[0] == 40]
S_mar, sd_mar, n_mar = signature(k30mar)
S_all, sd_all, n_all = signature(k30all)
S_jan, sd_jan, n_jan = signature(k30jan)
S_40, sd_40, n_40 = signature(k40)

say4('=== (a) residual signature (meas - baseline model, dB): mean +- sd over sheets, by frequency ===')
say4('  f(GHz)   30um Mar (n)        30um Jan (n)        30um all (n)        40um all (n)')
for fq in (2, 5, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 25, 30, 35):
    i = int(round((fq - 0.5)/0.25))
    def fmt(m, s, n):
        return f'{m[i]:+.2f}+-{s[i]:.2f} ({n[i]:2d})' if n[i] > 0 else '      -        '
    say4(f'  {fq:5.1f}   {fmt(S_mar, sd_mar, n_mar)}   {fmt(S_jan, sd_jan, n_jan)}   {fmt(S_all, sd_all, n_all)}   {fmt(S_40, sd_40, n_40)}')

say4('\n=== (b) 40 um residual after subtracting the 30 um signature; refits on the corrected curve ===')
say4('  signature = 30 um March mean (8 sheets, -7/-5 V);  [all12] uses all 12 30 um sheets')
say4(' D  V lab camp  | corrected residual: rms  @10  @15  @20  end | pole tau_x(ps) f_x rms | gauss f_g rms | k  rms | [all12] tau_x  f_g  k')
rows4 = []
for q in T[T.D == 40].itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    fm, pm, M0 = c['fm'], c['pm'], c['M0']; w = 2*np.pi*fm*1e9
    out = {}
    for tag, S in (('mar', S_mar), ('all', S_all)):
        Sf = np.interp(fm, grid, S)
        pc = pm - Sf; rc = pc - M0
        fp = least_squares(lambda p: pc - (M0 - 10*np.log10(1 + (w*p[0]*1e-12)**2)), [5.0], bounds=([0.0], [200.0]))
        a = max(0.0, -np.sum(rc*fm**2)/(K*np.sum(fm**4))); f_g = 1/np.sqrt(a) if a > 0 else np.inf
        fk = least_squares(lambda p: pc - (dB(H_ph(w, p[0])) - dB(H_ph(2*np.pi*1e6, p[0])) + c['Mckt']), [1.5], bounds=([0.0], [20.0]))
        at = lambda fq: float(np.interp(fq, fm, rc)) if fq <= fm[-1] else np.nan
        out[tag] = dict(rms=float(np.sqrt(np.mean(rc**2))), r10=at(10), r15=at(15), r20=at(20), rend=float(rc[-1]),
                        tau=fp.x[0], fx=1e3/(2*np.pi*fp.x[0]) if fp.x[0] > 0 else np.inf, rms_pole=float(np.sqrt(np.mean(fp.fun**2))),
                        f_g=f_g, rms_g=float(np.sqrt(np.mean((rc + K*a*fm**2)**2))), k=fk.x[0], rms_k=float(np.sqrt(np.mean(fk.fun**2))))
        if tag == 'mar':
            c['rc_mar'] = rc
    m, a2 = out['mar'], out['all']
    say4(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}  |   {m["rms"]:.2f}  {m["r10"]:+.2f} {m["r15"]:+.2f} {m["r20"]:+.2f} {m["rend"]:+.2f} |'
         f'   {m["tau"]:5.2f}  {m["fx"]:5.1f} {m["rms_pole"]:.3f} |  {m["f_g"]:5.1f} {m["rms_g"]:.3f} | {m["k"]:.2f} {m["rms_k"]:.3f} |'
         f'   {a2["tau"]:5.2f}  {a2["f_g"]:5.1f} {a2["k"]:.2f}')
    rows4.append(dict(D=q.D, V=q.V, lab=q.lab, camp=q.camp, **{f'{kk}_mar': vv for kk, vv in m.items()},
                      **{f'{kk}_all12': vv for kk, vv in a2.items()}))
D4 = pd.DataFrame(rows4); D4.to_csv(f'{OUT}/C_diff.csv', index=False)
for tag in ('mar', 'all12'):
    v = D4[f'tau_{tag}']; g = D4[f'f_g_{tag}'].replace(np.inf, np.nan); kk = D4[f'k_{tag}']
    say4(f'  [{tag}] 40 um corrected: tau_x mean {v.mean():.2f} sd {v.std(ddof=1):.2f} (min {v.min():.2f} max {v.max():.2f});'
         f'  f_g mean {g.mean():.1f} sd {g.std(ddof=1):.1f};  k mean {kk.mean():.2f} sd {kk.std(ddof=1):.2f}')
v7 = D4[D4.V == -7]
say4(f'  [mar] 40 um -7 V only: tau_x mean {v7.tau_mar.mean():.2f} sd {v7.tau_mar.std(ddof=1):.2f};  '
     f'f_g mean {v7.f_g_mar.mean():.1f};  k mean {v7.k_mar.mean():.2f}')
# same correction applied to each 30 um sheet against the signature of the OTHER 30 um sheets (leave-one-out) as a null test
say4('\n  null test: each 30 um sheet corrected by the leave-one-out signature of the other 30 um sheets (same campaign), pole refit:')
for k in k30all:
    others = [kk for kk in (k30mar if k[3] == 'Mar' else k30jan) if kk != k]
    S, _, _ = signature(others)
    c = curves[k]; fm, pm, M0 = c['fm'], c['pm'], c['M0']; w = 2*np.pi*fm*1e9
    Sf = np.interp(fm, grid, S, left=np.nan, right=np.nan); ok = np.isfinite(Sf)
    pc = (pm - Sf)[ok]; M = M0[ok]; ww = w[ok]
    fp = least_squares(lambda p: pc - (M - 10*np.log10(1 + (ww*p[0]*1e-12)**2)), [5.0], bounds=([0.0], [200.0]))
    rc = pc - M
    say4(f'    30 um {k[1]:3d} V {k[2]:>3s} {k[3]}: corrected rms {np.sqrt(np.mean(rc**2)):.2f}  tau_x {fp.x[0]:5.2f} ps   '
         f'(@10 {np.interp(10, fm[ok], rc):+.2f}, @15 {np.interp(15, fm[ok], rc):+.2f}, @20 {np.interp(20, fm[ok], rc) if fm[ok][-1] >= 20 else np.nan:+.2f})')

# ── (c) step at the 16.0 -> 16.5 GHz boundary, de-trended, in cal / raw / loss ──
say4('\n=== (c) de-trended step across the 16.0->16.5 GHz boundary (line fit 13-16.1 GHz vs 16.4-19.5 GHz, evaluated at 16.25 GHz) ===')
say4(' D  V lab camp  n_lo n_hi   step_cal(dB)  step_raw(dB)  step_loss(dB)   sheet')
rows5 = []
for q, r in zip(T.itertuples(), sel.itertuples()):
    key = (q.D, q.V, str(q.lab), str(q.camp)); s11rel, sheet = lookup[key]
    df = pd.read_excel(sheet_path(sheet), header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    cal = pd.to_numeric(df.iloc[:, 6], errors='coerce').values
    raw = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
    loss = pd.to_numeric(df.iloc[:, 3], errors='coerce').values
    m = np.isfinite(f) & np.isfinite(cal) & (f > 0) & (cal < 0)
    f, cal, raw, loss = f[m], cal[m], raw[m], loss[m]
    o = np.argsort(f, kind='stable'); f, cal, raw, loss = f[o], cal[o], raw[o], loss[o]
    lo = (f >= 13.0) & (f <= 16.1); hi = (f >= 16.4) & (f <= 19.5)
    def step(y):
        if lo.sum() < 3 or hi.sum() < 3:
            return np.nan
        a = np.polyfit(f[lo], y[lo], 1); b = np.polyfit(f[hi], y[hi], 1)
        return float(np.polyval(b, 16.25) - np.polyval(a, 16.25))
    sc, sr, sl = step(cal), step(raw), step(loss)
    say4(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}   {lo.sum():2d}   {hi.sum():2d}     {sc:+6.2f}        {sr:+6.2f}        {sl:+6.2f}       {os.path.basename(sheet)}')
    rows5.append(dict(D=q.D, V=q.V, lab=q.lab, camp=q.camp, n_lo=int(lo.sum()), n_hi=int(hi.sum()), step_cal=sc, step_raw=sr, step_loss=sl))
S5 = pd.DataFrame(rows5); S5.to_csv(f'{OUT}/C_step16.csv', index=False)
for D in (40, 30):
    v = S5[(S5.D == D)].step_cal.dropna()
    say4(f'  {D} um: step_cal mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} (n {len(v)}, min {v.min():+.2f} max {v.max():+.2f})')

# ── figure ─────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axs[0]
ax.fill_between(grid, S_mar - sd_mar, S_mar + sd_mar, color='#c0392b', alpha=0.15, label='30 um Mar mean ± sd (8)')
ax.plot(grid, S_mar, '-', color='#c0392b', lw=1.8)
ax.plot(grid, S_jan, '--', color='#e67e22', lw=1.4, label='30 um Jan mean (4)')
for k in k40:
    c = curves[k]; ax.plot(c['fm'], c['r0'], '-', color='#2e8b57', lw=0.9, alpha=0.8)
ax.plot([], [], '-', color='#2e8b57', label='40 um sheets (7)')
ax.axhline(0, color='k', lw=0.8); ax.set_xlim(0, 37); ax.set_ylim(-5, 2.5)
ax.set_xlabel('f (GHz)'); ax.set_ylabel('meas - baseline model (dB)'); ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=8)
ax.set_title('baseline residuals: 30 um signature vs 40 um', fontsize=10)
ax = axs[1]
for k in k40:
    c = curves[k]; ax.plot(c['fm'], c['rc_mar'], '-o', ms=2.5, lw=0.9, label=f'{k[2]} Ω {k[1]} V {k[3]}')
ax.axhline(0, color='k', lw=0.8); ax.set_xlim(0, 24); ax.set_ylim(-4, 1.5)
ax.set_xlabel('f (GHz)'); ax.set_ylabel('40 um residual minus 30 um Mar signature (dB)'); ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7)
ax.set_title('40 um-specific part', fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/C_diff.png', dpi=170, facecolor='white'); plt.close(fig)

open(f'{OUT}/C_diff.log', 'w').write(LOG4.getvalue())
say4('\nwrote inv40/C_diff.log, inv40/C_diff.csv, inv40/C_step16.csv, inv40/C_diff.png')
