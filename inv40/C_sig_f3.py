"""inv40 / dimension C, part 5: how much of the 40 um f_T deficit is the common-mode
(30 um) residual signature acting on a -3 dB point that lies inside 10-17 GHz?
For each device: pipeline-style cubic f_3dB of (a) the baseline model on the measured
grid, (b) model + 30 um March signature, (c) measured; implied f_T for each.
Also the 40 um-specific residual (after signature removal) at 15..22 GHz.
Output: inv40/C_sig_f3.log, inv40/C_sig_f3.csv
"""
import os, io, numpy as np, pandas as pd
exec(open('/home/user/UTC-PD-Simulation/inv40/C_diff.py').read().split("say4('\\n=== (b)")[0])
LOG5 = io.StringIO()
def say5(*a, **k):
    print(*a, **k); print(*a, **k, file=LOG5)

def cubic_f3(f, p):
    """pipeline: cubic fit, DC ref at f=0, first -3 dB crossing; NaN if the data never reach -3 dB."""
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]; f3 = float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]]))
    else:
        rel = p - ref; k = np.where(rel <= -3.0)[0]
        f3 = float(np.interp(-3.0, [rel[k[0]], rel[k[0]-1]], [f[k[0]], f[k[0]-1]])) if len(k) and k[0] > 0 else np.nan
    return f3
def fT(f3, fRC):
    return (1/f3**2 - 1/fRC**2)**-0.5 if np.isfinite(f3) and f3 < fRC else np.nan

say5('=== pipeline cubic f_3dB applied to: model only | model + 30um-Mar signature | measured ;  implied f_T = (1/f3^2 - 1/fRC^2)^-1/2 ===')
say5(' D  V lab camp   f_RC  | f3_model  fT_model | f3_model+sig  fT_model+sig | f3_meas  fT_meas | f3_meas-sig  fT_meas-sig')
rows6 = []
for q in T.itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    fm, pm, M0 = c['fm'], c['pm'], c['M0']
    Sf = np.interp(fm, grid, S_mar)
    f3m = cubic_f3(fm, M0); f3ms = cubic_f3(fm, M0 + Sf); f3x = cubic_f3(fm, pm); f3xs = cubic_f3(fm, pm - Sf)
    say5(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s}  {q.f_RC:6.2f} |  {f3m:6.2f}   {fT(f3m, q.f_RC):6.2f}  |    {f3ms:6.2f}       {fT(f3ms, q.f_RC):6.2f}   |'
         f'  {f3x:6.2f}   {fT(f3x, q.f_RC):6.2f}  |   {f3xs:6.2f}      {fT(f3xs, q.f_RC):6.2f}')
    rows6.append(dict(D=q.D, V=q.V, lab=q.lab, camp=q.camp, f_RC=q.f_RC, f3_model=f3m, fT_model=fT(f3m, q.f_RC),
                      f3_model_sig=f3ms, fT_model_sig=fT(f3ms, q.f_RC), f3_meas=f3x, fT_meas=fT(f3x, q.f_RC),
                      f3_meas_minus_sig=f3xs, fT_meas_minus_sig=fT(f3xs, q.f_RC)))
R6 = pd.DataFrame(rows6); R6.to_csv(f'{OUT}/C_sig_f3.csv', index=False)
for D in (40, 30):
    s = R6[R6.D == D]
    say5(f'  {D} um: fT_model mean {s.fT_model.mean():.1f} | fT_model+sig mean {s.fT_model_sig.mean():.1f} (min {s.fT_model_sig.min():.1f} max {s.fT_model_sig.max():.1f})'
         f' | fT_meas mean {s.fT_meas.mean():.1f} | fT_meas-sig mean {s.fT_meas_minus_sig.mean():.1f} (min {s.fT_meas_minus_sig.min():.1f} max {s.fT_meas_minus_sig.max():.1f})')

say5('\n=== 40 um-specific residual (meas - model - 30um Mar signature), dB, at fixed frequencies ===')
say5(' D  V lab camp    10    13    15    16    17    18    19    20    21    22   fmax')
for q in T[T.D == 40].itertuples():
    c = curves[(q.D, q.V, str(q.lab), str(q.camp))]
    fm = c['fm']; rc = c['r0'] - np.interp(fm, grid, S_mar)
    vals = [np.interp(fq, fm, rc) if fq <= fm[-1] else np.nan for fq in (10, 13, 15, 16, 17, 18, 19, 20, 21, 22)]
    say5(f'{q.D:3d} {q.V:3d} {q.lab:>3s} {q.camp:>4s} ' + ' '.join(f'{v:+5.2f}' if np.isfinite(v) else '   - ' for v in vals) + f'  {fm[-1]:5.2f}')

open(f'{OUT}/C_sig_f3.log', 'w').write(LOG5.getvalue())
say5('\nwrote inv40/C_sig_f3.log, inv40/C_sig_f3.csv')
