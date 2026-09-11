"""Dimension D, steps 1,2,4,5: loss curves per cal file, f3dB sensitivity to
the calibration, VOA / I_PD drift with frequency.

Inputs : inv40/D_losscurves.pkl, inv40/D_inventory.csv (from D_inventory.py)
Outputs: inv40/D_losscurves.png, inv40/D_lossref.csv, inv40/D_f3_calswap.csv,
         inv40/D_f3_calswap.png, inv40/D_voa_ipd_drift.csv, inv40/D_voa_drift.png,
         inv40/D_session_witness.png
All numbers here are exploratory; the project baseline is untouched.
"""
import os, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir('/home/user/UTC-PD-Simulation')

C = pickle.load(open('inv40/D_losscurves.pkl', 'rb'))
inv = pd.read_csv('inv40/D_inventory.csv')
inv['s2p'] = inv['s2p_loss'].astype(str).str.split('/').str[-1]

# ------------------------------------------------------------------ helpers
def clean(f, p):
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6]); return f[k], p[k]

def f3_of(f, p):
    """Same algorithm as ft_userbw.user_f3: cubic, DC ref at f=0, first -3 dB."""
    f, p = clean(f, p)
    if len(f) < 5: return np.nan, 'none', np.nan
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]; return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), 'poly3', f[-1]
    rel = p - ref; k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]; return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]])), 'raw', f[-1]
    return np.nan, 'none', f[-1]

def rel_at(f, p, fq):
    """relative response (dB) at fq from the cubic with DC ref at f=0, and the
    raw nearest-point value relative to the same reference."""
    f, p = clean(f, p)
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    out = {}
    for q in fq:
        if q <= f[-1]:
            j = np.argmin(np.abs(f - q))
            out[q] = (float(np.polyval(c, q) - ref), float(p[j] - ref), float(f[j]))
        else:
            out[q] = (np.nan, np.nan, np.nan)
    return out

# ------------------------------------------------------- 1. reference curves
# applied loss = Cal - Raw (this is what actually corrected the data; in the
# hybrid 55-ohm sheet it equals the Probe column, not the Total column)
def applied(d): return (d.cal - d.raw).values

refpts = {'A': [], 'B': []}
for _, r in inv.iterrows():
    p = r['path']
    if p not in C or r['n_pts'] < 5 or 'new cal' in p: continue
    d = C[p]; d = d[d.raw < 0]
    fam = 'A' if r['s2p'] == 'Cal.s2p' else 'B'
    refpts[fam].append(np.c_[d.f.values, applied(d)])
ref = {}
for fam in 'AB':
    a = np.vstack(refpts[fam]); a = a[np.isfinite(a).all(1)]
    df = pd.DataFrame(a, columns=['f', 'L']).groupby('f')['L'].agg(['median', 'std', 'count']).reset_index()
    ref[fam] = df
    print(f'family {fam}: {len(a)} pooled points, {len(df)} unique f, f range {df.f.min():.2f}-{df.f.max():.2f} GHz, '
          f'max within-f std {df["std"].max():.3f} dB (loss is deterministic in f)')
FQ = [1, 5, 10, 15, 20, 25, 30]
def Lref(fam, f):
    d = ref[fam]; return np.interp(f, d.f.values, d['median'].values)
tab = pd.DataFrame({'f_GHz': FQ, 'A_Cal.s2p_dB': [Lref('A', q) for q in FQ], 'B_S21_cal_dB': [Lref('B', q) for q in FQ]})
tab['A_minus_B_dB'] = tab['A_Cal.s2p_dB'] - tab['B_S21_cal_dB']
print('\nReference loss curves (linear interpolation of pooled per-point loss columns):')
print(tab.round(3).to_string(index=False))
pd.concat([ref['A'].assign(fam='A_Cal.s2p'), ref['B'].assign(fam='B_S21_cal')]).to_csv('inv40/D_lossref.csv', index=False)
fg = np.linspace(0.5, 30, 600)
dAB = Lref('A', fg) - Lref('B', fg)
print(f'A-B over 0.5-30 GHz: min {dAB.min():+.2f} dB at {fg[dAB.argmin()]:.1f} GHz, max {dAB.max():+.2f} dB at {fg[dAB.argmax()]:.1f} GHz; '
      f'|A-B| <= 20 GHz max {np.abs(dAB[fg<=20]).max():.2f} dB at {fg[fg<=20][np.abs(dAB[fg<=20]).argmax()]:.1f} GHz')

# classify every sheet by its applied-loss curve (not by header)
def classify(p):
    d = C[p]; d = d[d.raw < 0]
    if len(d) < 5: return 'n/a', np.nan, np.nan
    L = applied(d); f = d.f.values
    ea = np.sqrt(np.nanmean((L - Lref('A', f))**2)); eb = np.sqrt(np.nanmean((L - Lref('B', f))**2))
    return ('A' if ea < eb else 'B'), ea, eb
inv['fam_applied'] = [classify(p)[0] if p in C else 'n/a' for p in inv.path]
inv['rms_to_A'] = [classify(p)[1] if p in C else np.nan for p in inv.path]
inv['rms_to_B'] = [classify(p)[2] if p in C else np.nan for p in inv.path]
mism = inv[(inv.fam_applied != 'n/a') & (((inv.s2p == 'Cal.s2p') & (inv.fam_applied == 'B')) | ((inv.s2p != 'Cal.s2p') & (inv.fam_applied == 'A')))]
print('\nSheets whose applied loss curve does NOT match the S2P LOSS FILE named in their header:')
print(mism[['path', 'date', 's2p', 'fam_applied', 'rms_to_A', 'rms_to_B']].to_string(index=False))
inv.to_csv('inv40/D_inventory.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for _, r in inv.iterrows():
    p = r['path']
    if p not in C or r['n_pts'] < 5: continue
    d = C[p]; d = d[d.raw < 0]
    col = 'tab:blue' if r['fam_applied'] == 'A' else 'tab:red'
    ax[0].plot(d.f, applied(d), '-', color=col, lw=0.6, alpha=0.5)
ax[0].plot(ref['A'].f, ref['A']['median'], 'k-', lw=1.5, label='A = Cal.s2p (01/21, 02/03-04, 03/29 + "new cal")')
ax[0].plot(ref['B'].f, ref['B']['median'], 'k--', lw=1.5, label='B = S21_cal_bias_tee_cal.s2p (02/16, 02/25, 03/27)')
ax[0].set_xlabel('beat frequency (GHz)'); ax[0].set_ylabel('applied loss = Cal RF POW - Raw RF POW (dB)')
ax[0].set_title('Loss correction actually applied, every sheet'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_xlim(0, 40)
ax[1].plot(fg, dAB, 'k-'); ax[1].axhline(0, color='gray', lw=.5)
ax[1].set_xlabel('beat frequency (GHz)'); ax[1].set_ylabel('A - B (dB)')
ax[1].set_title('Difference between the two cal files'); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig('inv40/D_losscurves.png', dpi=130); plt.close(fig)

# --------------------------------------------- 4. f3dB under swapped cal
exec(open('ft_userbw.py').read().split('fg = np.linspace(1e6, 200e9')[0])   # PAIR, sheet_path
base = pd.read_csv('ft_userbw.csv')
rows = []
for (D, V, lab, camp, s11, sheet, note) in PAIR:
    p = sheet_path(sheet); d = C[p]; fam = inv.loc[inv.path == p, 'fam_applied'].iloc[0]
    f = d.f.values; raw = d.raw.values; cal = d.cal.values
    f3_rec, src_rec, fmax = f3_of(f, cal)
    f3_raw, src_raw, _ = f3_of(f, raw)
    f3_A, src_A, _ = f3_of(f, raw + Lref('A', f))
    f3_B, src_B, _ = f3_of(f, raw + Lref('B', f))
    other = 'B' if fam == 'A' else 'A'
    f3_other = f3_B if fam == 'A' else f3_A
    rr = rel_at(f, cal, [10, 15, 20]); rw = rel_at(f, raw, [10, 15, 20])
    b = base[(base.D == D) & (base.V == V) & (base.lab.astype(str) == lab) & (base.camp == camp)]
    rows.append(dict(D=D, V=V, lab=lab, camp=camp, sheet=os.path.basename(p), fam=fam, fmax=fmax,
                     f3_baseline_csv=float(b.f3.iloc[0]) if len(b) else np.nan,
                     f3_recorded=f3_rec, f3_nocal=f3_raw, f3_calA=f3_A, f3_calB=f3_B,
                     f3_othercal=f3_other, d_f3_swap=f3_other - f3_rec, d_f3_nocal=f3_raw - f3_rec,
                     src_rec=src_rec, src_other=src_B if fam == 'A' else src_A,
                     rel10_rec=rr[10][0], rel15_rec=rr[15][0], rel20_rec=rr[20][0],
                     rel20_raw=rw[20][0], L20_applied=Lref(fam, 20.0), L20_other=Lref(other, 20.0)))
sw = pd.DataFrame(rows)
sw.to_csv('inv40/D_f3_calswap.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print('\nf3dB (GHz) under different loss corrections; d_f3_swap = other cal file minus recorded:')
print(sw[['D', 'V', 'lab', 'camp', 'fam', 'fmax', 'f3_baseline_csv', 'f3_recorded', 'f3_nocal', 'f3_calA', 'f3_calB', 'd_f3_swap', 'd_f3_nocal', 'src_rec', 'src_other']].round(2).to_string(index=False))
print('\nrelative response (cubic, DC ref) at 10/15/20 GHz with recorded cal, raw at 20 GHz, and the loss applied at 20 GHz:')
print(sw[['D', 'V', 'lab', 'camp', 'fam', 'rel10_rec', 'rel15_rec', 'rel20_rec', 'rel20_raw', 'L20_applied', 'L20_other']].round(2).to_string(index=False))
for D in (30, 40):
    s = sw[sw.D == D]
    print(f'D={D}: |d_f3_swap| max {s.d_f3_swap.abs().max():.2f} GHz, mean {s.d_f3_swap.mean():+.2f}; |d_f3_nocal| max {s.d_f3_nocal.abs().max():.2f}, mean {s.d_f3_nocal.mean():+.2f}')

# implied f_T with swapped cal, using the baseline f_RC (from ft_userbw.csv)
sw = sw.merge(base[['D', 'V', 'lab', 'camp', 'f_RC']].assign(lab=lambda x: x.lab.astype(str)), on=['D', 'V', 'lab', 'camp'], how='left')
def fT(f3, frc): return 1.0 / np.sqrt(1.0 / f3**2 - 1.0 / frc**2) if (np.isfinite(f3) and f3 < frc) else np.nan
sw['fT_rec'] = [fT(a, b) for a, b in zip(sw.f3_recorded, sw.f_RC)]
sw['fT_other'] = [fT(a, b) for a, b in zip(sw.f3_othercal, sw.f_RC)]
sw['fT_nocal'] = [fT(a, b) for a, b in zip(sw.f3_nocal, sw.f_RC)]
print('\nimplied per-device f_T (GHz) with baseline f_RC: recorded / other cal file / no cal:')
print(sw[['D', 'V', 'lab', 'camp', 'f_RC', 'fT_rec', 'fT_other', 'fT_nocal']].round(1).to_string(index=False))
sw.to_csv('inv40/D_f3_calswap.csv', index=False)

fig, ax = plt.subplots(figsize=(7, 5))
for D, mk in ((30, 'o'), (40, 's'), (25, '^')):
    s = sw[sw.D == D]
    ax.plot(s.f3_recorded, s.f3_othercal, mk, label=f'{D} um, other cal file', alpha=.8)
    ax.plot(s.f3_recorded, s.f3_nocal, mk, mfc='none', label=f'{D} um, no cal', alpha=.8)
lim = [8, 38]; ax.plot(lim, lim, 'k-', lw=.6); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('f3dB, recorded calibration (GHz)'); ax.set_ylabel('f3dB, alternative calibration (GHz)')
ax.set_title('f3dB sensitivity to the loss correction'); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('inv40/D_f3_calswap.png', dpi=130); plt.close(fig)

# --------------------------------------------- 5. VOA / I_PD drift with f
drows = []
for _, r in inv.iterrows():
    p = r['path']
    if p not in C or r['n_pts'] < 8: continue
    d = C[p]; d = d[d.raw < 0].sort_values('f')
    f = d.f.values; v = d.voa.values; i = d.ipd.values
    ok = np.isfinite(v)
    sv = np.polyfit(f[ok], v[ok], 1)[0] if ok.sum() > 3 else np.nan
    si = np.polyfit(f, i, 1)[0] if len(f) > 3 else np.nan
    drows.append(dict(path=p, date=r['date'], fam=r['fam_applied'], fmax=f[-1],
                      voa_mean=np.nanmean(v), voa_range=np.nanmax(v) - np.nanmin(v),
                      voa_slope_dB_per_GHz=sv, voa_end_minus_start=v[ok][-1] - v[ok][0] if ok.sum() else np.nan,
                      voa_corr_f=np.corrcoef(f[ok], v[ok])[0, 1] if ok.sum() > 3 else np.nan,
                      ipd_mean=np.mean(i), ipd_rel_range_pct=100 * (np.max(i) - np.min(i)) / abs(np.mean(i)),
                      ipd_slope_pct_per_GHz=100 * si / abs(np.mean(i))))
dr = pd.DataFrame(drows); dr.to_csv('inv40/D_voa_ipd_drift.csv', index=False)
print('\nVOA P Actual and I_PD drift over each sweep (sheets with >= 8 points):')
dr['p'] = dr.path.str.replace('data_PD0008_1/Bandwidth/', 'B/', regex=False)
print(dr[['p', 'date', 'fam', 'fmax', 'voa_mean', 'voa_range', 'voa_slope_dB_per_GHz', 'voa_end_minus_start', 'voa_corr_f', 'ipd_rel_range_pct', 'ipd_slope_pct_per_GHz']].round(4).to_string(index=False))
print('\nper-date summary of VOA drift:')
print(dr.groupby('date').agg(n=('p', 'size'), voa_range_max=('voa_range', 'max'), voa_slope_mean=('voa_slope_dB_per_GHz', 'mean'),
                             voa_slope_min=('voa_slope_dB_per_GHz', 'min'), voa_slope_max=('voa_slope_dB_per_GHz', 'max'),
                             ipd_rel_range_max=('ipd_rel_range_pct', 'max')).round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for _, r in dr.iterrows():
    d = C[r['path']]; d = d[d.raw < 0].sort_values('f')
    if '40um' in r['path'] and 'data_bw_user' in r['path']:
        ax[0].plot(d.f, d.voa - d.voa.iloc[0], '-', lw=1, label=os.path.basename(r['path']).replace('.xlsx', ''))
    elif '30um' in r['path'] and 'data_bw_user' in r['path']:
        ax[1].plot(d.f, d.voa - d.voa.iloc[0], '-', lw=1, label=os.path.basename(r['path']).replace('.xlsx', ''))
for a, t in zip(ax, ('40 um baseline sheets', '30 um baseline sheets')):
    a.set_xlabel('beat frequency (GHz)'); a.set_ylabel('VOA P Actual - first point (dB)'); a.set_title(t)
    a.legend(fontsize=6, ncol=2); a.grid(alpha=.3); a.set_ylim(-0.2, 0.2)
fig.tight_layout(); fig.savefig('inv40/D_voa_drift.png', dpi=130); plt.close(fig)

# ------------------------- same-session witnesses: 15 um and 35 um sheets
print('\nSame-setup witnesses (Cal.s2p sessions 02/03-02/04): f3dB and relative response')
wit = ['data_PD0008_1/Bandwidth/15um/H.V/WO/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/35um/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/35um/Bias_-7V_Iph_1mA_diff_powermeter.xlsx',
       'data_PD0008_1/Bandwidth/40um/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/40um/100ohm/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/40um/40ohm/Bias_-7V_Iph_1mA_36ohm.xlsx',
       'data_PD0008_1/Bandwidth/40um/140hom/Bias_-7V_Iph_1mA_120ohm.xlsx',
       'data_PD0008_1/Bandwidth/40um/WO_bias__7V_Iph_1mA_#2.xlsx',
       'data_PD0008_1/Bandwidth/40um/Bias_-7V_Iph_1mA_diff_pol.xlsx',
       'data_PD0008_1/Bandwidth/40um/Bias_-9V_Iph_1mA_diff_pol.xlsx',
       'data_PD0008_1/Bandwidth/30um/WO/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/30um/120 ohm/Bias_-7V_Iph_1.034mA.xlsx',
       'data_PD0008_1/Bandwidth/30um/76 ohm/Bias_-7V_Iph_1mA.xlsx',
       'data_PD0008_1/Bandwidth/30um/40 ohm/Bias_-7V_Iph_1mA.xlsx']
fig, ax = plt.subplots(figsize=(8, 5))
for p in wit:
    d = C[p]; f, pc = clean(d.f.values, d.cal.values)
    f3, src, fm = f3_of(f, pc); rr = rel_at(f, pc, [10, 15, 20, 25])
    t = inv.loc[inv.path == p].iloc[0]
    print(f"  {p.replace('data_PD0008_1/Bandwidth/', ''):60s} {t['date']} {t['time']} fam {t['fam_applied']} fmax {fm:5.2f}  f3 {f3:6.2f} ({src})  "
          f"rel@10 {rr[10][0]:+.2f} @15 {rr[15][0]:+.2f} @20 {rr[20][0]:+.2f} @25 {rr[25][0]:+.2f} dB")
    c = np.polyfit(f, pc, 3); ref0 = np.polyval(c, 0)
    ax.plot(f, pc - ref0, '.-', lw=.8, ms=3, label=p.replace('data_PD0008_1/Bandwidth/', '').replace('Bias_-7V_Iph_1mA', '')[:45])
ax.axhline(-3, color='gray', lw=.6); ax.set_ylim(-14, 2); ax.set_xlim(0, 32)
ax.set_xlabel('beat frequency (GHz)'); ax.set_ylabel('Cal RF POW - cubic DC reference (dB)')
ax.set_title('Cal.s2p sessions: 15/35/40 um (02/04) and 30 um (01/21) sheets'); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('inv40/D_session_witness.png', dpi=130); plt.close(fig)
