"""Dimension D, step 1-3: inventory of every bandwidth sheet's calibration
metadata and per-point loss columns.

Writes inv40/D_inventory.csv (one row per sheet) and
inv40/D_losscurves.pkl (dict sheet -> DataFrame of f, raw, total, probe,
link, cal, voa).  Read-only on all project data.
"""
import os, glob, hashlib, pickle, re
import numpy as np, pandas as pd

ROOT = '/home/user/UTC-PD-Simulation'
os.chdir(ROOT)

paths = []
for sub in ['40um', '30um', '35um', '25 um', '15um']:
    paths += glob.glob(f'data_PD0008_1/Bandwidth/{sub}/**/*.xlsx', recursive=True)
paths += glob.glob('data_bw_user/*.xlsx')
paths = sorted(set(paths))

def meta(df0):
    """df0: header=None frame. Returns dict of the row-0..13 metadata."""
    d = {}
    for i in range(14):
        k = df0.iat[i, 0]
        if isinstance(k, str):
            d[k.strip()] = df0.iat[i, 1]
    return d

rows, curves = [], {}
for p in paths:
    try:
        df0 = pd.read_excel(p, header=None)
    except Exception as e:
        rows.append(dict(path=p, error=str(e))); continue
    hdr_row = None
    for i in range(min(30, len(df0))):
        if isinstance(df0.iat[i, 0], str) and df0.iat[i, 0].startswith('F_BEAT'):
            hdr_row = i; break
    md = meta(df0) if hdr_row == 14 else {}
    if hdr_row is None:
        rows.append(dict(path=p, error='no F_BEAT header')); continue
    df = pd.read_excel(p, header=hdr_row)
    cols = list(df.columns)
    def col(i):
        if i < df.shape[1]:
            return pd.to_numeric(df.iloc[:, i], errors='coerce').values.astype(float)
        return np.full(len(df), np.nan)
    f, ipd, raw, tot, prb, lnk, cal, voa = [col(i) for i in range(8)]
    if df.shape[1] < 8:
        print('NOTE: only', df.shape[1], 'columns in', p, cols)
    m = np.isfinite(f) & np.isfinite(raw)
    n_all = int(m.sum())
    n_sat = int((raw[m] >= 0).sum())
    resid = cal - (raw + tot)
    resid_tp = tot - (prb + lnk)
    with open(p, 'rb') as fh:
        h = hashlib.md5(fh.read()).hexdigest()
    r = dict(path=p, md5=h, n_pts=n_all, n_raw_ge0=n_sat,
             f_min=float(np.nanmin(f[m])) if n_all else np.nan,
             f_max=float(np.nanmax(f[m])) if n_all else np.nan,
             hdr_row=hdr_row,
             V=md.get('KEITHLEY VOLTAGE'), I0=md.get('INITIAL PHOTOCURRENT'),
             L3=md.get('STARTING WAVELENGTH FOR LASER 3'),
             L4=md.get('STARTING WAVELENGTH FOR LASER 4'),
             xlsx_loss=md.get('EXCEL LOSS FILE'), s2p_loss=md.get('S2P LOSS FILE'),
             date=md.get('DATE'), time=md.get('TIME'),
             link_max_abs=float(np.nanmax(np.abs(lnk[m]))) if n_all else np.nan,
             cal_minus_rawtot_maxabs=float(np.nanmax(np.abs(resid[m]))) if n_all else np.nan,
             tot_minus_prblnk_maxabs=float(np.nanmax(np.abs(resid_tp[m]))) if n_all else np.nan,
             ipd_mean=float(np.nanmean(ipd[m])) if n_all else np.nan,
             ipd_min=float(np.nanmin(ipd[m])) if n_all else np.nan,
             ipd_max=float(np.nanmax(ipd[m])) if n_all else np.nan,
             voa_min=float(np.nanmin(voa[m])) if n_all and np.isfinite(voa[m]).any() else np.nan,
             voa_max=float(np.nanmax(voa[m])) if n_all and np.isfinite(voa[m]).any() else np.nan,
             cols='|'.join(map(str, cols)))
    # probe loss at reference frequencies (nearest point)
    for fr in [1, 5, 10, 15, 20, 25, 30]:
        if n_all and np.nanmin(f[m]) <= fr <= np.nanmax(f[m]):
            j = np.nanargmin(np.abs(f - fr))
            r[f'probe_{fr}G'] = float(prb[j]); r[f'f_at_{fr}G'] = float(f[j])
        else:
            r[f'probe_{fr}G'] = np.nan; r[f'f_at_{fr}G'] = np.nan
    rows.append(r)
    curves[p] = pd.DataFrame(dict(f=f, ipd=ipd, raw=raw, tot=tot, prb=prb,
                                  lnk=lnk, cal=cal, voa=voa))[m]

inv = pd.DataFrame(rows)
inv.to_csv('inv40/D_inventory.csv', index=False)
with open('inv40/D_losscurves.pkl', 'wb') as fh:
    pickle.dump(curves, fh)

pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 70)
show = ['path', 'date', 'time', 'V', 'I0', 'L3', 'L4', 'n_pts', 'n_raw_ge0', 'f_min', 'f_max',
        'link_max_abs', 'cal_minus_rawtot_maxabs', 'tot_minus_prblnk_maxabs',
        'probe_10G', 'probe_15G', 'probe_20G', 'voa_min', 'voa_max']
print(inv[show].to_string())
print()
print('S2P LOSS FILE values:')
print(inv['s2p_loss'].value_counts(dropna=False).to_string())
print('EXCEL LOSS FILE values:')
print(inv['xlsx_loss'].value_counts(dropna=False).to_string())
print()
print('duplicate md5 groups:')
for h, g in inv.groupby('md5'):
    if len(g) > 1:
        print(' ', h[:8], list(g['path']))
