"""f_3dB from the user-supplied bandwidth sheets in data_bw_user/.

Reads the 'Cal RF POW (dBm)' column of every .xlsx in that folder, fits a
3rd-order polynomial and reports the -3 dB point. Two DC references are
reported side by side because the choice shifts f_3dB:
  ref0  : the polynomial extrapolated to f = 0
  refmin: the polynomial at the lowest measured frequency
"""
import os, re, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = 'data_bw_user'

def parse_name(fn):
    """Names follow ..._<D>um_<R>ohm_<run>.xlsx (the two may be swapped).

    One sheet carries two resistances, Bias_7V_Iph_1mA_120ohm_40um_140ohm_1:
    the last "<n>ohm" is the one in the canonical slot, and 140 is also what
    that device's S11 reads at DC (142.5 ohm), so the last match wins. Files
    with more than one resistance in the name are flagged in `ambiguous`.
    """
    b = os.path.basename(fn)
    D  = re.search(r'(\d+)\s*um', b, re.I)
    Rs = re.findall(r'(\d+)\s*ohm', b, re.I)
    V  = re.search(r'Bias[_-]?(-?\d+)\s*V', b, re.I)
    I  = re.search(r'Iph[_-]?(\d+(?:[._]\d+)?)\s*mA', b, re.I)
    # "WO" (without resistor) appears as a token: _WO_, _WO., -WO, ...
    isWO = re.search(r'(?:^|[_\s.-])WO(?:[_\s.-]|$)', b, re.I) is not None
    lab = 'WO' if isWO else (Rs[-1] if Rs else '?')
    run = re.search(r'_(\d+)\.xlsx$', b)
    return dict(D=int(D.group(1)) if D else None,
                lab=lab,
                run=int(run.group(1)) if run else 1,
                Rm=(np.inf if isWO else float(Rs[-1]) if Rs else np.nan),
                V=-abs(int(V.group(1))) if V else None,
                Iph=float(I.group(1).replace('_', '.')) if I else None,
                ambiguous=('/'.join(Rs) if len(Rs) > 1 else ''),
                file=os.path.basename(fn))

def load(path):
    df = pd.read_excel(path, header=14)
    f = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    p = pd.to_numeric(df.iloc[:, 6], errors='coerce').values   # Cal RF POW (dBm)
    m = np.isfinite(f) & np.isfinite(p) & (f > 0)
    f, p = f[m], p[m]
    bad = int((p >= 0).sum())                      # saturated / bogus rows
    m = p < 0.0
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    keep = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[keep], p[keep], bad, int((~keep).sum())

def f3(f, p, ref_at_zero):
    c = np.polyfit(f, p, 3)
    ref = np.polyval(c, 0.0 if ref_at_zero else f[0])
    ff = np.linspace(0.0, f[-1], 40001)
    pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if not len(i) or i[0] == 0:
        return np.nan, c, ref
    k = i[0]
    return np.interp(-3.0, [pp[k], pp[k-1]], [ff[k], ff[k-1]]), c, ref

recs = []
files = sorted(glob.glob(os.path.join(DIR, '*.xlsx')))
for path in files:
    meta = parse_name(path)
    f, p, bad, dup = load(path)
    a, ca, ra = f3(f, p, True)
    b, cb, rb = f3(f, p, False)
    reached = bool(np.any(p - ra <= -3.0))
    recs.append(dict(**meta, N=len(f), fmin=f[0], fmax=f[-1],
                     dropped=bad + dup, f3_ref0=a, f3_refmin=b,
                     crosses=reached, f=f, p=p, c=ca, ref=ra))

t = pd.DataFrame([{k: v for k, v in r.items() if k not in ('f', 'p', 'c', 'ref')}
                  for r in recs]).sort_values(['D', 'V', 'Rm', 'run'])
pd.set_option('display.width', 240)
print(t.drop(columns=['file']).to_string(index=False,
      float_format=lambda x: f'{x:.2f}'))
amb = t[t.ambiguous != '']
if len(amb):
    print('\nresistance ambiguous in the file name (last value used):')
    for _, r in amb.iterrows():
        print(f"  {r.file}  ->  {r.lab} ohm   (name contains {r.ambiguous})")
t.to_csv('bw_user_f3dB.csv', index=False)

n = len(recs)
cols = min(3, n); rows = int(np.ceil(n/cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.5*cols, 3.6*rows), squeeze=False)
for ax, r in zip(axs.ravel(), recs):
    ax.plot(r['f'], r['p'] - r['ref'], 'o', ms=3.2, mfc='none',
            color='#c0392b', label='measured')
    ff = np.linspace(0, r['f'][-1], 800)
    ax.plot(ff, np.polyval(r['c'], ff) - r['ref'], '-', color='k', lw=1.3,
            label='3rd-order fit')
    ax.axhline(-3, color='0.5', ls='--', lw=1.1)
    if np.isfinite(r['f3_ref0']):
        ax.axvline(r['f3_ref0'], color='#2471a3', ls=':', lw=1.3)
        ax.annotate(f"$f_{{3dB}}$ = {r['f3_ref0']:.2f} GHz",
                    xy=(0.04, 0.06), xycoords='axes fraction', fontsize=8.5,
                    color='#2471a3')
    ax.set_title(f"{r['D']} $\\mu$m, {r['lab']}"
                 + ('' if r['lab'] == 'WO' else ' $\\Omega$')
                 + f", {r['V']} V", fontsize=9.5)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Normalised response (dB)')
    ax.set_ylim(-8, 2); ax.grid(alpha=.3, ls=':')
    ax.legend(fontsize=7.5, loc='upper right')
for ax in axs.ravel()[n:]:
    ax.axis('off')
fig.tight_layout()
fig.savefig('bw_user_responses.png', dpi=200)
print('\nwrote bw_user_f3dB.csv, bw_user_responses.png')
