"""VA -- skeptic's independent re-check of the dimension-A (inventory) findings.

Own reader, own cubic -3 dB extraction, own level / loss-file / model checks.
Nothing here touches the project baseline; every model number is exploratory.
Output: printed sections 1-9 (one per finding) + inv40/VA_inventory.csv
"""
import os, re, glob, hashlib, numpy as np, pandas as pd
os.chdir('/home/user/UTC-PD-Simulation')
exec(open('ft_extraction_multiD.py').read().split('# ── main loop')[0])   # read_s11, H_ckt, f3dB_of, R_L

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

B = 'data_PD0008_1/Bandwidth'; S = 'data_PD0008_1/S11'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]

# ---------------------------------------------------------------- own reader
def hdr(path):
    d = pd.read_excel(path, header=None, nrows=14)
    m = {str(d.iloc[i, 0]).strip(): d.iloc[i, 1] for i in range(len(d))}
    num = lambda s: (float(re.search(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(s)).group(0))
                     if re.search(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(s)) else np.nan)
    return dict(date=str(m.get('DATE')), time=str(m.get('TIME')), V=num(m.get('KEITHLEY VOLTAGE')),
                I0=num(m.get('INITIAL PHOTOCURRENT')), lam3=num(m.get('STARTING WAVELENGTH FOR LASER 3')),
                lam4=num(m.get('STARTING WAVELENGTH FOR LASER 4')), delay=num(m.get('DELAY')),
                tsw=num(m.get('FREQUENCY SWEEP RUN TIME')), s2p=os.path.basename(str(m.get('S2P LOSS FILE'))),
                devno=str(m.get('DEVICE NUMBER')), comm=str(m.get('COMMENTS')))

def table(path):
    d = pd.read_excel(path, header=14)
    g = lambda i: pd.to_numeric(d.iloc[:, i], errors='coerce').values
    return dict(f=g(0), I=g(1), Praw=g(2), Ltot=g(3), Lprobe=g(4), Llink=g(5), Pcal=g(6), VOA=g(7))

def clean(f, p):
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[k], p[k]

def f3_cubic(f, p, fmax=None, order=3, ref_at=0.0):
    if fmax is not None:
        m = f <= fmax + 1e-9; f, p = f[m], p[m]
        if len(f) < 8: return np.nan, 'none'
    c = np.polyfit(f, p, order); ref = np.polyval(c, ref_at)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]; return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), f'poly{order}'
    rel = p - ref; k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]; return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]])), 'raw'
    return np.nan, 'none'

def Rm10(s11):
    f, Sm = read_s11(s11); return float((R_L*(1 + Sm[0])/(1 - Sm[0])).real)

def P_ideal(I_mA, Rm):
    Rp = R_L if not np.isfinite(Rm) else Rm*R_L/(Rm + R_L)
    return 10*np.log10((I_mA*1e-3)**2*Rp**2/(2*R_L)*1e3)

def analyse(path, Rm=np.inf):
    h = hdr(path); t = table(path)
    f, p = clean(t['f'], t['Pcal'])
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    f3, src = f3_cubic(f, p)
    at = lambda x: float(np.interp(x, f, p)) - ref if f[0] <= x <= f[-1] else np.nan
    fin = t['f'][np.isfinite(t['f'])]
    return dict(path=path, md5=md5(path), date=h['date'], time=h['time'], V=h['V'], I0=h['I0'],
                s2p=h['s2p'], lam4=h['lam4'], tsw=h['tsw'], devno=h['devno'], comm=h['comm'],
                N=len(fin), Nc=len(f), fmin=f[0], fmax=f[-1], restarts=int((np.diff(fin) <= 0).sum()),
                Pmin=p[0], Pdc=ref, dB10=at(10), dB15=at(15), dB20=at(20),
                resid=float(np.sqrt(np.mean((p - np.polyval(c, f))**2))),
                f3=f3, src=src, frac=f3/f[-1] if np.isfinite(f3) else np.nan,
                VOA=float(np.nanmean(t['VOA'])), Imean=float(np.nanmean(t['I'])),
                Imin=float(np.nanmin(t['I'])), Imax=float(np.nanmax(t['I'])),
                Rm=Rm, Pid=P_ideal(abs(h['I0']), Rm), off=p[0] - P_ideal(abs(h['I0']), Rm),
                nplus30=int(np.sum(t['Pcal'] >= 0)), link0=bool(np.all(np.nan_to_num(t['Llink']) == 0)),
                calcheck=float(np.nanmax(np.abs(t['Pcal'] - (t['Praw'] + t['Ltot'])))),
                f=f, p=p, c=c, t=t)

# ---------------------------------------------------------------- sheet list
S40 = [  # (tag, label, bias, path, S11 for Rm)
 ('WO_1649', 'WO', -7, f'{B}/40um/Bias_-7V_Iph_1mA.xlsx', None),
 ('WO_1700', 'WO', -7, f'{B}/40um/Bias_-7V_Iph_1mA_diff_pol.xlsx', None),
 ('WO_1711_9V', 'WO', -9, f'{B}/40um/Bias_-9V_Iph_1mA_diff_pol.xlsx', None),
 ('WO_1848_#2', 'WO', -7, f'{B}/40um/WO_bias__7V_Iph_1mA_#2.xlsx', None),
 ('R142', '140', -7, f'{B}/40um/140hom/Bias_-7V_Iph_1mA_120ohm.xlsx', f'{S}/40um/140ohm_V1/S11_-7V_120ohm.s1p'),
 ('R42', '40', -7, f'{B}/40um/40ohm/Bias_-7V_Iph_1mA_36ohm.xlsx', f'{S}/40um/40ohm_V1/S11_-7V_36ohm.s1p'),
 ('R107', '100', -7, f'{B}/40um/100ohm/Bias_-7V_Iph_1mA.xlsx', f'{S}/40um/100ohm_V1/S11_-7V.s1p'),
 ('V2_WO_5V', 'WO', -5, f'{B}/40um/V2/WO/Bias_-5V_Iph_1mA.xlsx', None),
 ('V2_R80_7V', '80', -7, f'{B}/40um/V2/40 ohm/Bias_-7V_Iph_1mA.xlsx', f'{S}/40um/80ohm_V2/S11_-7V.s1p'),
 ('V2_R80_5V', '80', -5, f'{B}/40um/V2/40 ohm/Bias_-5V_Iph_1mA.xlsx', f'{S}/40um/80ohm_V2/S11_-5V.s1p'),
 ('V2_WO_7V_1333', 'WO', -7, f'{B}/40um/V2/WO/Bias_-7V_Iph_1mA.xlsx', None),
 ('15um_HV_WO', 'WO', -7, f'{B}/15um/H.V/WO/Bias_-7V_Iph_1mA.xlsx', None),
 ('35_2208', 'WO', -7, f'{B}/35um/Bias_-7V_Iph_1mA.xlsx', None),
 ('35_diffpm', 'WO', -7, f'{B}/35um/Bias_-7V_Iph_1mA_diff_powermeter.xlsx', None),
 ('35_test', 'WO', -7, f'{B}/35um/test.xlsx', None),
]
M30 = f'{S}/30um/main_figure_03_30_2026'
S30 = [
 ('30Jan_WO_1mA', 'WO', -7, f'{B}/30um/WO/Bias_-7V_Iph_1mA.xlsx', None),
 ('30Jan_WO_6mA', 'WO', -7, f'{B}/30um/WO/Bias_-7V_Iph_6mA.xlsx', None),
 ('30Jan_120', '120', -7, f'{B}/30um/120 ohm/Bias_-7V_Iph_1.034mA.xlsx', f'{S}/30um/120 ohm/-7V.s1p'),
 ('30Jan_76', '76', -7, f'{B}/30um/76 ohm/Bias_-7V_Iph_1mA.xlsx', f'{S}/30um/71 ohm/-7V.s1p'),
 ('30Jan_38', '38', -7, f'{B}/30um/40 ohm/Bias_-7V_Iph_1mA.xlsx', f'{S}/30um/32 ohm/-7V.s1p'),
 ('30Mar_WO_7V', 'WO', -7, f'{B}/30um/Figure_03_27_2026/WO/Bias_-7V_Iph_1mA_30GHz.xlsx', None),
 ('30Mar_WO_5V_30G', 'WO', -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA_30GHz.xlsx', None),
 ('30Mar_WO_5V_19G', 'WO', -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA.xlsx', None),
 ('30Mar_WO_5V_0.5mA', 'WO', -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_0.5mA_30GHz.xlsx', None),
 ('30Mar_WO_5V_3mA', 'WO', -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_3mA_30GHz.xlsx', None),
 ('30Mar_WO_5V_5mA', 'WO', -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_5mA_30GHz_WO.xlsx', None),
 ('30Mar_WO_3V', 'WO', -3, f'{B}/30um/Figure_03_27_2026/WO/Bias_-3V_Iph_1mA.xlsx', None),
 ('30Mar_38_7V', '38', -7, f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-7V_Iph_1mA.xlsx', f'{M30}/33ohm/Bias_-7V_33ohm.s1p'),
 ('30Mar_60_7V', '60', -7, f'{B}/30um/Figure_03_27_2026/55 ohm/Bias_-7V_Iph_1mA.xlsx', f'{M30}/55ohm/Bias_-7V_55ohm.s1p'),
 ('30Mar_200_7V', '200', -7, f'{B}/30um/Figure_03_27_2026/200 ohm/Bias_-7V_Iph_1mA.xlsx', f'{M30}/200 ohm-1/Bias_-7V_200ohm-1.s1p'),
 ('25_7V', 'WO', -7, f'{B}/25 um/Bias_-7V_Iph_1mA_diff_probe_upto_30GHz.xlsx', None),
 ('25_5V', 'WO', -5, f'{B}/25 um/Bias_-5V_Iph_1mA_diff_probe_upto_30GHz.xlsx', None),
]
R = {}
for tag, lab, V, path, s11 in S40 + S30:
    R[tag] = analyse(path, Rm10(s11) if s11 else np.inf); R[tag]['tag'] = tag; R[tag]['lab'] = lab
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('f', 'p', 'c', 't')} for r in R.values()])
df.to_csv('inv40/VA_inventory.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 50)

print('=' * 100); print('SECTION 1: 40 um / 35 um / 15 um sheet inventory (own reader)')
cols = ['tag', 'md5', 'date', 'time', 'V', 'I0', 's2p', 'fmin', 'fmax', 'N', 'Nc', 'restarts', 'Pmin', 'f3', 'src', 'frac',
        'dB10', 'dB15', 'dB20', 'resid', 'tsw', 'VOA', 'lam4', 'nplus30', 'link0', 'calcheck']
print(df[df.tag.isin([s[0] for s in S40])][cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('DEVICE NUMBER / COMMENTS non-blank anywhere?', [(r['tag'], r['devno'], r['comm']) for r in R.values() if r['devno'] != 'nan' or r['comm'] != 'nan'])
# archive count
arch = sorted(glob.glob(f'{B}/40um/**/*.xlsx', recursive=True))
arch_noncal = [a for a in arch if os.path.basename(a) != 'cal.xlsx']
print(f'archive 40um xlsx (excl. cal.xlsx): {len(arch_noncal)} files, {len(set(md5(a) for a in arch_noncal))} distinct md5')
user40 = sorted(glob.glob('data_bw_user/*40um*.xlsx'))
print(f'data_bw_user 40um sheets: {len(user40)}; all have archive twin:',
      all(md5(u) in {md5(a) for a in arch_noncal} for u in user40))
for u in user40:
    twins = [a for a in arch_noncal if md5(a) == md5(u)]
    print(f'   {os.path.basename(u):48s} {md5(u)}  = {[a.replace(B + "/", "") for a in twins]}')

print('\n' + '=' * 100); print('SECTION 2: the 13:33 sweep (40um/V2/WO/-7V == 15um/H.V/WO)')
a, b = R['V2_WO_7V_1333'], R['15um_HV_WO']
print('md5 xlsx', a['md5'], b['md5'], md5('data_bw_user/Bias_7V_Iph_1mA_40um_WO_2.xlsx'))
print('header', a['date'], a['time'], a['s2p'], 'I0', a['I0'], 'VOA', round(a['VOA'], 3), 'lam4', a['lam4'], 'tsw', a['tsw'])
print(f"f3 {a['f3']:.2f} GHz ({a['src']}), span {a['fmin']}-{a['fmax']}, dB@10/15/20 = {a['dB10']:+.2f}/{a['dB15']:+.2f}/{a['dB20']:+.2f}, resid {a['resid']:.2f}")
print('trailing rows:', a['t']['f'][-3:], a['t']['Pcal'][-3:])
# times of everything in 40um/V2
for q in sorted(glob.glob(f'{B}/40um/V2/**/*.xlsx', recursive=True)):
    h = hdr(q); print(f"   {q.replace(B + '/', ''):45s} {h['date']} {h['time']}  V={h['V']}  I0={h['I0']}")
# model: f_RC and f3 for 40 um vs 15 um C_PD with fitted L's from ft_userbw.csv
fu = pd.read_csv('ft_userbw.csv')
r40 = fu[(fu.D == 40) & (fu.V == -7) & (fu.lab == 'WO')].iloc[0]
r30 = fu[(fu.D == 30) & (fu.V == -7) & (fu.lab == 'WO') & (fu.camp == 'Mar')].iloc[0]
fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg; Hp = H_ph(wg)
def model(Cpd_fF, L1, L2, Rm=np.inf, Lm=0.0):
    Hc = H_ckt(wg, Cpd_fF*1e-15, Rm, L1*1e-12, Lm*1e-12, L2*1e-12)
    d = lambda H, x: float(20*np.log10(np.abs(np.interp(x*1e9, fg, np.abs(H)))/np.abs(H[0])))
    return f3dB_of(fg, Hc)/1e9, f3dB_of(fg, Hp*Hc)/1e9, d(Hp*Hc, 10), d(Hp*Hc, 15), d(Hp*Hc, 20)
for name, C, L1, L2 in [('40um C227.6 L40', 227.6, r40.L1, r40.L2), ('15um C39.3 L40', 39.3, r40.L1, r40.L2),
                        ('15um C39.3 L30Mar', 39.3, r30.L1, r30.L2), ('20um C68.3 L40', 68.3, r40.L1, r40.L2),
                        ('25um C100.4 L40', 100.4, r40.L1, r40.L2)]:
    fr, f3m, d10, d15, d20 = model(C, L1, L2)
    print(f'   [exploratory] {name:22s} L1/L2={L1:.1f}/{L2:.1f} pH: f_RC={fr:.1f}  f3(Hph*Hckt)={f3m:.1f}  dB@10/15/20={d10:+.2f}/{d15:+.2f}/{d20:+.2f}')
# pure C_PD*R_L bound (no L, no Rs): the fastest any 40 um -7V RC can be
print(f'   RC bound with C_PD=227.6 fF into 50 ohm only: {1/(2*np.pi*50*227.6e-15)/1e9:.1f} GHz;  with Rs+R_L=58.92: {1/(2*np.pi*58.92*227.6e-15)/1e9:.1f} GHz')
# 30um 'new cal' WO misfiles
for q in [f'{B}/30um/Figure_03_27_2026/WO/new cal/Bias_-3V_Iph_1mA.xlsx', f'{B}/30um/Figure_03_27_2026/WO/new cal/Bias_-5V_Iph_1mA.xlsx',
          f'{B}/30um/Figure_03_27_2026/36 ohm/new cal/Bias_-3V_Iph_1mA.xlsx', f'{B}/30um/Figure_03_27_2026/36 ohm/new cal/Bias_-5V_Iph_1mA.xlsx',
          f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-3V_Iph_1mA.xlsx', f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-5V_Iph_1mA.xlsx',
          f'{B}/30um/Figure_03_27_2026/WO/Bias_-3V_Iph_1mA.xlsx', f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA.xlsx']:
    h = hdr(q); print(f"   {q.replace(B + '/', ''):60s} md5 {md5(q)} {h['date']} {h['time']} I0={h['I0']} s2p={h['s2p']}")

print('\n' + '=' * 100); print('SECTION 3: duplicate 40 um open runs at -7 V')
dup = ['WO_1649', 'WO_1700', 'WO_1848_#2']
for k in dup + ['WO_1711_9V', 'V2_WO_5V']:
    r = R[k]
    alt = {o: f3_cubic(r['f'], r['p'], order=o)[0] for o in (2, 3, 4, 5)}
    le148 = f3_cubic(r['f'], r['p'], fmax=14.8)[0]
    le12 = f3_cubic(r['f'], r['p'], fmax=12.0)[0]
    refmin = f3_cubic(r['f'], r['p'], ref_at=r['f'][0])[0]
    # single-pole fit |H|^2 = 1/(1+(f/fc)^2) in dB with free level
    from scipy.optimize import least_squares
    sp = least_squares(lambda x: (x[0] - 10*np.log10(1 + (r['f']/x[1])**2)) - r['p'], [r['p'][0], 12.0]).x
    print(f"   {k:12s} {r['time']} I0={r['I0']:.3f} VOA={r['VOA']:.3f} lam4={r['lam4']} span {r['fmin']:.2f}-{r['fmax']:.2f} N={r['Nc']} P(fmin)={r['Pmin']:.2f} "
          f"f3={r['f3']:.2f}({r['src']}) frac={r['frac']:.2f} | poly2/3/4/5 {alt[2]:.2f}/{alt[3]:.2f}/{alt[4]:.2f}/{alt[5]:.2f} | fit<=14.8 {le148:.2f} <=12 {le12:.2f} | refmin {refmin:.2f} | 1-pole fc {sp[1]:.2f} | dB@10 {r['dB10']:.2f} dB@15 {r['dB15']:.2f} resid {r['resid']:.2f}")
v = np.array([R[k]['f3'] for k in dup]); print(f'   f3 mean {v.mean():.2f} sd {v.std(ddof=1):.2f} ({v.std(ddof=1)/v.mean()*100:.0f}%) range {v.min():.2f}-{v.max():.2f}')
grid = np.arange(2.0, 14.81, 0.5)
P = np.array([np.interp(grid, R[k]['f'], R[k]['p']) for k in dup])
print(f'   raw dBm on common 2-14.8 grid: mean|dev| {np.mean(np.abs(P - P.mean(0))):.2f} dB, max {np.max(np.abs(P - P.mean(0))):.2f} dB')
print('   pairwise mean diff (dB) 1649-1700 %.2f, 1649-1848 %.2f, 1700-1848 %.2f' % ((P[0]-P[1]).mean(), (P[0]-P[2]).mean(), (P[1]-P[2]).mean()))
print('   diff 1848 - 1649 per grid point (dB):', np.round(P[2] - P[0], 2))
# implied f_T with project f_RC
fRC7 = float(r40.f_RC)
for k in dup:
    d = 1/R[k]['f3']**2 - 1/fRC7**2
    print(f"   implied f_T {k}: f3={R[k]['f3']:.2f} f_RC={fRC7:.3f} -> {np.sqrt(1/d) if d > 0 else np.inf:.1f} GHz")
# in-sweep repeatability for resumed sweeps
for k in ['V2_WO_5V', '35_diffpm']:
    t = R[k]['t']; f, p = t['f'], t['Pcal']; m = np.isfinite(f); f, p = f[m], p[m]
    kk = np.where(np.diff(f) <= 0)[0]
    if len(kk):
        j = kk[0] + 1; f1, p1, f2, p2 = f[:j], p[:j], f[j:], p[j:]
        lo, hi = max(f1.min(), f2.min()), min(f1.max(), f2.max()); g = np.linspace(lo, hi, 20)
        dd = np.interp(g, f1, p1) - np.interp(g, f2, p2)
        print(f'   resumed {k}: restart at row {j} (f={f[j]:.2f}), overlap {lo:.2f}-{hi:.2f}: mean {dd.mean():+.2f} rms {np.sqrt(np.mean(dd**2)):.2f} dB; exact-f duplicates: {len(set(f1) & set(f2))}')
        # exact duplicate frequencies
        com = sorted(set(np.round(f1, 2)) & set(np.round(f2, 2)))
        dif = [float(p1[np.round(f1, 2) == x][0] - p2[np.round(f2, 2) == x][0]) for x in com]
        print('      exact-f pass1-pass2:', np.round(dif, 2))
# probing order 02/04
print('   02/04 sweep order:')
allx = sorted(glob.glob(f'{B}/**/*.xlsx', recursive=True))
rows = []
for q in allx:
    try: h = hdr(q); t = table(q)
    except Exception: continue
    rows.append(dict(date=h['date'], time=h['time'], V=h['V'], I0=h['I0'], s2p=h['s2p'], lam4=h['lam4'], VOA=np.nanmean(t['VOA']),
                     fmax=np.nanmax(t['f']), N=int(np.isfinite(t['f']).sum()), path=q.replace(B + '/', ''), md5=md5(q),
                     praw0=t['Praw'][0] if len(t['Praw']) else np.nan))
ses = pd.DataFrame(rows).sort_values(['date', 'time'])
print(ses[ses.date == '02/04/2026'][['time', 'V', 'I0', 'VOA', 'fmax', 'N', 'path']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
cv = np.loadtxt('data_CV/CV_40um.txt', skiprows=3)
print(f'   CV_40um last rows: V={cv[-1,0]} C={cv[-1,1]*1e15:.1f} fF; V={cv[-2,0]} C={cv[-2,1]*1e15:.1f}; dC/0.2V at end = {(cv[-2,1]-cv[-1,1])*1e15:.1f} fF')

print('\n' + '=' * 100); print('SECTION 4: absolute levels')
print(f'   ideal 1 mA open into 50: {P_ideal(1.0, np.inf):.2f} dBm')
wo = df[(df.lab == 'WO') & ~df.tag.isin(['V2_WO_7V_1333', '15um_HV_WO', '35_test'])]
off = float(np.median(wo.off))
print(f'   open-device sheets used for median: {len(wo)}; median offset {off:+.3f} dB; mean {wo.off.mean():+.3f}; range {wo.off.min():+.2f}..{wo.off.max():+.2f}')
print(wo[['tag', 'I0', 'Pmin', 'Pid', 'off']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
df['Iimp'] = df.I0.abs()*10**((df.off - off)/20)
print('   resistor / all 40um sheets, implied I_ph after median offset:')
print(df[df.tag.isin([s[0] for s in S40]) | df.tag.str.startswith('30')][['tag', 'Rm', 'I0', 'Pmin', 'Pid', 'off', 'Iimp']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
# sensitivity: use a different DC-level estimator (cubic DC ref instead of P(fmin))
df['off_dc'] = df.Pdc - df.Pid
print('   offset using cubic DC ref instead of P(fmin): 40um WO ', np.round(df[df.tag.str.startswith('WO') | (df.tag == 'V2_WO_5V')].off_dc.values, 2))

print('\n' + '=' * 100); print('SECTION 5: I_PD column vs R_m')
for r in df[np.isfinite(df.Rm)].itertuples():
    Rr = abs(r.V)/(abs(r.I0) - r.Iimp)*1e3
    Rr_lab = abs(r.V)/(abs(r.I0) - 1.0)*1e3
    print(f'   {r.tag:14s} V={r.V:+.0f} I0={r.I0:9.3f} mA  Rm(S11)={r.Rm:7.2f}  R(IPD, I_imp {r.Iimp:.3f})={Rr:7.2f} ratio {Rr/r.Rm:.3f}   R(IPD, 1.0 mA)={Rr_lab:7.2f} ratio {Rr_lab/r.Rm:.3f}  I drift {r.Imin:.3f}..{r.Imax:.3f}')

print('\n' + '=' * 100); print('SECTION 6: loss files')
# collect (f, Ltot) per s2p name from every sheet, check consistency across dates
LC = {}
for q in allx:
    try: h = hdr(q); t = table(q)
    except Exception: continue
    m = np.isfinite(t['f']) & np.isfinite(t['Ltot'])
    for ff, ll in zip(t['f'][m], t['Ltot'][m]):
        LC.setdefault(h['s2p'], {}).setdefault(round(float(ff), 2), []).append((ll, h['date']))
for name, dct in LC.items():
    spread = [(k, np.ptp([x[0] for x in v]), sorted(set(x[1] for x in v))) for k, v in dct.items() if len(v) > 1]
    print(f'   {name}: {len(dct)} distinct f, max spread over sheets at same f = {max(s[1] for s in spread):.2f} dB; dates {sorted(set(d for v in dct.values() for _, d in v))}')
    bad = [s for s in spread if s[1] > 0.02]
    print('      f with spread >0.02 dB:', [(s[0], round(s[1], 2)) for s in bad][:20])
def curve(name):
    ks = sorted(LC[name]); return np.array(ks), np.array([np.mean([x[0] for x in LC[name][k]]) for k in ks])
fA, LA = curve('Cal.s2p'); fB_, LB = curve('S21_cal_bias_tee_cal.s2p')
for g in [1, 5, 10, 11, 11.5, 12, 15, 20, 20.5, 21, 21.5, 25, 30]:
    print(f'   {g:5.1f} GHz: Cal {np.interp(g, fA, LA):.2f}  S21 {np.interp(g, fB_, LB):.2f}  diff {np.interp(g, fB_, LB)-np.interp(g, fA, LA):+.2f}')
print('   S21 curve, 10-13 GHz:', [(float(x), round(float(y), 2)) for x, y in zip(fB_, LB) if 10 <= x <= 13])
print('   S21 curve, 19-23 GHz:', [(float(x), round(float(y), 2)) for x, y in zip(fB_, LB) if 19 <= x <= 23])
print('   Cal curve, 19-23 GHz:', [(float(x), round(float(y), 2)) for x, y in zip(fA, LA) if 19 <= x <= 23])
def swap(k):
    r = R[k]; t = r['t']; fo, Lo = (fB_, LB) if r['s2p'] == 'Cal.s2p' else (fA, LA)
    f, p = clean(t['f'], t['Praw'] + np.interp(t['f'], fo, Lo)); return f3_cubic(f, p)[0]
for k in ['WO_1649', 'WO_1700', 'WO_1848_#2', 'R142', 'R42', 'R107', 'V2_R80_7V', 'V2_WO_5V', '30Mar_WO_7V', '30Mar_WO_5V_30G', '30Mar_200_7V', '30Jan_WO_1mA', '30Mar_38_7V']:
    print(f"   swap {k:16s} {R[k]['s2p']:28s} f3 {R[k]['f3']:.2f} -> {swap(k):.2f}")

print('\n' + '=' * 100); print('SECTION 7: spans, steepness, f3 vs R_m')
for k in [s[0] for s in S40] + [s[0] for s in S30]:
    r = R[k]
    oct_ = r['dB20'] - r['dB10'] if np.isfinite(r['dB20']) else np.nan
    print(f"   {k:18s} span {r['fmin']:.2f}-{r['fmax']:.2f} N={r['Nc']} f3={r['f3']:.2f} frac={r['frac']:.2f} dB@10={r['dB10']:+.2f} dB@15={r['dB15']:+.2f} dB@20={r['dB20']:+.2f} 10->20 {oct_:+.2f} dB/oct resid={r['resid']:.2f} tsw={r['tsw']}")
# f3 vs Rm
print('   40um -7V f3 vs Rm:', [(round(R[k]['Rm'], 1), round(R[k]['f3'], 2)) for k in ['R42', 'V2_R80_7V', 'R107', 'R142', 'WO_1649']])
print('   30um -7V f3 vs Rm:', [(round(R[k]['Rm'], 1), round(R[k]['f3'], 2)) for k in ['30Jan_38', '30Mar_38_7V', '30Mar_60_7V', '30Jan_76', '30Jan_120', '30Mar_200_7V', '30Jan_WO_1mA', '30Mar_WO_7V']])
print(f"   40um 42.5->142.5 (x{142.5/42.5:.1f}): {R['R42']['f3']:.2f}->{R['R142']['f3']:.2f} = {(R['R142']['f3']/R['R42']['f3']-1)*100:+.0f}%")
print(f"   30um 38->115 (x{115/37.9:.1f}, Jan): {R['30Jan_38']['f3']:.2f}->{R['30Jan_120']['f3']:.2f} = {(R['30Jan_120']['f3']/R['30Jan_38']['f3']-1)*100:+.0f}%")
print(f"   30um 38->205 (x{205.5/38.15:.1f}, Mar): {R['30Mar_38_7V']['f3']:.2f}->{R['30Mar_200_7V']['f3']:.2f} = {(R['30Mar_200_7V']['f3']/R['30Mar_38_7V']['f3']-1)*100:+.0f}%")
# model prediction for the same ratio (f_RC only, and Hph*Hckt), from ft_userbw.csv
for D in (30, 40):
    q = fu[(fu.D == D) & (fu.V == -7)].sort_values('Rm')
    print(f'   {D}um -7V project f_RC vs Rm:', [(round(x, 1), round(y, 1)) for x, y in zip(q.Rm, q.f_RC)])

print('\n' + '=' * 100); print('SECTION 8: 35 um')
for k in ['35_2208', '35_diffpm', '35_test']:
    r = R[k]; print(f"   {k:10s} {r['date']} {r['time']} I0={r['I0']} span {r['fmin']}-{r['fmax']} N={r['Nc']} f3={r['f3']:.2f}({r['src']}) frac={r['frac']:.2f} dB@10/15/20 {r['dB10']:+.2f}/{r['dB15']:+.2f}/{r['dB20']:+.2f} off={r['off']:+.2f} fit<=14.8 {f3_cubic(r['f'], r['p'], fmax=14.8)[0]:.2f}")
    print('      rel-DC response (f, dB):', [(float(x), round(float(y - r['Pdc']), 2)) for x, y in zip(r['f'], r['p']) if x >= 12])
for name, C, L1, L2 in [('35um C184.8 L40', 184.8, r40.L1, r40.L2), ('35um C184.8 L30Mar', 184.8, r30.L1, r30.L2), ('40um C227.6 L40', 227.6, r40.L1, r40.L2), ('30um C133.5 L30Mar', 133.5, r30.L1, r30.L2)]:
    fr, f3m, d10, d15, d20 = model(C, L1, L2)
    print(f'   [exploratory] {name:20s} f_RC={fr:.1f} f3(Hph*Hckt)={f3m:.1f} model dB@10/15/20={d10:+.2f}/{d15:+.2f}/{d20:+.2f}')
    for k in ['35_2208', '35_diffpm']:
        d = 1/R[k]['f3']**2 - 1/fr**2
        print(f'        implied f_T from {k} f3={R[k]["f3"]:.2f}: {np.sqrt(1/d) if d > 0 else np.inf:.1f}')
print('   measured 30um Mar WO -7V dB@10/15/20:', round(R['30Mar_WO_7V']['dB10'], 2), round(R['30Mar_WO_7V']['dB15'], 2), round(R['30Mar_WO_7V']['dB20'], 2))
print('   measured 40um WO 16:49 dB@10/15/20:', round(R['WO_1649']['dB10'], 2), round(R['WO_1649']['dB15'], 2), round(R['WO_1649']['dB20'], 2))

print('\n' + '=' * 100); print('SECTION 9: sessions')
ses['mAmW'] = ses.I0.abs()/(10**(ses.VOA/10))
ses['fold'] = [p.split('/')[0] for p in ses.path]
ses['newcal'] = ses.path.str.contains('new cal')
for d, g in ses.groupby('date'):
    g2 = g[g.N > 3]
    print(f"   {d}: {len(g2)} sweep files ({g2.md5.nunique()} distinct md5, {len(g2[~g2.newcal])} excl 'new cal'), {g2.time.min()}-{g2.time.max()}, loss {sorted(g2.s2p.unique())}, "
          f"lam4 {g2.lam4.min():.3f}-{g2.lam4.max():.3f}, VOA {g2.VOA.min():.2f}-{g2.VOA.max():.2f}, fmax {g2.fmax.min():.1f}-{g2.fmax.max():.1f}, folders {sorted(set(g2.fold))}")
    g1 = g2[(g2.I0.abs() > 0.8) & (g2.I0.abs() < 1.3)]
    if len(g1): print(f"      ~1 mA sheets: VOA {g1.VOA.min():.2f}-{g1.VOA.max():.2f} dBm, mA/mW {g1.mAmW.min():.3f}-{g1.mAmW.max():.3f}")
# 'new cal' = same raw data?  check Praw identity with the non-newcal sibling
print("   'new cal' sheets: same raw sweep as the sibling?")
for q in [p for p in ses.path if 'new cal' in p]:
    sib = q.replace('new cal/', '').replace('0_5mA', '0.5mA')
    cands = [c for c in ses.path if c.startswith(os.path.dirname(sib)) and 'new cal' not in c and hdr(f'{B}/{c}')['time'] == hdr(f'{B}/{q}')['time']]
    print(f'      {q:70s} time {hdr(f"{B}/{q}")["time"]} s2p {hdr(f"{B}/{q}")["s2p"]:28s} same-time sibling: {cands}')
# mA/mW per diameter (~1 mA sheets)
for lab_, sel in [('40um 02/04', ses[(ses.fold == '40um') & (ses.date == '02/04/2026')]), ('40um 02/25', ses[(ses.fold == '40um') & (ses.date == '02/25/2026')]),
                  ('35um', ses[ses.fold == '35um']), ('15um', ses[ses.fold == '15um']), ('25um', ses[ses.fold == '25 um']),
                  ('30um Jan', ses[(ses.fold == '30um') & (ses.date == '01/21/2026')]), ('30um Mar27', ses[(ses.fold == '30um') & (ses.date == '03/27/2026')]), ('30um Mar29', ses[(ses.fold == '30um') & (ses.date == '03/29/2026')])]:
    s1 = sel[(sel.I0.abs() > 0.8) & (sel.I0.abs() < 1.3) & (sel.N > 3)]
    if len(s1): print(f'   {lab_:12s} ~1mA sheets N={len(s1)}: mA/mW {s1.mAmW.min():.3f}-{s1.mAmW.max():.3f}  VOA {s1.VOA.min():.2f}-{s1.VOA.max():.2f}')
# cal.xlsx single points
for q in sorted(glob.glob(f'{B}/**/cal.xlsx', recursive=True)):
    h = hdr(q); t = table(q); print(f"   {q.replace(B + '/', ''):45s} {h['date']} {h['time']} V={h['V']} I0={h['I0']} Praw={t['Praw'][0]} VOA={t['VOA'][0]}")
weak = min(R[k]['p'].min() for k in [s[0] for s in S40[:10]]); print(f'   weakest 40um point {weak:.2f} dBm; margin over -53.95: {weak + 53.95:.1f} dB -> {10*np.log10(1 + 10**(-(weak + 53.95)/10)):.3f} dB; over -48.53: {weak + 48.53:.1f} dB -> {10*np.log10(1 + 10**(-(weak + 48.53)/10)):.3f} dB')
