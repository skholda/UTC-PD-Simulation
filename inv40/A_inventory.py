"""DIMENSION A -- inventory and internal consistency of every 40 um bandwidth
measurement (plus 35 um / 15 um neighbours and the 30 / 25 um sheets needed
for the absolute-level comparison).

Nothing here changes the project baseline. All refits are exploratory and
labelled as such in the printed output and in the CSV column names (exp_*).

Outputs (all under inv40/):
  A_inventory.csv        one row per xlsx (both archive locations)
  A_overlay_WO.png       every 40 um open-device run, raw dBm and normalised
  A_overlay_R.png        40 um resistor devices vs the 30 um resistor devices
  A_levels.png           absolute low-frequency level vs the ideal I_ph^2 R/2
  A_losscal.png          the two RF-loss correction files actually applied
"""
import os, re, glob, hashlib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/home/user/UTC-PD-Simulation')
exec(open('ft_extraction_multiD.py').read().split('# ── main loop')[0])   # read_s11, H_ckt, ...

# H_ph exactly as in ft_userbw_plots.py (transit-time baseline, unchanged)
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

B = 'data_PD0008_1/Bandwidth'
S = 'data_PD0008_1/S11'
M30 = f'{S}/30um/main_figure_03_30_2026'

# ── every sheet of interest: (group, D, label, bias, xlsx, S11 for R_m) ─────
# group: '40V1' 02/04 session, '40V2' 02/25 session, '35', '15dup', '30Jan',
#        '30Mar', '25'. S11 = None for open devices (R_m = inf) or unknown.
SHEETS = [
 # ---- 40 um, archive location -------------------------------------------
 ('40V1', 40, 'WO',  -7, f'{B}/40um/Bias_-7V_Iph_1mA.xlsx',            None),
 ('40V1', 40, 'WO',  -7, f'{B}/40um/Bias_-7V_Iph_1mA_diff_pol.xlsx',   None),
 ('40V1', 40, 'WO',  -9, f'{B}/40um/Bias_-9V_Iph_1mA_diff_pol.xlsx',   None),
 ('40V1', 40, 'WO',  -7, f'{B}/40um/WO_bias__7V_Iph_1mA_#2.xlsx',      None),
 ('40V1', 40, '140', -7, f'{B}/40um/140hom/Bias_-7V_Iph_1mA_120ohm.xlsx',
                         f'{S}/40um/140ohm_V1/S11_-7V_120ohm.s1p'),
 ('40V1', 40, '40',  -7, f'{B}/40um/40ohm/Bias_-7V_Iph_1mA_36ohm.xlsx',
                         f'{S}/40um/40ohm_V1/S11_-7V_36ohm.s1p'),
 ('40V1', 40, '100', -7, f'{B}/40um/100ohm/Bias_-7V_Iph_1mA.xlsx',
                         f'{S}/40um/100ohm_V1/S11_-7V.s1p'),
 ('40V2', 40, 'WO',  -5, f'{B}/40um/V2/WO/Bias_-5V_Iph_1mA.xlsx',       None),
 ('40V2', 40, '80',  -7, f'{B}/40um/V2/40 ohm/Bias_-7V_Iph_1mA.xlsx',
                         f'{S}/40um/80ohm_V2/S11_-7V.s1p'),
 ('40V2', 40, '80',  -5, f'{B}/40um/V2/40 ohm/Bias_-5V_Iph_1mA.xlsx',
                         f'{S}/40um/80ohm_V2/S11_-5V.s1p'),
 # the sheet filed as 40um/V2/WO/-7V: dated 02/04 13:33, identical to 15um/H.V
 ('15dup', 40, 'WO', -7, f'{B}/40um/V2/WO/Bias_-7V_Iph_1mA.xlsx',       None),
 ('15dup', 15, 'WO', -7, f'{B}/15um/H.V/WO/Bias_-7V_Iph_1mA.xlsx',      None),
 # ---- 40 um, data_bw_user copies ----------------------------------------
 ('40V1', 40, 'WO',  -7, 'data_bw_user/Bias_7V_Iph_1mA_40um_WO_3.xlsx',   None),
 ('15dup',40, 'WO',  -7, 'data_bw_user/Bias_7V_Iph_1mA_40um_WO_2.xlsx',   None),
 ('40V2', 40, 'WO',  -5, 'data_bw_user/Bias_5V_Iph_1mA_40um_WO_2.xlsx',   None),
 ('40V1', 40, '100', -7, 'data_bw_user/Bias_7V_Iph_1mA_40um_100ohm_1.xlsx',
                         f'{S}/40um/100ohm_V1/S11_-7V.s1p'),
 ('40V1', 40, '40',  -7, 'data_bw_user/Bias_7V_Iph_1mA_40um_38ohm_1.xlsx',
                         f'{S}/40um/40ohm_V1/S11_-7V_36ohm.s1p'),
 ('40V1', 40, '140', -7, 'data_bw_user/Bias_7V_Iph_1mA_120ohm_40um_140ohm_1.xlsx',
                         f'{S}/40um/140ohm_V1/S11_-7V_120ohm.s1p'),
 ('40V2', 40, '80',  -7, 'data_bw_user/Bias_7V_Iph_1mA_40um_38ohm_2.xlsx',
                         f'{S}/40um/80ohm_V2/S11_-7V.s1p'),
 ('40V2', 40, '80',  -5, 'data_bw_user/Bias_5V_Iph_1mA_40um_38ohm_2.xlsx',
                         f'{S}/40um/80ohm_V2/S11_-5V.s1p'),
 # ---- 35 um neighbours (02/03-02/04, same rig, same loss file) ----------
 ('35', 35, 'WO', -7, f'{B}/35um/Bias_-7V_Iph_1mA.xlsx',                 None),
 ('35', 35, 'WO', -7, f'{B}/35um/Bias_-7V_Iph_1mA_diff_powermeter.xlsx', None),
 ('35', 35, 'WO', -7, f'{B}/35um/test.xlsx',                            None),
 # ---- 30 um and 25 um, for the level comparison --------------------------
 ('30Jan', 30, 'WO',  -7, f'{B}/30um/WO/Bias_-7V_Iph_1mA.xlsx',         f'{S}/30um/WO/-7V.s1p'),
 ('30Jan', 30, 'WO',  -7, f'{B}/30um/WO/Bias_-7V_Iph_6mA.xlsx',         f'{S}/30um/WO/-7V.s1p'),
 ('30Jan', 30, '120', -7, f'{B}/30um/120 ohm/Bias_-7V_Iph_1.034mA.xlsx', f'{S}/30um/120 ohm/-7V.s1p'),
 ('30Jan', 30, '120', -7, f'{B}/30um/120 ohm/Bias_-7V_Iph_6mA.xlsx',    f'{S}/30um/120 ohm/-7V.s1p'),
 ('30Jan', 30, '76',  -7, f'{B}/30um/76 ohm/Bias_-7V_Iph_1mA.xlsx',     f'{S}/30um/71 ohm/-7V.s1p'),
 ('30Jan', 30, '76',  -7, f'{B}/30um/76 ohm/Bias_-7V_Iph_6mA.xlsx',     f'{S}/30um/71 ohm/-7V.s1p'),
 ('30Jan', 30, '38',  -7, f'{B}/30um/40 ohm/Bias_-7V_Iph_1mA.xlsx',     f'{S}/30um/32 ohm/-7V.s1p'),
 ('30Jan', 30, '38',  -7, f'{B}/30um/40 ohm/Bias_-7V_Iph_6mA.xlsx',     f'{S}/30um/32 ohm/-7V.s1p'),
 ('30Jan', 30, '60',  -7, f'{B}/30um/60 ohm/Bias_-7V_Iph_1mA.xlsx',     'NOMINAL60'),
 ('30Mar', 30, 'WO',  -7, f'{B}/30um/Figure_03_27_2026/WO/Bias_-7V_Iph_1mA_30GHz.xlsx',   None),
 ('30Mar', 30, 'WO',  -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA.xlsx',         None),
 ('30Mar', 30, 'WO',  -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_1mA_30GHz.xlsx',   None),
 ('30Mar', 30, 'WO',  -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_0.5mA_30GHz.xlsx', None),
 ('30Mar', 30, 'WO',  -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_3mA_30GHz.xlsx',   None),
 ('30Mar', 30, 'WO',  -5, f'{B}/30um/Figure_03_27_2026/WO/Bias_-5V_Iph_5mA_30GHz_WO.xlsx',None),
 ('30Mar', 30, 'WO',  -3, f'{B}/30um/Figure_03_27_2026/WO/Bias_-3V_Iph_1mA.xlsx',         None),
 ('30Mar', 30, '38',  -7, f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-7V_Iph_1mA.xlsx',  f'{M30}/33ohm/Bias_-7V_33ohm.s1p'),
 ('30Mar', 30, '38',  -5, f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-5V_Iph_1mA.xlsx',  f'{M30}/33ohm/Bias_-5V_38ohm.s1p'),
 ('30Mar', 30, '38',  -3, f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-3V_Iph_1mA.xlsx',  f'{M30}/33ohm/Bias_-3V_33ohm.s1p'),
 ('30Mar', 30, '60',  -7, f'{B}/30um/Figure_03_27_2026/55 ohm/Bias_-7V_Iph_1mA.xlsx',  f'{M30}/55ohm/Bias_-7V_55ohm.s1p'),
 ('30Mar', 30, '60',  -5, f'{B}/30um/Figure_03_27_2026/55 ohm/Bias_-5V_Iph_1mA.xlsx',  f'{M30}/55ohm/Bias_-5V_60ohm.s1p'),
 ('30Mar', 30, '60',  -3, f'{B}/30um/Figure_03_27_2026/55 ohm/Bias_-3V_Iph_1mA.xlsx',  f'{M30}/55ohm/Bias_-3V_55ohm.s1p'),
 ('30Mar', 30, '200', -7, f'{B}/30um/Figure_03_27_2026/200 ohm/Bias_-7V_Iph_1mA.xlsx', f'{M30}/200 ohm-1/Bias_-7V_200ohm-1.s1p'),
 ('30Mar', 30, '200', -5, f'{B}/30um/Figure_03_27_2026/200 ohm/Bias_-5V_Iph_1mA.xlsx', f'{M30}/200 ohm-1/Bias_-5V_200ohm-1.s1p'),
 ('30Mar', 30, '200', -3, f'{B}/30um/Figure_03_27_2026/200 ohm/Bias_-3V_Iph_1mA.xlsx', f'{M30}/200 ohm-1/Bias_-3V_200ohm-1.s1p'),
 ('25',    25, 'WO',  -7, f'{B}/25 um/Bias_-7V_Iph_1mA_diff_probe_upto_30GHz.xlsx', None),
 ('25',    25, 'WO',  -5, f'{B}/25 um/Bias_-5V_Iph_1mA_diff_probe_upto_30GHz.xlsx', None),
]

# ── helpers ────────────────────────────────────────────────────────────────
def meta(path):
    df = pd.read_excel(path, header=None, nrows=14)
    m = {}
    for i in range(len(df)):
        m[str(df.iloc[i, 0]).strip()] = df.iloc[i, 1]
    def num(s):
        r = re.search(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(s))
        return float(r.group(0)) if r else np.nan
    return dict(date=str(m.get('DATE')), time=str(m.get('TIME')),
                V_hdr=num(m.get('KEITHLEY VOLTAGE')),
                Iph0=num(m.get('INITIAL PHOTOCURRENT')),
                lam3=num(m.get('STARTING WAVELENGTH FOR LASER 3')),
                lam4=num(m.get('STARTING WAVELENGTH FOR LASER 4')),
                delay=num(m.get('DELAY')),
                t_sweep=num(m.get('FREQUENCY SWEEP RUN TIME')),
                s2p=os.path.basename(str(m.get('S2P LOSS FILE'))))

def raw_table(path):
    d = pd.read_excel(path, header=14)
    g = lambda i: pd.to_numeric(d.iloc[:, i], errors='coerce').values
    return dict(f=g(0), I=g(1), Praw=g(2), Ltot=g(3), Pcal=g(6), VOA=g(7))

def clean(f, p):
    m = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p < 0)
    f, p = f[m], p[m]
    o = np.argsort(f, kind='stable'); f, p = f[o], p[o]
    k = np.concatenate([[True], np.diff(f) > 1e-6])
    return f[k], p[k], int((~m).sum()), int((~k).sum())

def f3_cubic(f, p, fmax=None):
    """project method: cubic, DC reference at f=0, raw-crossing fallback"""
    if fmax is not None:
        m = f <= fmax + 1e-9
        if m.sum() < 8:
            return np.nan, 'none', np.nan
        f, p = f[m], p[m]
    c = np.polyfit(f, p, 3); ref = np.polyval(c, 0.0)
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]
        return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]])), 'poly3', ref
    rel = p - ref
    k = np.where(rel <= -3.0)[0]
    if len(k) and k[0] > 0:
        j = k[0]
        return float(np.interp(-3.0, [rel[j], rel[j-1]], [f[j], f[j-1]])), 'raw', ref
    return np.nan, 'none', ref

def Rm_of(s11):
    if s11 is None:
        return np.inf
    if s11 == 'NOMINAL60':
        return 60.0
    f, Sm = read_s11(s11)
    return float((R_L*(1 + Sm[0])/(1 - Sm[0])).real)

def P_ideal_dBm(Iph_mA, Rm):
    """heterodyne, 100 % modulation depth: I_ac amplitude = I_dc.
    V_L = I_ph (R_m || R_L); P = V_L^2 / (2 R_L)"""
    Rp = R_L if not np.isfinite(Rm) else Rm*R_L/(Rm + R_L)
    return 10*np.log10((Iph_mA*1e-3)**2*Rp**2/(2*R_L)*1e3)

md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]

# ── build the inventory ────────────────────────────────────────────────────
rows, data = [], {}
for grp, D, lab, V, path, s11 in SHEETS:
    if not os.path.exists(path):
        print('MISSING', path); continue
    mt = meta(path); rt = raw_table(path)
    f, p, ndrop, ndup = clean(rt['f'], rt['Pcal'])
    f3, src, ref = f3_cubic(f, p)
    c = np.polyfit(f, p, 3)
    resid = p - np.polyval(c, f)
    # normalised response at fixed frequencies (linear interp on the cleaned data)
    at = lambda x: float(np.interp(x, f, p)) if f.min() <= x <= f.max() else np.nan
    Rm = Rm_of(s11)
    Iph_lab = abs(mt['Iph0'])
    Pid = P_ideal_dBm(Iph_lab, Rm)
    Ifile = re.search(r'Iph_(\d+(?:[._]\d+)?)mA', os.path.basename(path))
    I_name = float(Ifile.group(1).replace('_', '.')) if Ifile else np.nan
    key = (grp, D, lab, V, mt['date'], mt['time'])
    rec = dict(group=grp, D_um=D, label=lab, bias_V=V, file=path, md5=md5(path),
               date=mt['date'], time=mt['time'], V_header=mt['V_hdr'],
               Iph_initial_mA=mt['Iph0'], Iph_in_name_mA=I_name,
               s2p_loss_file=mt['s2p'], lam3_nm=mt['lam3'], lam4_nm=mt['lam4'],
               t_sweep_s=mt['t_sweep'],
               N_rows=len(rt['f']), N_clean=len(f), rows_dropped=ndrop, rows_dedup=ndup,
               f_min_GHz=f[0], f_max_GHz=f[-1],
               nonmono_restart=int((np.diff(rt['f'][np.isfinite(rt['f'])]) <= 0).sum()),
               P_at_fmin_dBm=p[0], P_dc_cubic_dBm=ref,
               P_10GHz_dBm=at(10.0), P_15GHz_dBm=at(15.0), P_20GHz_dBm=at(20.0),
               dB_10GHz_rel_dc=at(10.0) - ref, dB_15GHz_rel_dc=at(15.0) - ref,
               dB_20GHz_rel_dc=at(20.0) - ref,
               cubic_resid_rms_dB=float(np.sqrt(np.mean(resid**2))),
               f3dB_cubic_GHz=f3, f3dB_source=src,
               f3dB_over_fmax=f3/f[-1] if np.isfinite(f3) else np.nan,
               IPD_mean_mA=float(np.nanmean(rt['I'])), IPD_min_mA=float(np.nanmin(rt['I'])),
               IPD_max_mA=float(np.nanmax(rt['I'])),
               VOA_mean_dBm=float(np.nanmean(rt['VOA'])),
               Rm_S11_10MHz_ohm=Rm, S11_file=s11 if s11 else '',
               P_ideal_at_Iph_dBm=Pid, level_minus_ideal_dB=p[0] - Pid,
               Iph_implied_ideal_mA=Iph_lab*10**((p[0] - Pid)/20),
               R_from_IPD_ohm=(abs(V)/(abs(mt['Iph0']) - Iph_lab*10**((p[0]-Pid)/20))*1e3
                               if np.isfinite(Rm) else np.nan))
    rows.append(rec)
    data[path] = dict(f=f, p=p, c=c, ref=ref, f3=f3, rt=rt, rec=rec)

t = pd.DataFrame(rows)

# empirical open-device offset (all WO sheets, 0.5-6 mA): measured - ideal
wo = t[(t.label == 'WO') & (t.group != '15dup')]
off = float(np.median(wo.level_minus_ideal_dB))
t['Iph_implied_vs_WOoffset_mA'] = t.Iph_initial_mA.abs()*10**((t.level_minus_ideal_dB - off)/20)
t['R_from_IPD_vs_WOoffset_ohm'] = np.where(np.isfinite(t.Rm_S11_10MHz_ohm),
    t.bias_V.abs()/(t.Iph_initial_mA.abs() - t.Iph_implied_vs_WOoffset_mA)*1e3, np.nan)
t['R_IPD_over_R_S11'] = t.R_from_IPD_vs_WOoffset_ohm/t.Rm_S11_10MHz_ohm
# which sheets the project baseline (ft_userbw.py PAIR) actually uses
USED = {'Bias_7V_Iph_1mA_40um_38ohm_1.xlsx', 'Bias_7V_Iph_1mA_40um_100ohm_1.xlsx',
        'Bias_7V_Iph_1mA_120ohm_40um_140ohm_1.xlsx', 'Bias_7V_Iph_1mA_40um_WO_3.xlsx',
        'Bias_5V_Iph_1mA_40um_WO_2.xlsx', 'Bias_7V_Iph_1mA_40um_38ohm_2.xlsx',
        'Bias_5V_Iph_1mA_40um_38ohm_2.xlsx'}
t['used_in_ft_userbw'] = [os.path.basename(p) in USED for p in t.file]
# exploratory implied f_T per open-device run, using the project's own
# modelled f_RC for that device/bias (ft_userbw.csv): 40um WO -7V 13.776 GHz,
# 40um WO -5V 11.358 GHz.  f_T = (1/f3^2 - 1/fRC^2)^-1/2
FRC = {(40, 'WO', -7): 13.776, (40, 'WO', -5): 11.358, (40, '40', -7): 30.720,
       (40, '100', -7): 20.126, (40, '140', -7): 18.775, (40, '80', -7): 21.816,
       (40, '80', -5): 18.703}
def ft_dev(r):
    k = (r.D_um, r.label, r.bias_V)
    if k not in FRC or not np.isfinite(r.f3dB_cubic_GHz):
        return np.nan
    d = 1/r.f3dB_cubic_GHz**2 - 1/FRC[k]**2
    return np.sqrt(1/d) if d > 0 else np.inf
t['exp_fT_implied_GHz'] = [ft_dev(r) for r in t.itertuples()]

# ── exploratory: common-span truncation and loss-file swap ─────────────────
# (a) f3dB with the fit restricted to <= 14.8 GHz (the shortest 40 um WO span)
t['exp_f3dB_fit_le14p8_GHz'] = [f3_cubic(data[p]['f'], data[p]['p'], 14.8)[0] for p in t.file]
t['exp_f3dB_fit_le20_GHz']   = [f3_cubic(data[p]['f'], data[p]['p'], 20.0)[0] for p in t.file]
# (b) f3dB with DC reference at the lowest measured point instead of f=0
def f3_refmin(f, p):
    c = np.polyfit(f, p, 3); ref = np.polyval(c, f[0])
    ff = np.linspace(0, f[-1], 40001); pp = np.polyval(c, ff) - ref
    i = np.where(pp <= -3.0)[0]
    if len(i) and i[0] > 0:
        j = i[0]; return float(np.interp(-3.0, [pp[j], pp[j-1]], [ff[j], ff[j-1]]))
    return np.nan
t['exp_f3dB_refmin_GHz'] = [f3_refmin(data[p]['f'], data[p]['p']) for p in t.file]

# (c) the two loss files: build loss(f) from the sheets that extend furthest
def loss_curve(paths):
    F, L = [], []
    for q in paths:
        rt = raw_table(q); m = np.isfinite(rt['f']) & np.isfinite(rt['Ltot'])
        F.append(rt['f'][m]); L.append(rt['Ltot'][m])
    F = np.concatenate(F); L = np.concatenate(L)
    o = np.argsort(F); F, L = F[o], L[o]
    # average duplicates on a 0.25 GHz grid
    g = np.round(F*4)/4; u = np.unique(g)
    return u, np.array([L[g == x].mean() for x in u])
fA, LA = loss_curve([f'{B}/40um/Bias_-7V_Iph_1mA.xlsx', f'{B}/40um/V2/WO/Bias_-7V_Iph_1mA.xlsx',
                     f'{B}/30um/40 ohm/Bias_-7V_Iph_6mA.xlsx'])                     # Cal.s2p
fB, LB = loss_curve([f'{B}/40um/V2/40 ohm/Bias_-7V_Iph_1mA.xlsx',
                     f'{B}/25 um/Bias_-7V_Iph_1mA_diff_probe_upto_30GHz.xlsx',
                     f'{B}/30um/Figure_03_27_2026/36 ohm/Bias_-7V_Iph_1mA.xlsx'])   # S21_cal
def swap_loss(path, to):
    rt = raw_table(path)
    fo, Lo = (fB, LB) if to == 'S21' else (fA, LA)
    Lnew = np.interp(rt['f'], fo, Lo)
    f, p, _, _ = clean(rt['f'], rt['Praw'] + Lnew)
    return f3_cubic(f, p)[0]
t['exp_f3dB_other_lossfile_GHz'] = [swap_loss(p, 'S21' if r.s2p_loss_file == 'Cal.s2p' else 'Cal')
                                    for p, r in zip(t.file, t.itertuples())]

cols = ['group', 'D_um', 'label', 'bias_V', 'date', 'time', 'Iph_initial_mA', 's2p_loss_file',
        'N_rows', 'N_clean', 'f_min_GHz', 'f_max_GHz', 'nonmono_restart', 'P_at_fmin_dBm',
        'P_dc_cubic_dBm', 'dB_10GHz_rel_dc', 'dB_15GHz_rel_dc', 'cubic_resid_rms_dB',
        'f3dB_cubic_GHz', 'f3dB_source', 'f3dB_over_fmax', 'exp_f3dB_fit_le14p8_GHz',
        'exp_f3dB_refmin_GHz', 'exp_f3dB_other_lossfile_GHz',
        'Rm_S11_10MHz_ohm', 'P_ideal_at_Iph_dBm', 'level_minus_ideal_dB',
        'Iph_implied_vs_WOoffset_mA', 'R_IPD_over_R_S11', 'exp_fT_implied_GHz', 'used_in_ft_userbw',
        'VOA_mean_dBm', 'lam4_nm', 'md5', 'file']
t.to_csv('inv40/A_inventory.csv', index=False)
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 60)
print('open-device empirical offset (median over WO sheets, measured - ideal):', f'{off:+.2f} dB')
print(t[cols].to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# ── duplicate-run spread of the 40 um open device at -7 V ───────────────────
print('\n=== 40 um open device, -7 V, duplicate runs (02/04) ===')
dup = t[(t.group == '40V1') & (t.label == 'WO') & (t.bias_V == -7) & ~t.file.str.startswith('data_bw_user')]
for r in dup.itertuples():
    print(f"  {r.time}  span {r.f_min_GHz:.2f}-{r.f_max_GHz:.2f} GHz  N={r.N_clean}  P(fmin)={r.P_at_fmin_dBm:.2f}  "
          f"f3={r.f3dB_cubic_GHz:.2f} ({r.f3dB_source}, frac {r.f3dB_over_fmax:.2f})  fit<=14.8: {r.exp_f3dB_fit_le14p8_GHz:.2f}  "
          f"dB@10={r.dB_10GHz_rel_dc:.2f} dB@15={r.dB_15GHz_rel_dc:.2f}  resid rms {r.cubic_resid_rms_dB:.2f}")
print(f"  f3 (full span): mean {dup.f3dB_cubic_GHz.mean():.2f}, sd {dup.f3dB_cubic_GHz.std(ddof=1):.2f}, range {dup.f3dB_cubic_GHz.min():.2f}-{dup.f3dB_cubic_GHz.max():.2f}")
print(f"  f3 (fit<=14.8): mean {dup.exp_f3dB_fit_le14p8_GHz.mean():.2f}, sd {dup.exp_f3dB_fit_le14p8_GHz.std(ddof=1):.2f}")
print(f"  dB@10 rel DC: {dup.dB_10GHz_rel_dc.min():.2f}..{dup.dB_10GHz_rel_dc.max():.2f}; dB@15: {dup.dB_15GHz_rel_dc.min():.2f}..{dup.dB_15GHz_rel_dc.max():.2f}")
# raw-dBm agreement between runs on a common grid (2-14.8 GHz)
grid = np.arange(2.0, 14.81, 0.5)
P = np.array([np.interp(grid, data[p]['f'], data[p]['p']) for p in dup.file])
print(f"  raw dBm spread across the {len(dup)} runs on 2-14.8 GHz: mean |run - mean| = {np.mean(np.abs(P - P.mean(0))):.2f} dB, max {np.max(np.abs(P - P.mean(0))):.2f} dB")
m9 = t[(t.group == '40V1') & (t.label == 'WO') & (t.bias_V == -9)].iloc[0]
print(f"  -9 V run ({m9.time}): f3={m9.f3dB_cubic_GHz:.2f} GHz (span {m9.f_max_GHz:.2f}), dB@10={m9.dB_10GHz_rel_dc:.2f}, dB@15={m9.dB_15GHz_rel_dc:.2f}")

# in-session repeatability from the two interrupted/resumed sweeps
for q in [f'{B}/40um/V2/WO/Bias_-5V_Iph_1mA.xlsx', f'{B}/35um/Bias_-7V_Iph_1mA_diff_powermeter.xlsx']:
    rt = raw_table(q); f, p = rt['f'], rt['Pcal']
    k = np.where(np.diff(f) <= 0)[0][0] + 1
    f1, p1, f2, p2 = f[:k], p[:k], f[k:], p[k:]
    lo, hi = max(f1.min(), f2.min()), min(f1.max(), f2.max())
    g = np.linspace(lo, hi, 20)
    d = np.interp(g, f1, p1) - np.interp(g, f2, p2)
    print(f"  resumed sweep {os.path.basename(q)}: overlap {lo:.2f}-{hi:.2f} GHz, pass1-pass2 mean {d.mean():+.2f} dB, rms {np.sqrt(np.mean(d**2)):.2f} dB")

# ── the 13:33 sweep: 15 um vs 40 um plausibility (exploratory) ─────────────
print('\n=== 13:33 sweep (filed under both 40um/V2/WO and 15um/H.V/WO) ===')
a, b = f'{B}/40um/V2/WO/Bias_-7V_Iph_1mA.xlsx', f'{B}/15um/H.V/WO/Bias_-7V_Iph_1mA.xlsx'
print('  md5 40um/V2/WO :', md5(a), ' 15um/H.V/WO :', md5(b), ' data_bw_user WO_2 :', md5('data_bw_user/Bias_7V_Iph_1mA_40um_WO_2.xlsx'))
print('  cal.xlsx md5   :', md5(f'{B}/40um/V2/WO/cal.xlsx'), md5(f'{B}/15um/H.V/WO/cal.xlsx'))
r = data[a]['rec']
print(f"  f3 = {r['f3dB_cubic_GHz']:.2f} GHz ({r['f3dB_source']}), span to {r['f_max_GHz']:.2f}, dB@10={r['dB_10GHz_rel_dc']:+.2f}, dB@15={r['dB_15GHz_rel_dc']:+.2f}, dB@20={r['dB_20GHz_rel_dc']:+.2f}, resid rms {r['cubic_resid_rms_dB']:.2f}")
fg = np.linspace(1e6, 200e9, 40001); wg = 2*np.pi*fg
Hp = H_ph(wg)
# ladder inductances of the 40 um WO -7 V fit (ft_userbw.csv) and the 30 um Mar WO -7 V fit
for name, Cpd, L1, L2 in [('40 um, C_PD 227.6 fF, L from 40um WO fit', 227.6e-15, 98.10e-12, 163.16e-12),
                          ('15 um, C_PD 39.3 fF (CV_summary), same L', 39.3e-15, 98.10e-12, 163.16e-12),
                          ('15 um, C_PD 39.3 fF, L from 30um Mar WO fit', 39.3e-15, 59.18e-12, 128.23e-12),
                          ('35 um, C_PD 184.8 fF (CV_summary), L from 40um WO fit', 184.8e-15, 98.10e-12, 163.16e-12),
                          ('35 um, C_PD 184.8 fF, L from 30um Mar WO fit', 184.8e-15, 59.18e-12, 128.23e-12)]:
    Hc = H_ckt(wg, Cpd, np.inf, L1, 0.0, L2)
    print(f"  model {name}: f_RC = {f3dB_of(fg, Hc)/1e9:.1f} GHz, f_3dB(H_ph*H_ckt) = {f3dB_of(fg, Hp*Hc)/1e9:.1f} GHz   [exploratory]")
print('\n=== exploratory implied f_T per 40 um sheet (project f_RC, this sheet f_3dB) ===')
for r in t[(t.D_um == 40) & (t.group != '15dup') & ~t.file.str.startswith('data_bw_user')].itertuples():
    print(f"  {r.label:>3s} {r.bias_V:+d} V {r.time}  f3={r.f3dB_cubic_GHz:6.2f}  f_RC={FRC.get((r.D_um, r.label, r.bias_V), np.nan):6.2f}  "
          f"f_T={r.exp_fT_implied_GHz:7.2f}  used={r.used_in_ft_userbw}")

# ── loss files ─────────────────────────────────────────────────────────────
print('\n=== RF loss correction actually applied (Total RF Loss column) ===')
for g in [1, 5, 10, 15, 20, 25, 30]:
    print(f"  {g:2d} GHz: Cal.s2p {np.interp(g, fA, LA):.2f} dB   S21_cal_bias_tee_cal.s2p {np.interp(g, fB, LB):.2f} dB   diff {np.interp(g, fB, LB)-np.interp(g, fA, LA):+.2f}")

# ── session metadata summary ───────────────────────────────────────────────
print('\n=== session metadata ===')
allx = sorted(glob.glob(f'{B}/**/*.xlsx', recursive=True))
ses = []
for q in allx:
    try:
        m = meta(q); rt = raw_table(q)
    except Exception:
        continue
    if len(rt['f'][np.isfinite(rt['f'])]) < 5:
        continue
    ses.append(dict(date=m['date'], time=m['time'], s2p=m['s2p'], lam4=m['lam4'],
                    VOA=np.nanmean(rt['VOA']), fmax=np.nanmax(rt['f']),
                    N=int(np.isfinite(rt['f']).sum()), path=q.replace(B + '/', '')))
ses = pd.DataFrame(ses).sort_values(['date', 'time'])
for d, gdf in ses.groupby('date'):
    print(f"  {d}: {len(gdf)} sweeps {gdf.time.min()}-{gdf.time.max()}  loss={sorted(gdf.s2p.unique())}  "
          f"lam4 {gdf.lam4.min():.3f}-{gdf.lam4.max():.3f} nm  VOA {gdf.VOA.min():.2f}-{gdf.VOA.max():.2f} dBm  "
          f"fmax {gdf.fmax.min():.1f}-{gdf.fmax.max():.1f} GHz  N {gdf.N.min()}-{gdf.N.max()}  "
          f"folders={sorted(set(p.split('/')[0] for p in gdf.path))}")

# ── figures ────────────────────────────────────────────────────────────────
C = {'16:49:37': '#c0392b', '17:00:53': '#e67e22', '18:48:07': '#8e44ad', '17:11:05': '#2471a3',
     '20:52:01': '#16a085', '13:33:49': '0.35', '22:08:46': '#7f8c8d', '17:29:52': '#95a5a6'}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
sel = [(f'{B}/40um/Bias_-7V_Iph_1mA.xlsx',          '40 um WO -7 V 16:49 (WO_3, used)'),
       (f'{B}/40um/Bias_-7V_Iph_1mA_diff_pol.xlsx', '40 um WO -7 V 17:00 diff_pol'),
       (f'{B}/40um/WO_bias__7V_Iph_1mA_#2.xlsx',    '40 um WO -7 V 18:48 #2'),
       (f'{B}/40um/Bias_-9V_Iph_1mA_diff_pol.xlsx', '40 um WO -9 V 17:11 diff_pol'),
       (f'{B}/40um/V2/WO/Bias_-5V_Iph_1mA.xlsx',    '40 um WO -5 V 02/25 20:52 (V2)'),
       (f'{B}/40um/V2/WO/Bias_-7V_Iph_1mA.xlsx',    '13:33 sweep = 15um/H.V/WO (WO_2, excluded)'),
       (f'{B}/35um/Bias_-7V_Iph_1mA.xlsx',          '35 um WO -7 V 02/04 22:08'),
       (f'{B}/35um/Bias_-7V_Iph_1mA_diff_powermeter.xlsx', '35 um WO -7 V 02/03 diff_powermeter')]
for q, lb in sel:
    d = data[q]; col = C[d['rec']['time']]
    ls = '--' if '13:33' in lb else '-'
    lw = 1.0 if '35 um' in lb else 1.4
    axs[0].plot(d['f'], d['p'], ls, marker='o', ms=2.6, lw=lw, color=col, alpha=.85, label=lb)
    ff = np.linspace(0, d['f'][-1], 500)
    axs[1].plot(d['f'], d['p'] - d['ref'], 'o', ms=2.6, color=col, alpha=.5)
    axs[1].plot(ff, np.polyval(d['c'], ff) - d['ref'], ls, lw=lw, color=col,
                label=f"{lb}  f3dB {d['f3']:.1f} GHz" if np.isfinite(d['f3']) else lb)
axs[0].axhline(P_ideal_dBm(1.0, np.inf), color='k', ls=':', lw=1, label='ideal 1 mA open into 50 ohm (-16.0 dBm)')
axs[0].set_xlabel('Beat frequency (GHz)'); axs[0].set_ylabel('Cal RF POW (dBm), absolute')
axs[0].set_title('(a) 40 um open-device runs, raw calibrated dBm', fontsize=10)
axs[0].legend(fontsize=7, loc='lower left'); axs[0].grid(alpha=.3, ls=':'); axs[0].set_xlim(0, 30)
axs[1].axhline(-3, color='0.5', ls='--', lw=1)
axs[1].set_xlabel('Beat frequency (GHz)'); axs[1].set_ylabel('response rel. cubic DC reference (dB)')
axs[1].set_ylim(-14, 2); axs[1].set_xlim(0, 30); axs[1].grid(alpha=.3, ls=':')
axs[1].legend(fontsize=7, loc='lower left'); axs[1].set_title('(b) same, normalised, with the cubic fits', fontsize=10)
fig.tight_layout(); fig.savefig('inv40/A_overlay_WO.png', dpi=200)

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
selR = [(f'{B}/40um/40ohm/Bias_-7V_Iph_1mA_36ohm.xlsx',    '40 um 42.5 ohm -7 V (V1)', '#c0392b'),
        (f'{B}/40um/V2/40 ohm/Bias_-7V_Iph_1mA.xlsx',       '40 um 80.3 ohm -7 V (V2)', '#e67e22'),
        (f'{B}/40um/100ohm/Bias_-7V_Iph_1mA.xlsx',          '40 um 107.5 ohm -7 V (V1)', '#8e44ad'),
        (f'{B}/40um/140hom/Bias_-7V_Iph_1mA_120ohm.xlsx',   '40 um 142.5 ohm -7 V (V1)', '#2471a3'),
        (f'{B}/40um/V2/40 ohm/Bias_-5V_Iph_1mA.xlsx',       '40 um 77.2 ohm -5 V (V2)', '#d35400'),
        (f'{B}/40um/Bias_-7V_Iph_1mA.xlsx',                 '40 um open -7 V (V1)', 'k')]
sel30 = [(f'{B}/30um/40 ohm/Bias_-7V_Iph_1mA.xlsx', '30 um 37.9 ohm -7 V (Jan)'),
         (f'{B}/30um/Figure_03_27_2026/55 ohm/Bias_-7V_Iph_1mA.xlsx', '30 um 60.3 ohm -7 V (Mar)'),
         (f'{B}/30um/76 ohm/Bias_-7V_Iph_1mA.xlsx', '30 um 73.3 ohm -7 V (Jan)'),
         (f'{B}/30um/120 ohm/Bias_-7V_Iph_1.034mA.xlsx', '30 um 115 ohm -7 V (Jan)'),
         (f'{B}/30um/Figure_03_27_2026/200 ohm/Bias_-7V_Iph_1mA.xlsx', '30 um 205 ohm -7 V (Mar)'),
         (f'{B}/30um/Figure_03_27_2026/WO/Bias_-7V_Iph_1mA_30GHz.xlsx', '30 um open -7 V (Mar)')]
for q, lb, col in selR:
    d = data[q]
    axs[0].plot(d['f'], d['p'], '-o', ms=2.6, lw=1.3, color=col, alpha=.85,
                label=f"{lb}: P(fmin) {d['p'][0]:.1f}, ideal {d['rec']['P_ideal_at_Iph_dBm']:.1f} dBm")
    ff = np.linspace(0, d['f'][-1], 500)
    axs[1].plot(ff, np.polyval(d['c'], ff) - d['ref'], '-', lw=1.6, color=col,
                label=f"{lb}  f3dB {d['f3']:.1f}")
    axs[1].plot(d['f'], d['p'] - d['ref'], 'o', ms=2.4, color=col, alpha=.45)
for (q, lb), col in zip(sel30, ['#c0392b', '#e67e22', '#27ae60', '#2471a3', '#8e44ad', 'k']):
    d = data[q]; ff = np.linspace(0, d['f'][-1], 500)
    axs[1].plot(ff, np.polyval(d['c'], ff) - d['ref'], ':', lw=1.4, color=col,
                label=f"{lb}  f3dB {d['f3']:.1f}")
axs[0].set_xlabel('Beat frequency (GHz)'); axs[0].set_ylabel('Cal RF POW (dBm), absolute')
axs[0].set_title('(a) 40 um resistor devices, raw calibrated dBm', fontsize=10)
axs[0].legend(fontsize=7, loc='lower left'); axs[0].grid(alpha=.3, ls=':')
axs[1].axhline(-3, color='0.5', ls='--', lw=1)
axs[1].set_xlabel('Beat frequency (GHz)'); axs[1].set_ylabel('response rel. cubic DC reference (dB)')
axs[1].set_ylim(-12, 2); axs[1].set_xlim(0, 38); axs[1].grid(alpha=.3, ls=':')
axs[1].legend(fontsize=6.5, loc='lower left', ncol=1)
axs[1].set_title('(b) normalised: 40 um (solid) vs 30 um (dotted) resistor devices', fontsize=10)
fig.tight_layout(); fig.savefig('inv40/A_overlay_R.png', dpi=200)

# levels
fig, ax = plt.subplots(figsize=(11, 5))
tt = t[~t.file.str.startswith('data_bw_user')].copy()
tt['x'] = np.arange(len(tt))
colg = {'40V1': '#c0392b', '40V2': '#e67e22', '15dup': '0.4', '35': '#7f8c8d', '30Jan': '#2471a3', '30Mar': '#16a085', '25': '#8e44ad'}
for g, gdf in tt.groupby('group'):
    ax.scatter(gdf.x, gdf.level_minus_ideal_dB, color=colg[g], s=40, label=g, zorder=3)
ax.axhline(off, color='k', ls='--', lw=1, label=f'median open-device offset {off:+.2f} dB')
ax.axhline(0, color='0.5', ls=':', lw=1)
ax.set_xticks(tt.x)
ax.set_xticklabels([f"{r.D_um}um {r.label} {r.bias_V}V {abs(r.Iph_initial_mA):.2f}mA {r.date[:5]} {r.time[:5]}" for r in tt.itertuples()],
                   rotation=90, fontsize=6.5)
ax.set_ylabel('P(f_min) measured - ideal I_ph^2 (R_m||50)^2/(2*50)  (dB)')
ax.set_title('absolute low-frequency level vs the ideal heterodyne level at the labelled I_ph (R_m from S11 at 10 MHz)', fontsize=9)
ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('inv40/A_levels.png', dpi=200)

# loss files
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(fA, LA, '-o', ms=2.5, label='Cal.s2p (01/21, 02/03-04, 03/29, "new cal")')
ax.plot(fB, LB, '-s', ms=2.5, label='S21_cal_bias_tee_cal.s2p (02/16, 02/25, 03/27)')
ax.set_xlabel('Beat frequency (GHz)'); ax.set_ylabel('Total RF Loss added to Raw (dB)')
ax.grid(alpha=.3, ls=':'); ax.legend(fontsize=8)
ax.set_title('RF-loss correction applied, reconstructed from the sheets', fontsize=9)
fig.tight_layout(); fig.savefig('inv40/A_losscal.png', dpi=200)
print('\nwrote inv40/A_inventory.csv, A_overlay_WO.png, A_overlay_R.png, A_levels.png, A_losscal.png')
