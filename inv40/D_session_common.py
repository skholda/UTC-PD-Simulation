"""Dimension D step 5b: is there a session-common ripple / roll-off?
For each session, remove each sheet's own cubic (DC ref) and look at the mean
residual across sheets on a common grid. A chain artefact missed by the cal
would appear as a mean residual with structure; device-specific behaviour
averages out. Output: inv40/D_session_common.png and printed numbers."""
import os, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir('/home/user/UTC-PD-Simulation')
C = pickle.load(open('inv40/D_losscurves.pkl','rb')); inv = pd.read_csv('inv40/D_inventory.csv')
sess = {'01/21 30um (Cal.s2p)': [p for p in inv.path if '30um/' in p and 'Figure' not in p and 'sim' not in p and 'cal.xlsx' not in p],
        '02/04 40/35/15um (Cal.s2p)': [p for p in inv.path if (('40um/' in p and 'V2/40' not in p and 'V2/WO/Bias_-5V' not in p) or '35um/' in p or '15um/' in p) and 'cal.xlsx' not in p and 'test' not in p],
        '02/25 40um V2 (S21_cal)': [p for p in inv.path if '40um/V2' in p and 'cal.xlsx' not in p and not ('WO/Bias_-7V' in p)],
        '03/27 30um (S21_cal, original)': [p for p in inv.path if 'Figure_03_27' in p and 'new cal' not in p and '200 ohm' not in p and 'cal.xlsx' not in p and 'test' not in p],
        '03/29 30um 200ohm (Cal.s2p)': [p for p in inv.path if 'Figure_03_27' in p and 'new cal' not in p and '200 ohm' in p]}
grid = np.arange(0.5, 30.01, 0.5)
fig, ax = plt.subplots(len(sess), 1, figsize=(9, 2.6*len(sess)), sharex=True)
for k, (name, ps) in enumerate(sess.items()):
    R = []
    for p in ps:
        d = C[p]; d = d[d.raw < 0].sort_values('f').drop_duplicates('f')
        f, pc = d.f.values, d.cal.values
        if len(f) < 10: continue
        c = np.polyfit(f, pc, 3); res = pc - np.polyval(c, f)
        r = np.interp(grid, f, res, left=np.nan, right=np.nan); R.append(r)
        ax[k].plot(f, res, '-', lw=.5, alpha=.5)
    R = np.array(R); n = np.isfinite(R).sum(0); mean = np.nanmean(R, 0); sd = np.nanstd(R, 0)
    ok = n >= 3
    ax[k].plot(grid[ok], mean[ok], 'k-', lw=2, label=f'mean over {len(R)} sheets (n>=3)')
    ax[k].set_ylim(-1.5, 1.5); ax[k].set_ylabel('resid (dB)'); ax[k].set_title(name, fontsize=9); ax[k].grid(alpha=.3); ax[k].legend(fontsize=7)
    rms_mean = np.sqrt(np.nanmean(mean[ok]**2)); rms_ind = np.sqrt(np.nanmean(R[:, ok]**2))
    i = np.nanargmax(np.abs(mean[ok]))
    print(f'{name:34s} sheets {len(R):2d}  rms(mean resid) {rms_mean:.3f} dB, rms(individual resid) {rms_ind:.3f} dB, '
          f'largest |mean| {mean[ok][i]:+.2f} dB at {grid[ok][i]:.1f} GHz; mean resid 15-20 GHz: {np.nanmean(mean[ok & (grid>=15) & (grid<=20)]):+.2f} dB')
ax[-1].set_xlabel('beat frequency (GHz)'); fig.tight_layout(); fig.savefig('inv40/D_session_common.png', dpi=120)
