"""
Rigorous derivation of H_ph from the ORIGINAL paper equation
============================================================
Goal: start from J_tot = (1/W) integral [ J_e + J_h ] dx  (Ramo),
      NOT by arbitrarily bolting a hole term onto the pole.

Paper original (undep absorber, single-region electron diffusion):
    d/dx [ J_e + J_h ] = jωτ_R/(1+jωτ_R) · dJ_e/dx           (continuity+dielectric)
    J_e(x) = J_DC / (1+jωτ_A) · f(x)                          (diffusion pole)
    Terminal current = (1/W) ∫ J dx                           (Ramo/Shockley theorem)

We reconstruct the terminal (induced) current as the Ramo spatial average of the
PARTICLE currents actually flowing in each region, then normalize by DC.

Structure of the real device (z increasing from collector -> p-contact):
    Collector (InP)        W_C_coll  740 nm   depleted, e drift only
    Cliff (InP)             50 nm             depleted, e drift only
    Grading (Q1.1/Q1.4)     30 nm             depleted, e drift only
    Depleted absorber       160 nm  (W_Ad)    depleted InGaAs: e-h generated IN SITU
    Undepleted absorber     480 nm  (W_A)     p+ InGaAs: e diffuses, no field

Physical carrier bookkeeping:
  * Undep-absorber photo-electrons: diffuse across W_A (delay pole 1/(1+jωτ_A)),
    THEN drift as electrons through the full depleted stack W_C.
  * Dep-absorber photo-electrons: generated in situ inside the depleted absorber,
    NO diffusion delay; drift forward through (part of dep-abs + grading+cliff+coll).
  * Dep-absorber photo-holes: generated in situ, drift BACKWARD the short remaining
    distance to the p+ edge; NO τ_A delay (they never diffused).

Each drifting carrier that crosses a region of width Wr at velocity v induces,
by Ramo, a box current pulse of duration τr = Wr/v -> transfer sinc(ωτr/2)·exp(-jωτr/2).
The Ramo weight of that pulse in the terminal average is (Wr / W_norm).
"""
import numpy as np

# ── widths (m) ──────────────────────────────────────────────────────
W_A   = 480e-9      # undep absorber (diffusion)
W_Ad  = 160e-9      # depleted absorber (in-situ generation)
W_gcc = 820e-9      # grading+cliff+collector depleted InP stack (30+50+740)
W_C   = W_Ad + W_gcc  # 980 nm full depleted region
W_norm = W_A + W_C    # 1460 nm  <-- ONLY the physical widths; no double count

# ── transit times (s) ──────────────────────────────────────────────
tau_A  = 3.530e-12   # undep absorber diffusion pole (paper formula)
tau_R  = 0.070e-12   # dielectric relaxation (majority holes, p+ InGaAs)
# v(E) drift times over each depleted sub-region (Lumerical -7V,1.5mA):
tau_dep  = 2.026e-12   # electron crossing full dep-absorber (160 nm)
tau_gcc  = 7.295e-12   # electron crossing grading+cliff+collector (820 nm)
tau_C    = tau_dep + tau_gcc      # = 9.321 ps  full depleted electron transit
v_h      = 4.5e4       # m/s hole saturation velocity InGaAs
tau_h    = W_Ad / v_h  # 3.56 ps hole crossing dep-absorber (backward, full width avg)

sinc = lambda x: np.sinc(x/np.pi)

def H_ph_rigorous(w):
    """Ramo spatial-average of the particle currents, normalized by DC."""
    # 1) Undep-absorber electrons: diffusion pole, then drift the FULL depleted stack.
    #    Ramo box over W_C at electron velocity -> sinc·exp(τ_C). Weight W_A (the
    #    absorbed flux originates in W_A) but the induced pulse spans W_C.
    #    In the paper's 2-region form the diffusion region contributes its own
    #    (2+jωτ_R)/(2(1+jωτ_R)) shape over W_A, and the drift over W_C.
    e_diff = W_A * (2.0 + 1j*w*tau_R) / (2.0*(1.0 + 1j*w*tau_R)) / (1.0 + 1j*w*tau_A)
    e_drift_undep = W_C * sinc(w*tau_C/2) * np.exp(-1j*w*tau_C/2) / (1.0 + 1j*w*tau_A)

    # 2) Dep-absorber electrons: in situ, NO τ_A. Drift forward through the
    #    grading+cliff+collector stack (+ average half of dep-abs). Ramo box over
    #    ~W_gcc + half dep-abs; weight W_Ad.
    tau_e_ds = tau_gcc + 0.5*tau_dep
    e_drift_dep = W_Ad * sinc(w*tau_e_ds/2) * np.exp(-1j*w*tau_e_ds/2)

    # 3) Dep-absorber holes: in situ, NO τ_A. Drift backward over dep-abs; weight W_Ad.
    h_drift_dep = W_Ad * sinc(w*tau_h/2) * np.exp(-1j*w*tau_h/2)

    return (e_diff + e_drift_undep + e_drift_dep + h_drift_dep) / W_norm_eff(w)

def W_norm_eff(w):
    # DC normalization constant (w->0 total numerator weight)
    return (W_A + W_C + W_Ad + W_Ad)  # e_diff(W_A)+e_drift_undep(W_C)+e_dep(W_Ad)+h(W_Ad)

# ── DC check ────────────────────────────────────────────────────────
print("="*70)
print("RIGOROUS H_ph from original J_tot = (1/W)∫[J_e+J_h]dx")
print("="*70)
dc = H_ph_rigorous(1e-3)
print(f"DC normalization |H_ph(0)| = {abs(dc):.6f}   (must be 1.000000)")

# ── -3 dB transit-limited bandwidth ─────────────────────────────────
f = np.linspace(1e9, 200e9, 400000)
w = 2*np.pi*f
H = np.array([H_ph_rigorous(wi) for wi in w])
mag = np.abs(H)/abs(dc)
idx = np.where(mag <= 1/np.sqrt(2))[0]
f3 = f[idx[0]]/1e9 if len(idx) else np.nan
print(f"f_tr (-3 dB, WITH dep-abs hole)    = {f3:.2f} GHz")

# without hole
def H_no_hole(w):
    e_diff = W_A * (2.0 + 1j*w*tau_R) / (2.0*(1.0 + 1j*w*tau_R)) / (1.0 + 1j*w*tau_A)
    e_drift_undep = W_C * sinc(w*tau_C/2) * np.exp(-1j*w*tau_C/2) / (1.0 + 1j*w*tau_A)
    tau_e_ds = tau_gcc + 0.5*tau_dep
    e_drift_dep = W_Ad * sinc(w*tau_e_ds/2) * np.exp(-1j*w*tau_e_ds/2)
    return (e_diff + e_drift_undep + e_drift_dep) / (W_A + W_C + W_Ad)
Hn = np.array([H_no_hole(wi) for wi in w])
magn = np.abs(Hn)/abs(H_no_hole(1e-3))
idxn = np.where(magn <= 1/np.sqrt(2))[0]
f3n = f[idxn[0]]/1e9 if len(idxn) else np.nan
print(f"f_tr (-3 dB, WITHOUT hole)         = {f3n:.2f} GHz")

# ── compare to current baseline form ────────────────────────────────
W_A_p, W_C_p, W_Ad_p = 480e-9, 980e-9, 160e-9
W_norm_p = W_A_p + W_C_p + W_Ad_p
def H_baseline(w):
    abs_term  = W_A_p * (2.0 + 1j*w*tau_R) / (2.0*(1.0 + 1j*w*tau_R))
    col_term  = W_C_p * sinc(w*tau_C/2) * np.exp(-1j*w*tau_C/2)
    hole_term = W_Ad_p* sinc(w*tau_h/2) * np.exp(-1j*(w*tau_A + w*tau_h/2))
    return (abs_term + col_term + hole_term) / (W_norm_p*(1.0 + 1j*w*tau_A))
Hb = np.array([H_baseline(wi) for wi in w])
magb = np.abs(Hb)/abs(H_baseline(1e-3))
idxb = np.where(magb <= 1/np.sqrt(2))[0]
f3b = f[idxb[0]]/1e9 if len(idxb) else np.nan
print(f"f_tr (-3 dB, CURRENT baseline)     = {f3b:.2f} GHz")
print("="*70)
print("Key difference: baseline applies 1/(1+jωτ_A) to the WHOLE numerator,")
print("i.e. also to dep-absorber-generated carriers that never diffused.")
print("Rigorous form gives τ_A only to undep-absorber electrons.")
