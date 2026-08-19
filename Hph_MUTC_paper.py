"""
Exact paper H_MUTC(omega) — computed
====================================
Two generation groups:
  Bracket 1 (undep-absorber generated, weight eta_U/[W_T(1+jw tau_A)]):
    W_U (2+jw tR)/(2(1+jw tR)) + W_D D(tau_eD) + W_C sinc(w tC/2) e^{-jw(tau_eD+tC/2)}
  Bracket 2 (dep-absorber generated, weight eta_D/W_T, NO tau_A pole):
    W_U D(tau_h)
    + (W_D/jw tau_eD){1 - D(tau_eD)}          <- triangular (uniform generation), electron
    + (W_D/jw tau_h ){1 - D(tau_h) }          <- triangular, hole
    + W_C sinc(w tau_eD/2) sinc(w tC/2) e^{-jw(tau_eD+tC/2)}
  where D(tau) = sinc(w tau/2) e^{-jw tau/2},  sinc(x)=sin(x)/x.
"""
import numpy as np

# ── widths (m) ──────────────────────────────────────────────────────
W_U = 480e-9      # undepleted p-InGaAs absorber
W_D = 160e-9      # depleted InGaAs absorber
W_C = 820e-9      # electron-only collector (grading 30 + cliff 50 + collector 740)
W_T = W_U + W_D + W_C   # 1460 nm

# ── transit / relaxation times (s) ─────────────────────────────────
tau_A  = 3.530e-12   # undep absorber effective electron transit (diff+quasi-field)
tau_R  = 0.070e-12   # dielectric relaxation, p+ InGaAs
tau_eD = 2.026e-12   # depleted-absorber electron transit  (v(E) integral, 160 nm)
tau_C  = 7.295e-12   # collector-only electron transit     (v(E) integral, 820 nm)
tau_h  = W_D / 4.5e4 # depleted-absorber hole transit (3.56 ps), v_h=4.5e4 m/s

# ── generation fractions via Beer-Lambert (InGaAs alpha) ───────────
alpha = 0.68e6       # 1/m  (InGaAs at 1.55 um; 6800 /cm)
# Light path through the two InGaAs absorbers. Assume illumination reaches
# the depleted absorber first then undepleted (backside/collector side),
# OR undep first (topside). Only the InGaAs layers absorb (InP transparent).
def gen_fractions(order='undep_first'):
    if order == 'undep_first':   # light hits undep absorber (W_U) first
        A_U = 1 - np.exp(-alpha*W_U)
        A_D = np.exp(-alpha*W_U) * (1 - np.exp(-alpha*W_D))
    else:                        # 'dep_first': light hits dep absorber first
        A_D = 1 - np.exp(-alpha*W_D)
        A_U = np.exp(-alpha*W_D) * (1 - np.exp(-alpha*W_U))
    tot = A_U + A_D
    return A_U/tot, A_D/tot       # normalized: eta_U + eta_D = 1

sinc = lambda x: np.sinc(x/np.pi)          # sin(x)/x
def D(w, tau):                              # rectangular transit transfer fn
    return sinc(w*tau/2) * np.exp(-1j*w*tau/2)
def Tri(w, tau):                            # uniform-generation triangular fn
    x = 1j*w*tau
    return np.where(np.abs(x) < 1e-9, 0.5, (1.0 - D(w, tau))/x)

def H_MUTC(w, eta_U, eta_D):
    b1 = ( W_U*(2.0 + 1j*w*tau_R)/(2.0*(1.0 + 1j*w*tau_R))
         + W_D*D(w, tau_eD)
         + W_C*sinc(w*tau_C/2)*np.exp(-1j*w*(tau_eD + tau_C/2)) )
    b1 *= eta_U / (W_T*(1.0 + 1j*w*tau_A))
    b2 = ( W_U*D(w, tau_h)
         + W_D*Tri(w, tau_eD)
         + W_D*Tri(w, tau_h)
         + W_C*sinc(w*tau_eD/2)*sinc(w*tau_C/2)*np.exp(-1j*w*(tau_eD + tau_C/2)) )
    b2 *= eta_D / W_T
    return b1 + b2

# ── run ────────────────────────────────────────────────────────────
f = np.linspace(1e6, 200e9, 800000)
w = 2*np.pi*f

for order in ('undep_first', 'dep_first'):
    eU, eD = gen_fractions(order)
    H  = H_MUTC(w, eU, eD)
    dc = H_MUTC(np.array([2*np.pi*1e6]), eU, eD)[0]
    mag = np.abs(H)/np.abs(dc)
    i3 = np.where(mag <= 1/np.sqrt(2))[0]
    f3 = f[i3[0]]/1e9 if len(i3) else np.nan
    print(f"[{order:>11}]  eta_U={eU:.3f} eta_D={eD:.3f} | "
          f"|H(0)|={np.abs(dc):.6f} | f_tr(-3dB) = {f3:.2f} GHz")

# also: electron-only sensitivity (drop hole terms 2a & 2c)
print("-"*66)
def H_no_hole(w, eta_U, eta_D):
    b1 = ( W_U*(2.0 + 1j*w*tau_R)/(2.0*(1.0 + 1j*w*tau_R))
         + W_D*D(w, tau_eD)
         + W_C*sinc(w*tau_C/2)*np.exp(-1j*w*(tau_eD + tau_C/2)) )
    b1 *= eta_U / (W_T*(1.0 + 1j*w*tau_A))
    b2 = ( W_D*Tri(w, tau_eD)
         + W_C*sinc(w*tau_eD/2)*sinc(w*tau_C/2)*np.exp(-1j*w*(tau_eD + tau_C/2)) )
    b2 *= eta_D / W_T
    return b1 + b2
for order in ('undep_first', 'dep_first'):
    eU, eD = gen_fractions(order)
    H  = H_no_hole(w, eU, eD)
    dc = H_no_hole(np.array([2*np.pi*1e6]), eU, eD)[0]
    mag = np.abs(H)/np.abs(dc)
    i3 = np.where(mag <= 1/np.sqrt(2))[0]
    f3 = f[i3[0]]/1e9 if len(i3) else np.nan
    print(f"[{order:>11}]  (hole terms removed)      f_tr(-3dB) = {f3:.2f} GHz")
