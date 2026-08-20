# Transit-Time-Limited Photocurrent Transfer Function of the UTC-PD

## 1. The MUTC transfer function

The intrinsic (transit-time-limited) photocurrent response of the uni-traveling-carrier photodiode is modeled with the modified-UTC (MUTC) transfer function. Carriers are separated into two groups according to **where in the absorber the photon is absorbed**, and each group is propagated to the terminals by the Ramo–Shockley theorem (terminal current = spatial average of the particle current):

$$
H_\mathrm{MUTC}(\omega) = \frac{\eta_U}{W_T(1+j\omega\tau_A)}\left[
W_U\,\frac{2+j\omega\tau_R}{2(1+j\omega\tau_R)}
+ W_D\,\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)e^{-j\omega\tau_{eD}/2}
+ W_C\,\mathrm{sinc}\!\left(\frac{\omega\tau_C}{2}\right)e^{-j\omega(\tau_{eD}+\tau_C/2)}
\right]
$$

$$
+\ \frac{\eta_D}{W_T}\left[
W_U\,\mathrm{sinc}\!\left(\frac{\omega\tau_h}{2}\right)e^{-j\omega\tau_h/2}
+ \frac{W_D}{j\omega\tau_{eD}}\left\{1-\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)e^{-j\omega\tau_{eD}/2}\right\}
\right.
$$

$$
\left.
+ \frac{W_D}{j\omega\tau_h}\left\{1-\mathrm{sinc}\!\left(\frac{\omega\tau_h}{2}\right)e^{-j\omega\tau_h/2}\right\}
+ W_C\,\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)\mathrm{sinc}\!\left(\frac{\omega\tau_C}{2}\right)e^{-j\omega(\tau_{eD}+\tau_C/2)}
\right]
$$

with $\mathrm{sinc}(x)=\sin(x)/x$ and $\omega=2\pi f$.

**First bracket ($\eta_U$)** — carriers photogenerated in the *undepleted* absorber. Because these electrons must first diffuse to the depletion edge, the whole group carries the diffusion pole $1/(1+j\omega\tau_A)$. The three terms are, in order: the electron transport inside the undepleted absorber (diffusion shaped by dielectric relaxation), its subsequent drift across the depleted absorber, and its drift across the collector (with the cascaded delay $e^{-j\omega\tau_{eD}}$ accumulated while crossing the depleted absorber).

**Second bracket ($\eta_D$)** — carriers photogenerated *in situ* in the depleted absorber. They are born inside the field and drift immediately, so this group carries **no** $\tau_A$ pole. Because the generation is uniform across the depleted absorber, the electron and hole use the uniform-generation (triangular) transit transfer function $\{1-\mathrm{sinc}(\omega\tau/2)e^{-j\omega\tau/2}\}/(j\omega\tau)$ rather than the edge-injection (rectangular) form. The last term is the in-situ electron continuing into the collector, and the first term is the Ramo complement of the hole (so that the electron path plus the hole path of each pair spans the full $W_T$, i.e. each collected pair induces exactly one electronic charge).

At DC, $\eta_U+\eta_D=1$ gives $|H_\mathrm{MUTC}(0)|=1$.

## 2. Transit-time definitions

Each carrier that crosses a layer of width $W$ at drift velocity $v$ induces a current pulse of duration $\tau=W/v$. The four characteristic times are:

$$
\tau_A = \frac{W_U^{\,2}}{D_e\left(3+\ln\dfrac{p_\mathrm{max}}{p_\mathrm{min}}\right)}, \qquad
\tau_{eD} = \frac{W_D}{v_{eD}}, \qquad
\tau_C = \frac{W_C}{v_C}, \qquad
\tau_h = \frac{W_D}{v_h}
$$

- $\tau_A$ — effective electron transit of the **undepleted** absorber (diffusion plus the built-in quasi-field of the graded doping). Computed from the paper formula; not recomputed from the field.
- $\tau_{eD}$ — electron drift transit of the **depleted absorber** ($W_D$).
- $\tau_C$ — electron drift transit of the **collector** ($W_C$).
- $\tau_h$ — hole drift transit of the depleted absorber ($W_D$, backward toward the p-contact).
- $\tau_R$ — dielectric relaxation time; **neglected in the bandwidth calculation** ($\tau_R\to 0$, so the undepleted-absorber factor reduces to unity).

## 3. How the drift velocities are obtained — layer-average $v(E_\mathrm{avg})$

The electron drift transits $\tau_{eD}$ and $\tau_C$ are evaluated with the **layer-average field** method, combining two independent inputs:

1. **Device field** — the simulated E-field profile $E(z)$ from Lumerical CHARGE (bias $-7$ V, $I_\mathrm{ph}=0.5$ mA). The mean field of each layer is taken:
$$
E_\mathrm{avg} = \frac{1}{W}\int_{\mathrm{layer}} |E(z)|\,dz
$$
2. **Material velocity–field curve** — the drift velocity at that average field, $v(E_\mathrm{avg})$, read from the measured/simulated $v\!-\!E$ characteristic of the layer material (InGaAs for the depleted absorber, InP for the collector).

The transit time is then $\tau = W/v(E_\mathrm{avg})$. Averaging the field *before* looking up the velocity makes the result robust to thin low-field patches in the profile (which would otherwise dominate a point-by-point $\int dz/v(E(z))$ integral). The hole velocity $v_h$ is taken at its saturation value (InGaAs hole saturation velocity, literature).

## 4. Transit-time values

| Quantity | Layer | Width $W$ | $E_\mathrm{avg}$ | Drift velocity | Transit time |
|---|---|---|---|---|---|
| $\tau_A$ | Undepleted absorber (InGaAs) | 480 nm | — (diffusion) | — | **3.530 ps** |
| $\tau_{eD}$ | Depleted absorber (InGaAs) | 240 nm | 179.1 kV/cm | $0.79\times10^7$ cm/s | **3.039 ps** |
| $\tau_C$ | Collector (InP) | 740 nm | 27.2 kV/cm | $1.06\times10^7$ cm/s | **6.994 ps** |
| $\tau_h$ | Depleted absorber, holes (InGaAs) | 240 nm | (saturation) | $0.48\times10^7$ cm/s | **5.000 ps** |
| $\tau_R$ | Dielectric relaxation | — | — | — | neglected |

## 5. Geometry, generation fractions, and result

| Symbol | Meaning | Value |
|---|---|---|
| $W_U$ | Undepleted p-InGaAs absorber thickness | 480 nm |
| $W_D$ | Depleted InGaAs absorber thickness | 240 nm |
| $W_C$ | Electron-only collector thickness (InP) | 740 nm |
| $W_T$ | Total transport thickness $W_U+W_D+W_C$ | 1460 nm |
| $\eta_U$ | Undepleted-absorber generation fraction $W_U/(W_U+W_D)$ | 0.6667 |
| $\eta_D$ | Depleted-absorber generation fraction $W_D/(W_U+W_D)$ | 0.3333 |

Generation fractions assume uniform optical generation across the absorbers.

The DC value normalizes to $|H_\mathrm{MUTC}(0)|=1$, and the transit-time-limited bandwidth ($|H_\mathrm{MUTC}|=-3$ dB) is:

$$
f_{tr} = 30.74~\mathrm{GHz}
$$
