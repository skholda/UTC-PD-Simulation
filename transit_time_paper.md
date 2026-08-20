## Transit-Time-Limited Photocurrent Response

The intrinsic frequency response of the uni-traveling-carrier photodiode (UTC-PD) is limited by carrier transit across the depleted layers. Following the modified uni-traveling-carrier (MUTC) formalism, the photogenerated carriers are partitioned according to their point of generation and propagated to the terminals through the Ramo–Shockley theorem, which expresses the terminal current as the spatial average of the particle current. This yields the transit-time-limited photocurrent transfer function

$$
\begin{aligned}
H_\mathrm{MUTC}(\omega) =\ & \frac{\eta_U}{W_T\,(1+j\omega\tau_A)}\Bigg[
W_U\,\frac{2+j\omega\tau_R}{2(1+j\omega\tau_R)}
+ W_D\,\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)e^{-j\omega\tau_{eD}/2}
+ W_C\,\mathrm{sinc}\!\left(\frac{\omega\tau_C}{2}\right)e^{-j\omega(\tau_{eD}+\tau_C/2)}\Bigg] \\[4pt]
+\ & \frac{\eta_D}{W_T}\Bigg[
W_U\,\mathrm{sinc}\!\left(\frac{\omega\tau_h}{2}\right)e^{-j\omega\tau_h/2}
+ \frac{W_D}{j\omega\tau_{eD}}\left\{1-\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)e^{-j\omega\tau_{eD}/2}\right\} \\[2pt]
& \qquad
+ \frac{W_D}{j\omega\tau_h}\left\{1-\mathrm{sinc}\!\left(\frac{\omega\tau_h}{2}\right)e^{-j\omega\tau_h/2}\right\}
+ W_C\,\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)\mathrm{sinc}\!\left(\frac{\omega\tau_C}{2}\right)e^{-j\omega(\tau_{eD}+\tau_C/2)}\Bigg]
\end{aligned}
\tag{1}
$$

where $\mathrm{sinc}(x)=\sin(x)/x$ and $\omega=2\pi f$. The first bracket, weighted by the undepleted-absorber generation fraction $\eta_U$, represents carriers photogenerated in the undepleted p-InGaAs absorber. As these electrons must diffuse to the depletion edge before drifting, the entire group carries the diffusion pole $(1+j\omega\tau_A)^{-1}$; its three terms describe, respectively, transport within the undepleted absorber (diffusion shaped by dielectric relaxation), the subsequent drift across the depleted absorber, and the drift across the collector, the latter delayed by the cascade factor $e^{-j\omega\tau_{eD}}$ accumulated in crossing the depleted absorber. The second bracket, weighted by $\eta_D$, represents carriers generated in situ within the depleted absorber. Being created inside the field, they drift without diffusion delay and therefore carry no $\tau_A$ pole. Since the optical generation is uniform across the depleted absorber, the in-situ electron and hole are described by the uniform-generation (triangular) transit function $\{1-\mathrm{sinc}(\omega\tau/2)\,e^{-j\omega\tau/2}\}/(j\omega\tau)$ rather than the edge-injection (rectangular) form. The complementary hole term ensures that the electron and hole paths of each pair together span the full transport width $W_T$, so that every collected pair induces one electronic charge. With $\eta_U+\eta_D=1$, the response normalizes to $|H_\mathrm{MUTC}(0)|=1$.

Each carrier crossing a layer of width $W$ at drift velocity $v$ induces a current pulse of duration $\tau=W/v$. The characteristic times are

$$
\tau_A=\frac{W_U^{\,2}}{D_e\!\left(3+\ln\dfrac{p_\mathrm{max}}{p_\mathrm{min}}\right)},\qquad
\tau_{eD}=\frac{W_D}{v_{eD}},\qquad
\tau_C=\frac{W_C}{v_C},\qquad
\tau_h=\frac{W_D}{v_h},
\tag{2}
$$

in which $\tau_A$ is the effective electron transit of the undepleted absorber (carrier diffusion augmented by the quasi-field of the graded doping), $\tau_{eD}$ and $\tau_C$ are the electron drift transits of the depleted absorber and the collector, and $\tau_h$ is the hole drift transit of the depleted absorber. The dielectric-relaxation time $\tau_R=\varepsilon/\sigma$ is negligible over the modulation band and is set to zero, so that the undepleted-absorber factor in (1) reduces to unity.

The drift velocities entering $\tau_{eD}$ and $\tau_C$ are obtained by a layer-averaged evaluation of the simulated field. For each depleted layer the mean electric field is computed from the profile obtained by a drift-diffusion device simulation (Lumerical CHARGE, $-7$ V, $I_\mathrm{ph}=0.5$ mA),

$$
E_\mathrm{avg}=\frac{1}{W}\int_{\mathrm{layer}}\lvert E(z)\rvert\,dz,
\tag{3}
$$

and the corresponding drift velocity $v(E_\mathrm{avg})$ is read from the material velocity–field characteristic—InGaAs for the depleted absorber and InP for the collector—so that $\tau=W/v(E_\mathrm{avg})$. Averaging the field prior to the velocity lookup renders the transit time insensitive to thin low-field regions of the profile. The hole velocity is taken at the InGaAs saturation value.

The generation fractions follow from uniform optical generation across the absorbers,

$$
\eta_U=\frac{W_U}{W_U+W_D},\qquad
\eta_D=\frac{W_D}{W_U+W_D}.
\tag{4}
$$

The extracted parameters are summarized in Table I. With these values, the response is normalized to unity at DC and the transit-time-limited 3-dB bandwidth is $f_{tr}=30.74$ GHz.

**Table I.** Layer geometry, drift fields, velocities, and transit times of the UTC-PD model.

| Layer | Material | $W$ (nm) | $E_\mathrm{avg}$ (kV/cm) | $v$ ($10^7$ cm/s) | $\tau$ (ps) |
|---|---|---|---|---|---|
| Undepleted absorber | InGaAs | 480 | — (diffusion) | — | $\tau_A=3.530$ |
| Depleted absorber | InGaAs | 240 | 179.1 | 0.79 | $\tau_{eD}=3.039$ |
| Collector | InP | 740 | 27.2 | 1.06 | $\tau_C=6.994$ |
| Depleted absorber (holes) | InGaAs | 240 | saturation | 0.48 | $\tau_h=5.000$ |

Generation fractions: $\eta_U=0.667$, $\eta_D=0.333$; total transport thickness $W_T=1460$ nm; $\tau_R$ neglected.
