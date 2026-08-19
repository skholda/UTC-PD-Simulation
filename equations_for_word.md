# UTC-PD Model Equations (Baseline)

## 1. Photocurrent transfer function — MUTC model $H_\mathrm{MUTC}(\omega)$

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

with $W_T = W_U + W_D + W_C$, $\ \eta_U + \eta_D = 1$, and $\mathrm{sinc}(x)=\dfrac{\sin x}{x}$.

The first bracket ($\eta_U$) is the undepleted-absorber-generated group and carries the diffusion pole $1/(1+j\omega\tau_A)$; the second bracket ($\eta_D$) is the depleted-absorber in-situ group (electron + hole, uniform-generation triangular transit) and carries no $\tau_A$.

## 2. Transit times and generation fractions

Transit times from saturation-velocity crossing:

$$
\tau_{eD} = \frac{W_D}{v_{eD,\mathrm{sat}}}, \qquad
\tau_h = \frac{W_D}{v_{h,\mathrm{sat}}}, \qquad
\tau_C = \frac{W_C}{v_{C,\mathrm{sat}}}
$$

Undepleted-absorber effective transit (diffusion + quasi-field); $\tau_R$ is neglected in the bandwidth calculation:

$$
\tau_A = \frac{W_U^{\,2}}{D_e\left(3+\ln\frac{p_\mathrm{max}}{p_\mathrm{min}}\right)}
$$

Generation fractions (uniform optical generation over the absorbers):

$$
\eta_U = \frac{W_U}{W_U+W_D}, \qquad
\eta_D = \frac{W_D}{W_U+W_D}
$$

## 3. Circuit model

Input impedance and reflection coefficient:

$$
Z_\mathrm{in}(\omega) = j\omega L_\mathrm{CPW} + \frac{1}{\,j\omega C_\mathrm{CPW} + \dfrac{1}{R_p+j\omega L_{Rp}} + \dfrac{1}{Z_\mathrm{dev}}\,},
\qquad
Z_\mathrm{dev} = R_s + j\omega L_\mathrm{CPW2} + \frac{1}{j\omega C_j}
$$

$$
S_{11}(\omega) = \frac{Z_\mathrm{in}-Z_0}{Z_\mathrm{in}+Z_0}, \qquad Z_0 = 50~\Omega
$$

Normalized impedance:

$$
z_\mathrm{norm} = \frac{Z_\mathrm{in}}{Z_0} = \frac{1+S_{11}}{1-S_{11}}
$$

Transimpedance (photocurrent to load voltage):

$$
H_\mathrm{ckt}(\omega) = \frac{V_{R_L}}{I_\mathrm{ph}}
= \frac{\dfrac{R_L}{\,j\omega L_\mathrm{CPW}+R_L\,}}
{\,j\omega C_j + Y_A\,(1+j\omega C_j Z_s)\,},
\qquad
Y_A = j\omega C_\mathrm{CPW} + \frac{1}{R_p+j\omega L_{Rp}} + \frac{1}{j\omega L_\mathrm{CPW}+R_L}
$$

with $Z_s = R_s + j\omega L_\mathrm{CPW2}$.

## 4. Total response and RF power

$$
H_\mathrm{tot}(\omega) = H_\mathrm{ph}(\omega)\,H_\mathrm{ckt}(\omega), \qquad
P_\mathrm{RF}(\omega) = \frac{\big|\,I_\mathrm{ph}\,H_\mathrm{ph}(\omega)\,H_\mathrm{ckt}(\omega)\,\big|^{2}}{2\,R_L}
$$

## 5. Bias-dependent junction capacitance (C–V)

$$
C_j(V) \propto \frac{1}{\sqrt{V_\mathrm{bi}+V}}
\qquad\Rightarrow\qquad
C_j(-7~\mathrm{V}) = 131~\mathrm{fF}, \quad C_j(-5~\mathrm{V}) = 161~\mathrm{fF}
$$
