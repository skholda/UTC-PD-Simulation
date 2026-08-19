# UTC-PD Model Equations (Baseline)

## 1. Photocurrent transfer function — rigorous Ramo $J_\mathrm{tot}$ integral

$$
H_\mathrm{ph}(\omega) = \frac{1}{W_\mathrm{norm}}\Big[ H_1 + H_2 + H_3 + H_4 \Big]
$$

Term 1 — undepleted-absorber electron (diffusion; carries $\tau_A$):

$$
H_1 = \frac{W_A}{1+j\omega\tau_A}\cdot\frac{2+j\omega\tau_R}{2\,(1+j\omega\tau_R)}
$$

Term 2 — the same electron drifting the depleted region (carries $\tau_A$):

$$
H_2 = \frac{W_C}{1+j\omega\tau_A}\,\mathrm{sinc}\!\left(\frac{\omega\tau_C}{2}\right)e^{-j\omega\tau_C/2}
$$

Term 3 — depleted-absorber in-situ electron (no $\tau_A$):

$$
H_3 = W_{Ad}\,\mathrm{sinc}\!\left(\frac{\omega\tau_{eD}}{2}\right)e^{-j\omega\tau_{eD}/2}
$$

Term 4 — depleted-absorber in-situ hole (no $\tau_A$):

$$
H_4 = W_{Ad}\,\mathrm{sinc}\!\left(\frac{\omega\tau_h}{2}\right)e^{-j\omega\tau_h/2}
$$

with

$$
W_\mathrm{norm} = W_A + W_C + 2\,W_{Ad}, \qquad \mathrm{sinc}(x) = \frac{\sin x}{x}
$$

## 2. Transit times

$$
\tau_A = \frac{W_A^{\,2}}{D_e\left(3+\ln\frac{p_\mathrm{max}}{p_\mathrm{min}}\right)}, \qquad
\tau_C = \int_{W_C}\frac{dz}{v_e\big(E(z)\big)}, \qquad
\tau_{eD} = \tau_{gcc} + \tfrac{1}{2}\,\tau_{dep}
$$

$$
\tau_h = \frac{W_{Ad}}{v_h}, \qquad
\tau_R = \frac{\varepsilon}{\sigma}
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
