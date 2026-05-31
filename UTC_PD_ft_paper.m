%% UTC-PD Transit-Time-Limited Bandwidth (locked baseline)
%  Matches the Python UTC_PD_final.py framework:
%    - Paper 2-region H_ph(ω) formula
%    - τ_A from paper formula: W_A^2 / [D_e (3 + ln(p_max/p_min))]
%    - τ_C = Σ W_i / v_os,i  over the FULL depleted region (980 nm)
%    - τ_R = dielectric relaxation (small)
%    - Bandwidth = -3 dB point of |H_ph(ω)|  (no heuristic RC combination
%      unless device-specific Cj/Rs are provided)
%
%  Reference: Ishibashi (2000), §6.5

clear; clc; close all;

%% 1) Load device dimensions
dimTable = readtable('Dimenssion.xlsx');
dimTable.thickness = dimTable.z2 - dimTable.z1;

% Exclude only the non-active layers (top/bottom contacts and top-side gradings)
exclude = {'P+_cap','Contact_P+','Top contact','InP_n++','InP_P+', ...
           'Grading_Q1.4_1','Grading_Q1.1_2'};
dimTable(ismember(dimTable.region, exclude), :) = [];

% Split into undepleted absorber (W_A) and depleted region (W_C)
isUndep   = ~cellfun('isempty', regexp(dimTable.region, 'Undep'));
isDrift   = ~cellfun('isempty', regexp(dimTable.region, 'Dep_abs|Grading|Drift|Cliff'));
undepTable = dimTable(isUndep,:);
driftTable = dimTable(isDrift,:);

W_A   = sum(undepTable.thickness);     % undep absorber  ~ 480 nm
W_C   = sum(driftTable.thickness);     % depleted region ~ 980 nm
W_tot = W_A + W_C;                     %                 = 1460 nm

fprintf('Layer thicknesses:\n');
fprintf('  W_A (undep absorber) = %.0f nm\n', W_A*1e9);
fprintf('  W_C (depleted region)= %.0f nm\n', W_C*1e9);
fprintf('  W   (total active)   = %.0f nm\n\n', W_tot*1e9);

%% 2) τ_A — paper formula (diffusion + quasi-field)
%  τ_A = W_A^2 / [D_e (3 + ln(p_max/p_min))]
%  D_e = harmonic-weighted average across the three p-doped sub-layers
kT_q  = 0.02585;            % V (room temperature)
p_max = 5.0e18;             % cm^-3  (back contact side)
p_min = 4.0e17;             % cm^-3  (depletion edge side)
D_e   = 118e-4;             % m^2/s  (harmonic average; see Python script)

tau_A = W_A^2 / (D_e * (3 + log(p_max/p_min)));

fprintf('Absorber transit time:\n');
fprintf('  p_max / p_min = %.2g / %.2g cm^-3\n', p_max, p_min);
fprintf('  D_e (harmonic avg)  = %.1f cm^2/s\n', D_e*1e4);
fprintf('  τ_A = %.3f ps\n\n', tau_A*1e12);

%% 3) τ_C = Σ W_i / v_os,i  over the FULL depleted region
%  Ishibashi 2000 §6.5 overshoot velocities:
%     v_os(InP)     = 4.0e7 cm/s  = 4.0e5 m/s
%     v_os(InGaAs)  = 2.1e7 cm/s  = 2.1e5 m/s
%     v_os(InGaAsP) ≈ v_os(InGaAs) (Q1.1, Q1.4 close to InGaAs)
v_os_InP    = 4.0e5;
v_os_InGaAs = 2.1e5;
v_os_Q      = 2.1e5;

nLayers = height(driftTable);
v_layer = zeros(nLayers,1);
for i = 1:nLayers
    reg = driftTable.region{i};
    if contains(reg,'Q1.1') || contains(reg,'Q1.4')
        v_layer(i) = v_os_Q;
    elseif contains(reg,'Dep_abs')
        v_layer(i) = v_os_InGaAs;
    else                                     % Cliff, Drift_440nm, Drift_300nm (InP)
        v_layer(i) = v_os_InP;
    end
end
tau_layers = driftTable.thickness ./ v_layer;
tau_C      = sum(tau_layers);

fprintf('Collector transit time (per layer):\n');
Tcol = table(driftTable.region, driftTable.thickness*1e9, v_layer*1e2*1e-7, ...
             tau_layers*1e12, ...
             'VariableNames',{'Region','Thickness_nm','v_os_1e7cms','tau_ps'});
disp(Tcol);
fprintf('  τ_C (total) = %.3f ps\n\n', tau_C*1e12);

%% 4) τ_R — dielectric relaxation of majority holes (small)
eps0       = 8.854e-12;
eps_r      = 13.9;          % InGaAs relative permittivity
p0         = 1e18 * 1e6;    % m^-3  (effective p-doping in absorber)
mu_h       = 100e-4;        % m^2/Vs (hole mobility, InGaAs)
q          = 1.602e-19;
tau_R      = eps0*eps_r / (q*p0*mu_h);

fprintf('Dielectric relaxation:\n');
fprintf('  τ_R = %.3f ps  (negligible over our frequency range)\n\n', tau_R*1e12);

%% 5) Paper 2-region H_ph(ω)
%  H_ph(ω) = 1/(1+jωτ_A) · 1/W · [ W_A·(2+jωτ_R)/(2(1+jωτ_R))
%                                + W_C·sinc(ωτ_C/2)·exp(-jωτ_C/2) ]
%  Note: MATLAB sinc(x) = sin(πx)/(πx), so use sinc(ωτ_C/(2π)) to obtain
%        sin(ωτ_C/2)/(ωτ_C/2).
f  = linspace(0.01, 200, 40000) * 1e9;    % Hz
w  = 2*pi*f;

abs_term = W_A .* (2 + 1j*w*tau_R) ./ (2*(1 + 1j*w*tau_R));
col_term = W_C .* sinc(w*tau_C/(2*pi)) .* exp(-1j*w*tau_C/2);
H_ph     = (abs_term + col_term) ./ (W_tot * (1 + 1j*w*tau_A));

H_ph_dB  = 20*log10(abs(H_ph)./abs(H_ph(1)));

%% 6) Transit-time-limited -3 dB bandwidth
idx3 = find(H_ph_dB <= -3, 1, 'first');
f_t  = f(idx3);
fprintf('Transit-time-limited bandwidth:\n');
fprintf('  f_t = %.2f GHz   (|H_ph| = -3 dB)\n\n', f_t/1e9);

figure('Color','w'); hold on;
plot(f/1e9, H_ph_dB, 'b-', 'LineWidth', 1.8);
plot(f_t/1e9, -3, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
yline(-3, 'k--', '-3 dB', 'LabelHorizontalAlignment','left');
xline(f_t/1e9, 'r:', sprintf('f_t = %.1f GHz', f_t/1e9), ...
      'LabelOrientation','horizontal','LabelHorizontalAlignment','right');
xlabel('Frequency (GHz)'); ylabel('|H_{ph}| normalized (dB)');
title('Transit-time-limited intrinsic response (paper formula)');
grid on; xlim([0 100]); ylim([-15 1]);
set(gca,'FontSize',12);

%% 7) Optional — BW vs device diameter (Open device, transit + RC)
%  For the Open device (R_p = ∞) the bandwidth is roughly limited by
%  f_t (transit) and f_RC (junction capacitance × source/load resistance).
%
%  C_PD scales with area:    C_PD = ε·A / d_eff
%  Use the measured C_PD = 131 fF at d = 30 μm to back out the effective
%  d_eff (more reliable than assuming the geometric depletion width).
A_30  = pi*(15e-6)^2;
Cj_30 = 131e-15;
d_eff = eps0*eps_r*A_30 / Cj_30;
fprintf('Effective C_PD-derived depletion thickness: %.0f nm\n', d_eff*1e9);

d_um  = (15:60)';
r     = (d_um/2)*1e-6;
A     = pi*r.^2;
Cpd   = eps0*eps_r*A / d_eff;
Rs    = 8.92; Rload = 50;
f_RC  = 1./(2*pi*Cpd*(Rs+Rload));
f_3dB = 1./sqrt(1/f_t^2 + 1./f_RC.^2);

figure('Color','w'); hold on;
plot(d_um, repmat(f_t/1e9,size(d_um)), 'k-',  'LineWidth', 2.0, 'DisplayName','f_t');
plot(d_um, f_RC/1e9,                    'b--', 'LineWidth', 1.6, 'DisplayName','f_{RC}');
plot(d_um, f_3dB/1e9,                   'r-o', 'LineWidth', 1.6, 'DisplayName','f_{3dB} (heuristic)');
xlabel('Diameter (\mum)'); ylabel('Frequency (GHz)');
legend('Location','best');
title('Open-device bandwidth vs diameter');
grid on; xlim([15 60]); set(gca,'FontSize',12);
