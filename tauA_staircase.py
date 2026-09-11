"""tau_A for the undepleted p-InGaAs absorber: staircase (3-step doping) model.

Reproduces the number used in the project baseline (tau_A = 1.989 ps) and the
continuous-grading value it replaced (3.530 ps), with the intermediate
per-layer numbers.

Structure (top -> depletion edge), from the doping profile supplied by the
device owner:
    L1  p = 5e18 cm^-3   120 nm
    L2  p = 1.3e18       180 nm
    L3  p = 4e17         180 nm
    total 480 nm; electron diffusion coefficient D_e = 118 cm^2/s (given)

Staircase model
---------------
Inside each sublayer the doping is uniform, so there is no concentration
gradient and hence no quasi-field: transport within a layer is pure
diffusion. The built-in potential drop of the graded absorber is therefore
concentrated at the two doping steps, dE = kT ln(p_upper/p_lower), which are
treated as ideal one-way barriers: an electron that reaches a step is swept
into the lower layer and cannot diffuse back.

For an electron generated uniformly inside layer i (reflecting top surface,
absorbing bottom interface) the mean time to leave the layer is W_i^2/(3 D).
It then crosses every lower layer j as an edge-injected carrier, mean time
W_j^2/(2 D). Photogeneration is taken as uniform through the 480 nm (the
absorption length is much longer than the layer), so the layer delays are
averaged with the layer thicknesses as weights.

Continuous-grading model (previous value)
-----------------------------------------
tau_A = W^2 / [ D (3 + ln(p_max/p_min)) ]   with W = 480 nm — the standard
quasi-field-enhanced diffusion time for an exponentially graded absorber.
"""
import numpy as np

kT_meV = 25.85
D_e = 118.0e-4                      # m^2/s
layers = [(5.0e18, 120e-9), (1.3e18, 180e-9), (4.0e17, 180e-9)]   # (p, W)
W_tot = sum(W for _, W in layers)

print('Interface steps (one-way barriers):')
for (pa, _), (pb, _), tag in zip(layers[:-1], layers[1:], ('L1|L2', 'L2|L3')):
    print(f'  {tag}: dE = kT ln({pa:.1e}/{pb:.1e}) = {kT_meV*np.log(pa/pb):5.1f} meV'
          f'  ({np.log(pa/pb):.2f} kT)')

print('\nPer-layer delay (ps):   own layer W^2/3D   +  pass-through sum W_j^2/2D')
tau_layers = []
for i, (p, W) in enumerate(layers):
    own = W**2/(3*D_e)
    thru = sum(Wj**2/(2*D_e) for _, Wj in layers[i+1:])
    tau_layers.append(own + thru)
    print(f'  L{i+1} p={p:.1e} W={W*1e9:3.0f} nm : {own*1e12:6.3f} + {thru*1e12:6.3f}'
          f' = {(own+thru)*1e12:6.3f}')

tau_stair = sum(W*t for (_, W), t in zip(layers, tau_layers))/W_tot
p_max, p_min = layers[0][0], layers[-1][0]
tau_cont = W_tot**2/(D_e*(3 + np.log(p_max/p_min)))
print(f'\nthickness-weighted tau_A (staircase)      = {tau_stair*1e12:.3f} ps')
print(f'continuous grading W^2/[D(3+ln r)], r={p_max/p_min:.1f} = {tau_cont*1e12:.3f} ps')

# ── f_tr of the full 4-term H_ph for each tau_A ───────────────────────────
W_A, W_Ad, W_C = 480e-9, 160e-9, 820e-9
W_norm = W_A + W_C + 2*W_Ad
tau_eD, tau_C, tau_h = 2.026e-12, 7.794e-12, W_Ad/4.8e4

def H_ph(w, tau_A, tau_R=0.0):
    s = lambda x: np.sinc(x/np.pi)
    return (W_A/(1+1j*w*tau_A)*(2+1j*w*tau_R)/(2*(1+1j*w*tau_R))
            + W_C/(1+1j*w*tau_A)*s(w*tau_C/2)*np.exp(-1j*w*tau_C/2)
            + W_Ad*s(w*tau_eD/2)*np.exp(-1j*w*tau_eD/2)
            + W_Ad*s(w*tau_h/2)*np.exp(-1j*w*tau_h/2))/W_norm

f = np.linspace(1e6, 150e9, 300001); w = 2*np.pi*f
def f3(tau_A):
    m = 20*np.log10(np.abs(H_ph(w, tau_A)))
    return f[np.argmax(m <= -3.0)]/1e9

print(f'\nf_tr of H_ph:  staircase tau_A -> {f3(tau_stair):.2f} GHz   '
      f'continuous tau_A -> {f3(tau_cont):.2f} GHz')
for ft in (39.0, 35.5, 33.7, 44.4):
    # tau_A that would reproduce a given f_tr (bisection)
    lo, hi = 0.5e-12, 6e-12
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if f3(mid) > ft: lo = mid
        else: hi = mid
    print(f'  f_tr = {ft:4.1f} GHz  <->  tau_A = {0.5*(lo+hi)*1e12:.2f} ps')
