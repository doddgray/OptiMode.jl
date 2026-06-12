"""Si3N4 rectangular waveguide from Python: modes, dispersion and Kerr power sweep.

Mirrors examples/kerr_si3n4_waveguide.jl: a Si3N4 core (1.60 x 0.80 um) in SiO2 at
lambda = 1.55 um. Computes the fundamental mode, its group index and GVD, then sweeps
optical power and compares the Kerr effective-index shift with the textbook
self-phase-modulation estimate n2*P/Aeff.

Run from the repository root:
    python python/examples/si3n4_waveguide_kerr.py
"""

import numpy as np

import optimode as om

# --- structure & materials --------------------------------------------------------
lam = 1.55                    # vacuum wavelength [um]
omega = 1 / lam               # frequency [um^-1]  (c = 1 units)
grid = om.Grid(4.0, 3.0, 96, 72)

mats = [om.Si3N4, om.SiO2]
mat_vals = om.f_eps_mats(mats)([omega])              # (27, 2): eps, d_eps, d2_eps
core = om.box([0.0, 0.0], [1.60, 0.80], 1)           # material 1 = Si3N4
sm = om.smooth_eps([core], mat_vals, (1, 2), grid)   # background (2) = SiO2
eps_inv, deps, ddeps = om.inv_eps_slices(sm)

n2_vals = [om.kerr_n2(m, lam) for m in mats]         # [2.4e-7, 2.6e-8] um^2/W
n2_map = om.smooth_scalar([core], n2_vals, (1, 2), grid)
print(f"n2(Si3N4) = {n2_vals[0]:.2e} um^2/W, n2(SiO2) = {n2_vals[1]:.2e} um^2/W")

# --- linear mode ------------------------------------------------------------------
kmags, evecs = om.solve_k(omega, eps_inv, grid, nev=1, k_tol=1e-10)
neff = kmags[0] / omega
ng, gvd = om.ng_gvd(omega, kmags[0], evecs[0], eps_inv, deps, ddeps, grid)

# standard nonlinear effective area from the intensity profile at P = 1 W
I1 = om.mode_intensity(kmags[0], evecs[0], eps_inv, grid, 1.0)
Aeff = 1.0**2 / ((I1**2).sum() * grid.dV)
gamma = 2 * np.pi * n2_vals[0] / (lam * Aeff)        # [1/(W um)]
print(f"linear: neff = {neff:.5f}, ng = {ng:.4f}, GVD = {gvd:.4f} um, "
      f"Aeff = {Aeff:.3f} um^2, gamma = {gamma*1e6:.2f} /(W m)")

# --- Kerr power sweep -------------------------------------------------------------
print(f"\n{'P [W]':>8}  {'neff(P)':>12}  {'dneff (solve)':>14}  {'n2*P/Aeff':>12}")
for P in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    res = om.solve_k_kerr(omega, P, eps_inv, deps, n2_map, grid, nev=1, k_tol=1e-10)
    dneff = (res["kmags"][0] - res["kmags_lin"][0]) / omega
    print(f"{P:8.2f}  {res['kmags'][0]/omega:12.7f}  {dneff:14.4e}  "
          f"{n2_vals[0]*P/Aeff:12.4e}")

print("\nLarger sweeps (P x omega x geometry) deploy asynchronously with "
      "om.deploy_batch / om.frequency_sweep; see docs/python.md.")
