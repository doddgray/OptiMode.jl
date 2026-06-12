# optimode — Python interface to OptiMode.jl

A Python package wrapping the [OptiMode.jl](..) differentiable electromagnetic mode
solver via [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). It
exposes the full pipeline — material dispersion models, sub-pixel dielectric
smoothing, plane-wave Helmholtz eigensolves, mode analysis (group index, GVD,
effective area, polarization, Kerr power corrections) and asynchronous SLURM batch
sweeps — with Python/NumPy-native interfaces matching the Julia API.

Full interface documentation with usage examples: [`docs/python.md`](../docs/python.md).
The physics/mathematics of each component: [`docs/`](../docs).

## Installation

From a checkout of the OptiMode.jl repository (the package locates the Julia project
automatically when used in-repo):

```bash
pip install -e python/          # or: pip install juliacall numpy && use python/ on sys.path
```

Requirements: Python ≥ 3.9 and a Julia ≥ 1.10 installation (found on `PATH` or in
`~/.juliaup`; override with `PYTHON_JULIAPKG_EXE`). The first import starts an
embedded Julia runtime and loads the OptiMode packages (precompiling on first use).
Outside a checkout, point `OPTIMODE_JULIA_PROJECT` at a Julia project where
`OptiMode` is installed.

## Quick start

```python
import optimode as om

grid = om.Grid(4.0, 3.0, 128, 96)                       # 4×3 μm cell
mat_vals = om.f_eps_mats([om.Si3N4, om.SiO2])([1/1.55]) # (ε, ∂ωε, ∂²ωε) at ω = 1/λ
core = om.box([0, 0], [1.6, 0.8], 1)                    # Si₃N₄ core, SiO₂ background
sm = om.smooth_eps([core], mat_vals, (1, 2), grid)      # sub-pixel smoothing
eps_inv, deps, ddeps = om.inv_eps_slices(sm)

kmags, evecs = om.solve_k(1/1.55, eps_inv, grid, nev=2) # eigenmodes at fixed ω
neff = kmags[0] * 1.55
ng, gvd = om.ng_gvd(1/1.55, kmags[0], evecs[0], eps_inv, deps, ddeps, grid)

# Kerr: power-dependent effective index (n₂ from the material library)
n2_map = om.smooth_scalar([core], [om.kerr_n2(om.Si3N4), om.kerr_n2(om.SiO2)], (1, 2), grid)
res = om.solve_k_kerr(1/1.55, 1.0, eps_inv, deps, n2_map, grid)   # P = 1 W
dneff = (res["kmags"][0] - res["kmags_lin"][0]) * 1.55
```

A complete runnable example is in
[`examples/si3n4_waveguide_kerr.py`](examples/si3n4_waveguide_kerr.py); tests in
[`tests/test_optimode.py`](tests/test_optimode.py) exercise the full API surface
(run with `pytest python/tests`).

## Units

Same as OptiMode.jl (c = 1): lengths/wavelengths in μm, frequencies `ω = 1/λ` in
μm⁻¹, propagation constants `k = n_eff/λ` in μm⁻¹, powers in W, Kerr coefficients in
μm²/W.
