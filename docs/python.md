# Python interface (`optimode`)

The [`python/`](../python) directory contains **optimode**, a Python package exposing
the entire OptiMode.jl pipeline with Python/NumPy-native interfaces. It embeds a
Julia runtime via [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/);
all heavy computation runs in the same compiled Julia code as the native interface,
so results match to machine precision and there is no performance penalty beyond
NumPy↔Julia array conversion at the boundary.

```mermaid
flowchart LR
    A["Python user code<br/>(NumPy arrays, dicts)"] --> B["optimode<br/>(thin typed wrappers)"]
    B -->|JuliaCall| C["embedded Julia runtime<br/>OptiMode.jl packages"]
    C -->|"NumPy views / lists of dicts"| A
```

## Installation & configuration

```bash
pip install -e python/        # from a checkout of this repository
```

Requirements: Python ≥ 3.9, NumPy, and a Julia ≥ 1.10 installation. The wrapper finds
`julia` on `PATH` or in `~/.juliaup` (override with `PYTHON_JULIAPKG_EXE`) and, when
used from a repository checkout, automatically stacks the repo's Julia project onto
the load path. Outside a checkout set `OPTIMODE_JULIA_PROJECT` to a Julia project
with `OptiMode` installed. The first `import optimode` call starts Julia and loads
the packages (seconds when precompiled; longer on very first use).

## Name mapping

The Python API mirrors the Julia API one-to-one; Unicode names get ASCII equivalents:

| Julia | Python |
|---|---|
| `Si₃N₄`, `SiO₂`, `LiNbO₃`, `MgO_LiNbO₃`, `LiB₃O₅`, `αAl₂O₃` | `Si3N4`, `SiO2`, `LiNbO3`, `MgO_LiNbO3`, `LiB3O5`, `alpha_Al2O3` |
| `ε_fn(mat)(λ)` | `eps(mat, lam)` / `eps_fn(mat)` |
| `nn̂g`, `nĝvd` | `nng(mat, lam)`, `ngvd(mat, lam)` |
| `kerr_n2`, `with_kerr_n2`, `rotate` | same names |
| `_f_ε_mats(mats, (:ω,))` | `f_eps_mats(mats)` → callable `p -> (27, n) array` |
| `ε_views` | `eps_views(values, n_mats)` |
| `Grid(Δx, Δy, Nx, Ny)`; `x`, `δx`, `δV` | `Grid(Dx, Dy, Nx, Ny)`; `.x`, `.dx`, `.dV` properties |
| `MaterialShape(Cuboid(...), i)` etc. | `box(center, size, i)`, `polygon(verts, i)`, `ball(c, r, i)` |
| `smooth_ε`, `smooth_scalar` | `smooth_eps`, `smooth_scalar` |
| `sliceinv_3x3(selectdim(sm, 3, 1))` … | `inv_eps_slices(sm)` → `(eps_inv, deps, ddeps)` |
| `KrylovKitEigsolve()`, `GPUSolver(Float32)`, `MPBSolver()` | `KrylovKitEigsolve()`, `GPUSolver("f32", device=…)`, `MPBSolver()` |
| `solve_k(ω, ε⁻¹, grid, solver; nev, k_tol, …)` | `solve_k(omega, eps_inv, grid, solver=None, nev=1, **kw)` |
| `solve_ω²` | `solve_omega2` |
| `group_index`, `ng_gvd` | same names |
| `E⃗(k, ev, ε⁻¹, ∂ε_∂ω, grid; …)` | `E_field(k, evec, eps_inv, deps, grid, …)` |
| `E_relpower_xyz`, `count_E_nodes`, `mode_viable`, `𝓐`/`effective_area` | `rel_power_xyz`, `count_E_nodes`, `mode_viable`, `effective_area` |
| `poynting_z`, `mode_intensity`, `kerr_dielectric_perturbation`, `solve_k_kerr` | same names (`solve_k_kerr` returns a dict) |
| `param_grid(ω=…, …)`, `SlurmConfig`, `deploy_batch`, `frequency_sweep`, `load_batch`, `batch_status`, `gather_batch`, `load_summary`, `load_fields`, `run_task` | same names (Greek keywords like `ω` work in Python; rows are lists of dicts — `pandas.DataFrame(rows)` works) |

Conventions: NumPy arrays cross the boundary with index semantics preserved
(`eps_inv[i, j, ix, iy]` in Python is `ε⁻¹[i, j, ix, iy]` in Julia); eigenvectors are
1-D `complex128` arrays of length `2·Nx·Ny`; material objects, solver objects,
`Grid`s and `Batch`es are opaque handles. Units are identical to the Julia packages
(μm, `ω = 1/λ`, W, μm²/W).

## End-to-end example

```python
import numpy as np
import optimode as om

lam = 1.55; omega = 1/lam
grid = om.Grid(4.0, 3.0, 96, 72)

# materials → smoothed tensors (Kottke sub-pixel smoothing, exact ∂ω/∂ω² propagation)
mat_vals = om.f_eps_mats([om.Si3N4, om.SiO2])([omega])
core = om.box([0.0, 0.0], [1.60, 0.80], 1)
sm = om.smooth_eps([core], mat_vals, (1, 2), grid)
eps_inv, deps, ddeps = om.inv_eps_slices(sm)

# eigenmodes at fixed frequency (Newton-inverted dispersion relation)
kmags, evecs = om.solve_k(omega, eps_inv, grid, nev=2, k_tol=1e-10)
neff = kmags / omega

# dispersion & mode character
ng, gvd = om.ng_gvd(omega, kmags[0], evecs[0], eps_inv, deps, ddeps, grid)
E = om.E_field(kmags[0], evecs[0], eps_inv, deps, grid)
pol = om.rel_power_xyz(np.ascontiguousarray(sm[:, :, 0]), E)   # quasi-TE/TM

# Kerr power dependence (library n₂ values; SPM-validated)
n2_map = om.smooth_scalar([core], [om.kerr_n2(om.Si3N4), om.kerr_n2(om.SiO2)], (1, 2), grid)
res = om.solve_k_kerr(omega, 5.0, eps_inv, deps, n2_map, grid)         # P = 5 W
dneff = (res["kmags"][0] - res["kmags_lin"][0]) / omega
I = om.mode_intensity(kmags[0], evecs[0], eps_inv, grid, 5.0)          # ∫I dA = 5 W
Aeff = 5.0**2 / ((I**2).sum() * grid.dV)                               # ≈ 1 μm²
```

## Batched sweeps from Python

Sweeps deploy exactly as in Julia — the setup script stays a Julia file (it runs on
the workers), everything else is Python:

```python
import optimode as om

params = om.param_grid(ω=[1/1.6, 1/1.55, 1/1.5], w_top=[1.4, 1.7], P=0.0)
batch = om.deploy_batch("ridge_wg_setup.jl", params,
                        name="sweep", nev=2, backend="slurm",
                        slurm=om.SlurmConfig(time="0:30:00", max_concurrent=50))

batch.status()                       # {'total': 6, 'done': …, 'failed': …, 'pending': …}
rows = batch.gather()                # list of dicts; partial results OK while running
# import pandas as pd; df = pd.DataFrame(rows)

# later, in a new Python session:
batch = om.load_batch("modesweeps_sweep")
rows = om.gather_batch(batch)
fields = om.load_fields(batch, 3)    # full E-fields/eigenvectors (save_fields=True)
```

`backend="local"` runs the same workers as local processes (used by the test suite);
`om.frequency_sweep("setup.jl", ω=…, …)` is the frequency-sweep sugar. Kerr power
sweeps work by including `P` in the parameters and returning an `n₂` map from
`make_problem` (see [`examples/kerr_power_sweep_setup.jl`](../examples/kerr_power_sweep_setup.jl)).

## GPU and MPB backends

```python
kmags, evecs = om.solve_k(omega, eps_inv, grid, om.GPUSolver("f32", device="cuda"))
kmags, evecs = om.solve_k(omega, eps_inv, grid, om.MPBSolver())   # needs pymeep
```

Both run through the same `solve_k` interface; see
[Maxwell eigenmodes](maxwell_eigenmodes.md) for backend details.

## Notes & troubleshooting

- **First import is slow**: it boots Julia and loads/precompiles the packages.
  Subsequent imports reuse caches (seconds).
- **`InitError: could not load library …libssl.so`**: the wrapper preloads Julia's
  OpenSSL automatically; if your application imports `ssl` before `optimode`, import
  `optimode` first or `LD_PRELOAD` the artifact `libcrypto.so`/`libssl.so`.
- **Gradients**: the AD interfaces (Zygote/Enzyme/Mooncake) are Julia-side; from
  Python, call them through `om.julia()` (the raw JuliaCall `Main` module) — e.g.
  `om.julia().seval("om -> Zygote.gradient(...)")`. A NumPy-facing gradient API is a
  natural extension point.
