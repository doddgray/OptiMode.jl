# OptiMode.jl

A prototype differentiable eigensolver for the electromagnetic Helmholtz equation.

The mathematical model implemented in this code is a re-implementation of the plane-wave, transverse-polarization basis electromagnetic mode solver [MIT Photonic Bands (MPB)](https://mpb.readthedocs.io/en/latest/) [[1]](#1) and is also described in our paper [[2]](#2).
This package extends the functionality of MPB by implementing a "pull-back" (vector-Jacobian product) function to back-propagate gradients with respect to electromagnetic mode fields and spatial propagation constants to the parameters determining the modes, namely the dielectric tensor elements at each spatial grid point and the (temporal) frequency of the electromagnetic modes.
The "pull-back" function works by iteratively solving the adjoint equations for the electromagnetic Helmholtz eigen-problem.
As demonstrated in our paper, these gradients can be further back-propagated to parameters defining an optical waveguide geometry and used to optimize a waveguide for some desired modal properties.

## Documentation

Component-level documentation — with the physics and mathematics of each stage
(typeset equations, diagrams) and usage examples — lives in [`docs/`](docs):
[overview & units](docs/README.md) ·
[material dispersion](docs/material_dispersion.md) ·
[dielectric smoothing](docs/dielectric_smoothing.md) ·
[Maxwell eigenmodes](docs/maxwell_eigenmodes.md) ·
[mode analysis](docs/mode_analysis.md) ·
[mode sweeps](docs/mode_sweeps.md) ·
[automatic differentiation](docs/automatic_differentiation.md).
Function-level reference documentation is in docstrings (`?solve_k`, `?smooth_ε`, …
in the REPL). Runnable examples live in [`examples/`](examples).

A **Python interface** exposing the same pipeline with NumPy-native APIs lives in
[`python/`](python) (package `optimode`, via JuliaCall); see
[docs/python.md](docs/python.md) and
[`python/examples/`](python/examples).

## Package structure

OptiMode is organized as a monorepo of five component packages living in `lib/`, with
the top-level `OptiMode` module acting as a thin umbrella that re-exports all of them:

| Package | Purpose |
|---|---|
| [`MaterialDispersion`](lib/MaterialDispersion) | Symbolic dielectric material dispersion models (Sellmeier, thermo-optic, χ⁽²⁾, Kerr `n₂`), a material library (LiNbO₃, Si₃N₄, SiO₂, Si, Ge, …), and fast generated functions for ε(ω,T) and its frequency derivatives. |
| [`DielectricSmoothing`](lib/DielectricSmoothing) | Finite-difference spatial `Grid` types and sub-pixel ("Kottke") smoothing of dielectric tensors across material interfaces, mapping geometry + material data to smoothed ε/∂ωε/∂²ωε arrays. |
| [`MaxwellEigenmodes`](lib/MaxwellEigenmodes) | The plane-wave Helmholtz operator and iterative eigensolvers (`solve_ω²`, `solve_k`) operating on smoothed dielectric tensor data, with adjoint-method gradient rules. Includes an optional [MPB](https://mpb.readthedocs.io) backend (`MPBSolver`, Python `meep.mpb` via PythonCall.jl) and a CUDA-GPU-capable, Float32/Float64 backend (`GPUSolver`) with a device-resident adjoint. |
| [`ModeAnalysis`](lib/ModeAnalysis) | Post-processing of mode-solver results: group index, group velocity dispersion (`group_index`, `ng_gvd`), field reconstruction helpers, mode classification/filtering, and first-order Kerr (intensity-dependent index) mode corrections (`solve_k_kerr`). |
| [`ModeSweeps`](lib/ModeSweeps) | Batched/asynchronous deployment of mode simulations as SLURM array jobs (or local processes): parameter grids & frequency sweeps, persistent batch state, live status, partial gathering, summary-vs-full-field transfer, and tabular (CSV/TSV/JSON) result I/O. |

The dependency chain is `MaterialDispersion` ← `DielectricSmoothing` ← `MaxwellEigenmodes` ← `ModeAnalysis` (← `ModeSweeps`).
A typical calculation flows the same way:

```julia
using OptiMode   # re-exports all four packages
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid

# 1. material dispersion: generated function ω ↦ (ε, ∂ωε, ∂²ωε) for each material
mats = [Si₃N₄, SiO₂]
f_ε, _ = _f_ε_mats(mats, (:ω,))
ω = 1/1.55                                       # frequency in μm⁻¹
mat_vals = hcat(f_ε([ω]), vcat(vec([1.0 0 0; 0 1.0 0; 0 0 1.0]), zeros(18)))  # + vacuum

# 2. smooth dielectric tensors onto a spatial grid
grid   = Grid(6.0, 4.0, 128, 96)
core   = MaterialShape(Cuboid([0.0,0.0], [1.6,0.7], [1.0 0.0; 0.0 1.0]), 1)  # data = material index
sm     = smooth_ε((core,), mat_vals, (1,2), grid)            # (3,3,3,Nx,Ny): ε, ∂ωε, ∂²ωε
ε⁻¹    = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ε_∂ω  = copy(selectdim(sm, 3, 2))

# 3. solve for guided eigenmodes at frequency ω
kmags, evecs = solve_k(ω, ε⁻¹, grid, KrylovKitEigsolve(); nev=2)

# 4. post-process: effective & group index
neff = kmags[1]/ω
ng   = group_index(kmags[1], evecs[1], ω, ε⁻¹, ∂ε_∂ω, grid)
```

For modal group-velocity dispersion from a single mode solution and a worked
reproduction of the waveguide-dispersion design problem in
[[2]](#2) — group-index matching of the fundamental and second-harmonic
quasi-TE00 modes of a thin-film lithium niobate waveguide for broadband second-harmonic
generation — see [`examples/tfln_shg_dispersion.jl`](examples/tfln_shg_dispersion.jl).

## Automatic differentiation

Gradient support is provided at several levels:

- **ChainRules**: hand-written adjoint-method `rrule`s for the eigensolves (`solve_k`),
  the adjoint eigen-solver (`eig_adjt`), k-space basis fields (`mag_mn`), and
  post-processing (`group_index`), consumed directly by Zygote.
- **Mooncake.jl** (reverse mode): each package ships a `…MooncakeExt` extension that
  bridges these rules with `Mooncake.@from_rrule` and marks discrete bookkeeping
  functions zero-derivative. Pure generated code (dispersion functions, Kottke
  smoothing kernels) differentiates natively.
- **Enzyme.jl** (forward + reverse): each package ships an `…EnzymeExt` extension that
  imports the same rules with `Enzyme.@import_rrule` and marks discrete bookkeeping
  inactive. Generated scalar code differentiates natively in both modes. (The imported
  custom rules apply to positional calls; keyword calls lower to `Core.kwcall`, which
  `@import_rrule` does not cover.)
- **ForwardDiff**: forward mode works through the whole smoothing and post-processing
  stack (including FFTs, via AbstractFFTs' ForwardDiff extension).
- **Reactant.jl**: `MaterialDispersion.reactant_compile_dispersion` compiles generated
  dispersion functions to XLA via a `Reactant` package extension (the FFTW-planned
  eigensolver pipeline is not currently Reactant-traceable).

Gradient correctness is tested against `FiniteDifferences.jl` (and, where available,
exact symbolic Jacobians) in each package's test suite; see
`lib/*/test/runtests.jl`. Gradient efficiency benchmarks comparing primal vs.
gradient evaluation times for each backend live in `lib/*/benchmark/benchmarks.jl`.
Representative numbers from a 4-core CI-class container:

| objective | primal | Zygote | Enzyme (rev) | Enzyme (fwd) | Mooncake |
|---|---|---|---|---|---|
| `(ε,∂ωε,∂²ωε)(ω,T)`, 3 materials | 32 μs | — | 6.6× | 2.9× | 24× |
| Kottke kernel (single voxel) | 95 ns | — | 2.3× | — | 22× |
| `smooth_ε` (128×128 grid) | 1.2 s | 5.5× | — | — | — |
| `solve_k` (64×64 grid) | 2.1 s | 1.0× | 3.0× | — | 5.2× |
| `group_index` (64×64 grid) | 3.5 ms | 3.0× | 5.6× | — | 9.5× |

(× columns are gradient time relative to the primal; the `solve_k` gradient costing
≈1× the primal eigensolve is the expected behavior of the adjoint method.)

Known limitations:

- Whole-pipeline reverse mode through `smooth_ε`'s 768-voxel `mapreduce` is supported
  via Zygote; Mooncake/Enzyme cover the per-voxel Kottke kernels (compiling their
  reverse rules for the full pipeline takes impractically long).
- Geometry-*parameter* gradients (widths, thicknesses, sidewall angles, positions) are
  supported via the [`claude/geometry-gradient-ad-no6zct`](https://github.com/doddgray/GeometryPrimitives.jl/tree/claude/geometry-gradient-ad-no6zct)
  branch of `doddgray/GeometryPrimitives.jl` (parametric shape eltype, AD-compatible
  `surfpt_nearby`/`volfrac`): forward mode (ForwardDiff) through the full
  geometry→smoothing pipeline, and reverse mode (Mooncake) at the per-interface-pixel
  Kottke kernel. (Enzyme segfaults on the StaticArrays inverse in Cuboid
  `surfpt_nearby`; Zygote hits a non-`SVector` normal in `volfrac`.)
- Sensitivities of the mode quantities (effective index, group index, GVD, mode fields)
  w.r.t. geometry are obtained by composing forward-mode geometry→dielectric Jacobians
  with the reverse-mode (adjoint) eigensolve — the standard inverse-design pattern —
  verified against finite differences in the umbrella `geometry-parameter sensitivities`
  testset (`test/runtests.jl`).

```bash
# run tests for a component package
julia --project=lib/MaterialDispersion -e 'using Pkg; Pkg.test()'

# run gradient benchmarks for a component package
julia --project=lib/MaterialDispersion/benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=lib/MaterialDispersion/benchmark lib/MaterialDispersion/benchmark/benchmarks.jl
```

### Cluster sweeps (SLURM)

`ModeSweeps` deploys batched mode simulations asynchronously — e.g. parallelized
frequency sweeps combined with geometry/material parameter sweeps — as SLURM array
jobs on a cluster with the same packages installed:

```julia
batch = frequency_sweep("ridge_wg_setup.jl"; ω=0.55:0.005:0.75, w_top=[1.4,1.7,2.0],
                        nev=2, slurm=SlurmConfig(time="0:30:00", max_concurrent=50))
batch_status(batch)                      # live status while running (works via squeue too)
rows = gather_batch(batch)               # partial results OK; per-band neff, ng, GVD,
                                         # Aeff & polarization; writes summary.{csv,tsv,json}
rows = load_summary(".../summary.csv")   # reload anytime for analysis
```

Batch state is persisted at deployment, so status/gathering also work from new Julia
sessions; workers optionally store full mode-field data (HDF5) instead of only the
summary table. See [`lib/ModeSweeps`](lib/ModeSweeps) for details.

### Kerr nonlinearity (power-dependent modes)

Materials can carry an intensity-dependent refractive-index coefficient `n₂`
(μm²/W) — a constant or a symbolic function of wavelength — under the `:n₂` model key.
The library ships standard values for Si₃N₄ (`2.4e-7`) and SiO₂ (`2.6e-8`); materials
without an `:n₂` model are linear (`kerr_n2(mat) == 0`):

```julia
kerr_n2(Si₃N₄)                       # 2.4e-7 μm²/W at the default λ = 1.55 μm
m = with_kerr_n2(SiO₂, 2.0e-8 + 1.0e-8*λ^2)   # custom / wavelength-dependent model
```

`solve_k_kerr` applies a first-order power correction to each mode: the modal
intensity profile `I(x,y)` (z-Poynting flux, normalized so ∫I dA equals a specified
optical power `P` in W, assumed to reside entirely in that mode — no cross coupling)
induces `Δn = n₂(x,y)·I(x,y)`, and the mode is re-solved with the perturbed dielectric
tensor. The per-material `n₂` values are mapped onto the grid with the same sub-pixel
volume-fraction smoothing as the dielectric data:

```julia
n2_map = smooth_scalar(shapes, kerr_n2.(mats, λ), minds, grid)   # n₂(x,y) [μm²/W]
res = solve_k_kerr(ω, P, ε⁻¹, ∂ε_∂ω, n2_map, grid, KrylovKitEigsolve(); nev=1)
Δneff = (res.kmags[1] - res.kmags_lin[1]) / ω    # power-dependent index shift
```

For a 1.60×0.80 μm Si₃N₄ waveguide in SiO₂ at 1.55 μm this reproduces the textbook
self-phase-modulation estimate `Δneff ≈ n₂P/Aeff` to a few percent (γ ≈ 0.95 W⁻¹m⁻¹);
see [`examples/kerr_si3n4_waveguide.jl`](examples/kerr_si3n4_waveguide.jl). Power
sweeps deploy as `ModeSweeps` batches like any other parameter: if `make_problem`
returns an `n₂` map, parameter sets containing a power `P` are solved with the Kerr
correction and the gathered rows include `dneff_kerr` and `dn_max` columns
([`examples/kerr_power_sweep_setup.jl`](examples/kerr_power_sweep_setup.jl)).

### GPU backend

`MaxwellEigenmodes.GPUSolver` is a device- and precision-generic eigensolver backend:

```julia
using CUDA                                    # activates the extension (NVIDIA GPUs)
kmags, evecs = solve_k(ω, ε⁻¹, grid, GPUSolver(Float32); nev=2)          # GPU, single precision
kmags, evecs = solve_k(ω, ε⁻¹, grid, GPUSolver(Float64; device=:cpu))    # same code on CPU
```

The solver core uses only backend-agnostic operations (broadcast kernels, AbstractFFTs
plans — FFTW on `Array`, CUFFT on `CuArray` — and KrylovKit), so one code path serves
both devices and both precisions; inputs/outputs stay host `Float64` for pipeline
interoperability. The adjoint for `solve_k` is implemented in the same device-generic
style (`KrylovKit.linsolve` for the adjoint linear solve plus broadcast accumulation of
ε̄⁻¹ and k̄), so gradient back-propagation through GPU-accelerated solves also runs on
the GPU. Correctness vs. the native solver and vs. finite differences is tested at both
precisions on the CPU path in every test run; CUDA-device tests are opt-in via
`OPTIMODE_TEST_CUDA=true`. `benchmark/scaling.jl` benchmarks `solve_k` and its adjoint
gradient across backends as a function of grid size. CPU-path results from the same
4-core container (solve / adjoint-gradient seconds; GPU columns appear when run on a
machine with a functional CUDA device):

| grid | KrylovKit F64 | GPUSolver F64 (cpu) | GPUSolver F32 (cpu) | max |k| rel. dev. |
|---|---|---|---|---|
| 32×32 | 0.20 / 0.17 | 0.11 / 0.09 | 0.030 / 0.024 | 3×10⁻⁸ |
| 64×64 | 1.8 / 1.8 | 1.1 / 1.1 | 0.35 / 0.37 | 1×10⁻⁶ |
| 128×128 | 11 / 12 | 7.0 / 7.2 | 3.4 / 3.5 | 1×10⁻⁷ |
| 256×256 | 125 / 126 | 80 / 79 | 48 / 38 | 7×10⁻⁶ |

Even on the CPU the device-generic backend outpaces the legacy path (warm-started
Newton iterations save eigensolves), and Float32 roughly halves runtimes again; the
adjoint gradient costs ≈1× the primal solve at every size.

### MPB backend

`MaxwellEigenmodes.MPBSolver` runs the eigensolves with [MPB](https://mpb.readthedocs.io)
through the Python `meep.mpb` module, replacing the legacy PyCall bindings with a
PythonCall.jl package extension:

```julia
using PythonCall                      # activates the extension
# one-time Python setup: using CondaPkg; CondaPkg.add("pymeep")
kmags, evecs = solve_k(ω, ε⁻¹, grid, MPBSolver(); nev=2)
```

The smoothed dielectric tensors are passed to MPB as a material function (no files or
Python-side interpolation), so MPB and the native solvers share one discretization and
their results agree to solver tolerance; the solver-generic adjoint `rrule` makes
`solve_k` with the MPB backend differentiable as well. MPB tests are opt-in via
`OPTIMODE_TEST_MPB=true`.

Pre-refactor code that is not currently maintained (the old PyCall MPB bindings, HDF5
sweep I/O, old monolithic tests) is preserved under `legacy/`.

If you find this solver useful in your own research please consider citing our paper [[2]](#2) and the original MPB paper [[1]](#1).
If you find this solver broken or buggy please post an issue so that we can try to fix and improve it.
Good luck and happy mode solving.

## References
<a id="1">[1]</a>
S. G. Johnson and J. D. Joannopoulos, "Block-iterative frequency-domain methods for Maxwell’s equations in a planewave basis," [Optics Express 8, 173-190 (2001)](https://doi.org/10.1364/OE.8.000173)

<a id="2">[2]</a>
D. Gray, G. N. West, and R. J. Ram, "Inverse design for waveguide dispersion with a differentiable mode solver," [Optics Express 32, 30541-30554 (2024)](https://doi.org/10.1364/OE.530479)
