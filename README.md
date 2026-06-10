# OptiMode.jl

A prototype differentiable eigensolver for the electromagnetic Helmholtz equation.

The mathematical model implemented in this code is a re-implementation of the plane-wave, transverse-polarization basis electromagnetic mode solver [MIT Photonic Bands (MPB)](https://mpb.readthedocs.io/en/latest/) [[1]](#1) and is also described in our paper [[2]](#2).
This package extends the functionality of MPB by implementing a "pull-back" (vector-Jacobian product) function to back-propagate gradients with respect to electromagnetic mode fields and spatial propagation constants to the parameters determining the modes, namely the dielectric tensor elements at each spatial grid point and the (temporal) frequency of the electromagnetic modes.
The "pull-back" function works by iteratively solving the adjoint equations for the electromagnetic Helmholtz eigen-problem.
As demonstrated in our paper, these gradients can be further back-propagated to parameters defining an optical waveguide geometry and used to optimize a waveguide for some desired modal properties.

## Package structure

OptiMode is organized as a monorepo of four component packages living in `lib/`, with
the top-level `OptiMode` module acting as a thin umbrella that re-exports all of them:

| Package | Purpose |
|---|---|
| [`MaterialDispersion`](lib/MaterialDispersion) | Symbolic dielectric material dispersion models (Sellmeier, thermo-optic, χ⁽²⁾), a material library (LiNbO₃, Si₃N₄, SiO₂, Si, Ge, …), and fast generated functions for ε(ω,T) and its frequency derivatives. |
| [`DielectricSmoothing`](lib/DielectricSmoothing) | Finite-difference spatial `Grid` types and sub-pixel ("Kottke") smoothing of dielectric tensors across material interfaces, mapping geometry + material data to smoothed ε/∂ωε/∂²ωε arrays. |
| [`MaxwellEigenmodes`](lib/MaxwellEigenmodes) | The plane-wave Helmholtz operator and iterative eigensolvers (`solve_ω²`, `solve_k`) operating on smoothed dielectric tensor data, with adjoint-method gradient rules. |
| [`ModeAnalysis`](lib/ModeAnalysis) | Post-processing of mode-solver results: group index, group velocity dispersion (`group_index`, `ng_gvd`), field reconstruction helpers, and mode classification/filtering. |

The dependency chain is `MaterialDispersion` ← `DielectricSmoothing` ← `MaxwellEigenmodes` ← `ModeAnalysis`.
A typical calculation flows the same way:

```julia
using OptiMode   # re-exports all four packages
using OptiMode.DielectricSmoothing.GeometryPrimitives: Box

# 1. material dispersion: generated function ω ↦ (ε, ∂ωε, ∂²ωε) for each material
mats = [Si₃N₄, SiO₂]
f_ε, _ = _f_ε_mats(mats, (:ω,))
ω = 1/1.55                                       # frequency in μm⁻¹
mat_vals = hcat(f_ε([ω]), vcat(vec([1.0 0 0; 0 1.0 0; 0 0 1.0]), zeros(18)))  # + vacuum

# 2. smooth dielectric tensors onto a spatial grid
grid   = Grid(6.0, 4.0, 128, 96)
core   = Box([0.0,0.0], [1.6,0.7], [1.0 0.0; 0.0 1.0], 1)   # data = material index
sm     = smooth_ε((core,), mat_vals, (1,2), grid)            # (3,3,3,Nx,Ny): ε, ∂ωε, ∂²ωε
ε⁻¹    = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ε_∂ω  = copy(selectdim(sm, 3, 2))

# 3. solve for guided eigenmodes at frequency ω
kmags, evecs = solve_k(ω, ε⁻¹, grid, KrylovKitEigsolve(); nev=2)

# 4. post-process: effective & group index
neff = kmags[1]/ω
ng   = group_index(kmags[1], evecs[1], ω, ε⁻¹, ∂ε_∂ω, grid)
```

## Automatic differentiation

Gradient support is provided at several levels:

- **ChainRules**: hand-written adjoint-method `rrule`s for the eigensolves (`solve_k`),
  the adjoint eigen-solver (`eig_adjt`), and k-space basis fields (`mag_mn`), consumed
  directly by Zygote.
- **Mooncake.jl** (reverse mode): each package ships a `…MooncakeExt` extension that
  bridges these rules with `Mooncake.@from_rrule` and marks discrete bookkeeping
  functions zero-derivative. Pure generated code (dispersion functions, Kottke
  smoothing kernels) differentiates natively.
- **Enzyme.jl** (forward + reverse): each package ships an `…EnzymeExt` extension that
  imports the same rules with `Enzyme.@import_rrule` and marks discrete bookkeeping
  inactive. Generated scalar code differentiates natively in both modes.
- **ForwardDiff**: forward mode works through the whole post-processing stack
  (including FFTs, via AbstractFFTs' ForwardDiff extension).
- **Reactant.jl**: `MaterialDispersion.reactant_compile_dispersion` compiles generated
  dispersion functions to XLA via a `Reactant` package extension (the FFTW-planned
  eigensolver pipeline is not currently Reactant-traceable).

Gradient correctness is tested against `FiniteDifferences.jl` (and, where available,
exact symbolic Jacobians) in each package's test suite; see
`lib/*/test/runtests.jl`. Gradient efficiency benchmarks comparing primal vs.
gradient evaluation times for each backend live in `lib/*/benchmark/benchmarks.jl`.

```bash
# run tests for a component package
julia --project=lib/MaterialDispersion -e 'using Pkg; Pkg.test()'

# run gradient benchmarks for a component package
julia --project=lib/MaterialDispersion/benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=lib/MaterialDispersion/benchmark lib/MaterialDispersion/benchmark/benchmarks.jl
```

Pre-refactor code that is not currently maintained (MPB/PyCall bindings, HDF5 sweep
I/O, old monolithic tests) is preserved under `legacy/`.

If you find this solver useful in your own research please consider citing our paper [[2]](#2) and the original MPB paper [[1]](#1).
If you find this solver broken or buggy please post an issue so that we can try to fix and improve it.
Good luck and happy mode solving.

## References
<a id="1">[1]</a>
S. G. Johnson and J. D. Joannopoulos, "Block-iterative frequency-domain methods for Maxwell’s equations in a planewave basis," [Optics Express 8, 173-190 (2001)](https://doi.org/10.1364/OE.8.000173)

<a id="2">[2]</a>
D. Gray, G. N. West, and R. J. Ram, "Inverse design for waveguide dispersion with a differentiable mode solver," [Optics Express 32, 30541-30554 (2024)](https://doi.org/10.1364/OE.530479)
