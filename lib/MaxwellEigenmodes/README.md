# MaxwellEigenmodes.jl

Differentiable plane-wave electromagnetic eigenmode solver for the OptiMode photonics
tool suite, operating on smoothed dielectric tensor data on finite-difference grids
(following the MPB formulation).

- `HelmholtzMap`: matrix-free Maxwell operator `[(k+g)×] ε⁻¹ [(k+g)×]` in a transverse
  plane-wave basis with FFTW-planned transforms and a cheap preconditioner.
- `solve_ω²`: iterative eigensolves with KrylovKit, IterativeSolvers LOBPCG, or a
  vendored DFTK-style LOBPCG.
- `solve_k`: Newton solve for the propagation constant `|k|(ω)` of guided modes.
- `MPBSolver`: an [MPB](https://mpb.readthedocs.io) backend driven through the Python
  `meep.mpb` module via PythonCall.jl (a package extension; load `PythonCall` and make
  conda-forge `pymeep` importable, e.g. `using CondaPkg; CondaPkg.add("pymeep")`).
  The smoothed dielectric data is fed to MPB as a material function, so MPB and the
  native solvers operate on identical discretizations, and the solver-generic adjoint
  `rrule` makes the MPB backend differentiable too. Enable its tests with
  `OPTIMODE_TEST_MPB=true`.
- `GPUSolver(T; device)`: a device- and precision-generic backend (`T ∈ (Float32,
  Float64)`) built from broadcast-only kernels, AbstractFFTs plans, and KrylovKit, so
  the identical code runs on the CPU (`device=:cpu`, the tested reference path) and on
  NVIDIA GPUs (`device=:cuda` via a CUDA.jl package extension — `using CUDA`). Its
  adjoint (`rrule` for `solve_k`) is implemented with the same device-generic
  operations, so gradient back-propagation through GPU-accelerated mode solves also
  runs on the GPU. Enable CUDA-device tests with `OPTIMODE_TEST_CUDA=true`; grid-size
  scaling benchmarks across backends live in `benchmark/scaling.jl`.
- Adjoint-method gradients: ChainRules `rrule`s for `solve_k`, `eig_adjt` (generic
  iterative eigenpair adjoint), `my_linsolve`, and the k-space basis fields `mag_mn`.

## AD interfaces

Zygote consumes the ChainRules rules directly. Package extensions bridge the same
adjoint rules to Mooncake (`Mooncake.@from_rrule`) and Enzyme (`Enzyme.@import_rrule`),
so `solve_k` can sit inside larger programs differentiated with either engine.

Gradient correctness is verified in `test/runtests.jl` against FiniteDifferences.jl
(eigensolve-based finite differences of `|k|(ω)` and directional derivatives w.r.t.
ε⁻¹). Benchmarks: `benchmark/benchmarks.jl`.

Full documentation with the underlying physics/mathematics and usage examples:
[`docs/maxwell_eigenmodes.md`](../../docs/maxwell_eigenmodes.md) at the repository root.
