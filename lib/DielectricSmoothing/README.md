# DielectricSmoothing.jl

Sub-pixel ("Kottke") smoothing of dielectric tensors on finite-difference spatial grids
for the OptiMode differentiable photonics tool suite.

- `Grid{2}`/`Grid{3}` spatial grid types with pixel/voxel center & corner iterators and
  reciprocal-lattice utilities.
- Kottke interface-averaging kernels (`avg_param`, `εₑ_∂ωεₑ_∂²ωεₑ`, …) generated
  symbolically once at package load, including exact Jacobians/Hessians for propagating
  material dispersion derivatives through the smoothing.
- `smooth_ε(shapes, mat_vals, minds, grid)` maps a GeometryPrimitives.jl shape list +
  per-material `(ε, ∂ωε, ∂²ωε)` data to smoothed tensor fields on the grid.

## AD interfaces

The smoothing kernels are generated scalar code and differentiate natively with
ForwardDiff, Enzyme, and Mooncake; shape-index bookkeeping (`corner_sinds`,
`proc_sinds`) is marked non-differentiable for all backends (ChainRules
`@non_differentiable`, `Mooncake.@zero_adjoint`, `EnzymeRules.inactive`).

Gradient correctness (w.r.t. material tensors and geometry parameters) is verified in
`test/runtests.jl` against FiniteDifferences.jl. Benchmarks: `benchmark/benchmarks.jl`.
