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
- `smooth_scalar(shapes, vals, minds, grid)` maps per-material scalars (e.g. Kerr
  `n₂` coefficients) to a volume-fraction-averaged scalar field on the same grid.

## AD interfaces

The smoothing kernels are generated scalar code and differentiate natively with
ForwardDiff, Enzyme, and Mooncake; shape-index bookkeeping (`corner_sinds`,
`proc_sinds`) is marked non-differentiable for all backends (ChainRules
`@non_differentiable`, `Mooncake.@zero_adjoint`, `EnzymeRules.inactive`).

Gradients of the full geometry→smoothing pipeline w.r.t. the material tensor data are
supported in forward mode (ForwardDiff) and reverse mode (Zygote via ChainRules);
the per-voxel Kottke kernels are additionally covered by Enzyme and Mooncake.

Geometry-*parameter* sensitivities (widths, thicknesses, sidewall angles, positions)
are supported with GeometryPrimitives ≥ 0.6, whose parametric shape element type and
AD-compatible `surfpt_nearby`/`volfrac` let AD number types flow through shape
construction and the interface queries: forward mode (ForwardDiff) through the whole
pipeline, and reverse mode (Mooncake) at the per-interface-pixel kernel. Enzyme
segfaults on the StaticArrays inverse inside Cuboid `surfpt_nearby` and Zygote receives
a non-`SVector` normal in `volfrac`, so geometry-parameter reverse mode uses Mooncake.
The geometry queries remain `@non_differentiable` for ChainRules so that material-data
Zygote gradients treat them as constants (ForwardDiff and Mooncake bypass ChainRules).

Gradient correctness is verified in `test/runtests.jl` against FiniteDifferences.jl
(and exact symbolic Jacobians for the kernels). Benchmarks: `benchmark/benchmarks.jl`.

Full documentation with the underlying physics/mathematics and usage examples:
[`docs/dielectric_smoothing.md`](../../docs/dielectric_smoothing.md) at the repository root.
