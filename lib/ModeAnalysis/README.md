# ModeAnalysis.jl

Post-processing of electromagnetic eigenmode solver results for the OptiMode photonics
tool suite.

- `group_index(k, evec, ω, ε⁻¹, ∂ε_∂ω, grid)`: modal group index from a single mode
  solution.
- `ng_gvd` / `ng_gvd_E`: analytic adjoint-based group index *and* group velocity
  dispersion (plus the real-space E-field) from a single mode solution.
- Mode classification and filtering: `E_relpower_xyz`, `count_E_nodes`, `mode_viable`,
  `mode_idx`, effective area `𝓐`.

## AD interfaces

`group_index` composes FFTs and Tullio tensor contractions; its reverse rule is
assembled once with Zygote and exposed as a ChainRules `rrule`, which package
extensions bridge to Mooncake (`@from_rrule`) and Enzyme (`@import_rrule`). Forward
mode works with ForwardDiff through AbstractFFTs' Dual support. Gradient correctness is
verified in `test/runtests.jl` against FiniteDifferences.jl and against the analytic
GVD from `ng_gvd`. Benchmarks: `benchmark/benchmarks.jl`.
