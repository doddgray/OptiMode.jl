# ModeAnalysis.jl

Post-processing of electromagnetic eigenmode solver results for the OptiMode photonics
tool suite.

- `group_index(k, evec, ω, ε⁻¹, ∂ε_∂ω, grid)`: modal group index from a single mode
  solution.
- `ng_gvd` / `ng_gvd_E`: analytic adjoint-based group index *and* group velocity
  dispersion (plus the real-space E-field) from a single mode solution.
- Mode classification and filtering: `E_relpower_xyz`, `count_E_nodes`, `mode_viable`,
  `mode_idx`, effective area `𝓐`.
- First-order Kerr (intensity-dependent index) mode corrections: `solve_k_kerr(ω, P,
  ε⁻¹, ∂ε_∂ω, n₂, grid, solver)` re-solves each mode with the dielectric perturbation
  `Δε = 2n₀·n₂(x,y)·I(x,y)` induced by its own intensity profile at optical power `P`
  (W, all assumed in that mode); building blocks `poynting_z`, `mode_intensity` and
  `kerr_dielectric_perturbation` are exported separately.

## AD interfaces

`group_index` composes FFTs and Tullio tensor contractions; its reverse rule is
assembled once with Zygote and exposed as a ChainRules `rrule`, which package
extensions bridge to Mooncake (`@from_rrule`) and Enzyme (`@import_rrule`). Forward
mode works with ForwardDiff through AbstractFFTs' Dual support. Gradient correctness is
verified in `test/runtests.jl` against FiniteDifferences.jl and against the analytic
GVD from `ng_gvd`. Benchmarks: `benchmark/benchmarks.jl`.

Full documentation with the underlying physics/mathematics and usage examples:
[`docs/mode_analysis.md`](../../docs/mode_analysis.md) at the repository root.
