# MaterialDispersion.jl

Symbolic dielectric material dispersion models for the OptiMode differentiable photonics
tool suite.

- Material models (Sellmeier, Cauchy, thermo-optic Sellmeier, χ⁽²⁾ tensors) built with
  Symbolics.jl, with a library of common photonic materials (`LiNbO₃`, `MgO_LiNbO₃`,
  `Si₃N₄`, `SiO₂`, `LiB₃O₅`, `Si`, `Ge`, `αAl₂O₃`, …).
- `_f_ε_mats(mats, (:ω,:T))` generates fast numeric functions `p ↦ (ε, ∂ωε, ∂²ωε)` for a
  set of materials; `_fj_ε_mats`/`_fjh_ε_mats` additionally return exact symbolic
  Jacobians/Hessians w.r.t. the parameters.
- Material rotation (`rotate`), Miller's-delta χ⁽²⁾ scaling (`Δₘ`), and group
  index/GVD models (`ng_model`, `gvd_model`, `nn̂g_model`, `nĝvd_model`).
- Kerr (intensity-dependent index) coefficients `n₂` in μm²/W — constant or symbolic
  in wavelength — via `kerr_n2(mat, λ)` / `with_kerr_n2` / `set_kerr_n2!`; standard
  library values for Si₃N₄ (`2.4e-7`) and SiO₂ (`2.6e-8`), and `n₂ = 0` for materials
  without a specified model.

## AD interfaces

The generated dispersion functions are plain Julia code and differentiate natively with
ForwardDiff, Enzyme (forward & reverse), and Mooncake (reverse); Zygote works through
ChainRules. Package extensions (`ext/`) mark the symbolic/code-generation layer
non-differentiable for Mooncake and Enzyme, and `reactant_compile_dispersion` (with
Reactant loaded) compiles generated dispersion functions to XLA.

Gradient correctness is verified in `test/runtests.jl` against the exact symbolic
Jacobians and FiniteDifferences.jl. Benchmarks: `benchmark/benchmarks.jl`.

Full documentation with the underlying physics/mathematics and usage examples:
[`docs/material_dispersion.md`](../../docs/material_dispersion.md) at the repository root.
