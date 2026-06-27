# Enzyme.jl interface for ModePerturbations.
#
# The perturbation kernel `_perturbation_scalar` composes FFTs and the `HMₖH` quadratic
# form; its reverse rule is a ChainRules `rrule` and its forward rule a `frule` (computed
# with ForwardDiff because Enzyme forward cannot create FFTW plans). Both are imported into
# Enzyme here via the vendored `@vendored_import_rrule`/`@vendored_import_frule` generators
# (Enzyme's own importer code, defined locally so it is always available regardless of
# extension load order; see `_enzyme_chainrules_import.jl`), at module top level during
# precompilation. The purely-arithmetic exports (`payne_lacey_slab_loss`,
# `substrate_leakage_loss`, `cascaded_chi2_n2_eff`, `kerr_gamma`, …) differentiate natively
# in Enzyme (forward & reverse) with no rule.
#
# NB: the imported rules apply to positional calls of `_perturbation_scalar`.

module ModePerturbationsEnzymeExt

using ModePerturbations
using ModePerturbations: _perturbation_re, _perturbation_im
using DielectricSmoothing: Grid
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules

# The vendored `@vendored_import_rrule`/`@vendored_import_frule` generators register the
# rules regardless of extension load order (Enzyme's `EnzymeChainRulesCoreExt`, which
# provides the upstream `@import_*` macros, is not reliably loaded while this package
# extension precompiles; see `_enzyme_chainrules_import.jl`).
include("_enzyme_chainrules_import.jl")

# reverse (rrule) and forward (frule) rules of both perturbation kernels, for 2D & 3D grids
# and for both real (index) and complex (absorptive/loss) perturbation fields Δε.
for (TGrid, NEps) in ((:(Grid{2,Float64}), 4), (:(Grid{3,Float64}), 5))
    TEps = :(Array{Float64,$NEps})
    for TΔε in (:(Array{Float64,$NEps}), :(Array{ComplexF64,$NEps}))
        for kernel in (:_perturbation_re, :_perturbation_im)
            @eval @vendored_import_rrule(typeof($kernel), Float64, Array{ComplexF64,1},
                $TEps, $TΔε, $TGrid)
            @eval @vendored_import_frule(typeof($kernel), Float64, Array{ComplexF64,1},
                $TEps, $TΔε, $TGrid)
        end
    end
end

end # module
