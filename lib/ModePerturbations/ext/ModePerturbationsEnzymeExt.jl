# Enzyme.jl interface for ModePerturbations.
#
# The perturbation kernel `_perturbation_scalar` composes FFTs and the `HMₖH` quadratic
# form; its reverse rule is a ChainRules `rrule` and its forward rule a `frule` (computed
# with ForwardDiff because Enzyme forward cannot create FFTW plans). Both are imported into
# Enzyme here via `Enzyme.@import_rrule`/`@import_frule` (at module top level, during
# precompilation, like `ModeAnalysisEnzymeExt`). The purely-arithmetic exports
# (`payne_lacey_slab_loss`, `substrate_leakage_loss`, `cascaded_chi2_n2_eff`,
# `kerr_gamma`, …) differentiate natively in Enzyme (forward & reverse) with no rule.
#
# NB: the imported rules apply to positional calls of `_perturbation_scalar`.

module ModePerturbationsEnzymeExt

using ModePerturbations
using ModePerturbations: _perturbation_re, _perturbation_im
using DielectricSmoothing: Grid
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules

# `Enzyme.@import_rrule`/`@import_frule` are provided by Enzyme's ChainRulesCore extension,
# which is not always loaded while *this* package extension precompiles. Ensure it is, then
# import the rules; the `try` keeps a missing-extension corner case from breaking
# precompilation (ForwardDiff/Zygote still provide forward/reverse AD in that case).
Base.retry_load_extensions()

# reverse (rrule) and forward (frule) rules of both perturbation kernels, for 2D & 3D grids
# and for both real (index) and complex (absorptive/loss) perturbation fields Δε.
for (TGrid, NEps) in ((:(Grid{2,Float64}), 4), (:(Grid{3,Float64}), 5))
    TEps = :(Array{Float64,$NEps})
    for TΔε in (:(Array{Float64,$NEps}), :(Array{ComplexF64,$NEps}))
        for kernel in (:_perturbation_re, :_perturbation_im)
            try
                @eval Enzyme.@import_rrule(typeof($kernel), Float64, Array{ComplexF64,1},
                    $TEps, $TΔε, $TGrid)
                @eval Enzyme.@import_frule(typeof($kernel), Float64, Array{ComplexF64,1},
                    $TEps, $TΔε, $TGrid)
            catch err
                @warn "ModePerturbations: Enzyme rule import skipped for $kernel" exception = err
            end
        end
    end
end

end # module
