# Enzyme.jl interface for ModeAnalysis.
#
# `group_index` composes FFTs and Tullio tensor contractions; its reverse rule is defined
# as a ChainRules rrule in `src/ad_rules.jl` and imported into Enzyme here via
# `Enzyme._import_rrule` (the engine behind `Enzyme.@import_rrule`).
#
# NB: the imported custom rules apply to positional calls of `group_index`.
#
# `Enzyme._import_rrule` is provided by Enzyme's own ChainRulesCore extension. Extension
# load order is not guaranteed, so if that extension is not loaded yet when this one
# initializes, rule installation is deferred via `Base.package_callbacks`.

module ModeAnalysisEnzymeExt

using ModeAnalysis
using ModeAnalysis: group_index
using DielectricSmoothing: Grid
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(ModeAnalysis.count_E_nodes), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(ModeAnalysis.mode_idx), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(ModeAnalysis.mode_viable), args...; kwargs...) = nothing

const _rules_installed = Ref(false)

function _install_rules()
    _rules_installed[] && return nothing
    _rules_installed[] = true
    # evaluated (and thus macro-expanded) at runtime so hygiene is handled normally
    for (TGrid, TEps) in ((:(Grid{2,Float64}), :(Array{Float64,4})), (:(Grid{3,Float64}), :(Array{Float64,5})))
        # reverse (adjoint `rrule`) and forward (`frule`) rules for group_index
        Core.eval(@__MODULE__,
            :(Enzyme.@import_rrule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)))
        Core.eval(@__MODULE__,
            :(Enzyme.@import_frule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)))
    end
    return nothing
end

function __init__()
    if Base.get_extension(Enzyme, :EnzymeChainRulesCoreExt) !== nothing
        _install_rules()
    else
        # Defer until Enzyme's ChainRulesCore extension is loaded.
        push!(Base.package_callbacks, function (pkgid)
            if pkgid.name == "EnzymeChainRulesCoreExt" && !_rules_installed[]
                _install_rules()
            end
            return nothing
        end)
    end
end

end # module
