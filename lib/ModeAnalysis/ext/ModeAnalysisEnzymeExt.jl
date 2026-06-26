# Enzyme.jl interface for ModeAnalysis.
#
# `group_index` composes FFTs and Tullio tensor contractions; its reverse rule is a
# ChainRules `rrule` (`src/ad_rules.jl`) and its forward rule a `frule` (computed with
# ForwardDiff because Enzyme forward cannot create FFTW plans). Both are imported into
# Enzyme here via `Enzyme.@import_rrule`/`@import_frule`.
#
# The macro calls are made at module top level (during precompilation, so the generated
# EnzymeRules methods are cached). This Enzyme extension only loads once Enzyme is present
# and ChainRulesCore is a hard dependency, so Enzyme's `EnzymeChainRulesCoreExt`
# (which provides the `@import_*` machinery) is already loaded. Evaluating the macros from
# `__init__` instead would break incremental compilation on cached loads.
#
# NB: the imported rules apply to positional calls of `group_index`.

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

# reverse (adjoint `rrule`) and forward (`frule`) rules for group_index, for 2D & 3D grids.
# Guarded (with `retry_load_extensions`) because newer Enzyme versions provide the
# `@import_rrule` machinery from Enzyme's ChainRulesCore extension, which is not always
# loaded while this extension precompiles; on failure ForwardDiff/Zygote still differentiate
# group_index.
Base.retry_load_extensions()
try
    for (TGrid, TEps) in ((:(Grid{2,Float64}), :(Array{Float64,4})), (:(Grid{3,Float64}), :(Array{Float64,5})))
        @eval Enzyme.@import_rrule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)
        @eval Enzyme.@import_frule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)
    end
catch err
    @warn "ModeAnalysis: Enzyme rule import skipped (Enzyme/ChainRulesCore compat); \
           ForwardDiff/Zygote still provide forward/reverse AD" exception = err
end

end # module
