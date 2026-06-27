# Enzyme.jl interface for ModeAnalysis.
#
# `group_index` composes FFTs and Tullio tensor contractions; its reverse rule is a
# ChainRules `rrule` (`src/ad_rules.jl`) and its forward rule a `frule` (computed with
# ForwardDiff because Enzyme forward cannot create FFTW plans). Both are imported into
# Enzyme here via the vendored `@vendored_import_rrule`/`@vendored_import_frule` generators
# (Enzyme's own importer code, defined locally so it is always available — Enzyme's
# `EnzymeChainRulesCoreExt`, which provides the upstream `@import_*` macros, is not reliably
# loaded while this package extension precompiles; see `_enzyme_chainrules_import.jl`).
#
# The macro calls are made at module top level (during precompilation, so the generated
# EnzymeRules methods are cached).
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
# The vendored generators register the rules regardless of extension load order; the
# concrete types are spliced into each macrocall with `@eval` (loop variables are not
# visible to a once-expanded macro).
include("_enzyme_chainrules_import.jl")

for (TGrid, TEps) in ((:(Grid{2,Float64}), :(Array{Float64,4})), (:(Grid{3,Float64}), :(Array{Float64,5})))
    @eval @vendored_import_rrule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)
    @eval @vendored_import_frule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)
end

end # module
