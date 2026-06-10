# Enzyme.jl interface for ModeAnalysis.
#
# `group_index` composes FFTs and Tullio tensor contractions; its reverse rule is defined
# as a ChainRules rrule in `src/ad_rules.jl` and imported into Enzyme here.

module ModeAnalysisEnzymeExt

using ModeAnalysis
using ModeAnalysis: group_index
using DielectricSmoothing: Grid
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(ModeAnalysis.count_E_nodes), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(ModeAnalysis.mode_idx), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(ModeAnalysis.mode_viable), args...; kwargs...) = nothing

for (TGrid, TEps) in ((Grid{2,Float64}, Array{Float64,4}), (Grid{3,Float64}, Array{Float64,5}))
    @eval Enzyme.@import_rrule(typeof(group_index), Float64, Array{ComplexF64,1}, Float64, $TEps, $TEps, $TGrid)
end

end # module
