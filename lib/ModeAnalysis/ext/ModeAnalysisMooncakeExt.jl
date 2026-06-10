# Mooncake.jl interface for ModeAnalysis.
#
# `group_index` composes FFTs and Tullio tensor contractions; its reverse rule is defined
# as a ChainRules rrule in `src/ad_rules.jl` and bridged to Mooncake here.

module ModeAnalysisMooncakeExt

using ModeAnalysis
using ModeAnalysis: group_index
using DielectricSmoothing: Grid
using Mooncake
using Mooncake: @from_rrule, MinimalCtx, @zero_adjoint

# Mooncake's @from_rrule bridge lacks a tangent conversion for complex-vector arguments
# (the `evec` argument of `group_index`); supply it here.
function Mooncake.increment_and_get_rdata!(
    f::Vector{Mooncake.Tangent{@NamedTuple{re::T, im::T}}},
    r::Mooncake.NoRData,
    t::Vector{<:Complex},
) where {T<:Base.IEEEFloat}
    @inbounds for i in eachindex(f)
        fi = f[i].fields
        f[i] = Mooncake.Tangent((re = fi.re + T(real(t[i])), im = fi.im + T(imag(t[i]))))
    end
    return r
end

for (TGrid, TEps) in ((Grid{2,Float64}, Array{Float64,4}), (Grid{3,Float64}, Array{Float64,5}))
    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof(group_index),Float64,Array{ComplexF64,1},Float64,$TEps,$TEps,$TGrid},
        false,
    )
end

# Mode classification utilities are discrete-valued; treat as zero-derivative.
@zero_adjoint MinimalCtx Tuple{typeof(ModeAnalysis.count_E_nodes),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(ModeAnalysis.mode_idx),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(ModeAnalysis.mode_viable),Vararg}

end # module
