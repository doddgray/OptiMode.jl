# Mooncake.jl interface for ModePerturbations.
#
# The perturbation kernel `_perturbation_scalar` composes FFTs and the `HMₖH` quadratic
# form; its reverse rule is a ChainRules `rrule` (`src/ad_rules.jl`) bridged to Mooncake
# here with `@from_rrule`.

module ModePerturbationsMooncakeExt

using ModePerturbations
using ModePerturbations: _perturbation_re, _perturbation_im
using DielectricSmoothing: Grid
using Mooncake
using Mooncake: @from_rrule, MinimalCtx

Base.retry_load_extensions()   # ensure Mooncake's ChainRules bridge is available

# Mooncake's @from_rrule bridge lacks a tangent conversion for complex-vector arguments
# (the `evec` argument); supply it here (same shim as ModeAnalysisMooncakeExt).
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

for (TGrid, NEps) in ((Grid{2,Float64}, 4), (Grid{3,Float64}, 5))
    TEps = Array{Float64,NEps}
    for TΔε in (Array{Float64,NEps}, Array{ComplexF64,NEps})
        for kernel in (:_perturbation_re, :_perturbation_im)
            @eval @from_rrule(
                MinimalCtx,
                Tuple{typeof($kernel),Float64,Array{ComplexF64,1},$TEps,$TΔε,$TGrid},
                false,
            )
        end
    end
end

end # module
