# Reverse-mode rules for FFT/Tullio-based post-processing functions.
#
# `group_index`'s forward pass composes FFTs and Tullio tensor contractions. Zygote
# differentiates that program through the AbstractFFTs/Tullio reverse rules, so we use it
# (once, here) to expose a ChainRulesCore.rrule. Reverse-mode engines without native
# FFTW support (Mooncake via `@from_rrule`, Enzyme via `@import_rrule`) reuse this rule
# through the package extensions.

_zygote_tangent(x) = x === nothing ? ZeroTangent() : x

function ChainRulesCore.rrule(::typeof(group_index), k::Real, evec, ω, ε⁻¹, ∂ε_∂ω, grid::Grid)
    y, zpb = Zygote.pullback(
        (k_, ev_, ω_, ei_, de_) -> _group_index_kernel(k_, ev_, ω_, ei_, de_, grid),
        k, evec, ω, ε⁻¹, ∂ε_∂ω,
    )
    function group_index_pullback(ȳ)
        k̄, ēv, ω̄, eī, dē = zpb(ȳ)
        return (NoTangent(), _zygote_tangent(k̄), _zygote_tangent(ēv), _zygote_tangent(ω̄),
            _zygote_tangent(eī), _zygote_tangent(dē), NoTangent())
    end
    return y, group_index_pullback
end
