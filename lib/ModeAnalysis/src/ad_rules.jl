# Reverse-mode rules for FFT/Tullio-based post-processing functions.
#
# `group_index`'s forward pass composes FFTs and Tullio tensor contractions. Zygote
# differentiates that program through the AbstractFFTs/Tullio reverse rules, so we use it
# (once, here) to expose a ChainRulesCore.rrule. Reverse-mode engines without native
# FFTW support (Mooncake via `@from_rrule`, Enzyme via `@import_rrule`) reuse this rule
# through the package extensions.
#
# The pullback returns are concretely typed: Enzyme's imported rules require the
# cotangent of an `Active` scalar argument to be returned as a concrete float.

_scalar_tangent(x)::Float64 = Float64(real(x))
_scalar_tangent(::Nothing)::Float64 = 0.0
_array_tangent(x::AbstractArray) = x
_array_tangent(::Nothing) = NoTangent()

function ChainRulesCore.rrule(::typeof(group_index), k::Real, evec, ω, ε⁻¹, ∂ε_∂ω, grid::Grid)
    y, zpb = Zygote.pullback(
        (k_, ev_, ω_, ei_, de_) -> _group_index_kernel(k_, ev_, ω_, ei_, de_, grid),
        k, evec, ω, ε⁻¹, ∂ε_∂ω,
    )
    function group_index_pullback(ȳ)
        k̄, ēv, ω̄, eī, dē = zpb(ȳ)
        return (NoTangent(), _scalar_tangent(k̄), _array_tangent(ēv), _scalar_tangent(ω̄),
            _array_tangent(eī), _array_tangent(dē), NoTangent())
    end
    return y, group_index_pullback
end
