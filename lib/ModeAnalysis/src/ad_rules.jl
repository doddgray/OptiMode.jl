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

"""
Forward-mode rule for `group_index`. Enzyme forward mode cannot differentiate FFTW plan
creation inside the quadratic forms (`No forward mode derivative found for
fftw_plan_guru64_dft`); this frule computes the directional derivative with ForwardDiff,
whose AbstractFFTs extension differentiates FFTs by applying the plan to the dual parts
(never differentiating the plan itself). Bridged to Enzyme forward via
`Enzyme.@import_frule` in the package's Enzyme extension.
"""
function ChainRulesCore.frule((_, Δk, Δev, Δω, Δei, Δde, _), ::typeof(group_index),
        k::Real, evec, ω, ε⁻¹, ∂ε_∂ω, grid::Grid)
    y = _group_index_kernel(k, evec, ω, ε⁻¹, ∂ε_∂ω, grid)
    _v(Δ, x) = Δ isa AbstractZero ? zero(x) : Δ
    dk, dω = _v(Δk, k), _v(Δω, ω)
    dev, dei, dde = _v(Δev, evec), _v(Δei, ε⁻¹), _v(Δde, ∂ε_∂ω)
    ẏ = ForwardDiff.derivative(
        t -> _group_index_kernel(k + t*dk, evec .+ t .* dev, ω + t*dω,
                                 ε⁻¹ .+ t .* dei, ∂ε_∂ω .+ t .* dde, grid), 0.0)
    return y, ẏ
end

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
