"""
    EnzymeExt (ModeAnalysis)

Enzyme.jl AD extension for ModeAnalysis.jl.

Provides custom Enzyme rules for field conversion operations where Enzyme's
automatic source transformation may struggle (primarily FFTW-based operations
and in-place mutations).
"""
module EnzymeExt

using ModeAnalysis
using EigenModeSolver
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
using FFTW

# ──────────────────────────────────────────────────────────────────────────────
# _H2d! (H⃗ → D⃗): applies k× in reciprocal space, then IFFT
# Custom forward rule: d(d) = k× d(H), then IFFT(d(d))
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(ModeAnalysis._H2d!)},
    ::Type{<:EnzymeRules.Const},
    d_out::EnzymeRules.Duplicated{<:AbstractArray},
    H_in::EnzymeRules.Duplicated{<:AbstractArray},
    M̂::Const{<:EigenModeSolver.HelmholtzMap}
)
    # Primal computation
    ModeAnalysis._H2d!(d_out.val, H_in.val, M̂.val)
    # Tangent: linear in H_in → d(d) = k× d(H_in), then FFT
    ModeAnalysis._H2d!(d_out.dval, H_in.dval, M̂.val)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# _d2ẽ! (D⃗ → Ẽ): applies ε⁻¹ in real space, then IFFT
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(ModeAnalysis._d2ẽ!)},
    ::Type{<:EnzymeRules.Const},
    e_out::EnzymeRules.Duplicated{<:AbstractArray},
    d_in::EnzymeRules.Duplicated{<:AbstractArray},
    M̂::Const{<:EigenModeSolver.HelmholtzMap}
)
    ModeAnalysis._d2ẽ!(e_out.val, d_in.val, M̂.val)
    # Tangent: linear in d_in → d(e) = ε⁻¹ · d(d), then IFFT
    ModeAnalysis._d2ẽ!(e_out.dval, d_in.dval, M̂.val)
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# S⃗ (Poynting vector): E × H* — forward mode rule
# d(S) = dE × H* + E × d(H*)
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(ModeAnalysis.S⃗)},
    RT::Type{<:EnzymeRules.Duplicated},
    E::EnzymeRules.Duplicated{<:AbstractArray},
    H::EnzymeRules.Duplicated{<:AbstractArray}
)
    S = ModeAnalysis.S⃗(E.val, H.val)
    # S = real(E × conj(H)) / 2
    # dS = real(dE × conj(H) + E × conj(dH)) / 2
    dS = ModeAnalysis.S⃗(E.dval, H.val) .+ ModeAnalysis.S⃗(E.val, H.dval)
    return EnzymeRules.Duplicated(S, dS)
end

# ──────────────────────────────────────────────────────────────────────────────
# normE!: non-differentiable normalization
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(ModeAnalysis.normE!)},
    ::Type{<:EnzymeRules.Const},
    E::Const,
    args::Const...
)
    ModeAnalysis.normE!(E.val, (a.val for a in args)...)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(ModeAnalysis.normE!)},
    dret,
    tape,
    E::Const,
    args::Const...
)
    return (nothing, map(_ -> nothing, args)...)
end

end # module EnzymeExt
