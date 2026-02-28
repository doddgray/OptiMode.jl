"""
    EnzymeExt

Enzyme.jl automatic differentiation extension for MaterialModels.jl.

Provides custom forward- and reverse-mode AD rules for key MaterialModels
functions using Enzyme's `EnzymeRules` interface.
"""
module EnzymeExt

using MaterialModels
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Non-differentiable / inactive functions
# Code generation functions use Symbolics and RuntimeGeneratedFunctions which
# are not compatible with Enzyme's AD. Mark them as inactive.
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(MaterialModels.generate_fn)},
    ::Type{<:Const},
    mat::Const,
    model_name::Const,
    args::Const...
)
    fn = MaterialModels.generate_fn(mat.val, model_name.val, (a.val for a in args)...)
    return EnzymeRules.AugmentedReturn(fn, nothing, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(MaterialModels.generate_fn)},
    dret,
    tape,
    mat::Const,
    model_name::Const,
    args::Const...
)
    return (nothing, nothing, map(_ -> nothing, args)...)
end

# ──────────────────────────────────────────────────────────────────────────────
# ε_tensor: scalar n → 3×3 diagonal dielectric tensor
# Forward rule: dε = 2n·dn·I
# Reverse rule: dn̄ = 2n · tr(ε̄)
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(MaterialModels.ε_tensor)},
    RT::Type{<:EnzymeRules.Duplicated},
    n::EnzymeRules.Duplicated{<:Real}
)
    ε = MaterialModels.ε_tensor(n.val)
    dε = MaterialModels.ε_tensor(n.val) * (2 * n.dval / n.val)  # 2n·dn·I/n = 2dn·I
    # More directly: dε[i,i] = 2*n.val * n.dval
    dε_correct = 2 * n.val * n.dval * one(typeof(ε))
    return EnzymeRules.Duplicated(ε, dε_correct)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(MaterialModels.ε_tensor)},
    ::Type{<:EnzymeRules.Duplicated},
    n::EnzymeRules.Duplicated{<:Real}
)
    ε = MaterialModels.ε_tensor(n.val)
    return EnzymeRules.AugmentedReturn(ε, nothing, n.val)  # tape = n.val
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(MaterialModels.ε_tensor)},
    dret::EnzymeRules.Duplicated,
    tape,  # = n.val saved from augmented_primal
    n::EnzymeRules.Duplicated{<:Real}
)
    n_val = tape
    # dn̄ = 2n · ∑ᵢ ε̄[i,i]
    dn_bar = 2 * n_val * real(tr(dret.dval))
    n.dval += dn_bar
    return (nothing,)
end

# ──────────────────────────────────────────────────────────────────────────────
# rotate (2nd-order): χ → R^T χ R
# Forward: dχᵣ = R^T dχ R
# Reverse: dχ̄ = R dχᵣ̄ R^T
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(MaterialModels.rotate)},
    RT::Type{<:EnzymeRules.Duplicated},
    χ::EnzymeRules.Duplicated{<:AbstractMatrix},
    𝓡::Const{<:AbstractMatrix}
)
    χᵣ = MaterialModels.rotate(χ.val, 𝓡.val)
    dχᵣ = MaterialModels.rotate(χ.dval, 𝓡.val)
    return EnzymeRules.Duplicated(χᵣ, dχᵣ)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(MaterialModels.rotate)},
    ::Type{<:EnzymeRules.Duplicated},
    χ::EnzymeRules.Duplicated{<:AbstractMatrix},
    𝓡::Const{<:AbstractMatrix}
)
    χᵣ = MaterialModels.rotate(χ.val, 𝓡.val)
    return EnzymeRules.AugmentedReturn(χᵣ, nothing, 𝓡.val)  # tape = rotation matrix
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(MaterialModels.rotate)},
    dret::EnzymeRules.Duplicated,
    tape,  # = 𝓡.val
    χ::EnzymeRules.Duplicated{<:AbstractMatrix},
    𝓡::Const{<:AbstractMatrix}
)
    𝓡_val = tape
    # χ̄ = R dχᵣ̄ R^T (reverse of rotate)
    χ.dval .+= MaterialModels.rotate(dret.dval, 𝓡_val')
    return (nothing, nothing)
end

end # module EnzymeExt
