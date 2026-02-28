"""
    EnzymeExt (DielectricSmoother)

Enzyme.jl AD extension for DielectricSmoother.jl.

Provides custom Enzyme rules for non-standard operations in the smoothing pipeline.
Most arithmetic in this package (τ transform, rotation, averaging) is automatically
differentiable by Enzyme. Custom rules are needed only for:
1. Shape query functions (corner_sinds, proc_sinds) — marked as inactive/const
2. surfpt_nearby and volfrac from GeometryPrimitives — treated as non-diff w.r.t. shapes
"""
module EnzymeExt

using DielectricSmoother
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
using StaticArrays

# ──────────────────────────────────────────────────────────────────────────────
# corner_sinds: shape queries — non-differentiable
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(DielectricSmoother.corner_sinds)},
    ::Type{<:Const},
    shapes::Const,
    points::Const
)
    result = DielectricSmoother.corner_sinds(shapes.val, points.val)
    return EnzymeRules.AugmentedReturn(result, nothing, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(DielectricSmoother.corner_sinds)},
    dret,
    tape,
    shapes::Const,
    points::Const
)
    return (nothing, nothing)
end

# ──────────────────────────────────────────────────────────────────────────────
# proc_sinds: index processing — non-differentiable
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(DielectricSmoother.proc_sinds)},
    ::Type{<:Const},
    corner_sinds::Const
)
    result = DielectricSmoother.proc_sinds(corner_sinds.val)
    return EnzymeRules.AugmentedReturn(result, nothing, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(DielectricSmoother.proc_sinds)},
    dret,
    tape,
    corner_sinds::Const
)
    return (nothing,)
end

# ──────────────────────────────────────────────────────────────────────────────
# normcart: construct local Cartesian frame — forward rule for n vector
# Forward: dS/dn (directional derivative of the rotation matrix S)
# This is needed when differentiating through the Kottke smoothing formula
# w.r.t. shape parameters that affect the interface normal vector.
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(DielectricSmoother.normcart)},
    RT::Type{<:EnzymeRules.Duplicated},
    n::EnzymeRules.Duplicated{<:AbstractVector}
)
    # Primal
    S = DielectricSmoother.normcart(n.val)

    # Tangent: finite-difference approximation of dS/dn in direction dn
    ε = 1e-8
    n_fwd = n.val + ε * n.dval
    n_fwd_norm = n_fwd / norm(n_fwd)
    S_fwd = DielectricSmoother.normcart(n_fwd_norm)
    dS = (S_fwd - S) / ε

    return EnzymeRules.Duplicated(S, dS)
end

# ──────────────────────────────────────────────────────────────────────────────
# avg_param_rot: Kottke averaging in rotated frame
# Forward rule for mat_vals differentiation
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(DielectricSmoother.avg_param_rot)},
    RT::Type{<:EnzymeRules.Duplicated},
    ε₁ᵣ::EnzymeRules.Duplicated{<:AbstractMatrix},
    ε₂ᵣ::EnzymeRules.Duplicated{<:AbstractMatrix},
    r₁::Const{<:Real}
)
    # Primal
    εₑᵣ = DielectricSmoother.avg_param_rot(ε₁ᵣ.val, ε₂ᵣ.val, r₁.val)

    # Tangent via finite difference (complex step would be more accurate but less compatible)
    ε = 1e-8
    εₑᵣ_d1 = DielectricSmoother.avg_param_rot(ε₁ᵣ.val + ε*ε₁ᵣ.dval, ε₂ᵣ.val, r₁.val)
    εₑᵣ_d2 = DielectricSmoother.avg_param_rot(ε₁ᵣ.val, ε₂ᵣ.val + ε*ε₂ᵣ.dval, r₁.val)
    dεₑᵣ = (εₑᵣ_d1 + εₑᵣ_d2 - 2*εₑᵣ) / ε  # approximation via superposition

    return EnzymeRules.Duplicated(εₑᵣ, dεₑᵣ)
end

end # module EnzymeExt
