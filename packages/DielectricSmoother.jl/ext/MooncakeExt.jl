"""
    MooncakeExt (DielectricSmoother)

Mooncake.jl AD extension for DielectricSmoother.jl.

Key differentiable operations:
- `smooth_ε`: full grid smoothing (non-differentiable w.r.t. shape geometry,
   differentiable w.r.t. material tensor values `mat_vals`)
- `εₑ_∂ωεₑ_∂²ωεₑ`: Kottke-smoothed tensor computation (differentiable)
- `avg_param_rot` / `τ_trans` / `τ⁻¹_trans`: building blocks (differentiable)

Shape queries (`corner_sinds`, `proc_sinds`) are inherently non-differentiable
and are correctly marked with `@non_differentiable` in the main package.
"""
module MooncakeExt

using DielectricSmoother
using Mooncake
using Mooncake: CoDual, primal, tangent, zero_tangent, NoTangent, increment!!

# ──────────────────────────────────────────────────────────────────────────────
# corner_sinds and proc_sinds: shape queries, not differentiable
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(DielectricSmoother.corner_sinds), Any, Any}
function Mooncake.rrule!!(
    ::CoDual{typeof(DielectricSmoother.corner_sinds)},
    shapes::CoDual,
    points::CoDual
)
    result = DielectricSmoother.corner_sinds(primal(shapes), primal(points))
    return CoDual(result, NoTangent()), (dret) -> (NoTangent(), NoTangent(), NoTangent())
end

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(DielectricSmoother.proc_sinds), Any}
function Mooncake.rrule!!(
    ::CoDual{typeof(DielectricSmoother.proc_sinds)},
    corner_sinds::CoDual
)
    result = DielectricSmoother.proc_sinds(primal(corner_sinds))
    return CoDual(result, NoTangent()), (dret) -> (NoTangent(), NoTangent())
end

# ──────────────────────────────────────────────────────────────────────────────
# τ_trans: ε → τ(ε) transformation (differentiable)
# Kottke τ transform: permittivity tensor → generalized τ matrix
# Used in Kottke averaging: τ = [-1/ε₁₁, ε₁₂/ε₁₁, ...; ...]
# Mooncake handles this automatically via source transformation if all ops are
# differentiable. We just ensure no non-diff paths are taken.
# ──────────────────────────────────────────────────────────────────────────────

# τ_trans and τ⁻¹_trans are differentiable matrix operations — Mooncake's
# automatic differentiation handles them without custom rules.

# ──────────────────────────────────────────────────────────────────────────────
# normcart: n → S (local Cartesian frame from normal vector)
# Non-differentiable w.r.t. shapes (geometry is fixed), but differentiable
# w.r.t. the normal vector n in principle. In practice it's non-differentiable
# in smooth_ε because corner_sinds is non-diff.
# ──────────────────────────────────────────────────────────────────────────────

# smooth_ε is differentiable w.r.t. mat_vals (the dielectric tensor values)
# but non-differentiable w.r.t. shapes and minds (indices).
# The corner_sinds/proc_sinds calls are already @non_differentiable.
# Mooncake will propagate through the differentiable mat_vals automatically.

end # module MooncakeExt
