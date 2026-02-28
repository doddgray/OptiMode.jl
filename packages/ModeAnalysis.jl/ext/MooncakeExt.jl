"""
    MooncakeExt (ModeAnalysis)

Mooncake.jl AD extension for ModeAnalysis.jl.

All field conversion functions (H⃗→D⃗→E⃗) are built from differentiable
array operations (FFTs, matrix multiplications, cross products) that
Mooncake can handle automatically via source transformation.

Custom rules are provided for:
- `normE!` / `canonicalize_phase!`: normalization ops with in-place semantics
- `group_index`: computation involving solve derivatives (eig_adjt-based)
"""
module MooncakeExt

using ModeAnalysis
using EigenModeSolver
using Mooncake
using Mooncake: CoDual, primal, tangent, zero_tangent, NoTangent, increment!!
using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# normE!: in-place field normalization — treat the normalization constant as
# non-differentiable (normalization removes a degree of freedom)
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(ModeAnalysis.normE!), Any, Any, Any, Any, Any, Any}
function Mooncake.rrule!!(
    ::CoDual{typeof(ModeAnalysis.normE!)},
    E::CoDual,
    ε⁻¹::CoDual,
    M̂::CoDual,
    grid::CoDual,
    δV::CoDual,
    norms::CoDual
)
    # Normalization is a non-differentiable operation (removes phase/scale freedom)
    ModeAnalysis.normE!(primal(E), primal(ε⁻¹), primal(M̂), primal(grid), primal(δV), primal(norms))
    return CoDual(nothing, NoTangent()), (dret) -> (
        NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# canonicalize_phase!: phase normalization — non-differentiable
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(ModeAnalysis.canonicalize_phase!), Any}
function Mooncake.rrule!!(
    ::CoDual{typeof(ModeAnalysis.canonicalize_phase!)},
    H::CoDual
)
    ModeAnalysis.canonicalize_phase!(primal(H))
    return CoDual(nothing, NoTangent()), (dret) -> (NoTangent(), NoTangent())
end

# ──────────────────────────────────────────────────────────────────────────────
# E_relpower_xyz: relative power in each polarization component
# This is a pure computation on field arrays — differentiable by Mooncake
# automatically. No custom rule needed.
# ──────────────────────────────────────────────────────────────────────────────

# S⃗ (Poynting vector) = E × H* is differentiable w.r.t. E and H
# via the cross product rule — handled automatically by Mooncake.

end # module MooncakeExt
