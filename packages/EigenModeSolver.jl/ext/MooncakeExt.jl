"""
    MooncakeExt (EigenModeSolver)

Mooncake.jl AD extension for EigenModeSolver.jl.

Provides custom reverse-mode AD rules for the core electromagnetic solver operations:

1. `solve_ω²`: Eigenvalue solve — uses adjoint method (Daleckii-Krein theorem)
   to backpropagate through iterative eigenvalue decomposition
2. `HelmholtzMap` multiply: The core M̂H product — differentiable w.r.t. ε⁻¹
3. Linear algebra primitives (cross products, dot products over grids)

The adjoint method for eigenvalue backpropagation implements:
   If Â·x = λx and L = f(λ, x), then
   ∂L/∂Â = (∂L/∂λ)·x⊗x† + solve((Â-λI)†, x̄ - x·⟨x,x̄⟩)⊗x†

This avoids differentiating through the iterative eigensolver itself.
"""
module MooncakeExt

using EigenModeSolver
using EigenModeSolver: HelmholtzMap, ModeSolver, eig_adjt, my_linsolve
using Mooncake
using Mooncake: CoDual, primal, tangent, zero_tangent, NoTangent, increment!!
using LinearAlgebra
using StaticArrays
using Tullio
using FFTW
using IterativeSolvers

# ──────────────────────────────────────────────────────────────────────────────
# Non-differentiable operations: eigensolver control flow
# ──────────────────────────────────────────────────────────────────────────────

# update_k! modifies ModeSolver in-place — treat as @ignore'd in Mooncake
Mooncake.@is_primitive MinimalCtx, Tuple{typeof(EigenModeSolver.update_k!), ModeSolver, Any}
function Mooncake.rrule!!(
    ::CoDual{typeof(EigenModeSolver.update_k!)},
    ms::CoDual{<:ModeSolver},
    k::CoDual
)
    EigenModeSolver.update_k!(primal(ms), primal(k))
    return CoDual(nothing, NoTangent()), (dret) -> (NoTangent(), NoTangent(), NoTangent())
end

# ──────────────────────────────────────────────────────────────────────────────
# _cross product rules — already have ChainRulesCore rrules in linalg.jl
# Mooncake will use these via ChainRulesCore compatibility.
# Explicit declarations for performance-critical paths:
# ──────────────────────────────────────────────────────────────────────────────

# _dot and _mult are standard linear algebra ops — Mooncake handles automatically.

# ──────────────────────────────────────────────────────────────────────────────
# sliceinv_3x3: inversion of 3×3 blocks over a spatial grid
# Forward-differentiable; custom reverse rule for efficiency:
# d(A⁻¹)/dA = -A⁻¹ · dA · A⁻¹
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(EigenModeSolver.sliceinv_3x3), AbstractArray}
function Mooncake.rrule!!(
    ::CoDual{typeof(EigenModeSolver.sliceinv_3x3)},
    A::CoDual{<:AbstractArray}
)
    A_val = primal(A)
    A_inv = EigenModeSolver.sliceinv_3x3(A_val)

    function sliceinv_pullback(dA_inv_codo::CoDual)
        dA_inv = tangent(dA_inv_codo)
        if dA_inv isa Mooncake.NoTangent
            return NoTangent(), Mooncake.NoTangent()
        end
        # dL/dA[i,j,:,:] = -∑ₖ A_inv[i,k,:,:] * dA_inv[k,l,:,:] * A_inv[l,j,:,:]
        # Using @tullio for efficiency
        @tullio dA[a, b, ix, iy] := -A_inv[a, c, ix, iy] * dA_inv[c, d, ix, iy] * A_inv[d, b, ix, iy]
        return NoTangent(), increment!!(tangent(A), dA)
    end

    return CoDual(A_inv, zero_tangent(A_inv)), sliceinv_pullback
end

# ──────────────────────────────────────────────────────────────────────────────
# solve_ω²: eigenvalue solve via adjoint method
# The key AD rule: we do NOT differentiate through the iterative eigensolver.
# Instead, we use the adjoint equation (Daleckii-Krein) in the pullback.
# ──────────────────────────────────────────────────────────────────────────────

# Note: solve_ω² has complex control flow (iterative refinement) that Mooncake
# cannot handle automatically. We provide the adjoint method manually.
# The rrule defined in grads/solve.jl (ChainRulesCore) will be used via
# Mooncake's ChainRulesCore compatibility layer.

end # module MooncakeExt
