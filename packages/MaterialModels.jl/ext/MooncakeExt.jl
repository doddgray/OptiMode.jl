"""
    MooncakeExt

Mooncake.jl automatic differentiation extension for MaterialModels.jl.

Provides custom reverse-mode AD rules for key MaterialModels functions using
Mooncake's `rrule!!` interface. Most differentiable functions in this package
are automatically handled via Mooncake's ChainRules compatibility layer, but
some non-differentiable or custom-rule functions require explicit declarations.
"""
module MooncakeExt

using MaterialModels
using Mooncake
using Mooncake: CoDual, primal, tangent, zero_tangent, NoTangent, increment!!
using ChainRulesCore: @non_differentiable

# ──────────────────────────────────────────────────────────────────────────────
# Non-differentiable declarations
# These functions involve symbolic computation / code generation and should
# not be differentiated through — their outputs (compiled functions) are
# treated as constants during AD.
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(MaterialModels.generate_fn), Any, Any, Vararg}
function Mooncake.rrule!!(
    ::CoDual{typeof(MaterialModels.generate_fn)},
    mat::CoDual,
    model_name::CoDual,
    args::CoDual...
)
    fn = MaterialModels.generate_fn(primal(mat), primal(model_name), primal.(args)...)
    return CoDual(fn, NoTangent()), (dret) -> (NoTangent(), NoTangent(), NoTangent(), map(_->NoTangent(), args)...)
end

# ──────────────────────────────────────────────────────────────────────────────
# ε_tensor: n → n²·I₃ₓ₃ (scalar refractive index → dielectric tensor)
# Derivative: dε/dn = 2n · I₃ₓ₃
# ──────────────────────────────────────────────────────────────────────────────

Mooncake.@is_primitive MinimalCtx, Tuple{typeof(MaterialModels.ε_tensor), Real}
function Mooncake.rrule!!(
    ::CoDual{typeof(MaterialModels.ε_tensor)},
    n::CoDual{<:Real}
)
    n_val = primal(n)
    ε = MaterialModels.ε_tensor(n_val)
    function ε_tensor_pullback(dε_codo::CoDual)
        dε = tangent(dε_codo)
        # ε = n²·I, so ∂ε/∂n = 2n·I => dn = sum(dε .* 2n·I) = 2n·tr(dε)
        # For SMatrix, dε is a tangent (same type as ε)
        dn_tangent = if dε isa Mooncake.NoTangent
            zero(n_val)
        else
            # sum diagonal elements scaled by 2n
            2 * n_val * real(sum(i -> dε[i,i], 1:3))
        end
        return NoTangent(), increment!!(tangent(n), dn_tangent)
    end
    return CoDual(ε, zero_tangent(ε)), ε_tensor_pullback
end

# ──────────────────────────────────────────────────────────────────────────────
# rotate: tensor rotation χ → R'χR (or higher-order contractions)
# Uses Tullio-based implementations already differentiable via Zygote/ChainRules,
# so we just ensure Mooncake can handle them via ChainRules compat.
# ──────────────────────────────────────────────────────────────────────────────

# rotate is already handled by ChainRulesCore rrule if defined there.
# Mooncake will use the ChainRulesCore compatibility automatically.

end # module MooncakeExt
