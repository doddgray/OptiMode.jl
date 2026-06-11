# Mooncake.jl interface for DielectricSmoothing.
#
# The Kottke smoothing kernels (`εₑ_∂ωεₑ_∂²ωεₑ` and friends) are generated scalar Julia
# code which Mooncake differentiates natively. The shape-index bookkeeping
# (`corner_sinds`/`proc_sinds`) is integer-valued and explicitly marked zero-derivative,
# matching the `@non_differentiable` ChainRules declarations used by Zygote.

module DielectricSmoothingMooncakeExt

using DielectricSmoothing
using Mooncake
using Mooncake: @zero_adjoint, MinimalCtx

@zero_adjoint MinimalCtx Tuple{typeof(DielectricSmoothing.corner_sinds),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(DielectricSmoothing.proc_sinds),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(DielectricSmoothing.corners),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(DielectricSmoothing.matinds),Vararg}

end # module
