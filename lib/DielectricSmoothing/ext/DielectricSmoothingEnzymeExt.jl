# Enzyme.jl interface for DielectricSmoothing.
#
# The shape-index bookkeeping (`corner_sinds`/`proc_sinds`/`matinds`) is integer-valued
# and marked inactive, matching the `@non_differentiable` ChainRules declarations used by
# Zygote.
#
# NOTE on smoothing gradients with Enzyme: the Kottke kernels (`fjh_εₑᵣ` and friends) are
# *enormous* symbolically generated functions (the smoothed tensor together with its first
# and second frequency derivatives). Enzyme/GPUCompiler cannot compile a call path that
# contains them — even merely as the *primal* of a custom rule — and overflows the native
# stack (`StackOverflowError`) before any differentiation happens. This is a compiler
# scale limit, not a missing rule: bridging the `smooth_ε` `ChainRulesCore.rrule`/`frule`
# (defined in `src/smooth.jl`) via `@import_rrule`/`@import_frule` does *not* help, because
# Enzyme still has to compile the rule's primal execution of `smooth_ε`.
#
# Material- and geometry-parameter gradients of `smooth_ε` are therefore provided by
# ForwardDiff (forward mode) and Zygote (reverse mode), which differentiate the smoothing
# pipeline directly (validated in `test/runtests.jl`), and the per-interface-pixel Kottke
# kernel is additionally covered by Mooncake. Downstream of smoothing, the eigensolver and
# mode-analysis stack is fully Enzyme-differentiable (forward & reverse). A whole-pipeline
# gradient that must include the smoothing step should use ForwardDiff/Zygote; Enzyme is
# the fast path from the (inverse-)permittivity field onward.

module DielectricSmoothingEnzymeExt

using DielectricSmoothing
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(DielectricSmoothing.corner_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.proc_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.matinds), args...; kwargs...) = nothing

end # module
