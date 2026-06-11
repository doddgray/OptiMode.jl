# Enzyme.jl interface for DielectricSmoothing.
#
# The Kottke smoothing kernels (`εₑ_∂ωεₑ_∂²ωεₑ` and friends) are generated scalar Julia
# code which Enzyme differentiates natively. The shape-index bookkeeping
# (`corner_sinds`/`proc_sinds`) is integer-valued and marked inactive, matching the
# `@non_differentiable` ChainRules declarations used by Zygote.

module DielectricSmoothingEnzymeExt

using DielectricSmoothing
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(DielectricSmoothing.corner_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.proc_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.matinds), args...; kwargs...) = nothing

end # module
