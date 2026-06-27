# Enzyme.jl interface for DielectricSmoothing.
#
# `smooth_ε` now propagates a small 2nd-order Taylor jet (`J2`) through the closed-form
# Kottke transforms (`kottke.jl`) instead of a giant symbolically-generated kernel, so
# Enzyme differentiates the smoothing **material-data** gradients natively in both
# directions — no custom rule needed for the kernel itself.
#
# The integer pixel bookkeeping (`corner_sinds`/`proc_sinds`/`matinds`) and the geometry
# queries (`surfpt_nearby`/`volfrac`) are constant with respect to the material data, so
# they are marked `EnzymeRules.inactive` — matching the `@non_differentiable` ChainRules
# declarations used by Zygote. (Enzyme does not honour `@non_differentiable`; without these
# markers it tries to differentiate the StaticArrays matrix inverse inside Cuboid
# `surfpt_nearby`, which it cannot handle. Geometry-*parameter* gradients of `smooth_ε`
# therefore still go through ForwardDiff/Mooncake, as before.)

module DielectricSmoothingEnzymeExt

using DielectricSmoothing
using GeometryPrimitives: surfpt_nearby, volfrac
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(DielectricSmoothing.corner_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.proc_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.matinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(surfpt_nearby), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(volfrac), args...; kwargs...) = nothing
# `_interface_geometry` indexes the heterogeneous shapes tuple (`shapes[sidx1]`), producing a
# `Union` that Enzyme's strict type analysis cannot handle. It is constant w.r.t. the material
# data (the differentiated input of `smooth_ε`), so marking it inactive keeps the Union out of
# Enzyme's type analysis — fixing the `IllegalTypeAnalysisException` on the preallocated
# `smooth_ε` assembly with Enzyme ≥ 0.13.168 on Julia 1.11.
EnzymeRules.inactive(::typeof(DielectricSmoothing._interface_geometry), args...; kwargs...) = nothing

end # module
