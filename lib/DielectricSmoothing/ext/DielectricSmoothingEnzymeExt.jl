# Enzyme.jl interface for DielectricSmoothing.
#
# `smooth_ε` propagates a small 2nd-order Taylor jet (`J2`) through the closed-form Kottke
# transforms (`kottke.jl`) instead of a giant symbolically-generated kernel, so Enzyme
# differentiates the smoothing kernel **natively in both directions** — no custom rule needed
# for the kernel itself, for *either* the material-data or the geometry-*parameter* gradient.
#
# The integer pixel bookkeeping (`corner_sinds`/`proc_sinds`/`matinds`) is piecewise-constant
# (it returns which shape/material index owns a grid corner) — genuinely inactive w.r.t. *both*
# material data and geometry parameters almost everywhere, matching the `@non_differentiable`
# ChainRules declarations used by Zygote (Enzyme does not honour `@non_differentiable`, hence
# these explicit markers).
#
# `surfpt_nearby`/`volfrac`/`_interface_geometry`, by contrast, carry the actual continuous
# dependence of the smoothed dielectric on the *geometry* (they compute the interface location,
# fill fraction and orientation Kottke averages over) — marking them inactive is only valid for
# material-*data*-only differentiation, and silently gives a wrong all-zero geometry gradient
# otherwise. They are deliberately left active here so Enzyme differentiates geometry parameters
# too. Two historical blockers on this path are now resolved: (1) `Cuboid.surfpt_nearby` was
# reworked (in the GeometryPrimitives fork) to avoid `inv(s.p)`, whose StaticArrays
# implementation could crash Enzyme or yield silently wrong gradients; (2) the `Union` produced
# by `shapes[sidx1]` indexing a heterogeneous shapes tuple inside `_interface_geometry` — which
# used to raise `IllegalTypeAnalysisException` — no longer does with the pinned Enzyme version;
# both the homogeneous- and heterogeneous-shape-type cases now match ForwardDiff/finite
# differences exactly in forward and reverse mode (see `test/enzyme_geometry_gradient.jl`).

module DielectricSmoothingEnzymeExt

using DielectricSmoothing
using GeometryPrimitives: surfpt_nearby, volfrac
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(DielectricSmoothing.corner_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.proc_sinds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(DielectricSmoothing.matinds), args...; kwargs...) = nothing

# `Grid` is a fixed simulation-configuration struct (extents, point counts) — never itself a
# differentiated quantity, however many closures happen to capture it. Declaring the whole type
# inactive lets Enzyme's activity analysis skip it outright instead of inferring (sometimes
# ambiguously, across nested closures that each capture the same `Grid`) that it needs a shadow,
# which otherwise surfaces as `MethodError: no method matching zero(::Active{Grid{...}})` deep in
# an imported ChainRules rule.
EnzymeRules.inactive_type(::Type{<:DielectricSmoothing.Grid}) = true

end # module
