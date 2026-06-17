export MaterialShape, material

"""
    MaterialShape(shape::GeometryPrimitives.Shape, data)

Associate material `data` (a material index, `Material`, `NumMat`, or raw dielectric
data) with a geometric shape. GeometryPrimitives.jl ≥ 0.5 removed the `data` payload
that earlier versions attached to every shape; this wrapper restores that association
for use in geometry lists consumed by `smooth_ε` and friends.

All geometric queries (`in`, `surfpt_nearby`, `bounds`) delegate to the wrapped shape.
"""
struct MaterialShape{TS<:Shape,TD}
    shape::TS
    data::TD
end

Base.in(p::AbstractVector{<:Number}, ms::MaterialShape) = Base.in(p, ms.shape)
GeometryPrimitives.surfpt_nearby(p::AbstractVector{<:Number}, ms::MaterialShape) = surfpt_nearby(p, ms.shape)
GeometryPrimitives.bounds(ms::MaterialShape) = GeometryPrimitives.bounds(ms.shape)
GeometryPrimitives.normal(p::AbstractVector{<:Number}, ms::MaterialShape) = GeometryPrimitives.normal(p, ms.shape)

material(ms::MaterialShape) = ms.data
material(x) = x

# GeometryPrimitives' `rtol` is only defined for the element types it ships with; AD
# broadcast machinery (e.g. ForwardDiff `Dual`s through the smoothing pipeline) pushes
# other `Real` subtypes through it. Provide a generic `Real` fallback scaling by the
# Float64 machine tolerance. On GeometryPrimitives ≥ 0.6 (which defines `rtol(::Number)`)
# this `Real` method is simply more specific and non-ambiguous; on 0.5 it is required.
GeometryPrimitives.rtol(x::Real) = sqrt(eps(Float64)) * x

# Geometry-parameter gradients. GeometryPrimitives ≥ 0.6 stores shape fields with a
# parametric element type (`T<:Number`) and its geometric queries (`surfpt_nearby`,
# `volfrac`) are AD-compatible, so AD number types now flow through shape construction
# and the queries — enabling gradients of `smooth_ε` w.r.t. geometry parameters with:
#   - ForwardDiff (Dual propagation), through the full geometry→smoothing pipeline, and
#   - Mooncake (native reverse rules), at the per-interface-pixel kernel granularity.
# Both bypass ChainRules, so the `@non_differentiable` declarations below do not impede
# them. (Enzyme currently segfaults on the StaticArrays matrix inverse inside Cuboid
# `surfpt_nearby`, and Zygote hits a non-`SVector` normal in `volfrac`; geometry-param
# reverse mode is therefore Mooncake/ForwardDiff.)
#
# The ChainRules `@non_differentiable` markers are retained so that *material-data*
# reverse-mode gradients via Zygote — which traverse `smooth_ε` and would otherwise try
# to differentiate the geometry queries (constant w.r.t. material data) — treat the
# geometry queries and the integer pixel classification as constants. Grid corner
# positions likewise do not depend on geometry parameters.
@non_differentiable GeometryPrimitives.surfpt_nearby(::Any, ::Any)
@non_differentiable GeometryPrimitives.volfrac(::Any, ::Any, ::Any)
@non_differentiable corners(::Grid)
