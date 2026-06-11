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

Base.in(p::AbstractVector{<:Real}, ms::MaterialShape) = Base.in(p, ms.shape)
GeometryPrimitives.surfpt_nearby(p::AbstractVector{<:Real}, ms::MaterialShape) = surfpt_nearby(p, ms.shape)
GeometryPrimitives.bounds(ms::MaterialShape) = GeometryPrimitives.bounds(ms.shape)
GeometryPrimitives.normal(p::AbstractVector{<:Real}, ms::MaterialShape) = GeometryPrimitives.normal(p, ms.shape)

material(ms::MaterialShape) = ms.data
material(x) = x

# GeometryPrimitives.rtol(x) = sqrt(eps)·x is only defined for AbstractFloat. Zygote's
# broadcast machinery pushes ForwardDiff.Dual numbers through it; provide a generic
# Real fallback that scales by the same Float64 machine tolerance.
GeometryPrimitives.rtol(x::Real) = sqrt(eps(Float64)) * x

# GeometryPrimitives ≥ 0.5 stores shape fields as hardcoded Float64, so geometry
# parameters cannot carry AD number types through shape construction. Consistently,
# geometry queries are constants for AD purposes (gradients w.r.t. material data flow
# around them); geometry-parameter sensitivities require finite differences until a
# parametric-eltype shape library is available.
@non_differentiable GeometryPrimitives.surfpt_nearby(::Any, ::Any)
@non_differentiable GeometryPrimitives.volfrac(::Any, ::Any, ::Any)
@non_differentiable corners(::Grid)
