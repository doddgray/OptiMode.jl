# ──────────────────────────────────────────────────────────────────────────────
#  Cross-section de-duplication.
#
#  An EME stack of `num_cells` cells very often contains long uniform (or repeated)
#  runs — straight waveguide sections, identical grating periods — whose cells have
#  *geometrically identical* cross-sections. At a fixed grid and material list the
#  smoothed dielectric, and therefore the modal basis, is then bit-for-bit the same,
#  so the (expensive) mode solve only needs to run once per *unique* cross-section
#  and the result can be shared by every cell in that group.
#
#  This is the spatial-axis counterpart of caching the geometry smoothing across
#  frequency: here we collapse identical cells; the per-ω material reuse inside
#  `smooth_ε` is a separate (DielectricSmoothing-level) optimisation.
# ──────────────────────────────────────────────────────────────────────────────

export cross_section_key, dedup_groups

# Canonicalise a Float to a rounded integer multiple of `atol` so that geometrically
# identical cells (whose shape parameters are computed by identical arithmetic) hash and
# compare equal, while genuinely different cross-sections do not collide.
_q(x::Real; atol::Float64=1e-9) = round(Int, x / atol)

# Canonical, quantized key fragment for one shape's own geometry fields — dispatches per
# `GeometryPrimitives.Shape` subtype since each stores its geometry differently (`Cuboid`'s
# centre/half-widths/axes vs. `Polygon`'s vertex list — the latter is what a sidewall-angle
# `Trapezoid`/`Isosceles` rib actually constructs, see designer_dichroic_linbo3_litao3.jl).
# Add a new branch here for any further `GeometryPrimitives.Shape` subtype used in a
# `CrossSection`; the generic fallback keys on the shape's own `Base.hash`, which is exact for
# a bit-identical shape but won't dedup two structurally-equal-but-distinct shape instances.
_shape_key(b::GeometryPrimitives.Cuboid; atol::Float64) =
    (:Cuboid, Tuple(_q.(b.c; atol)), Tuple(_q.(2 .* b.r; atol)), Tuple(_q.(vec(Matrix(b.p)); atol)))
_shape_key(b::GeometryPrimitives.Polygon; atol::Float64) =
    (:Polygon, Tuple(_q.(vec(Matrix(b.v)); atol)))
_shape_key(b; atol::Float64) = (:Other, hash(b))

"""
    cross_section_key(cs::CrossSection; atol=1e-9) -> key

A hashable, `==`-comparable canonical key for a [`CrossSection`](@ref): two cells with equal
keys have the same shapes (geometry quantised to `atol` μm — see [`_shape_key`](@ref) for the
per-shape-type field lists) in the same order with the same material indices, hence an
identical smoothed dielectric on a given grid. Used by [`dedup_groups`](@ref) to solve each
unique cross-section's modes only once.
"""
function cross_section_key(cs::CrossSection; atol::Float64=1e-9)
    shape_keys = map(ms -> _shape_key(ms.shape; atol), cs.shapes)
    return (Tuple(shape_keys), Tuple(cs.minds))
end

"""
    dedup_groups(cells; atol=1e-9) -> (reps, gid)

Group cells by identical cross-section. Returns `reps`, the indices of the first cell of
each unique cross-section (the representatives to actually solve), and `gid`, a vector
mapping every cell to the position in `reps` of its group, so that

    modes_per_cell[i] = modes_of_representative[gid[i]]

shares one modal basis across all cells with the same geometry. For a stack of all-distinct
cross-sections (a fully tapered device) `reps == 1:length(cells)` and this is a no-op.
"""
function dedup_groups(cells::AbstractVector{Cell}; atol::Float64=1e-9)
    keys = [cross_section_key(c.cross_section; atol) for c in cells]
    reps = Int[]
    key_to_group = Dict{Any,Int}()
    gid = Vector{Int}(undef, length(cells))
    for (i, k) in enumerate(keys)
        g = get(key_to_group, k, 0)
        if g == 0
            push!(reps, i)
            g = length(reps)
            key_to_group[k] = g
        end
        gid[i] = g
    end
    return reps, gid
end

@non_differentiable cross_section_key(::Any)
@non_differentiable dedup_groups(::Any)
