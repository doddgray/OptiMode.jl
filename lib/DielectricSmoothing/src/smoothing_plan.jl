# ──────────────────────────────────────────────────────────────────────────────
#  Frequency-independent smoothing scaffold (geometry cache).
#
#  `smooth_ε(shapes, mat_vals, minds, grid)` does two kinds of work per pixel: a
#  *geometry* part — classify the pixel from its corners, and for an interface pixel
#  locate the surface (`surfpt_nearby`), integrate the fill fraction (`volfrac`) and
#  build the interface frame (`normcart`) — and a *material* part — evaluate the
#  dispersive Kottke kernel `εₑ_∂ωεₑ_∂²ωεₑ` on the two material columns. The geometry
#  part depends only on `(shapes, grid)`, not on frequency, yet it is redone at every
#  ω of a sweep (and, in EME, at every ω of every cell).
#
#  A `SmoothingPlan` precomputes that geometry scaffold once; `smooth_ε(plan, mat_vals)`
#  then applies only the AD-friendly Kottke algebra per ω. The result is identical to
#  `smooth_ε(shapes, mat_vals, minds, grid)` (same operations, same order) and stays
#  differentiable in `mat_vals` (the per-ω material data) in every backend; geometry is
#  frozen in the plan, so geometry-*parameter* gradients still use the original `smooth_ε`.
#
#  Note on the size of the win: benchmarking shows the geometry scaffold is only a modest
#  fraction (~10%) of `smooth_ε` — the per-pixel Kottke kernel and the output assembly,
#  which both paths share, dominate. So caching mainly removes redundant geometry work
#  across a sweep (and lets the EME (cell × ω) batch reuse one scaffold per cross-section);
#  the larger smoothing-cost lever is the per-pixel assembly, a separate optimisation.
# ──────────────────────────────────────────────────────────────────────────────

export SmoothingPlan, smoothing_plan

# Per-pixel kinds.
const _PLAN_UNIFORM = 0x01      # whole pixel in one material
const _PLAN_INTERFACE = 0x02    # one interface between two materials (Kottke)
const _PLAN_MULTI = 0x03        # ≥3 materials meet → naive corner average

"""
    SmoothingPlan{ND}

Precomputed, frequency-independent geometry scaffold for [`smooth_ε`](@ref) on a fixed
geometry and grid. Build it once with [`smoothing_plan`](@ref) and apply it per frequency
with `smooth_ε(plan, mat_vals)`. Pixels are stored in `corners(grid)` order; each carries
its kind and the geometry data the Kottke kernel needs — for an interface pixel the fill
fraction `rvol` and interface frame `S`, plus the material-column indices — so applying the
plan touches no geometry routines.
"""
struct SmoothingPlan{ND}
    gridsize::NTuple{ND,Int}
    NC::Int                                  # corners per pixel (4 in 2D, 8 in 3D)
    kind::Vector{UInt8}
    a::Vector{Int}                           # uniform: material col; interface: col 1; multi: index into `multi`
    b::Vector{Int}                           # interface: material col 2 (else 0)
    rvol::Vector{Float64}                    # interface: material-1 fill fraction
    S::Vector{SMatrix{3,3,Float64,9}}        # interface: normcart frame
    multi::Vector{Vector{Int}}               # multi pixels: the `NC` corner material cols
end

Base.size(p::SmoothingPlan) = p.gridsize
npixels(p::SmoothingPlan) = length(p.kind)

function Base.show(io::IO, p::SmoothingPlan{ND}) where {ND}
    nu = count(==(_PLAN_UNIFORM), p.kind)
    ni = count(==(_PLAN_INTERFACE), p.kind)
    nm = count(==(_PLAN_MULTI), p.kind)
    print(io, "SmoothingPlan{$ND}(", p.gridsize, ": $nu uniform, $ni interface, $nm multi)")
end

"""
    smoothing_plan(shapes, minds, grid) -> SmoothingPlan

Run the frequency-independent geometry pass of [`smooth_ε`](@ref) once: classify every
pixel and, for interface pixels, compute the surface location, fill fraction and interface
frame. The returned [`SmoothingPlan`](@ref) is reused across all frequencies (and, in EME,
all cells sharing the cross-section) via `smooth_ε(plan, mat_vals)`. `shapes`/`minds` are
the same foreground-ordered shapes and material-column map [`smooth_ε`](@ref) takes.
"""
function smoothing_plan(shapes, minds, grid::Grid{ND,TG}) where {ND,TG<:Real}
    crn = corners(grid)
    n = length(crn)
    kind = Vector{UInt8}(undef, n)
    a = Vector{Int}(undef, n)
    b = zeros(Int, n)
    rvol = zeros(Float64, n)
    Iframe = SMatrix{3,3,Float64}(I)
    S = fill(Iframe, n)
    multi = Vector{Int}[]
    NC = length(first(crn))
    for (k, crnrs) in enumerate(crn)
        ps = proc_sinds(corner_sinds(shapes, crnrs))
        if iszero(ps[2])
            kind[k] = _PLAN_UNIFORM
            a[k] = minds[first(ps)]
        elseif iszero(ps[3])
            sidx1, sidx2 = ps[1], ps[2]
            xyz = sum(crnrs) / NC
            r₀_n⃗ = surfpt_nearby(xyz, shapes[sidx1])
            kind[k] = _PLAN_INTERFACE
            a[k] = minds[sidx1]
            b[k] = minds[sidx2]
            rvol[k] = volfrac((vxlmin(crnrs), vxlmax(crnrs)), last(r₀_n⃗), first(r₀_n⃗))
            S[k] = normcart(vec3D(last(r₀_n⃗)))
        else
            kind[k] = _PLAN_MULTI
            push!(multi, Int[minds[i] for i in ps])
            a[k] = length(multi)
        end
    end
    return SmoothingPlan{ND}(size(grid), NC, kind, a, b, rvol, S, multi)
end

@non_differentiable smoothing_plan(shapes::Any, minds::Any, grid::Any)

# Per-pixel application of the plan, returning the 27-vector (ε, ∂ωε, ∂²ωε). Mirrors the
# three branches of `smooth_ε_single`, but with all geometry already resolved in the plan.
@inline function _apply_plan_single(p::SmoothingPlan, mat_vals, k::Int)
    kd = p.kind[k]
    if kd == _PLAN_UNIFORM
        return mat_vals[:, p.a[k]]
    elseif kd == _PLAN_INTERFACE
        return εₑ_∂ωεₑ_∂²ωεₑ(p.rvol[k], p.S[k], mat_vals[:, p.a[k]], mat_vals[:, p.b[k]])
    else
        cols = p.multi[p.a[k]]
        return sum(c -> mat_vals[:, c], cols) / p.NC
    end
end

"""
    smooth_ε(plan::SmoothingPlan, mat_vals) -> Array

Apply a precomputed [`SmoothingPlan`](@ref) to per-frequency material data `mat_vals`
(the same `(27, n_materials)` column matrix [`smooth_ε`](@ref) takes), returning the
identical `(3, 3, 3, size(grid)...)` smoothed-dielectric array — but running only the
Kottke kernel, with no geometry queries. Differentiable in `mat_vals` (forward and reverse)
just like the direct method, so material/ω gradients flow through unchanged.
"""
function smooth_ε(plan::SmoothingPlan, mat_vals)
    smoothed_vals = mapreduce(vcat, 1:npixels(plan)) do k
        _apply_plan_single(plan, mat_vals, k)
    end
    return reshape(smoothed_vals, (3, 3, 3, plan.gridsize...))
end
