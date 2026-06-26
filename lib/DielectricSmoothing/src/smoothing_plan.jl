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
#  Note on the size of the win: once the output assembly was made allocation-free (a
#  preallocated (27, N) fill replacing the old `mapreduce(vcat)` — ~20× faster), the
#  geometry pass became the *dominant* remaining cost of `smooth_ε`. So caching it removes
#  the now-largest term across a frequency sweep (and lets the EME (cell × ω) batch reuse
#  one scaffold per cross-section). Both the plan apply and the direct `smooth_ε` use the
#  same fast assembly and accept `threaded=true`.
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

# In-place per-pixel application of the plan: write the 27 smoothed values (ε, ∂ωε, ∂²ωε)
# into `dest`. Mirrors the three branches of `smooth_ε_single`, geometry already resolved.
@inline function _apply_plan_into!(dest, p::SmoothingPlan, mat_vals, v::Int)
    kd = p.kind[v]
    if kd == _PLAN_UNIFORM
        @inbounds @views dest .= mat_vals[:, p.a[v]]
    elseif kd == _PLAN_INTERFACE
        @inbounds @views dest .= εₑ_∂ωεₑ_∂²ωεₑ(p.rvol[v], p.S[v], mat_vals[:, p.a[v]], mat_vals[:, p.b[v]])
    else
        @inbounds cols = p.multi[p.a[v]]
        fill!(dest, zero(eltype(dest)))
        @inbounds for c in cols
            @views dest .+= mat_vals[:, c]
        end
        dest ./= p.NC
    end
    return dest
end

function _fill_plan!(flat, plan::SmoothingPlan, mat_vals, threaded::Bool)
    N = npixels(plan)
    if threaded && Threads.nthreads() > 1 && N > 1
        nt = min(Threads.nthreads(), N)
        @sync for ch in Iterators.partition(1:N, cld(N, nt))
            Threads.@spawn for v in ch
                @views _apply_plan_into!(flat[:, v], plan, mat_vals, v)
            end
        end
    else
        for v in 1:N
            @views _apply_plan_into!(flat[:, v], plan, mat_vals, v)
        end
    end
    return flat
end

"""
    smooth_ε(plan::SmoothingPlan, mat_vals; threaded=false) -> Array

Apply a precomputed [`SmoothingPlan`](@ref) to per-frequency material data `mat_vals`
(the same `(27, n_materials)` column matrix [`smooth_ε`](@ref) takes), returning the
identical `(3, 3, 3, size(grid)...)` smoothed-dielectric array — but running only the
Kottke kernel, with no geometry queries. Pixels are written into a preallocated `(27, N)`
buffer (no per-pixel allocation), optionally across threads (`threaded=true`).
Differentiable in `mat_vals` (forward via the primal, reverse via the `rrule` below), so
material/ω gradients flow through unchanged.
"""
function smooth_ε(plan::SmoothingPlan{ND}, mat_vals::AbstractMatrix; threaded::Bool=false) where {ND}
    flat = Matrix{promote_type(eltype(mat_vals), Float64)}(undef, 27, npixels(plan))
    _fill_plan!(flat, plan, mat_vals, threaded)
    return reshape(flat, (3, 3, 3, plan.gridsize...))
end

# Pixel-by-pixel material-data VJP. Uniform/multi pixels select/average columns (trivial);
# interface pixels apply the Kottke kernel's VJP through a small ForwardDiff Jacobian (as the
# kernel's own `rrule` does). Threaded with per-chunk accumulators — each task owns a buffer
# over a disjoint pixel range, summed at the end (race-free, no atomics).
function _plan_vjp(plan::SmoothingPlan, M::AbstractMatrix{T}, Ȳf, threaded::Bool) where {T}
    N = npixels(plan)
    accum! = (into, v) -> begin
        ȳ = collect(@view Ȳf[:, v])
        kd = plan.kind[v]
        if kd == _PLAN_UNIFORM
            @inbounds @views into[:, plan.a[v]] .+= ȳ
        elseif kd == _PLAN_INTERFACE
            a = plan.a[v]; b = plan.b[v]
            c1 = collect(@view M[:, a]); c2 = collect(@view M[:, b])
            r = plan.rvol[v]; S = plan.S[v]
            J1 = ForwardDiff.jacobian(c -> εₑ_∂ωεₑ_∂²ωεₑ(r, S, c, c2), c1)
            J2 = ForwardDiff.jacobian(c -> εₑ_∂ωεₑ_∂²ωεₑ(r, S, c1, c), c2)
            @inbounds @views into[:, a] .+= transpose(J1) * ȳ
            @inbounds @views into[:, b] .+= transpose(J2) * ȳ
        else
            @inbounds for c in plan.multi[plan.a[v]]
                @views into[:, c] .+= ȳ ./ plan.NC
            end
        end
    end
    if threaded && Threads.nthreads() > 1 && N > 1
        nt = min(Threads.nthreads(), N)
        chunks = collect(Iterators.partition(1:N, cld(N, nt)))
        bufs = Vector{Matrix{T}}(undef, length(chunks))
        @sync for (ci, ch) in enumerate(chunks)
            Threads.@spawn begin
                b = zeros(T, size(M))
                for v in ch; accum!(b, v); end
                bufs[ci] = b
            end
        end
        return reduce(+, bufs)
    else
        Δ = zeros(T, size(M))
        for v in 1:N; accum!(Δ, v); end
        return Δ
    end
end

# Reverse rule for the cached apply (Zygote, and Enzyme/Mooncake via the extension bridge).
# The plan (geometry) is constant ⇒ NoTangent; material-data cotangents come from `_plan_vjp`.
function ChainRulesCore.rrule(::typeof(smooth_ε), plan::SmoothingPlan, mat_vals::AbstractMatrix;
        threaded::Bool=false)
    y = smooth_ε(plan, mat_vals; threaded)
    M = collect(mat_vals)
    function smooth_ε_plan_pullback(Ȳ)
        Ȳf = reshape(collect(ChainRulesCore.unthunk(Ȳ)), 27, npixels(plan))
        Δ = _plan_vjp(plan, M, Ȳf, threaded)
        return (NoTangent(), NoTangent(), Δ)
    end
    return y, smooth_ε_plan_pullback
end
