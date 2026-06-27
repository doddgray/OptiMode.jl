export corner_sinds, proc_sinds, smooth_ε, smooth_ε_single, smooth_scalar, smooth_scalar_single, vec3D 

"""
    corner_sinds(shapes, points)

Return a Tuple of the index of the foreground shape index (in `shapes`) at each point in `points`.

The index `length(shapes)+1` will be returned for points in `points` outside of all shapes in `shapes`.

# Examples
```julia-repl

julia> shapes = ( Box(...), Polygon{5}(...), Sphere{2}(...), regpoly(3,...) );		  # a collection of four shapes

julia> points = ( SVector{2}(0.1,3.3), SVector{2}(0.5,-2.1) , SVector{2}(-1.1,0.3) ); # a collection of three points

julia> corner_sinds( shapes, points )
(2,5,1)		# `shapes[2]` and `shapes[1]` are the foreground shapes at `xyz[1]` and `xyz[3]`, 
			# and `xyz[2]` is outside each shape in `shapes` 
```
"""
function corner_sinds(shapes::Tuple,points::NTuple{NC,SVector{ND,T}}) where {NC,ND,T<:Real}
	ps = pairs(shapes)
	lsp1 = length(shapes) + 1
	map(points) do p
		let ps=ps, lsp1=lsp1
			for (i, a) in ps #pairs(s)
				in(p,a)::Bool && return i
			end
			return lsp1
		end
	end
end

@non_differentiable corner_sinds(shapes::Any,corners::Any)

"""
    proc_sinds(corner_sinds::NTuple{4}) / proc_sinds(corner_sinds::NTuple{8})

Classify a pixel (voxel) from the foreground-shape indices at its 4 (8) corners,
encoding the result so the smoothing kernel can branch cheaply:

- all corners in one shape → `(s, 0, 0, …)`: pixel is *uniform*, copy material `s`;
- exactly two shapes present → `(s₁, s₂, 0, …)`: pixel straddles *one interface*,
  apply Kottke averaging between materials `s₁` and `s₂`;
- otherwise the corner indices are returned unchanged (≥3 materials meet — a corner
  pixel) and the caller falls back to a naive arithmetic average.
"""
function proc_sinds(corner_sinds::NTuple{4,T}) where T<:Int
	unq = unique(corner_sinds)
	sinds_proc = isone(lastindex(unq)) ? (first(unq),0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) : corner_sinds )
	return sinds_proc
end

function proc_sinds(corner_sinds::NTuple{8,T}) where T<:Int
	unq = unique(corner_sinds)
	sinds_proc = isone(lastindex(unq)) ? (first(unq),0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) : corner_sinds )
	return sinds_proc
end

@non_differentiable proc_sinds(crnr_sinds::Any)

"""
    smooth_ε_single(shapes, mat_vals, minds, crnrs) -> Vector (length 27)

Smoothed `(ε, ∂ωε, ∂²ωε)` data for the single pixel whose corner positions are
`crnrs`. Uniform pixels copy their material's column of `mat_vals`; single-interface
pixels locate the interface (`surfpt_nearby`), compute the material-1 volume fraction
(`volfrac`) and apply the dispersion-propagating Kottke kernel
[`εₑ_∂ωεₑ_∂²ωεₑ`](@ref); pixels where ≥3 materials meet use a naive corner average.
See [`smooth_ε`](@ref) for the argument conventions.
"""
# In-place per-pixel smoothing: write the 27 smoothed values (ε, ∂ωε, ∂²ωε) for one pixel
# into `dest`, computing the geometry (classification, surface point/normal, fill fraction)
# inline so geometry-*parameter* AD (Dual-valued shapes) still flows through. This is the
# allocation-free core of the assembly rewrite — every pixel writes into a column of a
# preallocated output instead of allocating its own vector for a `mapreduce(vcat)`.
# Interface-pixel geometry: locate the surface (`surfpt_nearby`), the fill fraction
# (`volfrac`) and the interface frame (`normcart`) for the `sidx1↔sidx2` interface. The
# `shapes[sidx1]` index into the *heterogeneous* shapes tuple produces a small `Union`
# (`Polygon`/`Cuboid`/…) that Enzyme's strict type analysis rejects. This quantity is
# constant w.r.t. the material data, so the function is marked `EnzymeRules.inactive`
# (extension) — keeping the Union out of Enzyme's differentiated path — while ForwardDiff
# still propagates Dual *shape* coordinates through it (it ignores the marker), so
# geometry-parameter AD is unaffected.
# `@noinline` so the call survives for `EnzymeRules.inactive` to act on (inlining would
# expose the `Union` to Enzyme's type analysis again).
@noinline function _interface_geometry(shapes, sidx1::Int, crnrs::NTuple{NC,SVector{ND,T}}) where {NC,ND,T<:Real}
    xyz = sum(crnrs) / NC
    r₀_n⃗ = surfpt_nearby(xyz, shapes[sidx1])
    rvol = volfrac((vxlmin(crnrs), vxlmax(crnrs)), last(r₀_n⃗), first(r₀_n⃗))
    return rvol, normcart(vec3D(last(r₀_n⃗)))
end

@inline function smooth_ε_single!(dest, shapes, mat_vals, minds, crnrs::NTuple{NC,SVector{ND,T}}) where {NC,ND,T<:Real}
    ps = proc_sinds(corner_sinds(shapes, crnrs))
    if iszero(ps[2])
        @inbounds @views dest .= mat_vals[:, minds[first(ps)]]
    elseif iszero(ps[3])
        sidx1 = ps[1]; sidx2 = ps[2]
        rvol, S = _interface_geometry(shapes, sidx1, crnrs)
        @inbounds @views dest .= εₑ_∂ωεₑ_∂²ωεₑ(rvol, S, mat_vals[:, minds[sidx1]], mat_vals[:, minds[sidx2]])
    else
        fill!(dest, zero(eltype(dest)))
        @inbounds for i in ps
            @views dest .+= mat_vals[:, minds[i]]
        end
        dest ./= NC
    end
    return dest
end

function smooth_ε_single(shapes,mat_vals,minds,crnrs::NTuple{NC,SVector{ND,T}}) where{NC,ND,T<:Real}
    dest = Vector{_smooth_eltype(mat_vals, shapes)}(undef, 27)
    return smooth_ε_single!(dest, shapes, mat_vals, minds, crnrs)
end

"""
    smooth_ε(shapes, mat_vals, minds, grid) -> Array{Float64,(2+1+ND)}

Render a shape-based geometry to smoothed dielectric-tensor fields on `grid`.

Arguments:
- `shapes`: foreground-to-background-ordered tuple/list of `MaterialShape`s (the first
  shape containing a point wins); points outside all shapes belong to the background
  material `length(shapes)+1`.
- `mat_vals`: matrix whose `j`-th column is material `j`'s data
  `vcat(vec(ε), vec(∂ωε), vec(∂²ωε))` (27 entries, e.g. a column of
  `MaterialDispersion._f_ε_mats(mats, (:ω,))([ω])`).
- `minds`: map from shape/background slot to column of `mat_vals`,
  e.g. `(1, 2, 3)` for two shapes + background.

Returns a `(3, 3, 3, size(grid)...)` array whose third axis indexes
``(ε, ∂ε/∂ω, ∂^2ε/∂ω^2)``; each pixel is processed by [`smooth_ε_single`](@ref):
exact material data in uniform pixels, anisotropic Kottke averaging
([`avg_param`](@ref)) with exact derivative propagation across single material
interfaces.

Pixels are assembled into a preallocated `(3, 3, 3, size(grid)...)` array (each writes
its 27 values into one column of a `(27, N)` buffer) rather than the old per-pixel
`mapreduce(vcat)`, which is ~20× faster and ~50× lower-allocation at large grids
(10⁵–10⁷ points). Pass `threaded=true` to fill the independent pixels across all Julia
threads (`julia -t` on a compute node). Differentiable in `mat_vals` in forward mode
(ForwardDiff/Enzyme) and reverse mode (Zygote, via the `rrule`).

```julia
sm   = smooth_ε(shapes, mat_vals, (1, 2), grid)
ε    = copy(selectdim(sm, 3, 1))     # (3,3,Nx,Ny)
∂ωε  = copy(selectdim(sm, 3, 2))
ε⁻¹  = sliceinv_3x3(ε)               # input for solve_k
```
"""
function smooth_ε(shapes, mat_vals, minds, grid::Grid{ND,TG}; threaded::Bool=false) where {ND,TG<:Real}
	crn = corners(grid)
	flat = Matrix{_smooth_eltype(mat_vals, shapes)}(undef, 27, length(crn))
	_fill_smooth!(flat, shapes, mat_vals, minds, crn, threaded)
	return reshape(flat, (3, 3, 3, size(grid)...))
end

# NB: `smooth_ε` material-data gradients work in every backend, forward and reverse —
# ForwardDiff & Enzyme (fwd & rev) differentiate the preallocated fill natively (the Kottke
# kernel propagates a small 2nd-order Taylor jet, see `kottke.jl`); Zygote (and Mooncake via
# the extension bridge) consume the ChainRules `rrule` below, which routes material-data
# gradients through the frozen-geometry plan pullback. Geometry-*parameter* gradients go
# through the ForwardDiff (forward) primal, which keeps Dual shape coordinates.

# Output element type: material data and geometry coordinates (possibly AD Duals when
# differentiating shape parameters) both flow into the kernel, so the result promotes them.
_smooth_eltype(mat_vals, shapes) = promote_type(eltype(mat_vals), _shapes_eltype(shapes), Float64)
_shapes_eltype(shapes) = mapreduce(_shape_eltype, promote_type, shapes; init=Float64)
function _shape_eltype(ms)
	s = hasproperty(ms, :shape) ? ms.shape : ms
	T = Float64
	for f in fieldnames(typeof(s))
		v = getfield(s, f)
		v isa AbstractArray && eltype(v) <: Number && (T = promote_type(T, eltype(v)))
	end
	return T
end
@non_differentiable _smooth_eltype(::Any, ::Any)

# Fill a preallocated (27, N) buffer, one pixel per column. Disjoint columns ⇒ the threaded
# path is race-free; chunked `@spawn` keeps each task's pixels contiguous for cache locality.
function _fill_smooth!(flat, shapes, mat_vals, minds, crn, threaded::Bool)
	N = length(crn)
	if threaded && Threads.nthreads() > 1 && N > 1
		nt = min(Threads.nthreads(), N)
		@sync for ch in Iterators.partition(1:N, cld(N, nt))
			Threads.@spawn for v in ch
				@inbounds @views smooth_ε_single!(flat[:, v], shapes, mat_vals, minds, crn[v])
			end
		end
	else
		@inbounds for v in 1:N
			@views smooth_ε_single!(flat[:, v], shapes, mat_vals, minds, crn[v])
		end
	end
	return flat
end

# Reverse rule: the mutating assembly is not reverse-differentiable directly, so route
# material-data gradients through the frozen-geometry plan pullback (geometry is
# `@non_differentiable`, hence the `NoTangent`s for shapes/minds/grid).
function ChainRulesCore.rrule(::typeof(smooth_ε), shapes, mat_vals::AbstractMatrix, minds,
		grid::Grid; threaded::Bool=false)
	plan = smoothing_plan(shapes, minds, grid)
	y, plan_pb = ChainRulesCore.rrule(smooth_ε, plan, mat_vals; threaded)
	smooth_ε_pullback(Ȳ) = (NoTangent(), NoTangent(), plan_pb(Ȳ)[3], NoTangent(), NoTangent())
	return y, smooth_ε_pullback
end

"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

function vec3D(v::AbstractVector{T}) where T
	vout = (length(v)==3 ? v : SVector{3,T}(v[1],v[2],0.))
	return vout
end
vec3D(v::SVector{3}) = v
vec3D(v::SVector{2}) = SVector(v[1],v[2],zero(v[1]))
vec3D(v::SVector{1}) = SVector(v[1],zero(v[1]),zero(v[1]))


"""
    smooth_scalar_single(shapes, vals, minds, crnrs)

Volume-fraction-weighted value of a scalar material property (e.g. the Kerr
coefficient `n₂`) for one pixel/voxel with corners `crnrs`. Pixels inside a single
material take that material's value; interface pixels mix the two values linearly by
fill fraction (first-order accurate, appropriate for perturbative properties).
"""
function smooth_scalar_single(shapes, vals, minds, crnrs::NTuple{NC,SVector{ND,T}}) where {NC,ND,T<:Real}
    ps = proc_sinds(corner_sinds(shapes, crnrs))
    if iszero(ps[2])
        return Float64(vals[minds[first(ps)]])
    elseif iszero(ps[3])
        xyz = sum(crnrs) / NC
        r₀_n⃗ = surfpt_nearby(xyz, shapes[ps[1]])
        rvol = volfrac((vxlmin(crnrs), vxlmax(crnrs)), last(r₀_n⃗), first(r₀_n⃗))
        return rvol * Float64(vals[minds[ps[1]]]) + (1 - rvol) * Float64(vals[minds[ps[2]]])
    else
        return sum(i -> Float64(vals[minds[i]]), ps) / NC
    end
end

"""
    smooth_scalar(shapes, vals, minds, grid) -> Array{Float64}

Map a per-material scalar property (`vals`, indexed like the material columns used by
[`smooth_ε`](@ref), with `minds` assigning shapes → materials and the final entry the
background) onto the spatial grid with linear volume-fraction mixing at interfaces.
Used e.g. to build the Kerr-coefficient map `n₂(x,y)` for power-dependent mode solves.
"""
function smooth_scalar(shapes, vals::AbstractVector{<:Real}, minds, grid::Grid{ND,TG}) where {ND,TG<:Real}
    return reshape(map(crnrs -> smooth_scalar_single(shapes, vals, minds, crnrs), corners(grid)), size(grid))
end
