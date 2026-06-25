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
function smooth_ε_single(shapes,mat_vals,minds,crnrs::NTuple{NC,SVector{ND,T}}) where{NC,ND,T<:Real}
    ps = proc_sinds(corner_sinds(shapes,crnrs))
    if iszero(ps[2])
        return mat_vals[:,minds[first(ps)]]
	elseif iszero(ps[3])
        sidx1   =   ps[1]
        sidx2   =   ps[2]
        xyz     =   sum(crnrs) / NC # sum(crnrs)/NC
        r₀_n⃗    =   surfpt_nearby(xyz, shapes[sidx1])
        r₀      =   first(r₀_n⃗)
        n⃗       =   last(r₀_n⃗)
        rvol    =   volfrac((vxlmin(crnrs),vxlmax(crnrs)),n⃗,r₀)
        return εₑ_∂ωεₑ_∂²ωεₑ(
            rvol,
            normcart(vec3D(n⃗)),
            mat_vals[:,minds[sidx1]],
            mat_vals[:,minds[sidx2]],
        )
    else
        return sum(i->mat_vals[:,minds[i]],ps) / NC  # naive averaging to be used
    end
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

```julia
sm   = smooth_ε(shapes, mat_vals, (1, 2), grid)
ε    = copy(selectdim(sm, 3, 1))     # (3,3,Nx,Ny)
∂ωε  = copy(selectdim(sm, 3, 2))
ε⁻¹  = sliceinv_3x3(ε)               # input for solve_k
```
"""
function smooth_ε(shapes,mat_vals,minds,grid::Grid{ND,TG}) where {ND, TG<:Real}
	smoothed_vals = mapreduce(vcat,corners(grid)) do crnrs
		smooth_ε_single(shapes,mat_vals,minds,crnrs)
	end
	return reshape(smoothed_vals,(3,3,3,size(grid)...))
end

# NB: `smooth_ε` material-data gradients work in every backend, forward and reverse —
# ForwardDiff, Zygote, Mooncake, and Enzyme (fwd & rev). The Kottke kernel propagates a
# small 2nd-order Taylor jet through closed-form transforms (see `kottke.jl`), so the
# smoothing is type-stable and AD-friendly; Zygote consumes a ChainRules `rrule` on the
# kernel while the others differentiate the jet natively. Geometry-*parameter* gradients
# go through ForwardDiff (forward) / Mooncake (reverse, per-pixel).

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
