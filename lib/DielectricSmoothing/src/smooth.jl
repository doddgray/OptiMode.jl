export corner_sinds, proc_sinds, smooth_ε, smooth_ε_single, vec3D 

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

function smooth_ε(shapes,mat_vals,minds,grid::Grid{ND,TG}) where {ND, TG<:Real} 
	smoothed_vals = mapreduce(vcat,corners(grid)) do crnrs
		smooth_ε_single(shapes,mat_vals,minds,crnrs)
	end
	return reshape(smoothed_vals,(3,3,3,size(grid)...))
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
