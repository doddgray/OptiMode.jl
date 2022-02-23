export corner_sinds, corner_sinds!, proc_sinds, proc_sinds!, smooth, smooth!, _f_smooth, _f_smooth!
export vxl_minmax, hybridize, εₘₐₓ, ñₘₐₓ, nₘₐₓ, kguess # utility functions for automatic good guesses, move to geometry or solve?

# export εₛ, εₛ⁻¹,  corner_sinds, corner_sinds!, proc_sinds, proc_sinds!, avg_param, S_rvol, _εₛ⁻¹_init, _εₛ_init, nngₛ, nngₛ⁻¹
# export ngvdₛ, ngvdₛ⁻¹, εₛ_nngₛ_ngvdₛ, εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹, vxl_minmax, hybridize
# export make_εₛ⁻¹, make_εₛ⁻¹_fwd, make_KDTree # legacy junk to remove or update

# export kottke_smoothing, volfrac_smoothing

function corner_sinds(shapes::Vector{S},xyzc::AbstractArray{T}) where {S<:GeometryPrimitives.Shape,T<:SVector{N}} where N #where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
	ps = pairs(shapes)
	lsp1 = length(shapes) + 1
	map(xyzc) do p
		let ps=ps, lsp1=lsp1
			for (i, a) in ps #pairs(s)
				in(p::T,a::S)::Bool && return i
			end
			return lsp1
		end
	end
end

function corner_sinds!(corner_sinds,shapes::Vector{S},xyzc::AbstractArray{T}) where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
	ps = pairs(shapes)
	lsp1 = length(shapes) + 1
	map!(corner_sinds,xyzc) do p
		let ps=ps, lsp1=lsp1
			for (i, a) in ps #pairs(s)
				in(p::T,a::S)::Bool && return i
			end
			return lsp1
		end
	end
end

@non_differentiable corner_sinds(shapes,xyzc)

function proc_sinds(corner_sinds::AbstractArray{Int,2})
	unq = [0,0]
	sinds_proc = fill((0,0,0,0),size(corner_sinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds(geom::Vector{<:Shape},grid::Grid{2})
	csinds = corner_sinds(geom,x⃗c(grid)) # corner_sinds(geom,x⃗(grid),x⃗c(grid))
	unq = [0,0]
	sinds_proc = fill((0,0,0,0),size(csinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		csinds[I],
								csinds[I+CartesianIndex(1,0)],
								csinds[I+CartesianIndex(0,1)],
								csinds[I+CartesianIndex(1,1)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
				( 	csinds[I],
					csinds[I+CartesianIndex(1,0)],
					csinds[I+CartesianIndex(0,1)],
					csinds[I+CartesianIndex(1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds!(sinds_proc::AbstractArray{T,2},corner_sinds::AbstractArray{Int,2}) where T
	unq = [0,0]
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)]
				)
		)
	end
end


function proc_sinds(corner_sinds::AbstractArray{Int,3})
	unq = [0,0]
	sinds_proc = fill((0,0,0,0,0,0,0,0),size(corner_sinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0,0)],
								corner_sinds[I+CartesianIndex(0,1,0)],
								corner_sinds[I+CartesianIndex(1,1,0)]
			  				]
		# unique!( unq )
		unique!( sort!(unq) )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds(geom::Vector{<:Shape},grid::Grid{3})
	csinds = corner_sinds(geom,x⃗c(grid)) # corner_sinds(geom,x⃗(grid),x⃗c(grid))
	unq = [0,0]
	sinds_proc = fill((0,0,0,0,0,0,0,0),size(csinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		csinds[I],
								csinds[I+CartesianIndex(1,0,0)],
								csinds[I+CartesianIndex(0,1,0)],
								csinds[I+CartesianIndex(1,1,0)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) :
				( 	csinds[I],
					csinds[I+CartesianIndex(1,0,0)],
					csinds[I+CartesianIndex(0,1,0)],
					csinds[I+CartesianIndex(1,1,0)],
					csinds[I+CartesianIndex(0,0,1)],
					csinds[I+CartesianIndex(1,0,1)],
					csinds[I+CartesianIndex(0,1,1)],
					csinds[I+CartesianIndex(1,1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds!(sinds_proc::AbstractArray{T,3},corner_sinds::AbstractArray{Int,3}) where T
	unq = [0,0]
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0,0)],
								corner_sinds[I+CartesianIndex(0,1,0)],
								corner_sinds[I+CartesianIndex(1,1,0)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)]
				)
		)
	end
end

# @non_differentiable proc_sinds(geom,grid)

vec3D(v::SVector{3}) = v
vec3D(v::SVector{2}) = @inbounds SVector(v[1],v[2],zero(v[1]))
vec3D(v::SVector{1}) = @inbounds SVector(v[1],zero(v[1]),zero(v[1]))

# function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals,xx,vxl_min,vxl_max) where {NI,TI<:Int}
@inline n_voxel_verts(grid::Grid{2}) = 4
@inline n_voxel_verts(grid::Grid{3}) = 8

function smooth!(εg,∂ωεg,∂²ωεg, shapes,ε,∂ωε,∂²ωε,grid)
    xyz, xyzc = x⃗(grid), x⃗c(grid);			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
    vxlmin,vxlmax = vxl_minmax(xyzc);
    sinds = proc_sinds(corner_sinds(shapes,xyzc));
    minds   =   matinds(shapes);
    n_vxl_verts = n_voxel_verts(grid);
    @inbounds for grididx∈eachindex(grid)
        shape_inds = @inbounds sinds[grididx]
        if iszero(shape_inds[2]) # if only one shape detected, assign (ε, ∂ωε, ∂²ωε) from that shape's material model
            εg[:,:,grididx] = ε[minds[first(shape_inds)]]
            ∂ωεg[:,:,grididx] = ∂ωε[minds[first(shape_inds)]]
            ∂²ωεg[:,:,grididx] = ∂²ωε[minds[first(shape_inds)]]
        elseif iszero(shape_inds[3]) # if two shapes detected, use Kottke averaging to smooth (ε, ∂ωε, ∂²ωε)
            idx1, idx2 = shape_inds[1], shape_inds[2]
            r₀,n⃗ = surfpt_nearby(xyz[grididx], shapes[idx1])    # find normal vector `n⃗` of the interface between materials 1 & 2 (pointing 1→2)
            rvol = volfrac((vxlmin[grididx],vxlmax[grididx]),n⃗,r₀)      # find the fill fraction of material 1 in the smoothing voxel
            S = normcart(vec3D(n⃗) / norm(n⃗));
            εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12 = εₑ_∂ωεₑ_∂²ωεₑ(rvol,S,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2])
            εg[:,:,grididx] = εₑ_12
            ∂ωεg[:,:,grididx] = ∂ω_εₑ_12
            ∂²ωεg[:,:,grididx] = ∂ω²_εₑ_12
        else # if more than two shapes detected, use naive averaging to smooth (ε, ∂ωε, ∂²ωε)
            εg[:,:,grididx] = mapreduce(i->ε[minds[i]],+,shape_inds) / n_vxl_verts  
            ∂ωεg[:,:,grididx] = mapreduce(i->∂ωε[minds[i]],+,shape_inds) / n_vxl_verts  
            ∂²ωεg[:,:,grididx] = mapreduce(i->∂²ωε[minds[i]],+,shape_inds) / n_vxl_verts  
        end
    end;

    return nothing
end

function smooth(shapes::Vector{Shape{N1,N2,TD,TF1}},ε::AbstractArray{TF2},∂ωε,∂²ωε,grid) where{N1,N2,TD,TF1,TF2}
    εg = zeros(TF1,3,3,size(grid)...); #[zeros(3,3) for I ∈ eachindex(grid)];
    ∂ωεg = copy(εg);
    ∂²ωεg = copy(εg);
    smooth!(εg,∂ωεg,∂²ωεg, shapes,ε,∂ωε,∂²ωε,grid);
    return εg, ∂ωεg, ∂²ωεg
end

@inline function _f_smooth!(geom_fn,f_ε_mats,np_geom,np_mats,n_mats)    
    # np_mats =   length( p_mats_syms);
    # np_geom =   length(p_geom0);
    # mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
    # n_mats = length(mats);
    # f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
    # f_ε_mats(rand(np_mats));
    # function f_smooth!(εg,∂ωεg,∂²ωεg,p,grid)
    #     p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
    #     smooth!(εg,∂ωεg,∂²ωεg,geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
    #     return nothing
    # end
    # return f_smooth!
    (εg,∂ωεg,∂²ωεg,p,grid)->smooth!(εg,∂ωεg,∂²ωεg,geom_fn(p[(np_mats+1):(np_mats+np_geom)]),ε_views(f_ε_mats(p[1:np_mats]),n_mats)...,grid)
end

@inline function _f_smooth(geom_fn,f_ε_mats,np_geom,np_mats,n_mats)    
    # np_mats =   length( p_mats_syms);
    # np_geom =   length(p_geom0);
    # mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
    # n_mats = length(mats);
    # f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
    # f_ε_mats(rand(np_mats));
    # function f_smooth(p,grid)
    #     p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
    #     return smooth(geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
    # end
    # return f_smooth
    (p,grid)->smooth(geom_fn(p[(np_mats+1):(np_mats+np_geom)]),ε_views(f_ε_mats(p[1:np_mats]),n_mats)...,grid)
end


 
"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

function hybridize(A::AbstractArray{T,4},grid::Grid{2}) where T<:Number
	HybridArray{Tuple{3,3,Dynamic(),Dynamic()},T,4,4}(A)
end

function hybridize(A::AbstractArray{T,5},grid::Grid{3}) where T<:Number
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5}(A)
end


ñₘₐₓ(ε⁻¹::AbstractArray{<:SMatrix})::Float64 = √(maximum(3 ./ tr.(ε⁻¹)))
nₘₐₓ(ε::AbstractArray{<:SMatrix})::Float64 = √(maximum(reinterpret(Float64,ε)))
# function nₘₐₓ(ε::AbstractArray{T,4})::T where T<:Real
# 	sqrt(inv(minimum(hcat([ε⁻¹[a,a,:,:] for a=1:3]...))))
# end
function ñₘₐₓ(ε⁻¹::AbstractArray{T,4})::T where T<:Real
	sqrt(inv(minimum(hcat([ε⁻¹[a,a,:,:] for a=1:3]...))))
end

function vxl_minmax(xyzc::AbstractArray{TV,2}) where {TV<:AbstractVector}
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end]
	return vxl_min,vxl_max
end

function vxl_minmax(xyzc::AbstractArray{TV,3}) where {TV<:AbstractVector}
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	return vxl_min,vxl_max
end

"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""




