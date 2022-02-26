# using StaticArrays
# using GeometryPrimitives
# using OptiMode

# include("cse.jl")
# include("f_epse.jl")

# function corner_sinds(shapes::Vector{S},xyzc::AbstractArray{T}) where {S<:GeometryPrimitives.Shape,T<:SVector{N}} where N #where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
#     ps = pairs(shapes)
#     lsp1 = length(shapes) + 1
#     map(xyzc) do p
#             let ps=ps, lsp1=lsp1
#                     for (i, a) in ps #pairs(s)
#                             in(p::T,a::S)::Bool && return i
#                     end
#                     return lsp1
#             end
#     end
# end

# function proc_sinds(corner_sinds::AbstractArray{Int,2})
#     unq = [0,0]
#     sinds_proc = fill((0,0,0,0),size(corner_sinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
#     @inbounds for I ∈ CartesianIndices(sinds_proc)
#              unq = [                corner_sinds[I],
#                                                             corner_sinds[I+CartesianIndex(1,0)],
#                                                             corner_sinds[I+CartesianIndex(0,1)],
#                                                             corner_sinds[I+CartesianIndex(1,1)]
#                                                       ]
#             unique!( unq )
#             sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
#                     ( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
#                             (         corner_sinds[I],
#                                     corner_sinds[I+CartesianIndex(1,0)],
#                                     corner_sinds[I+CartesianIndex(0,1)],
#                                     corner_sinds[I+CartesianIndex(1,1)]
#                             )
#             )
#     end
#     return sinds_proc
# end

# function vxl_minmax(xyzc::AbstractArray{TV,2}) where {TV<:AbstractVector}
#     vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1)]
#     vxl_max = @view xyzc[min(2,end):end,min(2,end):end]
#     return vxl_min,vxl_max
# end

# function vec3D(v::AbstractVector{T}) where T
#     vout = (length(v)==3 ? v : SVector{3,T}(v[1],v[2],0.))
#     return vout
# end

# vec3D(v::SVector{3}) = 3
# vec3D(v::SVector{2}) = @inbounds SVector(v[1],v[2],zero(v[1]))
# vec3D(v::SVector{1}) = @inbounds SVector(v[1],zero(v[1]),zero(v[1]))

# # function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# @inline n_voxel_verts(grid::Grid{2}) = 4
# @inline n_voxel_verts(grid::Grid{3}) = 8

# function smooth!(εg,∂ωεg,∂²ωεg, shapes,ε,∂ωε,∂²ωε,grid)
#     xyz, xyzc = x⃗(grid), x⃗c(grid);			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
#     vxlmin,vxlmax = vxl_minmax(xyzc);
#     sinds = proc_sinds(corner_sinds(shapes,xyzc));
#     minds   =   matinds(shapes);
#     n_vxl_verts = n_voxel_verts(grid);
#     @inbounds for grididx∈eachindex(grid)
#         shape_inds = @inbounds sinds[grididx]
#         if iszero(shape_inds[2]) # if only one shape detected, assign (ε, ∂ωε, ∂²ωε) from that shape's material model
#             εg[:,:,grididx] = ε[minds[first(shape_inds)]]
#             ∂ωεg[:,:,grididx] = ∂ωε[minds[first(shape_inds)]]
#             ∂²ωεg[:,:,grididx] = ∂²ωε[minds[first(shape_inds)]]
#         elseif iszero(shape_inds[3]) # if two shapes detected, use Kottke averaging to smooth (ε, ∂ωε, ∂²ωε)
#             idx1, idx2 = shape_inds[1], shape_inds[2]
#             r₀,n⃗ = surfpt_nearby(xyz[grididx], shapes[idx1])    # find normal vector `n⃗` of the interface between materials 1 & 2 (pointing 1→2)
#             rvol = volfrac((vxlmin[grididx],vxlmax[grididx]),n⃗,r₀)      # find the fill fraction of material 1 in the smoothing voxel
#             S = normcart(vec3D(n⃗) / norm(n⃗));
#             εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12 = εₑ_∂ωεₑ_∂²ωεₑ(rvol,S,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2])
#             εg[:,:,grididx] = εₑ_12
#             ∂ωεg[:,:,grididx] = ∂ω_εₑ_12
#             ∂²ωεg[:,:,grididx] = ∂ω²_εₑ_12
#         else # if more than two shapes detected, use naive averaging to smooth (ε, ∂ωε, ∂²ωε)
#             εg[:,:,grididx] = mapreduce(i->ε[minds[i]],+,shape_inds) / n_vxl_verts  
#             ∂ωεg[:,:,grididx] = mapreduce(i->∂ωε[minds[i]],+,shape_inds) / n_vxl_verts  
#             ∂²ωεg[:,:,grididx] = mapreduce(i->∂²ωε[minds[i]],+,shape_inds) / n_vxl_verts  
#         end
#     end;

#     return nothing
# end

# function smooth(shapes::Vector{Shape{N1,N2,TD,TF1}},ε::AbstractArray{TF2},∂ωε,∂²ωε,grid) where{N1,N2,TD,TF1,TF2}
#     εg = zeros(TF1,3,3,size(grid)...); #[zeros(3,3) for I ∈ eachindex(grid)];
#     ∂ωεg = copy(εg);
#     ∂²ωεg = copy(εg);
#     smooth!(εg,∂ωεg,∂²ωεg, shapes,ε,∂ωε,∂²ωε,grid);
#     return εg, ∂ωεg, ∂²ωεg
# end

# @inline function _f_smooth!(geom_fn,f_ε_mats,np_geom,np_mats,n_mats)    
#     # np_mats =   length( p_mats_syms);
#     # np_geom =   length(p_geom0);
#     # mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
#     # n_mats = length(mats);
#     # f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
#     # f_ε_mats(rand(np_mats));
#     # function f_smooth!(εg,∂ωεg,∂²ωεg,p,grid)
#     #     p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
#     #     smooth!(εg,∂ωεg,∂²ωεg,geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
#     #     return nothing
#     # end
#     # return f_smooth!
#     (εg,∂ωεg,∂²ωεg,p,grid)->smooth!(εg,∂ωεg,∂²ωεg,geom_fn(p[(np_mats+1):(np_mats+np_geom)]),ε_views(f_ε_mats(p[1:np_mats]),n_mats)...,grid)
# end

# @inline function _f_smooth(geom_fn,f_ε_mats,np_geom,np_mats,n_mats)    
#     # np_mats =   length( p_mats_syms);
#     # np_geom =   length(p_geom0);
#     # mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
#     # n_mats = length(mats);
#     # f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
#     # f_ε_mats(rand(np_mats));
#     # function f_smooth(p,grid)
#     #     p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
#     #     return smooth(geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
#     # end
#     # return f_smooth
#     (p,grid)->smooth(geom_fn(p[(np_mats+1):(np_mats+np_geom)]),ε_views(f_ε_mats(p[1:np_mats]),n_mats)...,grid)
# end


# function _f_smooth!(geom_fn,p_geom0,p_mats_syms=(:ω,),T=SHermitianCompact{3,Float64,6})    
#     np_mats =   length( p_mats_syms);
#     np_geom =   length(p_geom0);
#     mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
#     n_mats = length(mats);
#     f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
#     f_ε_mats(rand(np_mats));
#     function f_smooth!(εg,∂ωεg,∂²ωεg,p,grid)
#         p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
#         smooth!(εg,∂ωεg,∂²ωεg,geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
#         return nothing
#     end
#     return f_smooth!
# end

# function _f_smooth(geom_fn,p_geom0,p_mats_syms=(:ω,),T=SHermitianCompact{3,Float64,6})    
#     np_mats =   length( p_mats_syms);
#     np_geom =   length(p_geom0);
#     mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
#     n_mats = length(mats);
#     f_ε_mats, f_ε_mats! = _f_ε_mats(mats,p_mats_syms);
#     f_ε_mats(rand(np_mats));
#     function f_smooth(p,grid)
#         p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
#         return smooth(geom_fn(p_geom),map(x->T.(x),ε_views(f_ε_mats(p_mats),n_mats))...,grid);
#     end
#     return f_smooth
# end

##### Tests  #####
using StaticArrays, GeometryPrimitives, OptiMode
# using OptiMode: _f_ε_mats
using ForwardDiff
using FiniteDifferences
using ReverseDiff
using Zygote
using Tracker
using Diffractor

rand_p_ω() = [ 0.2*rand(Float64)+0.8, ]
rand_p_ω_T() = vcat(rand_p_ω(),[20.0*rand(Float64)+20.0,])
rand_p_r_n() = [rand(), normalize(rand(3))...]
rand_p_εₑ() = [rand(), normalize(rand(3))..., (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2 ]
rand_p_mats() = [rand_p_ω_T()[1],]
rand_p_geom() = rand_w_t_ts()
rand_p() = vcat( rand_p_mats() , rand_p_geom())

##
Δx,Δy,Δz,Nx,Ny,Nz   =   12.0, 4.0, 1.0, 256, 128, 1;
grid                =   Grid(Δx,Δy,Nx,Ny);
# ridge_wg_slab_loaded(wₜₒₚ::Real,t_core::Real,θ::Real,t_slab::Real,edge_gap::Real,mat_core,mat_slab,mat_subs,Δx::Real,Δy::Real)
function ridge_wg_slab_loaded_sh(wₜₒₚ::Real,t_core::Real,θ::Real,t_slab::Real,edge_gap::Real,mat_core,mat_slab,mat_subs,Δx::Real,Δy::Real) #::Geometry{2}
    t_subs = (Δy -t_core - edge_gap )/2. - t_slab
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
	c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
	# t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	# c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_unetch/2.
	verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    # SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
					verts,
					mat_core,					                                    # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]
	# b_unetch = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
    #                 [0. , c_unetch_y],           	# c: center
    #                 [Δx - edge_gap, t_unetch ],	# r: "radii" (half span of each axis)
    #                 ax,	    		        	# axes: box axes
    #                 mat_core,					 # data: any type, data associated with box shape
    #             )
	b_slab = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
				[0. , c_slab_y],           	# c: center
				[Δx - edge_gap, t_slab ],	# r: "radii" (half span of each axis)
				ax,	    		        	# axes: box axes
				mat_slab,					 # data: any type, data associated with box shape
			)
	b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],           	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    mat_subs,					 # data: any type, data associated with box shape
                )
	return [core,b_slab,b_subs]
end
geom_fn(x)          =   ridge_wg_slab_loaded_sh(x[1],x[2],0.,x[3],0.5,Si₃N₄,MgO_LiNbO₃,SiO₂,Δx,Δy);
rand_w_t_ts() = [ 1.0*rand() + 0.8, 0.2*rand() + 0.1, 0.2*rand() + 0.1 ] ;
geom0 = geom_fn(rand_w_t_ts());
mats = vcat(materials(geom0),Vacuum);
n_mats = length(mats);
f_ε_mats,f_ε_mats! = _f_ε_mats(mats,(:ω,));
##

function ff1(x,p)
    sh1,sh2,sh3 = geom_fn(p)
    r1, n1 = surfpt_nearby(x,sh1)
    r2, n2 = surfpt_nearby(x,sh2)
    r3, n3 = surfpt_nearby(x,sh3)
    return n3[1]*r1[1]^2 + sin(n2[2])^r3[2]
end

ff1(rand(2),rand(4))
Zygote.gradient(ff1,rand(2),rand(4))


xyz, xyzc = x⃗(grid), x⃗c(grid);			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
vxlmin,vxlmax = vxl_minmax(xyzc);
sinds = proc_sinds(corner_sinds(geom0,xyzc));

import OptiMode: matinds
using ChainRulesCore
@non_differentiable OptiMode.materials(shapes::Vector{<:GeometryPrimitives.Shape})
@non_differentiable OptiMode.matinds(shapes::Vector{<:GeometryPrimitives.Shape})
@non_differentiable OptiMode.x⃗(grid::OptiMode.Grid)
@non_differentiable OptiMode.x⃗c(grid::OptiMode.Grid)
@non_differentiable OptiMode.vxl_minmax(xyzc::Array)
@non_differentiable OptiMode.corner_sinds(shapes::Vector{<:GeometryPrimitives.Shape},xyzc::Array)
@non_differentiable OptiMode.proc_sinds(csinds::Array)

function _smooth_single(ε,∂ωε,∂²ωε,shapes,minds,sinds,xyz,vxlmin,vxlmax)
    if iszero(sinds[2]) # if only one shape detected, assign (ε, ∂ωε, ∂²ωε) from that shape's material model
        # εg = ε[minds[first(sinds)]]
        # ∂ωεg = ∂ωε[minds[first(sinds)]]
        # ∂²ωεg = ∂²ωε[minds[first(sinds)]]
        return vcat( vec( ε[minds[first(sinds)]] ),vec( ∂ωε[minds[first(sinds)]] ),vec( ∂²ωε[minds[first(sinds)]] ) )
    elseif iszero(sinds[3]) # if two shapes detected, use Kottke averaging to smooth (ε, ∂ωε, ∂²ωε)
        idx1, idx2 = sinds[1], sinds[2]
        r₀,n⃗ = surfpt_nearby(xyz, shapes[idx1])    # find normal vector `n⃗` of the interface between materials 1 & 2 (pointing 1→2)
        rvol = volfrac((vxlmin,vxlmax),n⃗,r₀)      # find the fill fraction of material 1 in the smoothing voxel
        S = normcart(vec3D(n⃗) / norm(n⃗));
        εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12 = εₑ_∂ωεₑ_∂²ωεₑ(rvol,S,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2])
        # εg[:,:,grididx] = εₑ_12
        # ∂ωεg[:,:,grididx] = ∂ω_εₑ_12
        # ∂²ωεg[:,:,grididx] = ∂ω²_εₑ_12
        return vcat( vec( εₑ_12 ),vec( ∂ω_εₑ_12 ),vec( ∂ω²_εₑ_12 ) )
    else # if more than two shapes detected, use naive averaging to smooth (ε, ∂ωε, ∂²ωε)
        # εg[:,:,grididx] = mapreduce(i->ε[minds[i]],+,sinds) / n_vxl_verts  
        # ∂ωεg[:,:,grididx] = mapreduce(i->∂ωε[minds[i]],+,sinds) / n_vxl_verts  
        # ∂²ωεg[:,:,grididx] = mapreduce(i->∂²ωε[minds[i]],+,sinds) / n_vxl_verts
        return vcat( vec( mapreduce(i->ε[minds[i]],+,sinds) ),vec( mapreduce(i->∂ωε[minds[i]],+,sinds) ),vec( mapreduce(i->∂²ωε[minds[i]],+,sinds) ) ) / length(sinds)
    end
end

function smooth2(shapes,ε,∂ωε,∂²ωε,grid)
    xyz, xyzc = x⃗(grid), x⃗c(grid);			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
    vxlmin,vxlmax = vxl_minmax(xyzc);
    sinds = proc_sinds(corner_sinds(shapes,xyzc));
    minds   =   matinds(shapes);
    return reshape(mapreduce((ss,xx,vxn,vxp)->_smooth_single(ε,∂ωε,∂²ωε,shapes,minds,ss,xx,vxn,vxp),hcat,sinds,xyz,vxlmin,vxlmax),(3,3,3,size(grid)...))
end

mats = vcat(materials(geom0),Vacuum);
n_mats = length(mats);
f_ε_mats,f_ε_mats! = _f_ε_mats(mats,(:ω,));
ε,∂ωε,∂²ωε = ε_views(f_ε_mats([1.0,]),length(mats))

εg_∂ωεg_∂²ωεg = smooth2(geom0,ε,∂ωε,∂²ωε,grid)

εg_∂ωεg_∂²ωεg = smooth2(geom_fn(rand_w_t_ts()),ε_views(f_ε_mats([1.0,]),length(mats))...,grid)

smooth3(p,grid) = smooth2(geom_fn(p[2:4]),ε_views(f_ε_mats([p[1],]),length(mats))...,grid)

εg_∂ωεg_∂²ωεg = smooth3(rand_p(),grid);

Zygote.gradient(x->sum(smooth3(x,grid)),rand_p())

ff3(p) = sum(smooth3(p,Grid(8.0,5.0,256,256)));
ff3(rand_p())
Zygote.gradient(ff3,rand_p());


Zygote.gradient(x->sum(smooth2(geom_fn(x[2:4]),ε_views(f_ε_mats([x[1],]),length(mats))...,grid)),rand_p())

function ff2(p,grid)
    shapes = geom_fn(p)
    # mats = vcat(materials(shapes),Vacuum);
    # f_ε_mats,f_ε_mats! = _f_ε_mats(mats,(:ω,));
    ε,∂ωε,∂²ωε = ε_views(f_ε_mats([first(p),]),4)
    xyz, xyzc = x⃗(grid), x⃗c(grid);			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
    vxlmin,vxlmax = vxl_minmax(xyzc);
    # sinds = proc_sinds(corner_sinds(shapes,xyzc));
    # minds   =   matinds(shapes);
    I = CartesianIndex(50,60)
    idx1, idx2 = 1, 2
    r₀,n⃗ = surfpt_nearby(xyz[I], shapes[idx1])    # find normal vector `n⃗` of the interface between materials 1 & 2 (pointing 1→2)
    rvol = volfrac((vxlmin[I],vxlmax[I]),n⃗,r₀)      # find the fill fraction of material 1 in the smoothing voxel
    S = normcart(vec3D(n⃗) / norm(n⃗));
    εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12 = εₑ_∂ωεₑ_∂²ωεₑ(rvol,S,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2])
    return sum(εₑ_12)
end

ff2(rand_w_t_ts(),grid)
using OptiMode: εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ
sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))))
sum(sum(εₑ_∂ωεₑ_∂²ωεₑ(rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))))
Zygote.gradient((r,S,e1,e2,de1,de2,dde1,dde2)->sum(sum(εₑ_∂ωεₑ_∂²ωεₑ(r,S,e1,e2,de1,de2,dde1,dde2))),rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))
using OptiMode: fjh_εₑᵣ_herm,_f_εₑᵣ_sym, herm_vec, _fj_fjh_sym
f_εₑᵣ_sym, prot = _f_εₑᵣ_sym();
fj_εₑᵣ_sym, fjh_εₑᵣ_sym = _fj_fjh_sym(f_εₑᵣ_sym, prot);
f_εₑᵣ   = eval_fn_oop(f_εₑᵣ_sym,prot);
fj_εₑᵣ  = eval_fn_oop(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ  = eval_fn_oop(fjh_εₑᵣ_sym,prot);
f_εₑᵣ!   = eval_fn_ip(f_εₑᵣ_sym,prot);
fj_εₑᵣ!  = eval_fn_ip(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ! = eval_fn_ip(fjh_εₑᵣ_sym,prot);
fout_rot = f_εₑᵣ(rand(19));
fjout_rot = fj_εₑᵣ(rand(19));
fjhout_rot = fjh_εₑᵣ(rand(19));
f_εₑᵣ!(similar(fout_rot),rand(19));
fj_εₑᵣ!(similar(fjout_rot),rand(19));
fjh_εₑᵣ!(similar(fjout_rot,9,381),rand(19));

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ2(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑᵣ_12 = fjh_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    # fjh_εₑᵣ_12 = similar(ε₁,9,381) # fjh_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    # fjh_εₑᵣ!(fjh_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
    εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
    return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
end
sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ2(rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))))
Zygote.gradient((r,e1,e2,de1,de2,dde1,dde2)->sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ2(r,e1,e2,de1,de2,dde1,dde2))),rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))
val1 = sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ2(rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))))
grad1 = Zygote.gradient((r,e1,e2,de1,de2,dde1,dde2)->sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ2(r,e1,e2,de1,de2,dde1,dde2))),rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm2(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑᵣ_12 = fjh_εₑᵣ_herm(vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    # fjh_εₑᵣ_12 = similar(ε₁,6,183) # fjh_εₑᵣ_herm(fjh_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    # fjh_εₑᵣ_herm!(fjh_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:14], reshape(fjh_εₑᵣ_12[:,15:183],(6,13,13));
    εₑᵣ_12 = SHermitianCompact{3}(f_εₑᵣ_12,)
    v_∂ω, v_∂²ω = vcat(0.0,herm_vec(∂ω_ε₁),herm_vec(∂ω_ε₂)), vcat(0.0,herm_vec(∂²ω_ε₁),herm_vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = SHermitianCompact{3}( j_εₑᵣ_12 * v_∂ω,  );
    ∂ω²_εₑᵣ_12 = SHermitianCompact{3}( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:6] + j_εₑᵣ_12*v_∂²ω  );
    return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
end

sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm2(rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))))
sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm2(rand(),SHermitianCompact{3}(rand(6)),SHermitianCompact{3}(rand(6)),SHermitianCompact{3}(rand(6)),SHermitianCompact{3}(rand(6)),SHermitianCompact{3}(rand(6)),SHermitianCompact{3}(rand(6)))))
Zygote.gradient((r,e1,e2,de1,de2,dde1,dde2)->sum(sum(εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm2(r,e1,e2,de1,de2,dde1,dde2))),rand(),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3),rand(3,3))

##
np_mats = 1;
np_geom = 3;
p = rand_p(); p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)];
shapes = geom_fn(p_geom);

fsm1! = _f_smooth!(geom_fn,f_ε_mats,3,1,4)
fsm1 = _f_smooth(geom_fn,f_ε_mats,3,1,4)
p0 = rand_p()
εg1,∂ωεg1,∂²ωεg1 = fsm1(p0,grid);
εg2,∂ωεg2,∂²ωεg2 = similar(εg1),similar(εg1),similar(εg1);
fsm1!(εg2,∂ωεg2,∂²ωεg2,p0,grid);
@assert εg1 ≈ εg2
@assert ∂ωεg1 ≈ ∂ωεg2
@assert ∂²ωεg1 ≈ ∂²ωεg2

# om0, p_geom0 = 0.2*rand(Float64)+0.8, rand_w_t_ts()
# ff1 = x->fsm1([x,p_geom0...])[1]
# ff2 = x->fsm1([x,p_geom0...])[2]
# ff3 = x->fsm1([x,p_geom0...])[3]
# ∂ff1_∂om =  ForwardDiff.derivative(ff1,om0)
# ∂ff2_∂om =  ForwardDiff.derivative(ff2,om0)
# ∂ff3_∂om =  ForwardDiff.derivative(ff3,om0)
# ∂fsm1_∂p(p) =  ForwardDiff.derivative(fsm1,p)

∂fsm11_∂p_RM(p,grid) = ReverseDiff.gradient(x->sum(fsm1(x,grid)[1]),p)
Diffractor.∂☆(x->sum(fsm1(x,grid)[1]),rand_p())

sumeps1, pb1 = Zygote.pullback(x->sum(fsm1(x,grid)[1]),rand_p())



∂fsm11_∂p_FM(p,grid) =  reshape.(eachcol(ForwardDiff.jacobian(x->vec(fsm1(x,grid)[1]),p)),((3,3,size(grid)...),))
∂fsm12_∂p_FM(p,grid) =  reshape.(eachcol(ForwardDiff.jacobian(x->vec(fsm1(x,grid)[2]),p)),((3,3,size(grid)...),))
∂fsm13_∂p_FM(p,grid) =  reshape.(eachcol(ForwardDiff.jacobian(x->vec(fsm1(x,grid)[3]),p)),((3,3,size(grid)...),))
∂fsm11_∂p_FD(p,grid;n=3) =  reshape.(eachcol(first(FiniteDifferences.jacobian(central_fdm(n,1),x->vec(fsm1(x,grid)[1]),p))),((3,3,size(grid)...),))
∂fsm12_∂p_FD(p,grid;n=3) =  reshape.(eachcol(first(FiniteDifferences.jacobian(central_fdm(n,1),x->vec(fsm1(x,grid)[2]),p))),((3,3,size(grid)...),))
∂fsm13_∂p_FD(p,grid;n=3) =  reshape.(eachcol(first(FiniteDifferences.jacobian(central_fdm(n,1),x->vec(fsm1(x,grid)[3]),p))),((3,3,size(grid)...),))

∂εg_∂ω_FM, ∂εg_∂w_FM, ∂εg_∂t_FM, ∂εg_∂ts_FM = ∂fsm11_∂p_FM(p0,grid);
∂ωεg_∂ω_FM, ∂ωεg_∂w_FM, ∂ωεg_∂t_FM, ∂ωεg_∂ts_FM = ∂fsm12_∂p_FM(p0,grid);
∂∂²ωεg_∂ω_FM, ∂∂²ωεg_∂w_FM, ∂∂²ωεg_∂t_FM, ∂∂²ωεg_∂ts_FM = ∂fsm13_∂p_FM(p0,grid);

∂εg_∂ω_FD, ∂εg_∂w_FD, ∂εg_∂t_FD, ∂εg_∂ts_FD = ∂fsm11_∂p_FD(p0,grid);
∂ωεg_∂ω_FD, ∂ωεg_∂w_FD, ∂ωεg_∂t_FD, ∂ωεg_∂ts_FD = ∂fsm12_∂p_FD(p0,grid);
∂∂²ωεg_∂ω_FD, ∂∂²ωεg_∂w_FD, ∂∂²ωεg_∂t_FD, ∂∂²ωεg_∂ts_FD = ∂fsm13_∂p_FD(p0,grid);

function check_εg_grads(p,grid;n=3)
    εg, ∂ωεg, ∂²ωεg = fsm1(p,grid);
    ∂εg_∂ω_FM, ∂εg_∂w_FM, ∂εg_∂t_FM, ∂εg_∂ts_FM = ∂fsm11_∂p_FM(p,grid);
    ∂∂ωεg_∂ω_FM, ∂∂ωεg_∂w_FM, ∂∂ωεg_∂t_FM, ∂∂ωεg_∂ts_FM = ∂fsm12_∂p_FM(p,grid);
    ∂∂²ωεg_∂ω_FM, ∂∂²ωεg_∂w_FM, ∂∂²ωεg_∂t_FM, ∂∂²ωεg_∂ts_FM = ∂fsm13_∂p_FM(p,grid);
    ∂εg_∂ω_FD, ∂εg_∂w_FD, ∂εg_∂t_FD, ∂εg_∂ts_FD = ∂fsm11_∂p_FD(p,grid;n);
    ∂∂ωεg_∂ω_FD, ∂∂ωεg_∂w_FD, ∂∂ωεg_∂t_FD, ∂∂ωεg_∂ts_FD = ∂fsm12_∂p_FD(p,grid;n);
    ∂∂²ωεg_∂ω_FD, ∂∂²ωεg_∂w_FD, ∂∂²ωεg_∂t_FD, ∂∂²ωεg_∂ts_FD = ∂fsm13_∂p_FD(p,grid;n);

    @assert ∂ωεg ≈ ∂εg_∂ω_FM
    @assert ∂²ωεg ≈ ∂∂ωεg_∂ω_FM

    @assert ∂εg_∂ω_FM ≈ ∂εg_∂ω_FD
    @assert ∂εg_∂w_FM ≈ ∂εg_∂w_FD
    @assert ∂εg_∂t_FM ≈ ∂εg_∂t_FD
    @assert ∂εg_∂ts_FM ≈ ∂εg_∂ts_FD

    @assert ∂∂ωεg_∂ω_FM ≈ ∂∂ωεg_∂ω_FD
    @assert ∂∂ωεg_∂w_FM ≈ ∂∂ωεg_∂w_FD
    @assert ∂∂ωεg_∂t_FM ≈ ∂∂ωεg_∂t_FD
    @assert ∂∂ωεg_∂ts_FM ≈ ∂∂ωεg_∂ts_FD

    @assert ∂∂²ωεg_∂ω_FM ≈ ∂∂²ωεg_∂ω_FD
    @assert ∂∂²ωεg_∂w_FM ≈ ∂∂²ωεg_∂w_FD
    @assert ∂∂²ωεg_∂t_FM ≈ ∂∂²ωεg_∂t_FD
    @assert ∂∂²ωεg_∂ts_FM ≈ ∂∂²ωεg_∂ts_FD
end

check_εg_grads(rand_p(),grid);
check_εg_grads(rand_p(),Grid(8.3,5.5,256,256));
check_εg_grads(rand_p(),Grid(9.3,6.5,211,300));
### Enzyme pullbacks

# function something(x)
#     return a, b, c
# end
 
# function wrap(out, x)
#     out[:] = something(x)
# end

function wrapfsm1(out, x)
    fsm1!(out[1],out[2],out[3],x,grid)
end

εg = zeros(Float64,3,3,size(grid)...)
∂ωεg = zeros(Float64,3,3,size(grid)...)
∂²ωεg = zeros(Float64,3,3,size(grid)...)
εg_tuple = (εg, ∂ωεg, ∂²ωεg)
wrapfsm1(εg_tuple, rand_p())

∂z_∂εg = deepcopy(εg_tuple)
∂z_∂p = similar(rand_p())

Enzyme.autodiff(wrapfsm1, Const, Duplicated(εg_tuple, ∂z_∂εg), Duplicated(rand_p(), ∂z_∂p))

###

function wrapfsm12(x)
    fsm1!(out[1],out[2],out[3],x,grid)
end

# some objective function to work with
f(a, b) = sum(a' * b + a * b')

# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const f_tape = GradientTape(f, (rand(100, 100), rand(100, 100)))

# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
all_results = map(DiffResults.GradientResult, results)
cfg = GradientConfig(inputs)


###

# mats    =   vcat(materials(geom_fn(rand_p_geom())),Vacuum);
# f_ε_mats1, f_ε_mats1! = _f_ε_mats(mats,(:ω,));
# f_ε_mats1(rand_p_mats());
fsm1! = _f_smooth!(geom_fn,f_ε_mats,3,1,4)
fsm1 = _f_smooth(geom_fn,f_ε_mats,3,1,4)

εg1,∂ωεg1,∂²ωεg1 = (Array{Float64}(undef,3,3,size(grid)...) for i=1:3)
# εg1,∂ωεg1,∂²ωεg1 = fsm1(rand_p(),grid);
# εg2,∂ωεg2,∂²ωεg2 = copy.((εg1,∂ωεg1,∂²ωεg1));
εg2,∂ωεg2,∂²ωεg2 = (Array{Float64}(undef,3,3,size(grid)...) for i=1:3)
fsm1!(εg2,∂ωεg2,∂²ωεg2,rand_p(),grid);

function fsm12!(out,p,grid)
    @views fsm1!(out[:,:,:,:,1],out[:,:,:,:,2],out[:,:,:,:,3],p,grid);
    return nothing
end

out = Array{Float64}(undef,3,3,size(grid)...,3)
∂z_∂out = Array{Float64}(undef,3,3,size(grid)...,3)
∂z_∂p = similar(p)
Enzyme.autodiff(fsm12!, Const, Duplicated(out, ∂z_∂out), Duplicated(p, ∂z_∂p), grid)

####

rvol = 0.3
n̂ = rand() # normalize(rand(3))
idx1,idx2 = 1,2
p = rand_p()
p_mats, p_geom = @inbounds p[1:np_mats], p[(np_mats+1):(np_mats+np_geom)]
# ε,∂ωε,∂²ωε = ε_views(f_ε_mats(p_mats),n_mats)
ε,∂ωε,∂²ωε = map(x->SHermitianCompact{3,Float64,6}.(x),ε_views(f_ε_mats(p_mats),n_mats))
εₑ_∂ωεₑ_∂²ωεₑ(rvol,n̂,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2])
##
r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂ = rvol,n̂,ε[idx1],ε[idx2],∂ωε[idx1],∂ωε[idx2],∂²ωε[idx1],∂²ωε[idx2]

np_epse = 14
# fjh_εₑ_12 = fjh_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)));
# f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:(np_epse+1)], reshape(fjh_εₑ_12[:,(np_epse+2):end],(9,np_epse,np_epse));
fj_εₑ_12 = fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)));
f_εₑ_12, j_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:(np_epse+1)]
εₑ_12 = @views reshape(f_εₑ_12,(3,3))
v_∂ω = vcat(zeros(2),getproperty(∂ω_ε₁,:lowertriangle),getproperty(∂ω_ε₂,:lowertriangle))
# v_∂²ω = vcat(zeros(4),getproperty(∂²ω_ε₁,:lowertriangle),getproperty(∂²ω_ε₂,:lowertriangle));
∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
# ∂ω²_εₑ_12 = @views reshape( [dot(v_∂ω,h_εₑ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑ_12*v_∂²ω , (3,3) );





####

fsm1_eps(pp) = vec(fsm1(pp,grid)[1]);
fsm1_deps(pp) = vec(fsm1(pp,grid)[2]);
fsm1_ddeps(pp) = vec(fsm1(pp,grid)[3]);
∂fsm1_eps_FD(pp) = FiniteDifferences.jacobian(central_fdm(3,1),fsm1_eps,pp) |> first
∂fsm1_deps_FD(pp) = FiniteDifferences.jacobian(central_fdm(3,1),fsm1_deps,pp) |> first
∂fsm1_ddeps_FD(pp) = FiniteDifferences.jacobian(central_fdm(3,1),fsm1_ddeps,pp) |> first
∂fsm1_eps_FM(pp) = ForwardDiff.jacobian(fsm1_eps,pp)
∂fsm1_deps_FM(pp) = ForwardDiff.jacobian(fsm1_deps,pp)
∂fsm1_ddeps_FM(pp) = ForwardDiff.jacobian(fsm1_ddeps,pp)
pp0 = rand_p();
fsm1_eps0 = fsm1_eps(pp0); 
fsm1_deps0 = fsm1_deps(pp0); 
fsm1_ddeps0 = fsm1_ddeps(pp0); 
∂fsm1_eps_FD0 = ∂fsm1_eps_FD(pp0); 
∂fsm1_deps_FD0 = ∂fsm1_deps_FD(pp0); 
∂fsm1_ddeps_FD0 = ∂fsm1_ddeps_FD(pp0); 
∂fsm1_eps_FM0 = ∂fsm1_eps_FM(pp0); 
∂fsm1_deps_FM0 = ∂fsm1_deps_FM(pp0); 
∂fsm1_ddeps_FM0 = ∂fsm1_ddeps_FM(pp0); 

##
using StructArrays, StaticArrays, OptiMode
Δx,Δy,Δz,Nx,Ny,Nz   =   12.0, 4.0, 1.0, 256, 128, 1;
grid                =   Grid(Δx,Δy,Nx,Ny);
eps = StructArray{SHermitianCompact{3,Float64,6}}(undef,size(grid)...);
# eps = StructArray{SHermitianCompact{3,Complex{Float64},6}}(undef,size(grid)...;unwrap=T->!(T<:Real));
function ff1(grid)
    eps = StructArray{SHermitianCompact{3,Float64,6}}(undef,size(grid)...);
    for I in eachindex(grid)
        eps[I] = SHermitianCompact{3,Float64,6}([cos(1.0*I[1]), sqrt(1.0*I[2]), 1.0*I[1]*I[2]^3,log(1.0*I[1]), tan(1.0*I[2]), 1.0*I[1]^4*I[2]^2])
    end
    return eps
end