using LinearAlgebra, StaticArrays, GeometryPrimitives, OptiMode, ChainRules, FiniteDifferences, ForwardDiff, Zygote, BenchmarkTools, Test

function smooth_ε_single(sinds::NTuple{NI, TI},shapes,minds,mat_vals,xx,vxl_min,vxl_max) where {NI,TI<:Int}
	@inbounds if iszero(sinds[2])
		return mat_vals[:,minds[first(sinds)]]
	elseif iszero(sinds[3])
		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
		return εₑ_∂ωεₑ_∂²ωεₑ(
			rvol,
			normcart(vec3D(n⃗)),
			mat_vals[:,minds[sinds[1]]],
			mat_vals[:,minds[sinds[2]]],
		)
	else
		return @inbounds sum(i->mat_vals[:,minds[i]],sinds) / NI  # naive averaging to be used
	end
end

function smooth_ε(f_geom::F,p_geom::AbstractVector{T1},mat_vals::AbstractMatrix{T2},grid::Grid{ND}) where {F<:Function,T1<:Real,T2<:Real,ND}
	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
	vxlmin,vxlmax = vxl_minmax(xyzc)
	arr_flatB = Zygote.Buffer(p_geom,9*prod(size(grid)),3)
	arr_flat = Zygote.forwarddiff(p) do p
		geom = f_geom(p[(np_mats+1):(np_geom+np_mats)])
		shapes = getfield(geom,:shapes)
		# material_inds = getfield(geom,:material_inds)
		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
		smoothed_vals = mapreduce(vcat,sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
			smooth_ε_single(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp)
		end
		# smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
		# smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
		return smoothed_vals #_rr  # new spatially smoothed ε tensor array
	end
	copyto!(arr_flatB,copy(arr_flat))
	arr_flat_r = copy(arr_flatB)
	# Nx = size(grid,1)
	# Ny = size(grid,2)
	# fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
	# return fn_arrs
	return arr_flat_r
end

function smooth_ε(p_mats::AbstractVector{T1},p_geom::AbstractVector{T2},f_mats::F1,f_geom::F2,grid::Grid{ND}) where {ND,N,T1<:Real,T2<:Real,F1<:Function,F2<:Function}
	np_geom = length(p_geom)
	np_mats = length(p_mats)
	p = vcat(p_mats,p_geom)
	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
	vxlmin,vxlmax = vxl_minmax(xyzc)
	mat_vals = f_mats(p_mats)
	arr_flatB = Zygote.Buffer(p_geom,9*prod(size(grid)),3)
	arr_flat = Zygote.forwarddiff(p_geom) do p_geom
		geom = f_geom(p[(np_mats+1):(np_geom+np_mats)])
		shapes = getfield(geom,:shapes)
		# material_inds = getfield(geom,:material_inds)
		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
		smoothed_vals = mapreduce(vcat,sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
			smooth_ε_single(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp)
		end
		# smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
		# smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
		return smoothed_vals #_rr  # new spatially smoothed ε tensor array
	end
	copyto!(arr_flatB,copy(arr_flat))
	arr_flat_r = copy(arr_flatB)
	# Nx = size(grid,1)
	# Ny = size(grid,2)
	# fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
	# return fn_arrs
	return arr_flat_r
end
 

"""
Testing
"""
function foo_shapes(p::Vector{T}) where {T}
    wₜₒₚ        =   p[1]*1.0 + 0.8
    t_core      =   p[2]*0.2 + 0.5
    θ           =   p[3]*0.1
    t_slab      =   p[4]
    edge_gap::T    =   0.5
    Δx::T          =   8.0 #p[5]*4.0 + 7.0
    Δy::T          =   5.0 #p[6]*3.0 + 3.0

    t_subs = (Δy -t_core - edge_gap )/2. - t_slab
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
	c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
    wt_half = wₜₒₚ / 2.0
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2.0
	verts = SMatrix{4,2,T}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(	        # Instantiate 2D polygon, here a trapazoid
					verts,                      # v: polygon vertices in counter-clockwise order
					Si₃N₄,			        # data: any type, data associated with box shape
                )
    ax = SMatrix{2,2,T}(  [       1.    0.
                                0.    1.      ]     )
	b_slab = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
				SVector{2}(zero(c_slab_y), c_slab_y),           	    # c: center
				SVector{2}((Δx - edge_gap), t_slab ),	    # r: "radii" (half span of each axis)
				ax,	    		        	    # axes: box axes
				MgO_LiNbO₃,					    # data: any type, data associated with box shape
			)
	b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    SVector{2}(zero(c_subs_y) , c_subs_y),           	# c: center
                    SVector{2}((Δx - edge_gap), t_subs ),	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    SiO₂,					# data: any type, data associated with box shape
                )
	shapes = (core,b_slab,b_subs)
end

using OptiMode: vec3D

foo_geometry(p) = Geometry(foo_shapes(p))

p               =   rand(4);
sh1             =   foo_shapes(p);
# geom1           =   foo_geometry(p)
f_geom2(p) = (;shapes=foo_shapes(p),material_inds=matinds([shapes...,]))
f_geom3(p) = (shapes=foo_shapes(p); (;shapes=foo_shapes(p),material_inds=matinds([shapes...,])))
f_ε_mats, f_ε_mats! = _f_ε_mats(vcat(materials(sh1),Vacuum),(:ω,))
grid  = Grid(6.0,4.0,128,128)

f_geom = foo_shapes;
p_geom = p;
f_mats = f_ε_mats;
p_mats = [1.0,];
# smooth_ε(p_mats,p_geom,f_mats,f_geom3,grid)


function smoov1(shapes,mat_vals::AbstractMatrix{T2},minds,grid::Grid{ND}) where {T2<:Real,ND}
	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
	vxlmin,vxlmax = vxl_minmax(xyzc)
	sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
	smoothed_vals = mapreduce(vcat,sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
		smooth_ε_single(sinds,shapes,minds,mat_vals,xx,vn,vp)
	end
	return smoothed_vals
end

function smoov2(shapes,mat_vals::AbstractMatrix{T2},minds,grid::Grid{ND}) where {T2<:Real,ND}
	xyz = x⃗(grid)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel center
	xyzc = x⃗c(grid)
	vxlmin,vxlmax = vxl_minmax(xyzc)
	sinds = proc_sinds(corner_sinds(shapes,xyzc))
	smoothed_vals = mapreduce(vcat,sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
		smooth_ε_single(sinds,shapes,minds,mat_vals,xx,vn,vp)
	end
	return smoothed_vals
end

function smoov3(shapes,mat_vals::AbstractMatrix{T2},minds,grid::Grid{ND}) where {T2<:Real,ND}
	xyz = x⃗(grid)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel center
	xyzc = x⃗c(grid)
	vxlmin,vxlmax = vxl_minmax(xyzc)
	sinds = proc_sinds(corner_sinds(shapes,xyzc))
	smoothed_vals = mapreduce(vcat,eachindex(grid)) do I
		smooth_ε_single(sinds[I],shapes,minds,mat_vals,xyz[I],vxlmin[I],vxlmax[I])
	end
	return smoothed_vals
end

function smoov4(shapes,mat_vals::AbstractMatrix{T2},minds,grid::Grid{ND}) where {T2<:Real,ND}
	xyz = x⃗(grid)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel center
	xyzc = x⃗c(grid)
	vxlmin,vxlmax = vxl_minmax(xyzc)
	sinds = proc_sinds(corner_sinds(shapes,xyzc))
	smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
		smooth_ε_single(sinds,shapes,minds,mat_vals,xx,vn,vp)
	end
	return smoothed_vals
end

function smoov5(shapes,mat_vals::AbstractMatrix{T2},minds,grid::Grid{ND}) where {T2<:Real,ND}
	# xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	# xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
	# vxlmin,vxlmax = vxl_minmax(xyzc)
	
	sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
	smoothed_vals = mapreduce(vcat,sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
		smooth_ε_single(sinds,shapes,minds,mat_vals,xx,vn,vp)
	end
	return smoothed_vals
end


shapes = f_geom(p_geom);
matvals = f_mats(p_mats);
minds = collect(1:4);
epsm1 = smoov1(shapes,matvals,minds,grid);
epsm12 = smoov1(f_geom(p_geom),f_mats(p_mats),minds,grid);
gr_epsm1 = Zygote.gradient((pg,pm)->sum(smoov1(f_geom(pg),f_mats(pm),minds,grid)),p_geom,p_mats);

epsm2 = smoov2(shapes,matvals,minds,grid);
epsm22 = smoov2(f_geom(p_geom),f_mats(p_mats),minds,grid);
gr_epsm2 = Zygote.gradient((pg,pm)->sum(smoov2(f_geom(pg),f_mats(pm),minds,grid)),p_geom,p_mats);

gr_epsm2f = Zygote.gradient((pg,pm)->Zygote.forwarddiff(pp->sum(smoov2(f_geom(pp[2:end]),f_mats([pp[1],]),minds,grid)),vcat(pg,pm)),p_geom,p_mats);


epsm3 = smoov3(shapes,matvals,minds,grid);
epsm32 = smoov3(f_geom(p_geom),f_mats(p_mats),minds,grid);
gr_epsm3 = Zygote.gradient((pg,pm)->sum(smoov3(f_geom(pg),f_mats(pm),minds,grid)),p_geom,p_mats);


@benchmark          foo4($p4)
@btime          foo4($p4)
@show   gFD4    =   FiniteDifferences.grad(central_fdm(5,1,max_range=9e-4),foo4,p4)[1]
@show   gFM4    =   ForwardDiff.gradient(foo4,p4)
@show   gerrFM4  =   gFD4 .- gFM4
@test   gFD4 ≈ gFM4 rtol=1e-4

###########################################################################################################
###########################################################################################################
###
###									Minimal Examples of Similar Algorithms
###										& Their Gradients
###
###########################################################################################################
###########################################################################################################

## 1: As of yet unnamed example
using LinearAlgebra

x = LinRange(-5.0,5.0,100)
obj_fn(p) = (3.0*sin(p[1])^p[2] + p[3]*5.0 - 5.0)
n_obj = 10
p = rand(3,n_obj)


function foo1(p,x)
	objs = obj_fn.(eachcol(p))
	map(x) do xx
		objs[argmin((abs2(xx-obj) for obj in objs))]
	end
end

foo1(p,x)

using ChainRules, FiniteDifferences, ForwardDiff, Zygote, BenchmarkTools

##
function foo1_test(n_x,n_obj)
	println("##################")
	println("foo1 Test: ")
	println("\tNumber of grid points: $n_x")
	println("\tNumber of objects: $n_obj")
	println("\tNumber of parameters: $(3*n_obj)")
	
	# x = LinRange(-5.0,5.0,n_x)
	x = range(-5.0,5.0,length=n_x)
	p = rand(3,n_obj)
	gr1_FD = FiniteDifferences.grad(central_fdm(3,1),p->sum(foo1(p,x)),p)[1]
	gr1_FM = ForwardDiff.gradient(p->sum(foo1(p,x)),p)
	gr1_RM = Zygote.gradient(p->sum(foo1(p,x)),p)[1]
	println("foo1 FM grad max error magnitude")
	@show gr1_FM_max_err = maximum(gr1_FD .- gr1_FM)
	println("foo1 RM grad max error magnitude")
	@show gr1_RM_max_err = maximum(gr1_FD .- gr1_RM)
	println("foo1 FD grad benchmark")
	@btime FiniteDifferences.grad(central_fdm(3,1),p->sum(foo1(p,x)),p)[1]
	println("foo1 FM grad benchmark")
	@btime ForwardDiff.gradient(p->sum(foo1(p,x)),p)
	println("foo1 RM grad benchmark")
	@btime Zygote.gradient(p->sum(foo1(p,x)),p)[1]
	println("##################")
end

foo1_test(10,10)
foo1_test(10,100)
foo1_test(10,1000)

foo1_test(100,10)
foo1_test(100,100)
foo1_test(100,1000)

###########################################################################################################
###########################################################################################################
###
###									A "More Local" Smoothing Function
###										& Their Gradients
###
###########################################################################################################
###########################################################################################################

gr1 = Grid(6.0,4.0,256,128)
xv1 = ( SVector{3,T}(xx,yy,zero(T)) for xx in OptiMode.x(gr1), yy in OptiMode.y(gr1) )

yidx = 55
OptiMode.y(gr1)[yidx] ≈	gr1.Δy*((yidx-1)*inv(gr1.Ny)-0.5)

xidx = 155
OptiMode.x(gr1)[xidx] ≈	gr1.Δx*((xidx-1)*inv(gr1.Nx)-0.5)