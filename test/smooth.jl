using LinearAlgebra, StaticArrays, GeometryPrimitives, OptiMode, Test
using ChainRules, Zygote, FiniteDifferences, ForwardDiff
using OptiMode: vec3D
# using CairoMakie

function test_AD(f::Function,p;nFD=5)
    primal  =   f(p)
    gr_RM   =   first(Zygote.gradient(f,p))
    gr_FM   =   ForwardDiff.gradient(f,p)
    gr_FD   =   first(FiniteDifferences.grad(central_fdm(nFD,1),f,p))
    return isapprox(gr_RM,gr_FD,rtol=1e-4) && isapprox(gr_RM,gr_FD,rtol=1e-4)
end

function demo_shapes2D(p::Vector{T}=rand(17)) where T<:Real
    ε₁  =   diagm([p[1],p[1],p[1]])
    ε₂  =   diagm([p[2],p[3],p[3]])
    ε₃  =   diagm([1.0,1.0,1.0])

    b = Box(					# Instantiate N-D box, here N=2 (rectangle)
        p[4:5],					# c: center
        p[6:7],				# r: "radii" (half span of each axis)
        mapreduce(
            normalize,
            hcat,
            eachcol(reshape(p[8:11],(2,2))),
        ),			# axes: box axes
        ε₁,						# data: any type, data associated with box shape
        )

    s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
        p[12:13],					# c: center
        p[14],						# r: "radii" (half span of each axis)
        ε₂,						# data: any type, data associated with circle shape
        )

    t = regpoly(				# triangle::Polygon using regpoly factory method
        3,						# k: number of vertices
        p[15],					# r: distance from center to vertices
        π/2,					# θ: angle of first vertex
        p[16:17],					# c: center
        ε₃,						# data: any type, data associated with triangle
        )

    # return Geometry([ t, s, b ])
	return ( t, s, b )
end

function geom1(p)  # slab_loaded_ridge_wg
    wₜₒₚ        =   p[1]
    t_core      =   p[2]
    θ           =   p[3]
    t_slab      =   p[4]
    edge_gap    =   0.5
    mat_core    =   1
    mat_slab    =   2
    mat_subs    =   3
    Δx          =   6.0
    Δy          =   4.0
    t_subs = (Δy -t_core - edge_gap )/2. - t_slab
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
	c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
	# t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	# c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_unetch/2.
	verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(verts,mat_core)
    ax = SMatrix{2,2}( [      1.     0.   ;   0.     1.      ] )
	# b_unetch = GeometryPrimitives.Box( [0. , c_unetch_y], [Δx - edge_gap, t_unetch ],	ax,	mat_core )
	b_slab = GeometryPrimitives.Box( SVector{2}([0. , c_slab_y]), SVector{2}([Δx - edge_gap, t_slab ]),	ax, mat_slab, )
	b_subs = GeometryPrimitives.Box( SVector{2}([0. , c_subs_y]), SVector{2}([Δx - edge_gap, t_subs ]),	ax,	mat_subs, )
	return (core,b_slab,b_subs)
end

function geom2(p)  # slab_loaded_ridge_wg
    wₜₒₚ        =   p[1]
    t_core      =   p[2]
    θ           =   p[3]
    t_slab      =   p[4]
    edge_gap    =   0.5
    mat_core    =   1
    mat_slab    =   2
    mat_subs    =   3
    Δx          =   6.0
    Δy          =   4.0
    t_subs = (Δy -t_core - edge_gap )/2. - t_slab
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
	c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
	# t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	# c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_unetch/2.
	verts           =   SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core            =   GeometryPrimitives.Polygon(verts,mat_core)
    ax              =   SMatrix{2,2}( [      1.     0.   ;   0.     1.      ] )
    
    x_box_min       =   wb_half
    x_box_max       =   (Δx - edge_gap)/2
    y_box_min       =   c_slab_y + t_slab/2
    y_box_max       =   (Δy - edge_gap)/2
    # b_unetch = GeometryPrimitives.Box( [0. , c_unetch_y], [Δx - edge_gap, t_unetch ],	ax,	mat_core )
	b_slab = GeometryPrimitives.Box( SVector{2}([0. , c_slab_y]), SVector{2}([Δx - edge_gap, t_slab ]),	ax, mat_slab, )
	b_subs = GeometryPrimitives.Box( SVector{2}([0. , c_subs_y]), SVector{2}([Δx - edge_gap, t_subs ]),	ax,	mat_subs, )

    boxes = mapreduce(vcat,eachcol(reshape(p[5:end],(4,10)))) do pp
        c_bb_x      =   pp[1]*(x_box_max-x_box_min) + x_box_min
        r_bb_x      =   pp[2]*min((c_bb_x-x_box_min),(x_box_max-c_bb_x))
        c_bb_y      =   pp[3]*(y_box_max-y_box_min) + y_box_min
        r_bb_y      =   pp[4]*min((c_bb_y-y_box_min),(y_box_max-c_bb_y))
        [ GeometryPrimitives.Box( SVector{2}( c_bb_x , c_bb_y), SVector{2}(2*r_bb_x, 2*r_bb_y ), ax, mat_boxes, ),
          GeometryPrimitives.Box( SVector{2}(-c_bb_x , c_bb_y), SVector{2}(2*r_bb_x, 2*r_bb_y ), ax, mat_boxes, ), ]

    end
	return (boxes...,core,b_slab,b_subs)
end

# function ridge_wg_partial_etch3D(wₜₒₚ::Real,t_core::Real,etch_frac::Real,θ::Real,edge_gap::Real,mat_core,mat_subs,Δx::Real,Δy::Real,Δz::Real) #::Geometry{2}
function geom3(p)
    wₜₒₚ        =   p[1]
    t_core      =   p[2]
    etch_frac   =   p[3]
    θ           =   p[4]
    edge_gap    =   0.5
    mat_core    =   1
    mat_subs    =   2
    Δx          =   4.0
    Δy          =   3.0
    Δz          =   2.0
	t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
	t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_unetch/2.
	verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Prism(SVector(0.,0.,0.), Polygon(verts,nothing), Δz, SVector{3}([0.,0.,1.]), mat_core)
    ax = SMatrix{3,3}([      1.     0.		0.  ;   0.     1.      	0.  ;   0.		0.		1.		])
	b_unetch = GeometryPrimitives.Box(SVector{3}([0.,c_unetch_y,0.]), SVector{3}([Δx - edge_gap, t_unetch, Δz ]), ax, mat_core)
	b_subs = GeometryPrimitives.Box(SVector{3}([0. , c_subs_y, 0.]), SVector{3}([Δx - edge_gap, t_subs, Δz ]), ax, mat_subs)
    # return (core,b_unetch,b_subs)
	## etched pattern in slab cladding ##
	dx_sw_cyl = 0.5	# distance of etched cylinder centers from sidewalls
	r_cyl = 0.1
	z_cyl = 0.0
	c_cyl1 = SVector{2}(wb_half+dx_sw_cyl, c_unetch_y, z_cyl)
	c_cyl2 = SVector{2}(-wb_half-dx_sw_cyl, c_unetch_y, z_cyl)
	cyl1 = GeometryPrimitives.Cylinder(c_cyl1,r_cyl,t_unetch,SVector(0.,1.,0.),Vacuum)
	cyl2 = GeometryPrimitives.Cylinder(c_cyl2,r_cyl,t_unetch,SVector(0.,1.,0.),Vacuum)
	return (cyl1,cyl2,core,b_unetch,b_subs)
end

mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
mat_vars = (:ω,)
np_mats = length(mat_vars)
f_ε_mats, f_ε_mats! = _f_ε_mats(mats,mat_vars) # # f_ε_mats, f_ε_mats! = _f_ε_mats(vcat(materials(sh1),Vacuum),(:ω,))
ω               =   1.0
p               =   [ω, 2.0,0.8,0.1,0.1] #rand(4+np_mats);
mat_vals        =   f_ε_mats(p[1:np_mats]);
grid            =   Grid(6.,4.,256,128)
shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
minds           =   (1,2,3,4)
sm1             =   smooth_ε(shapes,mat_vals,minds,grid);
ε               =   copy(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(sm1,3,2)); # sm1[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);
kmags,evecs = find_k(ω,ε,grid);
ng1 = group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
ng_grads = Zygote.gradient((k,ev,om,epsi,deps_dom)->group_index(k,ev,om,epsi,deps_dom,grid),kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω);
E1 = E⃗(kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,grid)
kg1 = k_guess(ω,ε⁻¹)
k1 = kmags[1]
# kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
ms = ModeSolver(kmags[1], ε⁻¹, grid; nev=2, maxiter=100, tol=1e-8)
res1 = solve_ω²(ms)

##
# kz1,ev1 = solve_k(ms,ω,ε⁻¹;nev=2,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
ωₜ = 1.0
eigind = 1
ω²,ev1 = solve_ω²(ms,kmags[1];nev=2,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
Δω² = ω²[eigind] - ωₜ^2
∂ω²∂k = 2 * HMₖH(ev1[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
2 * HMₖH(vec(evecs[1]),ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
2 * HMₖH((ev1[:,eigind])./sqrt(sum(abs2,ev1[:,eigind])),ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) 
# 2 * HMₖH(ev1[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn[:,1,:,:],ms.M̂.mn[:,2,:,:])
mag3,mn3 = mag_mn(kmags[1],grid)
2 * HMₖH(vec(evecs[1]),ε⁻¹,mag3,mn3)


using CairoMakie
let fig = Figure(resolution = (600, 1200)), X=x(Grid(6.,4.,256,128)),Y=y(Grid(6.,4.,256,128)),Z=(ε[1,1,:,:], ∂ε_∂ω[1,1,:,:], ∂²ε_∂ω²[1,1,:,:]), labels=("ε","∂ε_∂ω","∂²ε_∂ω²"), cmaps=(:viridis,:magma,:RdBu)
    axes = [ Axis(fig[i, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect()) for i=1:length(Z) ]
    for (i,ZZ) in enumerate(Z)
        # ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
        hmap = heatmap!(axes[i],X,Y,ZZ; colormap = cmaps[i])
        Colorbar(fig[i, 2], hmap; label = labels[i],) # width = 15, height=200, ticksize = 15, tickalign = 1)
        scatter!(axes[i],Point2f(X[valinds[4]],Y[valinds[5]]))
    end
    # colsize!(fig.layout, 1, Aspect(1, 1.0))
    # colgap!(fig.layout, 1)
    display(fig)
end;
function plot_field!(pos,F,grid;cmap=:diverging_bkr_55_10_c35_n256,label_base=["x","y"],label="E",xlim=nothing,ylim=nothing,axind=1)
	xs = x(grid)
	ys = y(grid)
	xlim = isnothing(xlim) ? Tuple(extrema(xs)) : xlim
	ylim = isnothing(ylim) ? Tuple(extrema(ys)) : ylim

	# ax = [Axis(pos[1,j]) for j=1:2]
	# ax = [Axis(pos[j]) for j=1:2]
	labels = label.*label_base
	Fs = [view(F,j,:,:) for j=1:2]
	magmax = maximum(abs,F)
	hm = heatmap!(pos, xs, ys, real(Fs[axind]),colormap=cmap,label=labels[1],colorrange=(-magmax,magmax))
	# ax1 = pos[1]
	# hms = [heatmap!(pos[j], xs, ys, real(Fs[j]),colormap=cmap,label=labels[j],colorrange=(-magmax,magmax)) for j=1:2]
	# hm1 = heatmap!(ax1, xs, ys, real(Fs[1]),colormap=cmap,label=labels[1],colorrange=(-magmax,magmax))
	# ax2 = pos[2]
	# hm2 = heatmap!(ax2, xs, ys, real(Fs[2]),colormap=cmap,label=labels[2],colorrange=(-magmax,magmax))
	# hms = [hm1,hm2]
	# cbar = Colorbar(pos[1,3], heatmaps[2],  width=20 )
	# wfs_E = [wireframe!(ax_E[j], xs, ys, Es[j], colormap=cmap_E,linewidth=0.02,color=:white) for j=1:2]
	# map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:white), ax, labels )
	# hideydecorations!.(ax[2])
	# [ax[1].ylabel= "y [μm]" for axx in ax[1:1]]
	# for axx in ax
	# 	axx.xlabel= "x [μm]"
	# 	xlims!(axx,xlim)
	# 	ylims!(axx,ylim)
	# 	axx.aspect=DataAspect()
	# end
	# linkaxes!(ax...)
	return hm
end
function plot_field(F,grid;cmap=:diverging_bkr_55_10_c35_n256,label_base=["x","y"],label="E",xlim=nothing,ylim=nothing,axind=1)
	fig=Figure()
	ax = fig[1,1] = Axis(fig,aspect=DataAspect())
	hms = plot_field!(ax,F,grid;cmap,label_base,label,xlim,ylim,axind)
    Colorbar(fig[1,2],hms;label=label*" axis $axind")
	fig
end

plot_field(E1,grid;axind=1)

let fig = Figure(resolution = (600, 1200)), X=x(Grid(6.,4.,256,128)),Y=y(Grid(6.,4.,256,128)),Z=(realE1[1,1,:,:], ∂ε_∂ω[1,1,:,:], ∂²ε_∂ω²[1,1,:,:]), labels=("ε","∂ε_∂ω","∂²ε_∂ω²"), cmaps=(:viridis,:magma,:RdBu)
    axes = [ Axis(fig[i, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect()) for i=1:length(Z) ]
    for (i,ZZ) in enumerate(Z)
        # ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
        hmap = heatmap!(axes[i],X,Y,ZZ; colormap = cmaps[i])
        Colorbar(fig[i, 2], hmap; label = labels[i],) # width = 15, height=200, ticksize = 15, tickalign = 1)
        scatter!(axes[i],Point2f(X[valinds[4]],Y[valinds[5]]))
    end
    # colsize!(fig.layout, 1, Aspect(1, 1.0))
    # colgap!(fig.layout, 1)
    display(fig)
end;


##

let fig = Figure(resolution = (600, 1200)), X=x(Grid(6.,4.,256,128)),Y=y(Grid(6.,4.,256,128)),Z=(ε[1,1,:,:], ∂ε_∂ω[1,1,:,:], ∂²ε_∂ω²[1,1,:,:]), labels=("ε","∂ε_∂ω","∂²ε_∂ω²"), cmaps=(:viridis,:magma,:RdBu)
    axes = [ Axis(fig[i, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect()) for i=1:length(Z) ]
    for (i,ZZ) in enumerate(Z)
        # ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
        hmap = heatmap!(axes[i],X,Y,ZZ; colormap = cmaps[i])
        Colorbar(fig[i, 2], hmap; label = labels[i],) # width = 15, height=200, ticksize = 15, tickalign = 1)
        scatter!(axes[i],Point2f(X[valinds[4]],Y[valinds[5]]))
    end
    # colsize!(fig.layout, 1, Aspect(1, 1.0))
    # colgap!(fig.layout, 1)
    display(fig)
end;

yidx = 64 
xidx = findfirst(x->!iszero(x),sm1[1,2,1,:,yidx]) # index of first nonzero off-diagonal epsilon value along grid row at yidx
crnrs = corners(grid)[xidx,yidx]
ps = proc_sinds(corner_sinds(shapes,crnrs))
valinds = 1,1,1,xidx,yidx # last two are x,y indices of grid point on shape boundary
#∂²ε_∂ω²[:,:,xidx,yidx]


# AD through material dielectric fn
ff1(p) = sum(f_ε_mats(p[1:1]))
ff1([1.0,])
Zygote.gradient(ff1,[1.0,])
# AD through geometry fn, 1st shape property
ff2(p) = getindex( getproperty( first( geom1(p[2:5]) ) , :v) ,3,1)
ff2(p)
Zygote.gradient(ff2,p)
# AD through geometry fn, 2nd shape property
ff3(p) = getindex( getproperty( getindex( geom1(p[2:5]), 2 ) , :r) ,2)
ff3(p)
Zygote.gradient(ff3,p)
## AD through smoothing of dielectric fn at single grid point
function ff4(p)
    shapes = geom1(p[2:5])
    mat_vals = f_ε_mats([p[1],])
    minds = (1,2,3,4)
    grid = Grid(6.,4.,256,128)
    xidx = 84
    yidx = 64 
    crnrs = corners(grid)[xidx,yidx]
    sum(smoov1_single(shapes,mat_vals,minds,crnrs))
end
ff4(p)
Zygote.gradient(ff4,p)
Zygote.gradient(ff4,[1.1, 2.0,0.8,0.1,0.1])
FiniteDifferences.grad(central_fdm(5,1),ff4,p)
## AD through smoothing of dielectric fn with one material parameter (frequency)
ftest_geom1(p)  =   sum(smoov11(geom1(p[2:5]),f_ε_mats([p[1,],]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom1(p)
gr_ftest_geom1_RM = Zygote.gradient(ftest_geom1,p)
gr_ftest_geom1_FD = FiniteDifferences.grad(central_fdm(5,1),ftest_geom1,p)
gr_ftest_geom1_RM2 = Zygote.gradient(ftest_geom1,[1.1, 1.8,0.3,0.02,0.3])
gr_ftest_geom1_FD2 = FiniteDifferences.grad(central_fdm(5,1),ftest_geom1,[1.1, 1.8,0.3,0.02,0.3])
## AD through smoothing of dielectric fn with two material parameters (frequency, temperature)
mat_vars = (:ω,:T)
np_mats = length(mat_vars)
f_ε_mats2, f_ε_mats2! = _f_ε_mats(mats,mat_vars)
p               =   [1.0, 30.0, 2.0,0.8,0.1,0.1] #rand(4+np_mats);
mat_vals        =   f_ε_mats2(p[1:np_mats]);
grid            =   Grid(6.,4.,256,128)
shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
minds           =   (1,2,3,4)
sm1             =   smoov11(shapes,mat_vals,minds,grid);
ftest_geom2(p)  =   sum(smoov11(geom1(p[3:6]),f_ε_mats2(p[1:2]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom2(p)
gr_ftest_geom2_RM = Zygote.gradient(ftest_geom2,p)
gr_ftest_geom2_FD = FiniteDifferences.grad(central_fdm(5,1),ftest_geom2,p)
gr_ftest_geom2_RM2 = Zygote.gradient(ftest_geom2,[1.1,28.5, 1.8,0.3,0.02,0.3])
gr_ftest_geom2_FD2 = FiniteDifferences.grad(central_fdm(5,1),ftest_geom2,[1.1,28.5,1.8,0.3,0.02,0.3])
## AD through smoothing of dielectric fn with three material parameters (frequency, temperature, LN rotation angle)
using Symbolics
using Rotations: RotX, RotY, RotZ, MRP
@variables θ
rot1 = Matrix(RotY(θ))
LNrot = rotate(MgO_LiNbO₃,Matrix(RotY(θ)),name=:LiNbO₃_X);
mats = [LNrot,Si₃N₄,SiO₂,Vacuum];
mat_vars = (:ω,:T,:θ)
np_mats = length(mat_vars)
f_ε_mats3, f_ε_mats3! = _f_ε_mats(mats,mat_vars)
p               =   [1.0, 30.0, 0.4, 2.0,0.8,0.1,0.1] #rand(4+np_mats);
mat_vals        =   f_ε_mats3(p[1:np_mats]);
grid            =   Grid(6.,4.,256,128)
shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
minds           =   (1,2,3,4)
sm1             =   smoov11(shapes,mat_vals,minds,grid);
ftest_geom3(p)  =   sum(smoov11(geom1(p[4:7]),f_ε_mats3(p[1:3]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom3(p)
gr_ftest_geom3_RM = Zygote.gradient(ftest_geom3,p)
gr_ftest_geom3_FD = FiniteDifferences.grad(central_fdm(5,1),ftest_geom3,p)
gr_ftest_geom3_RM2 = Zygote.gradient(ftest_geom3,[1.1,28.5, 1.1, 1.8,0.3,0.02,0.3])
gr_ftest_geom3_FD2 = FiniteDifferences.grad(central_fdm(5,1),ftest_geom3,[1.1,28.5,1.1,1.8,0.3,0.02,0.3])
##
ftest_geom31(p)  =   sum(smoov11(geom1(p[4:7]),f_ε_mats3(p[1:3]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom32(p)  =   sum(smoov12(geom1(p[4:7]),f_ε_mats3(p[1:3]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom33(p)  =   sum(smoov13(geom1(p[4:7]),f_ε_mats3(p[1:3]),(1,2,3,4),Grid(6.,4.,256,128)))
ftest_geom31(p);
ftest_geom32(p);
ftest_geom33(p);
Zygote.gradient(ftest_geom31,p)
Zygote.gradient(ftest_geom32,p)
Zygote.gradient(ftest_geom33,p)

using BenchmarkTools
@btime sin(3.3)

@btime Zygote.gradient(ftest_geom31,p)
@btime Zygote.gradient(ftest_geom32,p)
@btime Zygote.gradient(ftest_geom33,p)

##
eps_mod1 = get_model(LNrot,:ε,:ω,:T,:θ)


model0 = get_model(LNrot.parent,:ε,:ω,:T)
model1 = rotate(get_model(LNrot.parent,:ε,:ω,:T),LNrot.rotation)
corners(grid)[116,64]
crnrs = corners(grid)[116,64]
smoov1_single(shapes,mat_vals,minds,crnrs)
sum(smoov1_single(shapes,mat_vals,minds,crnrs))



Zygote.gradient(x->sum(f_ε_mats2(x)),[1.1,30.0])
FiniteDifferences.grad(central_fdm(5,1),x->sum(f_ε_mats2(x)),[1.1,30.0])


























##
using ChainRulesCore
# ChainRulesCore differentiation rules for SArray and subtypes
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
# ChainRulesCore differentiation rules for SArray and subtypes
ChainRulesCore.rrule(T::Type{<:MArray}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MArray}, x::AbstractArray) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:MMatrix}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MMatrix}, x::AbstractMatrix) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:MVector}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MVector}, x::AbstractVector) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )

Zygote.refresh()
##

##














epse12 = εₑ_∂ωεₑ_∂²ωεₑ(rand(),normcart(vec3D(normalize(rand(2)))),mat_vals[:,1],mat_vals[:,2])
epse13 = εₑ_∂ωεₑ_∂²ωεₑ(rand(),normcart(vec3D(normalize(rand(2)))),mat_vals[:,1],mat_vals[:,3])



p2              =   rand(17)

sh2             =   demo_shapes2D(p2)
cs2             =   corner_sinds.((sh2,),corners(gr2))
ps2             =   proc_sinds(corner_sinds(sh2,collect(x⃗c(gr2))))
n_unique_sinds2 =   length.(unique.(cs2))
minds2          =   1:(length(sh2)+1)
mat_vals2       =   (getproperty.(sh2, (:data,))..., diagm([0.9,0.9,0.9]),)

sm1s1           =   smoov1_single(sh2,mat_vals2,minds2,first(corners(gr2)))
sm11            =   smoov11(sh2,mat_vals2,collect(minds2),gr2)
sm12            =   smoov12(sh2,mat_vals2,collect(minds2),gr2)
sm13            =   smoov13(sh2,mat_vals2,collect(minds2),gr2)


fig,ax1,hm11_11 = heatmap(reshape(getindex.(sm11,(1,),(1,)),size(grid)))

function foo2(p)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{2,Float64}(0.1,0.1), SVector{2,Float64}(0.2,0.2), SVector{2,Float64}(0.1,0.25), SVector{2,Float64}(-0.1,0.14) )
    return smoov1_single(shapes2,matvals2,minds2,crnrs)
end

foo2s(p) = sum(foo2(p))

function foo3(p)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{3,Float64}(0.1,0.1,0.1), SVector{3,Float64}(0.2,0.2,0.2), SVector{3,Float64}(0.1,0.3,0.25), SVector{3,Float64}(-0.1,0.13,0.14) )
    return smoov1_single(shapes2,matvals2,minds2,crnrs)
end

foo3s(p) = sum(foo3(p))

p0 = rand(17)
foo2(p0)
foo2s(p0)
Zygote.gradient(foo2s,p0)

grsm1s1          =   Zygote.gradient(p2) do p
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{3,Float64}(0.1,0.1,0.1), SVector{3,Float64}(0.2,0.2,0.2), SVector{3,Float64}(0.1,0.3,0.25), SVector{3,Float64}(-0.1,0.13,0.14) )
    smval = smoov1_single(shapes2,matvals2,minds2,crnrs)
    return sum(smval)
end

grsm1s2          =   Zygote.gradient(p2) do p
    grid  =   Grid(3.0,4.0,64,128)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    smval = smoov1_single(shapes2,matvals2,minds2,first(corners(grid)))
    return sum(smval)
end

##
grsm11          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov11(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end
grsm12          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov12(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end

grsm13          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov13(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end



function ff1(p)
    shapes = demo_shapes2D(p)
    # data_sum = sum( getproperty.(shapes,(:data,))) 
    data_sum = sum( map(ss->getproperty(ss,:data),shapes) ) 
    return sum(data_sum)
end
ff1(p2)
grff1 = Zygote.gradient(ff1,p2)[1]
typeof(grff1)

function ff2(p)
    shapes = demo_shapes2D(p)
    # data_sum = sum( getproperty.(shapes,(:data,))) 
    xyz = SVector{3,Float64}(0.0,0.1,0.2)
    data_sum = sum( map(ss->first(first(surfpt_nearby(xyz,ss))),shapes) ) 
    return data_sum
end
ff2(p2)
grff2 = Zygote.gradient(ff2,p2)[1]
typeof(grff2)


shapes = demo_shapes2D(rand(17))
ff31(x,y,z) = first(first(surfpt_nearby(SVector{3}(x,y,z),shapes[1])))
ff31(0.3,0.2,1.1)
Zygote.gradient(ff31,0.3,0.2,1.1)

ff312(x,y,z) = first(first(surfpt_nearby(SVector{2}(x,y),shapes[1])))
ff312(0.3,0.2,1.1)
Zygote.gradient(ff312,0.3,0.2,1.1)

ff322(x,y,z) = first(first(surfpt_nearby(SVector{2}(x,y),shapes[2])))
ff322(0.3,0.2,1.1)
Zygote.gradient(ff322,0.3,0.2,1.1)

ff32(x,y,z) = first(first(surfpt_nearby(SVector{3}(x,y,z),shapes[2])))
ff32(0.3,0.2,1.1)
Zygote.gradient(ff32,0.3,0.2,1.1)


##
println("smoov11 benchmark")
@btime smoov11(sh2,mat_vals2,collect(minds2),gr2)
println("smoov12 benchmark")
@btime smoov12(sh2,mat_vals2,collect(minds2),gr2)
println("smoov13 benchmark")
@btime smoov13(sh2,mat_vals2,collect(minds2),gr2)
##

## if plotting
using CairoMakie

let grid=gr2
    fig = Figure(resolution = (600, 400))
    x_grid = OptiMode.x(grid)
    y_grid = OptiMode.y(grid)

    ax11 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax12 = Axis(fig[1, 3]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax21 = Axis(fig[2, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax22 = Axis(fig[2, 3]; xlabel = "x", ylabel = "y", aspect=DataAspect())

    # heatmap of lower-left corner shape index ("sind") at each gridpoint
    hm2_sinds   = heatmap!(
        ax11,
        x_grid,
        y_grid,
        reshape(getindex.(cs2,(1,)),size(grid)),
    ) 
    cb11 = Colorbar(fig[1, 2], hm2_sinds; label = "foreground shape index", width = 15, ticksize = 15, tickalign = 1)
    # map of number of shapes at pixel corners, showing edges
    hm2_nunq    = heatmap!(
        ax12,
        x_grid,
        y_grid,
        reshape(n_unique_sinds2,size(grid)),
    )
    cb12 = Colorbar(fig[1, 4], hm2_nunq; label = "number of unique shape indices", width = 15, ticksize = 15, tickalign = 1)
    # heatmap of smoothed (1,1) tensor elements 
    hm2_sm11    = heatmap!(
        ax21,
        x_grid,
        y_grid,
        reshape(getindex.(sm11,(1,)),size(grid)),
    )
    cb21 = Colorbar(fig[2, 2], hm2_sm11; label = "smoothed diagonal tensor value", width = 15, ticksize = 15, tickalign = 1)
    # heatmap of smoothed (1,2) tensor elements 
    # hm2_sm12    = heatmap!(ax22,reshape(getindex.(sm11,(2,)),size(gr2)))
    # cb22 = Colorbar(fig[2, 4], hm2_sm12; label = "values", width = 15, ticksize = 15, tickalign = 1)

    
    
    #colsize!(fig.layout, 1, Aspect(1, 1.0))
    #colgap!(fig.layout, 7)
    # display(fig)
    fig
end


##
@show gr2_inds        =   rand(1:length(gr2)) # rand(eachindex(gr2))
crnrs2          =   collect(corners(gr2))[gr2_inds]
@show smval2          =   smoov_single1(sh2,mat_vals2,minds2,crnrs2)


sm11            =   smoov_single1(sh2,mat_vals2,minds2,first(corners(gr2)))
sm1             =   smoov1(sh2,mat_vals2,collect(minds2),gr2)
##


gr3 =   Grid(Δx,Δy,Δz,Nx,Ny,Nz)

@test xc(gr2) ≈ xc(gr3)
@test yc(gr2) ≈ yc(gr3)
# @test zc(gr2) ≈ zc(gr3)
@test xc(gr2)[1:end-1]  ≈   ( OptiMode.x(gr2) .- (δx(gr2)/2.0,) )
@test xc(gr2)[2:end]    ≈   ( OptiMode.x(gr2) .+ (δx(gr2)/2.0,) )
@test yc(gr2)[1:end-1]  ≈   ( OptiMode.y(gr2) .- (δy(gr2)/2.0,) )
@test yc(gr2)[2:end]    ≈   ( OptiMode.y(gr2) .+ (δy(gr2)/2.0,) )

@test xc(gr3)[1:end-1]  ≈   ( OptiMode.x(gr3) .- (δx(gr3)/2.0,) )
@test xc(gr3)[2:end]    ≈   ( OptiMode.x(gr3) .+ (δx(gr3)/2.0,) )
@test yc(gr3)[1:end-1]  ≈   ( OptiMode.y(gr3) .- (δy(gr3)/2.0,) )
@test yc(gr3)[2:end]    ≈   ( OptiMode.y(gr3) .+ (δy(gr3)/2.0,) )
@test zc(gr3)[1:end-1]  ≈   ( OptiMode.z(gr3) .- (δz(gr3)/2.0,) )
@test zc(gr3)[2:end]    ≈   ( OptiMode.z(gr3) .+ (δz(gr3)/2.0,) )