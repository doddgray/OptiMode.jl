using LinearAlgebra, StaticArrays, GeometryPrimitives, OptiMode, Test
using ChainRules, Zygote, FiniteDifferences, ForwardDiff, FiniteDiff
using OptiMode: vec3D
# using CairoMakie
# using CairoMakie
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)

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

function solve_k_compare(ω,ε⁻¹,grid;nev=2,eig_tol,k_tol)
    kmags_mpb,evecs_mpb = solve_k(ω,ε⁻¹,grid,MPB_Solver();nev,eig_tol,k_tol,overwrite=true)
    kmags_kk,evecs_kk   = solve_k(ω,ε⁻¹,grid,KrylovKitEigsolve();nev,eig_tol,k_tol)
    kmags_is,evecs_is   = solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    kmags_df,evecs_df   = solve_k(ω,ε⁻¹,grid,DFTK_LOBPCG();nev,eig_tol,k_tol)
    kmags_all   =   hcat(kmags_mpb,kmags_kk,kmags_is,kmags_df)
    evecs_all   =   hcat(evecs_mpb,evecs_kk,evecs_is,evecs_df)
    Es_all      =   map((k,ev)->E⃗(k,ev,ε⁻¹,∂ε_∂ω,grid;normalized=false,canonicalize=false),kmags_all,evecs_all)
    plot_fields_grid(permutedims(Es_all),grid;grid_xlabels=["mode $i" for i in 1:nev],
        grid_ylabels=["MPB","KrylovKit","IterativeSolvers","DFTK"],)
    # Evmms_all   =   map(val_magmax,Es_all)
    return kmags_all, evecs_all, Es_all
end

function compare_fields(f1,f2)
    print("field comparison\n")
    @show maximum(abs2.(f1))
    @show maximum(abs2.(f2))
    @show maximum(real(f1))
    @show minimum(real(f1))
    @show maximum(real(f2))
    @show minimum(real(f2))
    @show maximum(imag(f1))
    @show minimum(imag(f1))
    @show maximum(imag(f2))
    @show minimum(imag(f2))
    @show maximum(abs.(real(f1)))
    @show maximum(abs.(real(f2)))
    @show maximum(abs.(imag(f1)))
    @show maximum(abs.(imag(f2)))
    print("\n")
    return
end

function plot_compare_fields(F1,F2,grid;xlim=extrema(x(grid)),ylim=extrema(y(grid)),cmap=:diverging_bkr_55_10_c35_n256,)
	fig = Figure()
	@assert isequal(size(F1),size(F2))
    # ND1 = size(F1)[1]
    
    labels1r = ["real(F1_1)","real(F1_2)","real(F1_3)"]
    labels1i = ["imag(F1_1)","imag(F1_2)","imag(F1_3)"]
    labels2r = ["real(F2_1)","real(F2_2)","real(F2_3)"]
    labels2i = ["imag(F2_1)","imag(F2_2)","imag(F2_3)"]
    labelsdr = ["real((F1-F2)_1)","real((F1-F2)_2)","real((F1-F2)_3)"]
    labelsdi = ["imag((F1-F2)_1)","imag((F1-F2)_2)","imag((F1-F2)_3)"]

    ax11r = fig[1,1] = Axis(fig,backgroundcolor=:transparent,title=labels1r[1])
    ax12r = fig[1,2] = Axis(fig,backgroundcolor=:transparent,title=labels1r[2])
    ax13r = fig[1,3] = Axis(fig,backgroundcolor=:transparent,title=labels1r[3])
    ax11i = fig[2,1] = Axis(fig,backgroundcolor=:transparent,title=labels1i[1])
    ax12i = fig[2,2] = Axis(fig,backgroundcolor=:transparent,title=labels1i[2])
    ax13i = fig[2,3] = Axis(fig,backgroundcolor=:transparent,title=labels1i[3])

    ax21r = fig[3,1] = Axis(fig,backgroundcolor=:transparent,title=labels2r[1])
    ax22r = fig[3,2] = Axis(fig,backgroundcolor=:transparent,title=labels2r[2])
    ax23r = fig[3,3] = Axis(fig,backgroundcolor=:transparent,title=labels2r[3])
    ax21i = fig[4,1] = Axis(fig,backgroundcolor=:transparent,title=labels2i[1])
    ax22i = fig[4,2] = Axis(fig,backgroundcolor=:transparent,title=labels2i[2])
    ax23i = fig[4,3] = Axis(fig,backgroundcolor=:transparent,title=labels2i[3])

    axd1r = fig[5,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[1])
    axd2r = fig[5,2] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[2])
    axd3r = fig[5,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[3])
    axd1i = fig[6,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[1])
    axd2i = fig[6,2] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[2])
    axd3i = fig[6,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[3])

    ax1r = [ax11r,ax12r,ax13r]
    ax1i = [ax11i,ax12i,ax13i]
    ax2r = [ax21r,ax22r,ax23r]
    ax2i = [ax21i,ax22i,ax23i]
    axdr = [axd1r,axd2r,axd3r]
    axdi = [axd1i,axd2i,axd3i]

    Fd = F1 .- F2
    magmax1 = max(maximum(abs,real(F1)),maximum(abs,imag(F1)))
    magmax2 = max(maximum(abs,real(F2)),maximum(abs,imag(F2)))
    magmaxd = max(maximum(abs,real(Fd)),maximum(abs,imag(Fd)))
    x_um = x(grid)
    y_um = y(grid)
    hms1r = [heatmap!(ax1r[didx],x_um,y_um,real(F1[didx,:,:]);colorrange=(-magmax1,magmax1)) for didx=1:3]
    hms1i = [heatmap!(ax1i[didx],x_um,y_um,imag(F1[didx,:,:]);colorrange=(-magmax1,magmax1)) for didx=1:3]
    hms2r = [heatmap!(ax2r[didx],x_um,y_um,real(F2[didx,:,:]);colorrange=(-magmax2,magmax2)) for didx=1:3]
    hms2i = [heatmap!(ax2i[didx],x_um,y_um,imag(F2[didx,:,:]);colorrange=(-magmax2,magmax2)) for didx=1:3]
    hmsdr = [heatmap!(axdr[didx],x_um,y_um,real(Fd[didx,:,:]);colorrange=(-magmaxd,magmaxd)) for didx=1:3]
    hmsdi = [heatmap!(axdi[didx],x_um,y_um,imag(Fd[didx,:,:]);colorrange=(-magmaxd,magmaxd)) for didx=1:3]

    cb1r = Colorbar(fig[1,4], hms1r[1],  width=20 )
    cb1i = Colorbar(fig[2,4], hms1i[1],  width=20 )
    cb2r = Colorbar(fig[3,4], hms2r[1],  width=20 )
    cb2i = Colorbar(fig[4,4], hms2i[1],  width=20 )
    cbdr = Colorbar(fig[5,4], hmsdr[1],  width=20 )
    cbdi = Colorbar(fig[6,4], hmsdi[1],  width=20 )


    ax = (ax1r...,ax1i...,ax2r...,ax2i...,axdr...,axdi...)
    for axx in ax
        # axx.xlabel= "x [μm]"
        # xlims!(axx,xlim)
        # ylims!(axx,ylim)
        hidedecorations!(axx)
        axx.aspect=DataAspect()

    end
    return fig
end

mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
mat_vars = (:ω,)
np_mats = length(mat_vars)
f_ε_mats, f_ε_mats! = _f_ε_mats(mats,mat_vars) # # f_ε_mats, f_ε_mats! = _f_ε_mats(vcat(materials(sh1),Vacuum),(:ω,))
ω               =   1.1 #1.0
p               =   [ω, 2.0,0.8,0.1,0.1] #rand(4+np_mats);
mat_vals        =   f_ε_mats(p[1:np_mats]);
grid            =   Grid(6.,4.,256,128);
shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
minds           =   (1,2,3,4)
sm1             =   smooth_ε(shapes,mat_vals,minds,grid);
ε               =   copy(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(sm1,3,2)); # sm1[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);


@test herm(ε) ≈ ε
@test herm(ε⁻¹) ≈ ε⁻¹
@test herm(∂ε_∂ω) ≈ ∂ε_∂ω
@test _dot(ε⁻¹, ε) ≈ reshape(repeat([1,0,0,0,1,0,0,0,1],N(grid)),(3,3,size(grid)...))

##
function ff_eps1(p;Dx=6.0,Dy=4.0,Nx=256,Ny=256)
    grid            =   Grid(Dx,Dy,Nx,Ny)
    ω               =   p[1]
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid);
    ε               =   real(herm(copy(selectdim(sm1,3,1)))); 
    return  ε
end

function ff_deps1(p;Dx=6.0,Dy=4.0,Nx=256,Ny=256)
    grid            =   Grid(Dx,Dy,Nx,Ny)
    ω               =   p[1]
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid);
    ∂ε_∂ω           =   real(herm(copy(selectdim(sm1,3,2)))); 
    return  ∂ε_∂ω
end

function ff_ddeps1(p;Dx=6.0,Dy=4.0,Nx=256,Ny=256)
    grid            =   Grid(Dx,Dy,Nx,Ny)
    ω               =   p[1]
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid);
    ∂²ε_∂ω²         =   real(herm(copy(selectdim(sm1,3,3))));
    return     ∂²ε_∂ω²
end

p = [1.1,2.0,0.8,0.1,0.1]

eps1,deps1,ddeps1 = map(ff->ff(p),(ff_eps1,ff_deps1,ff_ddeps1))
deps1_FD,ddeps1_FD,dddeps1_FD = map((ff_eps1,ff_deps1,ff_ddeps1)) do ff
    FiniteDiff.finite_difference_derivative(oo->ff([oo,p[2:5]...]),p[1])
end


(k1,ng1), n1_ng1_pb = Zygote.pullback(ffnng1,p)
gr_n1_RM = n1_ng1_pb((1.0,nothing))[1]
gr_ng1_RM = n1_ng1_pb((nothing,1.0))[1]
gr_n1_FD = gradFD2(x->ffnng1(x)[1],p;rs=1e-5)
gr_ng1_FD = gradFD2(x->ffnng1(x)[2],p;rs=1e-5)


##
nev, eig_tol, k_tol = 2, 1e-7, 1e-7
kmags_mpb,evecs_mpb = solve_k(ω,ε⁻¹,grid,MPB_Solver();nev,eig_tol,k_tol,overwrite=true)
kmags_kk,evecs_kk   = solve_k(ω,ε⁻¹,grid,KrylovKitEigsolve();nev,eig_tol,k_tol)
kmags_is,evecs_is   = solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
kmags_df,evecs_df   = solve_k(ω,ε⁻¹,grid,DFTK_LOBPCG();nev,eig_tol,k_tol)
kmags_all   =   hcat(kmags_mpb,kmags_kk,kmags_is,kmags_df)
evecs_all   =   hcat(evecs_mpb,evecs_kk,evecs_is,evecs_df)
Es_all      =   map((k,ev)->E⃗(k,ev,ε⁻¹,∂ε_∂ω,grid;normalized=false,canonicalize=false),kmags_all,evecs_all)
plot_fields_grid(permutedims(Es_all),grid;grid_xlabels=["mode $i" for i in 1:size(Es_all,1)],
    grid_ylabels=["MPB","KrylovKit","IterativeSolvers","DFTK"],)

function ff1(ω,ε⁻¹;nev=2,eig_tol=1e-9,k_tol=1e-9,solver=DFTK_LOBPCG())
    kmags,evecs   = solve_k(ω,ε⁻¹,Grid(6.0,4.0,256,128),solver;nev,eig_tol,k_tol)
    return sum(kmags) + abs2(sum(sum(evecs)))
end

ff1(1.1,ε⁻¹)
Zygote.gradient(ff1,1.1,ε⁻¹)

function ffnng1(p;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=256,Ny=256,solver=IterativeSolversLOBPCG())
    grid            =   Grid(Dx,Dy,Nx,Ny)
    ω               =   p[1]
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid);
    ε               =   real(herm(copy(selectdim(sm1,3,1)))); # sm1[:,:,1,:,:]  
    ∂ε_∂ω           =   real(herm(copy(selectdim(sm1,3,2)))); # sm1[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    # kmags,evecs   = solve_k(p[1],ε⁻¹,grid,DFTK_LOBPCG();nev,eig_tol,k_tol)
    kmags,evecs   = solve_k(ω,ε⁻¹,grid,solver;nev,eig_tol,k_tol)
    k1 = kmags[1] #/ω
    ev1 = evecs[1]
    ng1 = group_index(kmags[1],first(evecs),ω,ε⁻¹,∂ε_∂ω,grid)
    # mag,mn = mag_mn(k1,grid) 
	# ng1 = (ω + HMH(vec(ev1), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(ev1),ε⁻¹,mag,mn)
    return k1,ng1
end
# ffnng1(p)
p = [1.1, 2.0,0.8,0.1,0.1]
(k1,ng1), n1_ng1_pb = Zygote.pullback(ffnng1,p)
gr_n1_RM = n1_ng1_pb((1.0,nothing))[1]
gr_ng1_RM = n1_ng1_pb((nothing,1.0))[1]
gr_n1_FD = gradFD2(x->ffnng1(x)[1],p;rs=1e-5)
gr_ng1_FD = gradFD2(x->ffnng1(x)[2],p;rs=1e-5)






##
using FFTW
p = [1.1, 2.0,0.8,0.1,0.1]
ω = p[1]
nev=2
eig_tol=1e-9
k_tol=1e-9
Dx = 6.0
Dy = 4.0
Nx=256
Ny=256
grid = Grid(Dx,Dy,Nx,Ny)
sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid);
ε               =   real(copy(selectdim(sm1,3,1))); # sm1[:,:,1,:,:]  
∂ε_∂ω           =   real(copy(selectdim(sm1,3,2))); # sm1[:,:,2,:,:]
ε⁻¹             =   sliceinv_3x3(ε);
kmags,evecs   = solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
##
k = kmags[1]
ev = first(evecs)
mag, mn = mag_mn(k1,g⃗(grid))
ev_grid = reshape(ev,(2,Nx,Ny))
nng = (ω/2) * ∂ε_∂ω + ε
norm_fact = inv(sqrt(δV(grid) * N(grid)) * ω)

D = fft(kx_tc(ev_grid,mn,mag),(2:3)) * norm_fact
E = ε⁻¹_dot( D, ε⁻¹)
real(_expect(∂ε_∂ω,E)) * δV(grid)
real(_expect(ε,E)) * δV(grid)
sum(abs2,E) *δV(grid)


HMH(ev,  ε⁻¹ ,mag,mn)
HMₖH(vec(ev),ε⁻¹,mag,mn)
ω / HMₖH(vec(ev),ε⁻¹,mag,mn)
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(1/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(1/2ω)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+ HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/(2ω))
(ω + HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(ev),ε⁻¹,mag,mn)
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(1/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(1/2ω)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))

# E = 1im * ε⁻¹_dot( fft( kx_tc( ev_grid,mns,mag), (2:1+ND) ), ε⁻¹)
# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
H1 = fft( tc(ev_grid,mn), (2:3) ) * (-ω)
P1 = 2*real(_sum_cross_z(conj(E1),H1))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
W1 = real(dot(E1,_dot(nng,E1))) + (N(grid)* (ω^2))     # energy density per unit length
ng11 = real( W1 / P1 )

om / HMₖH(vec(evec),ε⁻¹,mag,mn) * (1-(om/2)*HMH(evec, _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ ),mag,mn))

E1 = 1im * ε⁻¹_dot( fft( kx_tc(ev_grid,mn,mag), (2:3) ), real(ε⁻¹))
# H1 = fft(tc(kx_ct( ifft( E1, (2:3) ), mn,mag), mn),(2:3) ) 
H1 = 1im * fft( tc(ev_grid,mn), (2:3) )
P1 = 2*real(_sum_cross_z(conj(E1),H1))
W1 = real(dot(E1,_dot(nng,E1))) + (N(grid)* (ω^2))
@show ng11 = real( W1 / P1 )
ng11 / ω

real(dot(E1,_dot(ε,E1))) / real(dot(E1,E1))  * norm_fact

(2*real(dot(E1,_dot(ε,E1))) + ω * real(dot(E1,_dot(∂ε_∂ω ,E1))) ) / real(_sum_cross_z(conj(E1),H1))

@show ng12 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
@show ng13 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
@show ng14 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(ω/2)*HMH(ev, _dot( ε⁻¹, nng, ε⁻¹ ),mag,mn))
@show ng15 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(ω/2)*HMH(ev, _dot( ε⁻¹, nng, ε⁻¹ ),mag,mn))

@show ng12 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1+(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
@show ng13 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))

@show dk_dom_RM / ng11
@show inv(dk_dom_RM / ng11)
@show dk_dom_RM / ng12
@show inv(dk_dom_RM / ng12)
@show dk_dom_RM / ng13
@show inv(dk_dom_RM / ng13)




E1 = 1im * ε⁻¹_dot( fft( kx_tc(ev_grid,mn,mag), (2:3) ), real(ε⁻¹))
H1 = (-1im * ω) * fft( tc(ev_grid,mn), (2:3) )
P1 = 2*real(_sum_cross_z(E1,H1))
W1 = real(dot(E1,_dot(nng,E1))) + (N(grid)* (ω^2))
ng11 = real( W1 / P1 )


real(_expect(ε,E1)) / dot(E1,E1)

real(_expect(ε,E1))

W12 = real(dot(E1,_dot(ε,E1))) + (ω/2) * real(dot(E1,_dot(∂ε_∂ω,E1)))
W12 ≈ W1
_expect

D1 = 1im * fft( kx_tc( ev_grid,mn,mag), _fftaxes(grid) )
E1 = ε⁻¹_dot( D1, ε⁻¹)
# E = 1im * ε⁻¹_dot( fft( kx_tc( ev_grid,mns,mag), (2:1+ND) ), ε⁻¹)
# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
H1 = fft( tc(ev_grid,mn), (2:3) ) * (-ω)
P1 = 2*real(_sum_cross_z(conj(E1),H1))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
W1 = real(dot(E1,_dot(nng,E1))) + (N(grid)* (ω^2))     # energy density per unit length
ng11 = real( W1 / P1 )


E2 = E⃗(k,ev_grid,ε⁻¹,∂ε_∂ω,grid; canonicalize=true, normalized=true)

W = real(dot(E,_dot(nng,E))) + ( ω^2 * size(H,2) * size(H,3) )
@tullio P_z := conj(E)[1,ix,iy] * H[2,ix,iy] - conj(E)[2,ix,iy] * H[1,ix,iy]
ng11 = W / (2*real(P_z))



##
function ffng1(Hₜ::AbstractArray{Complex{T},3},ω,ε⁻¹,nng,mag,mn)::T where T<:Real
	E = 1im * ε⁻¹_dot( fft( kx_tc(Hₜ,mn,mag), (2:3) ), real(ε⁻¹))
	H = (-1im * ω) * fft( tc(Hₜ,mn), (2:3) )
	W = real(dot(E,_dot(nng,E))) + ( ω^2 * size(H,2) * size(H,3) )
	@tullio P_z := conj(E)[1,ix,iy] * H[2,ix,iy] - conj(E)[2,ix,iy] * H[1,ix,iy]
	return W / (2*real(P_z))
end

function ffng1(Hₜ::AbstractVector{Complex{T}},ω,ε⁻¹,nng,mag::AbstractArray{T,2},mn::AbstractArray{T,4})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(Hₜ,(2,Nx,Ny))
	ffng1(Ha,ω,ε⁻¹,nng,mag,mn)
end

k1 = kmags[1]
ev1 = first(evecs)
mag1, mn1 = mag_mn(k1,g⃗(grid))
mag12, mv1, nv1 = mag_m_n(k1,g⃗(grid))
m1,n1 = reinterpret(reshape,Float64,mv1), reinterpret(reshape,Float64,nv1)
ng11 = group_index(k1,ev1,ω,ε⁻¹,∂ε_∂ω,grid)
ng12 = ffng1(ev1,ω,ε⁻¹,∂ε_∂ω,mag1,mn1)
ng12 = ffng1(ev1,ω,ε⁻¹,(ε + (ω/2)*∂ε_∂ω),mag1,mn1)


##

ω               =   1.1 #1.0
p               =   [ω, 2.0,0.8,0.1,0.1] #rand(4+np_mats);
mat_vals        =   f_ε_mats(p[1:np_mats]);
grid            =   Grid(6.,4.,256,128);
shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
minds           =   (1,2,3,4)
sm1             =   smooth_ε(shapes,mat_vals,minds,grid);
ε               =   copy(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(sm1,3,2)); # sm1[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);
kmags,evecs     =   solve_k(ω,ε⁻¹,grid,DFTK_LOBPCG();nev=2,eig_tol=1e-8,k_tol=1e-8);
kmags_mpb,evecs_mpb = find_k(ω,ε,grid;num_bands=nev,eig_tol=1e-8,k_tol=1e-8,overwrite=true,save_efield=true,save_hfield=true);

@assert _dot(ε⁻¹ * ε) ≈  
##
eigind = 1
k, ev = kmags[eigind], evecs[eigind];
H1 = copy(reshape(ev,(2,size(ms.grid)...)));
mag,mn = mag_mn(k,grid)
ms = ModeSolver(k, ε⁻¹,grid; nev=2, maxiter=200)
Mev = ms.M̂ * copy(ev)
Mev2 = vec(kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(reshape(ev,(2,grid.Nx,grid.Ny)),mn,mag), (2:3) ), real(flat(ε⁻¹))), (2:3)),mn,mag))
Mkev = vec(kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(reshape(ev,(2,grid.Nx,grid.Ny)),mn), (2:3) ), real(flat(ε⁻¹))), (2:3)),mn,mag))
dot(ev,Mev)
dot(ev,Mev2)
HMH(H1,ε⁻¹,mag,mn)
HMₖH(H1,ε⁻¹,mag,mn)
ng11 = ω / HMₖH(vec(ev),ε⁻¹,mag,mn) * (1-(ω/2)*HMH(ev, _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn))
ω / HMₖH(H1,ε⁻¹,mag,mn) * (1.0 + (ω/2.0)*HMH(H1,_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn) )
group_index(k,ev,ω,ε⁻¹,∂ε_∂ω,Grid(6.,4.,256,128))

dot(H1, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H1,mn), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) )

ev12 = canonicalize_phase(ev,k,ms.M̂.ε⁻¹,ms.grid)
val_magmax(ev12) / val_magmax(ev)
ev12 ≈ ev

gr_k1_FD = gradFD2(x->((ffnng1(x)[1])*(x[1])),p;rs=1e-3)
dk_dom_FD = gr_k1_FD[1]

function ffnng1_mpb(p;nev=2,eig_tol=1e-9,k_tol=1e-9)
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128));
    ε               =   real(copy(selectdim(sm1,3,1))); # sm1[:,:,1,:,:]  
    ∂ε_∂ω           =   real(copy(selectdim(sm1,3,2))); # sm1[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs   = solve_k(p[1],ε⁻¹,Grid(6.,4.,256,128),MPB_Solver();nev,eig_tol,k_tol,overwrite=true)
    n1 = kmags[1]/p[1]
    ng1 = group_index(kmags[1],first(evecs),p[1],ε⁻¹,∂ε_∂ω,Grid(6.,4.,256,128))
    return n1,ng1
end


n1_mpb,ng1_mpb = ffnng1_mpb(p)
gr_k1_mpb_FD = gradFD2(x->((ffnng1_mpb(x)[1])*(x[1])),p;rs=1e-4)
dk_dom_mpb_FD = gr_k1_mpb_FD[1]
H1_mpb = copy(H1)
HMH(H1_mpb,ε⁻¹,mag,mn)
HMₖH(H1_mpb,ε⁻¹,mag,mn)
ω / HMₖH(H1_mpb,ε⁻¹,mag,mn) * (1.0 + (ω/2.0)*HMH(H1_mpb,_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn) )
ω / HMₖH(H1_mpb,ε⁻¹,mag,mn) * (1.0 + (ω/2.0)*HMH(H1_mpb,_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn) )
ω / HMₖH(H1_mpb,ε⁻¹,mag,mn)
dot(H1_mpb,H1_mpb)

function Mx1!(Hout,Hin,M̂)
    kx_tc!(M̂.d,Hin,M̂.mn,M̂.mag);
    mul!(M̂.d,M̂.𝓕!,M̂.d);
    eid!(M̂.e,M̂.ε⁻¹,M̂.d);
    mul!(M̂.e,M̂.𝓕⁻¹!,M̂.e);
    kx_ct!(Hout,M̂.e,M̂.mn,M̂.mag,M̂.Ninv);
	return Hout
end

function Mx1(Hin,M̂)
    d̃ = kx_tc(Hin,M̂.mn,M̂.mag);
    d = fft(d̃,2:3);
    e = ε⁻¹_dot(d,ε⁻¹);
    ẽ = ifft(e,2:3);
    Hout = kx_ct(ẽ,M̂.mn,M̂.mag);
	return Hout
end

Hin1 = copy(reshape(ev,(2,grid.Nx,grid.Ny)))
Hout1 = similar(Hin1)
Mx1!(Hout1,Hin1,ms.M̂)
Hout2 = Mx1(Hin1,ms.M̂)

dot(ev,vec(Hout1))
dot(ev,vec(Hout2))

HMH

    

##
mn ≈ ms.M̂.mn
mag ≈ ms.M̂.mag

using OptiMode: _H2d!, _d2ẽ!
H1 = reshape(ev,(2,size(ms.grid)...));
d1 = similar(ms.M̂.d);
e1 = similar(ms.M̂.e);
_H2d!(d1,H1,ms.M̂);
d2 = similar(d1);
kx_tc!(d2,H1,mn,mag);
mul!(d2,ms.M̂.𝓕!,d2);
d3 = fft(kx_tc(H1,mn,mag),2:3)
@test d1 ≈ d2
@test d2 ≈ 1im*d3

val_magmax(d1)
val_magmax(d3)

##
using OptiMode: load_field, tc
using FFTW
kmags_mpb,evecs_mpb = find_k(ω,ε,grid;num_bands=nev,eig_tol,k_tol,overwrite=true,save_efield=true,save_hfield=true);
E1_load_mpb = load_field("e.f01.b01.h5");
H1_load_mpb = load_field("h.f01.b01.h5");
norm_fact = inv(sqrt(δ(grid) * N(grid)) * ω)
E1_calc_mpb = norm_fact * E⃗(kmags_mpb[1],evecs_mpb[1],ε⁻¹,∂ε_∂ω,grid; canonicalize=false, normalized=false);
H1_calc_mpb = fft(tc(kx_ct( ifft( E1_calc_mpb, 2:3 ), mn,mag), mn), 2:3 ) / ω ; 

E1 = norm_fact * E⃗(kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,grid; canonicalize=false, normalized=false);
H1 = fft(tc(kx_ct( ifft( E1, 2:3 ), mn,mag), mn), 2:3 ) / ω ; 

E12 = norm_fact * E⃗(kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,grid; canonicalize=false, normalized=false);
val_magmax(E12)/val_magmax(E1)


plot_compare_fields(
    E1_load_mpb,
    E1_calc_mpb,
    grid,
)


val_magmax(E1)
val_magmax(E1_calc_mpb)
val_magmax(E1_load_mpb)
val_magmax(E1_calc_mpb) / val_magmax(E1_load_mpb)

val_magmax(H1)
val_magmax(H1_calc_mpb)
val_magmax(H1_load_mpb)
val_magmax(H1_calc_mpb) / val_magmax(H1_load_mpb)


inv(norm_fact)
# plot_compare_fields(
#     EE,
#     ε⁻¹_dot(fft(kx_tc(Hv,mn,mag),(2:3)),eepsi),
#     grid,
# )

# plot_compare_fields(
#     HH,
#     fft(tc(Hv,mn),(2:3)) * om,
#     grid,
# )

# plot_compare_fields(
#     HH,
#     fft(tc(kx_ct(ifft(EE,(2:3)),mn,mag),mm,nn),(2:3)) / om,
#     grid,
# )
