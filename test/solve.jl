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



# using KrylovKit, IterativeSolvers
# using DFTK: LOBPCG

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

function ff1(ω,ε⁻¹)
    kmags,evecs   = solve_k(ω,ε⁻¹,Grid(6.,4.,256,128),DFTK_LOBPCG();nev,eig_tol,k_tol)
    return sum(kmags) + abs2(sum(sum(evecs)))
end

ff1(1.0,ε⁻¹)
Zygote.gradient(ff1,1.0,ε⁻¹)

_adj(A::AbstractArray{<:Number,4}) = (At = permutedims(A,(2,1,3,4)); real(At) - 1.0im*imag(At))
_adj(A::AbstractArray{<:Real,4}) = permutedims(A,(2,1,3,4))
_herm(A::AbstractArray{<:Number,4}) = (A + _adj(A)) * 0.5
_herm(ε)
_herm(ε) ≈ ε
_herm(∂ε_∂ω) ≈ ∂ε_∂ω

function ff2(p;nev=2,eig_tol=1e-9,k_tol=1e-9)
    # mat_vals        =   f_ε_mats([ω,]); #f_ε_mats(p[1:np_mats]);
    # grid            =   Grid(6.,4.,256,128);
    # shapes          =   geom1(p[2:5]);
    # sm1             =   smooth_ε(shapes,mat_vals,(1,2,3,4),grid);
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128));
    # sm1             =   Zygote.forwarddiff(p) do p
    #     return smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128))
    # end
    ε               =   real(copy(selectdim(sm1,3,1))); # sm1[:,:,1,:,:]  
    ∂ε_∂ω           =   real(copy(selectdim(sm1,3,2))); # sm1[:,:,2,:,:]
    # ε               =   _herm(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
    # ∂ε_∂ω           =   _herm(selectdim(sm1,3,2)); # sm1[:,:,2,:,:]
    # ∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs   = solve_k(p[1],ε⁻¹,Grid(6.,4.,256,128),DFTK_LOBPCG();nev,eig_tol,k_tol)
    # kmags,evecs   = solve_k(ω,ε⁻¹,grid,MPB_Solver();nev,eig_tol,k_tol,overwrite=true)
    # ngs = map((k,ev)->group_index(k,ev,first(p),ε⁻¹,∂ε_∂ω,grid),zip(kmags,evecs))
    ng1 = group_index(kmags[1],first(evecs),p[1],ε⁻¹,∂ε_∂ω,Grid(6.,4.,256,128))
    return ng1
end
p = [1.0, 2.0,0.8,0.1,0.1]
ng11, gr_ng11_RM = Zygote.withgradient(ff2,p)
using FiniteDiff
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)
gr_ff2_FD = gradFD2(ff2,p;rs=1e-3)

##

function ff3(p;nev=2,eig_tol=1e-9,k_tol=1e-9)
    # p[1]
    # mat_vals        =   f_ε_mats([ω,]);
    # grid            =   Grid(6.,4.,256,128);
    # shapes          =   geom1(p[2:5]);
    # sm1             =   smooth_ε(shapes,mat_vals,(1,2,3,4),grid);
    sm1             =   smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128));
    # sm1             =   Zygote.forwarddiff(p) do p
    #     return smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128))
    # end
    ε               =   real(copy(selectdim(sm1,3,1))); # sm1[:,:,1,:,:]  
    ∂ε_∂ω           =   real(copy(selectdim(sm1,3,2))); # sm1[:,:,2,:,:]
    # ε               =   _herm(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
    # ∂ε_∂ω           =   _herm(selectdim(sm1,3,2)); # sm1[:,:,2,:,:]
    # ∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs   = solve_k(p[1],ε⁻¹,Grid(6.,4.,256,128),DFTK_LOBPCG();nev,eig_tol,k_tol)
    # kmags,evecs   = solve_k(ω,ε⁻¹,grid,MPB_Solver();nev,eig_tol,k_tol,overwrite=true)
    # ngs = map((k,ev)->group_index(k,ev,first(p),ε⁻¹,∂ε_∂ω,grid),zip(kmags,evecs))
    
    n1 = kmags[1]/p[1]
    ng1 = group_index(kmags[1],evecs[1],p[1],ε⁻¹,∂ε_∂ω,Grid(6.,4.,256,128))
    return n1, ng1
end
p = [1.0, 2.0,0.8,0.1,0.1]
(n1,ng1), n1_ng1_pb = Zygote.pullback(ff3,p)
gr_n1_RM = n1_ng1_pb((1.0,nothing))
gr_ng1_RM = n1_ng1_pb((nothing,1.0))
gr_n1_FD = gradFD2(x->ff3(x)[1],p;rs=1e-3)
gr_ng1_FD = gradFD2(x->ff3(x)[2],p;rs=1e-3)


##
#
p = [1.0, 2.0,0.8,0.1,0.1]
# mat_vals, mat_vals_pb  =   Zygote.pullback(p->f_ε_mats(p[1:np_mats]),p);
# grid            =   Grid(6.,4.,256,128);
# shapes, shapes_pb          =   Zygote.pullback(p->geom1(p[(np_mats+1):(np_mats+4)]),p);
# sm1, sm1_pb             =   Zygote.pullback((shapes,mat_vals)->smooth_ε(shapes,mat_vals,(1,2,3,4),grid),shapes,mat_vals);
sm1, sm1_pb             =   Zygote.pullback(p) do p
    return smooth_ε(geom1(p[(np_mats+1):(np_mats+4)]),f_ε_mats(p[1:np_mats]),(1,2,3,4),Grid(6.,4.,256,128))
end
ε , ε_pb               =   Zygote.pullback(sm1->_herm(selectdim(sm1,3,1)),sm1); # sm1[:,:,1,:,:]                 
∂ε_∂ω, ∂ε_∂ω_pb           =   Zygote.pullback(sm1->_herm(selectdim(sm1,3,2)),sm1); # sm1[:,:,2,:,:]
ε⁻¹, ε⁻¹_pb             =   Zygote.pullback(sliceinv_3x3,ε);
(kmags,evecs),kmags_evecs_pb   = Zygote.pullback((ω,ε⁻¹)->solve_k(ω,ε⁻¹,grid,DFTK_LOBPCG();nev=2,eig_tol=1e-10,k_tol=1e-12),p[1],ε⁻¹)
ng1,ng1_pb = Zygote.pullback((kmags,evecs,p,ε⁻¹,∂ε_∂ω)->group_index(first(kmags),first(evecs),first(p),ε⁻¹,∂ε_∂ω,grid),kmags,evecs,p,ε⁻¹,∂ε_∂ω)

ng12,ng12_pb = Zygote.pullback(kmags,evecs,p,ε⁻¹,∂ε_∂ω) do  kms,evs,pp,epsi,de_do
    grid = Grid(6.,4.,256,128)
    mag,mn = mag_mn(kms[1],grid)
    om = pp[1]
    return om / HMₖH(vec(evs[1]),epsi,mag,mn) * (1+(om/2)*HMH(evs[1], _dot( epsi, de_do, epsi ),mag,mn))
end
ng_z(evecs[1],1.0,ε⁻¹,∂ε_∂ω/2.,mag,mn)

mag,mn = mag_mn(kmags[1],Grid(6.,4.,256,128))
ms = ModeSolver(kmags[1], ε⁻¹,Grid(6.,4.,256,128); nev=2, maxiter=200)
Mev = ms.M̂ * copy(evecs[1])
Mev2 = vec(kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(reshape(evecs[1],(2,grid.Nx,grid.Ny)),mn,mag), (2:3) ), real(flat(ε⁻¹))), (2:3)),mn,mag))
Mkev = vec(kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(reshape(evecs[1],(2,grid.Nx,grid.Ny)),mn), (2:3) ), real(flat(ε⁻¹))), (2:3)),mn,mag))
dot(evecs[1],Mev)
dot(evecs[1],Mev2)




∂z_∂ng1 = 1.0
∂z_∂kmags,∂z_∂evecs,∂z_∂p_1,∂z_∂ε⁻¹_1,∂z_∂_∂ε_∂ω  =  ng1_pb(∂z_∂ng1)
∂z_∂kmags2,∂z_∂evecs2,∂z_∂p_12,∂z_∂ε⁻¹_12,∂z_∂_∂ε_∂ω2  =  ng12_pb(∂z_∂ng1)
∂z_∂ω,∂z_∂ε⁻¹_2 = kmags_evecs_pb((∂z_∂kmags,∂z_∂evecs))
∂z_∂p_2 = [∂z_∂ω, 0.0, 0.0, 0.0, 0.0  ]
∂z_∂ε = ε⁻¹_pb(∂z_∂ε⁻¹_1 + ∂z_∂ε⁻¹_2)[1]
∂z_∂sm1_1 = ε_pb(∂z_∂ε)[1]
∂z_∂sm1_2 = ∂ε_∂ω_pb(∂z_∂_∂ε_∂ω)[1]
∂z_∂p_3 = sm1_pb(∂z_∂sm1_1+∂z_∂sm1_2)[1]
# ∂z_∂shapes,∂z_∂matvals = sm1_pb(∂z_∂sm1_1+∂z_∂sm1_2)
# ∂z_∂p_3 = shapes_pb(∂z_∂shapes)[1]
# ∂z_∂p_4 = mat_vals_pb(∂z_∂matvals)[1]
@show ∂z_∂p = ∂z_∂p_1 + ∂z_∂p_2 + ∂z_∂p_3 
##
using OptiMode: HMₖH, ε⁻¹_bar, _H2d!, _d2ẽ!, mag_m_n, mag_mn, ∇ₖmag_m_n
using IterativeSolvers
using IterativeSolvers: bicgstabl!,bicgstabl,gmres,gmres!
T = Float64
ei_bar = zero(ε⁻¹)
ω_bar = 0.0
k = kmags[1]
ev = evecs[1]
k̄ = ∂z_∂kmags[1]
ēv = ∂z_∂evecs[1]
ms = ModeSolver(k, ε⁻¹,Grid(6.,4.,256,128); nev=2, maxiter=200)
gridsize= size(ms.grid)
λ⃗ = similar(ev)
λd =  similar(ms.M̂.d)
λẽ = similar(ms.M̂.d)
∂ω²∂k = 2 * HMₖH(ev,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
ev_grid = reshape(ev,(2,gridsize...))
λ⃗ 	-= 	 dot(ev,λ⃗) * ev
λ	=	reshape(λ⃗,(2,gridsize...))
d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), gridsize...)
# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
λd *=  ms.M̂.Ninv
λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ .* conj.(view(λ,2,axes(grid)...)) )
kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ .* conj.(view(λ,1,axes(grid)...)) )
m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,axes(grid)...])
n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,axes(grid)...])
māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
k̄ₕ = -∇ₖmag_m_n(
    māg,
    kx̄_m⃗.*ms.M̂.mag, # m̄,
    kx̄_n⃗.*ms.M̂.mag, # n̄,
    ms.M̂.mag,
    m⃗,
    n⃗;
    dk̂=SVector(0.,0.,1.), # dk⃗ direction
)

(mag2, m2, n2), mag_m_n_pb = Zygote.pullback(mag_m_n,k,ms.grid)
mag2 ≈ ms.M̂.mag
m2 ≈ m⃗
n2 ≈ n⃗

k̄ₕ2 = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1]

sum(abs2,λ⃗)
lm1_norm = dot(λ⃗,λ⃗)
ω² = ω^2
lm2 = similar(λ⃗)
res2,ch2 = bicgstabl!(
    lm2, # ms.adj_itr.x,	# recycle previous soln as initial guess
    ms.M̂ - real(ω²)*I, # A
    ēv - ev * dot(ev,ēv), # b,
    3;	# l = number of GMRES iterations per CG iteration
    Pl =ms.P̂, # Pl = HelmholtzPreconditioner(M̂), # left preconditioner
    log=true,
    abstol=1e-10,
    max_mv_products=500
)
dot(lm2,λ⃗)
dot(lm2,lm2)


dot(λ⃗,λ⃗)
dot(lm2,λ⃗)


lm3 = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=nothing, P̂=IterativeSolvers.Identity())
dot(lm3,λ⃗)
dot(lm3,lm3)
##


Zygote.gradient(ff2,p)[1] # [2.3832434960396154, 0.006884282902900199, 0.27005409524674673, 0.005702895939256883, 0.07660989286044446]

# calculated group indicex gradients vs. tolerance (ff2(p), p=[1.0, 2.0,0.8,0.1,0.1])

# Zygote:
# [2.3832434960396154, 0.006884282902900199,  0.27005409524674673,  0.005702895939256883, 0.07660989286044446 ] # (nev=2,eig_tol=1e-7,k_tol=1e-7)
# [2.3832434946779366, 0.006884285584123125,  0.2700541212712122,   0.005702897553505309, 0.0766099061958902  ] # (nev=2,eig_tol=1e-10,k_tol=1e-12)
# [2.383243496305781, 0.006884289989421895, 0.2700541216897883, 0.005702898375538622, 0.07660990622163796 ]

# FiniteDiff:
# [2.321140491701845, -0.009794340997459905, -0.04992521416458118, -0.00837669524988982, -0.04320451457173746 ] # (rs=1e-3, nev=2,eig_tol=1e-10,k_tol=1e-12)
# [2.321140350902695, -0.009794917577021778, -0.04992696425176745, -0.00837620506288772, -0.04320451103545508 ]

using FiniteDiff
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)
gr_ff2_FD = gradFD2(ff2,p;rs=1e-2)
gr_ff2_FD = gradFD2(ff2,p;rs=1e-3)
gr_ff2_FD = gradFD2(ff2,p;rs=1e-5)
# julia> gr_ff2_FD = gradFD2(ff2,p;rs=1e-2)
#     5-element Vector{Float64}:
#     2.3211544378370474
#     -0.009810432807899616
#     -0.049755946804519446
#     -0.008401555884840484
#     -0.04320452129880081

# julia> gr_ff2_FD = gradFD2(ff2,p;rs=1e-3)
#     5-element Vector{Float64}:
#     2.3211404942358183
#     -0.009794344028146718
#     -0.04992515774993045
#     -0.008376666784437603
#     -0.04320450436190448

function ff23(p)
    mat_vals        =   f_ε_mats(p[1:np_mats]);
    grid            =   Grid(6.,4.,256,128);
    shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
    sm1             =   smooth_ε(shapes,mat_vals,(1,2,3,4),grid);
    # ε               =   copy(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
    # ∂ε_∂ω           =   copy(selectdim(sm1,3,2)); # sm1[:,:,2,:,:] 
    # ∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
    # ε⁻¹             =   sliceinv_3x3(ε);
    # out = sum(ε⁻¹) + sum(∂ε_∂ω)
    out = sum(sm1)
    return out
end
p = [1.0, 2.0,0.8,0.1,0.1]
ff23(p)
gr_ff23_RM = Zygote.gradient(ff23,p)[1]
gr_ff23_FD = gradFD2(ff23,p;rs=1e-5)
# julia> gr_ff2_FD = gradFD2(ff2,p;rs=1e-2)
#     5-element Vector{Float64}:
#     2.3211544378370474
#     -0.009810432807899616
#     -0.049755946804519446
#     -0.008401555884840484
#     -0.04320452129880081

# julia> gr_ff2_FD = gradFD2(ff2,p;rs=1e-3)
#     5-element Vector{Float64}:
#     2.3211404942358183
#     -0.009794344028146718
#     -0.04992515774993045
#     -0.008376666784437603
#     -0.04320450436190448



# Enzyme
∂f_∂p = zero(p)
autodiff(ff23,Duplicated(p,∂f_∂p))

# ReverseDiff
# create_array(
#     ::Type{ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}},
#     ::Nothing, 
#     ::Val{2},
#     ::Val{(27, 4)},
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::Float64,
#     ::Float64,
#     ::Float64,
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing},
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64, ::Float64)
# rrule(::typeof(Code.create_array), A::Type{<:AbstractArray}, T, u::Val{j}, d::Val{dims}, elems...)

# ReverseDiff.@grad_from_chainrules f(x::TrackedReal, y::TrackedArray)
# ReverseDiff.@grad_from_chainrules f(x::TrackedReal, y::Array)
# ReverseDiff.@grad_from_chainrules f(x::Real, y::TrackedArray)

#


##
p = [0.0,0.0,2.0,2.0,-1.1,0.1]
b = Box(SVector{2}(p[1:2]),SVector{2}(p[3:4]))
x = SVector{2,Float64}(p[5:6])
ax = inv(b.p)  # axes: columns are unit vectors
# p_rownorm = map(norm,eachrow(bx.p))
@tullio p_rownorm[i] := b.p[i,j]^2 |> sqrt
@tullio n0[i,j] := b.p[i,j] / p_rownorm[i] # normalize
SVector{2,Float64}( @tullio p_rownorm[i] := b.p[i,j]^2 |> sqrt )
n1 = SMatrix{2,2,Float64}( n0 )
copysign.(one(eltype(d)),d)
# d= Array(b.p * (x - b.c))
d = b.p * (x - b.c)
cosθ = SVector{2,Float64}(diag(n0*ax))
# n = n0 .* SMatrix{1,2, Float64}([1.0 -1.0]) # ignore_derivatives( copysign.(one(eltype(d)),d) )
d_signs = copysign.(one(eltype(d)),d)
n = n1 .* d_signs 


##

function ff3(p)
    # mat_vals        =   f_ε_mats(p[1:np_mats]);
    grid            =   Grid(6.,4.,256,128);
    # shapes          =   geom1(p[(np_mats+1):(np_mats+4)]);
    # sm1             =   smooth_ε(shapes,mat_vals,(1,2,3,4),grid);
    sm1             =   Zygote.forwarddiff(p) do p
        return smooth_ε(geom1(p[2:5]),f_ε_mats(p[1:1]),(1,2,3,4),Grid(6.,4.,256,128))
    end
    ε               =   copy(selectdim(sm1,3,1)); # sm1[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(sm1,3,2)); # sm1[:,:,2,:,:] 
    # ∂²ε_∂ω²         =   copy(selectdim(sm1,3,3)); # sm1[:,:,3,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs   = solve_k(ω,ε⁻¹,grid,DFTK_LOBPCG();nev,eig_tol,k_tol)
    # ngs = map((k,ev)->group_index(k,ev,first(p),ε⁻¹,∂ε_∂ω,grid),zip(kmags,evecs))
    ng1 = group_index(first(kmags),first(evecs),first(p),ε⁻¹,∂ε_∂ω,grid)
    return ng1
end
p = [1.0, 2.0,0.8,0.1,0.1]
ff3(p)
Zygote.gradient(ff3,p)

using FiniteDiff
FiniteDiff.finite_difference_gradient(
    ff2,
    p,
    Val{:central},
    Float64,
    Val{false};
    epsilon_factor=1e-3
)

isapprox(evecs_mpb[1],evecs_df[1],atol=1e-3)
isapprox(evecs_is[1],evecs_df[1],atol=1e-10)

isapprox(evecs_mpb[2],evecs_df[2],atol=1e-3)
isapprox(evecs_is[2],evecs_df[2],atol=1e-10)

abs.( kmags_df .- kmags_mpb )
abs.( kmags_is .- kmags_mpb )
abs.( kmags_is .- kmags_df )
abs.( kmags_is .- kmags_df )

evecsc_all  =   map((k,ev)->canonicalize_phase(ev,k,ε⁻¹,∂ε_∂ω,grid),kmags_all,evecs_all)
Es_all2     =   map((k,ev)->E⃗(k,ev,ε⁻¹,∂ε_∂ω,grid;normalized=false,canonicalize=false),kmags_all,evecsc_all)
Evmms_all2  =   map(val_magmax,Es_all2)

using CairoMakie
using Printf
function plot_fields_grid(fields_matrix,gr::Grid,field_fn=F->real(F[ax_magmax(F),eachindex(gr)]); cmap=:diverging_bkr_55_10_c35_n256,
    zero_based=true,zero_centered=true,grid_xlabels=nothing,grid_ylabels=nothing,label_base=["x","y"],label="E",
    xlim=nothing,ylim=nothing,xlabel="x (μm)",ylabel="x (μm)",fig_pixels=(600, 1200))
    fig = Figure(resolution = fig_pixels)
    nrows,ncols = size(fields_matrix)
    axes = [Axis(fig[row, col], aspect=DataAspect()) for row in 1:nrows, col in 1:ncols]
    hidedecorations!.(axes, grid = false, label = false)
    X,Y = x(gr), y(gr) #xvals(gr), yvals(gr)
    # hms = [undef,(nrows,ncols)]
    for row in 1:nrows, col in 1:ncols
        F = fields_matrix[row,col]
        idx_mm = idx_magmax(F)
        val_mm = F[idx_mm]
        ax_mm = idx_mm[1] # = ax_magmax(F)
        Z = real(F[ax_mm,eachindex(gr)])
        
        # Z = field_fn(fields_matrix[row,col]) #real(F[ax_magmax(F),...])
        Zmin,Zmax = extrema(Z)
        # Zextrm_mag = abs.(Zextrm)
        Zminsign, Zmaxsign = sign(Zmin), sign(Zmax)
        if zero_centered    # diverging colormap, evenly distributed around zero
            crng = (-1,1) .* (max(abs(Zmin),abs(Zmax)),) 
        elseif zero_based && !( Zminsign<0.0 && Zmaxsign>0.0 ) # non-diverging colormap, extremal values on the same side of zero
            crng  =   Zminsign≥0.0 && Zmaxsign≥0.0  ?  (0.0,Zmax)   :   (Zmin,0.0)
        else    # zero_centered=false && ( zero_based=false || Zmin<0.0<Zmax  )  # nothing clever to do
            crng  =   (Zmin,Zmax)
        end
        # hms[row,col] = 
        heatmap!(axes[row, col], X, Y, Z,
            colormap=cmap,
            # label=labels[1],
            colorrange=crng,
        )

        # title_str = "real(" * ["E⃗ₓ","E⃗y","E⃗z"][ax_mm] * "), idx_magmax(E)=$idx_mm, val_magmax(E)=$val_mm"
        title_str = "idx_mm(E)=($(idx_mm[1]),$(idx_mm[2]),$(idx_mm[3]))\n" * "val_mm = $(@sprintf("%.2g", real(val_mm))) + $(@sprintf("%.2g", imag(val_mm)))i"
        axes[row,col].title = title_str
        # Label(axes[row, col, Top()], title_str, valign = :bottom, padding = (0, 0, 5, 0))
        # Colorbar(axes[row,col][1,2], heatmaps[row,col],  width=20 )
    end
    if !isnothing(grid_xlabels)
        for ix in 1:ncols
            axes[nrows, ix].xlabel = grid_xlabels[ix]
        end
    end
    if !isnothing(grid_ylabels)
        for iy in 1:nrows
            axes[iy, 1].ylabel = grid_ylabels[iy]
        end
    end
    return fig
end

plot_fields_grid(Es_all,grid)

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

# previously working (for mpb modes) E⃗(evec,k,grid) code from analyze.jl
# mags_mns = [mag_mn(kk,grid) for kk in ks]
# Es = [-1im * ε⁻¹_dot(fft(kx_tc(evecs[omidx,bndidx],mags_mns[omidx,bndidx][2],mags_mns[omidx,bndidx][1]),(2:3)),epsi[omidx]) for omidx=1:length(oms), bndidx=1:num_bands]
# Enorms =  [ EE[argmax(abs2.(EE))] for EE in Es ]
# Es = Es ./ Enorms

function Efield(k,evec::AbstractArray{<:Number},ε⁻¹,∂ε_∂ω,grid::Grid{ND}; normalized=true) where {ND}
    magmn = mag_mn(k,grid) 
    return -1im * ε⁻¹_dot(fft(kx_tc(evec,magmn[2],magmn[1]),(2:3)),ε⁻¹)
    #Enorms =  [ EE[argmax(abs2.(EE))] for EE in Es ]
    #Es = Es ./ Enorms
end

E⃗2(kmags_mpb1[1],evecs_mpb1[1],ε⁻¹,∂ε_∂ω,grid)



magmn = mag_mn(kmags_mpb1[1],grid) 
size(magmn[2])
E1m = -1im * ε⁻¹_dot(fft(kx_tc(evecs_mpb1[1],magmn[2],magmn[1]),(2:3)),ε⁻¹)
kx_tc(evecs_mpb1[1],magmn[2],magmn[1])

magmn2 = mag_mn(kmags_mpb2[1],grid) 
size(magmn2[2])
E2m = -1im * ε⁻¹_dot(fft(kx_tc(evecs_mpb2[1],magmn2[2],magmn2[1]),(2:3)),ε⁻¹)
isapprox(magmn[1],magmn2[1])
isapprox(magmn[2],magmn2[2])

sum(abs2,evecs_kk[1])

function E⃗2(ks,evecs::AbstractVector{TA},ε⁻¹,∂ε_∂ω,grid::Grid{ND}; normalized=true) where {ND,TA<:AbstractArray{Complex{T}} where T<:Real}
    map((k,evec)->E⃗2(k,evec,ε⁻¹,∂ε_∂ω,grid;normalized), zip(ks,evecs))
end


plot_field(E⃗2(kmags_mpb[1],evecs_mpb[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_mpb[1],evecs_mpb[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)

plot_field(abs2.(E⃗(kmags_kk[1],evecs_kk[1],ε⁻¹,∂ε_∂ω,grid)),grid;axind=3)
plot_field(Efield(kmags_kk[1],reshape(copy(evecs_kk[1]),(2,size(grid)...)),ε⁻¹,∂ε_∂ω,grid),grid;axind=2)

plot_field(E⃗(kmags_kk[1],evecs_kk[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_kk[2],evecs_kk[2],ε⁻¹,∂ε_∂ω,grid),grid;axind=2)

plot_field(E⃗(kmags_is[1],evecs_is[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_is[2],evecs_is[2],ε⁻¹,∂ε_∂ω,grid),grid;axind=2)

plot_field(E⃗(kmags_df[1],evecs_df[1],ε⁻¹,∂ε_∂ω,grid, normalized=false),grid;axind=1)
plot_field(E⃗(kmags_df[2],evecs_df[2],ε⁻¹,∂ε_∂ω,grid, normalized=false),grid;axind=2)


E_kk = E⃗
plot_field(E⃗(kmags_kk[1],evecs_kk[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_kk[2],evecs_kk[2],ε⁻¹,∂ε_∂ω,grid),grid;axind=2)

plot_field(E⃗(kmags_is[1],evecs_is[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_is[2],evecs_is[2],ε⁻¹,∂ε_∂ω,grid),grid;axind=2)

plot_field(E⃗(kmags_df[1],evecs_df[1],ε⁻¹,∂ε_∂ω,grid),grid;axind=1)
plot_field(E⃗(kmags_df[2],evecs_df[2],ε⁻¹,∂ε_∂ω,grid),grid;axind=2)


function foo1(x::Vector{T})::Tuple{Vector{T},Vector{Vector{Complex{T}}}} where {T<:Real}
    return 2*x, [x, x.^2, sin.(x)]
end
foo1(rand(4))

E1kk = E⃗(kmags_kk[1],evecs_kk[1],ε⁻¹,∂ε_∂ω,grid)
val_magmax(E1kk)
argmax(abs2.(E1kk))

eigs_itr = LOBPCGIterator(ms.M̂,false,ms.H⃗,ms.P̂)
res_is =  lobpcg!(eigs_itr; log=false,not_zeros=false,maxiter=200,tol=1e-8)
copyto!(ms.H⃗,res_is.X)
copyto!(ms.ω²,res_is.λ)
canonicalize_phase!(ms)
Hin = reshape(copy(ms.H⃗[:,1]),2,size(grid)...)
Ekk1,Ekk2 = copy(E⃗(ms,evecs_kk)) #E⃗(ms.M̂.k⃗[3],Hin,ε⁻¹,∂ε_∂ω,grid,normalized=false)
# Ekk1c,Ekk2c = E⃗(ms,canonicalize_phase!(ms,copy.(evecs_kk))
canonicalize_phase!(ms,evecs_kk)
evecs_kkc = canonicalize_phase(evecs_kk,ms)



E12 = E⃗(ms,2)
E11,E12 = E⃗(evecs_kk,ms)
E11c,E12c = E⃗(evecs_kkc,ms)
imagmax1, imagmax2 = argmax(abs2.(E11)), argmax(abs2.(E12))
E_magmax1 = E11[imagmax1]
E_magmax2 = E12[imagmax2]

imagmax1c, imagmax2c = argmax(abs2.(E11c)), argmax(abs2.(E12c))
E_magmax1c = E11c[argmax(abs2.(E11c))]
E_magmax2c = E12c[argmax(abs2.(E12c))]

Ephase_magmax1 = angle(E_magmax1)
Ephase_magmax2 = angle(E_magmax2)

E_canon = E11 * cis(-Ephase_magmax)
cis(-Ephase_magmax) |> abs2
E_canon_magmax = E_canon[argmax(abs2.(E_canon))]

E_canon2 = E⃗(ms.M̂.k⃗[3],Hin*cis(-Ephase_magmax),ε⁻¹,∂ε_∂ω,grid,normalized=false)
E_canon2_magmax = E_canon2[argmax(abs2.(E_canon2))]


# function canonicalize_phase!(Hin::AbstractMatrix,ms::ModeSolver,eig_idx::Int)
#     ms.H⃗ *= cis(-angle(magmax(E⃗(ms,eig_idx))))
# end

Hin_canon = Hin * 

# 

# kmag1_kk,evec1_kk = solve_k_single(ms,ω;nev=2,eigind=1)
# kmag2_kk,evec2_kk = solve_k_single(ms,ω;nev=2,eigind=2)
kmags_kk,evecs_kk = solve_k(ms,ω;nev=2)

kmags,evecs = copy(kmags_mpb),copy(evecs_mpb);
ng1 = group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
ng_grads = Zygote.gradient((k,ev,om,epsi,deps_dom)->group_index(k,ev,om,epsi,deps_dom,grid),kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω);
E1 = E⃗(kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,grid)
kg1 = k_guess(ω,ε⁻¹)
k1 = kmags[1]
# kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
using OptiMode: _solve_Δω²
ms = ModeSolver(1.9, ε⁻¹, grid; nev=2, maxiter=100, tol=1e-8)
evals_kk,evecs_kk,convinfo_kk = eigsolve(x->ms.M̂*x,copy(ms.H⃗[:,1]),2,:SR;maxiter=100,tol=1e-8,krylovdim=50,verbosity=2)
E1_kk = E⃗(kmags[1],evecs_kk[1],ε⁻¹,∂ε_∂ω,grid)
# evals_kk,evecs_kk = solve_ω²(ms;nev=2)
# Δω²,Δω²_∂ω²∂k = _solve_Δω²(ms,kmags[1],1.0;nev=2,eigind=1)

hcat(vec.(evecs_mpb)...)

copyto!(ms.H⃗,hcat(vec.(evecs_mpb)...))
x0 = copy(ms.H⃗[:,1]) # isapprox(x0,evecs_kk[1])
x1 = ms.M̂ * x0

dot(x0,x1)
dot(x0,x0)
dot(x1,x1)
dot(evecs_kk[1],evecs_kk[1])



isapprox(E1,E1_kk,rtol=1e-2)
_solve_Δω²(ms::ModeSolver{ND,T},k::TK,ωₜ::T;nev=1,eigind=1)

##
# kz1,ev1 = solve_k(ms,ω,ε⁻¹;nev=2,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
ωₜ = 1.0
eigind = 1
# ω²,ev1 = solve_ω²(ms,kmags[1];nev=2,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
Δω² = ω²[eigind] - ωₜ^2


∂ω²∂k_mpb = 2 * HMₖH(evecs_mpb[1],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
∂ω²∂k_kk  = 2 * HMₖH(evecs_kk[1], ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)


2 * HMₖH(vec(evecs[1]),ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
2 * HMₖH((ev1[:,eigind]),ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # ./sqrt(sum(abs2,ev1[:,eigind]))
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


plot_field(E1_kk,grid;axind=1)
plot_field(E1 .- E1_kk,grid;axind=1)

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