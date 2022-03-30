"""
Tests for calculations involving Mode Fields, both in plane-wave and real-space bases
"""

using LinearAlgebra, StaticArrays, FFTW, GeometryPrimitives, OptiMode, Test
using ChainRules, Zygote, FiniteDifferences, ForwardDiff, FiniteDiff
# using CairoMakie

function geom11p(p::AbstractVector{T}) where {T<:Real}  # fully-etched ridge_wg, Polygon core
    wₜₒₚ        =   p[1]
    t_core      =   p[2]
    θ           =   p[3]
    edge_gap    =   0.5
    mat_core    =   1
    mat_subs    =   2
    Δx          =   6.0
    Δy          =   4.0
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
	verts = SMatrix{4,2,T}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(verts,mat_core)
    ax = SMatrix{2,2,T}( [      1.     0.   ;   0.     1.      ] )
	b_subs = GeometryPrimitives.Box( SVector{2}([0. , c_subs_y]), SVector{2}([Δx - edge_gap, t_subs ]),	ax,	mat_subs, )
	return (core,b_subs)
end

function geom11b(p::AbstractVector{T}) where {T<:Real}  # fully-etched ridge_wg, Box core
    wₜₒₚ        =   p[1]
    t_core      =   p[2]
    edge_gap    =   0.5
    mat_core    =   1
    mat_subs    =   2
    Δx          =   6.0
    Δy          =   4.0
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.	
    ax = SMatrix{2,2,T}( [      1.     0.   ;   0.     1.      ] )
	b_core = GeometryPrimitives.Box( SVector{2,T}([0. , 0.]), SVector{2,T}([wₜₒₚ, t_core ]),	ax,	mat_subs, )
    b_subs = GeometryPrimitives.Box( SVector{2,T}([0. , c_subs_y]), SVector{2,T}([Δx - edge_gap, t_subs ]),	ax,	mat_core, )
	return (b_core,b_subs)
end

function geom12p(p::AbstractVector{T}) where {T<:Real}  # partially-etched ridge_wg, Polygon core
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
	t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_unetch/2.
	verts = SMatrix{4,2,T}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(verts,mat_core)
    ax = SMatrix{2,2,T}( [      1.     0.   ;   0.     1.      ] )
	# b_unetch = GeometryPrimitives.Box( [0. , c_unetch_y], [Δx - edge_gap, t_unetch ],	ax,	mat_core )
	b_slab = GeometryPrimitives.Box( SVector{2}([0. , c_slab_y]), SVector{2}([Δx - edge_gap, t_slab ]),	ax, mat_slab, )
	b_subs = GeometryPrimitives.Box( SVector{2}([0. , c_subs_y]), SVector{2}([Δx - edge_gap, t_subs ]),	ax,	mat_subs, )
	return (core,b_slab,b_subs)
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

## choose a geometry function and initialize the corresponding
## material models
geom_fn             =   geom1
mats                =   [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
mat_vars            =   (:ω,)
np_mats             =   length(mat_vars)
f_ε_mats, f_ε_mats! =   _f_ε_mats(mats,mat_vars) # # f_ε_mats, f_ε_mats! = _f_ε_mats(vcat(materials(sh1),Vacuum),(:ω,))
mat_vals            =   f_ε_mats(p[1:np_mats]);

## Set geometry parameters `p`, grid & solver settings
ω               =   1.1 
p               =   [ω, 2.0,0.8,0.1,0.1];
nev             =   2
eig_tol         =   1e-9
k_tol           =   1e-9
Dx              =   6.0
Dy              =   4.0
Nx              =   256
Ny              =   256
grid            =   Grid(Dx,Dy,Nx,Ny)
fftax           =   _fftaxes(grid)      # spatial grid axes of field arrays, 2:3 (2:4) for 2D (3D) using current field data format
Ngrid           =   length(grid)        # total number of grid points, Nx*Ny (Nx*Ny*Nz) in 2D (3D)
minds           =   (1,2,3,4)           # "material indices" for `shapes=geom1(p)`, the material of shape=geom1(p)[idx] is `mat[minds[idx]]`
                                        # TODO: automate determination of material indices, this is error-prone

## Calculate dielectric data using these parameters
ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); # TODO: automate unpacking of dielectric data into (ε, ∂ε_∂ω, ∂²ε_∂ω²)
ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);

# legacy dispersion tensor data....get rid of this asap
nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
nng⁻¹             =   sliceinv_3x3(nng);
ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω

## Calculate the first two eigeinmodes to generate fields for tests
kmags, evecs    =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)

# for now, focus on first eigenmode 
k = kmags[1]
ev = first(evecs)
mag, mn = mag_mn(k,g⃗(grid))
ev_grid = reshape(ev,(2,Nx,Ny))
nng = (ω/2) * ∂ε_∂ω + ε
norm_fact = inv(sqrt(δV(grid) * Ngrid) * ω)

# D       =   1im * fft( kx_tc( ev_grid,mn,mag), _fftaxes(grid) )
# E       =   ε⁻¹_dot( D, ε⁻¹)
# H       =   (1im * ω) * fft( tc(ev_grid,mn), (2:3) ) 

D           =   fft( kx_tc( ev_grid, mn, mag ), fftax )
E           =   ε⁻¹_dot( D, ε⁻¹)
H           =   ω * fft( tc(ev_grid,mn), fftax ) 
P           =   2*real(_sum_cross_z(conj(E),H))                 # `P`: integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
W_nondisp   =   real(_expect(ε,E)) + (N(grid)* (ω^2))           # `W_nondisp`: non-dispersive energy density per unit length ∫dA ( D⃗⋅E⃗ + |H⃗|² ) without magnetism μ = μ₍
W           =   real(_expect(ε + ω*∂ε_∂ω,E)) + (N(grid)* (ω^2)) # `W`: energy density with contribution from bulk dispersion = ∫dA ( E⃗⋅(ε + ω*∂ε_∂ω)⋅E⃗  + |H⃗|² ) without magnetism (μ = μ₀)

## check eigenvector normalization (should be 1.0)
@test real(sum(abs2,ev))    ≈   1.0

## check real-space field normalization (currently ∫D⃗⋅E⃗ = ∫|H⃗|² = ω^2 * Ngrid, Ngrid factor because I don't normalize after FFT
@test real(_expect(ε,E))    ≈   Ngrid * ω^2
@test sum(abs2,H)           ≈   Ngrid * ω^2
@test _expect(ε,E)          ≈   dot(E,_dot( ε ,E))

# compare non-dispersive group velocity calc'd with real-space E & H fields 
# against the same quantity calc'd directly from plane-wave basis eigenvectors 
ng_nondisp_rs   =   ( ( real(_expect(ε,E)) + sum(abs2,H) ) / ( 2*real(_sum_cross_z(conj(E),H)) ) )
ng_nondisp_ev   =   ( ω / HMₖH(ev,ε⁻¹,mag,mn) )
@test ng_nondisp_rs     ≈ ng_nondisp_ev
@show ng_nondisp_err    = ( ( real(_expect(ε,E)) + sum(abs2,H) ) / ( 2*real(_sum_cross_z(conj(E),H)) ) ) - ( ω / HMₖH(ev,ε⁻¹,mag,mn) )
@show ng_nondisp_relerr = abs(ng_nondisp_err) / ng_nondisp_ev

# compare group index (accounting for material dispersion) calc'd with real-space E & H fields 
# against the same quantity calc'd directly from plane-wave basis eigenvectors 
ng_rs   =   ( real(_expect(ε + ω*∂ε_∂ω,E)) + (N(grid)* (ω^2)) ) / ( 2*real(_sum_cross_z(conj(E),H)) ) 
ng_ev   =   (ω + HMH(vec(ev), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(ev),ε⁻¹,mag,mn)
@test ng_rs     ≈ ng_ev
@test ng_rs     ≈ ng_nondisp_rs + real(_expect(ω*∂ε_∂ω,E))  / ( 2*real(_sum_cross_z(conj(E),H)) ) 
@test ng_ev     ≈ ng_nondisp_ev + (HMH(vec(ev), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(ev),ε⁻¹,mag,mn)
@show ng_err    = ng_rs - ng_ev
@show ng_relerr = abs(ng_err) / ng_ev

# compare the group index calculated above with values of d|k|/dω calculated 
# via AD and finite differencing. they should all be equal
ng_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end

ng_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end |> first

# ng_RM = 2.3644757860483643 # temporary fix for syntax error, remove soon 
@test ng_RM ≈ ng_FD rtol = 1e-7
@test ng_rs ≈ ng_FD   # manual group index calculation matches d|k|/dω calculated via finite differencing
@test ng_rs ≈ ng_RM rtol = 1e-7 # manual group index calculation matches d|k|/dω calculated via AD

@show ng_RM_vs_FD_err    = ng_RM - ng_FD
@show ng_RM_vs_FD_relerr = abs(ng_RM_vs_FD_err) / ng_FD
@show ng_direct_vs_FD_err    = ng_FD - ng_ev
@show ng_direct_vs_FD_relerr = abs(ng_direct_vs_FD_err) / ng_ev
@show ng_direct_vs_RM_err    = ng_RM - ng_ev
@show ng_direct_vs_RM_relerr = abs(ng_direct_vs_RM_err) / ng_ev

gvd_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end

gvd_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end |> first

@test gvd_RM ≈ gvd_FD rtol = 1e-5
@show gvd_RM_vs_FD_err    = gvd_RM - gvd_FD
@show gvd_RM_vs_FD_relerr = abs(gvd_RM_vs_FD_err) / gvd_FD

# @show gvd_direct_vs_FD_err    = gvd_RM - gvd_ev
# @show gvd_direct_vs_FD_relerr = abs(gvd_direct_vs_FD_err) / gvd_FD
# @show gvd_direct_vs_RM_err    = gvd_RM - gvd_ev
# @show gvd_direct_vs_RM_relerr = abs(gvd_direct_vs_RM_err) / gvd_FD

# gvd_RM = Zygote.gradient(ω) do ω
#     ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
#     ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
#     ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
#     ε⁻¹             =   sliceinv_3x3(ε);
#     kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
#     ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
#     return ng
# end |> first
#  solve_k_pullback:
# k̄ (bar): -0.929874961497507
#  solve_k pullback for eigind=1:
#          ω² (target): 1.2100000000000002
#          ∂ω²∂k (recalc'd): 0.9638126793744675
# k̄ₕ_old = -((mag_m_n_pb((māg, kx̄_m⃗ .* ms.M̂.mag, kx̄_n⃗ .* ms.M̂.mag)))[1]) = 0.0279527504176263
# k̄ₕ = -(∇ₖmag_m_n(māg, kx̄_m⃗ .* ms.M̂.mag, kx̄_n⃗ .* ms.M̂.mag, ms.M̂.mag, m⃗, n⃗; dk̂ = SVector(0.0, 0.0, 1.0))) = 0.027952750417626303
# ω_bar += ((2ω) * (k̄ + k̄ₕ)) / ∂ω²∂k = -2.058728741422596
#  solve_k_pullback:
# k̄ (bar): 0.0
#  solve_k pullback for eigind=2:
#          ω² (target): 1.2100000000000002
#          ∂ω²∂k (recalc'd): 0.9564010245562874
# ω_bar += ((2ω) * (k̄ + k̄ₕ)) / ∂ω²∂k = -2.058728741422596
#gvd_RM = 0.14033455221012248

###############################################################################################################################
###############################################################################################################################

using Tullio
using OptiMode: ∇ₖmag_m_n, ∇ₖmag_mn

evg = ev_grid

function ng_AD_steps()
    ω                               =   1.1 
    p_geom                          =   [2.0,0.8,0.1,0.1];
    nev                             =   2
    eig_tol                         =   1e-9
    k_tol                           =   1e-9
    Dx                              =   6.0
    Dy                              =   4.0
    Nx                              =   256
    Ny                              =   256
    grid                            =   Grid(Dx,Dy,Nx,Ny)
    fftax                           =   _fftaxes(grid)      
    Ngrid                           =   length(grid)        
    minds                           =   (1,2,3,4) 

    (ε, ∂ε_∂ω, ∂²ε_∂ω²), ε_data_pb  =   Zygote.pullback(ω,p_geom) do  ω,p_geom
        ε_data = smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),minds,grid)
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
        ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:]
        return ε, ∂ε_∂ω, ∂²ε_∂ω²
    end

    ε⁻¹, ε⁻¹_pb                     =   Zygote.pullback(sliceinv_3x3,ε);
    (kmags,evecs), solve_k_pb       =   Zygote.pullback(ω,ε⁻¹) do  ω,ε⁻¹
        kmags,evecs = solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        return kmags,evecs
    end
    
    # ng, ng_pb                       =   Zygote.pullback(kmags,evecs,ω,ε⁻¹,∂ε_∂ω) do kmags,evecs,ω,ε⁻¹,∂ε_∂ω
    #     group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    # end

    ### ng = (ω + HMH(vec(evec), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(evec),ε⁻¹,mag,mn) ###
    (mag,mn), mag_mn_pb             = Zygote.pullback(kmags) do kmags
        mag,mn = mag_mn(kmags[1],grid)
        return mag, mn 
    end
	
    EdepsiE, EdepsiE_pb             = Zygote.pullback(evecs,ε⁻¹,∂ε_∂ω,mag,mn) do evecs,ε⁻¹,∂ε_∂ω,mag,mn
        HMH(vec(evecs[1]), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)
    end

    HMkH, HMkH_pb                   = Zygote.pullback(evecs,ε⁻¹,mag,mn) do evecs,ε⁻¹,mag,mn
        HMₖH(vec(evecs[1]),ε⁻¹,mag,mn)
    end

    ng, ng_pb                       = Zygote.pullback(ω,EdepsiE,HMkH) do ω,EdepsiE,HMkH
        return (ω + EdepsiE/2) / HMkH
    end


    @show ng
    
    # ∂ng_∂kmags, ∂ng_∂evecs, ∂ng_∂ω_1, ∂ng_∂ε⁻¹_1, ∂ng_∂∂ε_∂ω    =   ng_pb(1.0)
    # ∂ng_∂ω_2, ∂ng_∂ε⁻¹_2                                        =   solve_k_pb((∂ng_∂kmags,∂ng_∂evecs))
    # ∂ng_∂ε                                                      =   ε⁻¹_pb( ∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 )[1]

    ∂ng_∂ω_1, ∂ng_∂EdepsiE, ∂ng_∂HMkH                           =   ng_pb(1.0)
    # ∂ng_∂evecs_1,∂ng_∂ε⁻¹_1,∂ng_∂mag_1,∂ng_∂mn_1                =   HMkH_pb(∂ng_∂HMkH)
    # ∂ng_∂evecs_2,∂ng_∂ε⁻¹_2,∂ng_∂∂ε_∂ω,∂ng_∂mag_2,∂ng_∂mn_2     =   EdepsiE_pb(∂ng_∂EdepsiE)
    ∂ng_∂evecs_2,∂ng_∂ε⁻¹_2,∂ng_∂mag_2,∂ng_∂mn_2                =   HMkH_pb(∂ng_∂HMkH)
    ∂ng_∂evecs_1,∂ng_∂ε⁻¹_1,∂ng_∂∂ε_∂ω,∂ng_∂mag_1,∂ng_∂mn_1     =   EdepsiE_pb(∂ng_∂EdepsiE)
    ∂ng_∂kmags                                                  =   mag_mn_pb(( ∂ng_∂mag_1 + ∂ng_∂mag_2 , ∂ng_∂mn_1 + ∂ng_∂mn_2 ))[1]
    ∂ng_∂evecs                                                  =   [∂ng_∂evecs_1[1] + ∂ng_∂evecs_2[1], zero(evecs[1]) ] # ZeroTangent(evecs[1]) ]
    ∂ng_∂ω_2, ∂ng_∂ε⁻¹_3                                        =   solve_k_pb((∂ng_∂kmags, ∂ng_∂evecs ))
    ∂ng_∂ε                                                      =   ε⁻¹_pb( ∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 + ∂ng_∂ε⁻¹_3 )[1]
    ∂ng_∂ω_3, ∂ng_∂p_geom                                       =   ε_data_pb((∂ng_∂ε,∂ng_∂∂ε_∂ω,zero(ε)))

    @show ∂ng_∂ω                                                =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3
    @show gvd_RM
    @show gvd_FD
    
    @show ∂ng_∂p_geom

    ∂ng_∂mag_1_AD = copy(∂ng_∂mag_1)
    ∂ng_∂mn_1_AD = copy(∂ng_∂mn_1)    
    ∂ng_∂mag_2_AD = copy(∂ng_∂mag_2)
    ∂ng_∂mn_2_AD = copy(∂ng_∂mn_2)

    println("#######################################################################################")
    println("#######################################################################################")
    println("")
    println("")

    println("intermediate AD gradient values")

    println("")
    println("")

    @show ∂ng_∂EdepsiE
    @show ∂ng_∂HMkH
    @show ∂ng_∂ω_1

    @show val_magmax(∂ng_∂∂ε_∂ω)
    @show val_magmax(∂ng_∂ε⁻¹_1)
    ∂ng_∂evg_1 = first(∂ng_∂evecs_1)
    @show val_magmax(∂ng_∂evg_1)
    @show val_magmax(∂ng_∂mag_1)
    @show val_magmax(∂ng_∂mn_1)
    @show ∂ng_∂k_1 = mag_mn_pb(( ∂ng_∂mag_1, ∂ng_∂mn_1))[1][1]

    @show val_magmax(∂ng_∂ε⁻¹_2)
    ∂ng_∂evg_2 = first(∂ng_∂evecs_2)
    @show val_magmax(∂ng_∂evg_2)
    @show val_magmax(∂ng_∂mag_2)
    @show val_magmax(∂ng_∂mn_2)
    @show ∂ng_∂k_2 = mag_mn_pb(( ∂ng_∂mag_2, ∂ng_∂mn_2))[1][1]

    @show ∂ng_∂k = ∂ng_∂k_1 + ∂ng_∂k_2

    @show ∂ng_∂ω_2
    @show val_magmax(∂ng_∂ε⁻¹_3)
    @show val_magmax(∂ng_∂ε)

    @show ∂ng_∂ω_1
    @show ∂ng_∂ω_2
    @show ∂ng_∂ω_3

    println("#######################################################################################")
    println("#######################################################################################")
    println("")
    println("")

    println("manual calculation & comparison")

    println("")
    println("")

    ∂ng_∂ω_1, ∂ng_∂EdepsiE, ∂ng_∂HMkH       =       inv(HMkH),  inv(HMkH)/2,  -(ω + EdepsiE/2) * inv(HMkH^2)       
    
    @show ∂ng_∂EdepsiE
    @show ∂ng_∂HMkH
    @show ∂ng_∂ω_1
	
    ev  =   evecs[1]
    evg =   reshape(ev,(2,size(grid)...))
    T = Float64
    Ninv                =   inv(1.0 * length(grid))
    dk̂                  =   SVector(0.,0.,1.)
    mag2,m⃗,n⃗           =    mag_m_n(k,grid)
    one_mone = [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(ev_grid,mn), fftax )
    HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
    inv_HMkH            =   inv(HMkH)
    
    deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
    epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
    Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
    ng                  =   (ω + EdepsiE/2) * inv_HMkH
    
    ∂ng_∂ω_1            =   inv_HMkH
    ∂ng_∂EdepsiE        =   inv_HMkH/2
    ∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2
    
    ∂ng_∂∂ε_∂ω          =   _outer(E,E) * Ninv * ∂ng_∂EdepsiE
    ∂ng_∂ε⁻¹_1          =   herm(_outer(deps_E,D)) * Ninv * 2 * ∂ng_∂EdepsiE
    ∂ng_∂evg_1          =   kx_Fi_epsi_deps_E * 2 * ∂ng_∂EdepsiE
    ∂ng_∂kx_1           =  real(_outer(Fi_epsi_deps_E, evg)) * 2 * ∂ng_∂EdepsiE
    @tullio ∂ng_∂mag_1[ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j]  nograd=one_mone
    @tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] * one_mone[j]  nograd=one_mone
    ∂ng_∂k_1            =   ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)
    
    ### ∇HMₖH ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    # om̄₁₂ = dot(H,H̄) / ω
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn_2[i,j,ix,iy] := mag[ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[i,j,ix,iy]   +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]  
    @tullio ∂ng_∂mag_2[ix,iy] := mn[a,b,ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[a,b,ix,iy]
    
    @test real(∂ng_∂mn_2_AD[:,1,:,:]) ≈ real(∂ng_∂mn_2[:,1,:,:])
    @test real(∂ng_∂mn_2_AD[:,2,:,:]) ≈ real(∂ng_∂mn_2[:,2,:,:])
    ∂ng_∂k_2 = ∇ₖmag_mn(real(∂ng_∂mag_2),real(∂ng_∂mn_2),mag,mn)
    ∂ng_∂evg_2 = ( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag) + ω*ct(𝓕⁻¹_H̄,mn) ) 
    ∂ng_∂ε⁻¹_2 = real( herm( _outer(Ē,D) ) ) 
    ### end ∇HMₖH ###

    ### ∇solve_k ###
    k̄ = ∂ng_∂k_1 + ∂ng_∂k_2

    ∂ng_∂evg = vec(∂ng_∂evg_1) + vec(∂ng_∂evg_2)
    @show val_magmax(∂ng_∂evg_1)
    @show val_magmax(∂ng_∂evg_2)
    @show val_magmax(∂ng_∂evg)

    M̂ = HelmholtzMap(k,ε⁻¹,grid)
    P̂	= HelmholtzPreconditioner(M̂)
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		 # x⃗
        0.0, 							# ᾱ
        ∂ng_∂evg;					    # x̄
        # λ⃗₀=λ⃗₀,
        P̂	= P̂,
    )

    @show val_magmax(λ⃗)
    @show dot(ev,λ⃗)

    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    
    @show val_magmax(λd)
    
    ∂ng_∂ε⁻¹_31 = ε⁻¹_bar(vec(D), vec( λd ) , size(grid)...) * Ninv

    @show val_magmax(∂ng_∂ε⁻¹_31)

    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )

    @show val_magmax(λẽ)
    @show val_magmax(ẽ)

    λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
    ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
    m̄_kx = real.( λẽ_sv .* conj.(view(evg,2,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,2,axes(grid)...)) )	#NB: m̄_kx and n̄_kx would actually
    n̄_kx =  -real.( λẽ_sv .* conj.(view(evg,1,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,1,axes(grid)...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
    māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
    @show k̄ₕ		= -∇ₖmag_m_n(
                māg_kx, 		# māg total
                m̄_kx.*mag, 	# m̄  total
                n̄_kx.*mag,	  	# n̄  total
                mag, m⃗, n⃗; 
                dk̂=SVector(0.,0.,1.), # dk⃗ direction
            )
    λd2 = fft( kx_tc( ( (k̄ + k̄ₕ ) / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax ) * Ninv  # 2 * HMkH = ∂ω²∂k
    
	@show val_magmax(λd2)
    
    ∂ng_∂ε⁻¹_32 = ε⁻¹_bar(vec(D), vec( λd2 ) , size(grid)...)

    @show val_magmax(∂ng_∂ε⁻¹_32)

    ∂ng_∂ε⁻¹_3 = ∂ng_∂ε⁻¹_31 + ∂ng_∂ε⁻¹_32

    @show val_magmax(∂ng_∂ε⁻¹_3)

    @show ∂ng_∂ω_2 =  ω * (k̄ + k̄ₕ ) / HMkH 
    ### end ∇solve_k ###
    ∂ng_∂ε = _dot( -ε⁻¹, (∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 + ∂ng_∂ε⁻¹_3), ε⁻¹ )
    ∂ng_∂ω_3, ∂ng_∂p_geom                                       =   ε_data_pb((∂ng_∂ε,∂ng_∂∂ε_∂ω,zero(ε)))
    @show ∂ng_∂ω                                                =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3
    @show ∂ng_∂p_geom

    println("")
    println("")

    @show ∂ng_∂EdepsiE
    @show ∂ng_∂HMkH
    @show ∂ng_∂ω_1

    @show val_magmax(∂ng_∂∂ε_∂ω)
    @show val_magmax(∂ng_∂ε⁻¹_1)
    @show val_magmax(∂ng_∂evg_1)
    @show val_magmax(∂ng_∂mag_1)
    @show val_magmax(∂ng_∂mn_1)
    @show ∂ng_∂k_1  # = mag_mn_pb(( ∂ng_∂mag_1, ∂ng_∂mn_1))[1][1]

    @show val_magmax(∂ng_∂ε⁻¹_2)
    @show val_magmax(∂ng_∂evg_2)
    @show val_magmax(∂ng_∂mag_2)
    @show val_magmax(∂ng_∂mn_2)
    @show ∂ng_∂k_2

    @show ∂ng_∂k = ∂ng_∂k_1 + ∂ng_∂k_2

    @show ∂ng_∂ω_2
    @show val_magmax(∂ng_∂ε⁻¹_3)
    @show val_magmax(∂ng_∂ε)

    @show ∂ng_∂ω_1
    @show ∂ng_∂ω_2
    @show ∂ng_∂ω_3

    return nothing
    # return ∂ng_∂mag_1_AD, ∂ng_∂mn_1_AD, ∂ng_∂mag_2_AD, ∂ng_∂mn_2_AD, ∂ng_∂mag_1, ∂ng_∂mn_1, ∂ng_∂mag_2, ∂ng_∂mn_2
end

ng_AD_steps()

# ∂ng_∂mag_1_AD, ∂ng_∂mn_1_AD, ∂ng_∂mag_2_AD, ∂ng_∂mn_2_AD, ∂ng_∂mag_1, ∂ng_∂mn_1, ∂ng_∂mag_2, ∂ng_∂mn_2 = ng_AD_steps()

###############################################################################################################################
###############################################################################################################################

##

@show val_magmax(real(∂ng_∂mag_1_AD)) # val_magmax(∂ng_∂mag_1_AD)
@show val_magmax(real(∂ng_∂mn_1_AD)) # val_magmax(∂ng_∂mn_1_AD)
@show val_magmax(real(∂ng_∂mag_2_AD)) # val_magmax(∂ng_∂mag_2_AD)
@show val_magmax(real(∂ng_∂mn_2_AD)) # val_magmax(∂ng_∂mn_2_AD)
@show ∂ng_∂k_1_AD = ∇ₖmag_mn(real(∂ng_∂mag_1_AD),real(∂ng_∂mn_1_AD),mag,mn)
@show ∂ng_∂k_2_AD = ∇ₖmag_mn(real(∂ng_∂mag_2_AD),real(∂ng_∂mn_2_AD),mag,mn)

@show val_magmax(∂ng_∂mag_1)
@show val_magmax(∂ng_∂mn_1)
@show val_magmax(∂ng_∂mag_2)
@show val_magmax(∂ng_∂mn_2)
@show ∂ng_∂k_1 = ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)
@show ∂ng_∂k_2 = ∇ₖmag_mn(∂ng_∂mag_2,∂ng_∂mn_2,mag,mn)

# real(∂ng_∂mag_1_AD)[1:10,1:10]
# real(∂ng_∂mag_1)[1:10,1:10]
# real(∂ng_∂mag_1_AD-∂ng_∂mag_1)[1:10,1:10]
@test real(∂ng_∂mag_1_AD) ≈ ∂ng_∂mag_1

# real(∂ng_∂mn_1_AD)[:,1,1:4,1:4]
# real(∂ng_∂mn_1)[:,1,1:4,1:4]
# real(∂ng_∂mn_1_AD-∂ng_∂mn_1)[:,1,1:4,1:4]
@test real(∂ng_∂mn_1_AD[:,1,:,:]) ≈ ∂ng_∂mn_1[:,1,:,:]

# real(∂ng_∂mn_1_AD)[:,2,1:4,1:4]
# real(∂ng_∂mn_1)[:,2,1:4,1:4]
# real(∂ng_∂mn_1_AD - ∂ng_∂mn_1)[:,2,1:4,1:4]
@test real(∂ng_∂mn_1_AD[:,2,:,:]) ≈ ∂ng_∂mn_1[:,2,:,:]

# real(∂ng_∂mag_2_AD)[1:10,1:10]
# real(∂ng_∂mag_2)[1:10,1:10]
# real(∂ng_∂mag_2_AD-∂ng_∂mag_2)[1:10,1:10]
@test real(∂ng_∂mag_2_AD) ≈ ∂ng_∂mag_2


real(∂ng_∂mn_2_AD)[:,1,1:4,1:4]
real(∂ng_∂mn_2)[:,1,1:4,1:4]
real(∂ng_∂mn_2_AD-∂ng_∂mn_2)[:,1,1:4,1:4]
# wrong

real(∂ng_∂mn_2_AD)[:,2,1:4,1:4]
real(∂ng_∂mn_2)[:,2,1:4,1:4]
real(∂ng_∂mn_2_AD + ∂ng_∂mn_2)[:,2,1:4,1:4]
## wrong
ev  =   evecs[1]
evg =   reshape(ev,(2,size(grid)...))
T = Float64
Ninv                =   inv(1.0 * length(grid))
dk̂                  =   SVector(0.,0.,1.)
mag2,m⃗,n⃗           =    mag_m_n(k,grid)
one_mone = [1.0, -1.0]
D                   =   fft( kx_tc(evg,mn,mag), fftax )
E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
H                   =   ω * fft( tc(ev_grid,mn), fftax )
HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
inv_HMkH            =   inv(HMkH)

deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
ng                  =   (ω + EdepsiE/2) * inv_HMkH

∂ng_∂ω_1            =   inv_HMkH
∂ng_∂EdepsiE        =   inv_HMkH/2
∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2

∂ng_∂∂ε_∂ω          =   _outer(E,E) * Ninv * ∂ng_∂EdepsiE
∂ng_∂ε⁻¹_1          =   herm(_outer(deps_E,D)) * Ninv * 2 * ∂ng_∂EdepsiE
∂ng_∂evg_1          =   kx_Fi_epsi_deps_E * 2 * ∂ng_∂EdepsiE
∂ng_∂kx_1           =  real(_outer(Fi_epsi_deps_E, evg)) * 2 * ∂ng_∂EdepsiE
@tullio ∂ng_∂mag_1[ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j]  nograd=one_mone
@tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] * one_mone[j]  nograd=one_mone
∂ng_∂k_1            =   ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)

### ∇HMₖH ###
H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
# om̄₁₂ = dot(H,H̄) / ω
𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
@tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] * one_mone[j] nograd=one_mone
@tullio ∂ng_∂mn_2[i,j,ix,iy] := mag[ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[i,j,ix,iy]   +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]  
@tullio ∂ng_∂mag_2[ix,iy] := mn[a,b,ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[a,b,ix,iy]

@test real(∂ng_∂mn_2_AD[:,1,:,:]) ≈ real(∂ng_∂mn_2[:,1,:,:])
@test real(∂ng_∂mn_2_AD[:,2,:,:]) ≈ real(∂ng_∂mn_2[:,2,:,:])
∂ng_∂k_2 = ∇ₖmag_mn(real(∂ng_∂mag_2),real(∂ng_∂mn_2),mag,mn)
∂ng_∂evg_2 = ( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag) + ω*ct(𝓕⁻¹_H̄,mn) ) 
∂ng_∂ε⁻¹_2 = real( herm( _outer(Ē,D) ) ) 
##
@tullio outer1[i,j,ix,iy] := conj(evg)[j,ix,iy]*𝓕⁻¹_H̄[i,ix,iy]
outer2 = _outer(𝓕⁻¹_H̄,evg)
outer1 ≈ outer2
##

###############################################################################################################################
###############################################################################################################################

@show gvd_RM
@show gvd_FD

T = Float64
Ninv                =   inv(1.0 * length(grid))
dk̂                  =   SVector(0.,0.,1.)

D                   =   fft( kx_tc(evg,mn,mag), fftax )
E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
# HMkH_1            =   HMₖH(ev,ε⁻¹,mag,mn)
# HMkH_2            =   -real( dot(evg , kx_ct( ifft( _dot(ε⁻¹, fft( zx_tc(evg,mn), fftax ) ), fftax ), mn, mag )  )  )
# HMkH_3            =   -real( dot(evg , zx_ct( ifft( _dot(ε⁻¹, fft( kx_tc(evg,mn,mag), fftax ) ), fftax ), mn  )  )  )
# HMkH_4            =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
# HMkH_5            =   -real( dot(evg , zx_ct( bfft( E, fftax ), mn  )  )  ) * Ninv
HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
inv_HMkH            =   inv(HMkH)

deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
# EdepsiE_1         =   HMH(vec(ev), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)
# EdepsiE_2         =   real(_expect(∂ε_∂ω,E)) * Ninv
# EdepsiE_3         =   real( dot(D,epsi_deps_E) ) * Ninv
# EdepsiE_4         =   real( dot(evg,kx_Fi_epsi_deps) )

ng                  =   (ω + EdepsiE/2) * inv_HMkH

ng_RM

∂ng_∂ω_1            =   inv_HMkH
∂ng_∂EdepsiE        =   inv_HMkH/2
∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2

∂ng_∂∂ε_∂ω          =   _outer(E,E) * ∂ng_∂EdepsiE
∂ng_∂ε⁻¹_1          =   2 * herm(_outer(deps_E,D)) * ∂ng_∂EdepsiE
∂ng_∂evg_1          =   2 * kx_Fi_epsi_deps_E * ∂ng_∂EdepsiE

∂ng_∂kx_1           =  real(_outer(Fi_epsi_deps_E, evg)) * ∂ng_∂EdepsiE
@tullio ∂ng_∂mag_1[ix,iy] := conj(reverse(∂ng_∂kx_1,dims=2))[i,j,ix,iy] * mn[i,j,ix,iy] 
@tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] 
∂ng_∂k_1            =   ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)

# ∂ng_∂ε⁻¹_2          =   2 * herm(_outer(deps_E,D)) * ∂ng_∂EdepsiE
# ∂ng_∂evg_2          =   2 * kx_Fi_epsi_deps_E * ∂ng_∂EdepsiE

# ∂ng_∂kx_2           =  real(_outer(Fi_epsi_deps_E, evg)) * ∂ng_∂EdepsiE
# @tullio ∂ng_∂mag_1[ix,iy] := conj(reverse(∂ng_∂kx_1,dims=2))[i,j,ix,iy] * mn[i,j,ix,iy] 
# @tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] 
# ∂ng_∂k_1            =   ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)

∂ng_∂k_2, ∂ng_∂evg_2, ∂ng_∂ε⁻¹_2 = ∇HMₖH(k,evg,ε⁻¹,grid) .* ∂ng_∂HMkH
# ∂ng_∂evg_2          =   (-2 * ng * inv_HMkH) * _cross(dk̂, E)

### ∇solve_k ###
k̄ = ∂ng_∂k_1 + ∂ng_∂k_2

∂ng_∂evg = vec(∂ng_∂evg_1) + ∂ng_∂evg_2
@show val_magmax(∂ng_∂evg_1)
@show val_magmax(∂ng_∂evg_2)
@show val_magmax(∂ng_∂evg)

M̂ = HelmholtzMap(k,ε⁻¹,grid)
P̂	= HelmholtzPreconditioner(M̂)
λ⃗	= eig_adjt(
    M̂,								 # Â
    ω^2, 							# α
    ev, 					 		 # x⃗
    0.0, 							# ᾱ
    ∂ng_∂evg;					    # x̄
    # λ⃗₀=λ⃗₀,
    P̂	= P̂,
)

@show val_magmax(λ⃗)
λd = fft( kx_tc( reshape( λ⃗, (2,size(grid)...) ), mn, mag ), fftax )
∂ng_∂ε⁻¹_31 = ε⁻¹_bar(vec(D), vec( λd ) , size(grid)...)

λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
ẽ 	 =   ifft( E, fftax )
λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
m̄_kx = real.( λẽ_sv .* conj.(view(evg,2,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,2,axes(grid)...)) )	#NB: m̄_kx and n̄_kx would actually
n̄_kx =  -real.( λẽ_sv .* conj.(view(evg,1,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,1,axes(grid)...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
@show k̄ₕ		= -∇ₖmag_m_n(
            māg_kx, 		# māg total
            m̄_kx.*mag, 	# m̄  total
            n̄_kx.*mag,	  	# n̄  total
            mag, m⃗, n⃗; 
            dk̂=SVector(0.,0.,1.), # dk⃗ direction
        )
λd2 = fft( kx_tc( ( (k̄ + k̄ₕ ) / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax )  # 2 * HMkH = ∂ω²∂k
∂ng_∂ε⁻¹_32 = ε⁻¹_bar(vec(D), vec( λd2 ) , size(grid)...)
∂ng_∂ε⁻¹_3 = ∂ng_∂ε⁻¹_31 + ∂ng_∂ε⁻¹_32
@show ∂ng_∂ω_2 =  ω * (k̄ + k̄ₕ ) / HMkH 
### end ∇solve_k ###
∂ng_∂ε = _dot( -ε⁻¹, (∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 + ∂ng_∂ε⁻¹_3), ε⁻¹ )
∂ng_∂ω_3, ∂ng_∂p_geom                                       =   ε_data_pb((∂ng_∂ε,∂ng_∂∂ε_∂ω,zero(ε)))
@show ∂ng_∂ω                                                =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3
@show ∂ng_∂p_geom

# val_magmax(λ⃗)

@show ∂ng_∂EdepsiE
@show ∂ng_∂HMkH
@show ∂ng_∂ω_1

@show val_magmax(∂ng_∂∂ε_∂ω)
@show val_magmax(∂ng_∂ε⁻¹_1)
@show val_magmax(∂ng_∂evg_1)
@show val_magmax(∂ng_∂mag_1)
@show val_magmax(∂ng_∂mn_1)
@show ∂ng_∂k_1 = mag_mn_pb(( ∂ng_∂mag_1, ∂ng_∂mn_1))[1]

@show val_magmax(∂ng_∂ε⁻¹_2)
@show val_magmax(∂ng_∂evg_2)
# @show ∂ng_∂k_2 = mag_mn_pb(( ∂ng_∂mag_2, ∂ng_∂mn_2))[1]

@show ∂ng_∂k = ∂ng_∂k_1 + ∂ng_∂k_2

@show ∂ng_∂ω_2
@show val_magmax(∂ng_∂ε⁻¹_3)
@show val_magmax(∂ng_∂ε)

@show ∂ng_∂ω_1
@show ∂ng_∂ω_2
@show ∂ng_∂ω_3

##

###############################################################################################################################
###############################################################################################################################
using Tullio
using OptiMode: ∇ₖmag_m_n

nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
nng⁻¹             =   sliceinv_3x3(nng);
ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω
Hv = copy(ev)

# innards of this function: from scripts/solve_grads_old.jl
# function calc_∂²ω²∂k²(p,geom_fn,f_ε_mats,k,Hv,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}

(ε_recalc,ε⁻¹_recalc,nng_recalc,nng⁻¹_recalc,ngvd_recalc), eps_data_pb = Zygote.pullback(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),(1,2,3,4),grid); # TODO: automate unpacking of dielectric data into (ε, ∂ε_∂ω, ∂²ε_∂ω²)
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
    nng⁻¹             =   sliceinv_3x3(nng);
    ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω
    return ε,ε⁻¹,nng,nng⁻¹,ngvd
end



Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# mag,m⃗,n⃗ = mag_m_n(k,grid)
# ∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
mag,mns = mag_mn(k,grid)
# mns = copy(mn)
∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,real(mag),real(mns))

########################
#######################
# k̄, H̄, nngī  = ∇HMₖH(k,Hv,nng⁻¹,grid; eigind=1)

# inards of:
# function ∇HMₖH(k::Real,H⃗::AbstractArray{Complex{T}},nng⁻¹::AbstractArray{T2,N2},grid::Grid{ND};eigind=1) where {T<:Real,ND,T2<:Real,N2}
T = Float64
# Setup
zxtc_to_mn = SMatrix{3,3,Float64}(	[	0 	-1	  0
1 	 0	  0
0 	 0	  0	  ]	)

kxtc_to_mn = SMatrix{2,2,Float64}(	[	0 	-1
1 	 0	  ]	)

# g⃗s, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.ignore() do
Ninv 		= 		1. / N(grid)
Ns			=		size(grid)
g⃗s = g⃗(grid)
d0 = randn(Complex{T}, (3,Ns...))
𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
# return (g⃗s,Ninv,Ns,𝓕,𝓕⁻¹)
# end
mag, m⃗, n⃗  = mag_m_n(k,g⃗s)
H = reshape(Hv,(2,Ns...))
Hsv = reinterpret(reshape, SVector{2,Complex{T}}, H )

#TODO: Banish this quadruply re(shaped,interpreted) m,n,mns format back to hell
# mns = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
# m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
# n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
# mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
m = mns[:,1,:,:] #real(reinterpret(reshape,T,m⃗))
n = mns[:,2,:,:] # real(reinterpret(reshape,T,n⃗))
# mns = cat(reshape(m,(3,1,Ns...)),reshape(n,(3,1,Ns...));dims=2)

### calculate k̄ contribution from M̄ₖ ( from ⟨H|M̂ₖ|H⟩ )
Ā₁		=	conj.(Hsv)
A2_init = bfft( ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(nng⁻¹)), 2:3)
Ā₂ = reinterpret(reshape,SVector{3,Complex{T}}, A2_init)
# reshape(
# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
# 	(3,size(gr)...),
# 	),
    # 𝓕⁻¹ * ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(nng⁻¹)), )
    # bfft( ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(nng⁻¹)), 2:3), )
Ā 	= 	Ā₁  .*  transpose.( Ā₂ )
m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)

# # diagnostic for nngī accuracy
# B̄₁_old = reinterpret(
# 	reshape,
# 	SVector{3,Complex{T}},
# 	# 𝓕  *  kxtcsp	 *	vec(H),
# 	𝓕 * kx_tc( conj.(H) ,mns,mag),
# 	)
# B̄₂_old = reinterpret(
# 	reshape,
# 	SVector{3,Complex{T}},
# 	# 𝓕  *  zxtcsp	 *	vec(H),
# 	𝓕 * zx_tc( H * Ninv ,mns),
# 	)
# B̄_old 	= 	 SMatrix{3,3,Float64,9}.(real.(Hermitian.(  B̄₁_old  .*  transpose.( B̄₂_old )  )) )
# B̄_oldf = copy(flat(B̄_old))
# println("sum(B̄_oldf): $(sum(B̄_oldf))")
# println("maximum(B̄_oldf): $(maximum(B̄_oldf))")
# # end diagnostic for nngī accuracy

B̄₁ = fft( kx_tc( conj.(H) ,mns,mag) , 2:3) # 𝓕 * kx_tc( conj.(H) ,mns,mag)
B̄₂ = fft( zx_tc( H * Ninv ,mns) , 2:3) # 𝓕 * zx_tc( H * Ninv ,mns)
@tullio B̄[a,b,i,j] := B̄₁[a,i,j] * B̄₂[b,i,j] + B̄₁[b,i,j] * B̄₂[a,i,j]   #/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2

# # diagnostic for nngī accuracy
#
# # println("sum(B̄): $(sum(real(B̄)))")
# # println("maximum(B̄): $(maximum(real(B̄)))")
# B̄_herm = real(B̄)/2
# println("sum(B̄_herm): $(sum(B̄_herm))")
# println("maximum(B̄_herm): $(maximum(B̄_herm))")
# # end diagnostic for nngī accuracy

C1_init = bfft(ε⁻¹_dot(  𝓕 * -kx_tc( H * Ninv, mns, mag) , nng⁻¹), 2:3)
C̄₁ = reinterpret(reshape,SVector{3,Complex{T}},C1_init)
# reshape,
# SVector{3,Complex{T}},
# reshape(
# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
# 	(3,size(gr)...),
# 	),
# 𝓕⁻¹ * ε⁻¹_dot(  𝓕 * -kx_tc( H * Ninv, mns, mag) , nng⁻¹),
# )
C̄₂ =   conj.(Hsv)
C̄ 	= 	C̄₁  .*  transpose.( C̄₂ )
m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )

# Accumulate gradients and pull back
nngī 	=  real(B̄)/2 #( B̄ .+ transpose.(B̄) ) ./ 2
k̄	 	= ∇ₖmag_m_n(
    māg_Ā, 				# māg total
    m̄_Ā.*mag .+ m̄_C̄, 	  # m̄  total
    n̄_Ā.*mag .+ n̄_C̄,	  # n̄  total
    mag,
    m⃗,
    n⃗,
)
# H̄ = Mₖᵀ_plus_Mₖ(H⃗,k,nng⁻¹,grid)
# Y = zx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mns,mag), (2:3) ), nng⁻¹), (2:3)), mns )
# X = -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mns), (2:3) ), nng⁻¹), (2:3) ), mns, mag )

# nngif = real(flat(nng⁻¹))
# X = -kx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_tc(H,mns)		, nng⁻¹), mns, mag )
# Y =  zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nng⁻¹), mns )
X = -kx_ct( bfft( ε⁻¹_dot( fft( zx_tc(H,mns), 2:3)		, nng⁻¹), 2:3), mns, mag )
Y =  zx_ct( bfft( ε⁻¹_dot( fft( kx_tc(H,mns,mag), 2:3)	, nng⁻¹), 2:3), mns )
H̄ = vec(X + Y) * Ninv
# return k̄, H̄, nngī


########################
#######################

# ( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
#                                      (k,Hv),
#                                       ∂ω²∂k_nd,
#                                        ω,
#                                     ε⁻¹,
#                                     grid; eigind)

# ∇solve_k(ΔΩ, Ω, ∂ω²∂k, ω, ε⁻¹, grid)
# ΔΩ, Ω get unpacked immediately as:
    # k̄ₖ, H̄ = ΔΩ
    # k, Hv = Ω
k̄ₖ = copy(k̄)
∂ω²∂k = ∂ω²∂k_nd 
eigind=1

Ninv 		= 		1. / N(grid)
Ns			=		size(grid)
M̂ = HelmholtzMap(k,ε⁻¹,grid) # dropgrad(grid))
P̂	= HelmholtzPreconditioner(M̂)
# λ⃗₀0 = randn(eltype(Hv), size(Hv) )
# λ⃗₀ = normalize(λ⃗₀0 - Hv*dot(Hv,λ⃗₀0))
# if !iszero(H̄)
    # solve_adj!(λ⃗,M̂,H̄,ω^2,Hv,eigind)
λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        Hv, 					 		 # x⃗
        0.0, 							# ᾱ
        H̄;								 # x̄
        # λ⃗₀=λ⃗₀,
        P̂	= P̂,
    )
############################3
################################

# k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,Hv,grid)

# inards of
# function ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid::Grid{ND,T}) where {ND,T<:Real}

λ⃗ 	-= 	 dot(Hv,λ⃗) * ev
λ	=	reshape(λ⃗,(2,size(grid)...))
d = fft( kx_tc( H , mn, mag ), 2:3 ) * Ninv
λd = fft( kx_tc( λ , mn, mag ), 2:3 )  
# d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
# λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
eīₕ = ε⁻¹_bar(vec(d), vec(λd), size(grid)...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), gridsize...)

# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
λẽ  =   bfft( ε⁻¹_dot(λd * Ninv, ε⁻¹), 2:3 ) #flat(ε⁻¹)) # _d2ẽ!(λẽ , λd  ,M̂ )
ẽ 	 =   bfft( ε⁻¹_dot(d        , ε⁻¹), 2:3 )

λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,2,axes(grid)...)) )
kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,1,axes(grid)...)) )
# m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,..])
# n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,..])
māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)

# @show k̄ₕ = -∇ₖmag_m_n(
#     māg,
#     kx̄_m⃗.*ms.M̂.mag, # m̄,
#     kx̄_n⃗.*ms.M̂.mag, # n̄,
#     ms.M̂.mag,
#     m⃗,
#     n⃗;
#     dk̂=SVector(0.,0.,1.), # dk⃗ direction
# )

# TODO: check if this shoudl be negated
k̄ₕ		= -∇ₖmag_m_n(
            māg, #māg_kx, 		# māg total
            # m̄_kx.*mag, 	# m̄  total
            # n̄_kx.*mag,	  	# n̄  total
            kx̄_m⃗.* mag, # m̄,
            kx̄_n⃗.* mag, # n̄,
            mag, m⃗, n⃗;
            dk̂=SVector(0.,0.,1.),
        )
    

###############################
#########################
# else
#     eīₕ 	= zero(ε⁻¹) #fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
#     k̄ₕ 	= 0.0
# end
# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# println("")
# println("k̄ₖ = $(k̄ₖ)")
# println("k̄ₕ = $(k̄ₕ)")
# println("k̄ₖ + k̄ₕ = $(k̄ₖ+k̄ₕ)")
λ⃗ₖ	 = ( (k̄ₖ + k̄ₕ ) / ∂ω²∂k ) * Hv
H 	= reshape(Hv,(2,Ns...))
λₖ  = reshape(λ⃗ₖ, (2,Ns...))
# d	= 	𝓕 * kx_tc( H  , mns, mag ) * Ninv
# λdₖ	=	𝓕 * kx_tc( λₖ , mns, mag )
d	= 	fft(kx_tc( H  , mns, mag ),_fftaxes(grid)) * Ninv
λdₖ	=	fft(kx_tc( λₖ , mns, mag ),_fftaxes(grid))
eīₖ = ε⁻¹_bar(vec(d), vec(λdₖ), Ns...)
ω̄  =  2ω * (k̄ₖ + k̄ₕ ) / ∂ω²∂k

# ∇solve_k returned values 
om̄₁ = copy(ω̄ )  
eī₁ =  copy(eīₖ + eīₕ)

###############################
################################
###
println("")
println("\n manual calc.:")
om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
println("om̄₁: $(om̄₁)")
println("om̄₂: $(om̄₂)")
println("om̄₃: $(om̄₃)")
om̄ = om̄₁ + om̄₂ + om̄₃
println("om̄: $(om̄)")


#######

# calculate and print neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
# ev_grid = reshape(Hv,(2,Ns...))
# mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
EE = 1im * ε⁻¹_dot( fft( kx_tc( ev_grid,mns,mag), (2:1+ND) ), ε⁻¹)
HH = fft(tc(kx_ct( ifft( EE, (2:1+ND) ), mns,mag), mns),(2:1+ND) ) / ω
EEs = copy(reinterpret(reshape,SVector{3,ComplexF64},EE))
HHs = copy(reinterpret(reshape,SVector{3,ComplexF64},HH))
Sz = dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
PP = 2*real(sum(Sz))
WW = dot(EE,_dot((ε+nng),EE))
ng = WW / PP

# ∂ω²∂k_disp = 2 * HMₖH(Hv,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
∂ω²∂k_nd = 2 * HMₖH(Hv,nng⁻¹,real(mag),real(mns))
neff = k / ω
# ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )
gvd_alt1 = ω * 4 / ∂ω²∂k_disp^2 * om̄ - 2 / ∂ω²∂k_disp 
gvd_alt2 = 2 / ∂ω²∂k_disp + ω * 4 / ∂ω²∂k_disp^2 * om̄
# println("∂ω²∂k_disp: $(∂ω²∂k_disp)")
println("neff: $(neff)")
println("ng: $(ng)")
println("gvd: $(gvd)")

println("")
println("calc. with pullbacks:")
# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī2)) ) ./ 2
# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2

# om̄₂_pb = nngi_pb(herm(nngī))[1] #nngī2)
# om̄₃_pb = ei_pb(herm(eī₁))[1] #eī₁)

(ε,ε⁻¹,nng,nng⁻¹,ngvd), eps_data_pb

om̄₂_pb = eps_data_pb((nothing,nothing,nothing,herm(nngī),nothing))[1] #nngī2)
om̄₃_pb = eps_data_pb((nothing,herm(eī₁),nothing,nothing,nothing))[1] #eī₁)

println("om̄₁: $(om̄₁)")
println("om̄₂_pb: $(om̄₂_pb)")
println("om̄₃_pb: $(om̄₃_pb)")
om̄_pb = om̄₁ + om̄₂_pb + om̄₃_pb
println("om̄_pb: $(om̄_pb)")
gvd_pb = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄_pb #( ng / ω ) * ( 1. - ( ng * om̄ ) )
println("gvd_pb: $(gvd_pb)")
println("")















##################################################################################################################################
##################################################################################################################################
# check relationship between old dispersion-tensor convention (nng,ngvd) and newer (∂ε_∂ω,∂²ε_∂ω²)
nng = (ω * ∂ε_∂ω) + ε
W_nondisp = real(_expect(ε,E)) + (Ngrid* (ω^2))
W = real(_expect( ε + ω*∂ε_∂ω ,E)) + (Ngrid* (ω^2))
real(_expect(nng,E)) + (Ngrid* (ω^2))
W_old = real(dot(E,_dot(nng,E))) + (Ngrid* (ω^2))  
@test W_old ≈ W

# ∂ω²∂k_nd1 = 2 * HMₖH(ev,ε⁻¹,mag,mn) # TODO: replace this with ∂ω²∂k_nd = 2ω * P / W_nondisp after checking they are equal
# D           =   fft( kx_tc( ev_grid, mn, mag ), fftax )
# E           =   ε⁻¹_dot( D, ε⁻¹)
# H           =   ω * fft( tc(ev_grid,mn), fftax ) 
# # `P`: integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# P           =   2*real(_sum_cross_z(conj(E),H))
# # `W_nondisp`: non-dispersive energy density per unit length ∫dA ( D⃗⋅E⃗ + |H⃗|² ) without magnetism (μ = μ₀)
# W_nondisp   =   real(_expect(ε,E)) + (Ngrid* (ω^2)) 

# ∂ω²∂k_nd = (2*ω) * (P / W_nondisp)

# function group_index_and_gvd(ω::Real,ε::AbstractArray{<:Real},∂ε_∂ω::AbstractArray{<:Real},∂²ε_∂ω²::AbstractArray{<:Real},
# 	grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,max_eigsolves=60,maxiter=500,k_tol=1e-8,eig_tol=1e-8,
# 	log=false,kguess=nothing,Hguess=nothing,dk̂=SVector(0.0,0.0,1.0),log=false,f_filter=nothing) where {ND,T<:Real}

using Tullio

function E_ng_gvd(k::Real,evec,ω,ε,∂ε_∂ω,∂²ε_∂ω²,grid;dk̂=SVector{3}(0.0,0.0,1.0))
	
    fftax           =   _fftaxes(grid)      # spatial grid axes of field arrays, 2:3 (2:4) for 2D (3D) using current field data format
    Ngrid           =   length(grid)        # total number of grid points, Nx*Ny (Nx*Ny*Nz) in 2D (3D)
    ε⁻¹             =   sliceinv_3x3(ε);
    nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
    ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω(ε + ω*∂ε_∂ω) = 2*∂ε_∂ω + ω*∂²ε_∂ω²  TODO: check this
    # nng             =   (ω/2 * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
    # ngvd            =   (3/2) * ∂ε_∂ω  +  ω/2* ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω(ε + ω*∂ε_∂ω) = 2*∂ε_∂ω + ω*∂²ε_∂ω²  TODO: check this
    mag, mn = mag_mn(k,g⃗(grid))
    ev_grid = reshape(evec,(2,size(grid)...))
    Ninv = inv(1.0*length(grid))
    # nng = (ω/2) * ∂ε_∂ω + ε
    # norm_fact = inv(sqrt(δV(grid) * Ngrid) * ω)

    # old field phases for ref.
    # D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
    # E = ε⁻¹_dot( D, ε⁻¹)
    # H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)

    # D       =   1im * fft( kx_tc( ev_grid,mn,mag), _fftaxes(grid) )
    # E       =   ε⁻¹_dot( D, ε⁻¹)
    # H       =   (1im * ω) * fft( tc(ev_grid,mn), (2:3) ) 
    # P           =   2*real(_sum_cross_z(conj(E),H))
    
    D           =   fft( kx_tc( ev_grid, mn, mag ), fftax )
    E           =   ε⁻¹_dot( D, ε⁻¹)
    H           =   ω * fft( tc(ev_grid,mn), fftax ) 
    P           =   2*real(_sum_cross_z(conj(E),H)) # `P`: integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
    
    # `W_nondisp`: non-dispersive energy density per unit length ∫dA ( D⃗⋅E⃗ + |H⃗|² ) without magnetism (μ = μ₀)
    @show W_nondisp   =   real(_expect(ε,E)) + (Ngrid* (ω^2))               
    # `W`: dispersive energy density per unit length ∫dA ( E⃗⋅(ε + ω*∂ε_∂ω)⋅E⃗  + |H⃗|² ) without magnetism (μ = μ₀)
    @show W           =   W_nondisp + real(_expect( ω*∂ε_∂ω ,E))  # W = real(_expect( ε + ω*∂ε_∂ω ,E)) + (Ngrid* (ω^2))   
    # `W`version with first and second-order dispersive contributions to energy density, for consideration
    # W2 =   real(_expect( ε + ω*∂ε_∂ω + 2*ω^2*∂²ε_∂ω² ,E)) + (Ngrid* (ω^2))
	@show ng          =   W / P
    @show ∂ω²∂k_nd    =   (2*ω) * (P / W_nondisp)   # = 2ω/ng_nondisp,  used later 
    #previously: ∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n) 

    # NB: 
    # previous `nng` tensor field was defined such that the dispersive energy density was calc'd as
    # W = real(dot(E,_dot(nng,E))) + (Ngrid* (ω^2))     # energy density per unit length
    # If needed for compatiblity, `nng` should be defined as
    # nng = ω * ∂ε_∂ω + ε       

    # NB: ev_grid (eigenvector `evec` reshaped to (2,Nx,Ny) was previously called `Hₜ`

	# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
	W̄       =   inv(P)              # W̄:    ∂ng/∂W, TODO: rename ∂ng_∂W
	@show om̄₁₁    =   2*ω * Ngrid * W̄     # om̄₁₁: part of ∂ng/∂ω, TODO: rename ∂ng_∂ω_11
	nnḡ     =   _outer(E,E) * W̄     # nnḡ:  ∂ng/∂nng, TODO: rename ∂ng_∂nng
	# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
	# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
	H̄ = (-2*ng*W̄) * _cross(dk̂, E)                  # H̄:  ∂ng/∂H, TODO: rename ∂ng_∂H 
	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )     # Ē:  ∂ng/∂E, TODO: rename ∂ng_∂E
	@show om̄₁₂ = dot(H,H̄) / ω                            # om̄₁₂: part of ∂ng/∂ω, TODO: rename ∂ng_∂ω_12
	@show om̄₁₂_alt1 = dot(H,conj(H̄)) / ω
    @show om̄₁₂_alt2 = dot(H,-H̄) / ω
    @show om̄₁ = om̄₁₁ + om̄₁₂                              # om̄₁: part accumulation of ∂ng/∂ω, TODO: rename ∂ng_∂ω_1, consider inlining om̄₁₁ & om̄₁₂ 
	# eī₁ = _outer(Ē,D) ####################################
    eī₁ = _outer(Ē,D)
	𝓕⁻¹_ε⁻¹_Ē   = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))                  #   𝓕⁻¹_ε⁻¹_Ē: reciprocal space version of Ē, TODO: check if ε⁻¹ should be ε here? or is Ē actually D̄?
	𝓕⁻¹_H̄       = bfft( H̄ ,(2:3))                               #   𝓕⁻¹_H̄:     reciprocal space version of H̄
	
    ∂ng_∂evec = 1im*vec( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag) + ω*ct(𝓕⁻¹_H̄,mn) )     #   sum reciprocal space versions of Ē & H̄ and change to transverse polarization basis to get ∂ng_∂ev_grid
	
    local one_mone = [1.0im, -1.0im]

    # TODO: simplify this variable name and add description in comments
	@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(ev_grid;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
	

	######### back-propagate (∂ng_∂k & ∂ng_∂ev_grid) through `solve_k` to get ∂ng_∂ω & ∂ng_∂ε⁻¹ ############
	# solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
    λ⃗₀0 = randn(eltype(evec), size(evec) )
    λ⃗₀ = normalize(λ⃗₀0 - evec*dot(evec,λ⃗₀0))
    P̂	= HelmholtzPreconditioner(M̂)
	λ⃗	= eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		evec, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(∂ng_∂evec);					# x̄
		λ⃗₀ = λ⃗₀,
		P̂	= P̂,
	)
	### k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)
	λ = reshape(λ⃗,(2,size(grid)...))
	λd 	= 	fft(kx_tc( λ , mn, mag ),_fftaxes(grid))
	eīₕ	 = 	 ε⁻¹_bar(vec(D * (Ninv * -1.0im)), vec(λd), size(grid)...) ##########################
	λẽ  =   bfft(ε⁻¹_dot(λd , ε⁻¹),_fftaxes(grid))
	ẽ 	 =   bfft(E * -1.0im,_fftaxes(grid))
    
    # local one_mone = [1.0im, -1.0im]        # constant, used in a Tullio kernel below    <---- normal setting
    local one_mone2 = [1.0im, -1.0im] # [1.0, -1.0] # 

    # normally: top of these three is active
	@tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone2[i] * reverse(conj(ev_grid);dims=1)[i,ix,iy] * (Ninv*λẽ)[j,ix,iy] + -1.0im * one_mone2[i] * reverse(conj(λ);dims=1)[i,ix,iy] * (Ninv*ẽ)[j,ix,iy]  nograd=one_mone2
	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(ev_grid);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(ev_grid);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
	
    @tullio mn̄s[i,j,ix,iy] := mag[ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[i,j,ix,iy]  + 1im*ω*conj(ev_grid)[i,ix,iy]*𝓕⁻¹_H̄[j,ix,iy]
	
    # NB: `mn` axis order has changed since this was originally written
    # now it is (3,2,size(grid)...), prev. it was (2,3,size(grid)...)
    # As a temporary fix to debug, I'll definite mn_perm with the first two axes permuted
    mn_perm = copy(permutedims(mn,(2,1,3,4)))
    mn̄s_perm = real(copy(permutedims(mn̄s,(2,1,3,4))))

    @tullio māg[ix,iy] := mn_perm[a,b,ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[a,b,ix,iy]


	# k̄ = ∇ₖmag_mn(māg,mn̄s,mag,mn)
    @show k̄ = ∇ₖmag_mn(māg,mn̄s_perm,mag,mn)
    # TODO: replace this code usng `m` and `n` arrays with `mn` version
    m = copy(mn[:,1,:,:])
    n = copy(mn[:,2,:,:])

	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄s)[imn,i,ix,iy] * mn_perm[imn,mod(i-2),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄s)[imn,i,ix,iy] * mn_perm[imn,mod(i-1),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	@show k̄_old = k̄_mag + k̄_mn

    # λ⃗ 	-= 	 dot(evec,λ⃗) * evec
    # λ	=	reshape(λ⃗,(2,size(grid)...))
    # d = _H2d!(M̂.d, ev_grid * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * M̂.Ninv
    # λd = similar(M̂.d)
    # _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
    # eīₕ = ε⁻¹_bar(copy(vec(M̂.d)), copy(vec(λd)), size(grid)...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(M̂.d), vec(λd), size(grid)...)
    
    # # back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
    # λd *=  M̂.Ninv
    # λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,M̂ ) )
    # ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d,M̂) )
    # kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ .* conj.(view(λ,2,axes(grid)...)) )
    # kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ .* conj.(view(λ,1,axes(grid)...)) )
    # # m⃗ = reinterpret(reshape, SVector{3,Float64},M̂.mn[:,1,..])
    # # n⃗ = reinterpret(reshape, SVector{3,Float64},M̂.mn[:,2,..])
    # māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)

    # @show k̄ₕ_old = -mag_m_n_pb(( māg, kx̄_m⃗.*M̂.mag, kx̄_n⃗.*M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
    kx̄_m⃗ = reinterpret(reshape,SVector{3,Float64},mn̄s_perm[:,1,:,:])
    kx̄_n⃗ = reinterpret(reshape,SVector{3,Float64},mn̄s_perm[:,2,:,:])
    m⃗ = reinterpret(reshape,SVector{3,Float64},mn[:,1,:,:])
    n⃗ = reinterpret(reshape,SVector{3,Float64},mn[:,2,:,:])
    @show k̄_alt = ∇ₖmag_m_n(
        māg,
        kx̄_m⃗.*M̂.mag, # m̄,
        kx̄_n⃗.*M̂.mag, # n̄,
        M̂.mag,
        m⃗,
        n⃗;
        dk̂=SVector(0.,0.,1.), # dk⃗ direction
    )

	### end: k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)

	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	λₖ  = ( real(k̄) / ∂ω²∂k_nd ) * ev_grid # Hₜ #reshape(λ⃗ₖ, (2,size(grid)...))
	λdₖ	=	fft(kx_tc( λₖ , mn, mag ),_fftaxes(grid))
	eīₖ = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), size(grid)...) ####################################
	@show om̄₂  =  2*ω * real(k̄) / ∂ω²∂k_nd
	##### \grad solve k
	# @show om̄₃ = dot(herm(nnḡ), ngvd)
    @show om̄₃ = dot(herm(nnḡ), ngvd)

	@show om̄₄ = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ),  -1*_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ) ) #∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
    @show om̄₄ = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ),  -1*_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ) ) #∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
   
    
    @show om̄₄_alt1 = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * 1.0),D) ),  -1*_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ) ) #∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
    @show om̄₄_alt2 = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * 1.0im),D) ),  -1*_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ) ) #∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	@show om̄₄_old = dot( ( eīₖ + eīₕ + eī₁ ), -1*_dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ))
	@show om̄ = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )
    @show om̄_alt = real( om̄₁ + om̄₂ + om̄₃ )
	# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
	@show ∂ω²∂k_disp = (2*ω) / ng
	# ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
	@show gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) ) 
    @show gvd_1 = 2 / ∂ω²∂k_disp 
    @show gvd_2 = - ω * 4 / ∂ω²∂k_disp^2 * om̄ 
    @show gvd_alt = ( ng / ω ) * ( 1. - ( ng * om̄ ) )  
    @show gvd_alt1 = ( ng / ω ) 
    @show gvd_alt2 = ( ng / ω ) *  -1.0 * ( ng * om̄ )    
    
    @show om̄_alt2 = om̄₂ + om̄₃ + om̄₄
    @show (ng_ev / ω) * (1.0 - ng_ev * om̄_alt2)

    @show om̄_alt3 = om̄₁₁ + om̄₁₂ + om̄₂ + om̄₃  #+ om̄₄
    @show (ng_ev / ω) * (1.0 - ng_ev * om̄_alt3)

	return ( ng, gvd, E )
end

# group_index(k,ev,ω,ε,∂ε_∂ω,grid)

ng_out, gvd, E_out = E_ng_gvd(k,ev,ω,ε,∂ε_∂ω,∂²ε_∂ω²,grid)
##

using Zygote: dropgrad
nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
nng⁻¹             =   sliceinv_3x3(nng);
ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω(ε + ω*∂ε_∂ω) = 2*∂ε_∂ω + ω*∂²ε_∂ω²  TODO: check this
neff1,ng1,gvd1 = neff_ng_gvd(ω,ε,ε⁻¹,nng,nng⁻¹,ngvd,k,ev,grid;eigind=1,log=false)

##
function ∇ₖmag_mn(māg::AbstractArray{T1,2},mn̄,mag::AbstractArray{T2,2},mn;dk̂=SVector{3}(0.,0.,1.)) where {T1<:Number,T2<:Number}
	m = view(mn,:,1,:,:)
	n = view(mn,:,2,:,:)
	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄)[i,imn,ix,iy] * mn[mod(i-2),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄)[i,imn,ix,iy] * mn[mod(i-1),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

function ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	return +( k̄_mag, k̄_m, k̄_n )
end

gvd - gvd_FD
gvd - gvd_RM

# nng = inv.(nnginv)
# ε = inv.(ε⁻¹)
# ∂ε∂ω_man = (2/ω) * (nng .- ε)
# ∂ei∂ω_man = copy(flat(-(ε⁻¹.^2) .* ∂ε∂ω_man ))
# ∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * ε⁻¹.^2 .* (  inv.(nng⁻¹) .- inv.(ε⁻¹) ) #(2.0/ω) * ε⁻¹ .* (  ε⁻¹ .* inv.(nng⁻¹) - I )

# function ∂nng∂ω_man_LN(om)
# 	 ng = ng_MgO_LiNbO₃(inv(om))[1,1]
# 	 n = sqrt(ε_MgO_LiNbO₃(inv(om))[1,1])
# 	 gvd = gvd_MgO_LiNbO₃(inv(om))[1,1]  #/ (2π)
# 	 # om = 1/om
# 	 om*(ng^2 - n*ng) + n * gvd
# end

# previously working
# ∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * (  ε⁻¹.^2 .* inv.(nng⁻¹) .- ε⁻¹ )
# ∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω) = -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd) # (1.0/ω) * (nng⁻¹ .- ε⁻¹ ) .- (  ngvd .* (nng⁻¹).^2  )

"""
	∂ε⁻¹_∂ω(ε⁻¹,nng,ω) computes:
  ∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * (  ε⁻¹.^2 .* inv.(nng⁻¹) .- ε⁻¹ )
"""
function ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω)
	deps_dom = inv(ω) * (nng - ε)
	dei_dom = -1.0 * _dot(ε⁻¹,deps_dom,ε⁻¹)  #-(2.0/om) * ( _dot(ei,ei,nng) - ei )
end

"""
	∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω) computes:
  -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd)
"""
function ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω)
	dnngi_dom = -1*_dot(nng⁻¹, ngvd, nng⁻¹)
end

# ε⁻¹             =   sliceinv_3x3(ε);
nng             =   (ω * ∂ε_∂ω) + ε     # for backwards compatiblity with (nng,ngvd) dispersion tensor old convention
ngvd            =   2 * ∂ε_∂ω  +  ω * ∂²ε_∂ω² # I think ngvd = ∂/∂ω( nng ) = ∂/∂ω(ε + ω*∂ε_∂ω) = 2*∂ε_∂ω + ω*∂²ε_∂ω²  TODO: check this
@test (inv(ω) * (nng - ε)) ≈ ∂ε_∂ω

# ∇ₖmag_mn(māg,mn̄s,mag,mn)

##
3

# ## Old "calculate effective group index `ng`" from `group_index_and_gvd` just in case
# Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# Ninv 		= 		1. / Ngrid
# mag,m⃗,n⃗ = mag_m_n(k,grid)
# m = flat(m⃗)
# n = flat(n⃗)
# mns = copy(vcat(reshape(m,1,3,Ns...),reshape(n,1,3,Ns...)))
# Hₜ = reshape(Hv,(2,Ns...))
# D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
# E = ε⁻¹_dot( D, ε⁻¹)
# # E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
# # H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
# H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
# P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# # W = dot(E,_dot((ε+nng),E))             # energy density per unit length
# W = real(dot(E,_dot(nng,E))) + (Ngrid* (ω^2))     # energy density per unit length
# ng = real( W / P )



om_bar = calc_∂²ω²∂k²(p,geom_fn,f_ε_mats,k,ev,grid)








##############################################################################################################
##
##   Mildly Interesting Note: 
##
##   We can calculate the real-space H field in two nominally-equivalent different ways: 
##
##   (1)   H⃗(x,y,z) = ω * FFT( transverse_to_cartesian( H⃗_planewave_transverse ) )
##
##      or
##
##   (2)   H⃗(x,y,z) = ( 1 / ω ) * FFT( transverse_to_cartesian( iFFT( cartesian_to_transverse( ∇ × E⃗ ) ) ) )
##
##  where the real space E-field in (2) is calculated as
##
##      E⃗ = ε⁻¹ ⋅ FFT( ∇ × transverse_to_cartesian( H⃗_planewave_transverse ) )
##  
##  Below we compare H-fields calculated these two ways to see if one is more accurate  
##
###############################################################################################################
H1 =  ω * fft( tc(ev_grid,mn), (fftax) ) 
H2 = ( 1 / ω ) * fft(tc(kx_ct( ifft( E,  fftax ), mn,mag), mn), fftax )
@test val_magmax(H2) ≈ val_magmax(H1)
@test H2 ≈ H1
## comparing real-space group index calculations using these two H-fields with the
## corresponding plane-wave-basis-calculated group index (assumed to be ground truth),
## I have found the real space H-field calculated with fewer FFTs (`H1` above) gives
## more accurate results
@show ng_err1 = ( ( real(_expect(ε,E)) + sum(abs2,H1) ) / ( 2*real(_sum_cross_z(conj(E),H1)) ) ) - ( ω / HMₖH(ev,ε⁻¹,mag,mn) )
@show ng_err2 = ( ( real(_expect(ε,E)) + sum(abs2,H2) ) / ( 2*real(_sum_cross_z(conj(E),H2)) ) ) - ( ω / HMₖH(ev,ε⁻¹,mag,mn) )

# # W = dot(E,_dot((ε+nng),E))              # energy density per unit length
# W = real(dot(E,_dot(nng,E))) + (Ngrid* (ω^2))     # energy density per unit length
@test _expect(ε,E) ≈ dot(E,_dot( ε ,E))
@test real(_expect(ε,E)) ≈ Ngrid* (ω^2)
@test sum(abs2,H) ≈ Ngrid* (ω^2)

