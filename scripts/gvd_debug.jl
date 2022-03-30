"""
Functions for checking/debugging GVD calculation
"""
using LinearAlgebra, StaticArrays, FFTW, GeometryPrimitives, OptiMode, Tullio, Test
using ChainRules, Zygote, FiniteDifferences, ForwardDiff, FiniteDiff
using OptiMode: ∇ₖmag_m_n, ∇ₖmag_mn
# using CairoMakie

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

evg = ev_grid

##

function gvd_check1(ω)
    # ω                               =   1.1 
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

    @show ng_FD = FiniteDiff.finite_difference_derivative(ω) do ω
        ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        return kmags[1]
    end
    
    @show ng_RM = Zygote.gradient(ω) do ω
        ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        return kmags[1]
    end |> first
        
    @show gvd_FD = FiniteDiff.finite_difference_derivative(ω) do ω
        ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
        return ng
    end
    
    @show gvd_RM = Zygote.gradient(ω) do ω
        ε_data          =   smooth_ε(geom_fn(p[2:5]),f_ε_mats([ω,]),minds,grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
        return ng
    end |> first
    
    ###########################################################################
    #
    #       Calculate GVD using AD, but in steps for access to intermediate
    #       ChainRule components for diagnostics (comparison with manual calc.) 
    #
    ###########################################################################
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

    ∂ng_∂ω                                                =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3
    @show gvd_RM2 = ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3
    
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

    ###########################################################################
    #
    #       Calculate GVD manually with AD-differentiable functions
    #
    ###########################################################################

    ev  =   evecs[1]
    k   =   kmags[1]
    evg =   reshape(ev,(2,size(grid)...))
    T = Float64
    Ninv                =   inv(1.0 * length(grid))
    dk̂                  =   SVector(0.,0.,1.)
    # mag2,m⃗,n⃗           =    mag_m_n(k,grid)
    mag,mn              =   mag_mn(k,grid)
    m⃗ = reinterpret(reshape, SVector{3,Float64}, mn[:,1,axes(grid)...])
	n⃗ = reinterpret(reshape, SVector{3,Float64}, mn[:,2,axes(grid)...])
    one_mone = [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
    HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
    inv_HMkH            =   inv(HMkH)
    
    ### ∇⟨H|M̂(k,∂ε⁻¹/∂ω)|H⟩ ###
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
    ### end ∇⟨H|M̂(k,∂ε⁻¹/∂ω)|H⟩ ###

    ### ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    # om̄₁₂ = dot(H,H̄) / ω
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn_2[i,j,ix,iy] := mag[ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[i,j,ix,iy]   +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]  
    @tullio ∂ng_∂mag_2[ix,iy] := mn[a,b,ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[a,b,ix,iy]
    ∂ng_∂k_2 = ∇ₖmag_mn(real(∂ng_∂mag_2),real(∂ng_∂mn_2),mag,mn)
    ∂ng_∂evg_2 = ( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag) + ω*ct(𝓕⁻¹_H̄,mn) ) 
    ∂ng_∂ε⁻¹_2 = real( herm( _outer(Ē,D) ) ) 
    ### end ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###

    ### ∇solve_k ###
    k̄ = ∂ng_∂k_1 + ∂ng_∂k_2
    ∂ng_∂evg = vec(∂ng_∂evg_1) + vec(∂ng_∂evg_2)
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
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    ∂ng_∂ε⁻¹_31 = ε⁻¹_bar(vec(D), vec( λd ) , size(grid)...) * Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )
    λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
    ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
    m̄_kx = real.( λẽ_sv .* conj.(view(evg,2,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,2,axes(grid)...)) )	#NB: m̄_kx and n̄_kx would actually
    n̄_kx =  -real.( λẽ_sv .* conj.(view(evg,1,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,1,axes(grid)...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
    māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
    k̄ₕ		= -∇ₖmag_m_n(
                māg_kx, 		# māg total
                m̄_kx.*mag, 	# m̄  total
                n̄_kx.*mag,	  	# n̄  total
                mag, m⃗, n⃗; 
                dk̂=SVector(0.,0.,1.), # dk⃗ direction
            )
    λd2 = fft( kx_tc( ( (k̄ + k̄ₕ ) / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax ) * Ninv  # 2 * HMkH = ∂ω²∂k
    ∂ng_∂ε⁻¹_32 = ε⁻¹_bar(vec(D), vec( λd2 ) , size(grid)...)
    ∂ng_∂ε⁻¹_3 = ∂ng_∂ε⁻¹_31 + ∂ng_∂ε⁻¹_32
    ∂ng_∂ω_2 =  ω * (k̄ + k̄ₕ ) / HMkH 
    ### end ∇solve_k ###
    ∂ng_∂ε = _dot( -ε⁻¹, (∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 + ∂ng_∂ε⁻¹_3), ε⁻¹ )
    ∂ng_∂ω_3, ∂ng_∂p_geom                                       =   ε_data_pb((∂ng_∂ε,∂ng_∂∂ε_∂ω,zero(ε)))
    ∂ng_∂ω                                                      =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3

    gvd = ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3

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
    @show ∂ng_∂k_1  

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
    @show ∂ng_∂ω                                           
    @show ∂ng_∂p_geom
    # return nothing
    return ng_FD, gvd_FD, ng_RM, gvd_RM, ng, gvd, gvd_RM2
end

# ng_FD, gvd_FD, ng_RM, gvd_RM, ng, gvd, gvd_RM2 = gvd_check1(1.1)
##
oms = collect(0.7:0.1:1.6)
# gvd_data_check1_oms = map(gvd_check1,oms)  # ran in 15min or so on SC, output pasted below
gvd_data_check1_oms = [
    (2.3452384707502567, -0.06836404334647232, 2.345238589236614, -0.06836477684341258, 2.3452384706888623, -0.06836477684342057, -0.06836477684343589),
    (2.341964635577782, -0.00015313342249145131, 2.341964726698747, -0.00015362675047303487, 2.3419646344348166, -0.00015362675046559637, -0.00015362675046615148),
    (2.344738341317966, 0.05383841662936138, 2.344738420176899, 0.053838042110794904, 2.344738341327747, 0.05383804211081966, 0.05383804211081322),
    (2.3524563208106612, 0.09947696760509549, 2.3524563925252595, 0.09947666498111574, 2.3524563209584426, 0.09947666498110264, 0.09947666498110264),
    (2.3644757185595515, 0.1403348086021099, 2.3644757860483745, 0.1403345522101166, 2.364475718531385, 0.14033455221011948, 0.14033455221013147),
    (2.3804417409588905, 0.17872379351918355, 2.3804418075332587, 0.17872356788066213, 2.3804417422104085, 0.178723567880672, 0.17872356788066113),
    (2.4001921213733164, 0.21624923998783976, 2.4001921854791117, 0.21624903547077728, 2.4001921212014654, 0.21624903547077984, 0.216249035470787),
    (2.4237032263081684, 0.2541166689149096, 2.4237032904313356, 0.2541164810212183, 2.4237032264366705, 0.25411648102124196, 0.2541164810212432),
    (2.451059800648371, 0.2933109634589201, 2.451059864745382, 0.2933107931789597, 2.4510598004922994, 0.293310793178845, 0.2933107931788471),
    (2.482438886538844, 0.3347077963417258, 2.4824389510928273, 0.3347080557367859, 2.482438886173562, 0.33470805571661794, 0.33470805571661727), 
];
ng_FD_check1_oms            =   [xx[1] for xx in gvd_data_check1_oms]; 
gvd_FD_check1_oms           =   [xx[2] for xx in gvd_data_check1_oms]; 
ng_RM_check1_oms            =   [xx[3] for xx in gvd_data_check1_oms]; 
gvd_RM_check1_oms           =   [xx[4] for xx in gvd_data_check1_oms]; 
ng_check1_oms               =   [xx[5] for xx in gvd_data_check1_oms]; 
gvd_check1_oms              =   [xx[6] for xx in gvd_data_check1_oms]; 
gvd_RM2_check1_oms          =   [xx[7] for xx in gvd_data_check1_oms];

using CairoMakie
let fig = Figure(), oms=oms, ng_data=(ng_FD_check1_oms,ng_RM_check1_oms,ng_check1_oms), gvd_data=(gvd_FD_check1_oms,gvd_RM_check1_oms,gvd_check1_oms)
    λs = inv.(oms)
    ax_ng   = fig[1,1] = Axis(fig, xlabel = "vacuum wavelength (μm)", ylabel = "ng", rightspinevisible = false,  yminorticksvisible = false)
    ax_gvd  = fig[2,1] = Axis(fig, xlabel = "vacuum wavelength (μm)", ylabel = "gvd", rightspinevisible = false,  yminorticksvisible = false)
    ax_ng_err       =   fig[1,1]    =   Axis(fig, yaxisposition = :right, yscale = log10, ylabel = "ng_err", xgridstyle = :dash, ygridstyle = :dash, yminorticksvisible = true)
    ax_gvd_err      =   fig[2,1]    =   Axis(fig, yaxisposition = :right, yscale = log10, ylabel = "gvd_err", xgridstyle = :dash, ygridstyle = :dash, yminorticksvisible = true)

    plt_fns         =   [scatter!,scatter!,scatterlines!]
    colors          =   [:red,:black,:green]
    labels          =   ["FD","AD","manual"]
    markersizes     =   [16,16,12]
    markers         =   [:circle,:diamond,:cross]

    plts_ng = [pltfn(ax_ng,λs,ngs,color=clr,label=lbl,marker=mrkr,markersize=mrkrsz) for (pltfn,ngs,clr,lbl,mrkr,mrkrsz) in zip(plt_fns,ng_data,colors,labels,markers,markersizes)]
    plts_gvd = [pltfn(ax_gvd,λs,gvds,color=clr,label=lbl,marker=mrkr,markersize=mrkrsz) for (pltfn,gvds,clr,lbl,mrkr,mrkrsz) in zip(plt_fns,gvd_data,colors,labels,markers,markersizes)]

    ng_err_data         =   [abs.(ng_data[1] .- ng_data[2]),abs.(ng_data[1] .- ng_data[3]),abs.(ng_data[2] .- ng_data[3])]
    gvd_err_data        =   [abs.(gvd_data[1] .- gvd_data[2]),abs.(gvd_data[1] .- gvd_data[3]),abs.(gvd_data[2] .- gvd_data[3])]

    plt_fns_err         =   [lines!,lines!,lines!]
    colors_err          =   [:blue,:magenta,:orange]
    labels_err          =   ["|FD-AD|","|FD-manual|","|AD-manual|"]
    markersizes_err     =   [16,12,12]
    markers_err         =   [:circle,:diamond,:cross]

    plts_ng_err = [pltfn(ax_ng_err,λs,ng_errs,color=clr,label=lbl) for (pltfn,ng_errs,clr,lbl,mrkr,mrkrsz) in zip(plt_fns_err,ng_err_data,colors_err,labels_err,markers_err,markersizes_err)]
    plts_gvd_err = [pltfn(ax_gvd_err,λs,gvd_errs,color=clr,label=lbl) for (pltfn,gvd_errs,clr,lbl,mrkr,mrkrsz) in zip(plt_fns_err,gvd_err_data,colors_err,labels_err,markers_err,markersizes_err)]

    Legend(fig[1, 2], ax_gvd, merge = true)
    Legend(fig[2, 2], ax_gvd_err, merge = true)
    return fig
end

gvd_FD_check1_oms .- gvd_RM_check1_oms
gvd_FD_check1_oms .- gvd_RM2_check1_oms
gvd_FD_check1_oms .- gvd_check1_oms
gvd_RM_check1_oms .- gvd_check1_oms
gvd_RM2_check1_oms .- gvd_check1_oms


##

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


@test gvd_RM ≈ gvd_FD rtol = 1e-5
@show gvd_RM_vs_FD_err    = gvd_RM - gvd_FD
@show gvd_RM_vs_FD_relerr = abs(gvd_RM_vs_FD_err) / gvd_FD

##

function ng_gvd_single1(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid;dk̂=SVector{3,Float64}(0.0,0.0,1.0),adj_tol=1e-8)
    fftax               =   _fftaxes(grid)      
    evg                 =   reshape(ev,(2,size(grid)...))
    T = Float64
    Ninv                =   inv(1.0 * length(grid))
    mag,mn              =   mag_mn(k,grid)
    m⃗ = reinterpret(reshape, SVector{3,Float64},mn[:,1,axes(grid)...])
    n⃗ = reinterpret(reshape, SVector{3,Float64},mn[:,2,axes(grid)...])
    local one_mone      =   [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
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

    ### ∇⟨H|M̂(k,∂ε⁻¹/∂ω)|H⟩ ###
    ∂ng_∂∂ε_∂ω          =   _outer(E,E) * Ninv * ∂ng_∂EdepsiE
    ∂ng_∂ε⁻¹_1          =   herm(_outer(deps_E,D)) * Ninv * 2 * ∂ng_∂EdepsiE
    ∂ng_∂evg_1          =   kx_Fi_epsi_deps_E * 2 * ∂ng_∂EdepsiE
    ∂ng_∂kx_1           =  real(_outer(Fi_epsi_deps_E, evg)) * 2 * ∂ng_∂EdepsiE
    @tullio ∂ng_∂mag_1[ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j]  nograd=one_mone
    @tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] * one_mone[j]  nograd=one_mone
    ∂ng_∂k_1            =   ∇ₖmag_mn(∂ng_∂mag_1,∂ng_∂mn_1,mag,mn)
    ### end ∇⟨H|M̂(k,∂ε⁻¹/∂ω)|H⟩ ###

    ### ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    # om̄₁₂ = dot(H,H̄) / ω
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn_2[i,j,ix,iy] := mag[ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[i,j,ix,iy]   +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]  
    @tullio ∂ng_∂mag_2[ix,iy] := mn[a,b,ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[a,b,ix,iy]
    ∂ng_∂k_2 = ∇ₖmag_mn(real(∂ng_∂mag_2),real(∂ng_∂mn_2),mag,mn)
    ∂ng_∂evg_2 = ( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag) + ω*ct(𝓕⁻¹_H̄,mn) ) 
    ∂ng_∂ε⁻¹_2 = real( herm( _outer(Ē,D) ) ) 
    ### end ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###

    ### ∇solve_k ###
    k̄ = ∂ng_∂k_1 + ∂ng_∂k_2
    ∂ng_∂evg = vec(∂ng_∂evg_1) + vec(∂ng_∂evg_2)
    M̂ = HelmholtzMap(k,ε⁻¹,grid)
    P̂	= HelmholtzPreconditioner(M̂)
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		# x⃗
        0.0, 							# ᾱ
        ∂ng_∂evg;					    # x̄
        # λ⃗₀=λ⃗₀,
        P̂	= P̂,
    )
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    ∂ng_∂ε⁻¹_31 = ε⁻¹_bar(vec(D), vec( λd ) , size(grid)...) * Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )
    λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
    ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
    m̄_kx = real.( λẽ_sv .* conj.(view(evg,2,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,2,axes(grid)...)) )	#NB: m̄_kx and n̄_kx would actually
    n̄_kx =  -real.( λẽ_sv .* conj.(view(evg,1,axes(grid)...)) .+ ẽ_sv .* conj.(view(λ,1,axes(grid)...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
    māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
    k̄ₕ		= -∇ₖmag_m_n(
                māg_kx, 		# māg total
                m̄_kx.*mag, 	# m̄  total
                n̄_kx.*mag,	  	# n̄  total
                mag, m⃗, n⃗; 
                dk̂, # dk⃗ direction
            )
    λd2 = fft( kx_tc( ( (k̄ + k̄ₕ ) / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax ) * Ninv  # (2 * HMkH) = ∂ω²∂k
    ∂ng_∂ε⁻¹_32 = ε⁻¹_bar(vec(D), vec( λd2 ) , size(grid)...)
    ∂ng_∂ε⁻¹_3 = ∂ng_∂ε⁻¹_31 + ∂ng_∂ε⁻¹_32
    ∂ng_∂ω_2 =  ω * (k̄ + k̄ₕ ) / HMkH 
    ### end ∇solve_k ###
    ∂ng_∂ε              =   _dot( -ε⁻¹, (∂ng_∂ε⁻¹_1 + ∂ng_∂ε⁻¹_2 + ∂ng_∂ε⁻¹_3), ε⁻¹ )
    ∂ng_∂ω_3            =   dot( real(herm(∂ng_∂ε)), ∂ε_∂ω ) + dot( real(herm(∂ng_∂∂ε_∂ω)), ∂²ε_∂ω² )
    gvd                 =   ∂ng_∂ω_1 + ∂ng_∂ω_2 + ∂ng_∂ω_3     #  gvd = ∂ng/∂ω = ∂²|k|/∂ω²
    return E, ng, gvd
end

function ng_gvd_single2(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
    fftax               =   _fftaxes(grid)      
    evg                 =   reshape(ev,(2,size(grid)...))
    Ninv                =   inv(1.0 * length(grid))
    mag,mn              =   mag_mn(k,grid)
    local one_mone      =   [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
    HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
    inv_HMkH            =   inv(HMkH)
    deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
    epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
    Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
    ng                  =   (ω + EdepsiE/2) * inv_HMkH
    ∂ng_∂EdepsiE        =   inv_HMkH/2
    ∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2

    ### ∇⟨H|M̂(k,∂ε⁻¹/∂ω)|H⟩ ###
    ∂ng_∂kx_1           =  real(_outer(Fi_epsi_deps_E, evg)) * 2 * ∂ng_∂EdepsiE
    @tullio ∂ng_∂mag_1[ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j]  nograd=one_mone
    @tullio ∂ng_∂mn_1[i,j,ix,iy] := reverse(∂ng_∂kx_1,dims=2)[i,j,ix,iy] * mag[ix,iy] * one_mone[j]  nograd=one_mone
    
    ### ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn_2[i,j,ix,iy] := mag[ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[i,j,ix,iy]   +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]  
    @tullio ∂ng_∂mag_2[ix,iy] := mn[a,b,ix,iy] * real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)[a,b,ix,iy]

    ### ∇solve_k ###
    M̂ = HelmholtzMap(k,ε⁻¹,grid)
    P̂	= HelmholtzPreconditioner(M̂)
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		# x⃗
        0.0, 							# ᾱ
        # ∂ng_∂evg;					    # x̄
        (2 * ∂ng_∂EdepsiE) * vec(kx_Fi_epsi_deps_E) + vec(kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag)) + ω*vec(ct(𝓕⁻¹_H̄,mn));					    # x̄
        P̂	= P̂,
    )
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )
    ∂ng_∂kx_3           =  real(_outer(λẽ, evg)) + real(_outer(ẽ, λ))
    @tullio ∂ng_∂mag_3[ix,iy] := reverse(∂ng_∂kx_3,dims=2)[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j]  nograd=one_mone
    @tullio ∂ng_∂mn_3[i,j,ix,iy] := reverse(∂ng_∂kx_3,dims=2)[i,j,ix,iy] * mag[ix,iy] * one_mone[j]  nograd=one_mone
    k̄_tot		= ∇ₖmag_mn(
        real(∂ng_∂mag_1) + real(∂ng_∂mag_2) - real(∂ng_∂mag_3), 		# māg total
        real(∂ng_∂mn_1)  + real(∂ng_∂mn_2) - real(∂ng_∂mn_3),	  	# mn̄  total
        mag,
        mn,
    )
    λd2 = fft( kx_tc( ( k̄_tot / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax ) * Ninv  # (2 * HMkH) = ∂ω²∂k

    ### gvd = ∂ng/∂ω = ∂²|k|/∂ω² ###
    gvd  = dot(∂ε_∂ω,_dot( -ε⁻¹, real(herm( ( _outer(deps_E,D) * ( 2 * ∂ng_∂EdepsiE * Ninv )  +  _outer(Ē,D) +  ε⁻¹_bar(vec(D), vec( λd*Ninv + λd2 ) , size(grid)...) ) ) ), ε⁻¹ ) ) +
        (dot(∂²ε_∂ω²,real(herm(_outer(E,E)))) * ( ∂ng_∂EdepsiE * Ninv )) +
        ω * k̄_tot / HMkH +
        inv_HMkH

    return E, ng, gvd
end

function ng_gvd_single3(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
    fftax               =   _fftaxes(grid)      
    evg                 =   reshape(ev,(2,size(grid)...))
    Ninv                =   inv(1.0 * length(grid))
    mag,mn              =   mag_mn(k,grid)
    local one_mone      =   [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
    HMkH                =   -real( dot(evg , zx_ct( ifft( E, fftax ), mn  )  )  )
    inv_HMkH            =   inv(HMkH)
    deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
    epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
    Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
    ng                  =   (ω + EdepsiE/2) * inv_HMkH
    ∂ng_∂EdepsiE        =   inv_HMkH/2
    ∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2

    ### ∇⟨H| ∂/∂k M̂(k,ε⁻¹) |H⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)

    ### ∇solve_k ###
    M̂,P̂ = Zygote.ignore() do
        M̂ = HelmholtzMap(k,ε⁻¹,grid)
        P̂	= HelmholtzPreconditioner(M̂)
        return M̂,P̂
    end
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		# x⃗
        0.0, 							# ᾱ
        # ∂ng_∂evg;					    # x̄
        (2 * ∂ng_∂EdepsiE) * vec(kx_Fi_epsi_deps_E) + vec(kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag)) + ω*vec(ct(𝓕⁻¹_H̄,mn));					    # x̄
        P̂	= P̂,
    )
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )

    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] 
    ∂ng_∂kx           =  reverse( real(_outer( (2 * ∂ng_∂EdepsiE)*Fi_epsi_deps_E - λẽ, evg)) - real(_outer(ẽ, λ)) ,dims=2) + real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)
    @tullio ∂ng_∂mag[ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn[i,j,ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mag[ix,iy] * one_mone[j] +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]   nograd=one_mone
    k̄_tot		= ∇ₖmag_mn(
        real(∂ng_∂mag), 		# māg total
        real(∂ng_∂mn),	  	# mn̄  total
        mag,
        mn,
    )
    λd2 = fft( kx_tc( ( k̄_tot / ( 2 * HMkH ) ) * evg  , mn, mag ), fftax ) * Ninv  # (2 * HMkH) = ∂ω²∂k
    ### gvd = ∂ng/∂ω = ∂²|k|/∂ω² ###
    gvd  = dot(∂ε_∂ω,_dot( -ε⁻¹, real( _outer(  ( 2 * ∂ng_∂EdepsiE * Ninv ) * deps_E + Ē - (λd*Ninv + λd2) ,D)  ) , ε⁻¹ ) )  +
        (dot(∂²ε_∂ω²,real(herm(_outer(E,E)))) * ( ∂ng_∂EdepsiE * Ninv )) + ω * k̄_tot / HMkH + inv_HMkH
    return E, ng, gvd
end


E, ng, gvd1 = ng_gvd_single1(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid);
E, ng, gvd2 = ng_gvd_single2(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid);
E, ng, gvd3 = ng_gvd_single3(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid);
@show gvd1
@show gvd2
@show gvd3
@test gvd1 ≈ gvd2
@test gvd2 ≈ gvd3

ng4,gvd4,E4 = ng_gvd_E(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
@show gvd4
@show ng4
E5, ng5, gvd5 = ng_gvd_single3(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid);
@show gvd5
##
gvd31,gvd3_pb = Zygote.pullback(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²) do ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²
    ng_gvd_single3(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)[3]
end
gvd3_pb(1.0)

(neff32, ng32, gvd32, E32),gvd3_pb2 = Zygote.pullback(ω,p_geom) do ω,p_geom
    grid            =   Grid(6.0,4.0,256,256)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3,4),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    E, ng, gvd      =   ng_gvd_single3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1]/ω, ng, gvd, E
end;

neff32
ng32
gvd32
∇gvd32      =   gvd3_pb2((nothing,nothing,1.0,nothing))
∇ng32       =   gvd3_pb2((nothing,1.0,nothing,nothing))
∇neff32     =   gvd3_pb2((1.0,nothing,nothing,nothing))

(k, ng, gvd, E), k_ng_gvd_pb = Zygote.pullback(ω,p_geom) do ω,p_geom
    grid            =   Grid(6.0,4.0,256,256)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3,4),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd, E      =   ng_gvd_E(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd, E
end;
k
ng
gvd
∇gvd        =    k_ng_gvd_pb((nothing,nothing,1.0,nothing))
∇ng         =    k_ng_gvd_pb((nothing,1.0,nothing,nothing))
∇neff       =    k_ng_gvd_pb((1.0,nothing,nothing,nothing))


##
∇k_∇ng_∇gvd_FD = FiniteDiff.finite_difference_jacobian(vcat(ω,p_geom)) do ω_p
    ω               =   first(ω_p)
    p_geom          =   ω_p[2:5]
    grid            =   Grid(6.0,4.0,256,256)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3,4),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    _, ng, gvd      =   ng_gvd_single3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return [kmags[1], ng, gvd]
end

##
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



##
function test_neff_E(oms,p_geom;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=256,Ny=256,minds=(1,2,3,4))
    grid    =   Grid(Dx,Dy,Nx,Ny)
    nom     =   length(oms)
    neffs   =   Array{Float64}(undef,(nom,nev))
    Es      =   Array{ComplexF64}(undef,(3,size(grid)...,nom,nev))
    for (omidx,ω) in enumerate(oms)
        ε_data           =  smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),minds,grid)
        ε                =  copy(selectdim(ε_data,3,1));
        ∂ε_∂ω            =  copy(selectdim(ε_data,3,2));
        ε⁻¹              =  sliceinv_3x3(ε);
        kmags, evecs     =  solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        for bndidx in 1:nev
            neffs[omidx,bndidx]     =   kmags[bndidx]/ω
            Es[:,:,:,omidx,bndidx]  =   E⃗(kmags[bndidx],evecs[bndidx],ε⁻¹,∂ε_∂ω,grid;normalized=false);
        end
    end
    return neffs, Es
end

function test_neff_ng_E(oms,p_geom;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=256,Ny=256,minds=(1,2,3,4))
    grid    =   Grid(Dx,Dy,Nx,Ny)
    nom     =   length(oms)
    neffs   =   Array{Float64}(undef,(nom,nev))
    ngs     =   Array{Float64}(undef,(nom,nev))
    Es      =   Array{ComplexF64}(undef,(3,size(grid)...,nom,nev))
    for (omidx,ω) in enumerate(oms)
        ε_data           =  smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),minds,grid)
        ε                =  copy(selectdim(ε_data,3,1));
        ∂ε_∂ω            =  copy(selectdim(ε_data,3,2));
        ε⁻¹              =  sliceinv_3x3(ε);
        kmags, evecs     =  solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        for bndidx in 1:nev
            neffs[omidx,bndidx]     =   kmags[bndidx]/ω
            ngs[omidx,bndidx]       =   group_index(kmags[bndidx],evecs[bndidx],ω,ε⁻¹,∂ε_∂ω,grid)
            Es[:,:,:,omidx,bndidx]  =   E⃗(kmags[bndidx],evecs[bndidx],ε⁻¹,∂ε_∂ω,grid;normalized=false);
        end
    end
    return neffs, ngs, Es
end

function test_neff_ng_gvd_E(oms,p_geom;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=256,Ny=256,minds=(1,2,3,4))
    grid    =   Grid(Dx,Dy,Nx,Ny)
    nom     =   length(oms)
    neffs   =   Array{Float64}(undef,(nom,nev))
    ngs     =   Array{Float64}(undef,(nom,nev))
    gvds    =   Array{Float64}(undef,(nom,nev))
    Es      =   Array{ComplexF64}(undef,(3,size(grid)...,nom,nev))
    for (omidx,ω) in enumerate(oms)
        ε_data           =  smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),minds,grid)
        ε                =  copy(selectdim(ε_data,3,1));
        ∂ε_∂ω            =  copy(selectdim(ε_data,3,2));
        ∂²ε_∂ω²          =  copy(selectdim(ε_data,3,3));
        ε⁻¹              =  sliceinv_3x3(ε);
        kmags, evecs     =  solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        for bndidx in 1:nev
            k   =   kmags[bndidx]
            ev  =   evecs[bndidx]
            E, ng, gvd = ng_gvd_single1(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid);
            neffs[omidx,bndidx]     =   k/ω
            ngs[omidx,bndidx]       =   ng
            gvds[omidx,bndidx]      =   gvd
            Es[:,:,:,omidx,bndidx]  =   E
        end
    end
    return neffs, ngs, gvds, Es
end
##
# oms                     =  oms = collect(0.7:0.1:1.6);

p_geom                  =  [2.0,0.8,0.1,0.1];
## warmup
oms                     =  oms = collect(0.7:0.1:0.9);
neffs_Es                = test_neff_E(oms,p_geom;nev=2);
neffs_ngs_Es            = test_neff_ng_E(oms,p_geom;nev=2);
neffs, ngs, gvds, Es    = test_neff_ng_gvd_E(oms,p_geom;nev=2);
##
oms                             =   oms = collect(0.7:0.1:1.6);
@time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2);
@time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2);
@time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2);

"""
### SuperCloud (serial ω sweep) timings for length(oms)=10, nev=2, Nx=Ny=256, eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=256,Ny=256, eigensolver=IterativeSolversLOBPCG(), (SN-LN-slab-loaded-ridge)

### 128x128 grid

# julia> @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=128,Ny=128);
#  43.671565 seconds (14.86 M allocations: 20.155 GiB, 1.38% gc time, 0.83% compilation time)
# julia> @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=128,Ny=128);
#  43.779045 seconds (14.86 M allocations: 20.355 GiB, 1.17% gc time, 0.36% compilation time)
# julia> @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=128,Ny=128);
#  53.794012 seconds (15.63 M allocations: 24.616 GiB, 1.38% gc time, 0.39% compilation time)

### 256x256 grid

# julia> @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2);
# 178.743843 seconds (53.62 M allocations: 79.593 GiB, 1.21% gc time, 0.05% compilation time)
# julia> @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2);
# 178.666373 seconds (53.63 M allocations: 80.395 GiB, 1.76% gc time, 0.05% compilation time)
# julia> @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2);
# 235.750294 seconds (56.24 M allocations: 97.643 GiB, 1.47% gc time)

### 512x512 grid

# julia> @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# 831.623503 seconds (202.95 M allocations: 316.877 GiB, 1.99% gc time, 0.01% compilation time)
# julia> @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# 800.750017 seconds (202.96 M allocations: 320.082 GiB, 1.66% gc time)
# julia> @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# 1070.133483 seconds (213.68 M allocations: 382.723 GiB, 5.02% gc time, 0.02% compilation time)
"""


### 128x128 grid

## nω = 5
# oms                             =   oms = collect(0.7:0.2:1.6);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=128,Ny=128);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=128,Ny=128);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=128,Ny=128);

## nω = 10
oms                             =   oms = collect(0.7:0.1:1.6);
@time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=128,Ny=128);
@time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=128,Ny=128);
@time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=128,Ny=128);

## nω = 20
# oms                             =   oms = collect(0.7:0.05:1.65);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=128,Ny=128);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=128,Ny=128);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=128,Ny=128);


### 256x256 grid

## nω = 5
# oms                             =   oms = collect(0.7:0.2:1.6);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=256,Ny=256);

## nω = 10
# oms                             =   oms = collect(0.7:0.1:1.6);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=256,Ny=256);

## nω = 20
# oms                             =   oms = collect(0.7:0.05:1.65);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=256,Ny=256);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=256,Ny=256);


### 512x512 grid

## nω = 5
# oms                             =   oms = collect(0.7:0.2:1.6);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=512,Ny=512);

## nω = 10
oms                             =   oms = collect(0.7:0.1:1.6);
@time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=512,Ny=512);
@time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=512,Ny=512);
@time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=512,Ny=512);

## nω = 20
# oms                             =   oms = collect(0.7:0.05:1.65);
# @time   neffs_Es                =   test_neff_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# @time   neffs_ngs_Es            =   test_neff_ng_E(oms,p_geom;nev=2,Nx=512,Ny=512);
# @time   neffs, ngs, gvds, Es    =   test_neff_ng_gvd_E(oms,p_geom;nev=2,Nx=512,Ny=512);