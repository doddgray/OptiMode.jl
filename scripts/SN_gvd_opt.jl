"""
Functions for checking/debugging GVD calculation
"""

using LinearAlgebra, StaticArrays, FFTW, GeometryPrimitives, OptiMode, Tullio, Test
using ChainRules, Zygote, FiniteDifferences, ForwardDiff, FiniteDiff
using OptiMode: ∇ₖmag_m_n, ∇ₖmag_mn
# using CairoMakie

"""
    Si₃N₄_strip_wg(p)

Construct a parameterized Si₃N₄ strip waveguide with a rectangular core.

`p` should be a vector of real numbers specifying the following parameters:
    -   p[1]:   rectangular core width [μm]
    -   p[2]:   rectangular core thickness [μm]
"""
function Si₃N₄_strip_wg(p::AbstractVector{T};Δx=6.0,Δy=4.0) where {T<:Real}  
    w_core      =   p[1]    #   rectangular core width [μm]
    t_core      =   p[2]    #   rectangular core thickness [μm]
    edge_gap    =   0.5
    mat_core    =   1
    mat_subs    =   2
    # Δx          =   6.0
    # Δy          =   4.0
    t_subs      =   (Δy - t_core - edge_gap )/2.
    c_subs_y    =   -Δy/2. + edge_gap/2. + t_subs/2.
    ax = SMatrix{2,2,T}( [      1.     0.   ;   0.     1.      ] )
	core = GeometryPrimitives.Box( SVector{2,T}([0. , 0.]), SVector{2,T}([w_core, t_core]), ax, mat_core, )
	subs = GeometryPrimitives.Box( SVector{2,T}([0. , c_subs_y]), SVector{2,T}([Δx - edge_gap, t_subs ]),	ax,	mat_subs, )
	return (core, subs)
end

## choose a geometry function and initialize the corresponding
## material models
geom_fn             =   Si₃N₄_strip_wg
mats                =   [Si₃N₄,SiO₂,Vacuum];
mat_vars            =   (:ω,)
np_mats             =   length(mat_vars)
f_ε_mats, f_ε_mats! =   _f_ε_mats(mats,mat_vars) # # f_ε_mats, f_ε_mats! = _f_ε_mats(vcat(materials(sh1),Vacuum),(:ω,))

## Set geometry parameters `p`, grid & solver settings
ω               =   1.1 
p               =   [ω, 1.0, 0.7];
mat_vals1       =   f_ε_mats(p[1:np_mats]);
nev             =   2
eig_tol         =   1e-9
k_tol           =   1e-9
Dx              =   6.0
Dy              =   4.0
Nx              =   128
Ny              =   128
grid            =   Grid(Dx,Dy,Nx,Ny)
fftax           =   _fftaxes(grid)      # spatial grid axes of field arrays, 2:3 (2:4) for 2D (3D) using current field data format
Ngrid           =   length(grid)        # total number of grid points, Nx*Ny (Nx*Ny*Nz) in 2D (3D)
minds           =   (1,2,3)           # "material indices" for `shapes=geom1(p)`, the material of shape=geom1(p)[idx] is `mat[minds[idx]]`
                                        # TODO: automate determination of material indices, this is error-prone

## Calculate dielectric data using these parameters
ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); # TODO: automate unpacking of dielectric data into (ε, ∂ε_∂ω, ∂²ε_∂ω²)
ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);

# compare the group index calculated above with values of d|k|/dω calculated 
# via AD and finite differencing. they should all be equal
ng_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end

ng_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end |> first

@test ng_RM ≈ ng_FD rtol = 1e-7
@show ng_RM_vs_FD_err    = ng_RM - ng_FD

gvd_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end

gvd_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
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

function gvd_manual(p)
    ω               =   p[1]
    p_geom          =   p[2:end]
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return gvd
end
gvd_man1 = gvd_manual(p)
gvd_man2, gvd_manual_pb = Zygote.pullback(gvd_manual,p)
gvd_man2

function ng_manual(p)
    ω               =   p[1]
    p_geom          =   p[2:end]
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return ng
end
ng_man1 = ng_manual(p)
ng_man2, ng_manual_pb = Zygote.pullback(ng_manual,p)
ng_man2

##

function test_eig_adjt1(ω,k,ε⁻¹,ev,x̄)
    grid = Grid(6.0,4.0,128,128)
    (M̂,P̂) = Zygote.ignore() do 
        M̂ = HelmholtzMap(k,ε⁻¹,grid)
        P̂ = HelmholtzPreconditioner(M̂) 
        return M̂,P̂
    end
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		# x⃗
        0.0, 							# ᾱ
        x̄;	        				    # x̄
        P̂	= P̂,
    )
    return real(sum(λ⃗))
end

om = 1.1
kk = 2.2
ev = rand(ComplexF64,2*128*128);
x̄ = rand(ComplexF64,2*128*128);
tea1_out1 = test_eig_adjt1(om,kk,ε⁻¹,ev,x̄)
tea1_out2,tea1_pb = Zygote.pullback(test_eig_adjt1,om,kk,ε⁻¹,ev,x̄)
tea1_out1 ≈ tea1_out2

function test_eig_adjt2(ω,ε⁻¹,x̄)
    grid = Grid(6.0,4.0,128,128)
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev=2,eig_tol=1e-8,k_tol=1e-8)
    k = kmags[1]
    # (M̂,P̂) = Zygote.ignore() do 
    M̂ = HelmholtzMap(k,ε⁻¹,grid)
    # P̂ = HelmholtzPreconditioner(M̂) 
        # return M̂,P̂
    # end
    λ⃗	= eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        evecs[1], 					 		# x⃗
        0.0, 							# ᾱ
        x̄;	        				    # x̄
        P̂ = Zygote.ignore() do 
            return HelmholtzPreconditioner(M̂)
        end,
    )
    return real(sum(λ⃗))
end

om = 1.1
x̄ = rand(ComplexF64,2*128*128);
tea2_out1 = test_eig_adjt2(om,ε⁻¹,x̄)
tea2_out2,tea2_pb = Zygote.pullback(test_eig_adjt2,om,ε⁻¹,x̄)
dtea2_dom, dtea2_dei, dtea2_dxbar = tea2_pb(1.0)
tea2_out1 ≈ tea2_out2


##
ω = 1.1
x̄ = rand(ComplexF64,2*length(grid));
(kmags,evecs),solve_k_pb     =   Zygote.pullback((om,ei)->solve_k(om,ei,grid,IterativeSolversLOBPCG();nev=2,eig_tol=1e-8,k_tol=1e-8),ω,ε⁻¹)
k = kmags[1]
ev = evecs[1];
M̂, M̂_pb = Zygote.pullback(HelmholtzMap,k,ε⁻¹,grid)
# P̂ = HelmholtzPreconditioner(M̂) 
    # return M̂,P̂
# end
λ⃗, eig_adjt_pb	= Zygote.pullback(
    (M̂, ω, ev, x̄)->eig_adjt(
        M̂,								 # Â
        ω^2, 							# α
        ev, 					 		# x⃗
        0.0, 							# ᾱ
        x̄;	        				    # x̄
        P̂ = Zygote.ignore() do 
            return HelmholtzPreconditioner(M̂)
        end,
    ), M̂, ω, ev, x̄ )

M̂_bar, ω_bar, ev_bar, x̄_bar = eig_adjt_pb(ones(size(λ⃗)))
k_bar,ε⁻¹_bar1,grid_bar = M̂_pb(M̂_bar)

##
p = [1.1,1.0,0.7]
(k, ng, gvd), k_ng_gvd_pb = Zygote.pullback(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd
end;
k
ng
gvd
∇gvd_RM             =    k_ng_gvd_pb((nothing,nothing,1.0))[1]
∇ng_RM              =    k_ng_gvd_pb((nothing,1.0,nothing))[1]
∇k_RM               =    k_ng_gvd_pb((1.0,nothing,nothing))[1]
∂k_∂ω_RM, ∂k_∂w_RM, ∂k_∂t_RM = ∇k_RM
∂ng_∂ω_RM, ∂ng_∂w_RM, ∂ng_∂t_RM = ∇ng_RM
∂gvd_∂ω_RM, ∂gvd_∂w_RM, ∂gvd_∂t_RM = ∇gvd_RM
J_RM = [ ∇k_RM' ; ∇ng_RM' ; ∇gvd_RM'   ]

J_FD = FiniteDiff.finite_difference_jacobian(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return [kmags[1], ng, gvd]
end
#∇k_∇ng_∇gvd_FD

function ng_gvd1(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid;dk̂=SVector{3,Float64}(0.0,0.0,1.0),adj_tol=1e-8)
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
    M̂,P̂ = Zygote.ignore() do
        M̂ = HelmholtzMap(k,ε⁻¹,grid)
        P̂	= HelmholtzPreconditioner(M̂)
        return M̂, P̂
    end
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
    return ng, gvd
end

function ng_gvd2(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
    fftax               =   Zygote.ignore() do 
        _fftaxes(grid)
    end
    gridsize               =   Zygote.ignore() do 
        size(grid)
    end
    evg                 =   reshape(ev,(2,gridsize...))
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

    M̂ = Zygote.ignore() do 
        HelmholtzMap(k,ε⁻¹,grid)
    end
    P̂	= Zygote.ignore() do 
        HelmholtzPreconditioner(M̂)
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
    λ = reshape( λ⃗, (2,gridsize...) )
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
    gvd  = dot(∂ε_∂ω,_dot( -ε⁻¹, real( herm_back( _outer(deps_E,D) * ( 2 * ∂ng_∂EdepsiE * Ninv )  +  _outer(Ē,D) -  _outer(λd*Ninv + λd2, D  ) ) ), ε⁻¹ ) ) +
        (dot(∂²ε_∂ω²,real(herm_back(_outer(E,E)))) * ( ∂ng_∂EdepsiE * Ninv )) + ω * k̄_tot / HMkH + inv_HMkH

    return ng, gvd
end

function ng_gvd3(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
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
    return ng, gvd
end

function k_ng_gvd1(p)
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd1(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd
end
k11, ng11, gvd11 = k_ng_gvd1(p)
(k1, ng1, gvd1), k_ng_gvd_pb1 = Zygote.pullback(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd1(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd
end;
k1
ng1
gvd1
∇gvd1_RM             =    k_ng_gvd_pb1((nothing,nothing,1.0))[1]
∇ng1_RM              =    k_ng_gvd_pb1((nothing,1.0,nothing))[1]
∇k1_RM               =    k_ng_gvd_pb1((1.0,nothing,nothing))[1]
J1_RM = [ ∇k1_RM' ; ∇ng1_RM' ; ∇gvd1_RM'   ]


(k2, ng2, gvd2), k_ng_gvd_pb2 = Zygote.pullback(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd2(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd
end;
k2
ng2
gvd2
∇gvd2_RM             =    k_ng_gvd_pb2((nothing,nothing,1.0))[1]
∇ng2_RM              =    k_ng_gvd_pb2((nothing,1.0,nothing))[1]
∇k2_RM               =    k_ng_gvd_pb2((1.0,nothing,nothing))[1]
J2_RM = [ ∇k2_RM' ; ∇ng2_RM' ; ∇gvd2_RM'   ]

(k3, ng3, gvd3), k_ng_gvd_pb3 = Zygote.pullback(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return kmags[1], ng, gvd
end;
k3
ng3
gvd3
∇gvd3_RM             =    k_ng_gvd_pb3((nothing,nothing,1.0))[1]
∇ng3_RM              =    k_ng_gvd_pb3((nothing,1.0,nothing))[1]
∇k3_RM               =    k_ng_gvd_pb3((1.0,nothing,nothing))[1]
J3_RM = [ ∇k3_RM' ; ∇ng3_RM' ; ∇gvd3_RM'   ]
∂k_∂ω_RM, ∂k_∂w_RM, ∂k_∂t_RM = ∇k3_RM
∂ng_∂ω_RM, ∂ng_∂w_RM, ∂ng_∂t_RM = ∇ng3_RM
∂gvd_∂ω_RM, ∂gvd_∂w_RM, ∂gvd_∂t_RM = ∇gvd3_RM


J3_FD  = FiniteDiff.finite_difference_jacobian(p) do p
    ω               =   p[1]
    p_geom          =   p[2:end]
    grid            =   Grid(6.0,4.0,128,128)
    ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng, gvd         =   ng_gvd3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    return [kmags[1], ng, gvd]
end

# ∇k_FD,∇ng_FD,∇gvd_FD = collect(eachrow(∇k_∇ng_∇gvd_FD))
# ∂k_∂ω_FD, ∂k_∂w_FD, ∂k_∂tcore_FD, ∂k_∂θ_FD, ∂k_∂tslab_FD = ∇k_FD
# ∂ng_∂ω_FD, ∂ng_∂w_FD, ∂ng_∂tcore_FD, ∂ng_∂θ_FD, ∂ng_∂tslab_FD = ∇ng_FD
# ∂gvd_∂ω_FD, ∂gvd_∂w_FD, ∂gvd_∂tcore_FD, ∂gvd_∂θ_FD, ∂gvd_∂tslab_FD = ∇gvd_FD

3

# (k3, ng3, gvd3, E3),gvd3_pb = Zygote.pullback(ω,p_geom) do ω,p_geom
#     grid            =   Grid(6.0,4.0,128,128)
#     ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
#     ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
#     ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
#     ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
#     ε⁻¹             =   sliceinv_3x3(ε);
#     kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
#     E, ng, gvd      =   ng_gvd_single3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
#     return kmags[1], ng, gvd, E
# end;

# k3
# ng3
# gvd3
# ∇gvd3_RM            =   gvd3_pb((nothing,nothing,1.0,nothing))
# ∇ng3_RM             =   gvd3_pb((nothing,1.0,nothing,nothing))
# ∇k3_RM              =   gvd3_pb((1.0,nothing,nothing,nothing))
# ∂k3_∂ω_RM, ∂k3_∂w_RM, ∂k3_∂tcore_RM, ∂k3_∂θ_RM, ∂k3_∂tslab_RM = ∇k3_RM
# ∂ng3_∂ω_RM, ∂ng3_∂w_RM, ∂ng3_∂tcore_RM, ∂ng3_∂θ_RM, ∂ng3_∂tslab_RM = ∇ng3_RM
# ∂gvd3_∂ω_RM, ∂gvd3_∂w_RM, ∂gvd3_∂tcore_RM, ∂gvd3_∂θ_RM, ∂gvd3_∂tslab_RM = ∇gvd3_RM

# ∇k_∇ng_∇gvd3_FD = FiniteDiff.finite_difference_jacobian(vcat(ω,p_geom)) do ω_p
#     ω               =   first(ω_p)
#     p_geom          =   ω_p[2:5]
#     grid            =   Grid(6.0,4.0,128,128)
#     ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
#     ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
#     ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
#     ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
#     ε⁻¹             =   sliceinv_3x3(ε);
#     kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
#     _, ng, gvd      =   ng_gvd_single3(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
#     return [kmags[1], ng, gvd]
# end
# ∇k3_FD,∇ng3_FD,∇gvd3_FD = collect(eachrow(∇k_∇ng_∇gvd3_FD))
# ∂k3_∂ω_FD, ∂k3_∂w_FD, ∂k3_∂tcore_FD, ∂k3_∂θ_FD, ∂k3_∂tslab_FD = ∇k3_FD
# ∂ng3_∂ω_FD, ∂ng3_∂w_FD, ∂ng3_∂tcore_FD, ∂ng3_∂θ_FD, ∂ng3_∂tslab_FD = ∇ng3_FD
# ∂gvd3_∂ω_FD, ∂gvd3_∂w_FD, ∂gvd3_∂tcore_FD, ∂gvd3_∂θ_FD, ∂gvd3_∂tslab_FD = ∇gvd3_FD

##




























##

function Si₃N₄_strip_wg_ng_gvd(p;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=128,Ny=128,minds=(1,2,3),pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    ω                               =   p[1] 
    p_geom                          =   p[2:end];
    grid            =   Grid(Dx,Dy,Nx,Ny)     
    ε_data          =   smooth_ε(Si₃N₄_strip_wg(p_geom),f_ε_mats(p[1:1]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol);
    # eigind = Zygote.ignore() do
    #     E_fields    =   [E⃗(kk,ev,ε⁻¹,∂ε_∂ω,grid; canonicalize=true, normalized=false) for (kk,ev) in zip(kmags,evecs)]
    #     return mode_idx(E_fields,ε;pol_idx,mode_order,rel_amp_min)
    # end
    # E_fields    =   [E⃗(kk,ev,ε⁻¹,∂ε_∂ω,grid; canonicalize=true, normalized=false) for (kk,ev) in zip(kmags,evecs)]
    # eigind = mode_idx(E_fields,ε;pol_idx,mode_order,rel_amp_min)
    eigind = 1
    return ng_gvd(ω,kmags[eigind],evecs[eigind],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
    # return gvd
end
function Si₃N₄_strip_wg_ng(p;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=128,Ny=128,minds=(1,2,3),pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    return first(Si₃N₄_strip_wg_ng_gvd(p;nev,eig_tol,k_tol,Dx,Dy,Nx,Ny,minds,pol_idx,mode_order,rel_amp_min))
end
function Si₃N₄_strip_wg_gvd(p;nev=2,eig_tol=1e-9,k_tol=1e-9,Dx=6.0,Dy=4.0,Nx=128,Ny=128,minds=(1,2,3),pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    return last(Si₃N₄_strip_wg_ng_gvd(p;nev,eig_tol,k_tol,Dx,Dy,Nx,Ny,minds,pol_idx,mode_order,rel_amp_min))
end

Si₃N₄_strip_wg_gvd([1.1,1.0,0.7])
#Zygote.withgradient(Si₃N₄_strip_wg_gvd,[1.1,1.0,0.7])
Zygote.gradient(Si₃N₄_strip_wg_gvd,[0.8,1.0,0.7])[1]
FiniteDiff.FiniteDiff.finite_difference_gradient(Si₃N₄_strip_wg_gvd,[0.8,1.0,0.7])
##
p = [1.1,1.0,0.7]
ω                               =   p[1] 
p_geom                          =   p[2:end];
grid            =   Grid(Dx,Dy,Nx,Ny)     
ε_data          =   smooth_ε(Si₃N₄_strip_wg(p_geom),f_ε_mats(p[1:1]),minds,grid); 
ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,3,:,:] 
ε⁻¹             =   sliceinv_3x3(ε);
kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol);
# eigind = Zygote.ignore() do
#     E_fields    =   [E⃗(kk,ev,ε⁻¹,∂ε_∂ω,grid; canonicalize=true, normalized=false) for (kk,ev) in zip(kmags,evecs)]
#     return mode_idx(E_fields,ε;pol_idx,mode_order,rel_amp_min)
# end
E_fields    =   [E⃗(kk,ev,ε⁻¹,∂ε_∂ω,grid; canonicalize=true, normalized=false) for (kk,ev) in zip(kmags,evecs)]
eigind = mode_idx(E_fields,ε;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
# eigind = 1
ng1,gvd1 = ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
ng2,gvd2 = ng_gvd(ω,kmags[2],evecs[2],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
ng1_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end
ng1_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[1]
end |> first
gvd1_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end
gvd1_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[1],evecs[1],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end |> first
ng2_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[2]
end
ng2_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    return kmags[2]
end |> first
gvd2_FD = FiniteDiff.finite_difference_derivative(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[2],evecs[2],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end
gvd2_RM = Zygote.gradient(ω) do ω
    ε_data          =   smooth_ε(geom_fn(p[2:end]),f_ε_mats([ω,]),minds,grid); 
    ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
    ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:] 
    ε⁻¹             =   sliceinv_3x3(ε);
    kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
    ng              =   group_index(kmags[2],evecs[2],ω,ε⁻¹,∂ε_∂ω,grid)
    return ng
end |> first

function ∇ng_∇gvd(p)
    (ng1, gvd1, ng2, gvd2), ng_gvd_pb = Zygote.pullback(p) do p
        ω               =   first(p)
        p_geom          =   p[2:end]
        grid            =   Grid(6.0,4.0,256,256)
        ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
        ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        ng1, gvd1         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
        ng2, gvd2         =   ng_gvd(ω,kmags[2],evecs[2],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
        return ng1, gvd1, ng2, gvd2
    end;
    ∇gvd1_RM             =    ng_gvd_pb((nothing,1.0,nothing,nothing))
    ∇ng1_RM              =    ng_gvd_pb((1.0,nothing,nothing,nothing))
    ∇gvd2_RM             =    ng_gvd_pb((nothing,nothing,nothing,1.0))
    ∇ng2_RM              =    ng_gvd_pb((nothing,nothing,1.0,nothing))

    ∇ng_∇gvd_FD = FiniteDiff.finite_difference_jacobian(p) do p
        ω               =   first(p)
        p_geom          =   p[2:end]
        grid            =   Grid(6.0,4.0,256,256)
        ε_data          =   smooth_ε(geom_fn(p_geom),f_ε_mats([ω,]),(1,2,3),grid); 
        ε               =   copy(selectdim(ε_data,3,1)); # ε_data[:,:,1,:,:]  
        ∂ε_∂ω           =   copy(selectdim(ε_data,3,2)); # ε_data[:,:,2,:,:]
        ∂²ε_∂ω²         =   copy(selectdim(ε_data,3,3)); # ε_data[:,:,2,:,:]
        ε⁻¹             =   sliceinv_3x3(ε);
        kmags,evecs     =   solve_k(ω,ε⁻¹,grid,IterativeSolversLOBPCG();nev,eig_tol,k_tol)
        ng1, gvd1         =   ng_gvd(ω,kmags[1],evecs[1],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
        ng2, gvd2         =   ng_gvd(ω,kmags[2],evecs[2],ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)
        return [ng1, gvd1, ng2, gvd2]
    end
    ∇ng1_FD,∇gvd1_FD,∇ng2_FD,∇gvd2_FD = collect(eachrow(∇ng_∇gvd_FD))
    return ng1, gvd1, ng2, gvd2, hcat(vec(∇ng1_RM),vec(∇gvd1_RM),vec(∇ng2_RM),vec(∇gvd2_RM)), hcat(vec(∇ng1_FD),vec(∇gvd1_FD),vec(∇ng2_FD),vec(∇gvd2_FD)) 
end
ng1, gvd1, ng2, gvd2, ∇s_RM, ∇s_FD =  ∇ng_∇gvd(p)

let fig = Figure(resolution=(600,1200))
    ax1 = fig[1,1] = Axis(fig)
    ax2 = fig[2,1] = Axis(fig)
    ax3 = fig[3,1] = Axis(fig)
    
    hm1     =   heatmap!(ax1,x(grid),y(grid),ε[1,1,:,:])
    cbar1   =   Colorbar(fig[1,2], hm1,  width=20, height=350 )

    Z2      =   real(E_fields[1][1,:,:])
    magmax2 =   maximum(abs,Z2)
    hm2     =   heatmap!(ax2,x(grid),y(grid),Z2,colormap=:diverging_bkr_55_10_c35_n256,colorrange=(-magmax2,magmax2))
    cbar2   =   Colorbar(fig[2,2], hm2,  width=20, height=350 )

    Z3      =   real(E_fields[2][2,:,:])
    magmax3 =   maximum(abs,Z3)
    hm3     =   heatmap!(ax3,x(grid),y(grid),Z3,colormap=:diverging_bkr_55_10_c35_n256,colorrange=(-magmax3,magmax3))
    cbar3   =   Colorbar(fig[3,2], hm3,  width=20, height=350 )

    ax1.aspect = DataAspect()
    ax2.aspect = DataAspect()
    ax3.aspect = DataAspect()
    fig
end

##
grad = [0.0,0.0]
gvd, gvd_pb = Zygote.pullback(x->Si₃N₄_strip_wg_gvd(vcat(1.1,x)),[1.0,0.7])
grad[1:2] = gvd_pb(1.0)[1]
gvd
grad
##
using NLopt
function objective_fn(p,grad)
    gvd, gvd_pb = Zygote.pullback(x->Si₃N₄_strip_wg_gvd(vcat(1.2,x)),p)
    grad[1:2] = gvd_pb(1.0)[1]
    return abs2(gvd)
end
grad = [0.0,0.0]
objective_fn([1.0,0.7],grad)

opt = Opt(:LD_MMA,2)
opt.min_objective   =   objective_fn
opt.lower_bounds    =   [0.3, 0.2]
opt.upper_bounds    =   [2.5, 1.0]
opt.xtol_abs        =   0.01
opt.maxtime         =   2000
minf,minx,ret       =   optimize(opt,[1.0,0.7])
ret

##
grad = [0.0,0.0]
@time objective_fn([1.0,0.7],grad) # 98.334424 seconds (1.33 G allocations: 112.416 GiB, 8.07% gc time)
##
opt = Opt(:LD_LBFGS,2)
opt.min_objective   =   objective_fn
opt.lower_bounds    =   [0.3, 0.2]
opt.upper_bounds    =   [2.5, 1.0]
opt.xtol_abs        =   0.01
opt.maxtime         =   1000
@time minf,minx,ret       =   optimize(opt,[1.0,0.7])
ret

##











##
oms = collect(0.7:0.1:1.6)
gvd_data_check2_oms = map(oo->gvd_check2(vcat(oo,p[2:end])),oms)  # ran in 15min or so on SC, output pasted below
# gvd_data_check2_oms = [
#     (2.3452384696701603, -0.06836404359149875, 2.3452383428812027, -0.06836375594139099, 2.345238470688853, -0.06836477684340483, 2.345238470688853, -0.0683647768433997, 2.345238470688853, -0.06836404359149872),
#     (2.3419646344325478, -0.00015313129612798404, 2.341964513063431, -0.00015336275100708008, 2.34196463443484, -0.00015362675051179553, 2.34196463443484, -0.00015362675050045738, 2.34196463443484, -0.0001531312961282616),
#     (2.344738342130007, 0.05383841549444145, 2.3447377532720566, 0.05383896827697754, 2.3447383413277447, 0.05383804211082022, 2.3447383413277447, 0.05383804211083687, 2.3447383413277447, 0.05383841549444135),
#     (2.3524563224407564, 0.09947696760715216, 2.352456510066986, 0.099476158618927, 2.352456320958441, 0.09947666498112047, 2.352456320958441, 0.0994766649810992, 2.352456320958441, 0.0994769676071523),
#     (2.3644757208316465, 0.14033480981244004, 2.3644763231277466, 0.14033371751958673, 2.3644757185313865, 0.14033455221010016, 2.3644757185313865, 0.1403345522101378, 2.3644757185313865, 0.14033480981244018),
#     (2.3804417454062397, 0.17872379473945726, 2.3804418742656708, 0.1787236084540685, 2.380441742210411, 0.17872356788068025, 2.380441742210411, 0.1787235678806507, 2.380441742210411, 0.17872379473945732),
#     (2.4001921253905008, 0.2162492402059205, 2.400191930624155, 0.21624739353473368, 2.4001921212014623, 0.21624903547078878, 2.4001921212014623, 0.21624903547079155, 2.4001921212014623, 0.2162492402059204),
#     (2.4237032317379326, 0.2541166692148352, 2.423703649214336, 0.25411471724510193, 2.4237032264366865, 0.2541164810212212, 2.4237032264366865, 0.2541164810212222, 2.4237032264366865, 0.25411666921483533),
#     (2.451059807048243, 0.2933109686667302, 2.4510596990585327, 0.2933049400647481, 2.451059800492258, 0.29331079317893727, 2.451059800492258, 0.29331079317893427, 2.451059800492258, 0.2933109686667301),
#     (2.4824388941458135, 0.3347082212711727, 2.482438702136278, 0.33474817872047424, 2.4824388861733544, 0.33470805571719486, 2.4824388861733544, 0.33470805571719175, 2.4824388861733544, 0.3347082212711727),
# ]
ng_FD_check2_oms            =   [xx[3] for xx in gvd_data_check2_oms]; 
gvd_FD_check2_oms           =   [xx[4] for xx in gvd_data_check2_oms]; 
ng_check2_oms               =   [xx[1] for xx in gvd_data_check2_oms]; 
gvd_check2_oms              =   [xx[2] for xx in gvd_data_check2_oms]; 

ng1_check2_oms              =   [xx[5] for xx in gvd_data_check2_oms]; 
gvd1_check2_oms             =   [xx[6] for xx in gvd_data_check2_oms]; 
ng2_check2_oms              =   [xx[7] for xx in gvd_data_check2_oms]; 
gvd2_check2_oms             =   [xx[8] for xx in gvd_data_check2_oms]; 
ng3_check2_oms              =   [xx[9] for xx in gvd_data_check2_oms]; 
gvd3_check2_oms             =   [xx[10] for xx in gvd_data_check2_oms]; 



let fig = Figure(), oms=oms, ng_data=(ng_FD_check2_oms,ng_check2_oms), gvd_data=(gvd_FD_check2_oms,gvd_check2_oms)
    λs = inv.(oms)
    ax_ng   = fig[1,1] = Axis(fig, xlabel = "vacuum wavelength (μm)", ylabel = "ng", rightspinevisible = false,  yminorticksvisible = false)
    ax_gvd  = fig[2,1] = Axis(fig, xlabel = "vacuum wavelength (μm)", ylabel = "gvd", rightspinevisible = false,  yminorticksvisible = false)
    ax_ng_err       =   fig[1,1]    =   Axis(fig, yaxisposition = :right, yscale = log10, ylabel = "ng_err", xgridstyle = :dash, ygridstyle = :dash, yminorticksvisible = true)
    ax_gvd_err      =   fig[2,1]    =   Axis(fig, yaxisposition = :right, yscale = log10, ylabel = "gvd_err", xgridstyle = :dash, ygridstyle = :dash, yminorticksvisible = true)

    plt_fns         =   [scatter!,scatterlines!]
    colors          =   [:red,:black]
    labels          =   ["FD","manual"]
    markersizes     =   [16,12]
    markers         =   [:circle,:cross]

    plts_ng = [pltfn(ax_ng,λs,ngs,color=clr,label=lbl,marker=mrkr,markersize=mrkrsz) for (pltfn,ngs,clr,lbl,mrkr,mrkrsz) in zip(plt_fns,ng_data,colors,labels,markers,markersizes)]
    plts_gvd = [pltfn(ax_gvd,λs,gvds,color=clr,label=lbl,marker=mrkr,markersize=mrkrsz) for (pltfn,gvds,clr,lbl,mrkr,mrkrsz) in zip(plt_fns,gvd_data,colors,labels,markers,markersizes)]

    # ng_err_data         =   [abs.(ng_data[1] .- ng_data[2]), ]
    # gvd_err_data        =   [abs.(gvd_data[1] .- gvd_data[2]), ]

    # plt_fns_err         =   [lines!,]
    # colors_err          =   [:blue,]
    # labels_err          =   ["|FD-manual|",]
    # markersizes_err     =   [16,]
    # markers_err         =   [:square,]

    ng_err_data         =   [abs.(ng_data[1] .- ng_data[2]), abs.(ng_data[1] .- ng1_check2_oms), abs.(ng_data[1] .- ng2_check2_oms), abs.(ng_data[1] .- ng3_check2_oms) ]
    gvd_err_data        =   [abs.(gvd_data[1] .- gvd_data[2]) , abs.(gvd_data[1] .- gvd1_check2_oms) .+ eps(1.1), abs.(gvd_data[1] .- gvd2_check2_oms) .+ eps(1.1), abs.(gvd_data[1] .- gvd3_check2_oms) .+ eps(1.1) ]

    plt_fns_err         =   [lines!,lines!,lines!,lines!,]
    colors_err          =   [:blue,:red,:green,:black]
    labels_err          =   ["|FD-manual|","|FD-ng_gvd1|","|FD-ng_gvd2|","|FD-ng_gvd3|"]
    markersizes_err     =   [16,16,16,16,]
    markers_err         =   [:square,:square,:square,:square,]

    plts_ng_err = [pltfn(ax_ng_err,λs,ng_errs,color=clr,label=lbl) for (pltfn,ng_errs,clr,lbl,mrkr,mrkrsz) in zip(plt_fns_err,ng_err_data,colors_err,labels_err,markers_err,markersizes_err)]
    plts_gvd_err = [pltfn(ax_gvd_err,λs,gvd_errs,color=clr,label=lbl) for (pltfn,gvd_errs  ,clr,lbl,mrkr,mrkrsz) in zip(plt_fns_err,gvd_err_data,colors_err,labels_err,markers_err,markersizes_err)]

    Legend(fig[1, 2], ax_gvd, merge = true)
    Legend(fig[2, 2], ax_gvd_err, merge = true)
    return fig
end
