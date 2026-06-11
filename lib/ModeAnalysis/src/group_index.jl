export ng_gvd_E, ng_gvd, group_index

"""
######################################################################################
#
#		methods for calculating the group index from a mode field
#
######################################################################################
"""

"""
	`group_index(k::Real,evec,ω::Real,ε⁻¹,∂ε_∂ω,grid)`

Calculate the modal group index `ng = d|k|/dω` from the 
wavevector magnitude `k`, Helmholtz eigenvector `evec`, frequency `ω`, smoothed
inverse dielectric tensor and first-order dispersion `ε⁻¹` and `∂ε_∂ω`, 
and the corresponding spatial `grid<:Grid`.

This function should be compatible with reverse-mode auto-differentiation.
"""
group_index(k::Real,evec,ω,ε⁻¹,∂ε_∂ω,grid) = _group_index_kernel(k,evec,ω,ε⁻¹,∂ε_∂ω,grid)

# raw differentiable program backing `group_index`; kept separate so that the
# ChainRulesCore.rrule for `group_index` can be built with Zygote without recursing
function _group_index_kernel(k::Real,evec,ω,ε⁻¹,∂ε_∂ω,grid)
    mag,mn = mag_mn(k,grid) 
	return (ω + HMH(vec(evec), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(evec),ε⁻¹,mag,mn)
	# note that this formula assumes (HMH(...), HMₖH(...))>0 (positive eigenvalues)
end


"""
	ng_gvd_E(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)

Calculate the modal group index `ng`, group velocity dispersion `gvd` and real-space electric-field `E` for a single mode solution at frequency `ω`.
The mode solution is input as a wavenumber `k` and eigenvector `ev`, as retured by `solve_k(ω,ε⁻¹,...)`. 

The modal group index `ng` = ∂|k|/∂ω is calculated directly from the mode field and smoothed dielectric dispersion `∂ε_∂ω`.

The modal group velocity dispersion `gvd` = ∂ng/∂ω = ∂²|k|/∂ω² is calculated by solving the adjoint problem for the eigenmode solution.

The electric field `E` is calculated along the way and is frequently useful, so it is returned as well.
"""
function ng_gvd_E(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
    fftax               =   _fftaxes(grid)      
    evg                 =   reshape(ev,(2,size(grid)...))					# eigenvector, reshaped to (2,size(grid)...)
    Ninv                =   inv(1.0 * length(grid))
    mag,mn              =   mag_mn(k,grid)
    local one_mone      =   [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
    inv_HMkH            =   inv( -real( dot(evg , zx_ct( ifft( E, fftax ), mn ) ) ) )	# ⟨ev|∂M̂/∂k|ev⟩⁻¹ = dω²/dk 
    deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
    epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
    Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
    ng                  =   (ω + EdepsiE/2) * inv_HMkH						# modal group index, ng = d|k|/dω = ( 2ω + ⟨E|∂ε/∂ω|E⟩ ) / 2⟨ev|∂M̂/∂k|ev⟩ = (Energy density) / (Poynting flux)
    ∂ng_∂EdepsiE        =   inv_HMkH/2
    ∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2
    ### ∇⟨ev|∂M̂/∂k|ev⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    ### ∇solve_k ###
    M̂,P̂ = ignore_derivatives() do
        M̂ = HelmholtzMap(k,ε⁻¹,grid)
        P̂	= HelmholtzPreconditioner(M̂)
        return M̂,P̂
    end
    λ⃗	= eig_adjt(
        M̂,								 																		 # Â : operator or Matrix for which Â⋅x⃗ = αx⃗, here Â is the Helmholtz Operator M̂ = [∇× ε⁻¹ ∇×]
        ω^2, 																									# α	: primal eigenvalue, here α = ω²
        ev, 					 																				# x⃗ : primal eigenvector, here x⃗ = `ev` is the magnetic field in a plane wave basis (transverse polarization only) 
        0.0, 																									# ᾱ : sensitivity w.r.t eigenvalue, here this is always zero for ∇solve_k adjoint
        (2 * ∂ng_∂EdepsiE) * vec(kx_Fi_epsi_deps_E) + vec(kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag)) + ω*vec(ct(𝓕⁻¹_H̄,mn));		# x̄ : sensitivity w.r.t eigenvector, here x̄ =	∂ng/∂ev = ∂ng/∂⟨E|∂ε/∂ω|E⟩ * ∂⟨E|∂ε/∂ω|E⟩/∂ev + ∂ng/∂⟨ev|∂M̂/∂k|ev⟩ * ∂⟨ev|∂M̂/∂k|ev⟩/∂ev 
        P̂=P̂,																									  # P̂ : left preconditioner, here a cheaper-to-compute approximation of M̂⁻¹
    )
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] 
    ∂ng_∂kx           =  reverse( real(_outer( (2 * ∂ng_∂EdepsiE)*Fi_epsi_deps_E - λẽ, evg)) - real(_outer(ẽ, λ)) ,dims=2) + real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)
    @tullio ∂ng_∂mag[ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn[i,j,ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mag[ix,iy] * one_mone[j] +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]   nograd=one_mone
    ∂ng_∂k	=	∇ₖmag_mn(real(∂ng_∂mag),real(∂ng_∂mn),mag,mn)
    gvd  =	( ∂ng_∂EdepsiE * Ninv ) * dot( ∂²ε_∂ω², real(herm(_outer(E,E))) ) + inv_HMkH * ( ω * ∂ng_∂k + 1.0 ) -
		dot( 
			∂ε_∂ω,
			_dot( 
				ε⁻¹, 
				real( _outer(  ( 2 * ∂ng_∂EdepsiE * Ninv ) * deps_E + Ē - Ninv*(λd + fft( kx_tc( ( ∂ng_∂k * inv_HMkH/2 ) * evg  , mn, mag ), fftax ) ), D ) ),
				ε⁻¹,
			)
		)
    return real(ng), real(gvd), E
end

"""
	ng_gvd(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid)

Calculate the modal group index `ng`, group velocity dispersion `gvd` for a single mode solution at frequency `ω`.
The mode solution is input as a wavenumber `k` and eigenvector `ev`, as retured by `solve_k(ω,ε⁻¹,...)`. 

The modal group index `ng` = ∂|k|/∂ω is calculated directly from the mode field and smoothed dielectric dispersion `∂ε_∂ω`.

The modal group velocity dispersion `gvd` = ∂ng/∂ω = ∂²|k|/∂ω² is calculated by solving the adjoint problem for the eigenmode solution.
"""
function ng_gvd(ω,k,ev,ε⁻¹,∂ε_∂ω,∂²ε_∂ω²,grid::Grid{2,T};dk̂=SVector{3,T}(0.0,0.0,1.0),adj_tol=1e-8) where T<:Real
    fftax               =   _fftaxes(grid)      
    evg                 =   reshape(ev,(2,size(grid)...))					# eigenvector, reshaped to (2,size(grid)...)
    Ninv                =   inv(1.0 * length(grid))
    mag,mn              =   mag_mn(k,grid)
    local one_mone      =   [1.0, -1.0]
    D                   =   fft( kx_tc(evg,mn,mag), fftax )
    E                   =   _dot(ε⁻¹, D) #ε⁻¹_dot(D, ε⁻¹)
    H                   =   ω * fft( tc(evg,mn), fftax )
    inv_HMkH            =   inv( -real( dot(evg , zx_ct( ifft( E, fftax ), mn ) ) ) )	# ⟨ev|∂M̂/∂k|ev⟩⁻¹ = dω²/dk 
    deps_E              =   _dot(∂ε_∂ω,E)                                   # (∂ε/∂ω)|E⟩
    epsi_deps_E         =   _dot(ε⁻¹,deps_E)                                # (ε⁻¹)(∂ε/∂ω)|E⟩ = (∂(ε⁻¹)/∂ω)|D⟩
    Fi_epsi_deps_E      =   ifft( epsi_deps_E, fftax )                      # 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    kx_Fi_epsi_deps_E   =   kx_ct( Fi_epsi_deps_E , mn, mag  )              # [(k⃗+g⃗)×]cₜ ⋅ 𝓕⁻¹ ⋅ (ε⁻¹)(∂ε/∂ω)|E⟩
    EdepsiE             =   real( dot(evg,kx_Fi_epsi_deps_E) )              # ⟨E|∂ε/∂ω|E⟩ = ⟨D|∂(ε⁻¹)/∂ω|D⟩
    ng                  =   (ω + EdepsiE/2) * inv_HMkH						# modal group index, ng = d|k|/dω = ( 2ω + ⟨E|∂ε/∂ω|E⟩ ) / 2⟨ev|∂M̂/∂k|ev⟩ = (Energy density) / (Poynting flux)
    ∂ng_∂EdepsiE        =   inv_HMkH/2
    ∂ng_∂HMkH           =   -(ω + EdepsiE/2) * inv_HMkH^2
    ### ∇⟨ev|∂M̂/∂k|ev⟩ ###
    H̄ =  _cross(dk̂, E) * ∂ng_∂HMkH * Ninv / ω 
    Ē =  _cross(H,dk̂)  * ∂ng_∂HMkH * Ninv / ω 
    𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),fftax)
    𝓕⁻¹_H̄ = bfft( H̄ ,fftax)
    ### ∇solve_k ###
    M̂,P̂ = ignore_derivatives() do
        M̂ = HelmholtzMap(k,ε⁻¹,grid)
        P̂	= HelmholtzPreconditioner(M̂)
        return M̂,P̂
    end
    λ⃗	= eig_adjt(
        M̂,								 																		 # Â : operator or Matrix for which Â⋅x⃗ = αx⃗, here Â is the Helmholtz Operator M̂ = [∇× ε⁻¹ ∇×]
        ω^2, 																									# α	: primal eigenvalue, here α = ω²
        ev, 					 																				# x⃗ : primal eigenvector, here x⃗ = `ev` is the magnetic field in a plane wave basis (transverse polarization only) 
        0.0, 																									# ᾱ : sensitivity w.r.t eigenvalue, here this is always zero for ∇solve_k adjoint
        (2 * ∂ng_∂EdepsiE) * vec(kx_Fi_epsi_deps_E) + vec(kx_ct(𝓕⁻¹_ε⁻¹_Ē,mn,mag)) + ω*vec(ct(𝓕⁻¹_H̄,mn));		# x̄ : sensitivity w.r.t eigenvector, here x̄ =	∂ng/∂ev = ∂ng/∂⟨E|∂ε/∂ω|E⟩ * ∂⟨E|∂ε/∂ω|E⟩/∂ev + ∂ng/∂⟨ev|∂M̂/∂k|ev⟩ * ∂⟨ev|∂M̂/∂k|ev⟩/∂ev 
        P̂=P̂,																									  # P̂ : left preconditioner, here a cheaper-to-compute approximation of M̂⁻¹
    )
    λ = reshape( λ⃗, (2,size(grid)...) )
    λd = fft( kx_tc( λ, mn, mag ), fftax ) #* Ninv
    λẽ  =   ifft( _dot( ε⁻¹, λd ), fftax ) 
    ẽ 	 =   ifft( E, fftax )
    @tullio 𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ[i,j,ix,iy] :=  conj(𝓕⁻¹_ε⁻¹_Ē)[i,ix,iy] * reverse(evg;dims=1)[j,ix,iy] 
    ∂ng_∂kx           =  reverse( real(_outer( (2 * ∂ng_∂EdepsiE)*Fi_epsi_deps_E - λẽ, evg)) - real(_outer(ẽ, λ)) ,dims=2) + real(𝓕⁻¹_ε⁻¹_Ē_x_evgᵀ)
    @tullio ∂ng_∂mag[ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mn[i,j,ix,iy] * one_mone[j] nograd=one_mone
    @tullio ∂ng_∂mn[i,j,ix,iy] :=  ∂ng_∂kx[i,j,ix,iy] * mag[ix,iy] * one_mone[j] +   ω*real(_outer(𝓕⁻¹_H̄,evg))[i,j,ix,iy]   nograd=one_mone
    ∂ng_∂k	=	∇ₖmag_mn(real(∂ng_∂mag),real(∂ng_∂mn),mag,mn)
    gvd  =	( ∂ng_∂EdepsiE * Ninv ) * dot( ∂²ε_∂ω², real(herm(_outer(E,E))) ) + inv_HMkH * ( ω * ∂ng_∂k + 1.0 ) -
		dot( 
			∂ε_∂ω,
			_dot( 
				ε⁻¹, 
				real( _outer(  ( 2 * ∂ng_∂EdepsiE * Ninv ) * deps_E + Ē - Ninv*(λd + fft( kx_tc( ( ∂ng_∂k * inv_HMkH/2 ) * evg  , mn, mag ), fftax ) ), D ) ),
				ε⁻¹,
			)
		)
    return [real(ng), real(gvd)]
end

