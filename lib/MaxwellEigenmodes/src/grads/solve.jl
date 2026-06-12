#### AD Rules for Iterative eigensolves of Helmholtz Operator
using LinearAlgebra, StaticArrays, Tullio, ChainRulesCore, LinearMaps, IterativeSolvers
using LinearMaps: ⊗
using IterativeSolvers: gmres, lobpcg, lobpcg!

export ε⁻¹_bar, herm, herm_back, eig_adjt, my_linsolve, solve_adj!

# export ε⁻¹_bar!, ε⁻¹_bar, ∂ω²∂k_adj, Mₖᵀ_plus_Mₖ, ∂²ω²∂k², herm,
#      ∇ₖmag_m_n, ∇HMₖH, ∇M̂, ∇solve_k, ∇solve_k!, solve_adj!, 
#      neff_ng_gvd, ∂ε⁻¹_∂ω, ∂nng⁻¹_∂ω, ∇ₖmag_mn


"""
	herm(A) converts backpropagating gradients of nested Hermitian arrays/tensors to be Hermitian
"""
function herm(A::AbstractArray{TA,N}) where {TA<:AbstractMatrix,N}
	(real.(A) .+ transpose.(real.(A)) ) ./ 2
end

function herm(A::AbstractArray{T,4}) where {T<:Number}
	0.5 * real(A + conj(permutedims(A,(2,1,3,4))))
end

function herm(A::AbstractArray{T,5}) where {T<:Number}
	0.5 * real(A + conj(permutedims(A,(2,1,3,4,5))))
end


"""
    herm_back(∂z_∂A)

Pullback companion of [`herm`](@ref): map the cotangent of a hermitized tensor field
back to the cotangent of its raw argument,
``\\bar{A}_{ij} = w_{ij}(\\bar{H}_{ij} + \\bar{H}_{ji}^*)`` with weights ``w`` = ½ on
the diagonal and 1 off it — so that gradients of objectives written in terms of
`herm(A)` are correct for unsymmetrized `A` inputs.
"""
function herm_back(∂z_∂A::AbstractArray{T,4}) where {T<:Number}
	half_diag = [ 0.5 1.0 1.0 ; 1.0 0.5 1.0 ; 1.0 1.0 0.5 ]
	@tullio ∂z_∂A_herm[i,j,ix,iy] := ∂z_∂A[i,j,ix,iy]*half_diag[i,j] + conj(∂z_∂A)[j,i,ix,iy]*half_diag[i,j] nograd=half_diag
	return ∂z_∂A_herm
end

function herm_back(∂z_∂A::AbstractArray{T,5}) where {T<:Number}
	half_diag = [ 0.5 1.0 1.0 ; 1.0 0.5 1.0 ; 1.0 1.0 0.5 ]
	@tullio ∂z_∂A_herm[i,j,ix,iy,iz] := ∂z_∂A[i,j,ix,iy,iz]*half_diag[i,j] + conj(∂z_∂A)[j,i,ix,iy,iz]*half_diag[i,j] nograd=half_diag
	return ∂z_∂A_herm
end


"""

Generic Adjoint Solver Function Definitions

"""
function my_linsolve(Â, b⃗; x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	# x⃗ = isnothing(x⃗₀) ? randn(eltype(b⃗),first(size(b⃗))) : copy(x⃗₀)
	# x⃗ = isnothing(x⃗₀) ? zero(b⃗) : copy(x⃗₀)

	# return bicgstabl!(x⃗, Â, b⃗, 2; Pl=P̂, max_mv_products=5000)
	# return bicgstabl!(x⃗, Â, b⃗, 2; Pl=P̂, max_mv_products=3000)
	# bicgstabl(Â, b⃗, 3; Pl=P̂, max_mv_products=3000)
	# cg(Â, b⃗; Pl=P̂, maxiter=3000)
	# bicgstabl(Â, b⃗, 2; Pl=P̂, max_mv_products=10000)
	gmres(Â, b⃗; Pl=P̂, maxiter=1000)
end

function rrule(::typeof(my_linsolve), Â, b⃗;
		x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	x⃗ = my_linsolve(Â, b⃗; x⃗₀, P̂)
	function my_linsolve_pullback(x̄)
		λ⃗ = my_linsolve(Â', vec(x̄))
		Ā = (-λ⃗) ⊗ x⃗'
		return (NoTangent(), Ā, λ⃗)
	end
	return (x⃗, my_linsolve_pullback)
end

"""
	eig_adjt(A, α, x⃗, ᾱ, x̄; λ⃗₀, P̂)

Compute the adjoint vector `λ⃗` for a single eigenvalue/eigenvector pair (`α`,`x⃗`) of `Â` and
sensitivities (`ᾱ`,`x̄`). It is assumed (but not checked) that ``Â ⋅ x⃗ = α x⃗``. `λ⃗` is the
sum of two components,

	``λ⃗ = λ⃗ₐ + λ⃗ₓ``

where ``λ⃗ₐ = ᾱ x⃗`` and ``λ⃗ₓ`` correspond to `ᾱ` and `x̄`, respectively. When `x̄` is non-zero
``λ⃗ₓ`` is computed by iteratively solving

	``(Â - αÎ) ⋅ λ⃗ₓ = x̄ - (x⃗ ⋅ x̄)``

An inital guess can be supplied for `λ⃗ₓ` via the keyword argument `λ⃗₀`, otherwise a random
vector is used. A preconditioner `P̂` can also be supplied to improve convergeance.
"""
function eig_adjt(Â, α, x⃗, ᾱ, x̄; λ⃗₀=nothing, P̂=IterativeSolvers.Identity())
	if iszero(x̄)
		λ⃗ = iszero(ᾱ)	? zero(x⃗) : ᾱ * x⃗
 	else
		λ⃗ₓ₀ = my_linsolve(
			Â - α*I,
		 	x̄ - x⃗ * dot(x⃗,x̄);
			P̂,
			# x⃗₀=λ⃗₀,
		)
		# λ⃗ -= x⃗ * dot(x⃗,λ⃗)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		# λ⃗ += ᾱ * x⃗
		λ⃗ₓ = λ⃗ₓ₀  - x⃗ * dot(x⃗,λ⃗ₓ₀)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		λ⃗ = λ⃗ₓ + ᾱ * x⃗
	end

	return λ⃗
end

function rrule(::typeof(eig_adjt), Â, α, x⃗, ᾱ, x̄; λ⃗₀=nothing, P̂=IterativeSolvers.Identity())
	# primal
	if iszero(x̄)
		λ⃗ = iszero(ᾱ)	? zero(x⃗) : ᾱ * x⃗
 	else
		λ⃗ₓ₀ = my_linsolve(
			Â - α*I,
		 	x̄ - x⃗ * dot(x⃗,x̄);
			P̂,
			# x⃗₀=λ⃗₀,
		)
		λ⃗ₓ = λ⃗ₓ₀ - x⃗ * dot(x⃗,λ⃗ₓ₀)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		λ⃗ = λ⃗ₓ + ᾱ * x⃗
	end

	# define pullback
	# function eig_adjt_pullback(lm̄)
	# 	if lm̄ isa AbstractZero
	# 		return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
	# 	else
	# 		if iszero(x̄)
	# 			if iszero(ᾱ)
	# 				return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
	# 			else
	# 				x⃗_bar = ᾱ * lm̄
	# 				ᾱ_bar = dot(lm̄,x⃗)
	# 				return (NoTangent(), ZeroTangent(), ZeroTangent(), x⃗_bar, ᾱ_bar, ZeroTangent())
	# 			end
	# 	 	else	# λ⃗ₓ exists, must solve 2ⁿᵈ-order adjoint problem
	# 			lm̄_perp = lm̄ - x⃗ * dot(x⃗,lm̄)
	# 			ξ⃗ = my_linsolve(
	# 				Â - α*I,
	# 			 	lm̄_perp;
	# 				P̂,
	# 				# x⃗₀=λ⃗ₓ,
	# 			)
	# 			# println("")
	# 			# println("ξ⃗ err: $(sum(abs2,(Â - α*I)*ξ⃗-(lm̄ - x⃗ * dot(x⃗,lm̄))))")
	#
	# 			xv_dot_ξ⃗ = dot(x⃗,ξ⃗)
	# 			ξ⃗perp = ξ⃗ - x⃗ * xv_dot_ξ⃗
	#
	# 			# println("sum(abs2,ξ⃗): $(sum(abs2,ξ⃗)))")
	# 			# println("sum(abs2,ξ⃗perp): $(sum(abs2,ξ⃗perp)))")
	# 			# println("sum(abs2,lm̄): $(sum(abs2,lm̄)))")
	# 			# println("sum(abs2,lm̄_perp): $(sum(abs2,lm̄_perp)))")
	#
	# 			A_bar = (-ξ⃗perp) ⊗ λ⃗ₓ' # * (2*length(x⃗)))
	# 			α_bar = dot(ξ⃗,λ⃗ₓ)
	# 			x⃗_bar = dot(x̄,x⃗) * ξ⃗perp + ᾱ * lm̄ + dot(ξ⃗,x⃗) * x̄
	# 			ᾱ_bar = dot(lm̄,x⃗)
	# 			x̄_bar = ξ⃗perp
	#
	# 			# A_bar = -ξ⃗ ⊗ λ⃗ₓ'
	# 			# α_bar = dot(ξ⃗,λ⃗ₓ)
	# 			# x⃗_bar = dot(x̄,x⃗) * ξ⃗ + dot(ξ⃗,x⃗) * x̄  + ᾱ * lm̄
	# 			# ᾱ_bar = dot(lm̄,x⃗)
	# 			# x̄_bar = ξ⃗ - (x⃗ * dot(x⃗,ξ⃗))
	# 		end
	# 		# return (NoTangent(), A_bar, α_bar, x⃗_bar, ᾱ_bar, x̄_bar)
	# 		return (NoTangent(), A_bar, α_bar, x⃗_bar, ᾱ_bar, x̄_bar)
	# 		# return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
	# 	end
	# end

	# define pullback
	function eig_adjt_pullback(lm̄)
		if lm̄ isa AbstractZero
			return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
		else
			if iszero(x̄)
				if iszero(ᾱ)
					return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
				else
					x⃗_bar = ᾱ * lm̄
					ᾱ_bar = dot(lm̄,x⃗)
					return (NoTangent(), ZeroTangent(), ZeroTangent(), x⃗_bar, ᾱ_bar, ZeroTangent())
				end
		 	else	# λ⃗ₓ exists, must solve 2ⁿᵈ-order adjoint problem
				ξ⃗ = my_linsolve(
					Â - α*I,
				 	lm̄ - x⃗ * dot(x⃗,lm̄);
					P̂,
				)
				# println("")
				# println("ξ⃗ err: $(sum(abs2,(Â - α*I)*ξ⃗-(lm̄ - x⃗ * dot(x⃗,lm̄))))")
				A_bar = -ξ⃗ ⊗ λ⃗ₓ'
				α_bar = dot(ξ⃗,λ⃗ₓ)
				x⃗_bar = dot(x̄,x⃗) * ξ⃗ + dot(ξ⃗,x⃗) * x̄  + ᾱ * lm̄
				ᾱ_bar = dot(lm̄,x⃗)
				x̄_bar = ξ⃗ - (x⃗ * dot(x⃗,ξ⃗))
			end
			return (NoTangent(), A_bar, α_bar, x⃗_bar, ᾱ_bar, x̄_bar)
		end
	end

	# return primal and pullback
	return (λ⃗, eig_adjt_pullback)
end





function ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid::Grid{ND,T}) where {ND,T<:Real}
	# Nranges, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.ignore() do
	Ninv 		= 		1. / N(grid)
	Ns			=		size(grid)
	# g⃗s = g⃗(grid)
	Nranges		=		eachindex(grid)
	d0 = randn(Complex{T}, (3,Ns...))
	𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
	𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		# return (Nranges,Ninv,Ns,𝓕,𝓕⁻¹)
	# end
	mag, m⃗, n⃗  = mag_m_n(k,g⃗(grid))
	mns = hcat(reshape(copy(reinterpret(reshape,T,m⃗)),(3,1,Ns...)), reshape(copy(reinterpret(reshape,T,n⃗)),(3,1,Ns...)))
	H = reshape(H⃗,(2,Ns...))
	λ = reshape(λ⃗,(2,Ns...))
	d 	= 	𝓕 * kx_tc( H , mns, mag ) * Ninv
	λd 	= 	𝓕 * kx_tc( λ , mns, mag )
	eī	 = 	 ε⁻¹_bar(vec(d), vec(λd), Ns...)
	eif = ε⁻¹ #flat(ε⁻¹)
	# eif = reshape(reinterpret(reshape,T,ε⁻¹),3,3,Ns...) #flat(ε⁻¹)
	# eif = reshape(reinterpret(T,ε⁻¹),3,3,Ns...)
	λẽ  =   𝓕⁻¹ * ε⁻¹_dot(λd * Ninv, real(eif)) #flat(ε⁻¹)) # _d2ẽ!(λẽ , λd  ,M̂ )
	ẽ 	 =   𝓕⁻¹ * ε⁻¹_dot(d        , real(eif)) #flat(ε⁻¹)) # _d2ẽ!(M̂.e,M̂.d,M̂)
	λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
	ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
	m̄_kx = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ_sv .* conj.(view(λ,2,Nranges...)) )	#NB: m̄_kx and n̄_kx would actually
	n̄_kx =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ_sv .* conj.(view(λ,1,Nranges...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
	māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)

	# mag2, m⃗, n⃗  = mag_m_n(k,grid)
	# m̄n_kx = cat(
	# 	reshape(reinterpret(reshape,T,m̄_kx),(3,1,size(grid)...)), 
	# 	reshape(reinterpret(reshape,T,n̄_kx),(3,1,size(grid)...));
	# 	dims=2
	# )

	k̄		= ∇ₖmag_m_n(
				māg_kx, 		# māg total
				m̄_kx.*mag, 	# m̄  total
				n̄_kx.*mag,	  	# n̄  total
				mag, m⃗, n⃗,
			)

	# @show k̄_new		= ∇ₖmag_mn(
	# 		māg_kx, 		# māg total
	# 		m̄n_kx*mag,	  	# mn̄  total
	# 		mag, mns,
	# 	)
	return k̄, eī
end

function rrule(::typeof(HelmholtzMap), kz::T, ε⁻¹, grid::Grid; shift=0.) where {T<:Real}
	function HelmholtzMap_pullback(M̄)
		if M̄ isa AbstractZero
			k̄	= ZeroTangent()
			eī = ZeroTangent()
		else
			λ⃗ = -M̄.maps[1].lmap
			H⃗ = M̄.maps[2].lmap'
			k̄, eī = ∇M̂(kz,ε⁻¹,λ⃗,H⃗,grid)
		end

		return (NoTangent(), k̄, eī, ZeroTangent())
	end
	return HelmholtzMap(kz, ε⁻¹, grid; shift), HelmholtzMap_pullback
end


# function rrule(T::Type{<:LinearMaps.LinearCombination{Complex{T1}}},As::Tuple{<:HelmholtzMap,<:LinearMaps.UniformScalingMap}) where T1<:Real
# 	function LinComb_Helmholtz_USM_pullback(M̄)
# 		# return (NoTangent(), (M̄, M̄))
# 		return (NoTangent(), Composite{Tuple{LinearMap,LinearMap}}(M̄, M̄))
# 	end
# 	return LinearMaps.LinearCombination{Complex{T1}}(As), LinComb_Helmholtz_USM_pullback
# end
#
# function rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,N::Int) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/N, M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NoTangent(), ᾱ, ZeroTangent())
# 	end
# 	return LinearMaps.UniformScalingMap(α,N), USM_pullback
# end
#
# function rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,N::Int,N2::Int) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/N, M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NoTangent(), ᾱ, ZeroTangent(), ZeroTangent())
# 	end
# 	return LinearMaps.UniformScalingMap(α,N,N2), USM_pullback
# end
#
# function rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,Ns::Tuple{<:Int,<:Int}) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/first(Ns), M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NoTangent(), ᾱ, DoesNotExist())
# 	end
# 	return LinearMaps.UniformScalingMap(α,Ns), USM_pullback
# end












"""
Inversion/Conjugate-transposition equalities of Maxwell operator components
----------------------------------------------------------------------------

FFT operators:
--------------
	If 𝓕⁻¹ === `bfft` (symmetric, unnormalized case)

	(1a)	(𝓕)' 	= 	𝓕⁻¹

	(2a)	(𝓕⁻¹)' = 	𝓕

	If 𝓕⁻¹ === `ifft` (asymmetric, normalized case)

	(1a)	(𝓕)' 	= 	𝓕⁻¹ * N		( N := Nx * Ny * Nz )

	(2a)	(𝓕⁻¹)' = 	𝓕	 / N

Combined Curl+Basis-Change operators:
--------------------------------------
	(3)	( [(k⃗+g⃗)×]cₜ )' 	= 	-[(k⃗+g⃗)×]ₜc

	(4)	( [(k⃗+g⃗)×]ₜc )' 	= 	-[(k⃗+g⃗)×]cₜ

Combined Cross+Basis-Change operators:
--------------------------------------
	(3)	( [(ẑ)×]cₜ )' 	= 	[(ẑ)×]ₜc

	(4)	( [(ẑ)×]ₜc )' 	= 	[(ẑ)×]cₜ


--------------
"""

"""
Calculate k̄ contribution from M̄ₖ, where M̄ₖ is backpropagated from ⟨H|M̂ₖ|H⟩

Consider Mₖ as composed of three parts:

	(1) Mₖ	= 	[(k⃗+g⃗)×]cₜ  ⋅  [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]   ⋅  [ẑ×]ₜc
				------------	-----------------	  -------
  			= 		 A				    B                C

where the "cₜ" and "ₜc" labels on the first and third components denote
Cartesian-to-Transverse and Transverse-to-Cartesian coordinate transformations,
respectively.

From Giles, we know that if D = A B C, then

	(2)	Ā 	=	D̄ Cᵀ Bᵀ

	(3)	B̄ 	=	Aᵀ D̄ Cᵀ

	(4) C̄	=	Bᵀ Aᵀ D̄		(to the bone)

We also know that M̄ₖ corresponding to the gradient of ⟨H|M̂ₖ|H⟩ will be

	(5) M̄ₖ	=	|H*⟩⟨H|

where `*` denote complex conjugation.
Equations (2)-(5) give us formulae for the gradients back-propagated to the three
parameterized operators composing Mₖ

	(6) 	[k+g ×]̄		 = 	 |H⟩⟨H|	 ⋅  [[ẑ×]ₜc]ᵀ  ⋅  [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]ᵀ

							= 	|H⟩⟨ ( [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]  ⋅  [ẑ×]ₜc ⋅ H ) |

	(7)	[ 𝓕 nn̂g⁻¹ 𝓕⁻¹ ]̄	   = 	 [[k+g ×]cₜ]ᵀ  ⋅  |H⟩⟨H| ⋅  [[ẑ×]ₜc]ᵀ

						 	= 	-[k+g ×]ₜc  ⋅  |H⟩⟨H| ⋅  [ẑ×]cₜ

							=	-| [k+g ×]ₜc  ⋅ H ⟩⟨ [ẑ×]ₜc ⋅ H |

	(8)  ⇒ 	[ nn̂g⁻¹ ]̄	 	  =   -| 𝓕 ⋅ [k+g ×]ₜc  ⋅ H ⟩⟨ 𝓕 ⋅ [ẑ×]ₜc ⋅ H |

	(9)			[ẑ ×]̄	 	  =   [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]ᵀ ⋅ [[k+g×]cₜ]ᵀ ⋅ |H⟩⟨H|

							= 	-| [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]  ⋅  [k+g×]cₜ ⋅ H ⟩⟨H|

where `[ẑ ×]` operators are still parameterized by k⃗ because they involve
m⃗ & n⃗ orthonormal polarization basis vectors, which are determined by k⃗+g⃗
and thus k⃗-dependent.

Our [(k⃗+g⃗)×]cₜ and [ẑ ×]ₜc operators act locally in reciprocal space with the
following local structures

	(10) [ ( k⃗+g⃗[ix,iy,iz] ) × ]ₜc  =		[	-n⃗₁	m⃗₁
											  -n⃗₂	  m⃗₂
											  -n⃗₃	  m⃗₃	]

									=	  [	   m⃗     n⃗  	]  ⋅  [  0   -1
																	1 	 0	]


	(11) [ ( k⃗+g⃗[ix,iy,iz] ) × ]cₜ  =			 [	   n⃗₁	  n⃗₂	  n⃗₃
										  			-m⃗₁   -m⃗₂	  -m⃗₃	  ]

									=	[  0   -1 		⋅	[   m⃗ᵀ
										   1 	0  ]			n⃗ᵀ		]

										=	-(	[ ( k⃗+g⃗[ix,iy,iz] ) × ]ₜc	)ᵀ

	(12) [ ẑ × ]ₜc	 	 =	[	 -m⃗₂	-n⃗₂
								 m⃗₁	n⃗₁
								 0	    0	 ]

						=	[	0 	-1	  0		 		[	m⃗₁	 n⃗₁
								1 	 0	  0	 		⋅		m⃗₂	 n⃗₂
								0 	 0	  0	  ]				m⃗₃	 n⃗₃	]

	(13) [ ẑ × ]cₜ	 	 =	   [	 -m⃗₂	m⃗₁	 	0
									-n⃗₂	n⃗₁		0		]

						=	  [   m⃗ᵀ				[	0	 1	 0
								  n⃗ᵀ	]		⋅		-1	  0	  0
								  						0	 0	 0	]

						=	  (  [ ẑ × ]ₜc  )ᵀ
"""

"""
    ε⁻¹_bar(d⃗, λ⃗d, Nx, Ny[, Nz]) -> Array (3,3,Nx,Ny[,Nz])

Cotangent (gradient) of a Helmholtz-operator quadratic form w.r.t. the
inverse-permittivity field: accumulates the per-pixel 3×3 blocks of
``-\\mathrm{Re}\\,(λ_d ⊗ d^†)`` from the (vec'd, k-space-curl) D-field `d⃗` of the
forward pass and the corresponding adjoint-field product `λ⃗d`. Off-diagonal entries
hold the summed ``(i,j)+(j,i)`` sensitivity mirrored into both slots, consistent with
backpropagation through Hermitian tensor fields (see [`herm_back`](@ref)). This is
the workhorse of the `solve_k` adjoint `rrule`.
"""
function ε⁻¹_bar(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny, Nz) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = zeros(T,3,3,Nx,Ny,Nz)
	# @avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
	for iz=1:Nz,iy=1:Ny,ix=1:Nx
		# column-major linear voxel index of (:,ix,iy,iz) in the vec'd (3,Nx,Ny,Nz) field data
		q = (Nx * Ny * (iz-1) + Nx * (iy-1) + ix)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	end
	# eī = unflat(copy(eīf))
	# eī = reinterpret(reshape,SMatrix{3,3,T,9},reshape(copy(eīf),9,Nx,Ny,Nz))
	eī = copy(eīf)
	return eī
end

# 2D
function ε⁻¹_bar(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = zeros(T,3,3,Nx,Ny)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny))
	# @avx for iy=1:Ny,ix=1:Nx
	for iy=1:Ny,ix=1:Nx
		# column-major linear pixel index of (:,ix,iy) in the vec'd (3,Nx,Ny) field data
		q = (Nx * (iy-1) + ix)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
			eīf[a2+1,a2,ix,iy] = eīf[a2,a2+1,ix,iy]
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
		eīf[3,1,ix,iy] = eīf[1,3,ix,iy]
	end
	# eī = reinterpret(reshape,SMatrix{3,3,T,9},reshape(copy(eīf),9,Nx,Ny))
	eī = copy(eīf)
	return eī # inv( (eps' + eps) / 2)
end

# newer Tullio based version
function ε⁻¹_bar(d⃗::AbstractArray{Complex{T},N}, λ⃗d::AbstractArray{Complex{T},N}) where {T<:Real,N}
	-real( herm_back(_outer(λ⃗d,d⃗)))
end

#####
function solve_adj!(λ⃗,M̂::HelmholtzMap,H̄,ω²,H⃗,eigind::Int;log=false)
	# log=true
	res = bicgstabl!(
		λ⃗, # ms.adj_itr.x,	# recycle previous soln as initial guess
		M̂ - real(ω²)*I, # A
		H̄ - H⃗ * dot(H⃗,H̄), # b,
		2;	# l = number of GMRES iterations per CG iteration
		# Pl = HelmholtzPreconditioner(M̂), # left preconditioner
		log,
		abstol=1e-8,
		max_mv_products=500
		)
	if log
		copyto!(λ⃗,res[1])
		ch = res[2]
	else
		copyto!(λ⃗,res)
	end
	# println("#########  Adjoint Problem for kz = $( M̂.k⃗[3] ) ###########")
	# uplot(ch;name="log10( adj. prob. res. )")
	# println("\t\t\tadj converged?: $ch")
	# println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")
	return λ⃗
end

