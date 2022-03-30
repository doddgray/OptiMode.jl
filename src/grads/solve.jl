#### AD Rules for Iterative eigensolves of Helmholtz Operator
using LinearAlgebra, StaticArrays, Tullio, ChainRulesCore, LinearMaps, IterativeSolvers
using LinearMaps: ⊗
using IterativeSolvers: gmres, lobpcg, lobpcg!

export ε⁻¹_bar, herm, eig_adjt, my_linsolve, solve_adj!, ng_gvd_E, ng_gvd

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

function ε⁻¹_bar(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny, Nz) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = Buffer(Array{Float64,1}([2., 2.]),3,3,Nx,Ny,Nz) # bufferfrom(zero(T),3,3,Nx,Ny,Nz)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny,Nz))
	@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
		q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
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
	eīf = Buffer(Array{Float64,1}([2., 2.]),3,3,Nx,Ny) # bufferfrom(zero(T),3,3,Nx,Ny)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny))
	@avx for iy=1:Ny,ix=1:Nx
		q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
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
    M̂,P̂ = Zygote.ignore() do
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
    M̂,P̂ = Zygote.ignore() do
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

