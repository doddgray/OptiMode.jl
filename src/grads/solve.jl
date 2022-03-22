#### AD Rules for Iterative eigensolves of Helmholtz Operator
using LinearAlgebra, StaticArrays, Tullio, ChainRulesCore, LinearMaps, IterativeSolvers
using LinearMaps: ⊗
using IterativeSolvers: gmres, lobpcg, lobpcg!

export ε⁻¹_bar, herm, eig_adjt, my_linsolve, solve_adj!

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




function rrule(::typeof(solve_k), ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing) where {ND,T<:Real} 
	
	# ms = ModeSolver(k_guess(ω,ε⁻¹), ε⁻¹, grid; nev, maxiter, tol=eig_tol)
	# kmags,evecs = solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter,)
	# @show omsq_solns = copy(ms.ω²)
	# @show domsq_dk_solns = copy(ms.ms.∂ω²∂k)
	kmags,evecs = solve_k(ω, ε⁻¹, grid, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter,)

	# g⃗ = copy(ms.M̂.g⃗)
	# (mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
	# 	mag_m_n(x,dropgrad(g⃗))
	# end

	# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	# Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	# println("\tsolve_k:")
	# println("\t\tω² (target): $(ω^2)")
	# println("\t\tω² (soln): $(ms.ω²[eigind])")
	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
	# println("\t\tk: $k")
	# println("\t\t∂ω²∂k: $∂ω²∂k")
	# ∂ω²∂k = copy(ms.∂ω²∂k[eigind])
	gridsize = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	
	# ε⁻¹_copy = copy(ε⁻¹)
	# k = copy(k)
	# Hv = copy(Hv)
	function solve_k_pullback(ΔΩ)
		ei_bar = zero(ε⁻¹)
		ω_bar = zero(ω)
		k̄mags, ēvecs = ΔΩ
		for (eigind, k̄, ēv, k, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
			ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter)
			println("\tsolve_k_pullback:")
			println("k̄ (bar): $k̄")
			# update_k!(ms,k)
			# update_ε⁻¹(ms,ε⁻¹) #ε⁻¹)
			println("\tsolve_k pullback for eigind=$eigind:")
			println("\t\tω² (target): $(ω^2)")
			# println("\t\tω² (soln): $(omsq_solns[eigind])")
			# println("\t\tΔω² (soln): $(real(ω^2 - omsq_solns[eigind]))")
			
			# ms.∂ω²∂k[eigind] = ∂ω²∂k
			# copyto!(ms.H⃗, ev)
			ms.H⃗[:,eigind] = copy(ev)
			# replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
			λ⃗ = randn(eltype(ev),size(ev)) # similar(ev)
			λd =  similar(ms.M̂.d)
			λẽ = similar(ms.M̂.d)

			# println("\t\t∂ω²∂k (recorded): $(domsq_dk_solns[eigind])")
			∂ω²∂k = 2 * HMₖH(ev,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
			println("\t\t∂ω²∂k (recalc'd): $(∂ω²∂k)")
			# 
			# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
			# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
			(mag,m⃗,n⃗), mag_m_n_pb = Zygote.pullback(kk->mag_m_n(kk,g⃗(ms.grid)),k)

			ev_grid = reshape(ev,(2,gridsize...))
			# if typeof(k̄)==ZeroTangent()
			if isa(k̄,AbstractZero)
				k̄ = 0.0
			end
			# if typeof(ēv) != ZeroTangent()
			if !isa(ēv,AbstractZero)
				# λ⃗ = randn(eltype(ev),size(ev)) # similar(ev)
				# λd =  similar(ms.M̂.d)
				# λẽ = similar(ms.M̂.d)
				# solve_adj!(ms,ēv,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēv - dot(ev,ēv)*ev
				# solve_adj!(λ⃗,ms.M̂,ēv,ω^2,ev,eigind;log=false)
				λ⃗ = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=randn(eltype(ev),size(ev)), P̂=ms.P̂)
				# solve_adj!(ms,ēv,ω^2,ev,eigind;log=false)
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
				# m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,..])
				# n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,..])
				māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
				@show k̄ₕ_old = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
				@show k̄ₕ = -∇ₖmag_m_n(
					māg,
					kx̄_m⃗.*ms.M̂.mag, # m̄,
					kx̄_n⃗.*ms.M̂.mag, # n̄,
					ms.M̂.mag,
					m⃗,
					n⃗;
					dk̂=SVector(0.,0.,1.), # dk⃗ direction
				)
			else
				# eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
				k̄ₕ = 0.0
			end
			# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω_bar and eīₖ
			# copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * ev )
			λ⃗ = ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * ev
			d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,reshape(λ⃗,(2,gridsize...)),ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
			# ei_bar = eīₖ + eīₕ
			ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₖ # 
			@show ω_bar +=  ( 2ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
			# if !(typeof(k)<:SVector)
			# 	k̄_kx = k̄_kx[3]
			# end
			# ms.ω_bar = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
		end

		return (NoTangent(), ω_bar , ei_bar,ZeroTangent(),NoTangent())
	end
	return ((kmags, evecs), solve_k_pullback)
end