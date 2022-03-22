#### AD Rules for Iterative eigensolves of Helmholtz Operator

export ε⁻¹_bar!, ε⁻¹_bar, ∂ω²∂k_adj, Mₖᵀ_plus_Mₖ, ∂²ω²∂k², herm,
     ∇ₖmag_m_n, ∇HMₖH, ∇M̂, ∇solve_k, ∇solve_k!, solve_adj!, 
     neff_ng_gvd, ∂ε⁻¹_∂ω, ∂nng⁻¹_∂ω, ∇ₖmag_mn

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

function update_k_pb(M̂::HelmholtzMap{T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n), mag_m_n_pb = Zygote.pullback(k⃗) do x
		mag_m_n(x,dropgrad(M̂.g⃗))
	end
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
	M̂.m⃗ = m #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = n #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.k⃗ = k⃗
	return (mag, m, n), mag_m_n_pb
end

update_k_pb(M̂::HelmholtzMap{T},kz::T) where T<:Real = update_k_pb(M̂,SVector{3,T}(0.,0.,kz))

# 3D
function ε⁻¹_bar!(eī, d⃗, λ⃗d, Nx, Ny, Nz)
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
	eīf = flat(eī)
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
	return eī
end

# 2D
function ε⁻¹_bar!(eīf, d⃗, λ⃗d, Nx, Ny)
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
	# eīf = flat(eī)
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
	return eī # inv( (eps' + eps) / 2)

	# eīM = Matrix.(eī)
	# for iy=1:Ny,ix=1:Nx
	# 	q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
	# 	for a=1:3 # loop over diagonal elements: {11, 22, 33}
	# 		eīM[ix,iy][a,a] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
	# 	end
	# 	for a2=1:2 # loop over first off diagonal
	# 		eīM[ix,iy][a2,a2+1] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
	# 	end
	# 	# a = 1, set 1,3 and 3,1, second off-diagonal
	# 	eīM[ix,iy][1,3] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	# end
	# ēM = inv.(eīM)
	# eīMH = inv.( ( ēM .+ ēM' ) ./ 2 )
	# eī .= SMatrix{3,3}.( eīMH  ) # SMatrix{3,3}.(eīM)
	# return eī
end

# 3D
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



function solve_adj!(ms::ModeSolver,H̄,eigind::Int)
	ms.adj_itr = bicgstabl_iterator!(
		ms.adj_itr.x,	# recycle previous soln as initial guess
		ms.M̂ - real(ms.ω²)*I, # A
		H̄ - ms.H⃗ * dot(ms.H⃗,H̄), # b,
		3;	# l = number of GMRES iterations per CG iteration
		Pl = ms.P̂) # left preconditioner
	for (iteration, item) = enumerate(ms.adj_itr) end # iterate until convergence or until (iters > max_iters || mvps > max_mvps)
	copyto!(ms.λ⃗,ms.adj_itr.x) # copy soln. to ms.λ⃗ where other contributions/corrections can be accumulated
	# λ₀, ch = bicgstabl(
	# 	ms.adj_itr.x,	# recycle previous soln as initial guess
	# 	ms.M̂ - real(ms.ω²[eigind])*I, # A
	# 	H̄[:,eigind] - ms.H⃗[:,eigind] * dot(ms.H⃗[:,eigind],H̄[:,eigind]), # b,
	# 	3;	# l = number of GMRES iterations per CG iteration
	# 	Pl = ms.P̂, # left preconditioner
	# 	reltol = 1e-10,
	# 	log=true,
	# 	)
	# copyto!(ms.λ⃗,λ₀) # copy soln. to ms.λ⃗ where other contributions/corrections can be accumulated
	# println("\t\tAdjoint Problem for kz = $( ms.M̂.k⃗[3] ) ###########")
	# println("\t\t\tadj converged?: $ch")
	# println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")
	# uplot(ch;name="log10( adj. prob. res. )")
	return ms.λ⃗
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


using LinearMaps: ⊗
export eig_adjt, linsolve
using IterativeSolvers: gmres
function linsolve(Â, b⃗; x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	# x⃗ = isnothing(x⃗₀) ? randn(eltype(b⃗),first(size(b⃗))) : copy(x⃗₀)
	# x⃗ = isnothing(x⃗₀) ? zero(b⃗) : copy(x⃗₀)

	# return bicgstabl!(x⃗, Â, b⃗, 2; Pl=P̂, max_mv_products=5000)
	# return bicgstabl!(x⃗, Â, b⃗, 2; Pl=P̂, max_mv_products=3000)
	# bicgstabl(Â, b⃗, 3; Pl=P̂, max_mv_products=3000)
	# cg(Â, b⃗; Pl=P̂, maxiter=3000)
	# bicgstabl(Â, b⃗, 2; Pl=P̂, max_mv_products=10000)
	gmres(Â, b⃗; Pl=P̂, maxiter=1000)
end

function rrule(::typeof(linsolve), Â, b⃗;
		x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	x⃗ = linsolve(Â, b⃗; x⃗₀, P̂)
	function linsolve_pullback(x̄)
		λ⃗ = linsolve(Â', vec(x̄))
		Ā = (-λ⃗) ⊗ x⃗'
		return (NoTangent(), Ā, λ⃗)
	end
	return (x⃗, linsolve_pullback)
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
		λ⃗ₓ₀ = linsolve(
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
		λ⃗ₓ₀ = linsolve(
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
	# 			ξ⃗ = linsolve(
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
				ξ⃗ = linsolve(
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

function ∇solve_ω²(ΔΩ,Ω,k,ε⁻¹,grid)
	@show ω̄sq, H̄ = ΔΩ
	@show ω², H⃗ = Ω
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	H = reshape(H⃗,(2,Ns...))
	g⃗s = g⃗(dropgrad(grid))
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
	λd = similar(M̂.d)
	λẽ = similar(M̂.d)
	ẽ = similar(M̂.d)
	ε⁻¹_bar = similar(ε⁻¹)
	if typeof(ω̄sq)==ZeroTangent()
		ω̄sq = 0.
	end
	if typeof(H̄) != ZeroTangent()
		λ⃗ = solve_adj!(M̂,H̄,ω²,H⃗,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
		λ⃗ -= (ω̄sq + dot(H⃗,λ⃗)) * H⃗
	else
		λ⃗ = -ω̄sq * H⃗
	end
	λ = reshape(λ⃗,(2,Ns...))
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(ε⁻¹_bar, vec(M̂.d), vec(λd), Ns...)
	# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
	λd *=  M̂.Ninv
	λẽ .= reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  , M̂ ) )
	ẽ .= reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d, M̂) )
	kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
	kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
	māg .= dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
	k̄ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	return (NoTangent(), ZeroTangent(), k̄ , ε⁻¹_bar)
end

function rrule(::typeof(solve_ω²), k::Union{T,SVector{3,T}},shapes::Vector{<:Shape},grid::Grid{ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	# println("using new rrule")
	ms = @ignore(ModeSolver(k, shapes, grid)) # ; nev, eigind, maxiter, tol, log))
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	ω²H⃗ = solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
    solve_ω²_pullback(ΔΩ) = ∇solve_ω²(ΔΩ,ω²H⃗,k,ε⁻¹,grid)
    return (ω²H⃗, solve_ω²_pullback)
end

function rrule(::typeof(solve_ω²), ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ω²,H⃗ = solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
		mag_m_n(x,dropgrad(ms.M̂.g⃗))
	end
    function solve_ω²_pullback(ΔΩ)
		ω̄sq, H̄ = ΔΩ
		Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
		Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
		H = reshape(H⃗,(2,Ns...))
		# mn2 = vcat(reshape(ms.M̂.m,(1,3,Ns...)),reshape(ms.M̂.n,(1,3,Ns...)))
		if typeof(ω̄sq)==ZeroTangent()
			ω̄sq = 0.
		end
		if typeof(H̄) != ZeroTangent()
			solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
			ms.λ⃗ -= (ω̄sq[eigind] + dot(H⃗,ms.λ⃗)) * H⃗
		else
			ms.λ⃗ = -ω̄sq[eigind] * H⃗
		end
		λ = reshape(ms.λ⃗,(2,Ns...))
		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd), Ns...)
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		ms.λd *=  ms.M̂.Ninv
		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
		ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
		ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
		ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
		k̄ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
		# if !(typeof(k)<:SVector)
		# 	k̄_kx = k̄_kx[3]
		# end
		return (NoTangent(), ZeroTangent(), k̄ , ms.ε⁻¹_bar)
    end
    return ((ω², H⃗), solve_ω²_pullback)
end


"""
function mapping |H⟩ ⤇ ( (∂M/∂k)ᵀ + ∂M/∂k )|H⟩
"""
function Mₖᵀ_plus_Mₖ(H⃗::AbstractVector{Complex{T}},k,ε⁻¹,grid) where T<:Real
	Ns = size(grid)
	# g⃗s = g⃗(grid)
	# mag,m⃗,n⃗ = mag_m_n(k,g⃗s)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	H = reshape(H⃗,(2,Ns...))
	mn = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	X = -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(ε⁻¹)), (2:3) ), mn, mag )
	Y = zx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(ε⁻¹)), (2:3)), mn )
	vec(X + Y)
end

"""
solve the adjoint sensitivity problem corresponding to ∂ω²∂k = <H|∂M/∂k|H>
"""
function ∂ω²∂k_adj(M̂::HelmholtzMap,ω²,H⃗,H̄;eigind=1,log=false)
	res = bicgstabl(
		M̂ - real(ω²)*I, # A
		H̄ - H⃗ * dot(H⃗,H̄), # b,
		3;	# l = number of GMRES iterations per CG iteration
		# Pl = HelmholtzPreconditioner(M̂), # left preconditioner
		log,
		)
end

"""
pull back sensitivity w.r.t. ∂ω²∂k = 2⟨H|∂M/∂k|H⟩ to corresponding
k̄ (scalar) and nn̄g⁻¹ (tensor field) sensitivities
"""
function ∇HMₖH(k::Real,H⃗::AbstractArray{Complex{T}},nng⁻¹::AbstractArray{T2,N2},grid::Grid{ND};eigind=1) where {T<:Real,ND,T2<:Real,N2}
	# Setup
	local 	zxtc_to_mn = SMatrix{3,3}(	[	0 	-1	  0
											1 	 0	  0
											0 	 0	  0	  ]	)

	local 	kxtc_to_mn = SMatrix{2,2}(	[	0 	-1
											1 	 0	  ]	)

	g⃗s, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.ignore() do
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		g⃗s = g⃗(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		return (g⃗s,Ninv,Ns,𝓕,𝓕⁻¹)
	end
	mag, m⃗, n⃗  = mag_m_n(k,g⃗s)
	H = reshape(H⃗,(2,Ns...))
	Hsv = reinterpret(reshape, SVector{2,Complex{T}}, H )

	#TODO: Banish this quadruply re(shaped,interpreted) m,n,mns format back to hell
	# mns = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))

	### calculate k̄ contribution from M̄ₖ ( from ⟨H|M̂ₖ|H⟩ )
	Ā₁		=	conj.(Hsv)
	Ā₂ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(nng⁻¹)),
		)
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

	B̄₁ = 𝓕 * kx_tc( conj.(H) ,mns,mag)
	B̄₂ = 𝓕 * zx_tc( H * Ninv ,mns)
	@tullio B̄[a,b,i,j] := B̄₁[a,i,j] * B̄₂[b,i,j] + B̄₁[b,i,j] * B̄₂[a,i,j]   #/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2

	# # diagnostic for nngī accuracy
	#
	# # println("sum(B̄): $(sum(real(B̄)))")
	# # println("maximum(B̄): $(maximum(real(B̄)))")
	# B̄_herm = real(B̄)/2
	# println("sum(B̄_herm): $(sum(B̄_herm))")
	# println("maximum(B̄_herm): $(maximum(B̄_herm))")
	# # end diagnostic for nngī accuracy

	C̄₁ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * -kx_tc( H * Ninv, mns, mag) , nng⁻¹),
		)
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
	X = -kx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_tc(H,mns)		, nng⁻¹), mns, mag )
	Y =  zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nng⁻¹), mns )
	H̄ = vec(X + Y) * Ninv
	return k̄, H̄, nngī
	# return k̄, H̄, reinterpret(SMatrix{3,3,Float64,9},reshape( nngī ,9*128,128))
end

"""
pull back sensitivity w.r.t. ng_z = ∂kz/∂ω = ⟨E|(nng+ε)|E⟩ / ∫dA 2Re( conj(E) × H)⋅ẑ ) to corresponding
k̄ (scalar) and nn̄g⁻¹ (tensor field) sensitivities
"""
function ∇ng_z(k::Real,H⃗::AbstractArray{Complex{T}},nng⁻¹::AbstractArray{T2,N2},grid::Grid{ND};eigind=1) where {T<:Real,ND,T2<:Real,N2}
	# Setup
	local 	kxtc_to_mn = SMatrix{3,3}(	[	0 	-1	  0
											1 	 0	  0
											0 	 0	  0	  ]	)

	local 	kxtc_to_mn = SMatrix{2,2}(	[	0 	-1
											1 	 0	  ]	)

	g⃗s, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.ignore() do
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		g⃗s = g⃗(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		return (g⃗s,Ninv,Ns,𝓕,𝓕⁻¹)
	end
	mag, m⃗, n⃗  = mag_m_n(k,g⃗s)
	H = reshape(H⃗,(2,Ns...))
	Hsv = reinterpret(reshape, SVector{2,Complex{T}}, H )

	#TODO: Banish this quadruply re(shaped,interpreted) m,n,mns format back to hell
	# mns = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))

	### calculate k̄ contribution from M̄ₖ ( from ⟨H|M̂ₖ|H⟩ )
	Ā₁		=	conj.(Hsv)
	Ā₂ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(nng⁻¹)),
		)
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

	B̄₁ = 𝓕 * kx_tc( conj.(H) ,mns,mag)
	B̄₂ = 𝓕 * zx_tc( H * Ninv ,mns)
	@tullio B̄[a,b,i,j] := B̄₁[a,i,j] * B̄₂[b,i,j] + B̄₁[b,i,j] * B̄₂[a,i,j]   #/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2

	# # diagnostic for nngī accuracy
	#
	# # println("sum(B̄): $(sum(real(B̄)))")
	# # println("maximum(B̄): $(maximum(real(B̄)))")
	# B̄_herm = real(B̄)/2
	# println("sum(B̄_herm): $(sum(B̄_herm))")
	# println("maximum(B̄_herm): $(maximum(B̄_herm))")
	# # end diagnostic for nngī accuracy

	C̄₁ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * -kx_tc( H * Ninv, mns, mag) , nng⁻¹),
		)
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
	X = -kx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_tc(H,mns)		, nng⁻¹), mns, mag )
	Y =  zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nng⁻¹), mns )
	H̄ = vec(X + Y) * Ninv
	return k̄, H̄, nngī
	# return k̄, H̄, reinterpret(SMatrix{3,3,Float64,9},reshape( nngī ,9*128,128))
end


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
solve the adjoint sensitivity problem corresponding to ∂ω²∂k = <H|∂M/∂k|H>
"""
# function ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,Hv,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
function ∂²ω²∂k²(ω,p,geom_fn,k,Hv,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}

	# nng⁻¹, nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
	# ε⁻¹, epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)

	# ngvd = ngvdₛ(ω,geom,grid)
	# nng⁻¹, nnginv_pb = Zygote.pullback(x->nngₛ⁻¹(x,geom,grid),ω)
	# ε⁻¹, epsi_pb = Zygote.pullback(x->εₛ⁻¹(x,geom,grid),ω)

	# nng⁻¹, nnginv_pb = Zygote._pullback(Zygote.Context(),x->nngₛ⁻¹(x,dropgrad(geom),dropgrad(grid)),ω)
	# ε⁻¹, epsi_pb = Zygote._pullback(Zygote.Context(),x->εₛ⁻¹(x,dropgrad(geom),dropgrad(grid)),ω)

	# ε, nng, ngvd = εₛ_nngₛ_ngvdₛ(ω,geom,grid)
	# nng⁻¹ = inv.(nng)
	# ε⁻¹ = inv.(ε)

	mag,m⃗,n⃗ = mag_m_n(k,grid)

	ei,ei_pb = Zygote.pullback(ω) do ω
		ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],geom_fn,grid));
		return ε⁻¹
	end

	nngi,nngi_pb = Zygote.pullback(ω) do ω
		ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],geom_fn,grid));
		return nng⁻¹
	end

	ngvd0,ngvd_pb = Zygote.pullback(ω) do ω
		# ngvd,nng2,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs,:fnn̂gs),[false,false,true],geom_fn,grid,volfrac_smoothing));
		ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));
		return ngvd
	end
	nng20,nng2_pb = Zygote.pullback(ω) do ω
		# ngvd,nng2,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs,:fnn̂gs),[false,false,true],geom_fn,grid,volfrac_smoothing));
		ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));
		return nng2
	end

	nngi2,nngi2_pb = Zygote.pullback(ω) do ω
		# ngvd,nng2,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs,:fnn̂gs),[false,false,true],geom_fn,grid,volfrac_smoothing));
		ngvd,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,true],geom_fn,grid,volfrac_smoothing));
		return nngi2
	end

	ε,ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid));
	ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));

	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	k̄, H̄, nngī  = ∇HMₖH(k,Hv,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,Hv),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)


	println("")
	println("\n manual calc.:")
	om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	println("om̄₁: $(om̄₁)")
	println("om̄₂: $(om̄₂)")
	println("om̄₃: $(om̄₃)")
	om̄ = om̄₁ + om̄₂ + om̄₃
	println("om̄: $(om̄)")

	# k̄_nd, H̄_nd, nngī_nd  = ∇HMₖH(k,Hv,ε⁻¹,grid; eigind)
	# om̄₂_nd = dot(herm(nngī_nd), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω)) #dot(herm(nngī_nd), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	# om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	# println("om̄₁: $(om̄₁)")
	# println("om̄₂_nd: $(om̄₂_nd)")
	# println("om̄₃: $(om̄₃)")
	# om̄_nd = om̄₁ + om̄₂_nd + om̄₃
	# println("om̄_nd: $(om̄_nd)")


	nngis = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(nng⁻¹,(9,128,128))))
	eis = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(ε⁻¹,(9,128,128))))
	ngvds = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(ngvd,(9,128,128))))
	dnngi_dom_s = -(nngis.^2 ) .* ( ω*(eis.*inv.(nngis).^2 .- inv.(nngis)) .+ ngvds)
	dnngi_dom_sr = copy(flat(dnngi_dom_s))
	om̄₂2 = dot(herm(nngī), dnngi_dom_sr)
	println("om̄₂2: $(om̄₂2)")


	#######

	# calculate and print neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
	Hₜ = reshape(Hv,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	EE = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	HH = fft(tc(kx_ct( ifft( EE, (2:1+ND) ), mns,mag), mns),(2:1+ND) ) / ω
	EEs = copy(reinterpret(reshape,SVector{3,ComplexF64},EE))
	HHs = copy(reinterpret(reshape,SVector{3,ComplexF64},HH))
	Sz = dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
	PP = 2*real(sum(Sz))
	WW = dot(EE,_dot((ε+nng),EE))
	ng = WW / PP

	∂ω²∂k_disp = 2 * HMₖH(Hv,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	neff = k / ω
	# ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
	gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )
	# println("∂ω²∂k_disp: $(∂ω²∂k_disp)")
	println("neff: $(neff)")
	println("ng: $(ng)")
	println("gvd: $(gvd)")

	println("")
	println("calc. with pullbacks:")
	# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī2)) ) ./ 2
	# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
	om̄₂_pb = nngi_pb(herm(nngī))[1] #nngī2)
	om̄₃_pb = ei_pb(herm(eī₁))[1] #eī₁)
	println("om̄₁: $(om̄₁)")
	println("om̄₂_pb: $(om̄₂_pb)")
	println("om̄₃_pb: $(om̄₃_pb)")
	om̄_pb = om̄₁ + om̄₂_pb + om̄₃_pb
	println("om̄_pb: $(om̄_pb)")
	gvd_pb = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄_pb #( ng / ω ) * ( 1. - ( ng * om̄ ) )
	println("gvd_pb: $(gvd_pb)")
	println("")
	return om̄
end

function neff_ng_gvd(ω,geom,k,H⃗,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	ε, nng, ngvd = εₛ_nngₛ_ngvdₛ(ω,geom,grid)
	nng⁻¹ = inv.(nng)
	ε⁻¹ = inv.(ε)
	# calculate om̄ = ∂²ω²/∂k²
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	∂ω²∂k_nd = 2 * HMₖH(H⃗,ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,H⃗),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)

    # nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī)) ) ./ 2
	# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
	# om̄₂ = dot(nngī_herm, ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	# om̄₃ = dot(eī_herm, ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	om̄ = om̄₁ + om̄₂ + om̄₃
	# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
	∂ω²∂k_disp = 2 * HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	neff = k / ω
	ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
	gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )
	return neff, ng, gvd
end

function neff_ng_gvd(ω,ε,ε⁻¹,nng,nng⁻¹,ngvd,k,Hv,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	# ε, nng, ngvd = εₛ_nngₛ_ngvdₛ(ω,geom,grid)
	# nng⁻¹ = inv.(nng)
	# ε⁻¹ = inv.(ε)
	# calculate om̄ = ∂²ω²/∂k²
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗)))
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n)
	k̄, H̄, nngī  = ∇HMₖH(k,Hv,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,Hv),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)
	# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī)) ) ./ 2
	# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
	om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	om̄ = om̄₁ + om̄₂ + om̄₃
	# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
	∂ω²∂k_disp = 2 * HMₖH(Hv,nng⁻¹,mag,m,n)
	neff = k / ω
	# ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
	gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )

	Hₜ = reshape(Hv,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	EE = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	HH = inv(ω) * fft(tc(kx_ct( ifft( EE, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
	EEs = copy(reinterpret(reshape,SVector{3,Complex{T}},EE))
	HHs = copy(reinterpret(reshape,SVector{3,Complex{T}},HH))
	# Sz = dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
	Sz = getindex.(cross.(conj.(EEs),HHs),(3,))
	PP = 2*sum(Sz)
	# PP = 2*real( mapreduce((a,b)->dot(cross(conj(a),b),SVector(0.,0.,1.)),+,zip(EEs,HHs)))
	WW = dot(EE,_dot((ε+nng),EE))
	ng = real( WW / PP )

	return neff, ng, gvd
end

# function ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,H⃗,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
# 	ω² = ω^2
# 	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 	mag,m⃗,n⃗ = mag_m_n(k,grid)
# 	∂ω²∂k_nd = 2 * HMₖH(H⃗,ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
# 	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
# 	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
# 									 	(k,H⃗),
# 									  	∂ω²∂k_nd,
# 									   	ω,
# 									    ε⁻¹,
# 										grid; eigind)
# 	nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
# 	nngī_herm = (nngī2 .+ adjoint.(nngī2) ) / 2
# 	eī_herm = (eī₁ .+ adjoint.(eī₁) ) / 2
# 	return om̄₁, eī_herm, nngī_herm
# end

function ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid::Grid{ND,T}) where {ND,T<:Real}
	Nranges, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.ignore() do
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		# g⃗s = g⃗(grid)
		Nranges		=		eachindex(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		return (Nranges,Ninv,Ns,𝓕,𝓕⁻¹)
	end
	mag, m⃗, n⃗  = mag_m_n(k,grid)
	# mns = vcat(reshape(flat(m⃗),(1,3,Ns...)),reshape(flat(n⃗),(1,3,Ns...)))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
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
	k̄		= ∇ₖmag_m_n(
				māg_kx, 		# māg total
				m̄_kx.*mag, 	# m̄  total
				n̄_kx.*mag,	  	# n̄  total
				mag, m⃗, n⃗,
			)
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

function ∇solve_k(ΔΩ, Ω, ∂ω²∂k, ω, ε⁻¹, grid::Grid{ND,T}; eigind=1, λ⃗₀=nothing) where {ND,T<:Real}
	k̄ₖ, H̄ = ΔΩ
	k, Hv = Ω
	# Ninv, Ns, 𝓕 = Zygote.ignore() do
	# 	Ninv 		= 		1. / N(grid)
	# 	Ns			=		size(grid)
	# 	d0 = randn(Complex{T}, (3,Ns...))
	# 	𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
	# 	return (Ninv,Ns,𝓕)
	# end
	Ninv 		= 		1. / N(grid)
	Ns			=		size(grid)
	M̂ = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	g⃗s = g⃗(dropgrad(grid))
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	# mn = vcat(reshape(flat(m⃗),(1,3,Ns...)),reshape(flat(n⃗),(1,3,Ns...)))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
	if !iszero(H̄)
		# solve_adj!(λ⃗,M̂,H̄,ω^2,Hv,eigind)
		λ⃗	= eig_adjt(
				M̂,								 # Â
				ω^2, 							# α
				Hv, 					 		 # x⃗
				0.0, 							# ᾱ
				H̄;								 # x̄
				λ⃗₀,
				# P̂	= HelmholtzPreconditioner(M̂),
			)
		k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,Hv,grid)
	else
		eīₕ 	= zero(ε⁻¹) #fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
		k̄ₕ 	= 0.0
	end
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
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k ) # = 2ω * ω²̄
	return (NoTangent(), ZeroTangent(), ω̄  , eīₖ + eīₕ)
end

function ∇solve_k!(ΔΩ, Ω, ∂ω²∂k, ω::T, ε⁻¹, grid; eigind=1) where T<:Real
	k̄, H̄ = ΔΩ
	# println("k̄ = $(k̄)")
	k, Hv = Ω
	M̂ = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	H = reshape(Hv,(2,Ns...))
	g⃗s = g⃗(dropgrad(grid))
	mag = M̂.mag
	n⃗ 	 = M̂.n⃗
	m⃗ 	= M̂.m⃗
	λd = similar(M̂.d)
	λẽ = similar(M̂.d)
	ẽ = similar(M̂.d)
	eīₕ = similar(ε⁻¹)
	eīₖ = similar(ε⁻¹)
	λ⃗ = similar(Hv)
	λ = reshape(λ⃗,(2,Ns...))
	if iszero(k̄)
		k̄ = 0.
	end
	if !iszero(H̄) #typeof(H̄) != ZeroTangent()
		solve_adj!(λ⃗,M̂,H̄,ω^2,Hv,eigind)
		λ⃗ -= dot(Hv,λ⃗) * Hv
		d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
		λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
		ε⁻¹_bar!(eīₕ, vec(M̂.d), vec(λd), Ns...)
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λd *=  M̂.Ninv
		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,M̂ ) )
		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d,M̂) )
		m̄_kx = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )	#NB: m̄_kx and n̄_kx would actually
		n̄_kx =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
		māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
		k̄ₕ	= -∇ₖmag_m_n(māg_kx, 		# māg total
						m̄_kx.*mag, 	# m̄  total
						n̄_kx.*mag,	  	# n̄  total
						mag, m⃗, n⃗ )
	else
		eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
		k̄ₕ = 0.0
	end
	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
	λ = reshape(λ⃗,(2,Ns...))
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(eīₖ, vec(M̂.d), vec(λd),Ns...)
	ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k #[eigind]
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
	return (NoTangent(), ZeroTangent(), ω̄  , eīₖ + eīₕ)
end

# function rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
# 	kH⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log, f_filter)
#     solve_k_pullback(ΔΩ) = let kH⃗=kH⃗, ∂ω²∂k=ms.∂ω²∂k[eigind], ω=ω, ε⁻¹=ε⁻¹, grid=ms.grid, eigind=eigind
# 		∇solve_k!(ΔΩ,kH⃗,∂ω²∂k,ω,ε⁻¹,grid;eigind)
# 	end
#     return (kH⃗, solve_k_pullback)
# end

# 	println("#########  ∂ω²/∂k Adjoint Problem for kz = $( M̂.k⃗[3] ) ###########")
# 	uplot(ch;name="log10( adj. prob. res. )")
# 	println("\t\t\tadj converged?: $ch")
# 	println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")




# function rrule(::typeof(solve_k), ω::T,geom::Vector{<:Shape},gr::Grid{ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
#
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)
#
# 	Srvol,proc_sinds,mat_inds = ignore() do
# 		xyz = x⃗(gr)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 		xyzc = x⃗c(gr)
# 		ms = ModeSolver(kguess(ω,geom), geom, gr))
# 		corner_sinds!(ms.corner_sinds,geom,xyz,xyzc))
# 		proc_sinds!(ms.sinds_proc,ms.corner_sinds))
# 		Srvol(x) = let psinds=ms.sinds_proc, xyz=xyz, vxlmin=vxl_min(xyzc), vxlmax=vxl_max(xyzc)
# 			S_rvol(sinds_proc,xyz,vxlmin,vxlmax,x)
# 		end
# 		eism(om,x) =
# 		(Srvol, ms.sinds_proc)
# 	end
# 	# Srvol = S_rvol(proc_sinds,xyz,vxl_min(xyzc),vxl_max(xyzc),shapes)
# 	ε⁻¹ = εₛ⁻¹(ω,geom;ms=dropgrad(ms))
# 	kH⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
#     solve_k_pullback(ΔΩ) = let kH⃗=kH⃗, ∂ω²∂k=ms.∂ω²∂k, ω=ω, ε⁻¹=ε⁻¹, grid=ms.grid, eigind=eigind
# 		∇solve_k(ΔΩ,kH⃗,∂ω²∂k,ω,ε⁻¹,grid;eigind)
# 	end
#     return (kH⃗, solve_k_pullback)
# end


function rrule(::typeof(solve_k),ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};
	nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
	ε⁻¹, ε⁻¹_pb = Zygote.pullback(ω,p) do ω,p
		# smooth(ω,p,:fεs,true,geom_fn,grid)
		smooth(ω,p,(:fεs,:fnn̂gs),[true,false,],geom_fn,grid)[1]
	end
	ms = ModeSolver(k_guess(ω,ε⁻¹), ε⁻¹, grid; nev, maxiter, tol)
	k, Hv = solve_k(ms, ω; nev, eigind, maxiter, tol, log, f_filter)
	g⃗ = copy(ms.M̂.g⃗)
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
		mag_m_n(x,dropgrad(g⃗))
	end

	# Ns = copy(size(grid)) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	# Nranges = copy(eachindex(grid)) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	# println("\tsolve_k:")
	# println("\t\tω² (target): $(ω^2)")
	# println("\t\tω² (soln): $(ms.ω²[eigind])")
	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
	# println("\t\tk: $k")
	# println("\t\t∂ω²∂k: $∂ω²∂k")
	∂ω²∂k = copy(ms.∂ω²∂k[eigind])
	omsq_soln = copy(ms.ω²[eigind])
	# ε⁻¹_copy = copy(ε⁻¹)
	k = copy(k)
	Hv = copy(Hv)
    function solve_k_pullback(ΔΩ)
		k̄, H̄ = ΔΩ
		# println("\tsolve_k_pullback:")
		# println("k̄ (bar): $k̄")
		ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter, tol)
		update_k!(ms,k)
		update_ε⁻¹(ms,ε⁻¹) #ε⁻¹)
		ms.ω²[eigind] = omsq_soln # ω^2
		ms.∂ω²∂k[eigind] = ∂ω²∂k
		copyto!(ms.H⃗[:,eigind], Hv)
		# replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
		λ⃗ = similar(Hv)
		λd =  similar(ms.M̂.d)
		λẽ = similar(ms.M̂.d)
		Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
		Nranges = eachindex(grid)
		# ε⁻¹_bar = similar(ε⁻¹)
		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
		# Nranges = eachindex(ms.grid)

		H = reshape(Hv,(2,Ns...))
	    if iszero(k̄) # typeof(k̄)==ZeroTangent()
			k̄ = 0.
		end
		if !iszero(H̄) # if typeof(H̄) != ZeroTangent()
			# solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(Hv,H̄)*Hv
			solve_adj!(λ⃗,ms.M̂,H̄,omsq_soln,Hv,eigind;log=false)
			# solve_adj!(ms,H̄,ω^2,Hv,eigind)
			λ⃗ -= dot(Hv,λ⃗) * Hv
			λ = reshape(λ⃗,(2,Ns...))
			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
			# eīₕ = ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
			eīₕ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
			# eīₕ = copy(ε⁻¹_bar)
			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
			λd *=  ms.M̂.Ninv
			λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
			kx̄_m⃗ = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
			kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
			māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
			k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
		else
			eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
			k̄ₕ = 0.0
		end
		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
		copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
		λ = reshape(λ⃗,(2,Ns...))
		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
		λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
		# ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
		# eīₖ = copy(ε⁻¹_bar)
		eīₖ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
		# ε⁻¹_bar = eīₖ + eīₕ
		eibar = eīₖ + eīₕ
		ω̄_ε⁻¹, p̄ = ε⁻¹_pb(eibar)
		ω̄  =  ( 2ω * (k̄ + k̄ₕ ) / ∂ω²∂k ) + ω̄_ε⁻¹ #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
		# if !(typeof(k)<:SVector)
		# 	k̄_kx = k̄_kx[3]
		# end
		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄

		return (NoTangent(),  ω̄  , p̄, ZeroTangent(), ZeroTangent())
    end
    return ((k, Hv), solve_k_pullback)
end


# function rrule(::typeof(solve_k),ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};
# 	nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
function rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T};
		nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
		# ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T};nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing
	update_ε⁻¹(ms,ε⁻¹)
	k, Hv = solve_k(ms, ω; nev, eigind, maxiter, tol, log, f_filter)
	g⃗ = copy(ms.M̂.g⃗)
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
		mag_m_n(x,dropgrad(g⃗))
	end

	Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	# println("\tsolve_k:")
	# println("\t\tω² (target): $(ω^2)")
	# println("\t\tω² (soln): $(ms.ω²[eigind])")
	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
	# println("\t\tk: $k")
	# println("\t\t∂ω²∂k: $∂ω²∂k")
	∂ω²∂k = copy(ms.∂ω²∂k[eigind])
	omsq_soln = copy(ms.ω²[eigind])
	# ε⁻¹_copy = copy(ε⁻¹)
	k = copy(k)
	Hv = copy(Hv)
    function solve_k_pullback(ΔΩ)
		k̄, H̄ = ΔΩ
		# println("\tsolve_k_pullback:")
		# println("k̄ (bar): $k̄")
		update_k!(ms,k)
		update_ε⁻¹(ms,ε⁻¹) #ε⁻¹)
		ms.ω²[eigind] = omsq_soln # ω^2
		ms.∂ω²∂k[eigind] = ∂ω²∂k
		copyto!(ms.H⃗, Hv)
		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
		λ⃗ = similar(Hv)
		λd =  similar(ms.M̂.d)
		λẽ = similar(ms.M̂.d)
		# ε⁻¹_bar = similar(ε⁻¹)
		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
		# Nranges = eachindex(ms.grid)

		H = reshape(Hv,(2,Ns...))
	    # if typeof(k̄)==ZeroTangent()
		if isa(k̄,AbstractZero)
			k̄ = 0.
		end
		# if typeof(H̄) != ZeroTangent()
		if !isa(H̄,AbstractZero)
			# solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(Hv,H̄)*Hv
			solve_adj!(λ⃗,ms.M̂,H̄,omsq_soln,Hv,eigind;log=false)
			# solve_adj!(ms,H̄,ω^2,Hv,eigind)
			λ⃗ -= dot(Hv,λ⃗) * Hv
			λ = reshape(λ⃗,(2,Ns...))
			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
			# eīₕ = ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
			eīₕ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
			# eīₕ = copy(ε⁻¹_bar)
			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
			λd *=  ms.M̂.Ninv
			λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
			kx̄_m⃗ = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
			kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
			māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
			k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
		else
			eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
			k̄ₕ = 0.0
		end
		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
		copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
		λ = reshape(λ⃗,(2,Ns...))
		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
		λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
		# ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
		# eīₖ = copy(ε⁻¹_bar)
		eīₖ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
		# ε⁻¹_bar = eīₖ + eīₕ
		eibar = eīₖ + eīₕ
		ω̄  =  ( 2ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
		# if !(typeof(k)<:SVector)
		# 	k̄_kx = k̄_kx[3]
		# end
		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄


		return (NoTangent(), ZeroTangent(), ω̄  , eibar)
    end
    return ((k, Hv), solve_k_pullback)
end


# function rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
# 	k, Hv = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log, f_filter)
# 	# k, Hv = copy.(solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)) # ,ω²_tol)	 # returned data are refs to fields in ms struct. copy to preserve result for (possibly delayed) pullback closure.
# 	g⃗ = copy(ms.M̂.g⃗)
# 	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
# 		mag_m_n(x,dropgrad(g⃗))
# 	end
# 	∂ω²∂k = copy(ms.∂ω²∂k[eigind])
# 	Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 	Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
# 	# println("\tsolve_k:")
# 	# println("\t\tω² (target): $(ω^2)")
# 	# println("\t\tω² (soln): $(ms.ω²[eigind])")
# 	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
# 	# println("\t\tk: $k")
# 	# println("\t\t∂ω²∂k: $∂ω²∂k")
# 	omsq_soln = ms.ω²[eigind]
# 	ε⁻¹_copy = copy(ε⁻¹)
# 	k_copy = copy(k)
# 	Hv = copy(Hv)
#     function solve_k_pullback(ΔΩ)
# 		k̄, H̄ = ΔΩ
# 		# println("\tsolve_k_pullback:")
# 		# println("k̄ (bar): $k̄")
# 		update_k!(ms,k_copy)
# 		update_ε⁻¹(ms,ε⁻¹_copy) #ε⁻¹)
# 		ms.ω²[eigind] = omsq_soln # ω^2
# 		ms.∂ω²∂k[eigind] = ∂ω²∂k
# 		copyto!(ms.H⃗, Hv)
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
# 		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		# Nranges = eachindex(ms.grid)
#
# 		H = reshape(Hv,(2,Ns...))
# 	    if typeof(k̄)==ZeroTangent()
# 			k̄ = 0.
# 		end
# 		if typeof(H̄) != ZeroTangent()
# 			# solve_adj!(ms,H̄,eigind) 												# overwrite λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(Hv,H̄)*Hv
# 			solve_adj!(ms.λ⃗,ms.M̂,H̄,omsq_soln,Hv,eigind;log=false)
# 			# solve_adj!(ms,H̄,ω^2,Hv,eigind)
# 			ms.λ⃗ -= dot(Hv,ms.λ⃗) * Hv
# 			λ = reshape(ms.λ⃗,(2,Ns...))
# 			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 			λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 			ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd), Ns...)
# 			eīₕ = copy(ms.ε⁻¹_bar)
# 			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 			ms.λd *=  ms.M̂.Ninv
# 			λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
# 			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 			ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 			ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 			ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
# 			k̄ₕ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		else
# 			eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
# 			k̄ₕ = 0.0
# 		end
# 		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 		copyto!(ms.λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
# 		λ = reshape(ms.λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd),Ns...)
# 		eīₖ = copy(ms.ε⁻¹_bar)
# 		ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
# 		ε⁻¹_bar = eīₖ + eīₕ
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NoTangent(), ZeroTangent(), ω̄  , ε⁻¹_bar)
#     end
#     return ((k, Hv), solve_k_pullback)
# end




# ## prototype backprop through NLsolve of PDE
# # from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205

# using NLsolve
# using Zygote
# using Zygote: @adjoint
# using IterativeSolvers
# using LinearMaps
# using SparseArrays
# using LinearAlgebra
# using BenchmarkTools

# # nlsolve maps f to the solution x of f(x) = 0
# # We have ∂x = -(df/dx)^-1 ∂f, and so the adjoint is df = -(df/dx)^-T dx
# @adjoint nlsolve(f, x0; kwargs...) =
#     let result = nlsolve(f, x0; kwargs...)
#         result, function(vresult)
#             dx = vresult[].zero
#             x = result.zero
#             _, back_x = Zygote.pullback(f, x)

#             JT(df) = back_x(df)[1]
#             # solve JT*df = -dx
#             L = LinearMap(JT, length(x0))
#             df = gmres(L,-dx)

#             _, back_f = Zygote.pullback(f -> f(x), f)
#             return (back_f(df)[1], nothing, nothing)
#         end
#     end

# const NN = 10000
# const nonlin = 0.1
# const A = spdiagm(0 => fill(10.0, NN), 1 => fill(-1.0, NN-1), -1 => fill(-1.0, NN-1))
# const p0 = randn(NN)
# f(x, p) = A*x + nonlin*x.^2 - p
# solve_x(p) = nlsolve(x -> f(x, p), zeros(NN), method=:anderson, m=10).zero
# obj(p) = sum(solve_x(p))

# Zygote.refresh()
# g_auto, = gradient(obj, p0)
# g_analytic = gmres((A + Diagonal(2*nonlin*solve_x(p0)))', ones(NN))
# display(g_auto)
# display(g_analytic)
# @show sum(abs.(g_auto - g_analytic))

# @btime gradient(obj, p0);
# @btime gmres((A + Diagonal(2*nonlin*solve_x(p0)))', ones(NN));