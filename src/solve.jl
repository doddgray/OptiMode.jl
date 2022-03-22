export solve_ω², _solve_Δω², solve_k, solve_k_single, filter_eigs
export AbstractEigensolver

"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""
abstract type AbstractEigensolver end

# abstract type AbstractLinearSolver end

"""
	solve_ω²(ms::ModeSolver, solver::AbstractEigensolver; kwargs...)

	Find a few extremal eigenvalue/eigenvector pairs of the `HelmholtzOperator` map 
	in the modesolver object `ms`. The eigenvalues physically correspond to ω², the
	square of the temporal frequencies of electromagnetic resonances (modes) of the
	dielectric structure being modeled with Bloch wavevector k⃗, a 3-vector of spatial
	frequencies.
"""
function solve_ω²(ms::ModeSolver{ND,T}, solver::TS; kwargs...)::Tuple{Vector{T},Vector{Vector{Complex{T}}}} where {ND,T<:Real,TS<:AbstractEigensolver} end

"""
f_filter takes in a ModeSolver and an eigenvalue/vector pair `αX` and outputs boolean,
ex. f_filter = (ms,αX)->sum(abs2,𝓟x(ms.grid)*αX[2])>0.9
where the modesolver `ms` is passed for access to any auxilary information
"""
function filter_eigs(ms::ModeSolver{ND,T},f_filter::Function)::Tuple{Vector{T},Matrix{Complex{T}}} where {ND,T<:Real}
	ω²H_filt = filter(ω²H->f_filter(ms,ω²H), [(real(ms.ω²[i]),ms.H⃗[:,i]) for i=1:length(ms.ω²)] )
	return copy(getindex.(ω²H_filt,1)), copy(hcat(getindex.(ω²H_filt,2)...)) # ω²_filt, H_filt
	# return getindex.(ω²H_filt,1), hcat(getindex.(ω²H_filt,2)...) # ω²_filt, H_filt
end

# # function _solve_ω²(ms::ModeSolver{ND,T},::;nev=1,eigind=1,maxiter=100,tol=1.6e-8

# function solve_ω²(ms::ModeSolver{ND,T},solver::AbstractEigensolver;nev=1,maxiter=200,k_tol=1e-8,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
# 	evals,evecs = _solve_ω²(ms,solver;nev,eigind,maxiter,tol,log,f_filter)
# 	# @assert isequal(size(ms.H⃗,2),nev) # check that the modesolver struct is consistent with the number of eigenvalue/vector pairs `nev`
# 	# evals_res = evals[1:nev]
# 	# evecs_res = vec.(evecs[1:nev])
# 	# copyto!(ms.H⃗,hcat(evecs_res...)) 
# 	# copyto!(ms.ω²,evals_res)
	
# 	# res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol)

# 	# res = LOBPCG(ms.M̂,ms.H⃗,I,ms.P̂,tol,maxiter)
# 	# copyto!(ms.H⃗,res.X)
# 	# copyto!(ms.ω²,res.λ)


# 	# if isnothing(f_filter)
# 	# 	return   (copy(real(ms.ω²)), copy(ms.H⃗))
# 	# else
# 	# 	return filter_eigs(ms, f_filter)
# 	# end
# 	return evals, evecs
# end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,
	log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	# @ignore(update_k!(ms,k))
	update_k!(ms,k)
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(k::TK,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,maxiter=100,
	tol=1e-8,log=false,evecs_guess=nothing,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	ms = ignore() do
		ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = reshape(Hguess,(2*length(grid),2))
		end
		return ms
	end
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

"""
################################################################################
#																			   #
#						solve_k methods: (ε⁻¹, ω) --> (H, k)				   #
#																			   #
################################################################################
"""


"""
modified solve_ω version for Newton solver, which wants (x -> f(x), f(x)/f'(x)) as input to solve f(x) = 0
"""
function _solve_Δω²(ms::ModeSolver{ND,T},k::TK,ωₜ::T,evec_out::Vector{Complex{T}},solver::AbstractEigensolver;nev=1,
	eigind=1,maxiter=100,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK}
	# println("k: $(k)")
	evals,evecs = solve_ω²(ms,k,solver; nev, maxiter, tol=eig_tol, log, f_filter)
	evec_out[:] = copy(evecs[eigind]) #copyto!(evec_out,evecs[eigind])
	Δω² = evals[eigind] - ωₜ^2
	# ∂ω²∂k = 2 * HMₖH(evecs[eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	∂ω²∂k = 2 * HMₖH(evec_out,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	ms.∂ω²∂k[eigind] = ∂ω²∂k
	ms.ω²[eigind] = evals[eigind]
	# println("Δω²: $(Δω²)")
	# println("∂ω²∂k: $(∂ω²∂k)")
    return Δω² , ( Δω² / ∂ω²∂k )
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k_single(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,eigind=1,
	maxiter=100,max_eigsolves=60,k_tol=1e-10,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} #
    evec_out = Vector{Complex{T}}(undef,size(ms.H⃗,1))
	kmag = Roots.find_zero(
		x -> _solve_Δω²(ms,x,ω,evec_out,solver;nev,eigind,maxiter,eig_tol,f_filter),	# f(x), it will find zeros of this function
		ms.M̂.k⃗[3],				  # initial guess, previous |k|(ω) solution
		Roots.Newton(); 			# iterative zero-finding algorithm
		atol=k_tol,					# absolute |k| convergeance tolerance 
		maxevals=max_eigsolves,		# max Newton iterations before it gives up
		#verbose=true,
	)
	return kmag, evec_out #copy(ms.H⃗[:,eigind])
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,
	max_eigsolves=60,log=false,f_filter=nothing) where {ND,T<:Real} #
	kmags = Vector{T}(undef,nev)
	evecs = Matrix{Complex{T}}(undef,(size(ms.H⃗,1),nev))
	for (idx,eigind) in enumerate(1:nev)
		# idx>1 && copyto!(ms.H⃗,repeat(evecs[:,idx-1],1,size(ms.H⃗,2)))
		kmag, evec = solve_k_single(ms,ω,solver;nev,eigind,maxiter,max_eigsolves,k_tol,eig_tol,log)
		kmags[idx] = kmag
		evecs[:,idx] =  canonicalize_phase(evec,kmag,ms.M̂.ε⁻¹,ms.grid)
	end
	return kmags, collect(copy.(eachcol(evecs))) #evecs #[copy(ev) for ev in eachcol(evecs)] #collect(eachcol(evecs))
end

function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60, maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} 
	Zygote.@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter)
end

function solve_k(ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing) where {ND,T<:Real} 
	# ms = ignore() do
	# 	kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
	# 	ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, eig_tol)
	# 	if !isnothing(Hguess)
	# 		ms.H⃗ = reshape(Hguess,size(ms.H⃗))
	# 	end
	# 	return ms
	# end
	ms = ModeSolver(k_guess(ω,ε⁻¹), ε⁻¹, grid; nev, maxiter, tol=eig_tol)
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter,)
end





# function solve_k(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
# 	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
# 	ms = ignore() do
# 		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
# 		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
# 		if !isnothing(Hguess)
# 			ms.H⃗ = reshape(Hguess,size(ms.H⃗))
# 		end
# 		return ms
# 	end
# 	solve_k(ms, ω, solver; nev, maxiter, tol, log, f_filter)
# end

# function solve_k(ω::AbstractVector{T},p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
# 	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
# 	# ms = @ignore(ModeSolver(k_guess(first(ω),first(ε⁻¹)), first(ε⁻¹), grid; nev, maxiter, tol))
# 	ms = ignore() do
# 		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
# 		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
# 		if !isnothing(Hguess)
# 			ms.H⃗ = Hguess
# 		end
# 		return ms
# 	end
# 	nω = length(ω)
# 	k = Buffer(ω,nω)
# 	Hv = Buffer([1.0 + 3.0im, 2.1+4.0im],(size(ms.M̂)[1],nω))
# 	for ωind=1:nω
# 		@ignore(update_ε⁻¹(ms,ε⁻¹[ωind]))
# 		kHv = solve_k(ms,ω[ωind],solver; nev, maxiter, tol, log, f_filter)
# 		k[ωind] = kHv[1]
# 		Hv[:,ωind] = kHv[2]
# 	end
# 	return copy(k), copy(Hv)
# end













# function ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
# 	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
# 	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
# 	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
# 	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
# 	return +( k̄_mag, k̄_m, k̄_n )
# end



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
	# return eī # inv( (eps' + eps) / 2)
	return (real(eī) + permutedims(real(eī),(2,1,3,4)))/2.0
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
		Pl = HelmholtzPreconditioner(M̂), # left preconditioner
		log,
		abstol=1e-10,
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
export eig_adjt, linsolve, solve_adj!
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





