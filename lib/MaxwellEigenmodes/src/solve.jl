export solve_ω², _solve_Δω², solve_k, solve_k_single, filter_eigs
export AbstractEigensolver

"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""
abstract type AbstractEigensolver{L<:AbstractLogger} end

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


function solve_ω²(ms::ModeSolver{ND,T},k::TK,solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,
	log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	ignore_derivatives() do; update_k!(ms,k); end
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	ignore_derivatives() do
		update_k!(ms,k)
		update_ε⁻¹(ms,ε⁻¹)
	end
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(k::TK,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,maxiter=100,
	tol=1e-8,log=false,evecs_guess=nothing,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	ms = ignore_derivatives() do
		ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(evecs_guess)
			ms.H⃗ = reshape(hcat(evecs_guess...),(2*length(grid),nev))
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
	evals,evecs = solve_ω²(ms,k,solver; nev, maxiter, tol=eig_tol, log, f_filter)
	evec_out[:] = copy(evecs[eigind]) #copyto!(evec_out,evecs[eigind])
	Δω² = evals[eigind] - ωₜ^2
	∂ω²∂k = 2 * HMₖH(evec_out,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	ms.∂ω²∂k[eigind] = ∂ω²∂k
	ms.ω²[eigind] = evals[eigind]
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


function solve_k(ms::ModeSolver{ND,T},ω::T,solver::TS;nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,
	max_eigsolves=60,log=false,f_filter=nothing) where {ND,T<:Real,TS<:AbstractEigensolver{L} where L<:HDF5Logger} #
	@debug "Using HDF5-logging solve_k method"
	kmags = Vector{T}(undef,nev)
	evecs = Matrix{Complex{T}}(undef,(size(ms.H⃗,1),nev))
	for (idx,eigind) in enumerate(1:nev)
		# idx>1 && copyto!(ms.H⃗,repeat(evecs[:,idx-1],1,size(ms.H⃗,2)))
		kmag, evec = solve_k_single(ms,ω,solver;nev,eigind,maxiter,max_eigsolves,k_tol,eig_tol,log)
		kmags[idx] = kmag
		evecs[:,idx] =  canonicalize_phase(evec,kmag,ms.M̂.ε⁻¹,ms.grid)
	end
    with_logger(solver.logger) do
        solver_str = string(solver)
		k_dir = norm(Vector{Float64}(ms.M̂.k⃗))
		ε = sliceinv_3x3(ms.M̂.ε⁻¹) 
        @debug "solve_k" ω ε kmags evecs nev k_dir maxiter k_tol eig_tol max_eigsolves solver_str
    end
	return kmags, collect(copy.(eachcol(evecs))) #evecs #[copy(ev) for ev in eachcol(evecs)] #collect(eachcol(evecs))
end

function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60, maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} 
	ignore_derivatives() do; update_ε⁻¹(ms,ε⁻¹); end
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter)
end

"""
    solve_k(ω, ε⁻¹, grid, solver; nev=1, k_tol=1e-8, eig_tol=1e-8, ...)
        -> (kmags, evecs)

Find the propagation constants and fields of the first `nev` guided eigenmodes at
(temporal) frequency `ω` (μm⁻¹, with ``ω = 1/λ`` and ``c = 1``): the values
``|k|`` along ``\\hat{z}`` and transverse eigenvectors ``H`` satisfying

```math
\\hat{M}(k)\\,\\vec{H} = ω^2\\,\\vec{H},
```

where ``\\hat{M}`` is the plane-wave Maxwell operator ([`HelmholtzMap`](@ref)) built
from the smoothed inverse-permittivity field `ε⁻¹` (`(3,3,size(grid)...)`, e.g.
`sliceinv_3x3` of a `DielectricSmoothing.smooth_ε` slice).

Since eigensolvers naturally compute ``ω^2(k)`` ([`solve_ω²`](@ref)) but waveguide
problems fix ``ω``, the dispersion relation is *inverted* with a Newton iteration

```math
k_{j+1} = k_j - \\frac{ω^2(k_j) - ω^2}{∂ω^2/∂k|_{k_j}},
```

where the exact slope ``∂ω^2/∂k = \\langle H|\\hat{M}_k|H\\rangle`` comes from the
Hellmann–Feynman theorem ([`HMₖH`](@ref)) at no extra cost — each Newton step costs one
eigensolve, warm-started from the previous step. Returns mode effective indices via
`neff = kmags ./ ω`; eigenvectors are phase-canonicalized (largest E-field component
real and positive).

Keyword arguments: `nev` number of bands; `k_tol` Newton tolerance on `|k|`;
`eig_tol` eigensolver tolerance; `max_eigsolves` Newton iteration budget; `kguess`,
`Hguess` optional warm starts (used by some backends).

A `ChainRulesCore.rrule` implements the adjoint method for this function, giving
gradients of `(kmags, evecs)` w.r.t. `ω` and `ε⁻¹` at ≈1 extra eigensolve-equivalent
cost; see `eig_adjt`. Solver backends: `KrylovKitEigsolve` (native CPU), `DFTK_LOBPCG`,
`IterativeSolversLOBPCG`, `GPUSolver` (CUDA/Float32/Float64), `MPBSolver` (Python MPB).
"""
function solve_k(ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing,overwrite=false) where {ND,T<:Real}
	# `kguess`/`Hguess` are warm starts: an initial |k| for the Newton inversion and an
	# initial eigenvector basis for the eigensolver. They only seed the iterations — both
	# converge to the same eigenpair to tolerance — so the returned value (and its adjoint)
	# is independent of them. Supplying the neighbouring cell/frequency solution cuts the
	# Newton- and Lanczos-iteration counts (see `solve_cell_modes`' warm-start path).
	k0 = kguess === nothing ? k_guess(ω,ε⁻¹) : T(first(kguess))
	ms = ModeSolver(k0, ε⁻¹, grid; nev, maxiter, tol=eig_tol)
	if Hguess !== nothing
		_apply_Hguess!(ms.H⃗, Hguess)
	end
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter,)
end

# Write a warm-start eigenvector basis into `H⃗` (columns = bands), tolerating a single
# vector, a vector-of-vectors, or a matrix, and a mismatched band count (extra columns
# keep their random initialisation; missing bands are ignored).
function _apply_Hguess!(H⃗::AbstractMatrix, Hguess)
	cols = Hguess isa AbstractMatrix ? eachcol(Hguess) :
	       (eltype(Hguess) <: Number ? (Hguess,) : Hguess)
	for (j, h) in enumerate(cols)
		j ≤ size(H⃗, 2) || break
		length(h) == size(H⃗, 1) || continue
		@views H⃗[:, j] .= h
	end
	return H⃗
end


function rrule(::typeof(solve_k), ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::TS;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing,overwrite=false) where {ND,T<:Real,TS<:AbstractEigensolver} 
	
	kmags,evecs = solve_k(ω, ε⁻¹, grid, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, kguess, Hguess,
	f_filter, overwrite)
	# g⃗ = copy(ms.M̂.g⃗)
	# (mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
	# 	mag_m_n(x,dropgrad(g⃗))
	# end
	gridsize = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	# ε⁻¹_copy = copy(ε⁻¹)
	function solve_k_pullback(ΔΩ)
		ei_bar = zero(ε⁻¹)
		ω_bar = zero(ω)
		k̄mags, ēvecs = ΔΩ
		for (eigind, k̄, ēv, k, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
			ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter)
			ms.H⃗[:,eigind] = copy(ev)
			# replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
			λ⃗ = randn(eltype(ev),size(ev)) # similar(ev)
			λd =  similar(ms.M̂.d)
			λẽ = similar(ms.M̂.d)
			∂ω²∂k = 2 * HMₖH(ev,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
			ev_grid = reshape(ev,(2,gridsize...))
			if isa(k̄,AbstractZero)
				k̄ = 0.0
			end
			if !isa(ēv,AbstractZero)
				# solve_adj!(ms,ēv,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēv - dot(ev,ēv)*ev
				λ⃗ = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=randn(eltype(ev),size(ev)), P̂=ms.P̂)
				# @show val_magmax(λ⃗)
				# @show dot(ev,λ⃗)
				λ⃗ 	-= 	 dot(ev,λ⃗) * ev
				λ	=	reshape(λ⃗,(2,gridsize...))
				d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
				λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
				ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), gridsize...)
				# @show val_magmax(ei_bar)
				### back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
				λd *=  ms.M̂.Ninv
				λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
				ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
				# @show val_magmax(λẽ)
				# @show val_magmax(reinterpret(reshape,Complex{T},ẽ))
				kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ .* conj.(view(λ,2,axes(grid)...)) )
				kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ .* conj.(view(λ,1,axes(grid)...)) )
				m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,axes(grid)...])
				n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,axes(grid)...])
				māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
				# @show k̄ₕ_old = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
				k̄ₕ = -∇ₖmag_m_n(
					māg,
					kx̄_m⃗.*ms.M̂.mag, # m̄,
					kx̄_n⃗.*ms.M̂.mag, # n̄,
					ms.M̂.mag,
					m⃗,
					n⃗;
					dk̂=SVector(0.,0.,1.), # dk⃗ direction
				)
				# @show k̄ₕ
			else
				# eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
				k̄ₕ = 0.0
			end
			# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω_bar and eīₖ
			λ⃗ = ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * ev
			d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,reshape(λ⃗,(2,gridsize...)),ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )			
			ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₖ # epsinv_bar = eīₖ + eīₕ
			ω_bar +=  ( 2*ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  
			# @show ω_bar
		end
		return (NoTangent(), ω_bar , ei_bar, NoTangent(), NoTangent())
	end
	return ((kmags, evecs), solve_k_pullback)
end


function rrule(::typeof(solve_k), ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::TS;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing,overwrite=false) where {ND,T<:Real,TS<:AbstractEigensolver{L} where L<:HDF5Logger} 
	
	kmags,evecs = solve_k(ω, ε⁻¹, grid, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, kguess, Hguess,
	f_filter, overwrite)

	solver_logger= solver.logger

	# g⃗ = copy(ms.M̂.g⃗)
	# (mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
	# 	mag_m_n(x,dropgrad(g⃗))
	# end
	gridsize = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	# ε⁻¹_copy = copy(ε⁻¹)
	function solve_k_pullback(ΔΩ)
		ei_bar = zero(ε⁻¹)
		ω_bar = zero(ω)
		k̄mags, ēvecs = ΔΩ
		# λ₀s = zeros(ComplexF64,(size(first(evecs),1),nev))
		λ₀s = zeros(ComplexF64,(2*N(grid),nev))
		evecs_bar = zeros(ComplexF64,(2*N(grid),nev))
		kmags_bar = zeros(ComplexF64,nev)
		∂ω²∂ks = zeros(Float64,nev)
		for (eigind, k̄, ēv, k, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
			ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter)
			ms.H⃗[:,eigind] = copy(ev)
			# replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
			λ⃗ = randn(eltype(ev),size(ev)) # similar(ev)
			λd =  similar(ms.M̂.d)
			λẽ = similar(ms.M̂.d)
			∂ω²∂k = 2 * HMₖH(ev,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
			∂ω²∂ks[eigind] = ∂ω²∂k
			ev_grid = reshape(ev,(2,gridsize...))
			
			if isa(k̄,AbstractZero)
				k̄ = 0.0
			else
				kmags_bar[eigind] = k̄
			end
			if !isa(ēv,AbstractZero)
				evecs_bar[:,eigind] = ēv
				# solve_adj!(ms,ēv,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēv - dot(ev,ēv)*ev
				λ⃗ = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=randn(eltype(ev),size(ev)), P̂=ms.P̂)
				# @show val_magmax(λ⃗)
				# @show dot(ev,λ⃗)
				λ⃗ 	-= 	 dot(ev,λ⃗) * ev
				λ₀s[:,eigind] = λ⃗ 
				λ	=	reshape(λ⃗,(2,gridsize...))
				d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
				λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
				ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), gridsize...)
				# @show val_magmax(ei_bar)
				### back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
				λd *=  ms.M̂.Ninv
				λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
				ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
				# @show val_magmax(λẽ)
				# @show val_magmax(reinterpret(reshape,Complex{T},ẽ))
				kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ .* conj.(view(λ,2,axes(grid)...)) )
				kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ .* conj.(view(λ,1,axes(grid)...)) )
				m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,axes(grid)...])
				n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,axes(grid)...])
				māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
				# @show k̄ₕ_old = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
				k̄ₕ = -∇ₖmag_m_n(
					māg,
					kx̄_m⃗.*ms.M̂.mag, # m̄,
					kx̄_n⃗.*ms.M̂.mag, # n̄,
					ms.M̂.mag,
					m⃗,
					n⃗;
					dk̂=SVector(0.,0.,1.), # dk⃗ direction
				)
				# @show k̄ₕ
			else
				# eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
				k̄ₕ = 0.0
			end
			# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω_bar and eīₖ
			λ⃗ = ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * ev
			d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,reshape(λ⃗,(2,gridsize...)),ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )			
			ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₖ # epsinv_bar = eīₖ + eīₕ
			ω_bar +=  ( 2*ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  
			# @show ω_bar
		end

		@show kmags_bar 
		@show evecs_bar 
		@show ω_bar 
		@show ei_bar 
		@show λ₀s 
		@show ∂ω²∂ks 
		@show ω 
		@show kmags 
		@show nev 

		let kmags_bar=kmags_bar, evecs_bar=evecs_bar, ω_bar=ω_bar, ei_bar=ei_bar, λ₀s=λ₀s, ∂ω²∂ks=∂ω²∂ks, ω=ω, kmags=kmags, nev=nev, logger=solver_logger
			with_logger(logger) do
				@debug "solve_k_pullback" kmags_bar evecs_bar ω_bar ei_bar λ₀s ∂ω²∂ks ω kmags nev 
			end
		end
		return (NoTangent(), ω_bar , ei_bar, NoTangent(), NoTangent())
	end
	return ((kmags, evecs), solve_k_pullback)
end









