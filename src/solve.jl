export solve_ω², _solve_Δω², solve_k, solve_k_single, filter_eigs
export AbstractEigensolver

"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""

# add try/catch with
# res = DFTK.LOBPCG(ms.M̂,rand(ComplexF64,size(ms.M̂)[1],1),I,ms.P̂,1e-8,3000)

abstract type AbstractEigensolver end
# struct IterativeSolversLOBPCG <: Eigensolver end
# struct DFTK_LOBPCG <: Eigensolver end







# function __init__()
#     
# end

# function __init__()
#     @require PyCall="" include("solvers/mpb.jl")
# end

# abstract type AbstractLinearSolver end

function solve_ω²(ms::ModeSolver{ND,T},solver::TS;kwargs...) where {ND,T<:Real,TS<:AbstractEigensolver} end


# function _solve_ω²(ms::ModeSolver{ND,T},solver::MPB_Solver;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
# 	x₀ 		=	vec(ms.H⃗[:,1])	 # initial eigenvector guess, should be complex vector with length 2*prod(size(grid))
# 	howmany =	nev				# how many eigenvector/value pairs to find
# 	which	=	:SR				# :SR="Smallest Real" eigenvalues
# 	# evals,evecs,convinfo = eigsolve(x->ms.M̂*x,ms.H⃗[:,1],nev,:SR; maxiter, tol, krylovdim=50, verbosity=2)
# 	evals,evecs,info = eigsolve(x->ms.M̂*x,x₀,howmany,which;maxiter,tol,krylovdim=50) #,verbosity=2)
# 	info.converged < howmany && @warn "KrylovKit.eigsolve only found $(info.converged) eigenvector/value pairs while attempting to find $howmany"
# 	println("evals: $evals")
# 	n_results = min(nev,info.converged) # min(size(ms.H⃗,2),info.converged)
# 	evals_res = evals[1:n_results]
# 	evecs_res = vec.(evecs[1:n_results])
# 	copyto!(ms.ω²,evals_res)
# 	copyto!(ms.H⃗[:,1:n_results],hcat(evecs_res...))
# 	return real(evals_res), evecs_res
# end

# function _solve_ω²(ms::ModeSolver{ND,T},solver::DFTK_LOBPCG;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)
# 	x₀ 		=	vec(ms.H⃗[:,1])	 # initial eigenvector guess, should be complex vector with length 2*prod(size(grid))
# 	howmany =	nev				# how many eigenvector/value pairs to find
# 	which	=	:SR				# :SR="Smallest Real" eigenvalues
#   res		=	DFTK.LOBPCG(ms.M̂,rand(ComplexF64,size(ms.M̂)[1],1),I,ms.P̂,1e-8,3000)
# 	evals,evecs,convinfo = eigsolve(x->ms.M̂*x,ms.H⃗,size(ms.H⃗,2),:SR;maxiter,tol,krylovdim=50) #,verbosity=2)
# 	copyto!(ms.H⃗,hcat(evecs...)[1:size(ms.H⃗,2)])
# 	copyto!(ms.ω²,evals[1:size(ms.H⃗,2)])
# end

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

# function solve_ω²(ms::ModeSolver{ND,T},solver::AbstractEigensolver;nev=1,maxiter=200,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
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

function solve_ω²(ms::ModeSolver{ND,T},k::TK,solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	@ignore(update_k!(ms,k))
	solve_ω²(ms,solver; nev, maxiter, tol, log, solver, f_filter)
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND},solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms,solver; nev, maxiter, tol, log, solver, f_filter)
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
function _solve_Δω²(ms::ModeSolver{ND,T},k::TK,ωₜ::T,solver::AbstractEigensolver;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK}
	# println("k: $(k)")
	evals,evecs = solve_ω²(ms,k,solver; nev, maxiter, tol, log, solver, f_filter)
	Δω² = evals[eigind] - ωₜ^2
	∂ω²∂k = 2 * HMₖH(evecs[eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	ms.∂ω²∂k[eigind] = ∂ω²∂k
	println("Δω²: $(Δω²)")
	println("∂ω²∂k: $(∂ω²∂k)")
    return Δω² , ( Δω² / ∂ω²∂k )
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k_single(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,eigind=1,maxiter=100,tol=1e-8,atol=tol,maxevals=60,log=false,f_filter=nothing) where {ND,T<:Real} #
    kmag = Roots.find_zero(x -> _solve_Δω²(ms,x,ω,solver;nev,eigind,maxiter,tol,f_filter), ms.M̂.k⃗[3], Roots.Newton(); atol,maxevals,verbose=true,)
	return kmag, copy(ms.H⃗[:,eigind])
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,atol=tol,maxevals=60,log=false,f_filter=nothing) where {ND,T<:Real} #
	kmags = Vector{T}(undef,nev)
	evecs = Matrix{Complex{T}}(undef,(size(ms.H⃗,1),nev))
	for (idx,eigind) in enumerate(1:nev)
		idx>1 && copyto!(ms.H⃗,repeat(evecs[:,idx-1],1,size(ms.H⃗,2)))
		kmag_evec = solve_k_single(ms,ω,solver;nev,eigind,maxiter,tol,atol,maxevals,log)
		kmags[idx] = first(kmag_evec)
		copyto!(evecs[:,idx],last(kmag_evec))
	end
	return kmags, collect(eachcol(evecs))
end


function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} 
	Zygote.@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω, solver; nev, maxiter, tol, log, f_filter)
end

function solve_k(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = reshape(Hguess,size(ms.H⃗))
		end
		return ms
	end
	solve_k(ms, ω, solver; nev, maxiter, tol, log, f_filter)
end

function solve_k(ω::AbstractVector{T},p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
	# ms = @ignore(ModeSolver(k_guess(first(ω),first(ε⁻¹)), first(ε⁻¹), grid; nev, maxiter, tol))
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = Hguess
		end
		return ms
	end
	nω = length(ω)
	k = Buffer(ω,nω)
	Hv = Buffer([1.0 + 3.0im, 2.1+4.0im],(size(ms.M̂)[1],nω))
	for ωind=1:nω
		@ignore(update_ε⁻¹(ms,ε⁻¹[ωind]))
		kHv = solve_k(ms,ω[ωind],solver; nev, maxiter, tol, log, f_filter)
		k[ωind] = kHv[1]
		Hv[:,ωind] = kHv[2]
	end
	return copy(k), copy(Hv)
end


# # function find_k(ω::Real,ε::AbstractArray,grid::Grid{ND};num_bands=2,band_min=1,band_max=num_bands,filename_prefix="f01",data_path=pwd(),kwargs...) where ND
# function rrule(::typeof(find_k),ω::Real,ε⁻¹,grid::Grid{ND};nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing) where {ND}
# 		# ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T};nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing
# 	kmags, evecs = find_k(ω,ε,grid);
	
# 	function find_k_pullback(ΔΩ)
# 		k̄, ēvecs = ΔΩ
# 		ε⁻¹ = sliceinv_3x3(ε)
# 		ms = ModeSolver(kmags[1], ε⁻¹, grid; nev, maxiter, tol)
# 		# println("\tsolve_k_pullback:")
# 		# println("k̄ (bar): $k̄")
# 		update_k!(ms,k)
# 		update_ε⁻¹(ms,ε⁻¹) #ε⁻¹)
# 		ms.ω²[eigind] = omsq_soln # ω^2
# 		ms.∂ω²∂k[eigind] = ∂ω²∂k
# 		copyto!(ms.H⃗, Hv)
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		λ⃗ = similar(Hv)
# 		λd =  similar(ms.M̂.d)
# 		λẽ = similar(ms.M̂.d)
# 		# ε⁻¹_bar = similar(ε⁻¹)
# 		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
# 		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		# Nranges = eachindex(ms.grid)

# 		H = reshape(Hv,(2,Ns...))
# 		# if typeof(k̄)==ZeroTangent()
# 		if isa(k̄,AbstractZero)
# 			k̄ = 0.
# 		end
# 		# if typeof(ēvecs) != ZeroTangent()
# 		if !isa(ēvecs,AbstractZero)
# 			# solve_adj!(ms,ēvecs,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēvecs - dot(Hv,ēvecs)*Hv
# 			solve_adj!(λ⃗,ms.M̂,ēvecs,omsq_soln,Hv,eigind;log=false)
# 			# solve_adj!(ms,ēvecs,ω^2,Hv,eigind)
# 			λ⃗ -= dot(Hv,λ⃗) * Hv
# 			λ = reshape(λ⃗,(2,Ns...))
# 			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 			λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 			# eīₕ = ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
# 			eīₕ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
# 			# eīₕ = copy(ε⁻¹_bar)
# 			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 			λd *=  ms.M̂.Ninv
# 			λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
# 			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 			kx̄_m⃗ = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 			kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 			māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# 			k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		else
# 			eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
# 			k̄ₕ = 0.0
# 		end
# 		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 		copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
# 		λ = reshape(λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		# ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
# 		# eīₖ = copy(ε⁻¹_bar)
# 		eīₖ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
# 		# ε⁻¹_bar = eīₖ + eīₕ
# 		eibar = eīₖ + eīₕ
# 		ω̄  =  ( 2ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NoTangent(), ZeroTangent(), ω̄  , eibar)
# 	end

# 	return ((k, Hv), solve_k_pullback)
# end