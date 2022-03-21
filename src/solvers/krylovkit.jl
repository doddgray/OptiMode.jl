# following the solver-struct convention used in BifurcationKit.jl, eg:
# https://github.com/rveltz/BifurcationKit.jl/blob/master/src/LinearSolver.jl

####################################################################################################
# KrylovKit Solvers
####################################################################################################

# using .KrylovKit	# uncomment if using Requires.jl for optional dependancy

export KrylovKitEigsolve, solve_ω²

# """
# $(TYPEDEF)
# Create a linear solver based on GMRES from `KrylovKit.jl`. Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.
# $(TYPEDFIELDS)
# !!! tip "Different linear solvers"
#     By tuning the options, you can select CG, GMRES... see [here](https://jutho.github.io/KrylovKit.jl/stable/man/linear/#KrylovKit.linsolve)
# """
mutable struct KrylovKitEigsolve <: AbstractEigensolver end

function solve_ω²(ms::ModeSolver{ND,T},solver::KrylovKitEigsolve;nev=1,eigind=1,maxiter=200,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
	# x₀ 		=	 # copy(vec(ms.H⃗[:,1]))	 # initial eigenvector guess, should be complex vector with length 2*prod(size(grid))
	# howmany =	nev				# how many eigenvector/value pairs to find
	# which	=	:SR				# :SR="Smallest Real" eigenvalues
	# evals,evecs,convinfo = eigsolve(x->ms.M̂*x,copy(ms.H⃗[:,1]),nev,:SR; maxiter, tol, krylovdim=50) # , verbosity=2)
	evals,evecs,convinfo = eigsolve(x->ms.M̂*x,rand(Complex{T},size(ms.H⃗[:,1])),nev,:SR; maxiter, tol, krylovdim=50) # , verbosity=2)
	# evals,evecs,info = eigsolve(x->ms.M̂*x,x₀,howmany,which;maxiter,tol,krylovdim=50) #,verbosity=2)
	# info.converged < howmany && @warn "KrylovKit.eigsolve only found $(info.converged) eigenvector/value pairs while attempting to find $howmany"
	# println("evals: $evals")
	# n_results = min(nev,info.converged) # min(size(ms.H⃗,2),info.converged)
	evals_res = copy(evals[1:nev])
	# evecs_res = canonicalize_phase(vec.(evecs[1:n_results]),ms)
    #evecs_res = evecs#copy.(vec.(evecs[1:n_results]))
	evecs_res = [copy(evecs[i]) for i in 1:nev]
	# copyto!(ms.ω²,evals_res)
	# copyto!(ms.H⃗[:,1:n_results],hcat(evecs_res...))
	
	# copy!(ms.H⃗,hcat(evecs_res...))
    # copy!(ms.ω²,evals[1:nev])
    
	# canonicalize_phase!(ms)
    # copyto!(evecs_res,eachcol(ms.H⃗))
	# return real(evals_res), evecs_res #collect(eachcol(ms.H⃗))
    # return copy(real(ms.ω²)), #copy.(eachcol(ms.H⃗))
	return real(evals_res), evecs_res
end


# @with_kw mutable struct KrylovKitEigsolve{T, Tl} <: AbstractEigsolve
# 	"Krylov Dimension"
# 	dim::Int64 = KrylovDefaults.krylovdim

# 	"Absolute tolerance for solver"
# 	atol::T  = KrylovDefaults.tol

# 	"Relative tolerance for solver"
# 	rtol::T  = KrylovDefaults.tol

# 	"Maximum number of iterations"
# 	maxiter::Int64 = KrylovDefaults.maxiter

# 	"Verbosity ∈ {0,1,2}"
# 	verbose::Int64 = 0

# 	"If the linear map is symmetric, only meaningful if T<:Real"
# 	issymmetric::Bool = false

# 	"If the linear map is hermitian"
# 	ishermitian::Bool = false

# 	"If the linear map is positive definite"
# 	isposdef::Bool = false

# 	"Left preconditioner"
# 	Pl::Tl = nothing
# end

# @with_kw mutable struct GMRESKrylovKit{T, Tl} <: AbstractIterativeLinearSolver
# 	"Krylov Dimension"
# 	dim::Int64 = KrylovDefaults.krylovdim

# 	"Absolute tolerance for solver"
# 	atol::T  = KrylovDefaults.tol

# 	"Relative tolerance for solver"
# 	rtol::T  = KrylovDefaults.tol

# 	"Maximum number of iterations"
# 	maxiter::Int64 = KrylovDefaults.maxiter

# 	"Verbosity ∈ {0,1,2}"
# 	verbose::Int64 = 0

# 	"If the linear map is symmetric, only meaningful if T<:Real"
# 	issymmetric::Bool = false

# 	"If the linear map is hermitian"
# 	ishermitian::Bool = false

# 	"If the linear map is positive definite"
# 	isposdef::Bool = false

# 	"Left preconditioner"
# 	Pl::Tl = nothing
# end

# # this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# # the optional shift is only used for the Hopf Newton / Continuation
# function (l::GMRESKrylovKit{T, Tl})(J, rhs; a₀ = 0, a₁ = 1, kwargs...) where {T, Tl}
# 	if Tl == Nothing
# 		res, info = KrylovKit.linsolve(J, rhs, a₀, a₁; rtol = l.rtol, verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, atol = l.atol, issymmetric = l.issymmetric, ishermitian = l.ishermitian, isposdef = l.isposdef, kwargs...)
# 	else # use preconditioner
# 		res, info = KrylovKit.linsolve(x -> (out = apply(J, x); ldiv!(l.Pl, out)), ldiv!(l.Pl, copy(rhs)), a₀, a₁; rtol = l.rtol, verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, atol = l.atol, issymmetric = l.issymmetric, ishermitian = l.ishermitian, isposdef = l.isposdef, kwargs...)
# 	end
# 	info.converged == 0 && (@warn "KrylovKit.linsolve solver did not converge")
# 	return res, true, info.numops
# end