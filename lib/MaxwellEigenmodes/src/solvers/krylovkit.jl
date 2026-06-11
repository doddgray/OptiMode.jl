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
# mutable struct KrylovKitEigsolve <: AbstractEigensolver end

mutable struct KrylovKitEigsolve{L<:AbstractLogger} <: AbstractEigensolver{L}
    logger::L
end

KrylovKitEigsolve() = KrylovKitEigsolve(NullLogger())

function solve_ω²(ms::ModeSolver{ND,T},solver::KrylovKitEigsolve;nev=1,eigind=1,maxiter=200,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
	evals,evecs,convinfo = eigsolve(x->ms.M̂*x,rand(Complex{T},size(ms.H⃗[:,1])),nev,:SR; maxiter, tol, krylovdim=50) # , verbosity=2)

	evals_res = copy(evals[1:nev])
	evecs_res = [copy(evecs[i]) for i in 1:nev]
	return real(evals_res), evecs_res
end
