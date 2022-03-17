####################################################################################################
# LOBPCG Eigensolver from DFTK.jl
####################################################################################################

using .DFTK

export DFTK_LOBPCG

# Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.
# """
# $(TYPEDEF)
# Create an eigensolver based on `LOBPCG` from `DFTK.jl`. 
# $(TYPEDFIELDS)
# !!! tip "Different linear solvers"
#     The source code for LOBPCG is has lots of useful comments, see [here](https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/eigen/lobpcg_hyper_impl.jl)
# """

# function signature with keywords and default values copied from source code
# LOBPCG(A, X, B=I, precon=I, tol=1e-10, maxiter=100; miniter=1, ortho_tol=2eps(real(eltype(X))), n_conv_check=nothing, display_progress=false)
                         
mutable struct DFTK_LOBPCG <: AbstractEigensolver end
# @with_kw mutable struct DFTK_LOBPCG{T} <: AbstractEigensolver 
    
#     λ::Vector{T}
    
#     X::Matrix{Complex{T}}
    
#     n_matvec::Int64
    
#     residual_history::Matrix{T}
    
#     residual_norms::Vector{T}
# end

function solve_ω²(ms::ModeSolver{ND,T},solver::DFTK_LOBPCG;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
	# x₀ 		=	vec(ms.H⃗[:,1])	 # initial eigenvector guess, should be complex vector with length 2*prod(size(grid))
	# howmany =	nev				# how many eigenvector/value pairs to find
	# which	=	:SR				# :SR="Smallest Real" eigenvalues
	# # evals,evecs,convinfo = eigsolve(x->ms.M̂*x,ms.H⃗[:,1],nev,:SR; maxiter, tol, krylovdim=50, verbosity=2)
	res = LOBPCG(ms.M̂,ms.H⃗,I,ms.P̂,tol,maxiter)
    copyto!(ms.H⃗,res.X)
    copyto!(ms.ω²,res.λ)
    canonicalize_phase!(ms)
    # evals,evecs,info = eigsolve(x->ms.M̂*x,x₀,howmany,which;maxiter,tol,krylovdim=50) #,verbosity=2)
	# info.converged < howmany && @warn "KrylovKit.eigsolve only found $(info.converged) eigenvector/value pairs while attempting to find $howmany"
	# println("evals: $evals")
	# n_results = min(nev,info.converged) # min(size(ms.H⃗,2),info.converged)
	# evals_res = evals[1:n_results]
	# evecs_res = vec.(evecs[1:n_results])
	# copyto!(ms.ω²,evals_res)
	# copyto!(ms.H⃗[:,1:n_results],hcat(evecs_res...))
	# return real(evals_res), evecs_res
    return real(ms.ω²), collect(eachcol(ms.H⃗))
end
