####################################################################################################
# Solvers from IterativeSolvers.jl
####################################################################################################

# using .IterativeSolvers   # uncomment if using Requires.jl for optional dependancy

export IterativeSolversLOBPCG

# Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.
# """
# $(TYPEDEF)
# Create an eigensolver based on `lobpcg` from `IterativeSolvers.jl`. 
# $(TYPEDFIELDS)
# !!! tip "Different linear solvers"
#     By tuning the options, you can select CG, GMRES... see [here](https://jutho.github.io/KrylovKit.jl/stable/man/linear/#KrylovKit.linsolve)
# """
mutable struct IterativeSolversLOBPCG <: AbstractEigensolver end

function solve_ω²(ms::ModeSolver{ND,T},solver::IterativeSolversLOBPCG;nev=1,eigind=1,maxiter=200,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
    eigs_itr = LOBPCGIterator(ms.M̂,false,copy(ms.H⃗), ms.P̂) # ,constraint)
	res = lobpcg!(eigs_itr;log,not_zeros=false,maxiter,tol)
    # res = lobpcg(ms.M̂,false,copy(ms.H⃗);log,maxiter,tol,P=ms.P̂)
    # copyto!(ms.H⃗,res.X)
    # copyto!(ms.ω²,res.λ)
    copy!(ms.H⃗,res.X)
    copy!(ms.ω²,res.λ)
    # canonicalize_phase!(ms)
    # copyto!(res.X,ms.H⃗)

    # res = lobpcg(ms.M̂,ms.H⃗[:,1]; log,not_zeros=false,maxiter,tol)
    # evals,evecs,info = eigsolve(x->ms.M̂*x,x₀,howmany,which;maxiter,tol,krylovdim=50) #,verbosity=2)
	# info.converged < howmany && @warn "KrylovKit.eigsolve only found $(info.converged) eigenvector/value pairs while attempting to find $howmany"
	# println("evals: $evals")
	# return copy(real(ms.ω²)), copy.(eachcol(ms.H⃗))
    real(copy(res.λ)), [copy(res.X[:,i]) for i=1:nev] #[copy(ev) for ev in eachcol(res.X)] #copy(res.X) #
end


	# λ⃗ = randn(Complex{T},2*M̂.N)
	# b⃗ = similar(λ⃗)
	# adj_itr = bicgstabl_iterator!(λ⃗, M̂ - ( 1. * I ), b⃗, 2;		# last entry is `l`::Int = # of GMRES iterations
    #                          Pl = Identity(),
    #                          max_mv_products = size(M̂, 2),
    #                          abstol = zero(T),
    #                          reltol = sqrt(eps(T)),
    #                          initial_zero = false)