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
# mutable struct IterativeSolversLOBPCG <: AbstractEigensolver end

mutable struct IterativeSolversLOBPCG{L<:AbstractLogger} <: AbstractEigensolver{L}
    logger::L
end

IterativeSolversLOBPCG() = IterativeSolversLOBPCG(NullLogger())

function solve_ω²(ms::ModeSolver{ND,T},solver::IterativeSolversLOBPCG;nev=1,eigind=1,maxiter=200,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
    eigs_itr = LOBPCGIterator(ms.M̂,false,copy(ms.H⃗), ms.P̂) # ,constraint)
	res = lobpcg!(eigs_itr;log,not_zeros=false,maxiter,tol)
    copy!(ms.H⃗,res.X)
    copy!(ms.ω²,res.λ)
    real(copy(res.λ)), [copy(res.X[:,i]) for i=1:nev] #[copy(ev) for ev in eachcol(res.X)] #copy(res.X) #
end
