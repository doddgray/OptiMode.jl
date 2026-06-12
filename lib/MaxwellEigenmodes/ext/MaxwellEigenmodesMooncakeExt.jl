# Mooncake.jl interface for MaxwellEigenmodes.
#
# The eigenmode solves (`solve_k`) are iterative fixed-point computations whose gradients
# are defined by the adjoint-method ChainRules `rrule`s in `grads/solve.jl`. Here those
# rules are bridged to Mooncake with `Mooncake.@from_rrule`, so `solve_k` can sit inside
# a larger program differentiated in reverse mode with Mooncake.

module MaxwellEigenmodesMooncakeExt

using MaxwellEigenmodes
using MaxwellEigenmodes: solve_k, AbstractEigensolver, KrylovKitEigsolve,
    IterativeSolversLOBPCG, DFTK_LOBPCG, MPBSolver
using DielectricSmoothing: Grid
using Logging: NullLogger
using Mooncake
using Mooncake: @from_rrule, MinimalCtx, @zero_adjoint

# Bridge the adjoint-method rrule for solve_k (2D and 3D grids, all bundled eigensolvers).
for TSolver in (KrylovKitEigsolve{NullLogger}, IterativeSolversLOBPCG{NullLogger}, DFTK_LOBPCG{NullLogger}, MPBSolver{NullLogger})
    for (TGrid, TEps) in ((Grid{2,Float64}, Array{Float64,4}), (Grid{3,Float64}, Array{Float64,5}))
        @eval @from_rrule(
            MinimalCtx,
            Tuple{typeof(solve_k),Float64,$TEps,$TGrid,$TSolver},
            true,
        )
    end
end

# Solver bookkeeping that must not be differentiated through.
@zero_adjoint MinimalCtx Tuple{typeof(MaxwellEigenmodes.replan_ffts!),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(MaxwellEigenmodes.k_guess),Vararg}

end # module
