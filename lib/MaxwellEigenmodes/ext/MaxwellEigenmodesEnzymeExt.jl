# Enzyme.jl interface for MaxwellEigenmodes.
#
# The eigenmode solves (`solve_k`) are iterative fixed-point computations whose gradients
# are defined by the adjoint-method ChainRules `rrule`s in `grads/solve.jl`. Here those
# rules are imported into Enzyme with `Enzyme.@import_rrule`, so `solve_k` can sit inside
# a larger program differentiated with Enzyme.

module MaxwellEigenmodesEnzymeExt

using MaxwellEigenmodes
using MaxwellEigenmodes: solve_k, KrylovKitEigsolve, IterativeSolversLOBPCG, DFTK_LOBPCG
using DielectricSmoothing: Grid
using Logging: NullLogger
using Enzyme
using Enzyme: EnzymeRules

# Solver bookkeeping that must not be differentiated through.
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.replan_ffts!), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.k_guess), args...; kwargs...) = nothing

# Bridge the adjoint-method rrule for solve_k (2D and 3D grids, all bundled eigensolvers).
for TSolver in (KrylovKitEigsolve{NullLogger}, IterativeSolversLOBPCG{NullLogger}, DFTK_LOBPCG{NullLogger})
    for (TGrid, TEps) in ((Grid{2,Float64}, Array{Float64,4}), (Grid{3,Float64}, Array{Float64,5}))
        @eval Enzyme.@import_rrule(typeof(solve_k), Float64, $TEps, $TGrid, $TSolver)
    end
end

end # module
