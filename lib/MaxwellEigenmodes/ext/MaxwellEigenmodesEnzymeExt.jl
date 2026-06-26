# Enzyme.jl interface for MaxwellEigenmodes.
#
# The eigenmode solves (`solve_k`, `solve_k_periodic`) are iterative fixed-point
# computations whose gradients are defined by the adjoint-method ChainRules `rrule`s
# (`grads/solve.jl`, `grads/period.jl`) and the matching forward `frule`s
# (`grads/forward.jl`). The pointwise tensor-field inverse `sliceinv_3x3` has closed-form
# rules in `grads/linalg.jl`. Here those rules are imported into Enzyme via
# `Enzyme.@import_rrule`/`@import_frule`, so the functions can sit inside a larger program
# differentiated with Enzyme in either mode.
#
# The macro calls are made at module top level: this Enzyme extension only loads once
# Enzyme is present, and ChainRulesCore is a hard dependency, so Enzyme's own
# `EnzymeChainRulesCoreExt` (which provides `@import_rrule`/`@import_frule`) is already
# loaded by the time this module is compiled. (The previous approach evaluated the macros
# from `__init__`, which breaks incremental compilation when the extension is later
# loaded from its precompiled cache: "Evaluation into the closed module … breaks
# incremental compilation".)
#
# NB: the imported custom rules apply to positional calls (e.g.
# `solve_k(ω, ε⁻¹, grid, solver)`); keyword-argument calls lower to `Core.kwcall`, which
# `@import_rrule` does not cover.

module MaxwellEigenmodesEnzymeExt

using MaxwellEigenmodes
using MaxwellEigenmodes: solve_k, solve_k_periodic, sliceinv_3x3, KrylovKitEigsolve, IterativeSolversLOBPCG, DFTK_LOBPCG, MPBSolver
using DielectricSmoothing: Grid
using Logging: NullLogger
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules

# Solver bookkeeping that must not be differentiated through.
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.replan_ffts!), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.k_guess), args...; kwargs...) = nothing

# Bridge the adjoint-method rrule + forward frule for solve_k (2D & 3D grids, all bundled
# eigensolvers) and the period-Λ eigensolve, plus the closed-form sliceinv_3x3 rules.
# `@import_rrule`/`@import_frule` are macros, so the concrete types must be spliced into
# each macrocall with `@eval` (the loop variables are not visible to a once-expanded
# macro). This runs at top level during precompilation, so the generated EnzymeRules
# methods are cached — unlike evaluating from `__init__`, which is rerun on every cached
# load and would error with "Evaluation into the closed module … breaks incremental
# compilation".
# `Enzyme.@import_rrule`/`@import_frule` are provided by Enzyme's ChainRulesCore extension,
# which is not guaranteed to be loaded while *this* extension precompiles (newer Enzyme
# versions moved `_import_rrule` there). Ensure it is, and guard the imports so a
# missing-extension corner case degrades gracefully (ForwardDiff/Zygote still differentiate
# these functions) instead of failing the whole extension's precompilation.
Base.retry_load_extensions()
try
    for TSolver in (:(KrylovKitEigsolve{NullLogger}), :(IterativeSolversLOBPCG{NullLogger}), :(DFTK_LOBPCG{NullLogger}), :(MPBSolver{NullLogger}))
        for (TGrid, TEps) in ((:(Grid{2,Float64}), :(Array{Float64,4})), (:(Grid{3,Float64}), :(Array{Float64,5})))
            @eval Enzyme.@import_rrule(typeof(solve_k), Float64, $TEps, $TGrid, $TSolver)
            @eval Enzyme.@import_frule(typeof(solve_k), Float64, $TEps, $TGrid, $TSolver)
        end
        # period-Λ eigensolve (3D periodic waveguides): reverse + forward rules
        @eval Enzyme.@import_rrule(typeof(solve_k_periodic), Float64, Array{Float64,5}, Float64, Grid{3,Float64}, $TSolver)
        @eval Enzyme.@import_frule(typeof(solve_k_periodic), Float64, Array{Float64,5}, Float64, Grid{3,Float64}, $TSolver)
    end
    # pointwise ε ⇄ ε⁻¹ tensor-field inverse (closed-form per-pixel rules; bypasses the
    # `Threads.@threads` kernel that native Enzyme reverse mode cannot trace).
    for TArr in (:(Array{Float64,4}), :(Array{Float64,5}))
        @eval Enzyme.@import_rrule(typeof(sliceinv_3x3), $TArr)
        @eval Enzyme.@import_frule(typeof(sliceinv_3x3), $TArr)
    end
catch err
    @warn "MaxwellEigenmodes: Enzyme rule import skipped (Enzyme/ChainRulesCore compat); \
           ForwardDiff/Zygote still provide forward/reverse AD" exception = err
end

end # module
