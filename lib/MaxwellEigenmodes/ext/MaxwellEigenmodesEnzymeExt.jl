# Enzyme.jl interface for MaxwellEigenmodes.
#
# The eigenmode solves (`solve_k`) are iterative fixed-point computations whose gradients
# are defined by the adjoint-method ChainRules `rrule`s in `grads/solve.jl`. Here those
# rules are imported into Enzyme via `Enzyme._import_rrule` (the engine behind
# `Enzyme.@import_rrule`), so `solve_k` can sit inside a larger program differentiated
# with Enzyme.
#
# NB: the imported custom rules apply to positional calls `solve_k(ω, ε⁻¹, grid, solver)`;
# keyword-argument calls lower to `Core.kwcall`, which `@import_rrule` does not cover.
#
# `Enzyme._import_rrule` is provided by Enzyme's own ChainRulesCore extension. Extension
# load order is not guaranteed, so if that extension is not loaded yet when this one
# initializes, rule installation is deferred via `Base.package_callbacks`.

module MaxwellEigenmodesEnzymeExt

using MaxwellEigenmodes
using MaxwellEigenmodes: solve_k, KrylovKitEigsolve, IterativeSolversLOBPCG, DFTK_LOBPCG, MPBSolver
using DielectricSmoothing: Grid
using Logging: NullLogger
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules

# Solver bookkeeping that must not be differentiated through.
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.replan_ffts!), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaxwellEigenmodes.k_guess), args...; kwargs...) = nothing

const _rules_installed = Ref(false)

function _install_rules()
    _rules_installed[] && return nothing
    _rules_installed[] = true
    # Bridge the adjoint-method rrule for solve_k (2D & 3D grids, all bundled eigensolvers).
    # The macrocall is evaluated (and thus expanded) at runtime so that hygiene is handled
    # by the normal macro-expansion machinery.
    for TSolver in (:(KrylovKitEigsolve{NullLogger}), :(IterativeSolversLOBPCG{NullLogger}), :(DFTK_LOBPCG{NullLogger}), :(MPBSolver{NullLogger}))
        for (TGrid, TEps) in ((:(Grid{2,Float64}), :(Array{Float64,4})), (:(Grid{3,Float64}), :(Array{Float64,5})))
            Core.eval(@__MODULE__,
                :(Enzyme.@import_rrule(typeof(solve_k), Float64, $TEps, $TGrid, $TSolver)))
        end
    end
    return nothing
end

function __init__()
    if Base.get_extension(Enzyme, :EnzymeChainRulesCoreExt) !== nothing
        _install_rules()
    else
        # Defer until Enzyme's ChainRulesCore extension is loaded.
        push!(Base.package_callbacks, function (pkgid)
            if pkgid.name == "EnzymeChainRulesCoreExt" && !_rules_installed[]
                _install_rules()
            end
            return nothing
        end)
    end
end

end # module
