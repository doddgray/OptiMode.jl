####################################################################################################
# MPB (MIT Photonic Bands) eigensolver backend
#
# The solver methods are implemented in the `MaxwellEigenmodesPythonCallExt` package
# extension (see `ext/`), which loads when PythonCall.jl is available. This file defines
# the solver type and the function hooks the extension attaches to, so that `MPBSolver`
# can be constructed (and AD rules can be registered for it) without Python present.
####################################################################################################

export MPBSolver

"""
    MPBSolver(; verbose=false, parity=:NO_PARITY, mesh_size=1, n_guess_factor=0.9)

Eigensolver backend driven by MIT Photonic Bands (MPB) through the Python `meep.mpb`
module via PythonCall.jl. Usable with `solve_ω²` and `solve_k` like the native
eigensolvers; the smoothed dielectric tensor data is passed to MPB as a material
function, so both solvers operate on identical discretizations.

Requirements:
- load `PythonCall` (this activates the `MaxwellEigenmodesPythonCallExt` extension)
- the Python packages `meep` and `meep.mpb` must be importable in PythonCall's Python
  environment, e.g. via `using CondaPkg; CondaPkg.add("pymeep")` (conda-forge).

Options:
- `verbose`: when `false` (default), MPB's stdout chatter is suppressed.
- `parity`: a `Symbol` naming a meep parity constant (`:NO_PARITY`, `:EVEN_Z`, `:ODD_Z`,
  `:EVEN_Y`, `:ODD_Y`, …) used to constrain mode symmetry.
- `mesh_size`: MPB sub-pixel averaging mesh; defaults to 1 (no further averaging) since
  the input dielectric data is already smoothed by DielectricSmoothing.
- `n_guess_factor`: `|k|` search guess for `solve_k` is `n_guess_factor * n_max * ω`.

The adjoint-method ChainRules `rrule` for `solve_k` is solver-generic, so gradients of
`solve_k(ω, ε⁻¹, grid, ::MPBSolver)` work with Zygote (and with Mooncake/Enzyme through
the bridged rules) just as for the native backends.
"""
mutable struct MPBSolver{L<:AbstractLogger} <: AbstractEigensolver{L}
    logger::L
    verbose::Bool
    parity::Symbol
    mesh_size::Int
    n_guess_factor::Float64
end

MPBSolver(; logger=NullLogger(), verbose=false, parity=:NO_PARITY, mesh_size=1, n_guess_factor=0.9) =
    MPBSolver(logger, verbose, parity, mesh_size, n_guess_factor)

# Hooks extended by the PythonCall package extension.
function _mpb_solve_ω² end
function _mpb_solve_k end

const _MPB_HELP = """
The MPB backend requires PythonCall and the Python `meep`/`meep.mpb` modules.
Load it with:
    using PythonCall              # activates the MaxwellEigenmodesPythonCallExt extension
and make `meep` importable in PythonCall's Python environment, e.g.:
    using CondaPkg; CondaPkg.add("pymeep")
"""

_mpb_solve_ω²(args...; kwargs...) = error(_MPB_HELP)
_mpb_solve_k(args...; kwargs...) = error(_MPB_HELP)

function solve_ω²(ms::ModeSolver{ND,T}, solver::MPBSolver; nev=1, maxiter=100, tol=1e-8,
    log=false, f_filter=nothing) where {ND,T<:Real}
    return _mpb_solve_ω²(ms, solver; nev, tol)
end

function solve_k(ω::T, ε⁻¹::AbstractArray{T}, grid::Grid{ND,T}, solver::MPBSolver; nev=1,
    max_eigsolves=60, maxiter=100, k_tol=1e-8, eig_tol=1e-8, log=false, kguess=nothing,
    Hguess=nothing, f_filter=nothing, overwrite=false) where {ND,T<:Real}
    return _mpb_solve_k(ω, ε⁻¹, grid, solver; nev, k_tol, eig_tol)
end
