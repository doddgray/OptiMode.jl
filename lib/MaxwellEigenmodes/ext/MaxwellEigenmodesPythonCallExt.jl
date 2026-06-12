# PythonCall.jl-based MPB (MIT Photonic Bands) backend for MaxwellEigenmodes.
#
# This replaces the legacy PyCall implementation (preserved in `legacy/mpb.jl`). The
# smoothed dielectric tensor data is passed to MPB as a *material function* (a Julia
# closure called from Python per mesh point), so no files or scipy interpolation are
# involved, and both MPB and the native solvers operate on identical dielectric data.
#
# Eigenvector data layout: MPB returns plane-wave coefficient arrays of shape
# (N, 2, nb) with the grid flattened in C (row-major) order; the helpers below reshuffle
# each band into the column-major layout used by `HelmholtzMap`, following the recipe in
# the legacy implementation.

module MaxwellEigenmodesPythonCallExt

using MaxwellEigenmodes
using MaxwellEigenmodes: MPBSolver, ModeSolver, sliceinv_3x3
import MaxwellEigenmodes: _mpb_solve_ω², _mpb_solve_k
using DielectricSmoothing
using DielectricSmoothing: Grid, N, δx, δy, δz
using LinearAlgebra
using PythonCall

const meep = PythonCall.pynew()
const mpb = PythonCall.pynew()
const _py_wrap_fn = PythonCall.pynew()
const _py_wrap_bandfunc = PythonCall.pynew()
const _mpb_loaded = Ref(false)

function __init__()
    try
        PythonCall.pycopy!(meep, pyimport("meep"))
        PythonCall.pycopy!(mpb, pyimport("meep.mpb"))
        # pympb assigns attributes (e.g. `do_averaging`) to the material function object;
        # Julia callables wrapped by PythonCall reject setattr, so wrap them in a Python
        # lambda, which accepts arbitrary attributes.
        PythonCall.pycopy!(_py_wrap_fn, pyeval("lambda f: lambda p: f(p)", @__MODULE__))
        # pympb inspects `band_func.__code__.co_argcount`, which wrapped Julia callables
        # lack; a two-argument Python lambda provides it.
        PythonCall.pycopy!(_py_wrap_bandfunc, pyeval("lambda f: lambda ms, band: f(ms, band)", @__MODULE__))
        _mpb_loaded[] = true
    catch err
        @warn """
        MaxwellEigenmodes MPB backend: could not import the Python modules `meep`/`meep.mpb`.
        Install them in PythonCall's Python environment, e.g.
            using CondaPkg; CondaPkg.add("pymeep")
        The `MPBSolver` backend will error until they are importable.
        """ exception = err
    end
end

mpb_available() = _mpb_loaded[]

_check_loaded() = _mpb_loaded[] || error(MaxwellEigenmodes._MPB_HELP)

_parity(solver::MPBSolver) = pygetattr(meep, String(solver.parity))

"min/max refractive index of a (3,3,Nx,Ny[,Nz]) dielectric tensor array, from its diagonal"
function _n_range(ε::AbstractArray)
    lo, hi = Inf, -Inf
    @inbounds for I in CartesianIndices(size(ε)[3:end]), a in 1:3
        v = ε[a, a, I]
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return sqrt(lo), sqrt(hi)
end

# PythonCall may pass the `meep.Vector3` mesh point through as a wrapped Py object or
# auto-convert it to a Julia vector (Vector3 is array-like); accept both.
_coord(p::AbstractVector, i::Int) = Float64(p[i])
_coord(p, i::Int) = pyconvert(Float64, pygetattr(p, ("x", "y", "z")[i]))

"""
Material function: a Julia closure called by MPB (through Python) with a `meep.Vector3`
mesh point, returning the `meep.Medium` for the nearest pixel/voxel of the smoothed
dielectric tensor array `ε`. With the default `mesh_size=1` the query points coincide
with the pixel/voxel centers, so the lookup is exact.
"""
function _material_fn(ε::AbstractArray, grid::Grid{2})
    Nx, Ny = size(grid)
    function matfn(p)
        ix = clamp(round(Int, (_coord(p, 1) + grid.Δx / 2) / δx(grid)) + 1, 1, Nx)
        iy = clamp(round(Int, (_coord(p, 2) + grid.Δy / 2) / δy(grid)) + 1, 1, Ny)
        @inbounds return meep.Medium(
            epsilon_diag=meep.Vector3(ε[1, 1, ix, iy], ε[2, 2, ix, iy], ε[3, 3, ix, iy]),
            epsilon_offdiag=meep.Vector3(ε[1, 2, ix, iy], ε[1, 3, ix, iy], ε[2, 3, ix, iy]),
        )
    end
    return matfn
end

function _material_fn(ε::AbstractArray, grid::Grid{3})
    Nx, Ny, Nz = size(grid)
    function matfn(p)
        ix = clamp(round(Int, (_coord(p, 1) + grid.Δx / 2) / δx(grid)) + 1, 1, Nx)
        iy = clamp(round(Int, (_coord(p, 2) + grid.Δy / 2) / δy(grid)) + 1, 1, Ny)
        iz = clamp(round(Int, (_coord(p, 3) + grid.Δz / 2) / δz(grid)) + 1, 1, Nz)
        @inbounds return meep.Medium(
            epsilon_diag=meep.Vector3(ε[1, 1, ix, iy, iz], ε[2, 2, ix, iy, iz], ε[3, 3, ix, iy, iz]),
            epsilon_offdiag=meep.Vector3(ε[1, 2, ix, iy, iz], ε[1, 3, ix, iy, iz], ε[2, 3, ix, iy, iz]),
        )
    end
    return matfn
end

_lattice(grid::Grid{2}) = meep.Lattice(size=meep.Vector3(grid.Δx, grid.Δy, 1.0))
_lattice(grid::Grid{3}) = meep.Lattice(size=meep.Vector3(grid.Δx, grid.Δy, grid.Δz))
_resolution(grid::Grid{2}) = meep.Vector3(grid.Nx / grid.Δx, grid.Ny / grid.Δy, 1)
_resolution(grid::Grid{3}) = meep.Vector3(grid.Nx / grid.Δx, grid.Ny / grid.Δy, grid.Nz / grid.Δz)

"construct a `pympb.ModeSolver` fed with pre-smoothed dielectric data as its default material"
function _ms_mpb(ε::AbstractArray, grid::Grid, solver::MPBSolver, num_bands::Int, eig_tol::Real)
    return mpb.ModeSolver(;
        resolution=_resolution(grid),
        geometry_lattice=_lattice(grid),
        num_bands=num_bands,
        tolerance=eig_tol,
        mesh_size=solver.mesh_size,
        default_material=_py_wrap_fn(_material_fn(ε, grid)),
        deterministic=true,
        verbose=solver.verbose,
    )
end

"""
Reshuffle one MPB eigenvector — plane-wave coefficients of shape (N, 2) with the grid
flattened in C (row-major) order — into the flat column-major (2, Nx, Ny[, Nz]) layout
used by `HelmholtzMap`, and normalize it.
"""
function _to_julia_evec(ev::AbstractMatrix{<:Complex}, grid::Grid{ND}) where {ND}
    ev2 = permutedims(ev)                                   # (2, N), N in MPB's C order
    ev3 = reshape(ev2, (2, reverse(size(grid))...))         # last grid axis fastest
    perm = (1, (reverse(1:ND) .+ 1)...)
    return normalize(vec(permutedims(ev3, perm)))
end

"band function recording each solved eigenvector into `evecs` (1-based band offset `b0`)"
function _evec_recorder(evecs::Vector{Vector{ComplexF64}}, grid::Grid, b0::Int)
    function record_evec(ms_py, band)
        b = pyconvert(Int, band)
        ev = pyconvert(Array{ComplexF64,3}, ms_py.get_eigenvectors(b, 1))  # (N, 2, 1)
        evecs[b-b0+1] = _to_julia_evec(view(ev, :, :, 1), grid)
        return nothing
    end
    return _py_wrap_bandfunc(record_evec)
end

"run `body()` with MPB's (C-level) stdout suppressed unless `verbose`"
function _maybe_quiet(body, verbose::Bool)
    verbose && return body()
    return redirect_stdout(body, devnull)
end

function _mpb_solve_k(ω::T, ε⁻¹::AbstractArray{T}, grid::Grid{ND,T}, solver::MPBSolver;
    nev::Int=1, k_tol::Real=1e-8, eig_tol::Real=1e-8) where {ND,T<:Real}
    _check_loaded()
    ε = sliceinv_3x3(ε⁻¹)
    n_min, n_max = _n_range(ε)
    evecs = Vector{Vector{ComplexF64}}(undef, nev)
    kmags = _maybe_quiet(solver.verbose) do
        ms = _ms_mpb(ε, grid, solver, nev, eig_tol)
        kmags_py = ms.find_k(
            _parity(solver),                    # mode-symmetry constraint
            ω,                                  # frequency at which to solve for |k|
            1,                                  # band_min
            nev,                                # band_max
            meep.Vector3(0, 0, 1),              # k⃗ search direction
            k_tol,                              # |k| tolerance
            solver.n_guess_factor * n_max * ω,  # |k| guess
            n_min * ω,                          # |k| search lower bound
            n_max * ω,                          # |k| search upper bound
            mpb.fix_efield_phase,               # canonicalize eigenvector phase
            _evec_recorder(evecs, grid, 1),     # capture eigenvectors band-by-band
        )
        return pyconvert(Vector{T}, kmags_py)
    end
    return kmags, evecs
end

function _mpb_solve_ω²(ms::ModeSolver{ND,T}, solver::MPBSolver; nev::Int=1, tol::Real=1e-8) where {ND,T<:Real}
    _check_loaded()
    ε = sliceinv_3x3(ms.M̂.ε⁻¹)
    grid = ms.grid
    k⃗ = ms.M̂.k⃗
    evecs = Vector{Vector{ComplexF64}}(undef, nev)
    freqs = _maybe_quiet(solver.verbose) do
        msp = _ms_mpb(ε, grid, solver, nev, tol)
        msp.k_points = pylist([meep.Vector3(k⃗[1], k⃗[2], k⃗[3])])
        msp.run_parity(_parity(solver), true, mpb.fix_efield_phase, _evec_recorder(evecs, grid, 1))
        return pyconvert(Matrix{T}, msp.all_freqs)   # (n_kpoints=1, nev)
    end
    ω² = vec(freqs) .^ 2
    copyto!(ms.H⃗, hcat(evecs...))
    copyto!(ms.ω², complex.(ω²))
    return ω², evecs
end

end # module
