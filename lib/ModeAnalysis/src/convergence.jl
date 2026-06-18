# Forced grid-convergence mode solving.
#
# A waveguide eigenmode is computed on a *finite* finite-difference cell, so its
# effective indices carry two discretization errors:
#
#   1. truncation error — the periodic computational cell is finite, so the evanescent
#      cladding fields are clipped by the (periodic) boundaries. Controlled by the
#      distance from the waveguide center to the cell boundary (`Δx/2`, `Δy/2` for an
#      origin-centered `Grid`);
#   2. discretization error — the dielectric and fields are sampled on a finite grid.
#      Controlled by the spatial point density (points per μm²).
#
# `solve_k_converged` drives both errors down automatically: it re-runs the whole
# geometry → sub-pixel-smoothing → eigensolve pipeline on a sequence of grids, on each
# iteration multiplying the point density by `resolution_ramp` and the center-to-boundary
# distance by `boundary_ramp`, until the mode effective indices stop changing (to within
# `atol`/`rtol`) from one iteration to the next, or `max_iterations` is reached.

export ForceConvergenceSettings, ForceConvergenceResult, solve_k_converged

"""
    ForceConvergenceSettings(; rtol=1e-4, atol=1e-5, resolution_ramp=1.5,
                             boundary_ramp=1.25, max_iterations=8)

Settings for the forced grid-convergence loop of [`solve_k_converged`](@ref).

Fields:

- `rtol`: relative effective-index convergence tolerance. A band is converged once its
  effective index changes by less than `rtol · |neff_prev|` between successive iterations.
- `atol`: absolute effective-index convergence tolerance. A band is also converged once
  its effective index changes by less than `atol` between successive iterations. (A band
  satisfying *either* test is converged; all requested bands must converge.)
- `resolution_ramp`: factor (`> 1`) by which the spatial **point density** (points per
  μm²) is multiplied each iteration.
- `boundary_ramp`: factor (`> 1`) by which the **boundary distance** — the distance from
  the waveguide center to the grid boundary, i.e. `Δx/2` and `Δy/2` — is multiplied each
  iteration. The cell grows by this factor in each transverse dimension.
- `max_iterations`: maximum number of mode-simulation runs (including the initial run on
  the supplied grid). Convergence requires at least two runs.

Because each iteration multiplies the density by `resolution_ramp` and each transverse
extent by `boundary_ramp`, after `i` ramps the linear point density per axis grows by
`resolution_ramp^(i/2)` and each `N` by `resolution_ramp^(i/2) · boundary_ramp^i`; the
final grid size therefore encodes how many iterations were taken (see
[`ForceConvergenceResult`](@ref)).
"""
struct ForceConvergenceSettings
    rtol::Float64
    atol::Float64
    resolution_ramp::Float64
    boundary_ramp::Float64
    max_iterations::Int
end

function ForceConvergenceSettings(; rtol::Real=1e-4, atol::Real=1e-5,
        resolution_ramp::Real=1.5, boundary_ramp::Real=1.25, max_iterations::Integer=8)
    rtol >= 0 || throw(ArgumentError("rtol must be ≥ 0 (got $rtol)"))
    atol >= 0 || throw(ArgumentError("atol must be ≥ 0 (got $atol)"))
    resolution_ramp > 1 ||
        throw(ArgumentError("resolution_ramp must be > 1 to increase resolution (got $resolution_ramp)"))
    boundary_ramp >= 1 ||
        throw(ArgumentError("boundary_ramp must be ≥ 1 to grow the cell (got $boundary_ramp)"))
    max_iterations >= 2 ||
        throw(ArgumentError("max_iterations must be ≥ 2 (need ≥2 runs to compare; got $max_iterations)"))
    return ForceConvergenceSettings(Float64(rtol), Float64(atol), Float64(resolution_ramp),
        Float64(boundary_ramp), Int(max_iterations))
end

"""
    ForceConvergenceResult

Result of a [`solve_k_converged`](@ref) run. Fields:

- `kmags`, `evecs`: propagation constants `|k|` and (phase-canonicalized) eigenvectors of
  the final, most-refined solve — as returned by `solve_k`.
- `neff`: effective indices of the final solve (`kmags ./ ω`).
- `ε⁻¹`, `∂ε_∂ω`, `∂²ε_∂ω²`: the smoothed dielectric tensor fields on the final grid
  (`(3,3,size(grid)...)`), ready for [`group_index`](@ref) / [`ng_gvd`](@ref).
- `grid`: the final (most refined) `Grid`.
- `converged`: whether the effective indices met the `atol`/`rtol` tolerance before
  `max_iterations` was reached.
- `iterations`: number of mode-simulation runs performed (≥ 1).
- `neff_history`: effective indices from every iteration (`neff_history[end] == neff`).
- `grid_history`: the `Grid` used at every iteration (`grid_history[end] == grid`).

The number of iterations and whether convergence was achieved are recoverable from the
output grid alone (its size relative to the input grid reflects the ramping schedule), but
are also reported explicitly here for convenience.
"""
struct ForceConvergenceResult{ND,T}
    kmags::Vector{T}
    evecs::Vector{Vector{Complex{T}}}
    neff::Vector{T}
    ε⁻¹::Array{T}
    ∂ε_∂ω::Array{T}
    ∂²ε_∂ω²::Array{T}
    grid::Grid{ND,T}
    converged::Bool
    iterations::Int
    neff_history::Vector{Vector{T}}
    grid_history::Vector{Grid{ND,T}}
end

function Base.show(io::IO, r::ForceConvergenceResult)
    print(io, "ForceConvergenceResult(", r.converged ? "converged" : "NOT converged",
        " in ", r.iterations, " iteration", r.iterations == 1 ? "" : "s",
        ", final grid ", size(r.grid), ", neff ", round.(r.neff; digits=6), ")")
end

# round a count up to the nearest even integer ≥ 4 (even sizes are friendlier to the FFT
# basis of the eigensolver; 4 is a sane floor so a degenerate ramp can't collapse the grid)
_even_ge4(n::Real) = max(4, 2 * cld(round(Int, n), 2))

"point density of a 2D grid in points per μm²"
_point_density(g::Grid{2}) = length(g) / (g.Δx * g.Δy)

"""
    _ramped_grid(Δx, Δy, ρ) -> Grid{2}

Build an origin-centered 2D grid of transverse extents `Δx × Δy` (μm) with point density
as close as possible to `ρ` (points/μm²) while keeping the per-axis pitch isotropic
(`δx ≈ δy`): the linear density per axis is `√ρ`, so `Nx = √ρ·Δx`, `Ny = √ρ·Δy` (rounded
to even integers ≥ 4).
"""
function _ramped_grid(Δx::Real, Δy::Real, ρ::Real)
    lin = sqrt(ρ)                       # points per μm along each axis (isotropic pitch)
    return Grid(Float64(Δx), Float64(Δy), _even_ge4(lin * Δx), _even_ge4(lin * Δy))
end

"per-band absolute/relative convergence test between successive effective-index vectors"
function _neff_converged(neff, neff_prev, rtol::Real, atol::Real)
    length(neff) == length(neff_prev) || return false
    return all(zip(neff, neff_prev)) do (n, np)
        Δ = abs(n - np)
        return Δ < atol || Δ < rtol * abs(np)
    end
end

"""
    solve_k_converged(ω, shapes, mat_vals, minds, grid, solver; nev=1,
                      force_convergence=true,
                      force_convergence_settings=ForceConvergenceSettings(),
                      verbose=false, solver_kwargs...) -> ForceConvergenceResult

Solve for the first `nev` guided eigenmodes of a shape-based waveguide geometry at
frequency `ω`, optionally **forcing spatial-grid convergence** of the mode effective
indices.

The geometry is specified exactly as for [`smooth_ε`](@ref): `shapes` (a tuple of
`MaterialShape`s, foreground first), `mat_vals` (per-material dispersion columns,
e.g. from `MaterialDispersion._f_ε_mats`), and `minds` (shape/background → material-column
map). `grid` is the *initial* `Grid` and `solver` an `AbstractEigensolver`. Any extra
keyword arguments (`k_tol`, `eig_tol`, `max_eigsolves`, …) are forwarded to `solve_k`.

When `force_convergence` is `false`, the geometry is smoothed once onto `grid` and the
modes are solved once — a convenience wrapper over `smooth_ε`/`solve_k`.

When `force_convergence` is `true`, the full pipeline (sub-pixel dielectric smoothing →
eigensolve) is re-run on a sequence of progressively refined grids. Each iteration
multiplies the spatial **point density** (points/μm²) by
`force_convergence_settings.resolution_ramp` and the **boundary distance**
(waveguide-center → grid-boundary, i.e. `Δx/2`, `Δy/2`) by
`force_convergence_settings.boundary_ramp`, keeping the per-axis pixel pitch isotropic.
The loop stops once every band's effective index changes by less than `atol` (absolute)
or `rtol` (relative to the previous value) between successive iterations, or after
`max_iterations` runs.

Returns a [`ForceConvergenceResult`](@ref) carrying the final (most-refined) modes, the
smoothed dielectric fields on the final grid, and the convergence diagnostics
(`converged`, `iterations`, and the per-iteration `neff`/`grid` histories).

```julia
mats = [Si₃N₄, SiO₂]
f_ε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = f_ε([1/1.55])
core = MaterialShape(Cuboid([0.0, 0.0], [1.6, 0.7], [1.0 0.0; 0.0 1.0]), 1)
res = solve_k_converged(1/1.55, (core,), mat_vals, (1, 2), Grid(4.0, 3.0, 48, 36),
                        KrylovKitEigsolve();
                        force_convergence=true,
                        force_convergence_settings=ForceConvergenceSettings(rtol=1e-5))
res.converged, res.iterations, size(res.grid), res.neff
```
"""
function solve_k_converged(ω::Real, shapes, mat_vals, minds, grid::Grid{2,TG}, solver;
        nev::Integer=1, force_convergence::Bool=true,
        force_convergence_settings::ForceConvergenceSettings=ForceConvergenceSettings(),
        verbose::Bool=false, solver_kwargs...) where {TG<:Real}
    ω = Float64(ω)

    # one full pipeline run on a given grid: smooth the geometry, then solve the modes
    function run_grid(g::Grid{2})
        sm = smooth_ε(shapes, mat_vals, minds, g)
        ε⁻¹ = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
        ∂ε_∂ω = copy(selectdim(sm, 3, 2))
        ∂²ε_∂ω² = copy(selectdim(sm, 3, 3))
        kmags, evecs = solve_k(ω, ε⁻¹, g, solver; nev=Int(nev), solver_kwargs...)
        return kmags, evecs, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω²
    end

    # iteration 1: the supplied grid, as given
    g = Grid(Float64(grid.Δx), Float64(grid.Δy), grid.Nx, grid.Ny)
    kmags, evecs, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω² = run_grid(g)
    neff = kmags ./ ω
    neff_history = [neff]
    grid_history = Grid{2,Float64}[g]
    verbose && @info "solve_k_converged iteration 1" gridsize = size(g) neff

    if !force_convergence
        return ForceConvergenceResult(kmags, evecs, neff, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω², g,
            false, 1, neff_history, grid_history)
    end

    s = force_convergence_settings
    # track the *target* extents and density as floats so rounding never compounds
    Δx, Δy = Float64(g.Δx), Float64(g.Δy)
    ρ = _point_density(g)
    converged = false
    iter = 1
    while iter < s.max_iterations
        iter += 1
        Δx *= s.boundary_ramp
        Δy *= s.boundary_ramp
        ρ *= s.resolution_ramp
        g = _ramped_grid(Δx, Δy, ρ)
        neff_prev = neff
        kmags, evecs, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω² = run_grid(g)
        neff = kmags ./ ω
        push!(neff_history, neff)
        push!(grid_history, g)
        converged = _neff_converged(neff, neff_prev, s.rtol, s.atol)
        verbose && @info "solve_k_converged iteration $iter" gridsize = size(g) neff Δneff = neff .- neff_prev converged
        converged && break
    end

    return ForceConvergenceResult(kmags, evecs, neff, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω², g,
        converged, iter, neff_history, grid_history)
end
