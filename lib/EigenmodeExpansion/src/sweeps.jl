# ──────────────────────────────────────────────────────────────────────────────
#  Parameter-sweep / SLURM integration helpers.
#
#  EME's cost is dominated by the per-cell mode solves, which map one-to-one onto
#  ModeSweeps tasks (one cross-section per task). These helpers express a cell as
#  a ModeSweeps `make_problem` payload and re-assemble a device S-matrix from the
#  (remotely) solved per-cell fields. The actual SLURM deployment/gather lives in
#  the `EigenmodeExpansionModeSweepsExt` extension (loaded with `ModeSweeps`); the
#  functions here have no ModeSweeps dependency so they also work fully in-process.
# ──────────────────────────────────────────────────────────────────────────────

export cell_problem, assemble_eme, deploy_eme, gather_eme

"""
    deploy_eme(setup_file, cells; ω, nev=2, dedup=true, kwargs...) -> BatchInfo
    deploy_eme(setup_file, num_cells::Int; ω, nev=2, kwargs...) -> BatchInfo

Deploy the per-cell mode solves of an EME stack as a ModeSweeps batch. The expensive
work — `n_cells × n_ω` cross-section eigensolves — is fully independent and farmed as one
SLURM array job: pass a vector `ω` to sweep frequency and cells together in a single batch
(one task per `(cell, ω)`), then reduce per ω at gather time. With the `cells` method and
`dedup=true` (default), only the *unique* cross-sections are enqueued (see
[`dedup_groups`](@ref)); [`gather_eme`](@ref) maps every cell back to its representative
task, so uniform/repeated stacks cost one solve per distinct geometry per frequency.

Requires `ModeSweeps` to be loaded (see the `EigenmodeExpansionModeSweepsExt` extension).
The `setup_file`'s `make_problem(p)` should return the cell-`p.cell` cross-section problem
at frequency `p.ω` (see [`cell_problem`](@ref)).
"""
function deploy_eme end

"""
    gather_eme(batch, cells, materials, ω, grid; dedup=true, kwargs...)
        -> EMEResult | Vector{EMEResult}

Gather a completed [`deploy_eme`](@ref) batch and re-assemble the device S-matrix. With a
scalar `ω` one [`EMEResult`](@ref) is returned; with a vector `ω` one result per frequency
(in order) is returned. Each cell's modes are taken from its dedup representative's task
(`dedup` must match the `deploy_eme` call). Requires `ModeSweeps` to be loaded.
"""
function gather_eme end

# ── (cell, ω) → task-index mapping (pure; operates on the batch's parameter list) ──

# Quantise ω to an integer key so a gather-time ω matches the deploy-time ω despite any
# float round-trip; ω is O(1) μm⁻¹ so 1e-12 absolute resolution is unambiguous.
_ωkey(ω::Real; atol::Float64=1e-12) = round(Int, ω / atol)

"build a `(cell, ω) → task index` lookup from a batch's `(; cell, ω, …)` parameter list"
function _eme_task_map(params::AbstractVector; atol::Float64=1e-12)
    d = Dict{Tuple{Int,Int},Int}()
    for (t, p) in enumerate(params)
        d[(Int(p.cell), _ωkey(p.ω; atol))] = t
    end
    return d
end

_eme_task_index(d::AbstractDict, cell::Int, ω::Real; atol::Float64=1e-12) =
    d[(cell, _ωkey(ω; atol))]

"""
    cell_problem(cell, materials, ω, grid) -> NamedTuple

Build the ModeSweeps `make_problem`-style payload `(; ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω², grid)`
for a single EME [`Cell`](@ref). A ModeSweeps setup script can solve one cell per
SLURM array task with

    make_problem(p) = cell_problem(CELLS[p.cell], MATERIALS, p.ω, GRID)

so an entire EME stack's mode solves are deployed as one batch (and swept over ω
or geometry alongside the cell index).
"""
function cell_problem(cell::Cell, materials, ω, grid)
    mat_vals = _mat_vals(materials, ω)
    sm = smooth_ε(Tuple(cell.cross_section.shapes), mat_vals, Tuple(cell.cross_section.minds), grid)
    return (;
        ε⁻¹=sliceinv_3x3(copy(selectdim(sm, 3, 1))),
        ∂ε_∂ω=copy(selectdim(sm, 3, 2)),
        ∂²ε_∂ω²=copy(selectdim(sm, 3, 3)),
        grid,
    )
end

"""
    assemble_eme(cells, kmags_per_cell, evecs_per_cell, materials, ω, grid;
                 conjugate=false, reg=1e-9, reciprocity=true) -> EMEResult

Re-assemble a device S-matrix from per-cell eigensolutions (`kmags`/`evecs`,
e.g. gathered from a ModeSweeps batch). Mode fields are reconstructed locally
(cheap; no eigensolve) and the interface/propagation S-matrices are cascaded as
in [`eme`](@ref).
"""
function assemble_eme(cells::AbstractVector{Cell}, kmags_per_cell, evecs_per_cell,
                      materials, ω, grid; conjugate::Bool=false, reg::Real=1e-9,
                      reciprocity::Bool=true, passivity::Symbol=:invert, dedup::Bool=true)
    n = length(cells)
    reps, gid = dedup ? dedup_groups(cells) : (collect(1:n), collect(1:n))
    # reconstruct each unique cross-section's modes once, then share across its group
    rep_modes = map(eachindex(reps)) do k
        ci = reps[k]
        ε⁻¹, ∂ε_∂ω = cell_dielectric(cells[ci].cross_section, materials, ω, grid)
        [build_mode(ω, kmags_per_cell[k][j], evecs_per_cell[k][j], ε⁻¹, ∂ε_∂ω, grid; conjugate)
         for j in eachindex(kmags_per_cell[k])]
    end
    modes = [rep_modes[g] for g in gid]
    S = _assemble(modes, [c.length for c in cells]; conjugate, reg, reciprocity, passivity)
    T = typeof(float(ω))
    return EMEResult{T}(S, modes, [c.length for c in cells], T(ω))
end
