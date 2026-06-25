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
    deploy_eme(setup_file, num_cells; ω, nev=2, kwargs...) -> BatchInfo

Deploy the per-cell mode solves of an EME stack as a ModeSweeps batch (one SLURM
array task per cell). Requires `ModeSweeps` to be loaded; see the
`EigenmodeExpansionModeSweepsExt` extension. The `setup_file`'s `make_problem(p)`
should return the cell-`p.cell` cross-section problem (see [`cell_problem`](@ref)).
"""
function deploy_eme end

"""
    gather_eme(batch, cells, materials, ω, grid; kwargs...) -> EMEResult

Gather a completed [`deploy_eme`](@ref) batch and re-assemble the device
S-matrix. Requires `ModeSweeps` to be loaded.
"""
function gather_eme end

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
                      reciprocity::Bool=true)
    n = length(cells)
    modes = map(1:n) do i
        ε⁻¹, ∂ε_∂ω = cell_dielectric(cells[i].cross_section, materials, ω, grid)
        [build_mode(ω, kmags_per_cell[i][j], evecs_per_cell[i][j], ε⁻¹, ∂ε_∂ω, grid; conjugate)
         for j in eachindex(kmags_per_cell[i])]
    end
    S = _assemble(modes, [c.length for c in cells]; conjugate, reg, reciprocity)
    T = typeof(float(ω))
    return EMEResult{T}(S, modes, [c.length for c in cells], T(ω))
end
