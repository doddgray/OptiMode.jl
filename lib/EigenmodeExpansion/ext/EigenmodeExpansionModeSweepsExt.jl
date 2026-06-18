# ModeSweeps integration for EigenmodeExpansion.
#
# EME's per-cell mode solves are deployed as a ModeSweeps batch (one task per
# cell, swept over the cell index and frequency); the device S-matrix is then
# re-assembled locally from the gathered per-cell fields. This reuses the
# existing SLURM array-job / parameter-sweep machinery unchanged — each EME cell
# is just a `make_problem` cross-section.

module EigenmodeExpansionModeSweepsExt

using EigenmodeExpansion
using EigenmodeExpansion: Cell, assemble_eme
using ModeSweeps
using ModeSweeps: param_grid, deploy_batch, wait_batch, load_fields, SlurmConfig, BatchInfo

function EigenmodeExpansion.deploy_eme(setup_file::AbstractString, num_cells::Int;
        ω, nev::Int=2, name::AbstractString="eme", backend::Symbol=:slurm,
        slurm::SlurmConfig=SlurmConfig(), blocking::Bool=false, kwargs...)
    params = param_grid(; cell=1:num_cells, ω=ω)          # cell varies fastest
    return deploy_batch(setup_file, params; name, nev, save_fields=true,
        backend, slurm, blocking, kwargs...)
end

function EigenmodeExpansion.gather_eme(batch::BatchInfo, cells::AbstractVector{Cell},
        materials, ω, grid; conjugate::Bool=false, reg::Real=1e-9,
        reciprocity::Bool=true, wait::Bool=true)
    wait && wait_batch(batch)
    n = length(cells)
    kmags_per_cell = Vector{Vector{Float64}}(undef, n)
    evecs_per_cell = Vector{Vector{Vector{ComplexF64}}}(undef, n)
    for i in 1:n
        lf = load_fields(batch, i)                        # task i ↔ cell i (cell varies fastest)
        kmags_per_cell[i] = collect(Float64.(lf.kmags))
        evecs_per_cell[i] = [collect(ComplexF64.(ev)) for ev in lf.evecs]
    end
    return assemble_eme(cells, kmags_per_cell, evecs_per_cell, materials, ω, grid;
        conjugate, reg, reciprocity)
end

end # module
