# ModeSweeps integration for EigenmodeExpansion.
#
# EME's per-cell mode solves are deployed as a ModeSweeps batch, one task per
# (cell, ω): the whole n_cells × n_ω set of cross-section eigensolves is independent
# and farmed as a single SLURM array job. The device S-matrix is re-assembled locally
# per frequency from the gathered per-cell fields. Identical cross-sections are solved
# once (dedup) and shared. This reuses the existing SLURM array-job / parameter-sweep
# machinery unchanged — each EME cell is just a `make_problem` cross-section.

module EigenmodeExpansionModeSweepsExt

using EigenmodeExpansion
using EigenmodeExpansion: Cell, assemble_eme, dedup_groups, _eme_task_map, _eme_task_index
using ModeSweeps
using ModeSweeps: param_grid, deploy_batch, wait_batch, load_fields, SlurmConfig, BatchInfo

# Deploy with explicit cells: dedup identical cross-sections and farm (representative
# cell × ω) tasks. `gather_eme` maps every cell back to its representative.
function EigenmodeExpansion.deploy_eme(setup_file::AbstractString, cells::AbstractVector{Cell};
        ω, nev::Int=2, dedup::Bool=true, name::AbstractString="eme", backend::Symbol=:slurm,
        slurm::SlurmConfig=SlurmConfig(), blocking::Bool=false, kwargs...)
    reps, _ = dedup ? dedup_groups(cells) : (collect(eachindex(cells)), nothing)
    params = param_grid(; cell=reps, ω=ω)                 # (representative cell × ω), cell fastest
    return deploy_batch(setup_file, params; name, nev, save_fields=true,
        backend, slurm, blocking, kwargs...)
end

# Deploy by cell count (legacy, no dedup): one task per (cell, ω).
function EigenmodeExpansion.deploy_eme(setup_file::AbstractString, num_cells::Int;
        ω, nev::Int=2, name::AbstractString="eme", backend::Symbol=:slurm,
        slurm::SlurmConfig=SlurmConfig(), blocking::Bool=false, kwargs...)
    params = param_grid(; cell=1:num_cells, ω=ω)          # cell varies fastest
    return deploy_batch(setup_file, params; name, nev, save_fields=true,
        backend, slurm, blocking, kwargs...)
end

function EigenmodeExpansion.gather_eme(batch::BatchInfo, cells::AbstractVector{Cell},
        materials, ω, grid; dedup::Bool=true, conjugate::Bool=false, reg::Real=1e-9,
        reciprocity::Bool=true, passivity::Symbol=:invert, threaded::Bool=false, wait::Bool=true)
    wait && wait_batch(batch)
    n = length(cells)
    reps, _ = dedup ? dedup_groups(cells) : (collect(1:n), nothing)
    tmap = _eme_task_map(batch.params)
    ωs = ω isa Number ? [ω] : collect(ω)

    results = map(ωs) do ωj
        kmags_per_rep = Vector{Vector{Float64}}(undef, length(reps))
        evecs_per_rep = Vector{Vector{Vector{ComplexF64}}}(undef, length(reps))
        for (k, ci) in enumerate(reps)
            t = _eme_task_index(tmap, ci, ωj)             # task solving representative cell ci at ωj
            lf = load_fields(batch, t)
            kmags_per_rep[k] = collect(Float64.(lf.kmags))
            evecs_per_rep[k] = [collect(ComplexF64.(ev)) for ev in lf.evecs]
        end
        assemble_eme(cells, kmags_per_rep, evecs_per_rep, materials, ωj, grid;
            conjugate, reg, reciprocity, passivity, dedup, threaded)
    end
    return ω isa Number ? only(results) : results
end

end # module
