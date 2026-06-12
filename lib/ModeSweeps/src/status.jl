# Batch status: derived from per-task marker files (works for every backend, in any
# session, while the batch is still running), plus a `squeue` query for SLURM batches
# when available.

"""
    batch_status(batch::BatchInfo; verbose=true) -> NamedTuple

Report the status of a (possibly still running) batch:
`(; total, done, failed, pending, done_tasks, failed_tasks, slurm)`.

Counts come from the per-task completion markers in the batch directory, so this works
from any Julia session and at any time during execution. For SLURM batches with a
recorded job id, `slurm` additionally carries one line per still-queued/running array
task from `squeue` (or `nothing` when unavailable). In ssh mode the marker files are
first synced from the cluster.
"""
function batch_status(batch::BatchInfo; verbose::Bool=true)
    _maybe_fetch_markers!(batch)
    n = length(batch)
    done_tasks = Int[]
    failed_tasks = Int[]
    for i in 1:n
        base = _task_base(batch, i)
        if isfile(base * ".done")
            push!(done_tasks, i)
        elseif isfile(base * ".failed")
            push!(failed_tasks, i)
        end
    end
    pending = n - length(done_tasks) - length(failed_tasks)
    slurm = _squeue_lines(batch)
    st = (; total=n, done=length(done_tasks), failed=length(failed_tasks), pending,
        done_tasks, failed_tasks, slurm)
    if verbose
        @info "batch \"$(get(batch.manifest, "name", "?"))\": $(st.done)/$n done, " *
              "$(st.failed) failed, $(st.pending) pending"
    end
    return st
end

"fetch per-task summaries & markers from the remote cluster (ssh mode only)"
function _maybe_fetch_markers!(batch::BatchInfo; fields::Bool=false)
    ssh = get(batch.manifest, "ssh", "")
    isempty(ssh) && return nothing
    remote = get(batch.manifest, "remote_dir", "")
    isempty(remote) && return nothing
    incl = ["--include=*.json", "--include=*.done", "--include=*.failed"]
    fields && push!(incl, "--include=*.h5")
    cmd = Cmd(vcat(["rsync", "-a"], incl,
        ["--exclude=*", "$ssh:$remote/tasks/", joinpath(batch.dir, "tasks") * "/"]))
    try
        run(cmd)
    catch err
        @warn "could not sync task results from $ssh:$remote" exception = err
    end
    return nothing
end

"squeue lines for the batch's array job, or `nothing` when not applicable/available"
function _squeue_lines(batch::BatchInfo)
    jobid = get(batch.manifest, "slurm_jobid", "")
    (isempty(jobid) || get(batch.manifest, "backend", "") != "slurm") && return nothing
    ssh = get(batch.manifest, "ssh", "")
    cmd = ["squeue", "-j", jobid, "-h", "-o", "%i %T %M %R"]
    try
        out = isempty(ssh) ? read(Cmd(cmd), String) : read(_remote_cmd(ssh, cmd), String)
        return filter(!isempty, split(out, '\n'))
    catch
        return nothing
    end
end
