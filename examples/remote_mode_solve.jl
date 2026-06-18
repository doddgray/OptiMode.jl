# Remote, SLURM-managed mode solves & sweeps with ModeSweeps.
#
# This walkthrough shows how to deploy mode-solver jobs to a remote SLURM cluster
# (blocking and asynchronously), monitor and gather them from any Julia session or
# machine, choose between lightweight tabular summaries and full field data, and collect
# the annotated PNG mode-field images rendered on the workers. Everything transfers over
# rsync; the batch directory is fully self-describing, so you can disconnect and pick the
# batch back up later from a different computer.
#
# Run the heavy parts on a login node that can `sbatch`, or set `backend=:local` to try
# the whole machinery on your laptop (small grids recommended).
#
#     julia --project=. examples/remote_mode_solve.jl
#
# Prerequisite: the cluster has this project instantiated on a filesystem shared with the
# compute nodes (so the workers can `using OptiMode`).

using OptiMode                       # re-exports ModeSweeps
using ModeSweeps

const SETUP = joinpath(@__DIR__, "ridge_wg_setup.jl")

# Pick a backend. On a real cluster use ssh+rsync; `:local` runs background processes.
const REMOTE = false                 # flip to true on a login node / with ssh access
slurm = REMOTE ?
        SlurmConfig(time="0:30:00", partition="general", mem="8G", cpus_per_task=4,
            max_concurrent=50, ssh="me@login.cluster.edu", remote_dir="/scratch/me/sw_ridge") :
        SlurmConfig()
backend = REMOTE ? :slurm : :local

# ----------------------------------------------------------------------------------
# 1. Asynchronous deployment of a frequency × geometry sweep
# ----------------------------------------------------------------------------------
# `frequency_sweep` builds the Cartesian product of the swept parameters (ω varies
# fastest) and submits one SLURM array task per parameter set. It returns immediately —
# the jobs run in the background on the cluster.
batch = frequency_sweep(SETUP;
    ω=0.55:0.01:0.75,              # frequency sweep (μm⁻¹); λ = 1/ω from 1.33 to 1.82 μm
    w_top=[1.2, 1.6, 2.0],         # geometry sweep: core width (μm)
    h_core=0.7,                    # fixed core height (μm)
    nev=3,                         # 3 bands per task
    save_fields=false,             # summary-only (set true to also keep eigenvectors+fields)
    save_plots=true,               # render an annotated PNG per mode field on the worker
    mode_labels=true,              # classify each mode (TE₀₀, TM₁₀, …) via Hermite–Gaussian fit
    solver="KrylovKitEigsolve()",  # worker-side eigensolver (string; e.g. "GPUSolver(Float32)")
    solver_kwargs=(; k_tol=1e-9),
    backend, slurm)

@info "submitted" batch

# ----------------------------------------------------------------------------------
# 2. Monitor & gather — works at any time, from any session/machine
# ----------------------------------------------------------------------------------
# In a *fresh* Julia session (even on another computer that mounts the same scratch),
# re-attach to the batch by directory and inspect progress:
#
#     batch = load_batch("/scratch/me/sw_ridge")
#     batch_status(batch)        # done/failed/pending (+ live squeue lines)
#
# Partial gathering is race-free while the batch is still running:
rows = gather_batch(batch; partial=true)     # rsyncs summaries+PNGs back; writes summary.{csv,tsv,json}
@info "gathered so far" n = length(rows)

# Each row carries the worker-computed quantities. For example, the fundamental quasi-TE
# mode of the widest waveguide across the frequency sweep:
te00 = filter(r -> r.status == "done" && r.w_top == 2.0 && startswith(r.label, "TE"), rows)
for r in sort(te00; by=r -> r.ω)
    println("λ=", round(1 / r.ω, digits=3), " μm  ", r.label,
        "  neff=", round(r.neff, digits=4), "  ng=", round(r.ng, digits=4),
        "  GVD=", round(r.gvd, digits=2), "  Aeff=", round(r.Aeff, digits=3), " μm²",
        "  [", r.host, ", ", r.runtime_s, " s]")
end

# The annotated PNG of each mode field (rendered on the worker, transferred back):
@info "PNGs for task 1" plot_paths(batch, 1)

# `rows` is a Tables.jl row table — `using DataFrames; DataFrame(rows)` just works, as
# does `CSV.write("ridge.csv", rows)` or reloading later with `load_summary(path)`.

# ----------------------------------------------------------------------------------
# 3. Blocking (synchronous) deployment
# ----------------------------------------------------------------------------------
# Sometimes you just want to wait. `blocking=true` deploys and returns only once every
# task has finished; equivalently call `wait_batch` on a handle (e.g. after reconnecting
# from another machine with `load_batch`).
batch2 = deploy_batch(SETUP, param_grid(ω=1 / 1.55, w_top=[1.4, 1.8]);
    name="ridge_blocking", nev=2, save_fields=true, save_plots=true,
    backend, slurm, blocking=true)
rows2 = gather_batch(batch2)
@info "blocking batch complete" n = length(rows2)

# Full field data (only for batches deployed with save_fields=true): eigenvectors and
# E-fields per band, fetched on demand (large) and loaded per task.
gather_batch(batch2; fetch_fields=true)        # transfer the HDF5 field files in ssh mode
fd = load_fields(batch2, 1)                    # (; ω, kmags, evecs, Es, grid_Δ, grid_N)
@info "loaded full fields" task = 1 bands = length(fd.evecs) Esize = size(first(fd.Es))
