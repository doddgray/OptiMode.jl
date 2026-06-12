# ModeSweeps.jl

Batched, asynchronous deployment of OptiMode mode simulations — parameter sweeps run
as SLURM array jobs on a cluster with the same packages installed (or as local
background processes), with persistent batch state, live status, partial gathering,
and tabular result I/O.

## Quick start

```julia
using ModeSweeps

# 1. problem definition: a script defining `make_problem(p::NamedTuple)`
#    -> (; ε⁻¹, ∂ε_∂ω, grid [, ∂²ε_∂ω²]); by convention p.ω is the frequency.
#    It is copied into the batch directory and include'd by every worker.

# 2. deploy: a parallelized frequency sweep combined with geometry/material sweeps
batch = frequency_sweep("ridge_wg_setup.jl";
    ω      = 0.55:0.005:0.75,          # swept (varies fastest)
    w_top  = [1.4, 1.7, 2.0],          # swept geometry parameter
    T      = 35.0,                     # fixed material parameter
    nev    = 2,
    save_fields = false,               # summary-only transfer (set true for full fields)
    solver = "KrylovKitEigsolve()",    # or e.g. "GPUSolver(Float32)"
    slurm  = SlurmConfig(time="0:30:00", partition="general", max_concurrent=50))

# … or with explicit parameter sets / Cartesian grids:
batch = deploy_batch("ridge_wg_setup.jl", param_grid(ω=ωs, w_top=ws); nev=2)
batch = deploy_batch("ridge_wg_setup.jl", [(; ω=0.64, w_top=1.7), (; ω=0.66, w_top=1.4)])

# 3. anytime, in any Julia session:
batch = load_batch("modesweeps_freq_sweep")     # state persisted at deployment
batch_status(batch)                             # done/failed/pending + squeue info
rows  = gather_batch(batch)                     # partial results OK while running;
                                                # writes summary.{csv,tsv,json}
rows  = load_summary("modesweeps_freq_sweep/summary.csv")  # reload for analysis

# full mode-field data (batches deployed with save_fields=true)
fd = load_fields(batch, 7)                      # (; ω, kmags, evecs, Es, …)
```

Each summary row holds the swept parameters plus, per band: wavenumber `kmag`,
effective index `neff`, group index `ng`, group velocity dispersion `gvd`, effective
area `Aeff`, and polarization fractions `pol_x`/`pol_y`/`pol_z` with the dominant
`pol_axis`. Rows are Tables.jl-compatible (`DataFrame(rows)` just works).

## Execution backends

- `backend=:slurm` (default): one SLURM **array job**, one task per parameter set.
  Submission is local (`sbatch`, shared filesystem assumed) or remote over ssh
  (`SlurmConfig(ssh="user@cluster", remote_dir=...)`, using rsync for transfer —
  summaries only by default, field files with `gather_batch(...; fetch_fields=true)`).
- `backend=:local`: the same per-task worker runs in local background processes —
  useful for testing and small batches (this is how the test suite exercises the
  whole machinery end-to-end).
- `backend=:none` / `submit=false`: dry run; writes the batch directory including
  `job.sbatch` for manual inspection/submission.

## Batch state & fault tolerance

Everything needed to find, monitor, and gather a batch is written to the batch
directory at deployment time (`batch.toml`, `params.json`/`params.jls`, copied
`setup.jl`, generated `runtask.jl` and `job.sbatch`), so batches can be gathered from
new Julia sessions, days later. Workers write per-task JSON summaries (and optional
HDF5 field files) atomically with `.done`/`.failed` markers; `batch_status` and
`gather_batch(...; partial=true)` therefore work at any time during execution, failed
tasks are reported with their error text (`tasks/task_NNNNNN.failed`), and partial
results load exactly like completed ones.
