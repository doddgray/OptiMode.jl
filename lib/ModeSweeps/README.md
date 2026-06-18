# ModeSweeps.jl

Batched, **blocking or asynchronous** deployment of OptiMode mode simulations on a
remote SLURM cluster (or local background processes) — parameter sweeps run as SLURM
array jobs, with persistent batch state, live status, partial gathering across Julia
sessions and machines, rsync data transfer, and tabular result I/O. It also renders
**annotated PNG mode-field images** on the worker, and supports **remote automatic
differentiation** (forward & adjoint passes as separate SLURM tasks).

Highlights:

- **Lightweight summaries or full data.** Every batch produces a tabular summary
  (effective/group index, GVD, effective area, polarization, mode-character label,
  run time, host, …) computed on the worker; `save_fields=true` additionally keeps the
  full eigenvectors + E-fields (HDF5).
- **Annotated field PNGs** (`save_plots=true`) rendered on the worker — dependency-free,
  so they work on any headless node — and transferred back in either case.
- **Blocking or async.** `blocking=true` (or `wait_batch`) waits for completion; the
  default returns immediately and you monitor/gather later from any session.
- **Remote AD.** `deploy_forward`/`deploy_backward` run the mode solver and its adjoint
  as different SLURM tasks, exchanging state on the shared filesystem.
- **rsync transfer** of summaries, PNGs, and (on demand) field/AD data in ssh mode.

Full walkthroughs: [`docs/mode_sweeps.md`](../../docs/mode_sweeps.md),
[`examples/remote_mode_solve.jl`](../../examples/remote_mode_solve.jl), and
[`examples/remote_adjoint_optimization.jl`](../../examples/remote_adjoint_optimization.jl).

## Quick start

```julia
using ModeSweeps

# 1. problem definition: a script defining `make_problem(p::NamedTuple)`
#    -> (; ε⁻¹, ∂ε_∂ω, grid [, ∂²ε_∂ω², n₂]); by convention p.ω is the frequency.
#    It is copied into the batch directory and include'd by every worker.

# 2. deploy: a parallelized frequency sweep combined with geometry/material sweeps
batch = frequency_sweep("ridge_wg_setup.jl";
    ω      = 0.55:0.005:0.75,          # swept (varies fastest)
    w_top  = [1.4, 1.7, 2.0],          # swept geometry parameter
    T      = 35.0,                     # fixed material parameter
    nev    = 2,
    save_fields = false,               # summary-only transfer (set true for full fields)
    save_plots  = true,                # annotated PNG per mode field, transferred back
    blocking    = false,               # async (default); blocking=true waits for completion
    solver = "KrylovKitEigsolve()",    # or e.g. "GPUSolver(Float32)"
    slurm  = SlurmConfig(time="0:30:00", partition="general", max_concurrent=50,
                         ssh="me@cluster", remote_dir="/scratch/me/sw1"))  # rsync transfer

# … or with explicit parameter sets / Cartesian grids:
batch = deploy_batch("ridge_wg_setup.jl", param_grid(ω=ωs, w_top=ws); nev=2)
batch = deploy_batch("ridge_wg_setup.jl", [(; ω=0.64, w_top=1.7), (; ω=0.66, w_top=1.4)])

# 3. anytime, in any Julia session / on any machine:
batch = load_batch("/scratch/me/sw1")           # state persisted at deployment
batch_status(batch)                             # done/failed/pending + squeue info
wait_batch(batch)                               # (optional) block until complete
rows  = gather_batch(batch)                     # partial results OK while running;
                                                # rsyncs summaries+PNGs, writes summary.{csv,tsv,json}
rows  = load_summary("/scratch/me/sw1/summary.csv")   # reload for analysis

fd  = load_fields(batch, 7)                     # full fields (save_fields=true): (; ω, kmags, evecs, Es, …)
png = plot_paths(batch, 7)                       # annotated PNGs (save_plots=true), one per band
```

Each summary row holds the swept parameters plus, per band: wavenumber `kmag`,
effective index `neff`, group index `ng`, group velocity dispersion `gvd`, effective
area `Aeff`, polarization fractions `pol_x`/`pol_y`/`pol_z` (+ dominant `pol_axis`),
the mode-character label (`label`/`mode_pol`/`mode_m`/`mode_n`/`hg_rel_error`/`te_frac`),
the Kerr columns `dneff_kerr`/`dn_max` (zero for linear solves), and execution metadata
(`host`, `runtime_s`, `started`, `finished`, `slurm_job`, `slurm_task`). Rows are
Tables.jl-compatible (`DataFrame(rows)` just works).

**Remote automatic differentiation.** Run the mode solver and its adjoint as separate
SLURM tasks that exchange state on the shared filesystem:

```julia
fwd = deploy_forward("ridge_wg_setup.jl", [(; ω=1/1.55, w_top=1.7)]; nev=1, blocking=true)
sol = forward_solution(fwd)                                  # primal: (; ω, kmags, evecs, ε⁻¹, …)
bwd = deploy_backward(fwd, [(; k̄=[1/sol.ω])]; blocking=true) # objective L = neff = kmag/ω
g   = gradient_result(bwd)                                   # (; ω_bar, ε⁻¹_bar) = ∂L/∂(ω, ε⁻¹)
# …or both passes in one blocking call:
r   = remote_value_and_gradient("ridge_wg_setup.jl", (; ω=1/1.55, w_top=1.7), [1/(1/1.55)]; nev=1)
```

**Kerr power sweeps**: if `make_problem` returns an `n₂` map (μm²/W, e.g. from
`DielectricSmoothing.smooth_scalar` + `MaterialDispersion.kerr_n2`), any parameter set
containing an optical power `P` (W) is solved with the first-order power correction
(`ModeAnalysis.solve_k_kerr`), so power sweeps deploy exactly like any other parameter
sweep — see `examples/kerr_power_sweep_setup.jl` at the repository root.

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

Full documentation with the underlying physics/mathematics and usage examples:
[`docs/mode_sweeps.md`](../../docs/mode_sweeps.md) at the repository root.
