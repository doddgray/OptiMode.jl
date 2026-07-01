# OptiMode.jl examples

Every example is a standalone script: `julia --project=. examples/<script>.jl` from the repo
root. Most save plots to `examples/paper_reproduction_output/` (created on first run,
gitignored) and print the path at the end.

## The settings system

The paper-reproduction, EME, and AD-designer examples (everything that `include`s
`paper_reproductions_common.jl`, `eme_reproductions_common.jl`, or `designer_common.jl` — see
the catalog below) share one **a-la-carte settings** module, `example_settings.jl`, so accuracy
and run-mode knobs work the same way everywhere instead of being hardcoded per script.

Every setting can be given three equivalent ways, resolved with this precedence (**highest
wins**):

```
command-line flag  >  environment variable  >  script's own example_settings(...) keyword  >  built-in default
```

The script's own keyword is its pre-validated, tuned default (e.g. a dispersion sweep that's
known to need 13 points to resolve a feature) — CLI/ENV let you override it for a quick
accuracy/speed experiment without editing source.

| Setting             | CLI flag                  | Environment variable        | Default | Meaning |
|----------------------|----------------------------|------------------------------|---------|---------|
| `resolution_scale`   | `--resolution-scale=X`     | `OPTIMODE_RESOLUTION_SCALE`  | `1.0`   | Multiplies the grid point count at fixed physical domain size (finer/coarser mesh). |
| `domain_scale`       | `--domain-scale=X`         | `OPTIMODE_DOMAIN_SCALE`      | `1.0`   | Multiplies the physical domain size (how far the simulation boundary sits from the core); point count scales with it so resolution (points/µm) is unchanged. |
| `n_freqs`            | `--n-freqs=N`              | `OPTIMODE_N_FREQS`           | varies  | Number of wavelengths (or other swept parameter) in a *mode-solve* sweep — the expensive, per-point-eigensolve knob (dispersion sweeps, QPM tuning, supermode crossings, designer validation sweeps). |
| `n_dense`            | `--n-dense=N`              | `OPTIMODE_N_DENSE`           | varies  | Number of points in a *cheap, closed-form* dense curve (interpolated transmission, analytic gain/QPM-mismatch spectra) — free to make large; only affects plot smoothness. |
| `n_eme_freqs`        | `--n-eme-freqs=N`          | `OPTIMODE_N_EME_FREQS`       | `15`    | Number of wavelengths for a genuinely EME-solved (`eme`/`power_coupling`) dense transmission overlay, where an example provides one (currently `tfln_combiner_kwolek2026.jl`). |
| `n_cells`            | `--n-cells=N`              | `OPTIMODE_N_CELLS`           | `6`     | Number of cells in an EME cascade. |
| `run_mode`           | `--run-mode=local\|slurm`  | `OPTIMODE_RUN_MODE`          | `local` | See [Local vs. SLURM](#local-vs-slurm) below. |
| `slurm`              | *(keyword only)*           | —                            | `nothing` | A `ModeSweeps.SlurmConfig`; only settable as an `example_settings(slurm=...)` keyword (it has too many sub-fields for a single flag/env var) — see `remote_mode_solve.jl`. |

`--help` / `-h` on any of these scripts prints the same reference table and exits.

Every script prints its resolved settings on startup, e.g.:

```
== settings: ExampleSettings(resolution_scale=1.0, domain_scale=1.0, n_freqs=7, n_dense=400, n_eme_freqs=15, n_cells=6, run_mode=local) ==
```

### Usage examples

```sh
# Run an example at its default (validated) settings
julia --project=. examples/tantala_gvd_black2021.jl

# Finer grid, more dispersion-sweep points, via CLI flags
julia --project=. examples/tantala_gvd_black2021.jl --resolution-scale=2 --n-freqs=25

# The same, via environment variables (handy for batch scripts / CI)
OPTIMODE_RESOLUTION_SCALE=2 OPTIMODE_N_FREQS=25 julia --project=. examples/tantala_gvd_black2021.jl

# Push the simulation boundary further from the core (fixed resolution) — check for
# boundary-truncation artifacts on a leaky/weakly-confined mode
julia --project=. examples/si3n4_cw_opa_riemensberger2022.jl --domain-scale=1.5

# Cheaper/faster smoke-test run (coarse grid, few points) — good for CI or a quick check
julia --project=. examples/dichroic_filter_magden2018.jl --resolution-scale=0.5 --n-freqs=3 --n-dense=20

# Deeper EME overlay + finer grid together
julia --project=. examples/tfln_combiner_kwolek2026.jl --n-eme-freqs=30 --resolution-scale=1.5

# More EME cascade cells (validation step)
julia --project=. examples/tfln_combiner_kwolek2026.jl --n-cells=12

# List every available setting for a script
julia --project=. examples/tantala_gvd_black2021.jl --help
```

A script can also set its own defaults in code — this is what each example does for its
pre-tuned values (CLI/ENV still override them):

```julia
cfg = example_settings(n_freqs=13, n_dense=401)   # this script's validated defaults
grid = mk_grid(cfg, 5.0, 4.0, 128, 100)             # Grid(Lx0, Ly0, nx0, ny0) at scale=1
```

### Local vs. SLURM

`run_mode` is resolved consistently everywhere, but only the dedicated ModeSweeps-based
examples *act* on `:slurm` today:

- `remote_mode_solve.jl`, `remote_adjoint_optimization.jl`, `tfln_ppln_geometry_sweep_deploy.jl`
  already accept a `ModeSweeps.SlurmConfig` and a `backend` (`:local`/`:slurm`) and dispatch
  frequency/geometry sweeps to a cluster over ssh+rsync (see `SlurmConfig`'s docstring and
  `ridge_wg_setup.jl` for the worker-side "setup script" convention these use).
- The paper-reproduction / EME / AD-designer family (everything using `example_settings`) runs
  its mode-solve sweeps inline, in-process — they're a handful of points (`n_freqs`, typically
  3–15), fast enough on a workstation that cluster dispatch isn't needed. Passing
  `--run-mode=slurm` to one of these is accepted and printed in the resolved settings, but the
  script still runs locally; for cluster-scale sweeps of this physics, use the dedicated
  ModeSweeps entry points above (or increase `n_freqs`/`resolution_scale` and let it run
  longer locally).

## `tfln_combiner_kwolek2026.jl`: two ways to get the dense transmission curve

This example's combiner-response plot overlays two independently-computed curves so you can see
they agree:

1. **Analytic, interpolated** (`T_cross(λ)=sin²(πLΔn(λ)/λ)`, `cfg.n_dense` points): cheap, smooth,
   but `Δn(λ)` is linearly interpolated from the sparse (`cfg.n_freqs`-point) supermode dispersion
   sweep — it does *not* re-solve modes at every plotted wavelength.
2. **True EME** (`directional_coupler_transmission`, `cfg.n_eme_freqs` points): OptiMode's actual
   `eme` scattering matrix, solved fresh at each plotted wavelength (no interpolation), projected
   onto the physical bar/cross ports via `port_transmission`.

Both are exact for a *uniform* coupler in the sense that (1) is just an interpolated evaluation
of the same closed form (2) computes exactly — but (2) is the one that re-solves the modes, so
it's the ground truth wherever (1)'s interpolation is coarse. Increase `--n-eme-freqs` to shrink
the gap between them.

## Example catalog

### Paper reproductions — χ²/χ³ dispersion + OPA (share `paper_reproductions_common.jl`)
- `tantala_gvd_black2021.jl` — Ta₂O₅ GVD engineering + Kerr FWM gain (Black et al., Opt. Lett. 2021).
- `si3n4_cw_opa_riemensberger2022.jl` — Si₃N₄ dispersion + continuous-wave Kerr OPA (Riemensberger et al., Nature 2022).
- `pplt_allband_opa_kuznetsov2026.jl` — PPLT cascaded-χ² all-band OPA (Kuznetsov et al., arXiv:2605.22704).
- `ppln_reconfigurable_opa_han2026.jl` — x-cut TFLN χ² QPM OPA (Han et al., arXiv:2602.00246).
- `ppln_thermal_tuning_han2026.jl` — electro-thermal QPM tuning companion to the above.

### EME reproductions + AD designers (share `eme_reproductions_common.jl` / `designer_common.jl`)
- `dichroic_filter_magden2018.jl` — Si SOI thickness-contrast dichroic filter (Magden et al., Nat. Commun. 2018).
- `tfln_combiner_kwolek2026.jl` — TFLN >1-octave wavelength combiner (Kwolek et al., arXiv:2603.27034); see the dense-transmission note above.
- `designer_qpm_mgoln_1310.jl` — AD-optimized χ² SHG QPM design, MgO:LiNbO₃ rib, new 1310→655 nm target.
- `designer_dispersion_tantala_1p3um.jl` — AD-optimized zero-GVD design, Ta₂O₅ air-clad core, new 1.30 µm target.
- `designer_dichroic_si3n4.jl` — AD-optimized EME dichroic cutoff, Si₃N₄ thickness-contrast coupler, new 1.00 µm target.

### Dispersion-engineered PPLN (Jankowski et al., Optica 2020)
- `tfln_ppln_jankowski2020.jl` / `tfln_ppln_jankowski2020_common.jl` — dispersion-engineered nanophotonic PPLN reproduction.
- `tfln_ppln_geometry_sweep_setup.jl` / `tfln_ppln_geometry_sweep_deploy.jl` — the converged geometry-map version of the above, deployed as a ModeSweeps/SLURM batch.

### Adjoint / automatic differentiation
- `bragg_waveguide_period_adjoint.jl` — 3D periodic (Bragg) waveguide mode solving + adjoint sensitivity to the period.
- `tfln_bragg_waveguide_dispersion_adjoint.jl` — dispersion + adjoint sensitivities of a width-modulated TFLN Bragg waveguide.
- `tfln_shg_dispersion.jl` — forward SHG phase-matching dispersion calculation for TFLN.
- `tfln_shg_temperature_angle_ad.jl` — forward/reverse AD of SHG phase matching w.r.t. temperature and crystal orientation.
- `ad_backend_benchmarks.jl` — timing comparison of the AD backends (ForwardDiff/Zygote/Enzyme/Mooncake/hybrid).

### Remote / SLURM deployment (ModeSweeps)
- `remote_mode_solve.jl` — deploy/monitor/gather mode-solver sweeps on a SLURM cluster (or `:local`).
- `remote_adjoint_optimization.jl` — SLURM-managed automatic differentiation of the mode solver.
- `ridge_wg_setup.jl`, `kerr_power_sweep_setup.jl`, `eme_coupler_setup.jl` — ModeSweeps worker "setup scripts" used by the above.

### Perturbation theory (first-order index/loss corrections)
- `perturbation_kerr_spm.jl` — χ³ self-phase modulation.
- `perturbation_xpm.jl` — χ³ cross-phase modulation.
- `perturbation_tpa_loss.jl` — two-photon-absorption loss.
- `perturbation_cascaded_chi2.jl` — cascaded-χ² effective Kerr nonlinearity.
- `perturbation_shg_efficiency.jl` — χ² SHG normalized efficiency (TFLN).
- `perturbation_thermo_optic.jl` — thermo-optic tuning.
- `perturbation_substrate_leakage.jl` — substrate-leakage loss.
- `perturbation_surface_roughness_loss.jl` — sidewall-roughness scattering loss (Payne–Lacey).
- `perturbation_userdefined_index.jl` — arbitrary user-specified Δn(x,y) perturbation.

### Other mode-solver demos
- `kerr_si3n4_waveguide.jl` — Kerr power-dependent mode solves for a Si₃N₄ waveguide.
- `hermite_gaussian_mode_labeling.jl` — Hermite–Gaussian mode classification vs. node counting.
- `forced_grid_convergence.jl` — finite-grid convergence study for a Si₃N₄ ridge.
- `eme_adiabatic_coupler.jl` — EME of an adiabatic coupler driven from a GDSFactory GDS layout.
- `material_fitting_sellmeier.jl` — fitting Sellmeier material models with `MaterialFitting`.

These groups don't yet use the `example_settings` module (several have their own established
sweep/deploy conventions, e.g. the ModeSweeps setup-script pattern) — see each script's header
for its own configuration knobs.
