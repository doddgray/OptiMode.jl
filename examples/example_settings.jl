# Shared, a-la-carte accuracy/run-mode settings for the paper-reproduction, EME, and designer
# examples (the `*_reproduction*`/`*_designer*`/EME family that `include` this file — directly
# or via `paper_reproductions_common.jl` / `eme_reproductions_common.jl` / `designer_common.jl`).
#
# Every knob can be set three equivalent ways, resolved with this precedence
# (highest wins): command-line flag > environment variable > the script's own
# `example_settings(...)` keyword > the built-in default below. That lets a user override a
# script's tuned defaults at the command line or via the environment without editing source,
# while each example still ships sensible, pre-validated defaults.
#
#   julia examples/tantala_gvd_black2021.jl --resolution-scale=2 --n-freqs=25
#   OPTIMODE_N_FREQS=25 julia examples/tantala_gvd_black2021.jl
#   example_settings(n_freqs=25)   # inside the script itself
#
# See examples/README.md for the full settings reference and worked examples.

export ExampleSettings, example_settings, mk_grid

"""
    ExampleSettings

Resolved a-la-carte simulation settings for one example run.

- `resolution_scale` — multiplies every example's baseline grid point count (finer/coarser mesh
  at fixed physical domain size). `1.0` reproduces the script's validated default grid.
- `domain_scale` — multiplies every example's baseline physical domain size (`Lx`, `Ly`), i.e.
  how far the simulation boundary sits from the waveguide core; grid point counts scale with it
  too so the resolution (points/µm) stays fixed when only `domain_scale` changes.
- `n_freqs` — number of wavelengths (or other swept parameter) in a *mode-solve* sweep (isolated
  dispersion, QPM tuning, supermode crossing, …) — the expensive, per-point-eigensolve knob.
- `n_dense` — number of points in a *cheap, closed-form* dense curve (interpolated transmission,
  analytic gain/QPM-mismatch spectra) — free to make large; only affects plot smoothness.
- `n_eme_freqs` — number of wavelengths for a genuinely EME-solved (`eme`/`power_coupling`) dense
  transmission overlay, where the example provides one (see `tfln_combiner_kwolek2026.jl`).
- `n_cells` — number of cells in an EME cascade (`eme_transmission`/`Cell` count).
- `quality` — `:low`/`:medium`/`:high`/`:ultra` bundled preset for `resolution_scale`,
  `domain_scale`, and `n_cells` (see `_QUALITY_PRESETS` below). `:medium` reproduces the
  pre-existing `1.0`/`1.0`/`6` defaults. This is a convenience layer under the a-la-carte knobs:
  precedence for each of those three fields is (highest wins) CLI flag > env var > script kwarg >
  `quality` preset > built-in hardcoded default — so a script that passes its own tuned
  `resolution_scale`/`domain_scale`/`n_cells` keyword still wins over the preset, but a script
  that leaves them unset can be bumped from the command line with e.g. `--quality=high` without
  touching source.
- `run_mode` — `:local` (default) or `:slurm`. These example scripts run their (few-point)
  mode-solve sweeps inline; `:slurm` is accepted and resolved consistently but only *acted on* by
  scripts that say so in their header. For cluster-scale sweeps use the dedicated ModeSweeps
  entry points (`remote_mode_solve.jl`, `remote_adjoint_optimization.jl`,
  `tfln_ppln_geometry_sweep_deploy.jl`), which already accept a `SlurmConfig`.
- `slurm` — a `ModeSweeps.SlurmConfig` (or `nothing`); only settable via the `example_settings`
  keyword (not CLI/ENV, since it has many sub-fields) — see `remote_mode_solve.jl` for its API.
"""
Base.@kwdef struct ExampleSettings
    resolution_scale::Float64 = 1.0
    domain_scale::Float64 = 1.0
    n_freqs::Int = 9
    n_dense::Int = 400
    n_eme_freqs::Int = 15
    n_cells::Int = 6
    quality::Symbol = :medium
    run_mode::Symbol = :local
    slurm::Any = nothing
end

function Base.show(io::IO, cfg::ExampleSettings)
    print(io, "ExampleSettings(resolution_scale=", cfg.resolution_scale,
          ", domain_scale=", cfg.domain_scale, ", n_freqs=", cfg.n_freqs,
          ", n_dense=", cfg.n_dense, ", n_eme_freqs=", cfg.n_eme_freqs,
          ", n_cells=", cfg.n_cells, ", quality=", cfg.quality,
          ", run_mode=", cfg.run_mode, ")")
end

# Bundled resolution/boundary-distance/EME-cell-count presets (see `quality` docstring above).
const _QUALITY_PRESETS = (
    low    = (resolution_scale = 0.5, domain_scale = 0.8, n_cells = 3),
    medium = (resolution_scale = 1.0, domain_scale = 1.0, n_cells = 6),
    high   = (resolution_scale = 1.5, domain_scale = 1.2, n_cells = 10),
    ultra  = (resolution_scale = 2.5, domain_scale = 1.5, n_cells = 16),
)

# ---------------------------------------------------------------------------------------
# kwarg / CLI / ENV resolution
# ---------------------------------------------------------------------------------------

const _SETTINGS_SPEC = (
    (:resolution_scale, 1.0,     s -> parse(Float64, s)),
    (:domain_scale,      1.0,     s -> parse(Float64, s)),
    (:n_freqs,           9,       s -> parse(Int, s)),
    (:n_dense,           400,     s -> parse(Int, s)),
    (:n_eme_freqs,       15,      s -> parse(Int, s)),
    (:n_cells,           6,       s -> parse(Int, s)),
    (:quality,           :medium, s -> Symbol(lowercase(s))),
    (:run_mode,          :local,  s -> Symbol(lowercase(s))),
)

_cli_flag(name::Symbol) = "--" * replace(String(name), "_" => "-")
_env_key(name::Symbol) = "OPTIMODE_" * uppercase(String(name))

"Value of `--flag=value` or `--flag value` in `ARGS`, or `nothing` if absent."
function _cli_value(name::Symbol)
    flag = _cli_flag(name)
    for (i, a) in enumerate(ARGS)
        if startswith(a, flag * "=")
            return a[length(flag)+2:end]
        elseif a == flag && i < length(ARGS)
            return ARGS[i+1]
        end
    end
    return nothing
end

_env_value(name::Symbol) = get(ENV, _env_key(name), nothing)

function _print_settings_help()
    println("Available example settings (CLI flag / env var / keyword — highest precedence first):")
    for (name, default, _) in _SETTINGS_SPEC
        @printf("  %-22s  %-28s  %-14s  (default %s)\n",
                _cli_flag(name), _env_key(name), string(name), string(default))
    end
    println("  --help / -h            (this message)")
    println("\n`quality` bundles resolution_scale/domain_scale/n_cells (a-la-carte still wins):")
    for (name, vals) in pairs(_QUALITY_PRESETS)
        @printf("  %-8s  resolution_scale=%-4s domain_scale=%-4s n_cells=%s\n",
                name, vals.resolution_scale, vals.domain_scale, vals.n_cells)
    end
    println("\nExamples:")
    println("  julia --project=. examples/tantala_gvd_black2021.jl --resolution-scale=2 --n-freqs=25")
    println("  OPTIMODE_N_FREQS=25 julia --project=. examples/tantala_gvd_black2021.jl")
    println("  julia --project=. examples/tantala_gvd_black2021.jl --quality=high")
    println("See examples/README.md for the full reference.")
end

"""
    example_settings(; kwargs...) -> ExampleSettings

Resolve this run's settings. Each field may be set (in increasing precedence) by the built-in
default, a keyword passed here (a script's own tuned default), an `OPTIMODE_<NAME>` environment
variable, or a `--<name-with-dashes>[=value]` command-line flag. `--help`/`-h` prints the
settings reference and exits. `slurm` (a `ModeSweeps.SlurmConfig`) is keyword-only.
"""
function example_settings(; kwargs...)
    if "--help" in ARGS || "-h" in ARGS
        _print_settings_help()
        exit(0)
    end
    kw = Dict{Symbol,Any}(kwargs)

    # `quality` resolves first (same kwarg > env > cli precedence as everything else) since it
    # picks the preset that supplies the *defaults* for resolution_scale/domain_scale/n_cells.
    qv = haskey(kw, :quality) ? kw[:quality] : :medium
    qev = _env_value(:quality); qev === nothing || (qv = Symbol(lowercase(qev)))
    qcv = _cli_value(:quality); qcv === nothing || (qv = Symbol(lowercase(qcv)))
    haskey(_QUALITY_PRESETS, qv) ||
        throw(ArgumentError("quality must be one of $(join(keys(_QUALITY_PRESETS), ", ")), got $(repr(qv))"))
    preset = _QUALITY_PRESETS[qv]

    vals = Dict{Symbol,Any}(:quality => qv)
    for (name, default, parse_fn) in _SETTINGS_SPEC
        name === :quality && continue
        v = haskey(preset, name) ? preset[name] : default   # preset supplies the "default" tier
        v = haskey(kw, name) ? kw[name] : v                 # script kwarg beats the preset
        ev = _env_value(name); ev === nothing || (v = parse_fn(ev))
        cv = _cli_value(name); cv === nothing || (v = parse_fn(cv))
        vals[name] = v
    end
    vals[:run_mode] in (:local, :slurm) ||
        throw(ArgumentError("run_mode must be :local or :slurm, got $(repr(vals[:run_mode]))"))
    cfg = ExampleSettings(; vals..., slurm=get(kw, :slurm, nothing))
    println("== settings: ", cfg, " ==")
    cfg
end

# ---------------------------------------------------------------------------------------
# Grid construction from resolution_scale / domain_scale
# ---------------------------------------------------------------------------------------

"""
    mk_grid(cfg, Lx0, Ly0, nx0, ny0) -> Grid

Build the simulation `Grid` from an example's validated baseline domain size (`Lx0`, `Ly0`, µm)
and point counts (`nx0`, `ny0`) at `cfg.resolution_scale = cfg.domain_scale = 1`. `domain_scale`
scales the physical domain (how far the boundary sits from the core), carrying the point count
along so the resolution (points/µm) is unchanged; `resolution_scale` then further refines/coarsens
the mesh at whatever domain size resulted."""
function mk_grid(cfg::ExampleSettings, Lx0::Real, Ly0::Real, nx0::Integer, ny0::Integer)
    Lx, Ly = Lx0 * cfg.domain_scale, Ly0 * cfg.domain_scale
    nx = max(4, round(Int, nx0 * cfg.domain_scale * cfg.resolution_scale))
    ny = max(4, round(Int, ny0 * cfg.domain_scale * cfg.resolution_scale))
    Grid(Lx, Ly, nx, ny)
end
