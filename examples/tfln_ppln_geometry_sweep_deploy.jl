# Deploy the dispersion-engineered PPLN geometry sweep (Jankowski et al., Optica 7, 40
# (2020), Fig. 1b–e) as a ModeSweeps batch and build the four geometry maps:
#   (b) required poling period Λ = λ/(2(n_2ω − n_ω))
#   (c) normalized SHG efficiency η₀  (∝ 1/A_eff)
#   (d) group-velocity mismatch Δk′ = (n_g,2ω − n_g,ω)/c
#   (e) GVD k″_ω of the fundamental
#
# The sweep is the (top width × etch depth × {ω_FF, ω_SH}) Cartesian product. Each task
# solves the ridge modes; we pick the quasi-TE₀₀ band (the `label`/`pol_x` columns), pair
# the fundamental and second-harmonic rows at each geometry, and assemble the maps. This is
# exactly the kind of full-resolution, embarrassingly-parallel job ModeSweeps deploys to a
# cluster — the converged 2-µm-mode dispersion that a single workstation cannot reach.
#
#   REMOTE=false (default): a small, coarse LOCAL demo (reduced grid + few geometries) that
#                           runs on this machine and saves the maps.
#   REMOTE=true:            full-resolution SLURM array job over the paper's geometry ranges.
#
# Run:  julia --project=. examples/tfln_ppln_geometry_sweep_deploy.jl

using OptiMode
using ModeSweeps
using Printf
using CairoMakie

const SETUP = joinpath(@__DIR__, "tfln_ppln_geometry_sweep_setup.jl")
const C_UM_FS = 299792458.0 * 1e-9
const λF = 2.05
const ωF, ωSH = 1 / λF, 2 / λF
const REMOTE = false        # flip to true on a SLURM login node for the converged maps

# --- parameter ranges -------------------------------------------------------------------
if REMOTE
    widths = 1.80:0.025:2.00          # top width (µm)         — paper 1800–2000 nm
    etches = 0.31:0.01:0.39           # etch depth (µm)        — paper 310–390 nm
    gridkw = (;)                       # use the setup's converged 12×7 / 320×176 cell
    slurm = SlurmConfig(time="2:00:00", partition="general", mem="16G", cpus_per_task=8,
        max_concurrent=64, ssh="me@login.cluster.edu", remote_dir="/scratch/me/ppln_geo")
    backend = :slurm
    nev = 12
else
    widths = 1.80:0.10:2.00           # coarse 3-point demo
    etches = 0.32:0.04:0.36           # coarse 2-point demo
    gridkw = (; Lx=8.0, Ly=5.0, Nx=180, Ny=112)   # cheaper cell for a laptop demo
    slurm = SlurmConfig()
    backend = :local
    nev = 8
end

batch = frequency_sweep(SETUP;
    ω=[ωF, ωSH], w_top=collect(widths), etch=collect(etches), gridkw...,
    nev=nev, save_fields=false, save_plots=false, mode_labels=true,
    solver="KrylovKitEigsolve()", solver_kwargs=(; k_tol=1e-9, eig_tol=1e-9),
    name="ppln_geo_sweep", backend, slurm)
@info "submitted geometry sweep" tasks = length(widths) * length(etches) * 2
wait_batch(batch)
rows = gather_batch(batch)

# --- pick the ridge quasi-TE₀₀ band per (geometry, ω) and pair FF/SH --------------------
# This thick-slab geometry supports laterally-extended slab modes that are also Eₓ-polarized,
# so the ridge TE₀₀ is the *highest-index* Eₓ-dominant band (ridge confinement raises neff
# above the slab continuum); prefer the Hermite–Gaussian "TE₀₀" label when classification is
# on. For an even more robust, confinement-based selection set `save_fields=true` above and
# pick the band most localized to the ridge column (cf. `ridge_confinement` in the example).
isω(r, ω) = isapprox(r.ω, ω; rtol=1e-6)
function te00(rows, ω, w, e)
    cand = [r for r in rows if isω(r, ω) && r.w_top == w && r.etch == e && isfinite(r.neff)]
    isempty(cand) && return nothing
    te = filter(r -> r.pol_x > 0.5, cand)           # quasi-TE (Eₓ-dominant) bands
    pool = isempty(te) ? cand : te
    labeled = filter(r -> r.label == "TE₀₀", pool)
    isempty(labeled) ? pool[argmax(getfield.(pool, :neff))] : labeled[argmax(getfield.(labeled, :neff))]
end

Λ = fill(NaN, length(widths), length(etches))
η0 = similar(Λ); Δkp = similar(Λ); k2 = similar(Λ)
for (i, w) in enumerate(widths), (j, e) in enumerate(etches)
    F = te00(rows, ωF, w, e); S = te00(rows, ωSH, w, e)
    (F === nothing || S === nothing) && continue
    Λ[i, j] = λF / (2 * (S.neff - F.neff))
    Δkp[i, j] = 1e3 * (S.ng - F.ng) / C_UM_FS                       # fs/mm
    k2[i, j] = 1e3 * F.gvd / (2π * C_UM_FS^2)                        # fs²/mm
    η0[i, j] = 1 / sqrt(F.Aeff^2 * S.Aeff)                           # ∝ 1/A_eff (rel. units)
end
η0 .*= 1100 / maximum(filter(isfinite, η0))                          # scale to paper's peak

# --- maps (Fig. 1b–e) -------------------------------------------------------------------
wn = collect(widths) .* 1e3; en = collect(etches) .* 1e3
fig = Figure(size=(820, 640))
function panel(pos, data, title, cmap)
    ax = Axis(fig[pos...], xlabel="top width (nm)", ylabel="etch depth (nm)", title=title,
        yreversed=true)
    hm = heatmap!(ax, wn, en, data, colormap=cmap)
    Colorbar(fig[pos[1], pos[2]+1], hm)
    ax
end
panel((1, 1), Λ, "(b) poling period Λ (μm)", :viridis)
panel((1, 3), η0, "(c) normalized efficiency (%/W·cm², rel.)", :viridis)
axd = panel((2, 1), Δkp, "(d) group-velocity mismatch Δk′ (fs/mm)", :balance)
contour!(axd, wn, en, Δkp, levels=[0.0], color=:black, linewidth=2)            # Δk′ = 0
panel((2, 3), k2, "(e) GVD k″_ω (fs²/mm)", :balance)
out = joinpath(@__DIR__, "perturbation_output", "jankowski_geometry_maps.png")
save(out, fig)
@printf("saved %s   (Λ range %.2f–%.2f µm, Δk′ range %.0f–%.0f fs/mm)\n",
    out, minimum(filter(isfinite, Λ)), maximum(filter(isfinite, Λ)),
    minimum(filter(isfinite, Δkp)), maximum(filter(isfinite, Δkp)))
