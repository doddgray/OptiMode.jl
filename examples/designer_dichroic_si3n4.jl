# AD+EME DESIGNER — spectrally-selective dichroic filter, swept over cutoff wavelength and
# Si₃N₄ core thickness.
#
# Applies the mode-crossing / dichroic-filter design methodology of Magden et al., "Transmissive
# silicon photonic dichroic filters with spectrally selective waveguides," Nat. Commun. 9, 3009
# (2018) — reproduced on Si SOI in dichroic_filter_magden2018.jl — to a user-controlled
# **Si₃N₄-on-SiO₂** (fully buried/encapsulated) stack, swept over an ARRAY of cutoff targets
# λ_C = 1.00, 1.05, …, 1.50 µm and over SIX core thicknesses: 40, 60, 80, 100, 200, 400 nm.
#
# Dissimilar guides are made exactly as in the paper's Fig. 1b cross-section (not by thickness
# contrast): a solid strip **WGA** (width w_A) next to a sub-wavelength segmented **WGB** — three
# solid rail segments of width w_B separated by gaps g_B, same material and thickness as WGA.
# "This idea of mimicking a continuous waveguide using segments smaller than the wavelength is
# similar to a sub-wavelength grating... we independently engineer the effective and group
# indices... by controlling the widths and separations of individual segments" (§Coupled-mode
# description). The segmented WGB behaves like a lower-index, lower-group-index waveguide than
# the solid WGA — the paper states this is "essential for a good extinction ratio between the
# two output ports," since phase matching at a single λ_C together with a *large* group-index
# mismatch elsewhere is what gives a sharp, well-defined cutoff (§Power roll-off).
#
# ── Design sign convention (from the paper) ──────────────────────────────────────────────────
# Fig. 2a: "δ(λ < λ_C) > 0 and δ(λ > λ_C) < 0" for δ = β_A − β_B. Since dβ/dλ = −2π n_g/λ² < 0,
# a *decreasing* δ(λ) (as here) means d(β_A−β_B)/dλ ∝ −(n_gA − n_gB) < 0, i.e. **n_gA > n_gB** —
# consistent with "WGB possesses spectral characteristics equivalent to a waveguide with a
# smaller refractive index core, resulting in a smaller group index" (§Coupled-mode description).
# We therefore reward designs with n_gA − n_gB > 0, matching the paper's sign.
#
# ── Thin-core caveat ──────────────────────────────────────────────────────────────────────────
# As the Si₃N₄ core thins toward 40 nm, vertical confinement collapses: getting any usable
# n_A/n_B and a significant Δn_g out of a sub-wavelength segmented WGB requires MUCH wider
# lateral widths (2–7.5 µm, vs. ~0.1–0.4 µm at 400 nm) and correspondingly wider WGA–WGB and
# rail-to-rail gaps, a wider simulation domain to fit them, and finer grid spacing to resolve the
# thin core itself. `_si3n4_geometry_defaults`/`_si3n4_grid_baseline` below scale the design-space
# bounds and grid with thickness accordingly (grids are deliberately modest, not fully resolved,
# so the 6-thickness × 11-λ_C sweep finishes in a practical amount of time by default) — but,
# unlike the single-λ_C, single-thickness version of this script, these per-thickness
# bounds/starting points are heuristic (not individually feasibility-probed the way the original
# 400-nm point design was): rerun with
# `--quality=high`/`--quality=ultra` (see example_settings.jl) for production-fidelity results,
# and inspect each case's `*_report.png` for a genuine crossing with the correct Δn_g sign before
# trusting a given thickness's summary grid.
#
# ── Two-stage design per (thickness, λ_C) case (dichroic_designer_common.jl) ─────────────────
# Stage 1 (AD, Enzyme): optimize p=(w_A, w_B, g_B) for phase matching at λ_C with maximum
# n_gA−n_gB (paper sign), via a two-phase warm start that avoids a soft-penalty equilibrium bias
# (see dichroic_designer_common.jl header). Stage 2 (EME): shortest WGA–WGB gap-taper length
# whose quasi-even→quasi-odd mode mixing stays below tolerance at both band edges. Each case also
# gets a dense broadband transmission spectrum and TE00 |E| field profiles at the cutoff and at
# ±0.20 µm, and a combined per-case report figure + CSV trace/summary row
# (`dichroic_designer_common.jl`). Each thickness gets one summary grid of all 11 λ_C spectra on
# matching wavelength axes with the λ_C target marked.
#
# Settings (see examples/README.md): --n-freqs (dispersion/spectrum sweep nodes, default 11),
# --n-dense (dense spectrum resolution, default 400), --resolution-scale/--domain-scale/--n-cells
# or the bundled --quality=low|medium|high|ultra preset (grid + EME taper cell count).
#
# Run:  julia --project=. examples/designer_dichroic_si3n4.jl                    (needs CairoMakie)
#       julia --project=. examples/designer_dichroic_si3n4.jl --quality=high
#       OPTIMODE_QUALITY=high julia --project=. examples/designer_dichroic_si3n4.jl

include(joinpath(@__DIR__, "dichroic_designer_common.jl"))
using CairoMakie

cfg = example_settings(n_freqs=11, n_dense=400)
solver = KrylovKitEigsolve()
mats = (Si₃N₄, SiO₂)
mv = matvals_builder(collect(mats); air=false)   # built at top level — see DichroicGeometry docstring
λC_grid = collect(1.00:0.05:1.50)             # 11 targets, 1000–1500 nm
thicknesses_nm = (40, 60, 80, 100, 200, 400)   # Si₃N₄ core full thickness

"Heuristic (w_A, w_B, g_B) starting point + box bounds (µm) for one Si₃N₄ thickness `t` (µm);
see module header for why thinner cores need much wider structures."
function si3n4_geometry_defaults(t)
    t >= 0.30 && return (p0=[0.28, 0.14, 0.12], lo=[0.20, 0.10, 0.10], hi=[0.40, 0.18, 0.18])
    t >= 0.15 && return (p0=[0.45, 0.22, 0.18], lo=[0.30, 0.15, 0.12], hi=[0.70, 0.32, 0.30])
    t >= 0.09 && return (p0=[3.00, 1.50, 0.80], lo=[2.20, 0.90, 0.50], hi=[4.50, 2.40, 1.50])
    t >= 0.07 && return (p0=[3.60, 1.80, 1.00], lo=[2.60, 1.00, 0.60], hi=[5.50, 2.80, 1.80])
    t >= 0.05 && return (p0=[4.20, 2.10, 1.20], lo=[3.00, 1.20, 0.70], hi=[6.50, 3.40, 2.20])
    return (p0=[5.00, 2.50, 1.40], lo=[3.50, 1.50, 0.90], hi=[7.50, 4.00, 2.60])
end

"Baseline (Lx0, Ly0, nx0, ny0) at `cfg.resolution_scale=cfg.domain_scale=1` for one Si₃N₄
thickness `t` (µm): wider Lx and finer ny for thinner, laterally-wider designs. Kept modest
(rather than fully resolving these thin, wide structures) so the 6-thickness × 11-λC_C sweep
finishes in a practical amount of time by default; rerun a given thickness with
`--quality=high`/`--quality=ultra` for production-fidelity results."
function si3n4_grid_baseline(t)
    t >= 0.30 && return (Lx0=6.0, Ly0=3.0, nx0=48, ny0=30)
    t >= 0.15 && return (Lx0=8.0, Ly0=3.0, nx0=64, ny0=36)
    t >= 0.09 && return (Lx0=12.0, Ly0=2.6, nx0=84, ny0=34)
    t >= 0.07 && return (Lx0=14.0, Ly0=2.6, nx0=98, ny0=38)
    t >= 0.05 && return (Lx0=16.0, Ly0=2.6, nx0=112, ny0=42)
    return (Lx0=18.0, Ly0=2.6, nx0=126, ny0=46)
end

"Coupling-gap taper range (g_start, g_end) (µm): wider structures need wider separation gaps to
reach the weakly-coupled output state."
si3n4_taper_gaps(t) = t >= 0.15 ? (0.30, 2.00) : (0.80, 4.50)

"Build the DichroicGeometry (buried strip WGA / 3-rail segmented WGB) for one Si₃N₄ thickness
`t_nm` (nm)."
function si3n4_geometry(t_nm, cfg, mv)
    t = 1e-3 * t_nm
    rail(cx, w) = MaterialShape(Cuboid([cx, 0.0], [w, t], [1.0 0.0; 0.0 1.0]), 1)
    wgA_shapes(p) = (rail(0.0, p[1]),)
    wgB_shapes(p) = (w = p[2]; g = p[3]; (rail(-(w + g), w), rail(0.0, w), rail(w + g, w)))
    function coupled_shapes(p, gap)
        w_A, w_B, g_B = p
        xA, xB = -(gap/2 + w_A/2), gap/2 + w_B/2
        (rail(xA, w_A), rail(xB, w_B), rail(xB + w_B + g_B, w_B), rail(xB + 2*(w_B + g_B), w_B))
    end
    d = si3n4_geometry_defaults(t)
    gb = si3n4_grid_baseline(t)
    g_start, g_end = si3n4_taper_gaps(t)
    grid = mk_grid(cfg, gb.Lx0, gb.Ly0, gb.nx0, gb.ny0)
    DichroicGeometry(; tag=@sprintf("Si3N4_t%03dnm", t_nm), mats, mv, wgA_shapes, wgB_shapes, coupled_shapes,
        mindsA=(1, 2), mindsB=(1, 1, 1, 2), mindsAB=(1, 1, 1, 1, 2), grid,
        p0=d.p0, lo=d.lo, hi=d.hi, g_start=g_start, g_end=g_end)
end

println("== Si₃N₄ dichroic designer: λ_C ∈ [", λC_grid[1], ", ", λC_grid[end],
        "] µm × t ∈ ", thicknesses_nm, " nm ==")
all_results = Dict{Int,Any}()
for t_nm in thicknesses_nm
    @printf("\n---- Si₃N₄ t=%d nm ----\n", t_nm)
    geo = si3n4_geometry(t_nm, cfg, mv)
    all_results[t_nm] = run_dichroic_sweep(geo, λC_grid, cfg, solver; xlims=(0.6, 1.9))
end
println("\ndone: ", length(thicknesses_nm), " thicknesses × ", length(λC_grid), " cutoff targets → ",
        length(thicknesses_nm) * length(λC_grid), " cases. Summary grids + CSVs → ", OUTDIR)
