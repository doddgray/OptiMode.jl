# Shared, geometry-agnostic machinery for the dichroic-filter *designer* scripts
# (`designer_dichroic_si3n4.jl`, `designer_dichroic_linbo3_litao3.jl`): given a
# `DichroicGeometry` (a material stack + WGA/WGB/coupled shape builders for one physical
# waveguide family — buried strip, or rib-on-slab), sweep the two-stage AD+EME design of
# `designer_dichroic_si3n4.jl` over an array of target cutoff wavelengths λ_C, saving a
# per-case report figure + CSV row, then a per-configuration summary grid.
#
# Stage 1 (AD, Enzyme) and Stage 2 (EME taper-length search) are exactly the two-phase
# warm-start / adiabatic-taper-search methods validated for the Si₃N₄ solid/segmented design in
# the single-target version of this file; see that script's header for the physics and the
# soft-penalty-equilibrium-bias fix they encode. This file only factors those two stages, plus a
# dense transmission spectrum and TE00 field-profile helper, so they can be reused across the
# λ_C sweep and across material/geometry families without duplicating the optimization logic.

include(joinpath(@__DIR__, "designer_common.jl"))
include(joinpath(@__DIR__, "eme_reproductions_common.jl"))

"""
    DichroicGeometry

One physical waveguide family (a material stack + shape builders) for the dichroic designer.
`wgA_shapes`/`wgB_shapes` take the AD design vector `p = [w_A, w_B, g_B]` (µm) and return the
isolated-guide `MaterialShape` tuple; `coupled_shapes(p, gap)` returns the WGA+WGB cross-section
with edge-to-edge coupling gap `gap`, WGA centred at negative x and WGB at positive x so the
WGA|WGB divider is always x=0 (`x_split=0.0` is used uniformly below).

`mv` must be built at TOP-LEVEL script scope via `matvals_builder(collect(mats); air=false)`
before constructing this struct (i.e. passed in already-built, not built inside a function here):
`_f_ε_mats`/`matvals_builder` generate their returned closure via runtime `eval`, so calling it
from inside a nested function hits a Julia world-age error the first time the closure is used
later in that same call (`MethodError: ... method too new to be called from this world context`)
— building it at top level and threading it through as a field sidesteps that entirely.
"""
Base.@kwdef struct DichroicGeometry
    tag::String
    mats::Tuple
    mv::Function
    wgA_shapes::Function
    wgB_shapes::Function
    coupled_shapes::Function
    mindsA::Tuple
    mindsB::Tuple
    mindsAB::Tuple
    grid::Any
    p0::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
    g_start::Float64 = 0.30
    g_end::Float64 = 2.00
end

# ---------------------------------------------------------------------------------------
# Stage 1: two-phase AD crossing + max-Δn_g optimization (Enzyme)
# ---------------------------------------------------------------------------------------

"""
    optimize_dichroic_crossing(geo, λC_target, solver; β_gvm, iters1a, iters1b, lr1a, lr1b)

Two-phase warm-start AD optimization of `p=(w_A,w_B,g_B)`: phase 1a locks the crossing
`(n_A-n_B)²→0` at `λC_target` (β=0); phase 1b then does a short, small-step refinement of the
combined loss `(n_A-n_B)² - β·(n_gA-n_gB)` from that crossing, nudging the group-index mismatch
in the paper's sign (n_gA>n_gB) without reopening the gap (see module header)."""
function optimize_dichroic_crossing(geo::DichroicGeometry, λC_target, solver;
        β_gvm=0.002, iters1a=40, iters1b=12, lr1a=0.01, lr1b=0.002)
    om_C = 1 / λC_target
    mv = geo.mv
    grid = geo.grid
    nA_of(p, λ) = neff_of(diel_p(geo.wgA_shapes, mv, geo.mindsA, grid, p, 1/λ)[1], 1/λ, grid, solver)
    nB_of(p, λ) = neff_of(diel_p(geo.wgB_shapes, mv, geo.mindsB, grid, p, 1/λ)[1], 1/λ, grid, solver)

    function loss_grad_crossing(p)
        nA, gnA = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), geo.wgA_shapes, mv, geo.mindsA, grid, p, om_C)
        nB, gnB = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), geo.wgB_shapes, mv, geo.mindsB, grid, p, om_C)
        r = nA - nB
        (r^2, 2r .* (gnA .- gnB))
    end
    function loss_grad_stage1(p)
        nA, gnA   = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), geo.wgA_shapes, mv, geo.mindsA, grid, p, om_C)
        nB, gnB   = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), geo.wgB_shapes, mv, geo.mindsB, grid, p, om_C)
        ngA, gngA = geom_value_grad((ei, de) -> ng_of(ei, de, om_C, grid, solver), geo.wgA_shapes, mv, geo.mindsA, grid, p, om_C)
        ngB, gngB = geom_value_grad((ei, de) -> ng_of(ei, de, om_C, grid, solver), geo.wgB_shapes, mv, geo.mindsB, grid, p, om_C)
        r = nA - nB
        L = r^2 - β_gvm * (ngA - ngB)
        g = 2r .* (gnA .- gnB) .- β_gvm .* (gngA .- gngB)
        (L, g)
    end

    res0 = optimize_design(loss_grad_crossing, geo.p0; lo=geo.lo, hi=geo.hi, iters=iters1a, lr=lr1a, verbose=false)
    res1 = optimize_design(loss_grad_stage1, res0.p; lo=geo.lo, hi=geo.hi, iters=iters1b, lr=lr1b, verbose=false)
    p★ = res1.p
    nA★, nB★ = nA_of(p★, λC_target), nB_of(p★, λC_target)
    ngA★ = ng_of(diel_p(geo.wgA_shapes, mv, geo.mindsA, grid, p★, om_C)[1:2]..., om_C, grid, solver)
    ngB★ = ng_of(diel_p(geo.wgB_shapes, mv, geo.mindsB, grid, p★, om_C)[1:2]..., om_C, grid, solver)
    (; p=p★, mv, nA_of, nB_of, nA=nA★, nB=nB★, ngA=ngA★, ngB=ngB★, res0, res1)
end

# ---------------------------------------------------------------------------------------
# Stage 2: EME adiabatic taper length search
# ---------------------------------------------------------------------------------------

"Shortest K-cell WGA–WGB gap-taper length (of `Ls`) whose EME quasi-even→quasi-odd mode mixing
stays ≤ `tol` at both `λC_target∓0.15` µm."
function adiabatic_taper_length(geo::DichroicGeometry, p★, mats_vec, λC_target, n_cells, solver;
        tol=0.01, Ls=(5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0))
    grid = geo.grid
    λ_short, λ_long = λC_target - 0.15, λC_target + 0.15
    function adiabatic_loss(L, ω)
        edges = collect(range(0.0, L; length=n_cells + 1))
        gaps = collect(range(geo.g_start, geo.g_end; length=n_cells))
        cells = [Cell(i, (edges[i]+edges[i+1])/2, edges[i+1]-edges[i],
                      CrossSection(collect(MaterialShape, geo.coupled_shapes(p★, gaps[i])), collect(Int, geo.mindsAB)))
                 for i in 1:n_cells]
        res = eme(cells, mats_vec, ω, grid, solver; nev=2, k_tol=1e-7)
        power_coupling(res; in_mode=1, out_mode=2)
    end
    losses = Float64[]
    for L in Ls
        push!(losses, max(adiabatic_loss(L, 1/λ_short), adiabatic_loss(L, 1/λ_long)))
    end
    i_ok = findfirst(≤(tol), losses)
    L★ = i_ok === nothing ? Ls[end] : Ls[i_ok]
    (; Ls=collect(Ls), losses, L★, tol, ok=(i_ok !== nothing))
end

# ---------------------------------------------------------------------------------------
# Dense transmission spectrum + TE00 field profiles (quasi-even supermode of the *interaction*
# cross-section, i.e. coupled_shapes at gap=g_start — the ideal-adiabatic estimate of the finished
# device's response, following dichroic_filter_magden2018.jl's convention).
# ---------------------------------------------------------------------------------------

"Dense short-/long-pass transmission spectrum around `λ_center`, from `n_freqs` real supermode
solves of the coupled (interaction-gap) cross-section interpolated onto `n_dense` points."
function dichroic_dense_spectrum(geo::DichroicGeometry, p★, mv, solver, λ_center; n_freqs=11, n_dense=400, halfspan=0.35)
    λs = collect(range(λ_center - halfspan, λ_center + halfspan; length=n_freqs))
    fracA = zero(λs)
    minds = collect(Int, geo.mindsAB)
    for (j, λ) in enumerate(λs)
        sup = supermodes(geo.coupled_shapes(p★, geo.g_start), minds, mv, 1/λ, geo.grid, solver; nev=4)
        fracA[j] = port_fraction(sup[1].E, geo.grid, 0.0)
    end
    λdense = collect(range(λs[1], λs[end]; length=n_dense))
    T_short, T_long = dichroic_spectrum(λs, fracA, λdense)
    (; λs, fracA, λdense, T_short, T_long)
end

"TE00 (quasi-even supermode) |E| field of the coupled interaction cross-section at wavelength `λ`."
function dichroic_field(geo::DichroicGeometry, p★, mv, solver, λ)
    supermodes(geo.coupled_shapes(p★, geo.g_start), collect(Int, geo.mindsAB), mv, 1/λ, geo.grid, solver; nev=4)[1].E
end

# ---------------------------------------------------------------------------------------
# One (geometry, λC_target) case: run both stages, dense spectrum, 3 field profiles; save a
# combined per-case report figure + append a row to the configuration's summary CSV.
# ---------------------------------------------------------------------------------------

"""
    run_dichroic_case(geo, λC_target, cfg, solver) -> NamedTuple

Runs Stage 1 (AD crossing + max Δn_g), Stage 2 (EME adiabatic taper length), a dense broadband
transmission spectrum, and TE00 field profiles at the red-detuned/cutoff/blue-detuned test
wavelengths for one `(geo, λC_target)` case. Saves one combined report figure
`<geo.tag>_λC<λC*1000>nm_report.png` to `OUTDIR`."""
function run_dichroic_case(geo::DichroicGeometry, λC_target, cfg, solver; β_gvm=0.002, taper_tol=0.01)
    mats_vec = collect(geo.mats)
    stage1 = optimize_dichroic_crossing(geo, λC_target, solver; β_gvm=β_gvm)
    stage2 = adiabatic_taper_length(geo, stage1.p, mats_vec, λC_target, cfg.n_cells, solver; tol=taper_tol)
    spec = dichroic_dense_spectrum(geo, stage1.p, stage1.mv, solver, λC_target; n_freqs=cfg.n_freqs, n_dense=cfg.n_dense)

    λs_disp = collect(range(λC_target - 0.35, λC_target + 0.35; length=cfg.n_freqs))
    nA0 = [stage1.nA_of(geo.p0, λ) for λ in λs_disp]; nB0 = [stage1.nB_of(geo.p0, λ) for λ in λs_disp]
    nA1 = [stage1.nA_of(stage1.p, λ) for λ in λs_disp]; nB1 = [stage1.nB_of(stage1.p, λ) for λ in λs_disp]
    λC_achieved = crossing_wavelength(λs_disp, nA1, nB1)

    λ_show = (λC_target - 0.20, λC_target, λC_target + 0.20)   # red / at-cutoff / blue TE00 fields
    Efields = [dichroic_field(geo, stage1.p, stage1.mv, solver, λ) for λ in λ_show]

    tag_λ = @sprintf("%s_λC%04dnm", geo.tag, round(Int, 1000λC_target))
    @printf("[%s] target λC=%.3f µm → p=(%.3f,%.3f,%.3f) µm  n_A=%.4f n_B=%.4f  n_gA-n_gB=%+.3f  λC_achieved=%s  L★=%.0f µm%s\n",
            tag_λ, λC_target, stage1.p..., stage1.nA, stage1.nB, stage1.ngA - stage1.ngB,
            isnan(λC_achieved) ? "none" : string(round(λC_achieved; digits=3)), stage2.L★, stage2.ok ? "" : " (tol not reached)")
    flush(stdout)

    # --- per-case combined report figure -------------------------------------------------------
    xc, yc = grid_coords(geo.grid)
    fig = Figure(size=(1500, 760))
    ax1 = Axis(fig[1, 1], xlabel="Adam iteration", ylabel="loss (log)", yscale=log10, title="Stage 1: AD optimization")
    lines!(ax1, 1:length(stage1.res0.history), max.(stage1.res0.history, 1e-12), color=:dodgerblue, linewidth=2, label="1a: (n_A−n_B)²")
    lines!(ax1, length(stage1.res0.history) .+ (1:length(stage1.res1.history)), max.(abs.(stage1.res1.history), 1e-12),
        color=:crimson, linewidth=2, label="1b: combined")
    axislegend(ax1, position=:rt, labelsize=8)

    ax2 = Axis(fig[1, 2], xlabel="wavelength (µm)", ylabel="n_eff", title="isolated WGA/WGB crossing")
    lines!(ax2, λs_disp, nB0, color=:black, linewidth=1.5, linestyle=:dash, label="WGB start")
    lines!(ax2, λs_disp, nA0, color=:dodgerblue, linewidth=1.5, linestyle=:dash, label="WGA start")
    lines!(ax2, λs_disp, nB1, color=:gray30, linewidth=2, label="WGB opt")
    lines!(ax2, λs_disp, nA1, color=:crimson, linewidth=2, label="WGA opt")
    vlines!(ax2, [λC_target], color=:gray, linestyle=:dot)
    axislegend(ax2, position=:rt, labelsize=8)

    ax3 = Axis(fig[1, 3], xlabel="taper length L (µm)", ylabel="mode mixing", xscale=log10, yscale=log10,
        title="Stage 2: taper length search")
    scatterlines!(ax3, stage2.Ls, max.(stage2.losses, 1e-12), color=:seagreen, linewidth=2, marker=:circle)
    hlines!(ax3, [stage2.tol], color=:gray, linestyle=:dash)
    vlines!(ax3, [stage2.L★], color=:crimson, linestyle=:dot)

    ax4 = Axis(fig[1, 4], xlabel="wavelength (µm)", ylabel="transmission",
        title=@sprintf("dense spectrum (λC target=%.0f nm)", 1000λC_target))
    lines!(ax4, spec.λdense, spec.T_short, color=:dodgerblue, linewidth=2, label="short-pass (WGA)")
    lines!(ax4, spec.λdense, spec.T_long, color=:crimson, linewidth=2, label="long-pass (WGB)")
    vlines!(ax4, [λC_target], color=:gray, linestyle=:dash)
    axislegend(ax4, position=:rc, labelsize=8)

    for (col, (E, λ)) in enumerate(zip(Efields, λ_show))
        lbl = col == 1 ? "red (λ_C−0.2µm)" : col == 2 ? "at λ_C" : "blue (λ_C+0.2µm)"
        ax = Axis(fig[2, col], xlabel="x (µm)", ylabel="y (µm)", aspect=DataAspect(),
            title=@sprintf("TE00 |E|: %s (%.0f nm)", lbl, 1000λ))
        heatmap!(ax, xc, yc, absE_norm(E), colormap=:turbo)
    end
    Label(fig[2, 4], @sprintf("%s\nλC_target=%.0f nm", geo.tag, 1000λC_target); tellwidth=false, tellheight=false)
    save(joinpath(OUTDIR, tag_λ * "_report.png"), fig)

    # --- optimization trace data ----------------------------------------------------------------
    open(joinpath(OUTDIR, tag_λ * "_trace.csv"), "w") do io
        println(io, "phase,iter,loss")
        for (i, ℓ) in enumerate(stage1.res0.history); println(io, "1a,", i, ",", ℓ); end
        for (i, ℓ) in enumerate(stage1.res1.history); println(io, "1b,", i, ",", ℓ); end
    end

    (; geo, λC_target, λC_achieved, p=stage1.p, nA=stage1.nA, nB=stage1.nB,
       ngA=stage1.ngA, ngB=stage1.ngB, L★=stage2.L★, taper_ok=stage2.ok, spec, Efields, λ_show,
       λs_disp, nA0, nB0, nA1, nB1)
end

# ---------------------------------------------------------------------------------------
# Full λC sweep for one geometry: run every case, write a summary CSV, and produce a summary
# grid of subplots (dense transmission spectra, matching x-axes, λC_target as a vertical dashed
# line) across all swept designs.
# ---------------------------------------------------------------------------------------

"""
    run_dichroic_sweep(geo, λC_grid, cfg, solver; xlims=(0.5, 2.0)) -> Vector of case results

Runs [`run_dichroic_case`](@ref) for every `λC_target` in `λC_grid`, writes
`<geo.tag>_sweep_summary.csv`, and saves `<geo.tag>_summary_grid.png` — a grid of subplots (one
per λC_target) of the dense transmission spectrum, all sharing the x-axis range `xlims` so the
cutoffs are directly comparable, each with its target λC_target marked by a vertical dashed
line."""
function run_dichroic_sweep(geo::DichroicGeometry, λC_grid, cfg, solver; xlims=(0.5, 2.0), β_gvm=0.002, taper_tol=0.01)
    cases = [run_dichroic_case(geo, λC, cfg, solver; β_gvm=β_gvm, taper_tol=taper_tol) for λC in λC_grid]

    open(joinpath(OUTDIR, geo.tag * "_sweep_summary.csv"), "w") do io
        println(io, "λC_target_um,λC_achieved_um,w_A_um,w_B_um,g_B_um,n_A,n_B,ngA_minus_ngB,L_star_um,taper_ok")
        for c in cases
            println(io, join((c.λC_target, c.λC_achieved, c.p..., c.nA, c.nB, c.ngA - c.ngB, c.L★, c.taper_ok), ","))
        end
    end

    ncols = min(4, length(cases))
    nrows = cld(length(cases), ncols)
    fig = Figure(size=(320ncols, 220nrows + 40))
    Label(fig[0, 1:ncols], geo.tag * " — dichroic filter λC sweep (dense transmission spectra)"; fontsize=16, tellwidth=false)
    for (i, c) in enumerate(cases)
        row, col = fld(i - 1, ncols) + 1, mod(i - 1, ncols) + 1
        ax = Axis(fig[row, col], xlabel="wavelength (µm)", ylabel="transmission",
            title=@sprintf("λC=%.0f nm", 1000c.λC_target))
        xlims!(ax, xlims...)
        ylims!(ax, -0.02, 1.02)
        lines!(ax, c.spec.λdense, c.spec.T_short, color=:dodgerblue, linewidth=1.5)
        lines!(ax, c.spec.λdense, c.spec.T_long, color=:crimson, linewidth=1.5)
        vlines!(ax, [c.λC_target], color=:gray, linestyle=:dash)
    end
    save(joinpath(OUTDIR, geo.tag * "_summary_grid.png"), fig)
    println("saved: ", geo.tag, "_summary_grid.png, ", geo.tag, "_sweep_summary.csv, and ",
            length(cases), " per-case *_report.png/*_trace.csv → ", OUTDIR)
    cases
end
