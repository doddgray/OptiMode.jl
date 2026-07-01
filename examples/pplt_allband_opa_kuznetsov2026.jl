# Reproduction of the dispersion + cascaded-χ² / QPM design of
#
#   N. Kuznetsov, Z. Li, T. J. Kippenberg, "All-band photonic integrated optical parametric
#   amplification," arXiv:2605.22704 (2026).
#
# Device (Fig. 2): x-cut thin-film periodically-poled lithium tantalate (PPLT) — a ~690 nm
# LiTaO₃ film with a retained ~100 nm slab, top width ~1.8 µm, poling period Λ ≈ 5305 nm — that
# realizes a broadband cascaded-χ² (phase-mismatched SHG ⇒ effective Kerr) parametric amplifier
# pumped in the telecom band. d₃₃ ≈ 10.7 pm/V; the LiTaO₃ index follows Moutzouris et al.
#
# Using OptiMode's mode solver + χ² tools this script reproduces:
#   (1) the modal dispersion  β₂(λ), n_eff, n_g of the quasi-TE₀₀ telecom mode;
#   (2) the χ² QPM phase-matching spectrum Δk(λ) and the first-order poling period Λ
#       (target ≈ 5.305 µm) — the SHG/DFG phase-matching that seeds the cascade;
#   (3) the nonlinear coupling: SHG normalized efficiency η₀ and the cascaded-χ² effective Kerr
#       n₂,eff(Δk) (sign-changing across phase matching — the amplifier's working nonlinearity);
#   (4) the FH (1550 nm) and SH (775 nm) quasi-TE₀₀ mode profiles.
#
# Settings (see examples/README.md): --n-freqs (dispersion-sweep points, default 5), --n-dense
# (cascaded-n₂ curve resolution, default 401), --resolution-scale / --domain-scale (grid).
#
# Run:  julia --project=. examples/pplt_allband_opa_kuznetsov2026.jl   (needs CairoMakie)
#       julia --project=. examples/pplt_allband_opa_kuznetsov2026.jl --n-freqs=9 --resolution-scale=1.5

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode: rotate
using OptiMode.MaterialDispersion: LiTaO₃
using OptiMode.ModePerturbations: shg_normalized_efficiency, shg_effective_area, cascaded_chi2_n2_eff
using CairoMakie

cfg = example_settings(n_freqs=5, n_dense=401)
solver = KrylovKitEigsolve()
λF = 1.55; λS = λF/2                          # 1550 nm FH → 775 nm SH (telecom pump)
w, etch, film = 1.80, 0.59, 0.69            # 690 nm LiTaO₃, ~100 nm slab, 1.8 µm top width

# x-cut LiTaO₃: rotate the c-axis in-plane along x (extraordinary d₃₃ seen by the quasi-TE mode)
const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
LiTaO₃_xcut = rotate(LiTaO₃, _RY; name=:LiTaO₃_xcut)
mv = matvals_builder([LiTaO₃_xcut, SiO₂]; air=true)      # columns: LiTaO₃, SiO₂, air
shapes, minds = ridge_on_slab(w, etch, film)
slab = film - etch
grid = mk_grid(cfg, 8.0, 5.0, 116, 76)
D33 = 10.7; deff = (2/π) * D33 * 1e-12       # first-order-QPM d_eff (m/V), LiTaO₃ d₃₃≈10.7 pm/V

# --- (1) telecom-band modal dispersion --------------------------------------------------
println("== PPLT x-cut ridge: telecom (1550 nm) band dispersion ==")
λFH = range(1.48, 1.62; length=cfg.n_freqs)
swF = sweep_dispersion(shapes, minds, mv, λFH, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=6)
β2F, β4F = dispersion_betas(swF, λF)
@printf("β₂(1550) = %+.1f fs²/mm,  β₄(1550) = %+.0f fs⁴/mm\n", β2F, β4F)

# --- SH-band dispersion (for the QPM tuning curve) --------------------------------------
println("== PPLT x-cut ridge: second-harmonic (775 nm) band dispersion ==")
swS = sweep_dispersion(shapes, minds, mv, λFH ./ 2, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=9)

# --- design-point FH & SH modes: Λ, η₀, A_eff -------------------------------------------
mF = solve_fundamental(shapes, minds, mv, 1/λF, grid, solver; nev=6, pol=:TE, w=w, h=film, yc=slab)
mS = solve_fundamental(shapes, minds, mv, 1/λS, grid, solver; nev=9, pol=:TE, w=w, h=film, yc=slab)
nF, nS = mF.neff, mS.neff
Λ = poling_period(nF, nS, λF)
Aeff = shg_effective_area(mF.E, mS.E, grid)
η0 = shg_normalized_efficiency(mF.E, mS.E, grid; deff=deff, λ1=λF*1e-6, n1=nF, n2=nS)
@printf("n_FF=%.4f  n_SH=%.4f  Λ=%.3f µm (paper ≈ 5.305)  A_eff=%.3f µm²  η₀=%.0f %%/W/cm²\n",
        nF, nS, Λ, Aeff, η0)

# --- (2) χ² QPM phase-matching spectrum -------------------------------------------------
qpm = qpm_mismatch_spectrum(swF.neff, swS.neff, swF.λ, Λ)

# --- (3) cascaded-χ² effective Kerr n₂,eff(Δk) ------------------------------------------
Δk_range = range(-4000, 4000; length=cfg.n_dense)     # phase mismatch (rad/m)
n2c = [cascaded_chi2_n2_eff(; deff=deff, λ1=λF*1e-6, n1=nF, n2=nS, Δk=Δk) for Δk in Δk_range]

# --- plots ------------------------------------------------------------------------------
xc, yc = grid_coords(grid)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="GVD β₂ (fs²/mm)",
    title="PPLT x-cut — telecom-band dispersion")
lines!(ax1, swF.λ, swF.β2, color=:darkorange3, linewidth=2)
hlines!(ax1, [0.0], color=:gray, linestyle=:dash); vlines!(ax1, [λF], color=:gray, linestyle=:dot)
ax1b = Axis(fig1[1, 2], xlabel="FH wavelength (µm)", ylabel="Δk (rad/mm)",
    title=@sprintf("χ² QPM phase mismatch (Λ=%.2f µm)", Λ))
lines!(ax1b, qpm.λ, qpm.Δk, color=:teal, linewidth=2)
hlines!(ax1b, [0.0], color=:red, linestyle=:dash)
save(joinpath(OUTDIR, "pplt_dispersion_qpm.png"), fig1)

fig2 = Figure(size=(560, 380))
ax2 = Axis(fig2[1, 1], xlabel="phase mismatch Δk (rad/m)", ylabel="cascaded n₂,eff (m²/W)",
    title=@sprintf("Cascaded-χ² effective Kerr (d_eff=%.1f pm/V)", 2/π*D33))
lines!(ax2, collect(Δk_range), n2c, color=:purple, linewidth=2)
hlines!(ax2, [0.0], color=:gray, linestyle=:dash); vlines!(ax2, [0.0], color=:gray, linestyle=:dash)
save(joinpath(OUTDIR, "pplt_cascaded_n2.png"), fig2)

fig3 = Figure(size=(900, 340))
for (col, (E, ttl)) in enumerate(((mF.E, "TE₀₀ FH (1550 nm)"), (mS.E, "TE₀₀ SH (775 nm)")))
    ax = Axis(fig3[1, col], xlabel="x (µm)", ylabel="y (µm)", title=ttl, aspect=DataAspect())
    hm = heatmap!(ax, xc, yc, absE_norm(E), colormap=:turbo)
    lines!(ax, [-w/2, w/2, w/2, -w/2, -w/2], [slab+etch, slab+etch, slab, slab, slab+etch], color=:white, linewidth=1)
    lines!(ax, [xc[1], xc[end]], [slab, slab], color=:white, linewidth=0.6)
    lines!(ax, [xc[1], xc[end]], [0, 0], color=:white, linestyle=:dash, linewidth=0.6)
    xlims!(ax, -3.0, 3.0); ylims!(ax, -1.0, 1.6)
    Colorbar(fig3[1, col == 1 ? 0 : 3], hm, label="|E| (norm.)")
end
save(joinpath(OUTDIR, "pplt_mode_profiles.png"), fig3)

println("saved: pplt_dispersion_qpm.png, pplt_cascaded_n2.png, pplt_mode_profiles.png → ", OUTDIR)
