# Reproduction of the dispersion + Kerr four-wave-mixing parametric-gain design of
#
#   J. Riemensberger, N. Kuznetsov, J. Liu, J. He, R. N. Wang, T. J. Kippenberg,
#   "A photonic integrated continuous-travelling-wave parametric amplifier,"
#   Nature 612, 56 (2022).  https://doi.org/10.1038/s41586-022-05329-1
#
# Device (Fig. 1b): a dispersion-engineered, SiO₂-clad, stoichiometric Si₃N₄ waveguide of
# cross-section 910 nm (height) × 2450 nm (width), pumped in the C band (1550 nm). The paper
# reports β₂ = −124 fs²/mm, β₄ = 50 fs⁴/mm and a ~1 THz single-pump amplification bandwidth.
#
# Using only OptiMode's mode solver + Kerr tools this script reproduces:
#   (1) the modal dispersion spectra  n_eff(λ), n_g(λ), GVD β₂(λ);
#   (2) the Kerr nonlinear coupling coefficient  γ = ω n₂/(c A_eff)  and effective area A_eff;
#   (3) the degenerate-FWM parametric-gain spectrum g(Ω) vs signal wavelength (Eq. 2); and
#   (4) the fundamental quasi-TE₀₀ mode profile (Fig. 1d inset).
#
# Settings (see examples/README.md): --n-freqs (dispersion-sweep points, default 13), --n-dense
# (FWM gain-spectrum resolution, default 321), --resolution-scale / --domain-scale (grid).
#
# Run:  julia --project=. examples/si3n4_cw_opa_riemensberger2022.jl
# (needs CairoMakie in the active environment; grid is moderate so it runs in a few minutes —
#  the converged design sweep deploys via ModeSweeps/SLURM.)
#       julia --project=. examples/si3n4_cw_opa_riemensberger2022.jl --n-freqs=25 --resolution-scale=1.5

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode.MaterialDispersion: kerr_n2
using CairoMakie

cfg = example_settings(n_freqs=13, n_dense=321)
solver = KrylovKitEigsolve()
w, h = 2.45, 0.91                       # 2450 nm width × 910 nm height (Riemensberger 2022)
λp = 1.55                                # C-band pump
mats = [Si₃N₄, SiO₂]
mv = matvals_builder(mats; air=false)   # Si₃N₄ core fully buried in SiO₂
shapes, minds = buried_core(w, h)
grid = mk_grid(cfg, 6.0, 4.0, 128, 96)

# --- (1) modal dispersion spectra -------------------------------------------------------
println("== Si₃N₄ 910×2450 nm: modal dispersion sweep ==")
λs = range(1.40, 1.70; length=cfg.n_freqs)
sw = sweep_dispersion(shapes, minds, mv, λs, grid, solver; pol=:TE, w=w, h=h, nev=4)
β2_p = sw.β2[argmin(abs.(sw.λ .- λp))]
@printf("β₂(1.55 µm) ≈ %.0f fs²/mm   (paper: −124 fs²/mm)\n", β2_p)

# --- (2) Kerr nonlinear coupling coefficient at the pump --------------------------------
mp = solve_fundamental(shapes, minds, mv, 1/λp, grid, solver; nev=4, pol=:TE, w=w, h=h)
Aeff = effective_area_kerr(mp.k, mp.ev, mp.εi, grid)
n2 = kerr_n2(Si₃N₄, λp)                  # µm²/W
γ = kerr_gamma(n2, Aeff, λp)             # W⁻¹m⁻¹
@printf("A_eff = %.3f µm²   n₂ = %.2e µm²/W   γ = %.3f W⁻¹m⁻¹\n", Aeff, n2, γ)

# --- (3) degenerate-FWM parametric-gain spectrum ---------------------------------------
P, L = 2.0, 1.0                          # 2 W on-chip pump, 1 m waveguide (metre-scale spiral)
fw = fwm_gain_spectrum(sw, λp, P, γ; L=L, Ω_THz=range(-8, 8; length=cfg.n_dense))
β2_fit = fw.β2 * 1e27      # s²/m → fs²/mm
β4_fit = fw.β4 * 1e57      # s⁴/m → fs⁴/mm
@printf("fit β₂ = %+.1f fs²/mm, β₄ = %+.0f fs⁴/mm;  peak gain %.1f dB,  3-dB BW %.2f THz\n",
        β2_fit, β4_fit, maximum(fw.gain_dB), fw.bw3_THz)

# --- plots ------------------------------------------------------------------------------
xc, yc = grid_coords(grid)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="GVD β₂ (fs²/mm)",
    title="Si₃N₄ 910×2450 nm — modal dispersion")
lines!(ax1, sw.λ, sw.β2, color=:dodgerblue, linewidth=2)
hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
scatter!(ax1, [λp], [β2_p], color=:red, markersize=10)
ax1b = Axis(fig1[1, 2], xlabel="wavelength (µm)", ylabel="index",
    title="effective / group index")
lines!(ax1b, sw.λ, sw.neff, color=:seagreen, linewidth=2, label="n_eff")
lines!(ax1b, sw.λ, sw.ng, color=:darkorange, linewidth=2, label="n_g")
axislegend(ax1b, position=:rc)
save(joinpath(OUTDIR, "si3n4_dispersion.png"), fig1)

fig2 = Figure(size=(560, 380))
ax2 = Axis(fig2[1, 1], xlabel="signal wavelength (µm)", ylabel="parametric gain (dB)",
    title=@sprintf("FWM parametric gain (γ=%.2f /W/m, P=%.0f W, L=%.0f m)", γ, P, L))
lines!(ax2, fw.λs_signal, fw.gain_dB, color=:purple, linewidth=2)
vlines!(ax2, [λp], color=:gray, linestyle=:dash)
save(joinpath(OUTDIR, "si3n4_fwm_gain.png"), fig2)

fig3 = Figure(size=(520, 360))
ax3 = Axis(fig3[1, 1], xlabel="x (µm)", ylabel="y (µm)", aspect=DataAspect(),
    title="quasi-TE₀₀ mode |E| (1550 nm)")
hm = heatmap!(ax3, xc, yc, absE_norm(mp.E), colormap=:turbo)
lines!(ax3, [-w/2, w/2, w/2, -w/2, -w/2], [-h/2, -h/2, h/2, h/2, -h/2], color=:white, linewidth=1)
xlims!(ax3, -2.5, 2.5); ylims!(ax3, -1.6, 1.6)
Colorbar(fig3[1, 2], hm, label="|E| (norm.)")
save(joinpath(OUTDIR, "si3n4_mode_profile.png"), fig3)

println("saved: si3n4_dispersion.png, si3n4_fwm_gain.png, si3n4_mode_profile.png → ", OUTDIR)
