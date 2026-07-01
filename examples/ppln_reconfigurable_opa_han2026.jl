# Reproduction of the dispersion + χ² quasi-phase-matched OPA design of
#
#   G. Han, W. Deng, Y. Wang, Z. Feng, W. Wang, M. Tian, Y. Liu, S. Biswas, C. A. Meriles,
#   A. Alù, Q. Guo, "On-chip electrically reconfigurable octave-bandwidth optical
#   amplification from visible to near-infrared," arXiv:2602.00246 (2026).
#
# Device (Fig. 1c,d): an x-cut thin-film lithium niobate (TFLN / PPLN) ridge engineered for
# near-zero β₂ and low β₄ at the 1064 nm fundamental (paper: β₂ = −3.6 fs²/mm, β₄ = 543 fs⁴/mm),
# with a 532 nm second harmonic. χ² (d₃₃) QPM decouples the phase mismatch from pump power and,
# with high-order dispersion engineering + electro-thermal QPM tuning, gives near-octave gain.
#
# Using OptiMode's mode solver + χ² tools this script reproduces:
#   (1) the modal dispersion  β₂(λ), β₄ near 1064 nm (Fig. 1d), n_eff, n_g;
#   (2) the χ² QPM phase-matching spectrum  Δk(λ) = β_SH − 2β_FF − 2π/Λ and poling period Λ;
#   (3) the χ² nonlinear coupling: normalized efficiency η₀ and SHG effective area A_eff;
#   (4) the FH (1064 nm) and SH (532 nm) fundamental-mode profiles (Fig. 1c).
#
# Run:  julia --project=. examples/ppln_reconfigurable_opa_han2026.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode: rotate
using OptiMode.MaterialDispersion: LiNbO₃
using OptiMode.ModePerturbations: shg_normalized_efficiency, shg_effective_area
using CairoMakie

solver = KrylovKitEigsolve()
λF = 1.064; λS = λF/2                        # 1064 nm FH → 532 nm SH
w, etch, film = 1.40, 0.35, 0.70            # x-cut TFLN ridge (representative near-zero-β₂ design)

# x-cut LiNbO₃: rotate so the extraordinary (c) axis lies in-plane along x (d₃₃ for quasi-TE)
const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
LiNbO₃_xcut = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
mv = matvals_builder([LiNbO₃_xcut, SiO₂]; air=true)     # columns: LN, SiO₂, air
shapes, minds = ridge_on_slab(w, etch, film)
slab = film - etch
grid = Grid(6.0, 4.0, 116, 78)
D33 = 20.5; deff = (2/π) * D33 * 1e-12       # first-order-QPM d_eff (m/V), LiNbO₃ d₃₃≈20.5 pm/V

# --- (1) FH-band modal dispersion -------------------------------------------------------
println("== TFLN x-cut ridge: fundamental (1064 nm band) dispersion ==")
λFH = range(1.01, 1.13; length=5)
swF = sweep_dispersion(shapes, minds, mv, λFH, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=6)
β2F, β4F = dispersion_betas(swF, λF)
@printf("β₂(1064) = %+.1f fs²/mm,  β₄(1064) = %+.0f fs⁴/mm   (paper: −3.6, 543)\n", β2F, β4F)

# --- SH-band dispersion (for the QPM tuning curve) --------------------------------------
println("== TFLN x-cut ridge: second-harmonic (532 nm band) dispersion ==")
swS = sweep_dispersion(shapes, minds, mv, λFH ./ 2, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=9)

# --- design-point FH & SH modes: Λ, η₀, A_eff -------------------------------------------
mF = solve_fundamental(shapes, minds, mv, 1/λF, grid, solver; nev=6, pol=:TE, w=w, h=film, yc=slab)
mS = solve_fundamental(shapes, minds, mv, 1/λS, grid, solver; nev=9, pol=:TE, w=w, h=film, yc=slab)
nF, nS = mF.neff, mS.neff
Λ = poling_period(nF, nS, λF)
Aeff = shg_effective_area(mF.E, mS.E, grid)
η0 = shg_normalized_efficiency(mF.E, mS.E, grid; deff=deff, λ1=λF*1e-6, n1=nF, n2=nS)
@printf("n_FF=%.4f  n_SH=%.4f  Λ=%.3f µm  A_eff=%.3f µm²  η₀=%.0f %%/W/cm²\n", nF, nS, Λ, Aeff, η0)

# --- (2) χ² QPM phase-matching spectrum -------------------------------------------------
qpm = qpm_mismatch_spectrum(swF.neff, swS.neff, swF.λ, Λ)

# --- χ² OPA gain bandwidth near degeneracy (SH-pumped) ----------------------------------
P_SH = 0.05                                   # 50 mW SH pump
Γ = sqrt((η0/100*1e4) * P_SH)                 # drive √(η₀ P_SH), 1/m
gn = chi2_opa_gain_spectrum(swF, λF, Γ; L=0.01, Ω_THz=range(-150, 150; length=601))
@printf("χ² OPA drive Γ=%.1f /m,  peak gain %.1f dB,  3-dB BW %.0f THz\n",
        Γ, maximum(gn.gain_dB), gn.bw3_THz)

# --- plots ------------------------------------------------------------------------------
xc, yc = grid_coords(grid)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="GVD β₂ (fs²/mm)",
    title="x-cut TFLN — FH-band dispersion")
lines!(ax1, swF.λ, swF.β2, color=:crimson, linewidth=2)
hlines!(ax1, [0.0], color=:gray, linestyle=:dash); vlines!(ax1, [λF], color=:gray, linestyle=:dot)
ax1b = Axis(fig1[1, 2], xlabel="FH wavelength (µm)", ylabel="Δk (rad/mm)",
    title=@sprintf("χ² QPM phase mismatch (Λ=%.2f µm)", Λ))
lines!(ax1b, qpm.λ, qpm.Δk, color=:teal, linewidth=2)
hlines!(ax1b, [0.0], color=:red, linestyle=:dash)
save(joinpath(OUTDIR, "ppln_han_dispersion_qpm.png"), fig1)

fig2 = Figure(size=(560, 380))
ax2 = Axis(fig2[1, 1], xlabel="signal wavelength (µm)", ylabel="parametric gain (dB)",
    title=@sprintf("χ² OPA gain near degeneracy (η₀=%.0f %%/W/cm², Γ=%.0f/m)", η0, Γ))
lines!(ax2, gn.λs_signal, gn.gain_dB, color=:purple, linewidth=2)
vlines!(ax2, [λF], color=:gray, linestyle=:dash)
xlims!(ax2, 0.7, 1.7)
save(joinpath(OUTDIR, "ppln_han_gain.png"), fig2)

fig3 = Figure(size=(900, 340))
for (col, (E, ttl)) in enumerate(((mF.E, "TE₀₀ FH (1064 nm)"), (mS.E, "TE₀₀ SH (532 nm)")))
    ax = Axis(fig3[1, col], xlabel="x (µm)", ylabel="y (µm)", title=ttl, aspect=DataAspect())
    hm = heatmap!(ax, xc, yc, absE_norm(E), colormap=:turbo)
    lines!(ax, [-w/2, w/2, w/2, -w/2, -w/2], [slab+etch, slab+etch, slab, slab, slab+etch], color=:white, linewidth=1)
    lines!(ax, [xc[1], xc[end]], [slab, slab], color=:white, linewidth=0.6)
    lines!(ax, [xc[1], xc[end]], [0, 0], color=:white, linestyle=:dash, linewidth=0.6)
    xlims!(ax, -2.6, 2.6); ylims!(ax, -1.0, 1.6)
    Colorbar(fig3[1, col == 1 ? 0 : 3], hm, label="|E| (norm.)")
end
save(joinpath(OUTDIR, "ppln_han_mode_profiles.png"), fig3)

println("saved: ppln_han_dispersion_qpm.png, ppln_han_gain.png, ppln_han_mode_profiles.png → ", OUTDIR)
