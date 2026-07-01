# Reproduction of the tantala (Ta₂O₅) group-velocity-dispersion engineering of
#
#   J. A. Black, R. Streater, K. F. Lamee, D. R. Carlson, S.-P. Yu, S. B. Papp,
#   "Group-velocity dispersion engineering of tantala integrated photonics,"
#   Optics Letters 46, 817 (2021).  [arXiv:2009.14190]
#
# Device (Fig. 1b): a tantala core on a SiO₂ substrate with air top/side cladding; the survey
# uses thickness t with width w = 1.25·t. Here we take the t = 1 µm, w = 1.25 µm cross-section
# (the mode-profile inset of Fig. 1b) and reproduce, with OptiMode's mode solver + Kerr tools:
#   (1) the modal dispersion:  GVD β₂(λ) (→ the paper's D-style GVD survey), n_eff, n_g;
#   (2) the Kerr nonlinear coupling γ = ω n₂/(c A_eff) with tantala n₂ = 6.2×10⁻¹⁹ m²/W;
#   (3) the degenerate-FWM parametric-gain spectrum (tantala is a χ³ platform); and
#   (4) the fundamental quasi-TE₀₀ mode profile (Fig. 1b inset).
#
# Run:  julia --project=. examples/tantala_gvd_black2021.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode.MaterialDispersion: kerr_n2
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using CairoMakie

solver = KrylovKitEigsolve()
t, w = 1.0, 1.25                          # thickness 1 µm, width 1.25·t (Black 2021, Fig. 1b)
λp = 1.55
mats = [Ta₂O₅, SiO₂]
mv = matvals_builder(mats; air=true)     # columns: Ta₂O₅, SiO₂, air

# tantala core sitting on a SiO₂ substrate (core bottom at y=0), air top + sides
shapes = ( MaterialShape(Cuboid([0.0, t/2], [w, t], [1.0 0.0; 0.0 1.0]), 1),          # core (Ta₂O₅)
           MaterialShape(Cuboid([0.0, -1.5], [100.0, 3.0], [1.0 0.0; 0.0 1.0]), 2) )  # SiO₂ substrate
minds = (1, 2, 3)                        # core→Ta₂O₅, substrate→SiO₂, background→air
grid = Grid(5.0, 4.0, 128, 100)
yc = t/2                                  # core center height for confinement mask

# --- (1) modal dispersion spectra -------------------------------------------------------
println("== Ta₂O₅ 1.0×1.25 µm on SiO₂ (air clad): dispersion sweep ==")
λs = range(0.9, 2.3; length=13)
sw = sweep_dispersion(shapes, minds, mv, λs, grid, solver; pol=:TE, w=w, h=t, yc=yc, nev=4)
# zero-dispersion wavelengths (β₂ sign changes)
zdw = Float64[]
for i in 1:length(sw.λ)-1
    if sign(sw.β2[i]) != sign(sw.β2[i+1])
        push!(zdw, sw.λ[i] - sw.β2[i]*(sw.λ[i+1]-sw.λ[i])/(sw.β2[i+1]-sw.β2[i]))
    end
end
println("zero-dispersion wavelength(s): ", isempty(zdw) ? "none in range" : join(round.(zdw; digits=3), ", "), " µm")

# --- (2) Kerr nonlinear coupling at 1.55 µm --------------------------------------------
mp = solve_fundamental(shapes, minds, mv, 1/λp, grid, solver; nev=4, pol=:TE, w=w, h=t, yc=yc)
Aeff = effective_area_kerr(mp.k, mp.ev, mp.εi, grid)
n2 = kerr_n2(Ta₂O₅, λp)
γ = kerr_gamma(n2, Aeff, λp)
@printf("A_eff = %.3f µm²   n₂ = %.2e µm²/W   γ = %.3f W⁻¹m⁻¹\n", Aeff, n2, γ)

# --- (3) degenerate-FWM parametric-gain spectrum ---------------------------------------
P, L = 5.0, 0.1                           # 5 W, 10 cm (tantala nonlinear waveguide)
fw = fwm_gain_spectrum(sw, λp, P, γ; L=L, Ω_THz=range(-30, 30; length=401))
@printf("fit β₂ = %+.1f fs²/mm, β₄ = %+.0f fs⁴/mm;  peak gain %.1f dB,  3-dB BW %.1f THz\n",
        fw.β2*1e27, fw.β4*1e57, maximum(fw.gain_dB), fw.bw3_THz)

# --- plots ------------------------------------------------------------------------------
xc, ycoord = grid_coords(grid)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="GVD β₂ (fs²/mm)",
    title="Ta₂O₅ 1.0×1.25 µm — GVD (Black 2021, Fig. 1b)")
lines!(ax1, sw.λ, sw.β2, color=:goldenrod3, linewidth=2)
hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
for z in zdw; vlines!(ax1, [z], color=:red, linestyle=:dot); end
ax1b = Axis(fig1[1, 2], xlabel="wavelength (µm)", ylabel="index", title="effective / group index")
lines!(ax1b, sw.λ, sw.neff, color=:seagreen, linewidth=2, label="n_eff")
lines!(ax1b, sw.λ, sw.ng, color=:darkorange, linewidth=2, label="n_g")
axislegend(ax1b, position=:rc)
save(joinpath(OUTDIR, "tantala_dispersion.png"), fig1)

fig2 = Figure(size=(560, 380))
ax2 = Axis(fig2[1, 1], xlabel="signal wavelength (µm)", ylabel="parametric gain (dB)",
    title=@sprintf("Ta₂O₅ FWM gain (γ=%.2f /W/m, P=%.0f W, L=%.0f cm)", γ, P, 100L))
lines!(ax2, fw.λs_signal, fw.gain_dB, color=:purple, linewidth=2)
vlines!(ax2, [λp], color=:gray, linestyle=:dash)
save(joinpath(OUTDIR, "tantala_fwm_gain.png"), fig2)

fig3 = Figure(size=(520, 360))
ax3 = Axis(fig3[1, 1], xlabel="x (µm)", ylabel="y (µm)", aspect=DataAspect(),
    title="quasi-TE₀₀ mode |E| (1550 nm)")
hm = heatmap!(ax3, xc, ycoord, absE_norm(mp.E), colormap=:turbo)
lines!(ax3, [-w/2, w/2, w/2, -w/2, -w/2], [0.0, 0.0, t, t, 0.0], color=:white, linewidth=1)
lines!(ax3, [xc[1], xc[end]], [0.0, 0.0], color=:white, linestyle=:dash, linewidth=0.7)  # substrate top
xlims!(ax3, -2.2, 2.2); ylims!(ax3, -1.2, 2.0)
Colorbar(fig3[1, 2], hm, label="|E| (norm.)")
save(joinpath(OUTDIR, "tantala_mode_profile.png"), fig3)

println("saved: tantala_dispersion.png, tantala_fwm_gain.png, tantala_mode_profile.png → ", OUTDIR)
