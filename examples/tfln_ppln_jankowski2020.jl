# Reproduction of the dispersion-engineered nanophotonic PPLN waveguide of
#   M. Jankowski, C. Langrock, B. Desiatov, A. Marandi, C. Wang, M. Zhang, C. R. Phillips,
#   M. Lončar, M. M. Fejer, "Ultrabroadband nonlinear optics in nanophotonic periodically
#   poled lithium niobate waveguides," Optica 7, 40 (2020).
#   https://doi.org/10.1364/OPTICA.7.000040
#
# Design device (Fig. 1): x-cut MgO:LiNbO₃, 700-nm film, top width 1850 nm, etch depth
# 340 nm, SHG of ~2050 nm → ~1025 nm. This script reproduces:
#   • Fig. 1(a): the quasi-TE₀₀ fundamental and second-harmonic modal field profiles
#   • the design-point χ² figures of merit: poling period Λ, normalized efficiency η₀,
#     effective area A_eff, group-velocity mismatch Δk′, and GVD k″_ω
#   • Fig. 3(d): the broadband SHG phase-matching transfer function sinc²(Δk(Ω)L/2)
#
# NOTE on convergence: the 2-µm fundamental is weakly guided and its effective index (and
# the delicate engineered Δk′ ≈ 0) converge slowly with the cell size — e.g. n_FF rises
# 1.68 → 1.86 and Λ rises 2.7 → 4.0 µm going from a 6-µm to a 9-µm-wide cell, trending
# toward the paper's Λ = 5.11 µm. Reproducing the fs/mm-level engineered dispersion to
# paper precision requires a large, fully converged grid — deploy the geometry sweep as a
# ModeSweeps/SLURM batch (examples/tfln_ppln_geometry_sweep_setup.jl). Here we use a
# moderate cell and report the trend honestly.
#
# Run:  julia --project=. examples/tfln_ppln_jankowski2020.jl

include(joinpath(@__DIR__, "tfln_ppln_jankowski2020_common.jl"))
using OptiMode.ModePerturbations: shg_normalized_efficiency, shg_effective_area
using OptiMode.DielectricSmoothing: δx, δy
using Printf
using CairoMakie

solver = KrylovKitEigsolve()
w, etch = 1.85, 0.34                       # design: 1850 nm top width, 340 nm etch
λF = 2.05; ωF = 1 / λF; ωSH = 2ωF          # fundamental 2050 nm → SH 1025 nm
L_mm = 6.0                                  # 6-mm device

# --- modal fields (Fig. 1a) on a common moderate cell ----------------------------------
# One cell for both harmonics so the χ² mode-overlap integral (η₀, A_eff) is well defined.
# n_o > n_e in LiNbO₃, so the higher-index quasi-TM modes sit above TE₀₀; we use enough
# bands that the (Eₓ-dominant) quasi-TE₀₀ is captured at each harmonic. A modest cell keeps
# the second-harmonic band count tractable (the converged 2-µm dispersion is the cluster
# batch's job — see header note and tfln_ppln_geometry_sweep_deploy.jl).
grid = Grid(7.0, 4.5, 176, 112)
mF = solve_te00(w, etch, ωF, grid, solver; nev=6)
mS = solve_te00(w, etch, ωSH, grid, solver; nev=6)
nF, nS = mF.k / ωF, mS.k / ωSH
@printf("ridge quasi-TE₀₀:  n_FF = %.4f (TE %.3f, conf %.2f),  n_SH = %.4f (TE %.3f, conf %.2f)\n",
    nF, mF.te_frac, mF.conf, nS, mS.te_frac, mS.conf)

# |E| maps, normalized to peak (Fig. 1a style)
Nx, Ny = size(grid)
xc = collect((-grid.Δx / 2) .+ (0.5:Nx) .* δx(grid))
yc = collect((-grid.Δy / 2) .+ (0.5:Ny) .* δy(grid))
absE(E) = (m = sqrt.(dropdims(sum(abs2, E; dims=1); dims=1)); m ./ maximum(m))
fig1 = Figure(size=(820, 320))
for (col, (E, ttl)) in enumerate(((mF.E, "TE₀₀ fundamental (2050 nm)"), (mS.E, "TE₀₀ second harmonic (1025 nm)")))
    ax = Axis(fig1[1, col], xlabel="x (μm)", ylabel="y (μm)", title=ttl, aspect=DataAspect())
    hm = heatmap!(ax, xc, yc, absE(E), colormap=:turbo)
    # waveguide outline: ridge (top width w, height etch) on slab (FILM-etch), SiO₂ below
    slab = FILM_NM / 1e3 - etch
    lines!(ax, [-w/2, w/2, w/2, -w/2, -w/2], [slab+etch, slab+etch, slab, slab, slab+etch], color=:white, linewidth=1)
    lines!(ax, [xc[1], xc[end]], [slab, slab], color=:white, linewidth=0.7)
    lines!(ax, [xc[1], xc[end]], [0, 0], color=:white, linestyle=:dash, linewidth=0.7)
    xlims!(ax, -3.5, 3.5); ylims!(ax, -1.2, 1.6)
    Colorbar(fig1[1, col == 1 ? 0 : 3], hm, label="|E| (norm.)")
end
out1 = joinpath(@__DIR__, "perturbation_output", "jankowski_modal_fields.png")
save(out1, fig1)
println("saved ", out1)

# --- design-point χ² figures of merit ---------------------------------------------------
Λ = poling_period(nF, nS, λF)
ngF = group_index(mF.k, mF.ev, ωF, mF.εi, mF.∂ωε, grid)
ngS = group_index(mS.k, mS.ev, ωSH, mS.εi, mS.∂ωε, grid)
_, gvdF = ng_gvd(ωF, mF.k, mF.ev, mF.εi, mF.∂ωε, mF.∂²ωε, grid)
_, gvdS = ng_gvd(ωSH, mS.k, mS.ev, mS.εi, mS.∂ωε, mS.∂²ωε, grid)
Δkp = gvm_fs_per_mm(ngF, ngS)
k2F = gvd_fs2_per_mm(gvdF)
chi2mask = smooth_scalar(ppln_shapes(w, etch), [1.0, 0.0, 0.0], PPLN_MINDS, grid)
Aeff = shg_effective_area(mF.E, mS.E, grid)
η0 = shg_eta0_eq1(nF, nS, λF, Aeff)
@printf("""
design-point figures of merit (this cell)        paper (Jankowski 2020)
  poling period Λ      = %6.3f μm                 5.11 μm
  normalized eff. η₀   = %6.0f %%/W·cm²            1100 %%/W·cm²
  effective area A_eff = %6.3f μm²                1.6 μm²
  group-vel. mismatch Δk′ = %+7.1f fs/mm          5 fs/mm
  GVD k″_ω             = %+7.1f fs²/mm           -15 fs²/mm
(Λ, Δk′, k″ are converging toward the paper as the cell grows; see header note.)
""", Λ, η0, Aeff, Δkp, k2F)

# --- SHG transfer function (Fig. 3d): sinc²(Δk(Ω)L/2) -----------------------------------
# 2nd-order Taylor of Δk(Ω) = β_SH(2ω+2Ω) − 2β_FF(ω+Ω) − 2π/Λ about phase matching:
#   Δk(Ω) = 2Δk′·Ω + (2β″_SH − β″_FF)·Ω²,   β′ = n_g/c,  β″ = GVD.
# Compare the engineered waveguide (small Δk′) with a conventional GVM-dominated device.
βpp_F = gvdF / (2π * C_UM_FS^2)            # FF GVD β″ in fs²/µm (OptiMode units → physical)
βpp_S = gvdS / (2π * C_UM_FS^2)
Δkp_um(fs_per_mm) = fs_per_mm * 1e-3        # fs/mm → fs/µm  (= Δβ′ in fs/µm)
L_um = L_mm * 1e3
ω_a(λ) = 2π * 299792458.0 * 1e-9 / λ        # angular freq (rad/fs) at vacuum λ (µm)
Ωof(λ) = ω_a(λ) - ω_a(λF)                   # detuning (rad/fs)
λs = range(1.90, 2.25; length=400)
Δk_of(Ω, Δkp_fsmm) = 2 * Δkp_um(Δkp_fsmm) * Ω + (2 * βpp_S - βpp_F) * Ω^2   # fs/µm × rad/fs = 1/µm
T(Ω, Δkp_fsmm) = (x = Δk_of(Ω, Δkp_fsmm) * L_um / 2; iszero(x) ? 1.0 : (sin(x) / x)^2)
T_eng  = [T(Ωof(λ), 5.0)   for λ in λs]     # engineered Δk′ = 5 fs/mm (paper)
T_conv = [T(Ωof(λ), 100.0) for λ in λs]     # conventional bulk-like Δk′ ≈ 100 fs/mm (20× larger)
T_wg   = [T(Ωof(λ), Δkp)   for λ in λs]     # this waveguide (container-budget Δk′)

bw3(T) = (m = T .≥ 0.5; (count(m) > 0) ? (λs[findlast(m)] - λs[findfirst(m)]) * 1e3 : 0.0)
@printf("3-dB SHG bandwidth: engineered(Δk′=5) %.0f nm, conventional(Δk′=100) %.0f nm  (paper >110 nm)\n",
    bw3(T_eng), bw3(T_conv))

fig3 = Figure(size=(620, 380))
ax = Axis(fig3[1, 1], xlabel="fundamental wavelength (nm)", ylabel="SHG transfer function",
    title="Jankowski 2020 Fig. 3(d): dispersion-engineered SHG (L = 6 mm)")
lines!(ax, λs .* 1e3, T_eng, color=:dodgerblue, linewidth=2, label="engineered Δk′=5 fs/mm (theory)")
lines!(ax, λs .* 1e3, T_conv, color=:orange, linewidth=2, label="conventional Δk′≈100 fs/mm")
lines!(ax, λs .* 1e3, T_wg, color=:seagreen, linestyle=:dash, label=@sprintf("this cell Δk′=%.0f fs/mm", Δkp))
axislegend(ax, position=:rt)
xlims!(ax, 1950, 2200)
out3 = joinpath(@__DIR__, "perturbation_output", "jankowski_shg_transfer_function.png")
save(out3, fig3)
println("saved ", out3)
