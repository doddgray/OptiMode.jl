# Thermo-optic tuning of a Si₃N₄ waveguide by first-order perturbation theory.
#
# A temperature change ΔT shifts each material's index by (dn/dT)·ΔT; the resulting weak,
# spatially-varying Δε = 2 n (dn/dT) ΔT perturbs the modal effective index by the
# Hellmann–Feynman first-order amount Δneff, and a microring resonance by
# Δλ/λ = Δneff/n_g. This reproduces the measured Si₃N₄-ring thermo-optic sensitivity
# dλ/dT ≈ 18–20 pm/K at 1550 nm:
#   Arbabi & Goddard, Opt. Lett. 38, 3878 (2013): dn/dT(Si₃N₄)=2.45e-5, dn/dT(SiO₂)=0.95e-5
#   Ilie et al., Sci. Rep. 12, 17815 (2022): annealed bare SiN ring 18.0 pm/K
#   Xue et al., Opt. Express 24, 687 (2016): ≈21 pm/K
#
# Run (from the repo root):  julia --project=. examples/perturbation_thermo_optic.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations
using OptiMode.ModePerturbations: thermo_optic_dneff_dT, thermo_optic_Δneff,
    resonance_shift_dλ_dT
using LinearAlgebra, Printf
using CairoMakie

λ = 1.55; ω = 1 / λ
grid = Grid(4.0, 3.0, 96, 72)
solver = KrylovKitEigsolve()
mats = [Si₃N₄, SiO₂]
fε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = hcat(fε([ω]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18)))   # +air (unused here)
core = MaterialShape(Cuboid([0.0, 0.0], [1.0, 0.4], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)
sm = smooth_ε(shapes, mat_vals, minds, grid)
εi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ωε = copy(selectdim(sm, 3, 2))
km, ev = solve_k(ω, copy(εi), grid, solver; nev=1, k_tol=1e-11)
k0, ev0 = km[1], ev[1]
ng = group_index(k0, ev0, ω, εi, ∂ωε, grid)

# per-material dn/dT maps (Arbabi & Goddard 2013), sub-pixel-smoothed onto the grid
dndT_SiN, dndT_SiO2 = 2.45e-5, 0.95e-5
dndT_map = smooth_scalar(shapes, [dndT_SiN, dndT_SiO2], minds, grid)

dneff_dT = thermo_optic_dneff_dT(k0, ev0, ω, εi, dndT_map, grid)
dλdT = resonance_shift_dλ_dT(dneff_dT, ng, λ) * 1e6     # μm/K → pm/K
@printf("Si₃N₄ WG: neff=%.4f  n_g=%.4f\n", k0/ω, ng)
@printf("dneff/dT = %.3e /K   →   dλ/dT = %.2f pm/K  (meas. 18–21 pm/K)\n", dneff_dT, dλdT)

# temperature sweep: Δλ(ΔT) is linear; slope is dλ/dT
ΔTs = 0:5:80
Δλ_pm = [thermo_optic_Δneff(k0, ev0, ω, εi, dndT_map, ΔT, grid) / ng * λ * 1e6 for ΔT in ΔTs]

fig = Figure(size=(760, 320))
ax1 = Axis(fig[1, 1], xlabel="ΔT (K)", ylabel="resonance shift Δλ (pm)",
    title=@sprintf("Si₃N₄ ring: dλ/dT = %.1f pm/K (meas. 18–21)", dλdT))
scatterlines!(ax1, collect(ΔTs), Δλ_pm, color=:firebrick)

# material comparison: dλ/dT for Si₃N₄ vs SiO₂-dominant vs (literature) Si
ax2 = Axis(fig[1, 2], ylabel="dλ/dT (pm/K)", title="platform comparison (1550 nm)",
    xticks=(1:3, ["Si₃N₄\n(this calc)", "Si₃N₄\n(meas)", "SOI Si\n(meas)"]))
barplot!(ax2, [1, 2, 3], [dλdT, 18.0, 77.0],
    color=[:firebrick, :grey70, :grey70])
out = joinpath(@__DIR__, "perturbation_output", "thermo_optic_Si3N4.png")
save(out, fig)
println("saved ", out)
