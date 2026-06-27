# χ⁽³⁾ cross-phase modulation (XPM): the index shift a "probe" mode sees from a co-
# propagating "pump" mode, by first-order perturbation theory.
#
# For two co-propagating waves the XPM index shift is *twice* the SPM shift at equal
# intensity (Δn_XPM = 2 n₂ I_pump vs Δn_SPM = n₂ I_self) — the χ⁽³⁾ degeneracy factor of 2
# (Agrawal, Nonlinear Fiber Optics). Here a Si₃N₄ waveguide carries a strong pump in its
# TE₀₀ mode; we compute the XPM Δneff the same TE₀₀ mode would see as a weak probe and
# confirm it is exactly 2× the SPM shift of the pump at the same power. Both curves are
# AD-differentiable in the pump power.
#
# Run:  julia --project=. examples/perturbation_xpm.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations: kerr_spm_Δneff, kerr_xpm_Δneff, effective_area_kerr, kerr_gamma
using LinearAlgebra, Printf
using CairoMakie

λ = 1.55; ω = 1 / λ
grid = Grid(4.0, 3.0, 96, 72)
solver = KrylovKitEigsolve()
mats = [Si₃N₄, SiO₂]
fε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = fε([ω])
core = MaterialShape(Cuboid([0.0, 0.0], [1.6, 0.8], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)
sm = smooth_ε(shapes, mat_vals, minds, grid)
εi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ωε = copy(selectdim(sm, 3, 2))
n2map = smooth_scalar(shapes, [kerr_n2(m, λ) for m in mats], minds, grid)
km, ev = solve_k(ω, copy(εi), grid, solver; nev=1, k_tol=1e-10)
k0, ev0 = km[1], ev[1]
Aeff = effective_area_kerr(k0, ev0, εi, grid)
@printf("Si₃N₄ WG: neff=%.4f  Aeff=%.3f μm²  γ=%.3f W⁻¹m⁻¹\n", k0/ω, Aeff, kerr_gamma(kerr_n2(Si₃N₄,λ), Aeff, λ))

powers = 0.0:0.5:10.0
# SPM: shift the pump sees from its own power
Δneff_spm = [kerr_spm_Δneff(k0, ev0, ω, εi, ∂ωε, n2map, grid, P) for P in powers]
# XPM: shift a co-propagating (here degenerate TE₀₀) probe sees from the pump's power
Δneff_xpm = [kerr_xpm_Δneff(k0, ev0, ω, εi, ∂ωε, n2map, grid, k0, ev0, εi, P) for P in powers]
ratio = [P > 0 ? x / s : NaN for (x, s, P) in zip(Δneff_xpm, Δneff_spm, powers)]
@printf("XPM/SPM ratio (P=10 W) = %.4f  (expected 2.000)\n", Δneff_xpm[end] / Δneff_spm[end])

fig = Figure(size=(760, 330))
ax1 = Axis(fig[1, 1], xlabel="pump power P (W)", ylabel="Δneff",
    title="XPM is 2× SPM (Si₃N₄, 1.55 μm)")
scatterlines!(ax1, collect(powers), Δneff_xpm, color=:crimson, label="XPM (2 n₂ I_pump)")
scatterlines!(ax1, collect(powers), Δneff_spm, color=:navy, label="SPM (n₂ I_self)")
axislegend(ax1, position=:lt)
ax2 = Axis(fig[1, 2], xlabel="pump power P (W)", ylabel="Δneff_XPM / Δneff_SPM",
    title="degeneracy factor")
scatterlines!(ax2, collect(powers)[2:end], ratio[2:end], color=:purple)
hlines!(ax2, [2.0], color=:grey, linestyle=:dash)
ylims!(ax2, 1.8, 2.2)
out = joinpath(@__DIR__, "perturbation_output", "xpm_Si3N4.png")
save(out, fig)
println("saved ", out)
