# χ⁽³⁾ two-photon absorption (TPA): the intensity-dependent modal power loss of a silicon
# waveguide, by first-order perturbation theory.
#
# TPA contributes an intensity-dependent absorption α_TPA = β_TPA·I, modeled here as an
# imaginary index Δn″ = β_TPA I λ/(4π) fed to the complex-Δε loss engine; the returned
# modal loss is the field-weighted average over the cross-section. For a 450×220 nm SOI
# wire at 1550 nm with β_TPA ≈ 0.8 cm/GW (Bristow, APL 90, 191104 (2007); Lin/Painter/
# Agrawal, Opt. Express 15, 16604 (2007)) the loss grows linearly with power and reaches
# the dB/cm scale that limits silicon nonlinear photonics. AD-differentiable in the power.
#
# Run:  julia --project=. examples/perturbation_tpa_loss.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations: tpa_modal_loss, cm_per_GW_to_μm_per_W, effective_area_kerr,
    mode_intensity
using LinearAlgebra, Printf
using CairoMakie

λ = 1.55; ω = 1 / λ
grid = Grid(2.5, 2.0, 120, 96)
solver = KrylovKitEigsolve()
mats = [silicon, SiO₂]
fε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = fε([ω])
core = MaterialShape(Cuboid([0.0, 0.0], [0.45, 0.22], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)
sm = smooth_ε(shapes, mat_vals, minds, grid)
εi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ωε = copy(selectdim(sm, 3, 2))
km, ev = solve_k(ω, copy(εi), grid, solver; nev=2, k_tol=1e-10)
k0, ev0 = km[1], ev[1]
Aeff = effective_area_kerr(k0, ev0, εi, grid)
@printf("Si wire 450×220 nm: neff=%.4f  Aeff=%.4f μm²\n", k0/ω, Aeff)

β_TPA = cm_per_GW_to_μm_per_W(0.8)               # 0.8 cm/GW → μm/W
βmap = smooth_scalar(shapes, [β_TPA, 0.0], minds, grid)   # TPA in the Si core only
powers = 0.0:0.02:0.5                            # W
α_dBcm = [tpa_modal_loss(k0, ev0, εi, ∂ωε, βmap, grid, P, λ) for P in powers]
# textbook estimate α = β_TPA·P/Aeff (peak/effective intensity), for comparison
α_est = [4.342944819032518e4 * (0.8e-5) * P / Aeff for P in powers]  # cm/GW→μm/W, W/μm², Np/μm→dB/cm
@printf("α_TPA(P=0.5 W) = %.3f dB/cm\n", α_dBcm[end])

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="optical power P (W)", ylabel="modal TPA loss (dB/cm)",
    title="Two-photon absorption loss (Si wire, 1550 nm, β=0.8 cm/GW)")
scatterlines!(ax, collect(powers), α_dBcm, color=:darkred, label="perturbative (mode-weighted)")
lines!(ax, collect(powers), α_est, color=:orange, linestyle=:dash, label="β_TPA·P/Aeff estimate")
axislegend(ax, position=:lt)
out = joinpath(@__DIR__, "perturbation_output", "tpa_loss_Si.png")
save(out, fig)
println("saved ", out)
