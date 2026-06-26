# χ⁽³⁾ self-phase-modulation index shift of a Si₃N₄ waveguide by first-order perturbation.
#
# The modal intensity I(x,y) at power P induces Δn = n₂(x,y)·I(x,y); the Hellmann–Feynman
# engine gives the power-dependent Δneff(P) as a *fast first-order* correction (no re-solve),
# matching both the full re-solve `ModeAnalysis.solve_k_kerr` and the textbook n₂P/Aeff.
# For a 1.60×0.80 μm Si₃N₄/SiO₂ core at 1.55 μm this reproduces γ ≈ 0.95 W⁻¹m⁻¹
# (Ikeda, Opt. Express 16, 12987 (2008); n₂=2.4e-19 m²/W = 2.4e-7 μm²/W).
#
# Run:  julia --project=. examples/perturbation_kerr_spm.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations: effective_area_kerr, kerr_gamma, kerr_spm_Δneff
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
γ = kerr_gamma(kerr_n2(Si₃N₄, λ), Aeff, λ)
@printf("Si₃N₄ WG: neff=%.4f  Aeff=%.3f μm²  γ=%.3f W⁻¹m⁻¹ (Ikeda 2008)\n", k0/ω, Aeff, γ)

powers = 0.0:0.5:10.0
Δneff_pert = [kerr_spm_Δneff(k0, ev0, ω, εi, ∂ωε, n2map, grid, P) for P in powers]
Δneff_txt = [kerr_n2(Si₃N₄, λ) * P / Aeff for P in powers]

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="optical power P (W)", ylabel="Δneff (SPM)",
    title=@sprintf("Si₃N₄ SPM: γ = %.2f W⁻¹m⁻¹", γ))
scatterlines!(ax, collect(powers), Δneff_pert, label="perturbative ⟨E|Δε|E⟩", color=:purple)
lines!(ax, collect(powers), Δneff_txt, label="textbook n₂P/Aeff", color=:orange, linestyle=:dash)
axislegend(ax, position=:lt)
out = joinpath(@__DIR__, "perturbation_output", "kerr_spm_Si3N4.png")
save(out, fig)
println("saved ", out)
