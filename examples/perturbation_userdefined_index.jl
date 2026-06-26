# Arbitrary user-specified spatially-varying index perturbation Δn(x,y).
#
# Any weak index map — a localized stress/doping region, an electro-optic Δn from an
# applied field, a probe of mode sensitivity — perturbs the mode via Δε = 2 n₀ Δn and the
# Hellmann–Feynman engine. Here we sweep a Gaussian index "bump" across a Si₃N₄ waveguide
# and plot Δneff vs. bump position: the response traces the modal energy density |E|²,
# directly visualizing where the mode is sensitive to index changes. The whole curve is
# AD-differentiable in the perturbation strength and position.
#
# Run:  julia --project=. examples/perturbation_userdefined_index.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.DielectricSmoothing: δx, δy
using OptiMode.ModePerturbations: index_perturbation_Δneff
using LinearAlgebra, Printf
using CairoMakie

λ = 1.55; ω = 1 / λ
grid = Grid(4.0, 3.0, 80, 60)
solver = KrylovKitEigsolve()
mats = [Si₃N₄, SiO₂]
fε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = fε([ω])
core = MaterialShape(Cuboid([0.0, 0.0], [1.2, 0.5], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)
sm = smooth_ε(shapes, mat_vals, minds, grid)
εi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
km, ev = solve_k(ω, copy(εi), grid, solver; nev=1, k_tol=1e-11)
k0, ev0 = km[1], ev[1]

Nx, Ny = size(grid)
xc = (-grid.Δx / 2) .+ (0.5:Nx) .* δx(grid)
yc = (-grid.Δy / 2) .+ (0.5:Ny) .* δy(grid)
# Gaussian index bump (peak Δn=1e-3, width 0.25 μm) scanned along x at the core center
bump(x0) = [1e-3 * exp(-((x - x0)^2 + y^2) / (2 * 0.25^2)) for x in xc, y in yc]
x0s = -1.8:0.1:1.8
Δneff = [index_perturbation_Δneff(k0, ev0, ω, εi, bump(x0), grid) for x0 in x0s]
@printf("peak Δneff (bump at center) = %.3e\n", maximum(Δneff))

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="index-bump position x₀ (μm)", ylabel="Δneff",
    title="Δneff from a scanned Gaussian Δn(x,y) bump (Si₃N₄ WG)")
scatterlines!(ax, collect(x0s), Δneff, color=:indigo)
vlines!(ax, [-0.6, 0.6], color=:grey, linestyle=:dash)
text!(ax, 0.0, maximum(Δneff) * 0.5, text="core edges", align=(:center, :center), color=:grey, fontsize=11)
out = joinpath(@__DIR__, "perturbation_output", "userdefined_index_scan.png")
save(out, fig)
println("saved ", out)
