# Benchmarks for ModeAnalysis: primal group-index evaluation vs. its gradients
# (Zygote-assembled rrule, bridged to Mooncake and Enzyme), and the analytic
# adjoint-based ng+gvd evaluation `ng_gvd`.
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using LinearAlgebra
using MaterialDispersion
using DielectricSmoothing
using DielectricSmoothing.GeometryPrimitives: Cuboid
using MaxwellEigenmodes
using ModeAnalysis
using ModeAnalysis: Zygote
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using ForwardDiff

const SUITE = BenchmarkGroup()

function gaussian_wg_eps_dispersive(ω, grid::Grid{2})
    wx, wy = 1.0, 0.6
    xs, ys = x(grid), y(grid)
    Nx, Ny = size(grid)
    eps = zeros(3, 3, Nx, Ny)
    deps = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        G = exp(-(xx^2 / wx^2 + yy^2 / wy^2))
        for a in 1:3
            eps[a, a, ix, iy] = (1.5 + 0.4 * ω^2) + ((4.0 + 2.0 * ω^2) - (1.5 + 0.4 * ω^2)) * G
            deps[a, a, ix, iy] = 0.8 * ω + (4.0 * ω - 0.8 * ω) * G
        end
    end
    return eps, deps
end

grid = Grid(6.0, 4.0, 64, 64)
ω0 = 1 / 1.55
eps0, deps0 = gaussian_wg_eps_dispersive(ω0, grid)
ddeps0 = zero(deps0)
epsi0 = sliceinv_3x3(eps0)
kmags0, evecs0 = solve_k(ω0, copy(epsi0), grid, KrylovKitEigsolve(); nev=1)
k0, ev0 = kmags0[1], evecs0[1]

f_ω(om) = group_index(k0, ev0, om, epsi0, deps0, grid)

SUITE["primal"]["group_index_64x64"] = @benchmarkable group_index($k0, $ev0, $ω0, $epsi0, $deps0, $grid)
SUITE["primal"]["ng_gvd_64x64"] = @benchmarkable ng_gvd($ω0, $k0, $ev0, $epsi0, $deps0, $ddeps0, $grid)
SUITE["gradient"]["Zygote(reverse)"] = @benchmarkable Zygote.gradient($f_ω, $ω0)
SUITE["gradient"]["Mooncake(reverse)"] = @benchmarkable DI.derivative($f_ω, AutoMooncake(; config=nothing), $ω0)
SUITE["gradient"]["Enzyme(reverse)"] = @benchmarkable DI.derivative($f_ω, AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const), $ω0)

# --- geometry-parameter sensitivities (forward geometry + reverse adjoint) ----------
# Hybrid AD gradient of mode quantities w.r.t. waveguide geometry (a Si₃N₄ core (w,h) in
# SiO₂): ForwardDiff Jacobian of the smoothed dielectric fields w.r.t. geometry composed
# with the reverse-mode adjoint of the eigensolve/post-processing. Requires the GP AD
# branch for Duals to flow through shape construction + Kottke smoothing.
const _geom_grid = Grid(4.0, 3.0, 32, 24)
const _geom_solver = KrylovKitEigsolve()
let
    fε, _ = _f_ε_mats([Si₃N₄, SiO₂], (:ω,))
    global _geom_matvals = hcat(fε([ω0]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18)))
end
_geom_shapes(p) = (MaterialShape(Cuboid([0.0, 0.0], [p[1], p[2]], [1.0 0.0; 0.0 1.0]), 1),)
function _geom_diel(p)
    sm = smooth_ε(_geom_shapes(p), _geom_matvals, (1, 2), _geom_grid)
    sliceinv_3x3(copy(selectdim(sm, 3, 1))), copy(selectdim(sm, 3, 2))
end
const _p_geom = [1.6, 0.8]
const _ei_g, _de_g = _geom_diel(_p_geom)
const _Npix_g = length(vec(_ei_g))
_diel_flat(q) = (d = _geom_diel(q); vcat(vec(d[1]), vec(d[2])))
# forward part: geometry Jacobian of the dielectric fields
_fwd_geom_jac() = ForwardDiff.jacobian(_diel_flat, _p_geom)
# reverse part: adjoint cotangents of n_eff / n_g w.r.t. the dielectric fields
_neff_diel(ei) = solve_k(ω0, ei, _geom_grid, _geom_solver; nev=1, k_tol=1e-11)[1][1] / ω0
function _ng_diel(ei, de)
    k, ev = solve_k(ω0, ei, _geom_grid, _geom_solver; nev=1, k_tol=1e-11)
    group_index(k[1], ev[1], ω0, ei, de, _geom_grid)
end
# full hybrid geometry gradients
function _grad_neff_geom()
    J = _fwd_geom_jac()
    ḡei = Zygote.gradient(_neff_diel, copy(_ei_g))[1]
    J[1:_Npix_g, :]' * vec(ḡei)
end
function _grad_ng_geom()
    J = _fwd_geom_jac()
    gei, gde = Zygote.gradient(_ng_diel, copy(_ei_g), copy(_de_g))
    J[1:_Npix_g, :]' * vec(gei) .+ J[_Npix_g+1:2_Npix_g, :]' * vec(gde)
end

SUITE["primal"]["solve_k_32x24"] = @benchmarkable solve_k($ω0, copy($_ei_g), $_geom_grid, $_geom_solver; nev=1, k_tol=1e-11)
SUITE["geometry-gradient"]["fwd_diel_jacobian"] = @benchmarkable _fwd_geom_jac()
SUITE["geometry-gradient"]["neff(forward+reverse)"] = @benchmarkable _grad_neff_geom()
SUITE["geometry-gradient"]["ng(forward+reverse)"] = @benchmarkable _grad_ng_geom()

function run_and_report(suite)
    results = run(suite; verbose=false)
    t_primal = minimum(results["primal"]["group_index_64x64"]).time
    println("\n=== ModeAnalysis benchmark summary (64×64 grid) ===")
    println("primal group_index:  $(t_primal/1e6) ms")
    println("primal ng_gvd:       $(minimum(results["primal"]["ng_gvd_64x64"]).time/1e6) ms")
    for name in keys(results["gradient"])
        t = minimum(results["gradient"][name]).time
        println(rpad("gradient $name:", 28), "$(t/1e6) ms   ($(round(t/t_primal; digits=1))× primal)")
    end
    t_solve = minimum(results["primal"]["solve_k_32x24"]).time
    println("\n=== geometry-parameter sensitivities (32×24 grid) ===")
    println("primal solve_k:      $(t_solve/1e6) ms")
    for name in ("fwd_diel_jacobian", "neff(forward+reverse)", "ng(forward+reverse)")
        t = minimum(results["geometry-gradient"][name]).time
        println(rpad("geom gradient $name:", 36), "$(t/1e6) ms   ($(round(t/t_solve; digits=1))× solve_k)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
