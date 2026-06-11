# Benchmarks for ModeAnalysis: primal group-index evaluation vs. its gradients
# (Zygote-assembled rrule, bridged to Mooncake and Enzyme), and the analytic
# adjoint-based ng+gvd evaluation `ng_gvd`.
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using LinearAlgebra
using DielectricSmoothing
using MaxwellEigenmodes
using ModeAnalysis
using ModeAnalysis: Zygote
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake

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
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
