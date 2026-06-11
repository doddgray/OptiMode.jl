# Benchmarks for MaxwellEigenmodes: primal eigenmode solve (solve_k) vs. its
# adjoint-method gradient through Zygote (ChainRules), Mooncake, and Enzyme.
# An efficient adjoint costs O(1) extra eigensolve-equivalents, so the reported
# gradient/primal ratios should be small single-digit multiples.
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using LinearAlgebra
using DielectricSmoothing
using MaxwellEigenmodes
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using Zygote

const SUITE = BenchmarkGroup()

function gaussian_wg_epsi(p, grid::Grid{2})
    ε_core, ε_bg, wx, wy = p
    xs, ys = x(grid), y(grid)
    Nx, Ny = size(grid)
    epsi = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        ε = ε_bg + (ε_core - ε_bg) * exp(-(xx^2 / wx^2 + yy^2 / wy^2))
        for a in 1:3
            epsi[a, a, ix, iy] = inv(ε)
        end
    end
    return epsi
end

grid = Grid(6.0, 4.0, 64, 64)
epsi0 = gaussian_wg_epsi([4.2, 2.1, 1.0, 0.6], grid)
ω0 = 1 / 1.55
solver = KrylovKitEigsolve()

solve_k_ω(om) = solve_k(om, copy(epsi0), grid, solver; nev=1)[1][1]

SUITE["primal"]["solve_k_64x64"] = @benchmarkable $solve_k_ω($ω0)
SUITE["gradient"]["Zygote(adjoint rrule)"] = @benchmarkable Zygote.gradient($solve_k_ω, $ω0)
SUITE["gradient"]["Mooncake(reverse)"] = @benchmarkable DI.derivative($solve_k_ω, AutoMooncake(; config=nothing), $ω0)
SUITE["gradient"]["Enzyme(reverse)"] = @benchmarkable DI.derivative($solve_k_ω, AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const), $ω0)

function run_and_report(suite)
    results = run(suite; verbose=false, seconds=20)
    t_primal = minimum(results["primal"]["solve_k_64x64"]).time
    println("\n=== MaxwellEigenmodes benchmark summary (64×64 grid) ===")
    println("primal solve_k:  $(t_primal/1e9) s")
    for name in keys(results["gradient"])
        t = minimum(results["gradient"][name]).time
        println(rpad("gradient $name:", 32), "$(t/1e9) s   ($(round(t/t_primal; digits=1))× primal)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
