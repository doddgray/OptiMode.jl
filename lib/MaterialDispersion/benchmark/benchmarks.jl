# Benchmarks for MaterialDispersion: primal dispersion-function evaluation vs. AD
# gradients (Enzyme forward/reverse, Mooncake reverse) and the symbolically generated
# Jacobian, which serves as the "speed of light" reference for gradient efficiency.
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using MaterialDispersion
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake

const SUITE = BenchmarkGroup()

mats = [MgO_LiNbO₃, Si₃N₄, SiO₂]
f_ε, f_ε! = _f_ε_mats(mats, (:ω, :T))
fj_ε, _ = _fj_ε_mats(mats, (:ω, :T))
p0 = [1 / 1.55, 30.0]
loss(p) = sum(abs2, f_ε(p))

backends = Dict(
    "Enzyme(reverse)" => AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
    "Enzyme(forward)" => AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    "Mooncake(reverse)" => AutoMooncake(; config=nothing),
)

SUITE["primal"]["f_ε_mats"] = @benchmarkable $f_ε($p0)
SUITE["primal"]["loss"] = @benchmarkable $loss($p0)
SUITE["symbolic"]["fj_ε_mats"] = @benchmarkable $fj_ε($p0)

for (name, backend) in backends
    prep = DI.prepare_gradient(loss, backend, p0)
    SUITE["gradient"][name] = @benchmarkable DI.gradient($loss, $prep, $backend, $p0)
end

function run_and_report(suite)
    results = run(suite; verbose=false)
    t_primal = minimum(results["primal"]["loss"]).time
    println("\n=== MaterialDispersion benchmark summary ===")
    println("primal loss eval:           $(t_primal/1e3) μs")
    println("symbolic value+Jacobian:    $(minimum(results["symbolic"]["fj_ε_mats"]).time/1e3) μs")
    for (name, _) in backends
        t = minimum(results["gradient"][name]).time
        println(rpad("gradient $name:", 28), "$(t/1e3) μs   ($(round(t/t_primal; digits=1))× primal)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
