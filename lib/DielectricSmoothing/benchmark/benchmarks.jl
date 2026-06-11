# Benchmarks for DielectricSmoothing: primal Kottke smoothing of a 2D waveguide
# cross-section vs. AD gradients w.r.t. the material tensor values (Enzyme & Mooncake
# reverse mode).
#
# Run with:
#   julia --project=benchmark benchmark/benchmarks.jl

using BenchmarkTools
using StaticArrays
using LinearAlgebra
using GeometryPrimitives
using MaterialDispersion
using DielectricSmoothing
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using Zygote
using ForwardDiff

const SUITE = BenchmarkGroup()

"simple 2D slab-loaded ridge waveguide geometry, shapes carry material indices"
function ridge_wg(p)
    w_top, t_core, θ, t_slab = p
    edge_gap = 0.5
    Δx, Δy = 6.0, 4.0
    t_subs = (Δy - t_core - edge_gap) / 2.0 - t_slab
    c_subs_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs / 2.0
    c_slab_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs + t_slab / 2.0
    wt_half = w_top / 2
    wb_half = wt_half + (t_core * tan(θ))
    tc_half = t_core / 2
    # vertices as columns (x;y), counter-clockwise
    verts = SMatrix{2,4}(wt_half, tc_half, -wt_half, tc_half, -wb_half, -tc_half, wb_half, -tc_half)
    core = MaterialShape(GeometryPrimitives.Polygon(verts), 1)
    ax = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
    b_slab = MaterialShape(GeometryPrimitives.Cuboid(SVector(0.0, c_slab_y), SVector(Δx - edge_gap, t_slab), ax), 2)
    b_subs = MaterialShape(GeometryPrimitives.Cuboid(SVector(0.0, c_subs_y), SVector(Δx - edge_gap, t_subs), ax), 3)
    return (core, b_slab, b_subs)
end

mats = [Si₃N₄, SiO₂, MgO_LiNbO₃]
f_ε, _ = _f_ε_mats(mats, (:ω,))
mat_vals0 = hcat(f_ε([1 / 1.55]), vcat(vec([1.0 0 0; 0 1.0 0; 0 0 1.0]), zeros(18)))
shapes0 = ridge_wg([1.7, 0.7, π / 14.0, 0.2])
minds0 = (1, 2, 3, 4)
grid = Grid(6.0, 4.0, 128, 128)

loss(mv) = sum(abs2, smooth_ε(shapes0, mv, minds0, grid))

# Reverse mode through the full mapreduce pipeline is impractically slow to *compile*
# with Mooncake/Enzyme (the per-voxel kernels are covered below); the supported
# full-pipeline gradients are Zygote (reverse) and ForwardDiff (forward).
backends_pipeline = Dict(
    "Zygote(reverse)" => AutoZygote(),
    "ForwardDiff" => AutoForwardDiff(),
)
# Per-voxel Kottke kernel (no dispersion derivatives): all reverse backends apply.
S0 = collect(normcart(normalize([0.3, 0.9, 0.1])))
xk0 = vcat(0.37, mat_vals0[1:9, 1], mat_vals0[1:9, 2])
loss_kernel(x) = sum(abs2, DielectricSmoothing.f_εₑᵣ(x))
backends_kernel = Dict(
    "Enzyme(reverse)" => AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
    "Mooncake(reverse)" => AutoMooncake(; config=nothing),
    "ForwardDiff" => AutoForwardDiff(),
)

SUITE["primal"]["smooth_ε_128x128"] = @benchmarkable smooth_ε($shapes0, $mat_vals0, $minds0, $grid)
SUITE["primal"]["loss"] = @benchmarkable $loss($mat_vals0)
SUITE["primal"]["kottke_kernel"] = @benchmarkable $loss_kernel($xk0)
for (name, backend) in backends_pipeline
    prep = DI.prepare_gradient(loss, backend, mat_vals0)
    SUITE["gradient(pipeline)"][name] = @benchmarkable DI.gradient($loss, $prep, $backend, $mat_vals0)
end
for (name, backend) in backends_kernel
    prep = DI.prepare_gradient(loss_kernel, backend, xk0)
    SUITE["gradient(kernel)"][name] = @benchmarkable DI.gradient($loss_kernel, $prep, $backend, $xk0)
end

function run_and_report(suite)
    results = run(suite; verbose=false)
    t_primal = minimum(results["primal"]["loss"]).time
    t_kernel = minimum(results["primal"]["kottke_kernel"]).time
    println("\n=== DielectricSmoothing benchmark summary (128×128 grid) ===")
    println("primal smoothing loss:  $(t_primal/1e6) ms")
    for (name, _) in backends_pipeline
        t = minimum(results["gradient(pipeline)"][name]).time
        println(rpad("pipeline gradient $name:", 36), "$(t/1e6) ms   ($(round(t/t_primal; digits=1))× primal)")
    end
    println("primal Kottke kernel:   $(t_kernel/1e3) μs")
    for (name, _) in backends_kernel
        t = minimum(results["gradient(kernel)"][name]).time
        println(rpad("kernel gradient $name:", 36), "$(t/1e3) μs   ($(round(t/t_kernel; digits=1))× primal)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
