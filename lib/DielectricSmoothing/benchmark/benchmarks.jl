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

# Geometry-parameter loss: gradient flows through shape construction and the interface
# queries (surfpt_nearby/volfrac) into the smoothing. ForwardDiff propagates through the
# whole pipeline (GeometryPrimitives ≥ 0.6 parametric shapes); Mooncake handles the
# per-interface-pixel kernel (below).
loss_geom(p) = sum(abs2, smooth_ε(ridge_wg(p), mat_vals0, minds0, grid))
p_geom0 = [1.7, 0.7, π / 14.0, 0.2]

# Reverse mode through the full mapreduce pipeline is impractically slow to *compile*
# with Mooncake/Enzyme (the per-voxel kernels are covered below); the supported
# full-pipeline gradients are Zygote (reverse) and ForwardDiff (forward).
backends_pipeline = Dict(
    "Zygote(reverse)" => AutoZygote(),
    "ForwardDiff" => AutoForwardDiff(),
)
# Geometry parameters: forward mode through the full pipeline.
backends_geom_pipeline = Dict(
    "ForwardDiff" => AutoForwardDiff(),
)
# Per-interface-pixel geometry kernel: shape params → surfpt_nearby/volfrac → Kottke.
ε1_b = SMatrix{3,3}(reshape(mat_vals0[1:9, 1], 3, 3))
ε2_b = SMatrix{3,3}(reshape(mat_vals0[1:9, 2], 3, 3))
xyz_b = SVector(0.8, 0.0)
vmin_b = SVector(0.75, -0.05)
vmax_b = SVector(0.85, 0.05)
function loss_geom_kernel(p)
    w, h, cx = p
    core = GeometryPrimitives.Cuboid(SVector(cx, 0.0), SVector(w, h), SMatrix{2,2}(1.0, 0.0, 0.0, 1.0))
    r = GeometryPrimitives.surfpt_nearby(xyz_b, core)
    rvol = GeometryPrimitives.volfrac((vmin_b, vmax_b), last(r), first(r))
    return sum(abs2, avg_param(ε1_b, ε2_b, normcart(DielectricSmoothing.vec3D(last(r))), rvol))
end
pk0 = [1.6, 0.8, 0.0]
backends_geom_kernel = Dict(
    "Mooncake(reverse)" => AutoMooncake(; config=nothing),
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
SUITE["primal"]["loss_geom"] = @benchmarkable $loss_geom($p_geom0)
SUITE["primal"]["kottke_kernel"] = @benchmarkable $loss_kernel($xk0)
SUITE["primal"]["geom_kernel"] = @benchmarkable $loss_geom_kernel($pk0)
for (name, backend) in backends_pipeline
    prep = DI.prepare_gradient(loss, backend, mat_vals0)
    SUITE["gradient(pipeline)"][name] = @benchmarkable DI.gradient($loss, $prep, $backend, $mat_vals0)
end
for (name, backend) in backends_kernel
    prep = DI.prepare_gradient(loss_kernel, backend, xk0)
    SUITE["gradient(kernel)"][name] = @benchmarkable DI.gradient($loss_kernel, $prep, $backend, $xk0)
end
# Geometry-parameter gradients (material data fixed, shape parameters varied)
for (name, backend) in backends_geom_pipeline
    prep = DI.prepare_gradient(loss_geom, backend, p_geom0)
    SUITE["gradient(geometry,pipeline)"][name] = @benchmarkable DI.gradient($loss_geom, $prep, $backend, $p_geom0)
end
for (name, backend) in backends_geom_kernel
    prep = DI.prepare_gradient(loss_geom_kernel, backend, pk0)
    SUITE["gradient(geometry,kernel)"][name] = @benchmarkable DI.gradient($loss_geom_kernel, $prep, $backend, $pk0)
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
    # geometry-parameter gradients
    t_geom = minimum(results["primal"]["loss_geom"]).time
    t_gk = minimum(results["primal"]["geom_kernel"]).time
    println("\nprimal geometry loss:   $(t_geom/1e6) ms")
    for (name, _) in backends_geom_pipeline
        t = minimum(results["gradient(geometry,pipeline)"][name]).time
        println(rpad("geom pipeline gradient $name:", 36), "$(t/1e6) ms   ($(round(t/t_geom; digits=1))× primal)")
    end
    println("primal geometry kernel: $(t_gk/1e3) μs")
    for (name, _) in backends_geom_kernel
        t = minimum(results["gradient(geometry,kernel)"][name]).time
        println(rpad("geom kernel gradient $name:", 36), "$(t/1e3) μs   ($(round(t/t_gk; digits=1))× primal)")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_and_report(SUITE)
end
