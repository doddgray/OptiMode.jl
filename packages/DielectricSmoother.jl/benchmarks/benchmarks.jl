"""
Benchmarks for DielectricSmoother.jl

Measures:
- Grid construction and coordinate access
- τ_trans / τ⁻¹_trans transforms
- Kottke averaging (avg_param_rot)
- Full grid smoothing (smooth_ε)
- AD gradients (Zygote, Mooncake, Enzyme)
"""
using BenchmarkTools
using DielectricSmoother
using MaterialModels
using StaticArrays
using LinearAlgebra
using GeometryPrimitives

println("=== DielectricSmoother.jl Benchmarks ===\n")

# Setup
g = Grid(2.0, 2.0, 64, 64)
λ = 1.55

# Silicon waveguide geometry
box = Box(SVector(0.0, 0.0, 0.0), SVector(0.45, 0.22, 0.0),
          Matrix{Float64}(I, 3, 3), NumMat(silicon))
shapes = (box,)
mats = [NumMat(silicon), NumMat(vacuum)]
mat_vals = reduce(hcat, [
    vcat(vec(m.fε(λ)), vec(m.fnng(λ)), vec(m.fngvd(λ))) for m in mats
])
minds = matinds(shapes, mats)

# ──────────────────────────────────────────────────────────────────────────────
println("1. Grid coordinate access")
@btime x($g)
@btime xc($g)
@btime corners($g)

# ──────────────────────────────────────────────────────────────────────────────
println("\n2. τ_trans / τ⁻¹_trans")
ε_test = [2.0 0.1 0.0; 0.1 1.9 0.05; 0.0 0.05 2.2]
@btime τ_trans($ε_test)
@btime τ⁻¹_trans(τ_trans($ε_test))

# ──────────────────────────────────────────────────────────────────────────────
println("\n3. avg_param_rot (Kottke averaging)")
ε1 = 2.0 * Matrix{Float64}(I, 3, 3)
ε2 = 12.0 * Matrix{Float64}(I, 3, 3)
@btime avg_param_rot($ε1, $ε2, 0.5)

# ──────────────────────────────────────────────────────────────────────────────
println("\n4. Full grid smoothing (smooth_ε)")
println("  Grid: $(size(g)), $(length(g)) cells")
@btime smooth_ε($shapes, $mat_vals, $minds, $g)

# ──────────────────────────────────────────────────────────────────────────────
println("\n5. Kottke kernel benchmarks (εₑ_∂ωεₑ_∂²ωεₑ)")
r₁ = 0.6
S = normcart(SVector{3}(0.0, 1.0, 0.0))
∂ωε1 = zeros(3,3)
∂ωε2 = zeros(3,3)
∂²ωε1 = zeros(3,3)
∂²ωε2 = zeros(3,3)
@btime εₑ_∂ωεₑ_∂²ωεₑ($r₁, $S, $ε1, $ε2, $∂ωε1, $∂ωε2, $∂²ωε1, $∂²ωε2)

# ──────────────────────────────────────────────────────────────────────────────
println("\n6. AD gradient benchmarks (smooth_ε w.r.t. mat_vals)")

try
    using Zygote
    f_zygote(mv) = sum(smooth_ε(shapes, mv, minds, g))
    println("  Zygote gradient (64×64 grid):")
    @btime Zygote.gradient($f_zygote, $mat_vals)
catch e
    println("  Zygote not available: $e")
end

try
    using Mooncake
    f_mc(mv) = sum(smooth_ε(shapes, mv, minds, g))
    rule = Mooncake.build_rrule(f_mc, mat_vals)
    println("  Mooncake gradient (compiled rule):")
    @btime $rule(Mooncake.CoDual($mat_vals, zero($mat_vals)))
catch e
    println("  Mooncake not available: $e")
end

println("\n=== Benchmarks complete ===")
