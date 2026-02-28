"""
Benchmarks for MaterialModels.jl

Measures performance of:
- Material model evaluation (ε(λ))
- Symbolic Jacobian/Hessian generation
- AD gradient computation (Mooncake, Enzyme, FiniteDifferences)
"""
using BenchmarkTools
using MaterialModels
using StaticArrays
using LinearAlgebra

println("=== MaterialModels.jl Benchmarks ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dielectric tensor evaluation
# ──────────────────────────────────────────────────────────────────────────────
println("1. ε_tensor construction")
n0 = 1.5
@btime ε_tensor($n0)
@btime ε_tensor($n0, $(1.6), $(2.0))

# ──────────────────────────────────────────────────────────────────────────────
# 2. Sellmeier model evaluation
# ──────────────────────────────────────────────────────────────────────────────
println("\n2. Sellmeier dispersion (n²_sym_fmt1)")
λ0 = 1.55
@btime n²_sym_fmt1($λ0; A₀=2.0, B₁=0.5, C₁=0.01, B₂=0.1, C₂=0.1)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Material model function evaluation (pre-compiled)
# ──────────────────────────────────────────────────────────────────────────────
println("\n3. NumMat precompiled function evaluation")
nmat = NumMat(silicon)
@btime $(nmat.fε)($(1.55))
@btime $(nmat.fnng)($(1.55))
@btime $(nmat.fngvd)($(1.55))

# ──────────────────────────────────────────────────────────────────────────────
# 4. Symbolic Jacobian generation
# ──────────────────────────────────────────────────────────────────────────────
println("\n4. Symbolic Jacobian generation (_fj_ε_mats)")
mats = [silicon, SiO₂]
println("  Generating f_ε_mats for [silicon, SiO2]...")
t_gen = @elapsed _f_ε_mats(mats, (:ω,))
println("  Generation time: $(round(t_gen, sigdigits=3)) s")

f_ε, _ = _f_ε_mats(mats, (:ω,))
ω0 = 2π / 1.55
println("\n5. Compiled _f_ε_mats evaluation")
@btime $f_ε([$ω0])

# ──────────────────────────────────────────────────────────────────────────────
# 5. AD gradient benchmarks
# ──────────────────────────────────────────────────────────────────────────────
println("\n6. AD gradient benchmarks")

try
    using FiniteDifferences
    f_sum_ε(n) = sum(ε_tensor(n))
    println("  FiniteDifferences (central_fdm):")
    @btime FiniteDifferences.grad(central_fdm(5,1), $f_sum_ε, $n0)
catch e
    println("  FiniteDifferences not available: $e")
end

try
    using Enzyme
    f_sum_ε(n) = sum(ε_tensor(n))
    println("  Enzyme (forward mode):")
    @btime Enzyme.autodiff(Enzyme.Forward, $f_sum_ε, Enzyme.Duplicated($n0, 1.0))
    println("  Enzyme (reverse mode):")
    @btime Enzyme.autodiff(Enzyme.Reverse, $f_sum_ε, Enzyme.Active($n0))
catch e
    println("  Enzyme not available: $e")
end

try
    using Mooncake
    f_sum_ε(n) = sum(ε_tensor(n))
    rule = Mooncake.build_rrule(f_sum_ε, n0)
    println("  Mooncake (reverse mode):")
    @btime $rule(Mooncake.CoDual($n0, 1.0))
catch e
    println("  Mooncake not available: $e")
end

println("\n=== Benchmarks complete ===")
