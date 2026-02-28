"""
Benchmarks for EigenModeSolver.jl

Measures:
- HelmholtzMap application time (core inner-loop operation)
- Full eigensolve time (solve_ω²)
- AD gradient computation (Zygote, Mooncake, Enzyme)
- Scaling with grid size (Nx × Ny)
"""
using BenchmarkTools
using EigenModeSolver
using DielectricSmoother
using MaterialModels
using StaticArrays
using LinearAlgebra
using FFTW

println("=== EigenModeSolver.jl Benchmarks ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# Setup functions
# ──────────────────────────────────────────────────────────────────────────────

function make_modesolver(; Nx=32, Ny=32, ε_val=2.25, k=0.5)
    g = Grid(2.0, 2.0, Nx, Ny)
    ε⁻¹_flat = zeros(3, 3, Nx, Ny)
    for ix=1:Nx, iy=1:Ny
        ε⁻¹_flat[:,:,ix,iy] = I/ε_val
    end
    return ModeSolver(k, ε⁻¹_flat, g)
end

# ──────────────────────────────────────────────────────────────────────────────
# 1. HelmholtzMap application
# ──────────────────────────────────────────────────────────────────────────────
println("1. HelmholtzMap (M̂H) application")
for (Nx, Ny) in [(16,16), (32,32), (64,64)]
    ms = make_modesolver(; Nx, Ny)
    n_modes = 2*Nx*Ny
    H_in = randn(ComplexFloat64, n_modes)
    H_out = similar(H_in)
    println("  $(Nx)×$(Ny) grid:")
    @btime mul!($H_out, $(ms.M̂), $H_in)
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. Full eigensolve (KrylovKit)
# ──────────────────────────────────────────────────────────────────────────────
println("\n2. Full eigensolve (solve_ω², KrylovKit, nev=4)")
for (Nx, Ny) in [(16,16), (32,32)]
    ms = make_modesolver(; Nx, Ny)
    solver = KrylovKitEigsolve()
    println("  $(Nx)×$(Ny) grid:")
    @btime solve_ω²($ms, $solver; nev=4)
end

# ──────────────────────────────────────────────────────────────────────────────
# 3. LOBPCG eigensolver
# ──────────────────────────────────────────────────────────────────────────────
println("\n3. Full eigensolve (solve_ω², LOBPCG, nev=4)")
for Nx in [16, 32]
    ms = make_modesolver(; Nx, Ny=Nx)
    solver = IterativeSolversLOBPCG()
    println("  $(Nx)×$(Nx) grid:")
    @btime solve_ω²($ms, $solver; nev=4)
end

# ──────────────────────────────────────────────────────────────────────────────
# 4. sliceinv_3x3
# ──────────────────────────────────────────────────────────────────────────────
println("\n4. sliceinv_3x3 (3×3 block matrix inversion)")
for (Nx, Ny) in [(32,32), (64,64), (128,128)]
    A = zeros(3, 3, Nx, Ny)
    for ix=1:Nx, iy=1:Ny
        M = randn(3,3)
        A[:,:,ix,iy] = M'*M + 2*I
    end
    println("  $(Nx)×$(Ny) grid:")
    @btime sliceinv_3x3($A)
end

# ──────────────────────────────────────────────────────────────────────────────
# 5. AD gradient benchmarks
# ──────────────────────────────────────────────────────────────────────────────
println("\n5. AD gradient benchmarks (32×32 grid)")
ms = make_modesolver(; Nx=32, Ny=32)
solver = KrylovKitEigsolve()
ε_val = 2.25
ε⁻¹_flat = zeros(3, 3, 32, 32)
for ix=1:32, iy=1:32
    ε⁻¹_flat[:,:,ix,iy] = I/ε_val
end

function loss(ε⁻¹)
    ms = ModeSolver(0.5, ε⁻¹, Grid(2.0, 2.0, 32, 32))
    ω²s, _ = solve_ω²(ms, KrylovKitEigsolve(); nev=1)
    return ω²s[1]
end

try
    using Zygote
    println("  Zygote (reverse mode):")
    @btime Zygote.gradient($loss, $ε⁻¹_flat)
catch e
    println("  Zygote: $e")
end

try
    using Mooncake
    rule = Mooncake.build_rrule(loss, ε⁻¹_flat)
    println("  Mooncake (compiled rule):")
    @btime $rule(Mooncake.CoDual($ε⁻¹_flat, zero($ε⁻¹_flat)))
catch e
    println("  Mooncake: $e")
end

# ──────────────────────────────────────────────────────────────────────────────
# 6. eig_adjt: adjoint solve
# ──────────────────────────────────────────────────────────────────────────────
println("\n6. eig_adjt (adjoint eigensolver for reverse AD)")
ms = make_modesolver(; Nx=16, Ny=16)
solver = KrylovKitEigsolve()
ω²s, Hvecs = solve_ω²(ms, solver; nev=2)
α = ω²s[1]
x⃗ = Hvecs[1]
ᾱ = 1.0
x̄ = randn(ComplexFloat64, length(x⃗))
@btime eig_adjt($(ms.M̂), $α, $x⃗, $ᾱ, $x̄)

println("\n=== Benchmarks complete ===")
