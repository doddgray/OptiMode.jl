"""
Benchmarks for ModeAnalysis.jl

Measures:
- Field conversion H⃗ → E⃗ (via D⃗)
- Poynting vector computation
- Group index computation
- AD gradient computation (Zygote, Mooncake, Enzyme)
- Overlap integral computation
"""
using BenchmarkTools
using ModeAnalysis
using EigenModeSolver
using DielectricSmoother
using MaterialModels
using StaticArrays
using LinearAlgebra
using FFTW

println("=== ModeAnalysis.jl Benchmarks ===\n")

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
function setup_mode(; Nx=64, Ny=64, ε_val=2.25, k=0.5)
    g = Grid(2.0, 2.0, Nx, Ny)
    ε⁻¹_flat = zeros(3, 3, Nx, Ny)
    for ix=1:Nx, iy=1:Ny
        ε⁻¹_flat[:,:,ix,iy] = I/ε_val
    end
    ms = ModeSolver(k, ε⁻¹_flat, g)
    solver = KrylovKitEigsolve()
    ω²s, Hvecs = solve_ω²(ms, solver; nev=2)
    H = ComplexFloat64.(reshape(Hvecs[1], 2, g.Nx, g.Ny))
    return ms, H, g, ω²s[1], ε⁻¹_flat
end

println("Setting up test modes (64×64 grid)...")
ms, H, g, ω², ε⁻¹_flat = setup_mode()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Field conversion
# ──────────────────────────────────────────────────────────────────────────────
println("\n1. Field conversion")
d = zeros(ComplexFloat64, 3, g.Nx, g.Ny)
e = zeros(ComplexFloat64, 3, g.Nx, g.Ny)

println("  _H2d! (H→D, k× + FFT):")
@btime _H2d!($d, $H, $(ms.M̂))

println("  _d2ẽ! (D→E, ε⁻¹· + IFFT):")
_H2d!(d, H, ms.M̂)  # populate d first
@btime _d2ẽ!($e, $d, $(ms.M̂))

println("  E⃗ (full H→E pipeline):")
@btime E⃗($H, $(ms.M̂))

# ──────────────────────────────────────────────────────────────────────────────
# 2. Poynting vector
# ──────────────────────────────────────────────────────────────────────────────
println("\n2. Poynting vector S⃗")
E = E⃗(H, ms.M̂)
@btime S⃗($E, $E)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Field analysis
# ──────────────────────────────────────────────────────────────────────────────
println("\n3. Field analysis")
println("  E_relpower_xyz:")
@btime E_relpower_xyz($E)

println("  val_magmax:")
@btime val_magmax($H)

println("  canonicalize_phase:")
H_copy = copy(H)
@btime canonicalize_phase!($H_copy)

# ──────────────────────────────────────────────────────────────────────────────
# 4. AD gradient benchmarks
# ──────────────────────────────────────────────────────────────────────────────
println("\n4. AD gradients (64×64 grid)")

loss_E(H_in) = sum(abs2, E⃗(H_in, ms.M̂))

try
    using Zygote
    println("  Zygote gradient of E⃗ norm:")
    @btime Zygote.gradient($loss_E, $H)
catch e
    println("  Zygote: $e")
end

try
    using Mooncake
    rule = Mooncake.build_rrule(loss_E, H)
    println("  Mooncake gradient of E⃗ norm (compiled):")
    @btime $rule(Mooncake.CoDual($H, zero($H)))
catch e
    println("  Mooncake: $e")
end

try
    using Enzyme
    println("  Enzyme forward gradient of E⃗ norm:")
    dH = randn(ComplexFloat64, size(H))
    @btime Enzyme.autodiff(Enzyme.Forward, $loss_E, Enzyme.Duplicated($H, $dH))
catch e
    println("  Enzyme: $e")
end

# ──────────────────────────────────────────────────────────────────────────────
# 5. Scaling with grid size
# ──────────────────────────────────────────────────────────────────────────────
println("\n5. Scaling with grid size (E⃗ computation)")
for (Nx, Ny) in [(16,16), (32,32), (64,64), (128,128)]
    ms_n, H_n, g_n, _, _ = setup_mode(; Nx, Ny)
    println("  $(Nx)×$(Ny):")
    @btime E⃗($H_n, $(ms_n.M̂))
end

println("\n=== Benchmarks complete ===")
