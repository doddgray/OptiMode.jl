# Automatic-differentiation backend benchmarks for the OptiMode eigenmode stack.
#
# Times one gradient/derivative evaluation (after a warm-up call to exclude compilation)
# for every working AD backend on the core differentiable kernels:
#
#   * solve_k(ω)              — Helmholtz eigensolve wavenumber, adjoint + forward rules
#   * solve_k_periodic(Λ)     — z-periodic (Bragg) eigensolve, period derivative
#   * group_index(ω)          — group-index post-processing (FFT/Tullio)
#   * neff(ε field)           — sliceinv_3x3 → solve_k (the Enzyme-able sub-stack)
#
# Backends: ForwardDiff, Zygote, Mooncake, Enzyme (forward), Enzyme (reverse), and
# FiniteDifferences as a reference. Each is also checked against the finite-difference
# value so the timing table doubles as a correctness table. Backends that do not support a
# given function (or are not meaningful for it) are reported as "n/a".
#
# Run:  julia --project=<env-with-AD-backends> examples/ad_backend_benchmarks.jl

using LinearAlgebra, StaticArrays, Printf
using MaterialDispersion, DielectricSmoothing, MaxwellEigenmodes, ModeAnalysis
using ForwardDiff, FiniteDifferences, Zygote, Enzyme, Mooncake
using DifferentiationInterface
import DifferentiationInterface as DI

const FD5 = central_fdm(5, 1)
const REV   = AutoEnzyme(; mode = set_runtime_activity(Enzyme.Reverse))
const FWD   = AutoEnzyme(; mode = set_runtime_activity(Enzyme.Forward))
solver = KrylovKitEigsolve()

"Median wall time (s) of `f()` over `n` trials after one warm-up."
function timeit(f; n = 5)
    f()                                   # warm up (compile)
    ts = Float64[]
    for _ in 1:n
        t = @elapsed f()
        push!(ts, t)
    end
    sort!(ts)
    return ts[(end + 1) ÷ 2]
end

"Run a scalar-input derivative across backends, printing time + relative error vs FD."
function bench_scalar(name, f, x0, backends)
    ref = FD5(f, x0)
    @printf("\n%s   (FD ref = %.6g)\n", name, ref)
    for (bname, b) in backends
        try
            g = DI.derivative(f, b, x0)
            t = timeit(() -> DI.derivative(f, b, x0))
            @printf("  %-16s % .6g   rel=%.2e   %8.2f ms\n", bname, g, abs(g - ref) / abs(ref), 1e3t)
        catch e
            @printf("  %-16s n/a (%s)\n", bname, first(split(sprint(showerror, e), '\n')))
        end
    end
end

"Run an array-input gradient across backends, printing time + relative error vs a reference gradient."
function bench_array(name, f, x0, gref, backends)
    @printf("\n%s   (‖gref‖ = %.6g)\n", name, norm(gref))
    for (bname, b) in backends
        try
            g = DI.gradient(f, b, x0)
            t = timeit(() -> DI.gradient(f, b, x0))
            @printf("  %-16s rel=%.2e   %8.2f ms\n", bname, norm(g .- gref) / norm(gref), 1e3t)
        catch e
            @printf("  %-16s n/a (%s)\n", bname, first(split(sprint(showerror, e), '\n')))
        end
    end
end

# ---------------------------------------------------------------------------------------
# Test structures
# ---------------------------------------------------------------------------------------
fε, _ = _f_ε_mats([Si₃N₄, SiO₂], (:ω,))
grid  = Grid(4.0, 3.0, 12, 10)
core  = MaterialShape(DielectricSmoothing.GeometryPrimitives.Cuboid([0.0, 0.0], [1.6, 0.7], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)
ω0   = 1 / 1.55
mv0  = hcat(fε([ω0]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18)))
sm   = smooth_ε(shapes, mv0, minds, grid)
εf   = copy(selectdim(sm, 3, 1))
∂ωε  = copy(selectdim(sm, 3, 2))
epsi = sliceinv_3x3(εf)

# 3D periodic (Bragg) structure
grid3 = Grid(4.0, 3.0, 0.30, 16, 12, 8); Λ0 = grid3.Δz; ω3 = 1 / 1.55
epsi3 = let (Nx, Ny, Nz) = size(grid3), xs = x(grid3), ys = y(grid3)
    e = zeros(3, 3, Nx, Ny, Nz)
    for iz in 1:Nz, (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        bump = exp(-(xx^2 / 0.5^2 + yy^2 / 0.4^2)); εb = 2.0 + 2.0 * bump * (1 + 0.15 * cospi(2 * (iz - 1) / Nz))
        for a in 1:3; e[a, a, ix, iy, iz] = inv(εb); end
    end
    e
end

k0, ev0 = solve_k(ω0, epsi, grid, solver; nev = 1)

println("="^78)
println("OptiMode AD backend benchmarks (median of 5, after warm-up)")
println("="^78)

allb = (("ForwardDiff", AutoForwardDiff()), ("Zygote", AutoZygote()),
        ("Mooncake", AutoMooncake(; config = nothing)), ("Enzyme-fwd", FWD), ("Enzyme-rev", REV))

bench_scalar("solve_k(ω) — 2D ridge", om -> solve_k(om, epsi, grid, solver; nev = 1)[1][1], ω0, allb)
bench_scalar("solve_k_periodic(Λ) — 3D Bragg",
             Λ -> solve_k_periodic(ω3, epsi3, Λ, grid3, solver; nev = 1, eig_tol = 1e-12, k_tol = 1e-12)[1][1], Λ0, allb)
bench_scalar("group_index(ω)", om -> group_index(k0[1], ev0[1], om, epsi, ∂ωε, grid), ω0, allb)
function ng_end_to_end(om)
    k, ev = solve_k(om, epsi, grid, solver; nev = 1)
    return group_index(k[1], ev[1], om, epsi, ∂ωε, grid)
end
bench_scalar("ng(solve_k(ω)) end-to-end", ng_end_to_end, ω0, allb)

# array-input: ε-field gradient of n_eff through sliceinv → solve_k. Reference via Zygote
# (rrule through the eigensolver); ForwardDiff cannot trace solve_k, so it is reported n/a.
neff_eps(e) = solve_k(ω0, sliceinv_3x3(e), grid, solver; nev = 1)[1][1] / ω0
bench_array("neff(ε field) — sliceinv→solve_k", neff_eps, εf, Zygote.gradient(neff_eps, εf)[1], allb)

println("\nDone.")
