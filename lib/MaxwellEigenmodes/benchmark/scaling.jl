# Grid-size scaling benchmark for the eigensolver backends: primal `solve_k` time and
# adjoint-gradient (Zygote `dk/dω`) time as a function of the spatial grid size.
#
# Backends compared:
#   - KrylovKitEigsolve()                 native Float64 CPU solver (FFTW + scalar kernels)
#   - GPUSolver(Float64; device=:cpu)     device-generic backend, CPU reference path
#   - GPUSolver(Float32; device=:cpu)     same, single precision
#   - GPUSolver(Float64/Float32; device=:cuda)   when a functional CUDA GPU is available
#
# Run with:
#   julia --project=benchmark benchmark/scaling.jl [sizes...]
# e.g. `julia --project=benchmark benchmark/scaling.jl 64 128 256 512`; default 32 64 128.

using LinearAlgebra
using DielectricSmoothing
using MaxwellEigenmodes
using Zygote
using Printf

const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

"analytic isotropic Gaussian-bump waveguide ε⁻¹ on an Nx×Ny grid"
function gaussian_wg_epsi(grid::Grid{2})
    xs, ys = x(grid), y(grid)
    Nx, Ny = size(grid)
    epsi = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        e = 2.1 + (4.2 - 2.1) * exp(-(xx^2 / 1.0^2 + yy^2 / 0.6^2))
        for a in 1:3
            epsi[a, a, ix, iy] = inv(e)
        end
    end
    return epsi
end

"minimum runtime of `f()` over `n` evaluations (after one warmup call)"
function _mintime(f, n::Int=3)
    f()
    return minimum(@elapsed(f()) for _ in 1:n)
end

function run_scaling(sizes)
    ω = 1 / 1.55
    backends = Any[
        ("KrylovKit (F64)", KrylovKitEigsolve()),
        ("GPUSolver F64 cpu", GPUSolver(Float64; device=:cpu)),
        ("GPUSolver F32 cpu", GPUSolver(Float32; device=:cpu)),
    ]
    if HAVE_CUDA
        push!(backends, ("GPUSolver F64 cuda", GPUSolver(Float64; device=:cuda)))
        push!(backends, ("GPUSolver F32 cuda", GPUSolver(Float32; device=:cuda)))
    end

    println("\n=== solve_k scaling benchmark (times in s; grad = Zygote dk/dω adjoint) ===")
    header = @sprintf("%-10s", "grid")
    for (name, _) in backends
        header *= @sprintf(" | %-18s %-18s", name * " solve", name * " grad")
    end
    println(header)

    results = Dict{Tuple{Int,String},NamedTuple}()
    for N in sizes
        grid = Grid(6.0, 4.0, N, N)
        epsi = gaussian_wg_epsi(grid)
        row = @sprintf("%-10s", "$(N)×$(N)")
        kref = Ref(0.0)
        for (name, solver) in backends
            t_solve = _mintime(() -> solve_k(ω, copy(epsi), grid, solver; nev=1))
            k1 = solve_k(ω, copy(epsi), grid, solver; nev=1)[1][1]
            kref[] == 0.0 && (kref[] = k1)
            f = om -> solve_k(om, copy(epsi), grid, solver; nev=1)[1][1]
            t_grad = _mintime(() -> Zygote.gradient(f, ω))
            results[(N, name)] = (; t_solve, t_grad, k=k1, k_rel_dev=abs(k1 - kref[]) / kref[])
            row *= @sprintf(" | %-18.4g %-18.4g", t_solve, t_grad)
        end
        println(row)
        # cross-backend agreement at this size
        devs = [(name, results[(N, name)].k_rel_dev) for (name, _) in backends]
        agree = join((@sprintf("%s: %.2e", n, d) for (n, d) in devs), ",  ")
        println(" "^10, "|k| rel. deviation vs $(backends[1][1]):  ", agree)
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    sizes = isempty(ARGS) ? [32, 64, 128] : parse.(Int, ARGS)
    HAVE_CUDA || println("(no functional CUDA GPU detected — running CPU backends only)")
    run_scaling(sizes)
end
