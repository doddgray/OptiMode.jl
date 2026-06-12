using Test
using LinearAlgebra
using StaticArrays
using DielectricSmoothing
using MaxwellEigenmodes
using FiniteDifferences
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using Zygote

# MPB-backend tests are opt-in: they need PythonCall plus the Python `meep`/`meep.mpb`
# modules (e.g. conda-forge `pymeep` via CondaPkg). Enable with OPTIMODE_TEST_MPB=true.
const TEST_MPB = get(ENV, "OPTIMODE_TEST_MPB", "false") == "true"
TEST_MPB && @eval using PythonCall

# CUDA-device tests are opt-in (require a functional GPU): OPTIMODE_TEST_CUDA=true.
# The same GPUSolver code path is always tested on the CPU (device=:cpu).
const TEST_CUDA = get(ENV, "OPTIMODE_TEST_CUDA", "false") == "true"
TEST_CUDA && @eval using CUDA

"""
Analytic, smoothly-varying isotropic dielectric profile for a 2D waveguide-like structure:
a Gaussian index bump on a uniform background. Returns the (3,3,Nx,Ny) inverse dielectric
tensor array. Smooth in all parameters, so finite-difference references are well behaved.
"""
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

const grid = Grid(6.0, 4.0, 16, 16)
const p_wg = [4.2, 2.1, 1.0, 0.6]
const epsi0 = gaussian_wg_epsi(p_wg, grid)
const ω0 = 1 / 1.55
const solver = KrylovKitEigsolve()

@testset "MaxwellEigenmodes" begin
    @testset "HelmholtzMap operator" begin
        k0 = k_guess(ω0, epsi0)
        M̂ = HelmholtzMap(k0, copy(epsi0), grid)
        @test size(M̂) == (2 * 16 * 16, 2 * 16 * 16)
        @test ishermitian(M̂)
        v = randn(ComplexF64, 2 * 16 * 16)
        w = randn(ComplexF64, 2 * 16 * 16)
        # Hermiticity check on random vectors
        @test isapprox(dot(w, M̂ * v), conj(dot(v, M̂ * w)); rtol=1e-9)
        # out-of-place HMH agrees with mutating operator quadratic form
        mag, mn = mag_mn(k0, grid)
        @test HMH(v, copy(epsi0), mag, mn) ≈ real(dot(v, M̂ * v)) rtol = 1e-6
    end

    @testset "solve_ω² and solve_k consistency" begin
        k0 = k_guess(ω0, epsi0)
        ms = ModeSolver(k0, copy(epsi0), grid; nev=2)
        evals, evecs = solve_ω²(ms, solver; nev=2)
        @test length(evals) == 2
        @test all(evals .> 0)
        # eigenpair residual
        M̂ = HelmholtzMap(k0, copy(epsi0), grid)
        for (α, v) in zip(evals, evecs)
            @test norm(M̂ * v - α * v) / norm(v) < 1e-4
        end
        # mode is guided: effective index between background and core
        kmags, kevecs = solve_k(ω0, copy(epsi0), grid, solver; nev=1)
        neff = kmags[1] / ω0
        @test sqrt(p_wg[2]) < neff < sqrt(p_wg[1])
        # round trip: solving ω² at k(ω₀) returns ω₀²
        ms2 = ModeSolver(kmags[1], copy(epsi0), grid; nev=1)
        evals2, _ = solve_ω²(ms2, solver; nev=1)
        @test evals2[1] ≈ ω0^2 rtol = 1e-6
    end

    # scalar objective for AD tests: first wavenumber eigenvalue.
    # NB: called positionally (no kwargs) so that the custom Enzyme rule imported from
    # the ChainRules rrule applies (Enzyme.@import_rrule does not cover Core.kwcall).
    solve_k_ω(om) = solve_k(om, copy(epsi0), grid, solver)[1][1]
    solve_k_ei(ei) = solve_k(ω0, ei, grid, solver)[1][1]

    @testset "solve_k adjoint gradients (ChainRules/Zygote)" begin
        dk_dω_FD = FiniteDifferences.central_fdm(5, 1)(solve_k_ω, ω0)
        dk_dω_zyg = Zygote.gradient(solve_k_ω, ω0)[1]
        @test dk_dω_zyg ≈ dk_dω_FD rtol = 1e-4
        # group index of a guided mode is bounded by material indices (sanity)
        @test sqrt(p_wg[2]) < dk_dω_zyg < 1.5 * sqrt(p_wg[1])

        # ε⁻¹ gradient: directional derivative against finite differences.
        # Use tight solver tolerances so eigensolver noise doesn't pollute the FD reference.
        solve_k_ei_tight(ei) = solve_k(ω0, ei, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)[1][1]
        g_ei = Zygote.gradient(solve_k_ei_tight, copy(epsi0))[1]
        # The adjoint stores Hermitian-tensor gradients with off-diagonal entries holding
        # the summed (i,j)+(j,i) sensitivity mirrored into both entries, so probe along a
        # diagonal-only direction where ⟨g, dir⟩ is unambiguous.
        dir = zero(epsi0)
        for a in 1:3
            dir[a, a, :, :] .= randn(size(grid)) .* 1e-3
        end
        dk_dir_FD = FiniteDifferences.central_fdm(5, 1; factor=1e8)(t -> solve_k_ei_tight(epsi0 .+ t .* dir), 0.0)
        @test dot(g_ei, dir) ≈ dk_dir_FD rtol = 1e-3
    end

    @testset "solve_k gradients (Mooncake & Enzyme via bridged rules)" begin
        dk_dω_FD = FiniteDifferences.central_fdm(5, 1)(solve_k_ω, ω0)
        for (name, backend) in (
            ("Mooncake(reverse)", AutoMooncake(; config=nothing)),
            ("Enzyme(reverse)", AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)),
        )
            @testset "$name" begin
                g = DI.derivative(solve_k_ω, backend, ω0)
                @test g ≈ dk_dω_FD rtol = 1e-4
            end
        end
    end

    @testset "MPB backend (PythonCall)" begin
        solver_mpb = MPBSolver()
        @test solver_mpb isa MaxwellEigenmodes.AbstractEigensolver
        if !TEST_MPB
            # without PythonCall the backend must fail with an instructive error
            err = try
                solve_k(ω0, copy(epsi0), grid, solver_mpb)
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("PythonCall", err.msg)
        else
            ext = Base.get_extension(MaxwellEigenmodes, :MaxwellEigenmodesPythonCallExt)
            @test ext !== nothing
            @test ext.mpb_available()

            # MPB and the native solver share the same plane-wave discretization and the
            # same (pre-smoothed) dielectric data, so their |k|(ω) must agree closely.
            kmags_kk, evecs_kk = solve_k(ω0, copy(epsi0), grid, solver; nev=2)
            kmags_mpb, evecs_mpb = solve_k(ω0, copy(epsi0), grid, solver_mpb; nev=2)
            @test length(kmags_mpb) == 2
            @test kmags_mpb[1] ≈ kmags_kk[1] rtol = 1e-4
            @test kmags_mpb[2] ≈ kmags_kk[2] rtol = 1e-4

            # MPB's eigenvectors are valid eigenvectors of our HelmholtzMap
            for (km, ev) in zip(kmags_mpb, evecs_mpb)
                M̂ = HelmholtzMap(km, copy(epsi0), grid)
                @test norm(M̂ * ev - ω0^2 .* ev) / norm(ev) < 1e-3
                @test HMH(Vector(ev), copy(epsi0), mag_mn(km, grid)...) ≈ ω0^2 rtol = 1e-4
            end

            # solve_ω² path (fixed k, MPB `run`)
            ms_mpb = ModeSolver(kmags_mpb[1], copy(epsi0), grid; nev=2)
            ω²_mpb, _ = solve_ω²(ms_mpb, solver_mpb; nev=2)
            @test ω²_mpb[1] ≈ ω0^2 rtol = 1e-4

            # the solver-generic adjoint rrule makes the MPB backend differentiable
            solve_k_ω_mpb(om) = solve_k(om, copy(epsi0), grid, solver_mpb)[1][1]
            dk_dω_FD = FiniteDifferences.central_fdm(3, 1)(solve_k_ω_mpb, ω0)
            dk_dω_zyg = Zygote.gradient(solve_k_ω_mpb, ω0)[1]
            @test dk_dω_zyg ≈ dk_dω_FD rtol = 1e-3
        end
    end

    @testset "GPUSolver (device- & precision-generic backend)" begin
        kk_ref, _ = solve_k(ω0, copy(epsi0), grid, solver; nev=2, k_tol=1e-11, eig_tol=1e-11)
        dk_dω_FD = FiniteDifferences.central_fdm(5, 1)(solve_k_ω, ω0)

        for (T, rtol_k, atol_res, rtol_g) in (
            (Float64, 1e-7, 1e-8, 1e-4),
            (Float32, 1e-5, 1e-4, 5e-3),
        )
            @testset "T=$T (cpu device)" begin
                solver_g = GPUSolver(T; device=:cpu)
                # |k|(ω) matches the native Float64 solver to precision-appropriate tolerance
                kg, evg = solve_k(ω0, copy(epsi0), grid, solver_g; nev=2)
                @test kg ≈ kk_ref rtol = rtol_k
                # eigenvectors are valid eigenvectors of the (Float64) HelmholtzMap
                for (km, ev) in zip(kg, evg)
                    M̂ = HelmholtzMap(km, copy(epsi0), grid)
                    @test norm(M̂ * ev - ω0^2 .* ev) / norm(ev) < atol_res
                end
                # solve_ω² round-trip at the solved k
                ms_g = ModeSolver(kg[1], copy(epsi0), grid; nev=2)
                ω²_g, _ = solve_ω²(ms_g, solver_g; nev=2)
                @test ω²_g[1] ≈ ω0^2 rtol = 10 * rtol_k
                # adjoint: dk/dω through the device-generic rrule vs finite differences
                fT = om -> solve_k(om, copy(epsi0), grid, solver_g)[1][1]
                @test Zygote.gradient(fT, ω0)[1] ≈ dk_dω_FD rtol = rtol_g
            end
        end

        @testset "adjoint dε⁻¹ directional derivative (Float64, cpu device)" begin
            solver_g = GPUSolver(Float64; device=:cpu)
            fei = ei -> solve_k(ω0, ei, grid, solver_g; nev=1, k_tol=1e-11, eig_tol=1e-11)[1][1]
            g_ei = Zygote.gradient(fei, copy(epsi0))[1]
            dir = zero(epsi0)
            for a in 1:3
                dir[a, a, :, :] .= randn(size(grid)) .* 1e-3
            end
            d_FD = FiniteDifferences.central_fdm(5, 1; factor=1e8)(t -> fei(epsi0 .+ t .* dir), 0.0)
            @test dot(g_ei, dir) ≈ d_FD rtol = 1e-3
        end

        if !TEST_CUDA
            # without the CUDA extension the :cuda device must fail with an instructive error
            err = try
                solve_k(ω0, copy(epsi0), grid, GPUSolver(Float32))  # device defaults to :cuda
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("CUDA", err.msg)
        else
            @testset "CUDA device" begin
                @test CUDA.functional()
                for (T, rtol_k, rtol_g) in ((Float64, 1e-7, 1e-4), (Float32, 1e-5, 5e-3))
                    solver_c = GPUSolver(T; device=:cuda)
                    kc, evc = solve_k(ω0, copy(epsi0), grid, solver_c; nev=2)
                    @test kc ≈ kk_ref rtol = rtol_k
                    # GPU and CPU runs of the same generic code agree to solver tolerance
                    kg, _ = solve_k(ω0, copy(epsi0), grid, GPUSolver(T; device=:cpu); nev=2)
                    @test kc ≈ kg rtol = 10 * rtol_k
                    fC = om -> solve_k(om, copy(epsi0), grid, solver_c)[1][1]
                    @test Zygote.gradient(fC, ω0)[1] ≈ dk_dω_FD rtol = rtol_g
                end
            end
        end
    end

    @testset "eig_adjt adjoint solver on dense Hermitian problem" begin
        n = 40
        A0 = randn(n, n)
        A = (A0 + A0') / 2
        F = eigen(A)
        α, v = F.values[1], F.vectors[:, 1]
        # gradient of f(A) = v'Bv via eigenvector adjoint, B fixed random symmetric
        B0 = randn(n, n)
        B = (B0 + B0') / 2
        x̄ = 2 * B * v
        λ = eig_adjt(A, α, v, 0.0, x̄)
        # finite-difference check of d(v'Bv)/dA along random symmetric direction
        dA0 = randn(n, n)
        dA = (dA0 + dA0') / 2 * 1e-2
        f = t -> (Ft = eigen(A + t * dA); vt = Ft.vectors[:, 1] * sign(dot(Ft.vectors[:, 1], v)); dot(vt, B * vt))
        df_FD = FiniteDifferences.central_fdm(5, 1)(f, 0.0)
        df_adj = -real(dot(λ, dA * v))   # ⟨λ|dA|v⟩ adjoint formula
        @test df_adj ≈ df_FD rtol = 1e-6
    end
end
