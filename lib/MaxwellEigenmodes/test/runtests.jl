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

    # scalar objective for AD tests: first wavenumber eigenvalue
    solve_k_ω(om) = solve_k(om, copy(epsi0), grid, solver; nev=1)[1][1]
    solve_k_ei(ei) = solve_k(ω0, ei, grid, solver; nev=1)[1][1]

    @testset "solve_k adjoint gradients (ChainRules/Zygote)" begin
        dk_dω_FD = FiniteDifferences.central_fdm(5, 1)(solve_k_ω, ω0)
        dk_dω_zyg = Zygote.gradient(solve_k_ω, ω0)[1]
        @test dk_dω_zyg ≈ dk_dω_FD rtol = 1e-4
        # group index of a guided mode is bounded by material indices (sanity)
        @test sqrt(p_wg[2]) < dk_dω_zyg < 1.5 * sqrt(p_wg[1])

        # ε⁻¹ gradient: directional derivative against finite differences
        g_ei = Zygote.gradient(solve_k_ei, copy(epsi0))[1]
        dir = randn(size(epsi0)) .* 1e-3
        # symmetrize perturbation direction the same way the solver sees ε⁻¹ (it uses full tensor)
        dk_dir_FD = FiniteDifferences.central_fdm(5, 1)(t -> solve_k_ei(epsi0 .+ t .* dir), 0.0)
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
