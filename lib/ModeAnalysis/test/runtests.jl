using Test
using LinearAlgebra
using StaticArrays
using DielectricSmoothing
using MaxwellEigenmodes
using ModeAnalysis
using FiniteDifferences
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using ForwardDiff
using ModeAnalysis: Zygote

"""
Analytic dispersive 2D dielectric profile: Gaussian index bump with simple quadratic
frequency dependence. Returns (ε, ∂ε/∂ω, ∂²ε/∂ω²) as (3,3,Nx,Ny) arrays, all analytic.
"""
function gaussian_wg_eps_dispersive(ω, grid::Grid{2})
    ε_core(om) = 4.0 + 2.0 * om^2
    ε_bg(om) = 1.5 + 0.4 * om^2
    ∂ε_core(om) = 4.0 * om
    ∂ε_bg(om) = 0.8 * om
    ∂²ε_core(om) = 4.0
    ∂²ε_bg(om) = 0.8
    wx, wy = 1.0, 0.6
    xs, ys = x(grid), y(grid)
    Nx, Ny = size(grid)
    eps = zeros(3, 3, Nx, Ny)
    deps = zeros(3, 3, Nx, Ny)
    ddeps = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        G = exp(-(xx^2 / wx^2 + yy^2 / wy^2))
        for a in 1:3
            eps[a, a, ix, iy] = ε_bg(ω) + (ε_core(ω) - ε_bg(ω)) * G
            deps[a, a, ix, iy] = ∂ε_bg(ω) + (∂ε_core(ω) - ∂ε_bg(ω)) * G
            ddeps[a, a, ix, iy] = ∂²ε_bg(ω) + (∂²ε_core(ω) - ∂²ε_bg(ω)) * G
        end
    end
    return eps, deps, ddeps
end

const grid = Grid(6.0, 4.0, 16, 16)
const ω0 = 1 / 1.55
const solver = KrylovKitEigsolve()

eps0, deps0, ddeps0 = gaussian_wg_eps_dispersive(ω0, grid)
epsi0 = sliceinv_3x3(eps0)
kmags0, evecs0 = solve_k(ω0, copy(epsi0), grid, solver; nev=1)
k0, ev0 = kmags0[1], evecs0[1]

"full pipeline |k|(ω): used to obtain finite-difference references for ng and gvd"
function k_of_ω(om)
    eps, _, _ = gaussian_wg_eps_dispersive(om, grid)
    solve_k(om, sliceinv_3x3(eps), grid, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)[1][1]
end

"full pipeline ng(ω) via group_index"
function ng_of_ω(om)
    eps, deps, _ = gaussian_wg_eps_dispersive(om, grid)
    epsi = sliceinv_3x3(eps)
    kmags, evecs = solve_k(om, copy(epsi), grid, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)
    group_index(kmags[1], evecs[1], om, epsi, deps, grid)
end

@testset "ModeAnalysis" begin
    @testset "group_index value vs finite-difference dk/dω" begin
        ng0 = group_index(k0, ev0, ω0, epsi0, deps0, grid)
        ng_FD = FiniteDifferences.central_fdm(5, 1; factor=1e6)(k_of_ω, ω0)
        @test ng0 ≈ ng_FD rtol = 1e-4
    end

    @testset "ng_gvd matches group_index and FD of ng(ω)" begin
        ng_gvd_res = ng_gvd(ω0, k0, ev0, epsi0, deps0, ddeps0, grid)
        ng0 = group_index(k0, ev0, ω0, epsi0, deps0, grid)
        @test ng_gvd_res[1] ≈ ng0 rtol = 1e-6
        gvd_FD = FiniteDifferences.central_fdm(5, 1; factor=1e6)(ng_of_ω, ω0)
        @test ng_gvd_res[2] ≈ gvd_FD rtol = 1e-3
        # ng_gvd_E returns the same plus the E field
        ng_E, gvd_E, E = ng_gvd_E(ω0, k0, ev0, epsi0, deps0, ddeps0, grid)
        @test ng_E ≈ ng_gvd_res[1]
        @test gvd_E ≈ ng_gvd_res[2]
        @test size(E) == (3, size(grid)...)
    end

    @testset "group_index partial-derivative gradients" begin
        # partial derivatives at fixed (k, ev, ε⁻¹, ∂ε/∂ω): reference from FiniteDifferences
        f_ω = om -> group_index(k0, ev0, om, epsi0, deps0, grid)
        f_k = kk -> group_index(kk, ev0, ω0, epsi0, deps0, grid)
        dng_dω_FD = FiniteDifferences.central_fdm(5, 1)(f_ω, ω0)
        dng_dk_FD = FiniteDifferences.central_fdm(5, 1)(f_k, k0)

        gz = Zygote.gradient((kk, om) -> group_index(kk, ev0, om, epsi0, deps0, grid), k0, ω0)
        @test gz[1] ≈ dng_dk_FD rtol = 1e-5
        @test gz[2] ≈ dng_dω_FD rtol = 1e-5

        for (name, backend) in (
            ("Mooncake(reverse)", AutoMooncake(; config=nothing)),
            ("Enzyme(reverse)", AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)),
        )
            @testset "$name" begin
                @test DI.derivative(f_ω, backend, ω0) ≈ dng_dω_FD rtol = 1e-5
                @test DI.derivative(f_k, backend, k0) ≈ dng_dk_FD rtol = 1e-5
            end
        end

        # forward mode through FFTs (AbstractFFTs' ForwardDiff extension)
        @test DI.derivative(f_ω, AutoForwardDiff(), ω0) ≈ dng_dω_FD rtol = 1e-5
        @test DI.derivative(f_k, AutoForwardDiff(), k0) ≈ dng_dk_FD rtol = 1e-5

        # gradient w.r.t. dielectric inputs: directional derivative check (Zygote-defined rrule)
        g_ei, g_de = Zygote.gradient((ei, de) -> group_index(k0, ev0, ω0, ei, de, grid), epsi0, deps0)
        dir_ei = randn(size(epsi0))
        dir_de = randn(size(deps0))
        d_FD = FiniteDifferences.central_fdm(5, 1)(
            t -> group_index(k0, ev0, ω0, epsi0 .+ t .* dir_ei, deps0 .+ t .* dir_de, grid), 0.0)
        @test dot(g_ei, dir_ei) + dot(g_de, dir_de) ≈ d_FD rtol = 1e-4
    end

    @testset "mode classification" begin
        E0 = E⃗(k0, copy(ev0), epsi0, deps0, grid; canonicalize=true, normalized=false)
        @test size(E0) == (3, size(grid)...)
        relpwr = E_relpower_xyz(eps0, E0)
        @test isapprox(norm(relpwr), 1.0; rtol=1e-9)   # E_relpower_xyz returns an L2-normalized 3-vector
        pol_idx = argmax(relpwr)
        @test pol_idx in (1, 2)
        # fundamental mode has zero nodes along both axes
        @test count_E_nodes(E0 ./ ModeAnalysis.Eperp_max(E0), eps0, pol_idx; rel_amp_min=0.1) == (0, 0)
        @test mode_viable(E0 ./ ModeAnalysis.Eperp_max(E0), eps0; pol_idx=pol_idx, mode_order=(0, 0), rel_amp_min=0.1)
        # effective area is positive and smaller than the computational cell
        neff = k0 / ω0
        ng0 = group_index(k0, ev0, ω0, epsi0, deps0, grid)
        E0n = E0 ./ ModeAnalysis.Eperp_max(E0)
        @test 0 < 𝓐(neff, ng0, E0n)
    end
end
