# Adjoint gradients for 3D waveguides periodic along the propagation axis (ẑ)
# — Bragg / photonic-crystal-defect waveguides — including the derivative with
# respect to the absolute spatial period Λ ≡ grid.Δz. Validated against finite
# differences for isotropic and anisotropic (incl. off-diagonal) materials.

using Test
using LinearAlgebra
using StaticArrays
using DielectricSmoothing
using MaxwellEigenmodes
using FiniteDifferences
using Zygote
using Enzyme
using DifferentiationInterface
import DifferentiationInterface as DI

"""
Inverse-permittivity field (3,3,Nx,Ny,Nz) of a z-periodic waveguide: a transverse
Gaussian index bump (physical x,y) modulated along z by a grating defined in INDEX
space, so the structure is attached to the unit cell and stretches with the period
Λ. `aniso`/`offdiag` introduce material anisotropy; `binary=true` makes a blocky
Bragg stack (soft-edged for FD smoothness) instead of a sinusoid.
"""
function periodic_wg_epsi(grid::Grid{3}; ε_core=4.0, ε_clad=2.0, wx=0.5, wy=0.4, δ=0.15,
        aniso=(1.0, 1.0, 1.0), offdiag=0.0, binary=false, duty=0.5, smoothz=0.06)
    Nx, Ny, Nz = size(grid)
    xs, ys = x(grid), y(grid)
    ax, ay, az = aniso
    epsi = zeros(3, 3, Nx, Ny, Nz)
    for iz in 1:Nz
        frac = (iz - 1) / Nz
        modz = binary ? (1 + δ * tanh((duty - frac) / smoothz)) : (1 + δ * cospi(2 * frac))
        for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
            bump = exp(-(xx^2 / wx^2 + yy^2 / wy^2))
            εb = ε_clad + (ε_core - ε_clad) * bump * modz
            ε = @SMatrix [ax*εb offdiag*εb 0.0; offdiag*εb ay*εb 0.0; 0.0 0.0 az*εb]
            epsi[:, :, ix, iy, iz] .= inv(ε)
        end
    end
    return epsi
end

@testset "3D periodic-waveguide adjoint (period Λ derivative)" begin
    solver = KrylovKitEigsolve()
    ω = 1 / 1.55
    # non-cubic, non-square grid: catches x/y/z index mix-ups in the adjoints
    grid = Grid(4.0, 3.0, 0.30, 16, 12, 8)
    Λ = grid.Δz

    cases = (
        ("isotropic Bragg", (; δ=0.15)),
        ("anisotropic", (; δ=0.12, aniso=(1.0, 1.15, 0.9))),
        ("anisotropic + off-diagonal", (; δ=0.10, aniso=(1.05, 0.95, 1.0), offdiag=0.08)),
        ("binary Bragg stack", (; binary=true, δ=0.18, ε_core=4.5)),
    )

    for (name, kw) in cases
        @testset "$name" begin
            epsi = periodic_wg_epsi(grid; kw...)

            # primal: a guided Bloch mode, effective index between clad and core
            kmags, evecs = solve_k_periodic(ω, epsi, Λ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)
            k = kmags[1]
            @test sqrt(2.0) < k / ω < sqrt(6.0)   # guided: clad index < n_eff < core index

            # (a) eigenvalue channel: ∂ω²/∂Λ at fixed k vs FD of solve_ω²
            ms = ModeSolver(k, epsi, grid; nev=1)
            ev = evecs[1] / norm(evecs[1])
            dω²dΛ_adj = MaxwellEigenmodes.∂ω²_∂Λ(ev, ms.M̂.ε⁻¹, ms.M̂.mag, ms.M̂.mn, grid)
            function ω²_of_Λ(ΛΛ)
                g = Grid(grid.Δx, grid.Δy, ΛΛ, grid.Nx, grid.Ny, grid.Nz)
                solve_ω²(ModeSolver(k, epsi, g; nev=1), solver; nev=1, tol=1e-12)[1][1]
            end
            @test dω²dΛ_adj ≈ central_fdm(5, 1)(ω²_of_Λ, Λ) rtol = 1e-4

            # (b) full period derivative ∂kz/∂Λ via the rrule vs FD of solve_k_periodic
            kΛ(ΛΛ) = solve_k_periodic(ω, epsi, ΛΛ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)[1][1]
            dkdΛ_adj = Zygote.gradient(kΛ, Λ)[1]
            @test dkdΛ_adj ≈ central_fdm(5, 1)(kΛ, Λ) rtol = 1e-3

            # (c) cross-check ∂kz/∂ω (group index) vs FD
            kω(ωω) = solve_k_periodic(ωω, epsi, Λ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)[1][1]
            @test Zygote.gradient(kω, ω)[1] ≈ central_fdm(5, 1)(kω, ω) rtol = 1e-4

            # (d) eigenvector (H̄) channel: a phase-invariant mode-field objective.
            Ncomp = 2 * length(grid)
            B = [0.5 + 0.5 * sin(0.1 * i) for i in 1:Ncomp]
            objsol(km, ev) = real(dot(ev[1], B .* ev[1])) + 0.3 * km[1]
            fobj(ωω, ei, ΛΛ) = objsol(solve_k_periodic(ωω, ei, ΛΛ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)...)
            gω, gei, gΛ = Zygote.gradient(fobj, ω, epsi, Λ)
            @test gΛ ≈ central_fdm(5, 1)(ΛΛ -> fobj(ω, epsi, ΛΛ), Λ) rtol = 1e-3
            @test gω ≈ central_fdm(5, 1)(ωω -> fobj(ωω, epsi, Λ), ω) rtol = 1e-4
            dir = zero(epsi)
            for a in 1:3
                dir[a, a, :, :, :] .= randn(size(grid)) .* 1e-3
            end
            d_FD = central_fdm(5, 1; factor=1e8)(t -> fobj(ω, epsi .+ t .* dir, Λ), 0.0)
            @test dot(gei, dir) ≈ d_FD rtol = 2e-3

            # (e) internal consistency: solve_k_periodic ω/ε gradients reproduce the
            # existing solve_k rrule on the same grid (only the Λ term is new)
            fplain(ωω, ei) = objsol(solve_k(ωω, ei, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)...)
            gω2, gei2 = Zygote.gradient(fplain, ω, epsi)
            @test gω ≈ gω2 rtol = 1e-6
            @test gei ≈ gei2 rtol = 1e-6
        end
    end
end

@testset "3D periodic-waveguide solve_k_periodic — Enzyme forward & reverse" begin
    # The period-Λ eigensolve has both an adjoint rrule and a forward frule, bridged to
    # Enzyme; verify forward and reverse Enzyme derivatives w.r.t. the absolute period Λ
    # and the frequency ω against finite differences on a guided 3D Bragg mode.
    solver = KrylovKitEigsolve()
    ω = 1 / 1.55
    grid = Grid(4.0, 3.0, 0.30, 16, 12, 8)
    Λ = grid.Δz
    epsi = periodic_wg_epsi(grid; δ=0.15)
    kΛ(ΛΛ) = solve_k_periodic(ω, epsi, ΛΛ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)[1][1]
    kω(ωω) = solve_k_periodic(ωω, epsi, Λ, grid, solver; nev=1, eig_tol=1e-12, k_tol=1e-12)[1][1]
    dΛ_FD = central_fdm(5, 1)(kΛ, Λ)
    dω_FD = central_fdm(5, 1)(kω, ω)
    for (name, b) in (("Enzyme(reverse)", AutoEnzyme(; mode=Enzyme.Reverse)),
                      ("Enzyme(forward)", AutoEnzyme(; mode=Enzyme.Forward)))
        @testset "$name" begin
            @test DI.derivative(kΛ, b, Λ) ≈ dΛ_FD rtol = 1e-3
            @test DI.derivative(kω, b, ω) ≈ dω_FD rtol = 1e-4
        end
    end
end
