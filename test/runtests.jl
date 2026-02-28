"""
Top-level test suite for OptiMode.jl

Tests that the umbrella package correctly re-exports all symbols from
the four sub-packages.
"""
using Test
using OptiMode

@testset "OptiMode.jl (integration tests)" begin

    @testset "MaterialModels re-exports" begin
        @test @isdefined ε_tensor
        @test @isdefined εᵥ
        @test @isdefined Material
        @test @isdefined NumMat
        @test @isdefined silicon
        @test @isdefined vacuum
        @test @isdefined n²_sym_fmt1
        @test @isdefined rotate
        @test @isdefined eval_fn_oop
    end

    @testset "DielectricSmoother re-exports" begin
        @test @isdefined Grid
        @test @isdefined smooth_ε
        @test @isdefined normcart
        @test @isdefined τ_trans
        @test @isdefined Geometry
        @test @isdefined my_fftfreq
        @test @isdefined corners
    end

    @testset "EigenModeSolver re-exports" begin
        @test @isdefined HelmholtzMap
        @test @isdefined ModeSolver
        @test @isdefined solve_ω²
        @test @isdefined solve_k
        @test @isdefined KrylovKitEigsolve
        @test @isdefined eig_adjt
        @test @isdefined sliceinv_3x3
        @test @isdefined _cross
    end

    @testset "ModeAnalysis re-exports" begin
        @test @isdefined E⃗
        @test @isdefined S⃗
        @test @isdefined normE!
        @test @isdefined canonicalize_phase!
        @test @isdefined E_relpower_xyz
        @test @isdefined group_index
        @test @isdefined val_magmax
    end

    @testset "End-to-end integration: simple waveguide mode" begin
        using StaticArrays, LinearAlgebra

        # Small test grid
        g = Grid(2.0, 2.0, 8, 8)

        # Uniform dielectric slab
        ε_val = 2.25
        ε⁻¹_flat = zeros(3, 3, 8, 8)
        for ix=1:8, iy=1:8
            ε⁻¹_flat[:,:,ix,iy] = I/ε_val
        end

        # Solve
        k = 0.5
        ms = ModeSolver(k, ε⁻¹_flat, g)
        ω²s, Hvecs = solve_ω²(ms, KrylovKitEigsolve(); nev=2)

        @test length(ω²s) == 2
        @test all(ω²s .> 0)

        # Field analysis
        H = ComplexFloat64.(reshape(Hvecs[1], 2, g.Nx, g.Ny))
        E = E⃗(H, ms.M̂)
        @test size(E) == (3, g.Nx, g.Ny)
        @test all(isfinite, E)

        px, py, pz = E_relpower_xyz(E)
        @test px + py + pz ≈ 1.0 atol=0.01
    end

end
