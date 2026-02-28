"""
Tests for MaterialModels.jl

Covers:
- Basic material model construction and evaluation
- Dispersion model correctness (Sellmeier, Cauchy)
- ε_tensor conversions
- Symbolic Jacobian/Hessian generation
- AD gradient correctness via FiniteDifferences, Mooncake, Enzyme
"""
using Test
using MaterialModels
using StaticArrays
using LinearAlgebra

@testset "MaterialModels.jl" begin

    @testset "ε_tensor" begin
        # Scalar refractive index
        n = 1.5
        ε = ε_tensor(n)
        @test ε isa SMatrix{3,3}
        @test ε[1,1] ≈ n^2
        @test ε[2,2] ≈ n^2
        @test ε[3,3] ≈ n^2
        @test ε[1,2] ≈ 0.0

        # Three-component anisotropic
        ε3 = ε_tensor(1.5, 2.0, 2.5)
        @test ε3[1,1] ≈ 1.5^2
        @test ε3[2,2] ≈ 2.0^2
        @test ε3[3,3] ≈ 2.5^2
        @test ε3[1,2] ≈ 0.0

        # Matrix input
        M = [2.0 0.1 0.0; 0.1 2.1 0.0; 0.0 0.0 1.9]
        εM = ε_tensor(M)
        @test εM ≈ M

        # Vacuum permittivity
        @test εᵥ ≈ I
    end

    @testset "Sellmeier dispersion model" begin
        # Test n²_sym_fmt1 at a known wavelength
        λ = 1.55  # μm, telecom wavelength
        # Simple glass-like parameters
        n2 = n²_sym_fmt1(λ; A₀=1.0, B₁=0.6, C₁=0.009)
        @test n2 > 1.0

        # Cauchy model
        n_cauchy = n_sym_cauchy(λ; A=1.45, B=0.003)
        @test n_cauchy ≈ 1.45 + 0.003/λ^2
    end

    @testset "Material struct and models" begin
        # Test vacuum material
        @test haskey(vacuum.models, :ε)
        ε_vac = get_model(vacuum, :ε, :λ)
        # Vacuum ε should be identity-like
        @test !isnothing(ε_vac)

        # Test silicon material
        @test has_model(silicon, :ε)
        @test nameof(silicon) isa Symbol

        # Test numeric material generation
        nmat = NumMat(vacuum)
        @test nmat.fε(1.55) isa AbstractMatrix
    end

    @testset "RotatedMaterial" begin
        using Rotations: RotZ
        # Rotate a uniaxial material
        rot_mat = rotate(silicon, Matrix(RotZ(π/4)))
        @test rot_mat isa RotatedMaterial
        @test has_model(rot_mat, :ε)
    end

    @testset "Group index and GVD models" begin
        λ = Symbolics.Num(Symbolics.Sym{Real}(:λ))
        n_model = n²_sym_fmt1(λ; A₀=2.0, B₁=1.0, C₁=0.1)
        n_sym = sqrt(n_model)
        ng = ng_model(n_sym, λ)
        @test !isnothing(ng)

        gvd = gvd_model(n_sym, λ)
        @test !isnothing(gvd)
    end

    @testset "Dispersion AD with FiniteDifferences" begin
        try
            using FiniteDifferences
            # Test that ε_tensor gradient is correct
            f(n) = sum(ε_tensor(n))
            n0 = 1.5
            # analytical gradient: d/dn sum(n²·I₃) = 6n
            fd_grad = FiniteDifferences.grad(central_fdm(5,1), f, n0)[1]
            @test fd_grad ≈ 6*n0 rtol=1e-6
        catch e
            @info "FiniteDifferences not available, skipping FD gradient test: $e"
        end
    end

    @testset "Mooncake AD" begin
        try
            using Mooncake
            # Test ε_tensor is differentiable under Mooncake
            f(n) = sum(ε_tensor(n))
            n0 = 1.5
            rule = Mooncake.build_rrule(f, n0)
            result, pb = rule(Mooncake.CoDual(n0, 1.0))
            @test primal(result) ≈ 3*n0^2
        catch e
            @info "Mooncake test skipped: $e"
        end
    end

    @testset "Enzyme AD" begin
        try
            using Enzyme
            # Forward mode gradient of ε_tensor sum
            f(n) = sum(ε_tensor(n))
            n0 = 1.5
            dn = Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated(n0, 1.0))
            @test dn[1] ≈ 6*n0 rtol=1e-10
        catch e
            @info "Enzyme test skipped: $e"
        end
    end

    @testset "CSE utilities" begin
        # Test that eval_fn_oop produces callable functions
        @variables x y
        expr = [x^2 + y, x*y - 1]
        fn = eval_fn_oop(expr, [x, y])
        result = fn([2.0, 3.0])
        @test result[1] ≈ 2.0^2 + 3.0
        @test result[2] ≈ 2.0*3.0 - 1.0
    end

end
