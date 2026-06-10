using Test
using LinearAlgebra
using MaterialDispersion
using Rotations
using FiniteDifferences
using ForwardDiff
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake

const backends_reverse = Dict(
    "Enzyme(reverse)" => AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
    "Mooncake(reverse)" => AutoMooncake(; config=nothing),
)
const backends_forward = Dict(
    "Enzyme(forward)" => AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    "ForwardDiff" => AutoForwardDiff(),
)

@testset "MaterialDispersion" begin
    @testset "refractive index values" begin
        # SiO₂ ordinary index at 1.55 μm (Malitson) ≈ 1.444
        n_SiO₂_1550 = sqrt(ε_fn(SiO₂)(1.55)[1, 1])
        @test isapprox(n_SiO₂_1550, 1.444; atol=2e-3)
        # Si₃N₄ index at 1.55 μm ≈ 1.99-2.0
        n_Si₃N₄_1550 = sqrt(ε_fn(Si₃N₄)(1.55)[1, 1])
        @test isapprox(n_Si₃N₄_1550, 1.99; atol=2e-2)
        # LiNbO₃ is uniaxial: εxx == εyy ≠ εzz
        ε_LN = ε_fn(LiNbO₃)(1.55)
        @test ε_LN[1, 1] == ε_LN[2, 2]
        @test ε_LN[1, 1] != ε_LN[3, 3]
        @test isapprox(sqrt(ε_LN[1, 1]), 2.211; atol=5e-3)   # nₒ @ 1.55 μm
        @test isapprox(sqrt(ε_LN[3, 3]), 2.138; atol=5e-3)   # nₑ @ 1.55 μm
    end

    @testset "group index models" begin
        # group index from symbolic model: ng = n - λ dn/dλ. Check against ForwardDiff of n(λ).
        n_fn = let fε = ε_fn(SiO₂)
            lm -> sqrt(fε(lm)[1, 1])
        end
        λ0 = 1.35
        ng_sym = generate_fn(SiO₂, ng_model(SiO₂), :λ)(λ0)[1, 1]
        ng_fd = n_fn(λ0) - λ0 * ForwardDiff.derivative(n_fn, λ0)
        @test isapprox(ng_sym, ng_fd; rtol=1e-6)
    end

    mats = [LiNbO₃, Si₃N₄, SiO₂]
    n_mats = length(mats)
    f_ε, f_ε! = _f_ε_mats(mats, (:ω,))
    fj_ε, fj_ε! = _fj_ε_mats(mats, (:ω,))
    fjh_ε, fjh_ε! = _fjh_ε_mats(mats, (:ω,))
    ω0 = 1 / 1.55
    p0 = [ω0]

    @testset "generated dispersion functions" begin
        f0 = f_ε(p0)
        ε, ∂ωε, ∂²ωε = ε_views(f0, n_mats)
        # consistency with single-material models
        @test isapprox(Array(ε[3]), Array(ε_fn(SiO₂)(1 / ω0)); rtol=1e-9)
        # in-place == out-of-place
        f0ip = similar(vec(f0))
        f_ε!(f0ip, p0)
        @test vec(f0) ≈ f0ip
        # fj/fjh agree with f
        fj0 = fj_ε(p0)
        @test vec(f0) ≈ fj0[:, 1]
        fjh0 = fjh_ε(p0)
        @test vec(f0) ≈ fjh0[:, 1]
        @test fj0[:, 2] ≈ fjh0[:, 2]
        # the symbolic ω-Jacobian of ε must equal the symbolically-differentiated ∂ωε model
        nflat = 9 * n_mats
        @test fj0[1:nflat, 2] ≈ vec(f0)[nflat+1:2nflat] rtol = 1e-9
    end

    @testset "AD gradient correctness" begin
        fd_jac = FiniteDifferences.jacobian(central_fdm(5, 1), f_ε, p0)[1]
        j_sym = fj_ε(p0)[:, 2:end]
        @test fd_jac ≈ j_sym rtol = 1e-6

        loss = p -> sum(abs2, f_ε(p))
        g_ref = FiniteDifferences.grad(central_fdm(5, 1), loss, p0)[1]

        for (name, backend) in backends_reverse
            @testset "$name" begin
                g = DI.gradient(loss, backend, p0)
                @test g ≈ g_ref rtol = 1e-6
            end
        end
        for (name, backend) in backends_forward
            @testset "$name" begin
                g = DI.gradient(loss, backend, p0)
                @test g ≈ g_ref rtol = 1e-6
            end
        end
    end

    @testset "AD with temperature-dependent models" begin
        matsT = [MgO_LiNbO₃, Si₃N₄, SiO₂]
        fT, _ = _f_ε_mats(matsT, (:ω, :T))
        fjT, _ = _fj_ε_mats(matsT, (:ω, :T))
        pT0 = [ω0, 35.0]
        lossT = p -> sum(abs2, fT(p))
        gT_ref = FiniteDifferences.grad(central_fdm(5, 1), lossT, pT0)[1]
        # symbolic-Jacobian-based reference for the same loss
        fT0, jT0 = fjT(pT0)[:, 1], fjT(pT0)[:, 2:end]
        @test transpose(jT0) * (2 .* fT0) ≈ gT_ref rtol = 1e-6
        for (name, backend) in merge(backends_reverse, backends_forward)
            @testset "$name" begin
                g = DI.gradient(lossT, backend, pT0)
                @test g ≈ gT_ref rtol = 1e-6
            end
        end
    end

    @testset "rotated materials" begin
        R = Matrix(RotZ(π / 4))
        LN_rot = rotate(LiNbO₃, R; name=:LiNbO₃_rot45)
        ε_rot = generate_fn(LN_rot, :ε, :λ)(1.55)
        ε_unrot = ε_fn(LiNbO₃)(1.55)
        @test isapprox(Array(ε_rot), R' * Array(ε_unrot) * R; rtol=1e-9) ||
              isapprox(Array(ε_rot), R * Array(ε_unrot) * R'; rtol=1e-9)
    end

    if get(ENV, "OPTIMODE_TEST_REACTANT", "false") == "true"
        @testset "Reactant compilation" begin
            using Reactant
            f_c, p_ra = reactant_compile_dispersion(f_ε, p0)
            @test Array(f_c(Reactant.to_rarray(p0))) ≈ Array(f_ε(p0)) rtol = 1e-6
        end
    end
end
