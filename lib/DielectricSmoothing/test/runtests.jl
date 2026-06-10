using Test
using LinearAlgebra
using StaticArrays
using GeometryPrimitives
using MaterialDispersion
using DielectricSmoothing
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

"simple 2D slab-loaded ridge waveguide geometry, shapes carry material indices"
function ridge_wg(p)
    w_top, t_core, θ, t_slab = p
    edge_gap = 0.5
    Δx, Δy = 6.0, 4.0
    t_subs = (Δy - t_core - edge_gap) / 2.0 - t_slab
    c_subs_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs / 2.0
    c_slab_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs + t_slab / 2.0
    wt_half = w_top / 2
    wb_half = wt_half + (t_core * tan(θ))
    tc_half = t_core / 2
    # vertices as columns (x;y), counter-clockwise
    verts = SMatrix{2,4}(wt_half, tc_half, -wt_half, tc_half, -wb_half, -tc_half, wb_half, -tc_half)
    core = MaterialShape(GeometryPrimitives.Polygon(verts), 1)
    ax = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
    b_slab = MaterialShape(GeometryPrimitives.Cuboid(SVector(0.0, c_slab_y), SVector(Δx - edge_gap, t_slab), ax), 2)
    b_subs = MaterialShape(GeometryPrimitives.Cuboid(SVector(0.0, c_subs_y), SVector(Δx - edge_gap, t_subs), ax), 3)
    return (core, b_slab, b_subs)
end

@testset "DielectricSmoothing" begin
    @testset "Grid" begin
        g2 = Grid(6.0, 4.0, 64, 32)
        @test size(g2) == (64, 32)
        @test ndims(g2) == 2
        @test δx(g2) ≈ 6.0 / 64
        @test δV(g2) ≈ (6.0 / 64) * (4.0 / 32)
        @test length(x(g2)) == 64
        @test x(g2)[1] ≈ -3.0
        @test all(isapprox.(diff(x(g2)), δx(g2)))
        @test length(corners(g2)) == 64 * 32
        @test eltype(g2) == SVector{3,Float64}
        g3 = Grid(6.0, 4.0, 2.0, 16, 8, 4)
        @test size(g3) == (16, 8, 4)
        @test ndims(g3) == 3
    end

    @testset "Kottke kernel values" begin
        # isotropic sanity limit: equal materials -> smoothing returns the same tensor
        ε1 = Matrix(3.0I, 3, 3)
        S = normcart([1.0, 0.0, 0.0])
        @test avg_param(ε1, ε1, S, 0.3) ≈ ε1
        # r₁ → 1 limit returns ε₁; r₁ → 0 limit returns ε₂
        ε2 = Matrix(Diagonal([5.0, 4.0, 4.5]))
        @test avg_param(ε1, ε2, S, 1.0) ≈ ε1
        @test avg_param(ε1, ε2, S, 0.0) ≈ ε2
        # generated kernel f_εₑᵣ matches avg_param_rot
        r1 = 0.4
        x0 = vcat(r1, vec(ε1), vec(ε2))
        @test reshape(DielectricSmoothing.f_εₑᵣ(x0), (3, 3)) ≈ avg_param_rot(ε1, ε2, r1)
        # diagonal-anisotropic, axis-aligned interface: perpendicular component is the
        # harmonic mean, parallel components the arithmetic mean (classic Kottke/MEEP limit)
        εsm = avg_param(ε1, ε2, normcart([0.0, 1.0, 0.0]), 0.5)
        harm(a, b) = 2 / (1 / a + 1 / b)
        @test εsm[2, 2] ≈ harm(3.0, 4.0) rtol = 1e-9
        @test εsm[1, 1] ≈ (3.0 + 5.0) / 2 rtol = 1e-9
        @test εsm[3, 3] ≈ (3.0 + 4.5) / 2 rtol = 1e-9
    end

    @testset "Kottke kernel gradients" begin
        ε1 = Matrix(Diagonal([4.41, 4.41, 4.6]))
        ε2 = Matrix(2.1I, 3, 3)
        x0 = vcat(0.37, vec(ε1), vec(ε2))
        loss = x -> sum(abs2, DielectricSmoothing.f_εₑᵣ(x))
        g_ref = FiniteDifferences.grad(central_fdm(5, 1), loss, x0)[1]
        # symbolic Jacobian reference
        fj0 = DielectricSmoothing.fj_εₑᵣ(x0)
        @test transpose(fj0[:, 2:end]) * (2 .* fj0[:, 1]) ≈ g_ref rtol = 1e-6
        for (name, backend) in merge(backends_reverse, backends_forward)
            @testset "$name" begin
                g = DI.gradient(loss, backend, x0)
                @test g ≈ g_ref rtol = 1e-6
            end
        end
    end

    mats = [Si₃N₄, SiO₂, MgO_LiNbO₃]
    n_mats = length(mats) + 1 # +1 for vacuum background
    f_ε, _ = _f_ε_mats(mats, (:ω,))
    ω0 = 1 / 1.55
    mat_vals0 = hcat(f_ε([ω0]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))) # append vacuum
    p_geom0 = [1.7, 0.7, π / 14.0, 0.2]
    shapes0 = ridge_wg(p_geom0)
    minds0 = (1, 2, 3, 4)
    grid = Grid(6.0, 4.0, 32, 24)

    @testset "smooth_ε primal" begin
        sm = smooth_ε(shapes0, mat_vals0, minds0, grid)
        @test size(sm) == (3, 3, 3, 32, 24)
        εg = view(sm, :, :, 1, :, :)
        # pixel deep inside the substrate (material 3, MgO:LiNbO₃): smoothed ε equals material ε
        ε_subs = ε_views(mat_vals0[:, 3], 1) |> first |> first
        @test εg[:, :, 16, 4] ≈ ε_subs rtol = 1e-9
        # pixel deep inside the core (material 1, Si₃N₄)
        ε_SiN = ε_views(mat_vals0[:, 1], 1) |> first |> first
        @test εg[:, :, 16, 13] ≈ ε_SiN rtol = 1e-6
        # all smoothed diagonal entries bounded by material extremes
        diags = [εg[i, i, ix, iy] for i in 1:3, ix in 1:32, iy in 1:24]
        @test minimum(diags) >= 1.0 - 1e-9
        @test maximum(diags) <= maximum([mat_vals0[i, j] for i in (1, 5, 9), j in 1:4]) + 1e-9
        # smoothed tensors symmetric
        @test maximum(abs, εg .- permutedims(εg, (2, 1, 3, 4))) < 1e-9
    end

    @testset "smooth_ε_single (interface voxel) reverse-mode gradients" begin
        # find a voxel whose corners straddle a material interface, so the full Kottke
        # smoothing path (surface normal, fill fraction, tensor rotation) is exercised
        crnrs_list = corners(grid)
        idx = findfirst(crnrs_list) do crnrs
            ps = proc_sinds(corner_sinds(shapes0, crnrs))
            !iszero(ps[2]) && iszero(ps[3])
        end
        @test !isnothing(idx)
        crnrs = crnrs_list[idx]
        loss_mv = mv -> sum(abs2, smooth_ε_single(shapes0, mv, minds0, crnrs))
        g_ref = FiniteDifferences.grad(central_fdm(5, 1), loss_mv, mat_vals0)[1]
        for (name, backend) in backends_reverse
            @testset "$name" begin
                g = DI.gradient(loss_mv, backend, mat_vals0)
                @test g ≈ g_ref rtol = 1e-5
            end
        end
        # gradient w.r.t. geometry parameters through surfpt_nearby/volfrac
        loss_p = p -> sum(abs2, smooth_ε_single(ridge_wg(p), mat_vals0, minds0, crnrs))
        gp_ref = FiniteDifferences.grad(central_fdm(5, 1), loss_p, p_geom0)[1]
        @test any(!iszero, gp_ref)  # interface voxel: geometry must matter
        for (name, backend) in backends_reverse
            @testset "$name" begin
                gp = DI.gradient(loss_p, backend, p_geom0)
                @test gp ≈ gp_ref rtol = 1e-4
            end
        end
    end

    @testset "smooth_ε full-pipeline forward-mode gradients" begin
        # Forward mode (ForwardDiff Duals) traverses the entire smoothing pipeline.
        # (Reverse mode through the full `mapreduce` pipeline is intentionally not tested
        # here: deriving reverse rules for the whole 768-voxel program takes Mooncake and
        # Enzyme impractically long to compile. The adjoint-relevant smoothing physics is
        # covered voxel-wise by the reverse-mode tests above.)
        loss_mv = mv -> sum(abs2, smooth_ε(shapes0, mv, minds0, grid))
        g_ref = FiniteDifferences.grad(central_fdm(3, 1), loss_mv, mat_vals0)[1]
        @test DI.gradient(loss_mv, AutoForwardDiff(), mat_vals0) ≈ g_ref rtol = 1e-5

        loss_p = p -> sum(abs2, smooth_ε(ridge_wg(p), mat_vals0, minds0, grid))
        gp_ref = FiniteDifferences.grad(central_fdm(3, 1), loss_p, p_geom0)[1]
        @test DI.gradient(loss_p, AutoForwardDiff(), p_geom0) ≈ gp_ref rtol = 1e-4
    end
end
