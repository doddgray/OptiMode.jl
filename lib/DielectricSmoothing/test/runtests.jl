using Test
using LinearAlgebra
using StaticArrays
using GeometryPrimitives
using MaterialDispersion
using DielectricSmoothing
using ChainRulesCore: rrule
using FiniteDifferences
using ForwardDiff
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme
using Mooncake
using Zygote

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
        # the dispersion-propagating kernel's value slice matches the plain Kottke average
        r1 = 0.4
        sm = εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r1, vcat(vec(ε1), zeros(18)), vcat(vec(ε2), zeros(18)))
        @test reshape(sm[1:9], (3, 3)) ≈ avg_param_rot(ε1, ε2, r1)
        # diagonal-anisotropic, axis-aligned interface: perpendicular component is the
        # harmonic mean, parallel components the arithmetic mean (classic Kottke/MEEP limit)
        εsm = avg_param(ε1, ε2, normcart([0.0, 1.0, 0.0]), 0.5)
        harm(a, b) = 2 / (1 / a + 1 / b)
        @test εsm[2, 2] ≈ harm(3.0, 4.0) rtol = 1e-9
        @test εsm[1, 1] ≈ (3.0 + 5.0) / 2 rtol = 1e-9
        @test εsm[3, 3] ≈ (3.0 + 4.5) / 2 rtol = 1e-9
    end

    @testset "Kottke kernel gradients" begin
        # Material-data gradient of the smoothed tensor through the closed-form Kottke
        # average. Every backend (forward & reverse) differentiates the small, type-stable
        # `avg_param_rot` kernel directly — no symbolic Jacobian needed.
        ε1 = Matrix(Diagonal([4.41, 4.41, 4.6]))
        ε2 = Matrix(2.1I, 3, 3)
        r1 = 0.37
        x0 = vcat(vec(ε1), vec(ε2))   # 18 material-tensor entries (r₁ held fixed)
        loss = x -> sum(abs2, vec(avg_param_rot(reshape(x[1:9], 3, 3), reshape(x[10:18], 3, 3), r1)))
        g_ref = FiniteDifferences.grad(central_fdm(5, 1), loss, x0)[1]
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

    @testset "smooth_scalar (per-material scalar maps, e.g. Kerr n₂)" begin
        # per-material scalar values, e.g. Kerr coefficients n₂ [μm²/W]; vacuum bg = 0
        vals = [2.4e-7, 1.0e-8, 5.0e-9, 0.0]
        n2 = smooth_scalar(shapes0, vals, minds0, grid)
        @test size(n2) == size(grid)
        # pixels deep inside a single material take exactly that material's value
        @test n2[16, 13] ≈ vals[1] rtol = 1e-9   # core (same pixel as smooth_ε primal test)
        @test n2[16, 4] ≈ vals[3] rtol = 1e-9    # substrate
        # all smoothed values bounded by the material extremes …
        @test all(v -> minimum(vals) - 1e-15 <= v <= maximum(vals) + 1e-15, n2)
        # … and interface pixels are strict volume-fraction mixtures
        @test any(v -> minimum(vals) + 1e-12 < v < maximum(vals) - 1e-12, n2)
    end

    @testset "Kottke smoothing-with-dispersion kernel gradients" begin
        S = normcart(normalize([0.3, 0.9, 0.1]))
        v1, v2 = mat_vals0[:, 1], mat_vals0[:, 2]
        # The dispersion kernels carry a 2nd-order Taylor jet through the closed-form Kottke
        # average (no symbolic Jacobian/Hessian), so they are small and type-stable and
        # every backend — forward and reverse — differentiates them directly.
        allbackends = merge(backends_reverse, backends_forward)
        # first-derivative propagation kernel (εₑ_∂ωεₑ): value + ∂ωεₑ
        x0 = vcat(0.37, v1[1:18], v2[1:18])
        loss = x -> sum(abs2, εₑ_∂ωεₑ(x[1], S, x[2:19], x[20:37]))
        g_ref = FiniteDifferences.grad(central_fdm(5, 1), loss, x0)[1]
        for (name, backend) in allbackends
            @testset "∂ωεₑ kernel: $name" begin
                @test DI.gradient(loss, backend, x0) ≈ g_ref rtol = 1e-5
            end
        end
        # second-derivative propagation kernel (εₑ_∂ωεₑ_∂²ωεₑ): value + ∂ωεₑ + ∂²ωεₑ
        x2 = vcat(0.37, v1, v2)
        loss2 = x -> sum(abs2, εₑ_∂ωεₑ_∂²ωεₑ(x[1], S, x[2:28], x[29:55]))
        g2_ref = FiniteDifferences.grad(central_fdm(5, 1), loss2, x2)[1]
        for (name, backend) in allbackends
            @testset "∂²ωεₑ kernel: $name" begin
                @test DI.gradient(loss2, backend, x2) ≈ g2_ref rtol = 1e-5
            end
        end
    end

    @testset "smooth_ε full-pipeline material gradients (all backends)" begin
        # Material-data gradients traverse the entire smoothing pipeline. The Kottke kernel
        # propagates a small, type-stable Taylor jet (no symbolic kernel), so the smoothing
        # differentiates in forward mode (ForwardDiff, Enzyme-forward) and reverse mode
        # (Zygote via the `smooth_ε` rrule; Enzyme-reverse natively, with the heterogeneous
        # shape-tuple index isolated in the inactive `_interface_geometry`).
        loss_mv = mv -> sum(abs2, smooth_ε(shapes0, mv, minds0, grid))
        g_ref = FiniteDifferences.grad(central_fdm(3, 1), loss_mv, mat_vals0)[1]
        # ForwardDiff & Zygote are exact on the full material gradient.
        for (name, backend) in (("ForwardDiff", AutoForwardDiff()), ("Zygote", AutoZygote()))
            @testset "$name" begin
                @test DI.gradient(loss_mv, backend, mat_vals0) ≈ g_ref rtol = 1e-5
            end
        end
        # Enzyme (fwd & rev) on Julia 1.11 + Enzyme 0.13.168: isolating the heterogeneous
        # `shapes[sidx1]` index in the inactive `_interface_geometry` fixes the
        # `IllegalTypeAnalysisException`, and Enzyme is then exact for every *real* material.
        # It still mis-accumulates the gradient w.r.t. the *vacuum* background column's
        # frequency-dispersion entries (∂ωε/∂²ωε, structurally zero and never optimization
        # variables) — an upstream Enzyme regression (ForwardDiff/Zygote/Mooncake are exact
        # there). Validate Enzyme on the real-material columns; track the vacuum entry as
        # broken so it flags if a future Enzyme fixes it.
        real_cols = 1:(n_mats - 1)
        for (name, backend) in (
                ("Enzyme(reverse)", AutoEnzyme(; mode=set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)),
                ("Enzyme(forward)", AutoEnzyme(; mode=set_runtime_activity(Enzyme.Forward), function_annotation=Enzyme.Const)))
            @testset "$name" begin
                g = DI.gradient(loss_mv, backend, mat_vals0)
                @test g[:, real_cols] ≈ g_ref[:, real_cols] rtol = 1e-5
                @test_broken g ≈ g_ref rtol = 1e-5   # vacuum-dispersion entry: upstream Enzyme 0.13 limitation
            end
        end

        # Geometry-parameter sensitivities: with GeometryPrimitives ≥ 0.6 the shape
        # element type is parametric and the geometric queries (`surfpt_nearby`,
        # `volfrac`) are AD-compatible, so AD number types flow through shape
        # construction. Forward mode (ForwardDiff) propagates through the full
        # geometry→smoothing pipeline; verify it against finite differences.
        loss_p = p -> sum(abs2, smooth_ε(ridge_wg(p), mat_vals0, minds0, grid))
        gp_ref = FiniteDifferences.grad(central_fdm(5, 1), loss_p, p_geom0)[1]
        @test any(!iszero, gp_ref)
        @test all(isfinite, gp_ref)
        @test DI.gradient(loss_p, AutoForwardDiff(), p_geom0) ≈ gp_ref rtol = 1e-4

        # Enzyme (fwd & rev) geometry gradient, on `ridge_wg`'s *heterogeneous* shapes tuple
        # (Polygon + 2×Cuboid) — the case `_interface_geometry`'s `shapes[sidx1]` indexing used
        # to raise `IllegalTypeAnalysisException` for, before `surfpt_nearby`/`volfrac`/
        # `_interface_geometry` stopped being marked `EnzymeRules.inactive` (that marking is
        # only valid for material-data-only differentiation; it silently zeroed geometry
        # gradients otherwise). Now exact, matching ForwardDiff/finite differences.
        for (name, backend) in (
                ("Enzyme(reverse)", AutoEnzyme(; mode=set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)),
                ("Enzyme(forward)", AutoEnzyme(; mode=set_runtime_activity(Enzyme.Forward), function_annotation=Enzyme.Const)))
            @testset "$name" begin
                @test DI.gradient(loss_p, backend, p_geom0) ≈ gp_ref rtol = 1e-4
            end
        end
    end

    @testset "geometry-parameter gradients (Kottke kernel)" begin
        # Geometry parameters enter the smoothing through the interface queries
        # `surfpt_nearby` (surface point + normal) and `volfrac` (fill fraction) feeding
        # the Kottke kernel. At one interface pixel — held fixed in space while the shape
        # boundary sweeps across it — these compose into a clean, union-free function that
        # reverse-mode Mooncake handles (in addition to forward-mode ForwardDiff). The
        # full-pipeline `mapreduce` over a union of pixel branches is forward-mode /
        # finite-difference only for the reverse backends, as for material data.
        ε1 = SMatrix{3,3}(reshape(mat_vals0[1:9, 1], 3, 3))   # Si₃N₄ tensor
        ε2 = SMatrix{3,3}(reshape(mat_vals0[1:9, 2], 3, 3))   # SiO₂ tensor
        xyz = SVector(0.8, 0.0)                               # fixed pixel center
        vmin = SVector(0.75, -0.05)
        vmax = SVector(0.85, 0.05)
        # (w, h, cx): the core's right edge cx + w/2 sweeps the fixed pixel
        function kernel_geom(p)
            w, h, cx = p
            core = GeometryPrimitives.Cuboid(SVector(cx, 0.0), SVector(w, h),
                SMatrix{2,2}(1.0, 0.0, 0.0, 1.0))
            r = GeometryPrimitives.surfpt_nearby(xyz, core)
            rvol = GeometryPrimitives.volfrac((vmin, vmax), last(r), first(r))
            return sum(abs2, avg_param(ε1, ε2, normcart(DielectricSmoothing.vec3D(last(r))), rvol))
        end
        pk0 = [1.6, 0.8, 0.0]
        gk_ref = FiniteDifferences.grad(central_fdm(5, 1), kernel_geom, pk0)[1]
        @test any(!iszero, gk_ref)                            # nonzero edge sensitivity
        @test DI.gradient(kernel_geom, AutoForwardDiff(), pk0) ≈ gk_ref rtol = 1e-4
        @test DI.gradient(kernel_geom, AutoMooncake(; config=nothing), pk0) ≈ gk_ref rtol = 1e-4
        for (name, backend) in (
                ("Enzyme(reverse)", AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)),
                ("Enzyme(forward)", AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const)))
            @testset "$name" begin
                @test DI.gradient(kernel_geom, backend, pk0) ≈ gk_ref rtol = 1e-4
            end
        end
    end

    @testset "smoothing geometry cache (SmoothingPlan)" begin
        # The plan precomputes the frequency-independent geometry scaffold once; applying it
        # to per-ω material data must reproduce the direct `smooth_ε` exactly.
        plan = smoothing_plan(shapes0, minds0, grid)
        @test plan isa SmoothingPlan{2}
        @test size(plan) == size(grid)
        # every pixel is classified into exactly one kind
        nu = count(==(0x01), plan.kind); ni = count(==(0x02), plan.kind); nm = count(==(0x03), plan.kind)
        @test nu + ni + nm == prod(size(grid))
        @test ni > 0                                          # the ridge has interfaces

        sm_direct = smooth_ε(shapes0, mat_vals0, minds0, grid)
        sm_cached = smooth_ε(plan, mat_vals0)
        @test sm_cached == sm_direct                          # identical operations → bit-exact

        # Reuse one plan across frequencies: apply at a second ω and match the direct path.
        ω1 = 1 / 1.31
        mat_vals1 = hcat(f_ε([ω1]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18)))
        @test smooth_ε(plan, mat_vals1) == smooth_ε(shapes0, mat_vals1, minds0, grid)

        # Material-data gradients still flow through the cached apply, and match the direct
        # path (the cache freezes geometry but preserves AD in the material data).
        loss_cached = mv -> sum(abs2, smooth_ε(plan, mv))
        loss_direct = mv -> sum(abs2, smooth_ε(shapes0, mv, minds0, grid))
        g_ref = FiniteDifferences.grad(central_fdm(3, 1), loss_cached, mat_vals0)[1]
        for (name, backend) in (("ForwardDiff", AutoForwardDiff()), ("Zygote", AutoZygote()))
            @testset "$name" begin
                @test DI.gradient(loss_cached, backend, mat_vals0) ≈ g_ref rtol = 1e-5
            end
        end
        @test DI.gradient(loss_cached, AutoZygote(), mat_vals0) ≈
              DI.gradient(loss_direct, AutoZygote(), mat_vals0) rtol = 1e-9
    end

    @testset "preallocated / threaded assembly" begin
        # The assembly fills a preallocated (27,N) buffer instead of mapreduce(vcat). The
        # threaded fill must be bit-identical to the serial fill (disjoint columns), for both
        # the geometry (shapes) and cached (plan) entry points.
        plan = smoothing_plan(shapes0, minds0, grid)
        @test smooth_ε(shapes0, mat_vals0, minds0, grid; threaded=true) ==
              smooth_ε(shapes0, mat_vals0, minds0, grid; threaded=false)
        @test smooth_ε(plan, mat_vals0; threaded=true) == smooth_ε(plan, mat_vals0; threaded=false)
        @info "assembly threads" nthreads = Threads.nthreads()

        # The reverse rule's threaded pixel-VJP accumulation must match the serial one.
        Ȳ = randn(size(smooth_ε(plan, mat_vals0)))
        _, pb_s = rrule(smooth_ε, plan, mat_vals0; threaded=false)
        _, pb_t = rrule(smooth_ε, plan, mat_vals0; threaded=true)
        @test pb_s(Ȳ)[3] ≈ pb_t(Ȳ)[3] rtol = 1e-12

        # Scale sanity: a larger grid assembles, stays finite, and matches plan vs direct.
        big = Grid(6.0, 4.0, 192, 160)            # ~31k pixels
        planbig = smoothing_plan(shapes0, minds0, big)
        smbig = smooth_ε(shapes0, mat_vals0, minds0, big; threaded=true)
        @test size(smbig) == (3, 3, 3, 192, 160)
        @test all(isfinite, smbig)
        @test smbig == smooth_ε(planbig, mat_vals0; threaded=true)
    end
end
