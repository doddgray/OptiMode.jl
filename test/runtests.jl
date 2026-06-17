# Umbrella smoke + cross-package integration tests for OptiMode. The substantive unit,
# gradient-correctness, and benchmark suites live with the component packages:
#   lib/MaterialDispersion/test, lib/DielectricSmoothing/test,
#   lib/MaxwellEigenmodes/test,  lib/ModeAnalysis/test
# End-to-end geometry-parameter sensitivities (which span all of them) live here.
using Test
using LinearAlgebra
using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModeAnalysis: Zygote
using ForwardDiff
using FiniteDifferences

@testset "OptiMode umbrella" begin
    # re-exports from all four component packages resolve
    @test OptiMode.SiO₂ isa OptiMode.Material            # MaterialDispersion
    @test OptiMode.Grid(6.0, 4.0, 16, 16) isa OptiMode.Grid  # DielectricSmoothing
    @test OptiMode.KrylovKitEigsolve() isa OptiMode.AbstractEigensolver  # MaxwellEigenmodes
    @test OptiMode.group_index isa Function              # ModeAnalysis

    # minimal end-to-end pipeline: dispersion → smoothing → eigensolve → analysis
    mats = [Si₃N₄, SiO₂]
    f_ε, _ = _f_ε_mats(mats, (:ω,))
    ω = 1 / 1.55
    mat_vals = hcat(f_ε([ω]), vcat(vec([1.0 0 0; 0 1.0 0; 0 0 1.0]), zeros(18)))
    grid = Grid(4.0, 3.0, 16, 12)
    core = MaterialShape(OptiMode.DielectricSmoothing.GeometryPrimitives.Cuboid([0.0, 0.0], [1.6, 0.7], [1.0 0.0; 0.0 1.0]), 1)
    shapes = (core,)
    minds = (1, 2)
    sm = smooth_ε(shapes, mat_vals, minds, grid)
    @test size(sm) == (3, 3, 3, 16, 12)
    epsi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
    kmags, evecs = solve_k(ω, epsi, grid, KrylovKitEigsolve(); nev=1)
    @test sqrt(1.444^2) < kmags[1] / ω < 2.1
    deps_dom = copy(selectdim(sm, 3, 2))
    ng = group_index(kmags[1], evecs[1], ω, epsi, deps_dom, grid)
    @test kmags[1] / ω < ng < 3.0
end

# Sensitivities of the mode quantities (effective index, group index, GVD, mode field)
# with respect to waveguide *geometry* parameters, end to end across the whole stack:
#
#   p (geometry) ──ForwardDiff──▶ ε⁻¹, ∂ωε, ∂²ωε ──Zygote adjoint──▶ neff / ng / field
#
# The geometry → smoothed-dielectric map is differentiated in *forward* mode (Duals flow
# through GeometryPrimitives' parametric shapes and the Kottke smoothing — enabled by the
# doddgray/GeometryPrimitives.jl `claude/geometry-gradient-ad-no6zct` branch), while the
# expensive eigensolve and post-processing are differentiated in *reverse* mode via the
# adjoint-method `rrule`s (`solve_k`, `group_index`). Composing the two by the chain rule
# gives the exact geometry gradient, verified against finite differences of the full
# pipeline. This is the standard adjoint pattern for waveguide inverse design.
@testset "geometry-parameter sensitivities (forward + reverse AD)" begin
    ω = 1 / 1.55
    grid = Grid(4.0, 3.0, 24, 18)
    solver = KrylovKitEigsolve()
    mats = [Si₃N₄, SiO₂]
    fε, _ = _f_ε_mats(mats, (:ω,))
    matvals(om) = hcat(fε([om]), vcat(vec(Matrix(1.0I, 3, 3)), zeros(18)))
    # geometry: a Si₃N₄ core of width/height p = (w, h) in SiO₂, centered
    geom(p) = (MaterialShape(Cuboid([0.0, 0.0], [p[1], p[2]], [1.0 0.0; 0.0 1.0]), 1),)
    function diel(p, om)
        sm = smooth_ε(geom(p), matvals(om), (1, 2), grid)
        (sliceinv_3x3(copy(selectdim(sm, 3, 1))),
            copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
    end
    p0 = [1.6, 0.8]
    ei0, de0, dde0 = diel(p0, ω)
    Npix = length(vec(ei0))

    # forward-mode Jacobian of the dielectric fields w.r.t. geometry (FFT-free; the GP
    # AD branch lets Duals flow through shape construction + Kottke smoothing)
    dflat(q) = (d = diel(q, ω); vcat(vec(d[1]), vec(d[2]), vec(d[3])))
    J = ForwardDiff.jacobian(dflat, p0)
    @test all(isfinite, J) && any(!iszero, J)
    Jei, Jde, Jdde = J[1:Npix, :], J[Npix+1:2Npix, :], J[2Npix+1:3Npix, :]
    # chain rule: ∇ₚ q = Jᵀ · (∂q/∂diel), with the cotangents from reverse-mode AD
    hybrid(ḡei, ḡde, ḡdde) = Jei' * vec(ḡei) .+ Jde' * vec(ḡde) .+ Jdde' * vec(ḡdde)

    @testset "effective index n_eff" begin
        neff_diel(ei) = solve_k(ω, ei, grid, solver; nev=1, k_tol=1e-11)[1][1] / ω
        fd = FiniteDifferences.grad(central_fdm(5, 1), q -> neff_diel(diel(q, ω)[1]), p0)[1]
        ḡei = Zygote.gradient(neff_diel, copy(ei0))[1]          # reverse adjoint of solve_k
        @test hybrid(ḡei, zero(de0), zero(dde0)) ≈ fd rtol = 3e-3
    end

    @testset "group index n_g" begin
        function ng_diel(ei, de)
            k, ev = solve_k(ω, ei, grid, solver; nev=1, k_tol=1e-11)
            group_index(k[1], ev[1], ω, ei, de, grid)
        end
        fd = FiniteDifferences.grad(central_fdm(5, 1), q -> (d = diel(q, ω); ng_diel(d[1], d[2])), p0)[1]
        gei, gde = Zygote.gradient(ng_diel, copy(ei0), copy(de0))
        @test hybrid(gei, gde, zero(dde0)) ≈ fd rtol = 1e-2
    end

    @testset "mode field functional ∫|E|²" begin
        function field_diel(ei, de)
            k, ev = solve_k(ω, ei, grid, solver; nev=1, k_tol=1e-11)
            E = E⃗(k[1], copy(ev[1]), ei, de, grid; canonicalize=true, normalized=true)
            real(sum(abs2, E))
        end
        fd = FiniteDifferences.grad(central_fdm(5, 1), q -> (d = diel(q, ω); field_diel(d[1], d[2])), p0)[1]
        gei, gde = Zygote.gradient(field_diel, copy(ei0), copy(de0))
        @test hybrid(gei, gde, zero(dde0)) ≈ fd rtol = 2e-2
    end

    @testset "group-velocity dispersion GVD" begin
        # GVD = ∂n_g/∂ω. `ng_gvd`'s hand-rolled adjoint is not itself reverse-mode
        # differentiable, so the geometry gradient of GVD is obtained as the frequency
        # derivative of the (exact AD) geometry gradient of n_g: the high-dimensional
        # geometry sensitivity stays exact AD, only the scalar ω-derivative is finite-differenced.
        function grad_ng(p, om)
            e, d, _ = diel(p, om)
            N = length(vec(e))
            Jf = ForwardDiff.jacobian(q -> (dd = diel(q, om); vcat(vec(dd[1]), vec(dd[2]))), p)
            function ngf(ei, de)
                k, ev = solve_k(om, ei, grid, solver; nev=1, k_tol=1e-11)
                group_index(k[1], ev[1], om, ei, de, grid)
            end
            ge, gd = Zygote.gradient(ngf, copy(e), copy(d))
            Jf[1:N, :]' * vec(ge) .+ Jf[N+1:2N, :]' * vec(gd)
        end
        Δ = 1e-3
        gvd_grad_AD = (grad_ng(p0, ω + Δ) .- grad_ng(p0, ω - Δ)) ./ (2Δ)
        function gvd_p(p)
            e, d, dd = diel(p, ω)
            k, ev = solve_k(ω, e, grid, solver; nev=1, k_tol=1e-11)
            ng_gvd(ω, k[1], ev[1], e, d, dd, grid)[2]
        end
        fd = FiniteDifferences.grad(central_fdm(5, 1), gvd_p, p0)[1]
        @test gvd_grad_AD ≈ fd rtol = 5e-2
    end
end
