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

# Modal group index and group-velocity dispersion from a *single* eigenmode solution, for
# a thin-film lithium niobate waveguide with realistic anisotropic, dispersive materials —
# the calculation behind Gray, West & Ram, "Inverse design for waveguide dispersion with a
# differentiable mode solver," Opt. Express 32, 30541 (2024).
#
# `group_index` evaluates the modal group index n_g = ∂|k|/∂ω directly from one mode field
# (paper Eq. 12, ratio of energy density to Poynting flux), and `ng_gvd`/`ng_gvd_E` add the
# GVD = ∂n_g/∂ω = ∂²|k|/∂ω² via a single adjoint solve (Supplement 1) using the smoothed
# second frequency derivative ∂²ε/∂ω² of the dielectric tensor. The paper's central claim
# for these formulas is that they reproduce the dispersion that finite differences of full
# re-solves give — without the finite-difference truncation error. This testset verifies
# exactly that for an x-cut LiNbO₃ rib on SiO₂: the single-solution n_g and GVD match
# high-order finite differences of the fully re-solved |k|(ω) and n_g(ω).
@testset "modal GVD from a single mode solution (TFLN, anisotropic dispersive)" begin
    # x-cut TFLN: rotate the bundled LiNbO₃ (extraordinary/c axis along z) by RotY(π/2) so
    # the c-axis lies in-plane along x; the quasi-TE mode (dominant Eₓ) then sees nₑ.
    Ry = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
    LiNbO₃_xcut = rotate(LiNbO₃, Ry; name=:LiNbO₃_xcut)
    mats = [LiNbO₃_xcut, SiO₂]
    fε, _ = _f_ε_mats(mats, (:ω,))
    air_col = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))     # air cladding: ε = I, no dispersion
    matvals(om) = hcat(fε([om]), air_col)                  # 27 × 3 columns (LiNbO₃, SiO₂, air)

    grid = Grid(6.0, 4.0, 48, 36)
    solver = KrylovKitEigsolve()
    # LiNbO₃ ridge (1.0 × 0.6 μm) on an SiO₂ substrate, air above; shapes foreground-first.
    w, t = 1.0, 0.6
    geom = (
        MaterialShape(Cuboid([0.0, t / 2], [w, t], [1.0 0.0; 0.0 1.0]), 1),         # core
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2), # substrate
    )
    minds = (1, 2, 3)   # core→LiNbO₃, substrate→SiO₂, background→air

    "smoothed dielectric fields (ε⁻¹, ∂ωε, ∂²ωε) at frequency `om`"
    function diel(om)
        sm = smooth_ε(geom, matvals(om), minds, grid)
        (sliceinv_3x3(copy(selectdim(sm, 3, 1))),
            copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
    end

    "fundamental quasi-TE mode (largest Eₓ power fraction) at frequency `om`"
    function te_mode(om)
        ei, de, _ = diel(om)
        ε = sliceinv_3x3(copy(ei))
        kmags, evecs = solve_k(om, copy(ei), grid, solver; nev=4, k_tol=1e-12, eig_tol=1e-12)
        fracs = [E_relpower_xyz(ε,
            E⃗(kmags[i], copy(evecs[i]), ei, de, grid; canonicalize=true, normalized=true))[1]
                 for i in eachindex(evecs)]
        i = argmax(fracs)
        return kmags[i], evecs[i], fracs[i]
    end

    ω0 = 1 / 1.55
    ei0, de0, dde0 = diel(ω0)
    k0, ev0, tefrac0 = te_mode(ω0)

    @test k0 / ω0 > sqrt(2.09)        # guided above the SiO₂ substrate index (n ≈ 1.45)
    @test tefrac0 > 0.8               # genuinely quasi-TE (Eₓ-dominant), so it sees nₑ

    # finite-difference references from full re-solves with the real material dispersion
    k_of_ω(om) = te_mode(om)[1]
    function ng_of_ω(om)
        ei, de, _ = diel(om)
        k, ev, _ = te_mode(om)
        group_index(k, ev, om, ei, de, grid)
    end

    ng_direct = group_index(k0, ev0, ω0, ei0, de0, grid)
    ng, gvd = ng_gvd(ω0, k0, ev0, ei0, de0, dde0, grid)

    # n_g (Eq. 12) reproduces both the FD slope d|k|/dω and the ng_gvd value
    ng_FD = FiniteDifferences.central_fdm(5, 1; factor=1e6)(k_of_ω, ω0)
    @test ng_direct ≈ ng_FD rtol = 1e-4
    @test ng ≈ ng_direct rtol = 1e-6
    @test ng > k0 / ω0                # group index exceeds phase index in this dispersive WG

    # GVD (single adjoint solve) reproduces the FD frequency-derivative of n_g. Strong
    # waveguide dispersion makes ∂n_g/∂ω negative (anomalous) for this geometry at 1.55 μm.
    gvd_FD = FiniteDifferences.central_fdm(5, 1; factor=1e6)(ng_of_ω, ω0)
    @test isfinite(gvd)
    @test sign(gvd) == sign(gvd_FD)
    @test gvd ≈ gvd_FD rtol = 5e-3

    # ng_gvd_E returns the same n_g and GVD plus the real-space E field
    ngE, gvdE, E = ng_gvd_E(ω0, k0, ev0, ei0, de0, dde0, grid)
    @test ngE ≈ ng
    @test gvdE ≈ gvd
    @test size(E) == (3, size(grid)...)
end
