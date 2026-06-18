using Test
using LinearAlgebra
using StaticArrays
using DielectricSmoothing
using DielectricSmoothing.GeometryPrimitives: Cuboid
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

"""
Rectangular (optionally anisotropic, diagonal) core dielectric built analytically: a
`w × h` μm core of principal refractive indices `(nx, ny, nz)` in an isotropic cladding
of index `nclad`, returned as a `(3, 3, Nx, Ny)` ε array. Used to exercise the
Hermite–Gaussian mode classifier on Si₃N₄- and LiNbO₃-like multimode waveguides without
pulling in the material-dispersion / geometry-smoothing machinery.
"""
function rect_wg_eps(grid::Grid, w, h, nx, ny, nz, nclad)
    Nx, Ny = size(grid)
    eps = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(y(grid)), (ix, xx) in enumerate(x(grid))
        inside = (abs(xx) <= w / 2) && (abs(yy) <= h / 2)
        eps[1, 1, ix, iy] = (inside ? nx : nclad)^2
        eps[2, 2, ix, iy] = (inside ? ny : nclad)^2
        eps[3, 3, ix, iy] = (inside ? nz : nclad)^2
    end
    return eps
end

"""
Solve the lowest `nev` modes of `eps` at `ω` and label every *guided* mode
(``n_eff > n_clad``) two ways: the node-counting classifier (`count_E_nodes`, whose raw
count is twice the Hermite–Gaussian order — each zero crossing flips the field sign) and
the Hermite–Gaussian fit classifier (`hg_mode_label`). Returns a vector of NamedTuples
`(neff, old=(pol,m,n), new=(pol,m,n), relerr, te_frac, label)`.
"""
function guided_mode_labels(eps, grid, ω, nclad; nev=8, max_order=4)
    epsi = sliceinv_3x3(copy(eps))
    km, ev = solve_k(ω, copy(epsi), grid, KrylovKitEigsolve(); nev=nev, k_tol=1e-11, eig_tol=1e-11)
    out = NamedTuple[]
    for i in eachindex(ev)
        neff = km[i] / ω
        neff > nclad + 1e-3 || continue
        E = E⃗(km[i], copy(ev[i]), epsi, epsi, grid; canonicalize=true, normalized=false)
        En = E ./ ModeAnalysis.Eperp_max(E)
        pol = argmax(E_relpower_xyz(eps, En))
        nodes = count_E_nodes(En, eps, pol; rel_amp_min=0.1)
        old = (pol == 1 ? :TE : :TM, Int(nodes[1]) ÷ 2, Int(nodes[2]) ÷ 2)
        nw = hg_mode_label(E, grid; max_order)
        push!(out, (neff=neff, old=old, new=(nw.pol, nw.m, nw.n),
            relerr=nw.rel_error, te_frac=nw.te_frac, label=nw.label))
    end
    return out
end

"""
One material-dispersion column for `smooth_ε`/`solve_k_converged`: an isotropic material
of index `n` with diagonal `∂ε/∂ω = dε` and `∂²ε/∂ω² = ddε`, packed as
`vcat(vec(ε), vec(∂ωε), vec(∂²ωε))` (27 entries). Avoids pulling MaterialDispersion into
the ModeAnalysis test environment while still exercising the geometry→smoothing→solve
pipeline that the forced-convergence loop drives.
"""
function diag_mat_col(n; dε=0.0, ddε=0.0)
    vcat(vec(Matrix(Float64(n)^2 * I, 3, 3)),
        vec(Matrix(Float64(dε) * I, 3, 3)),
        vec(Matrix(Float64(ddε) * I, 3, 3)))
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

    @testset "Kerr power-dependent mode correction" begin
        # Si₃N₄-like rectangular waveguide (1.6 × 0.8 μm core) in SiO₂-like cladding at
        # λ = 1.55 μm, with the standard Kerr coefficients n₂(Si₃N₄) = 2.4e-7 μm²/W and
        # n₂(SiO₂) = 2.6e-8 μm²/W.
        gK = Grid(4.0, 3.0, 32, 24)
        ωK = 1 / 1.55
        n_core, n_clad = 1.996, 1.444
        n2_core, n2_clad = 2.4e-7, 2.6e-8
        w, h = 1.6, 0.8
        NxK, NyK = size(gK)
        epsK = zeros(3, 3, NxK, NyK)
        depsK = zeros(3, 3, NxK, NyK)
        n2map = zeros(NxK, NyK)
        # realistic material dispersion ∂ε/∂ω = 2n·(ng-n)/ω with ng(Si₃N₄)≈2.07, ng(SiO₂)≈1.46
        dε_core = 2 * n_core * (2.07 - n_core) * 1.55
        dε_clad = 2 * n_clad * (1.462 - n_clad) * 1.55
        for (iy, yy) in enumerate(y(gK)), (ix, xx) in enumerate(x(gK))
            inside = (abs(xx) <= w / 2) && (abs(yy) <= h / 2)
            for a in 1:3
                epsK[a, a, ix, iy] = (inside ? n_core : n_clad)^2
                depsK[a, a, ix, iy] = inside ? dε_core : dε_clad
            end
            n2map[ix, iy] = inside ? n2_core : n2_clad
        end
        epsiK = sliceinv_3x3(epsK)

        # P = 0 reproduces the linear solve exactly
        res0 = solve_k_kerr(ωK, 0.0, epsiK, depsK, n2map, gK, solver; nev=1, k_tol=1e-10)
        @test res0.kmags == res0.kmags_lin
        @test res0.evecs[1] == res0.evecs_lin[1]
        @test res0.dn_max == [0.0]
        k_lin, ev_lin = res0.kmags_lin[1], res0.evecs_lin[1]
        @test k_lin / ωK > n_clad   # guided

        # modal intensity is normalized to carry the full power: ∫ I dA = P
        P = 5.0
        I = mode_intensity(k_lin, ev_lin, epsiK, gK, P)
        @test sum(I) * δV(gK) ≈ P rtol = 1e-12
        @test maximum(I) * δV(gK) < P / 10   # power spread over many pixels, none dominant

        # the Kerr perturbation is diagonal, nonnegative, and Δε = 2 n₀ Δn
        Δε, Δn = kerr_dielectric_perturbation(I, n2map, epsK)
        @test minimum(Δn) >= -1e-6 * maximum(Δn)   # ≥ 0 up to spectral ringing
        @test maximum(Δn) ≈ n2_core * maximum(I) rtol = 1e-12
        @test Δε[1, 1, 16, 12] ≈ 2 * n_core * Δn[16, 12] rtol = 1e-12
        @test iszero(Δε[1, 2, :, :])

        # power-corrected solves: Δneff > 0 and ∝ P to first order
        res1 = solve_k_kerr(ωK, P, epsiK, depsK, n2map, gK, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)
        res2 = solve_k_kerr(ωK, 2P, epsiK, depsK, n2map, gK, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)
        Δneff1 = (res1.kmags[1] - res1.kmags_lin[1]) / ωK
        Δneff2 = (res2.kmags[1] - res2.kmags_lin[1]) / ωK
        @test Δneff1 > 0
        @test res1.dn_max[1] > 0
        @test Δneff2 ≈ 2 * Δneff1 rtol = 5e-2
        @test res2.dn_max[1] ≈ 2 * res1.dn_max[1] rtol = 1e-9

        # magnitude agrees with the textbook SPM estimate Δneff ≈ n₂(core)·P/Aeff with
        # the standard nonlinear effective area Aeff = (∫I dA)² / ∫I² dA
        AeffK = P^2 / (sum(abs2, I) * δV(gK))
        @test 0.3 < AeffK < 3.0   # μm²-scale mode
        @test 0.2 < Δneff1 / (n2_core * P / AeffK) < 1.5

        # adjoint consistency: the Kerr Δk matches ⟨∂k/∂ε⁻¹, Δ(ε⁻¹)⟩ from the
        # solve_k adjoint rule (the perturbation is small enough to be linear)
        P10 = 10.0
        I10 = mode_intensity(k_lin, ev_lin, epsiK, gK, P10)
        Δε10, _ = kerr_dielectric_perturbation(I10, n2map, epsK)
        epsi_NL = sliceinv_3x3(epsK .+ Δε10)
        kref = solve_k(ωK, copy(epsiK), gK, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)[1][1]
        k_NL = solve_k(ωK, copy(epsi_NL), gK, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)[1][1]
        g_ei = Zygote.gradient(
            ei -> solve_k(ωK, ei, gK, solver; nev=1, k_tol=1e-12, eig_tol=1e-12)[1][1],
            copy(epsiK))[1]
        @test dot(g_ei, epsi_NL .- epsiK) ≈ k_NL - kref rtol = 5e-2
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

    # Alternative mode-classification scheme: instead of node-counting, label a mode by
    # which elliptical Hermite–Gaussian template (order (m,n), polarization TE/TM) best
    # fits its transverse field in a least-squares sense (`hg_mode_label`). This is
    # threshold-free and returns a quantitative goodness-of-fit (`rel_error`). The tests
    # below verify (i) the Hermite-polynomial primitive, (ii) exact recovery of synthetic
    # Hermite–Gaussian fields, and (iii) agreement with the node-counting classifier on
    # the guided modes of Si₃N₄- and LiNbO₃-core multimode waveguides.
    @testset "Hermite–Gaussian mode labeling" begin
        # (i) physicist's Hermite polynomials Hₙ(x): H₀=1, H₁=2x, H₂=4x²−2, H₃=8x³−12x
        @test hermite_H(0, 0.3) == 1.0
        @test hermite_H(1, 0.3) ≈ 0.6
        @test hermite_H(2, 0.3) ≈ 4 * 0.09 - 2
        @test hermite_H(3, 0.3) ≈ 8 * 0.027 - 12 * 0.3
        @test hermite_H(2, [0.0, 1.0]) ≈ [-2.0, 2.0]   # broadcastable form
        @test hg_label_string(:TE, 0, 0) == "TE₀₀"
        @test hg_label_string(:TM, 2, 1) == "TM₂₁"

        # (ii) synthetic recovery: an exact elliptical Hermite–Gaussian placed in one
        # transverse polarization (plus a small π/2-shifted longitudinal component) must
        # be labeled with its true polarization and order at near-zero residual.
        gsyn = Grid(8.0, 6.0, 80, 60)
        xs, ys = x(gsyn), y(gsyn)
        for (mt, nt, pol) in [(0, 0, 1), (1, 0, 1), (2, 1, 1), (0, 2, 2), (3, 0, 2)]
            ψ = hg_template(mt, nt, xs, ys, 0.2, -0.1, 1.3, 1.0)
            E = zeros(ComplexF64, 3, 80, 60)
            E[pol, :, :] .= ψ
            E[3, :, :] .= 0.03im .* ψ
            lbl = hg_mode_label(E, gsyn; max_order=4)
            @test lbl.pol == (pol == 1 ? :TE : :TM)
            @test (lbl.m, lbl.n) == (mt, nt)
            @test lbl.rel_error < 0.02
            # the analytic centroid/width seed is recovered by the fit
            @test isapprox(lbl.fit.x₀, 0.2; atol=0.05)
            @test isapprox(lbl.fit.y₀, -0.1; atol=0.05)
        end

        # (iii) Si₃N₄ multimode waveguide: 2.5 × 0.7 μm, n_core = 2.0, n_clad = 1.444.
        gSi = Grid(7.0, 4.0, 112, 64)
        epsSi = rect_wg_eps(gSi, 2.5, 0.7, 2.0, 2.0, 2.0, 1.444)
        labsSi = guided_mode_labels(epsSi, gSi, ω0, 1.444; nev=8)
        @test length(labsSi) >= 6                                  # genuinely multimode
        labelset_Si = Set(l.new for l in labsSi)
        @test (:TE, 0, 0) in labelset_Si                           # quasi-TE₀₀ guided
        @test (:TM, 0, 0) in labelset_Si                           # quasi-TM₀₀ guided
        # HG-fit labels match the (factor-of-two-corrected) node-counting labels exactly
        @test all(l.old == l.new for l in labsSi)
        # the fundamental is an excellent Hermite–Gaussian; te_frac cleanly splits TE/TM
        @test labsSi[1].relerr < 0.05
        @test all(l.new[1] == :TE ? l.te_frac > 0.8 : l.te_frac < 0.2 for l in labsSi)

        # (iii) x-cut LiNbO₃-like ridge: anisotropic core nₑ = 2.14 ∥ x, nₒ = 2.21 ∥ y,z,
        # 1.4 × 0.6 μm, n_clad = 1.444 — quasi-TE sees nₑ, quasi-TM sees nₒ.
        gLN = Grid(7.0, 4.0, 112, 64)
        epsLN = rect_wg_eps(gLN, 1.4, 0.6, 2.14, 2.21, 2.21, 1.444)
        labsLN = guided_mode_labels(epsLN, gLN, ω0, 1.444; nev=8)
        @test length(labsLN) >= 4
        labelset_LN = Set(l.new for l in labsLN)
        @test (:TE, 0, 0) in labelset_LN
        @test (:TM, 0, 0) in labelset_LN
        @test all(l.old == l.new for l in labsLN)
        @test all(l.new[1] == :TE ? l.te_frac > 0.8 : l.te_frac < 0.2 for l in labsLN)
    end

    # Forced grid-convergence mode solving: re-run the whole geometry → sub-pixel
    # smoothing → eigensolve pipeline on a sequence of grids, increasing the spatial
    # point density and the waveguide-center → boundary distance each iteration until the
    # mode effective indices stop changing to within atol/rtol. A 1.2 × 0.5 μm core of
    # index 2.0 (∂ε/∂ω diagonal) in an n = 1.444 cladding at λ = 1.55 μm. Grids are kept
    # deliberately small (≤ ~48×32) so the repeated eigensolves stay fast.
    @testset "forced grid convergence" begin
        mat_vals = hcat(diag_mat_col(2.0; dε=0.4), diag_mat_col(1.444; dε=0.1))
        core = MaterialShape(Cuboid([0.0, 0.0], [1.2, 0.5], [1.0 0.0; 0.0 1.0]), 1)
        shapes, minds = (core,), (1, 2)
        grid0 = Grid(3.0, 2.0, 24, 16)
        ωc = 1 / 1.55

        # settings constructor: validation + stored fields
        @test_throws ArgumentError ForceConvergenceSettings(resolution_ramp=1.0)   # not > 1
        @test_throws ArgumentError ForceConvergenceSettings(boundary_ramp=0.9)      # not ≥ 1
        @test_throws ArgumentError ForceConvergenceSettings(max_iterations=1)       # need ≥ 2
        @test_throws ArgumentError ForceConvergenceSettings(rtol=-1e-3)
        s = ForceConvergenceSettings(; rtol=2e-3, atol=1e-5, resolution_ramp=1.4,
            boundary_ramp=1.2, max_iterations=5)
        @test (s.rtol, s.atol, s.resolution_ramp, s.boundary_ramp, s.max_iterations) ==
              (2e-3, 1e-5, 1.4, 1.2, 5)

        # force_convergence=false ⇒ a single solve on the supplied grid
        r0 = solve_k_converged(ωc, shapes, mat_vals, minds, grid0, solver;
            nev=1, force_convergence=false)
        @test r0 isa ForceConvergenceResult
        @test r0.iterations == 1
        @test !r0.converged
        @test r0.grid === r0.grid_history[end]
        @test size(r0.grid) == (24, 16)
        @test length(r0.neff_history) == 1
        @test r0.neff ≈ r0.kmags ./ ωc
        @test r0.neff[1] > 1.444                                   # genuinely guided
        @test size(r0.ε⁻¹) == (3, 3, 24, 16)
        @test size(r0.∂ε_∂ω) == (3, 3, 24, 16)
        @test size(r0.∂²ε_∂ω²) == (3, 3, 24, 16)

        # force_convergence=true ⇒ ramp until converged
        r = solve_k_converged(ωc, shapes, mat_vals, minds, grid0, solver;
            nev=1, force_convergence=true, force_convergence_settings=s)
        @test r.converged                                          # converges before the cap
        @test 2 <= r.iterations <= s.max_iterations
        @test length(r.neff_history) == r.iterations
        @test length(r.grid_history) == r.iterations
        @test r.grid === r.grid_history[end]
        @test r.neff === r.neff_history[end]
        @test r.neff ≈ r.kmags ./ ωc
        @test size(r.ε⁻¹) == (3, 3, size(r.grid)...)               # dielectric on final grid

        # every iteration strictly increases BOTH the boundary distance (Δ/2) and the
        # spatial point density (points/μm²), by ≈ the configured ramp factors (the density
        # ratio carries some slack from rounding the point counts to even integers)
        density(g) = length(g) / (g.Δx * g.Δy)
        for i in 2:r.iterations
            gp, g = r.grid_history[i-1], r.grid_history[i]
            @test g.Δx > gp.Δx && g.Δy > gp.Δy
            @test g.Δx / gp.Δx ≈ s.boundary_ramp rtol = 1e-12      # boundary ramp exact
            @test g.Δy / gp.Δy ≈ s.boundary_ramp rtol = 1e-12
            @test density(g) > density(gp)
            @test density(g) / density(gp) ≈ s.resolution_ramp rtol = 0.15  # ≈ density ramp
        end

        # the convergence criterion actually holds on the final step (abs OR rel)
        Δ = abs(r.neff_history[end][1] - r.neff_history[end-1][1])
        @test Δ < s.atol || Δ < s.rtol * abs(r.neff_history[end-1][1])

        # the more-refined result differs from the initial coarse solve; the iteration
        # count and convergence status are recoverable from the output grid size alone (it
        # is strictly larger than the input grid)
        @test prod(size(r.grid)) > prod(size(grid0))
        @test r.neff[1] != r0.neff[1]

        # an unsatisfiable tolerance (atol = rtol = 0) runs the full iteration budget and
        # reports non-convergence rather than stopping early
        s_strict = ForceConvergenceSettings(; rtol=0.0, atol=0.0, resolution_ramp=1.3,
            boundary_ramp=1.15, max_iterations=3)
        r_cap = solve_k_converged(ωc, shapes, mat_vals, minds, grid0, solver;
            nev=1, force_convergence=true, force_convergence_settings=s_strict)
        @test !r_cap.converged
        @test r_cap.iterations == s_strict.max_iterations

        # multiple bands: convergence requires ALL requested bands to settle, and the
        # per-iteration histories carry one effective index per band
        r2 = solve_k_converged(ωc, shapes, mat_vals, minds, grid0, solver;
            nev=2, force_convergence=true,
            force_convergence_settings=ForceConvergenceSettings(; rtol=3e-3, atol=1e-5,
                resolution_ramp=1.4, boundary_ramp=1.2, max_iterations=4))
        @test all(length(nh) == 2 for nh in r2.neff_history)
        @test issorted(r2.neff; rev=true)                          # bands ordered by |k|
    end
end
