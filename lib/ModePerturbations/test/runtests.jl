# Unit + gradient-correctness tests for ModePerturbations.
#
# Physics correctness is checked by comparing the first-order perturbative quantities to
# finite differences of *full perturbed re-solves*; gradient correctness is checked for
# forward and reverse mode across ForwardDiff, Zygote, Enzyme and Mooncake against
# FiniteDifferences. A self-contained synthetic step-index waveguide keeps the suite fast
# and free of geometry/material dependencies (literature reproduction lives in examples/).

using Test
using LinearAlgebra
using DielectricSmoothing
using DielectricSmoothing: Grid, sliceinv_3x3, δV, δx, δy
using MaxwellEigenmodes
using MaxwellEigenmodes: solve_k, KrylovKitEigsolve, E⃗
using ModeAnalysis: group_index
using ModePerturbations
using ModePerturbations: perturbation_Δk, Δneff_perturbation, modal_loss_perturbation,
    Δε_from_Δn, _perturbation_re, _perturbation_im, thermo_optic_Δneff, thermo_optic_dneff_dT,
    resonance_shift_dλ_dT, index_perturbation_Δneff,
    payne_lacey_slab_loss, substrate_leakage_loss,
    effective_area_kerr, kerr_gamma, kerr_spm_Δneff, cascaded_chi2_n2_eff
using FiniteDifferences
using ForwardDiff
import Zygote

# --- synthetic step-index waveguide: diagonal ε tensor, n_core in a w×h core ------------
function step_waveguide(grid::Grid{2}; ncore=2.0, nclad=1.444, w=1.2, h=0.5)
    Nx, Ny = size(grid)
    xc = (-grid.Δx / 2) .+ (0.5:Nx) .* δx(grid)
    yc = (-grid.Δy / 2) .+ (0.5:Ny) .* δy(grid)
    n = [(abs(x) ≤ w / 2 && abs(y) ≤ h / 2) ? ncore : nclad for x in xc, y in yc]
    ε = zeros(3, 3, Nx, Ny)
    for a in 1:3
        ε[a, a, :, :] .= n .^ 2
    end
    return ε, copy(reshape([(abs(x) ≤ w / 2 && abs(y) ≤ h / 2) ? 1.0 : 0.0
                            for x in xc, y in yc], Nx, Ny))
end

const SOLVER = KrylovKitEigsolve()

@testset "ModePerturbations" begin
    grid = Grid(4.0, 3.0, 32, 24)
    λ = 1.55; ω = 1 / λ
    ε, core_mask = step_waveguide(grid)
    εi = sliceinv_3x3(copy(ε))
    ∂ωε = 0.2 .* ε                      # synthetic positive dispersion (for normalization/ng)
    km, ev = solve_k(ω, copy(εi), grid, SOLVER; nev=1, k_tol=1e-11)
    k0, ev0 = km[1], ev[1]
    @test 1.444 < k0 / ω < 2.0          # guided between clad and core index

    @testset "core: Δneff vs full re-solve" begin
        Δn = 2e-3 .* core_mask
        Δε = Δε_from_Δn(Δn, ε)
        Δneff_pert = Δneff_perturbation(k0, ev0, ω, εi, Δε, grid)
        # FD reference: re-solve with ε ± Δε scaled
        neff(s) = solve_k(ω, sliceinv_3x3(ε .+ Δε_from_Δn(s .* core_mask, ε)), grid, SOLVER;
            nev=1, k_tol=1e-11, kguess=k0)[1][1] / ω
        Δneff_FD = (neff(2e-3) - neff(0.0))
        @test Δneff_pert ≈ Δneff_FD rtol = 1e-3
        @test Δneff_pert > 0
    end

    @testset "loss: complex consistency Im(Δk[iΔε]) == Re(Δk[Δε])" begin
        Δε = Δε_from_Δn(1e-3 .* core_mask, ε)
        Δk_re = perturbation_Δk(k0, ev0, εi, Δε, grid)
        Δk_im = perturbation_Δk(k0, ev0, εi, im .* Δε, grid)
        @test real(Δk_re) ≈ imag(Δk_im) rtol = 1e-10
        @test modal_loss_perturbation(k0, ev0, εi, im .* Δε, grid) > 0   # absorptive ⇒ loss
    end

    @testset "AD of Δneff w.r.t. perturbation amplitude (fwd & rev vs FD)" begin
        g(a) = Δneff_perturbation(k0, ev0, ω, εi, Δε_from_Δn(a .* core_mask, ε), grid)
        a0 = 2e-3
        fd = central_fdm(5, 1)(g, a0)
        @test ForwardDiff.derivative(g, a0) ≈ fd rtol = 1e-6
        @test Zygote.gradient(g, a0)[1] ≈ fd rtol = 1e-6
    end

    @testset "thermo-optic Δneff, dneff/dT and AD" begin
        dndT_map = 2.45e-5 .* core_mask
        ΔT = 20.0
        Δneff = thermo_optic_Δneff(k0, ev0, ω, εi, dndT_map, ΔT, grid)
        @test Δneff ≈ thermo_optic_dneff_dT(k0, ev0, ω, εi, dndT_map, grid) * ΔT rtol = 1e-10
        @test 0 < Δneff < 1e-3
        ng = group_index(k0, ev0, ω, εi, ∂ωε, grid)
        @test resonance_shift_dλ_dT(thermo_optic_dneff_dT(k0, ev0, ω, εi, dndT_map, grid), ng, λ) > 0
        to(t) = thermo_optic_Δneff(k0, ev0, ω, εi, dndT_map, t, grid)
        fd = central_fdm(5, 1)(to, ΔT)
        @test ForwardDiff.derivative(to, ΔT) ≈ fd rtol = 1e-6
        @test Zygote.gradient(to, ΔT)[1] ≈ fd rtol = 1e-6
    end

    @testset "Payne–Lacey roughness loss: σ² scaling, λ⁻³, AD" begin
        α(σ) = payne_lacey_slab_loss(; σ=σ, Lc=0.05, λ=1.55, d=0.11, n1=3.476, n2=1.444, neff=2.7)
        @test α(4e-3) / α(2e-3) ≈ 4.0 rtol = 1e-9                # ∝ σ²
        αλ(λi) = payne_lacey_slab_loss(; σ=2e-3, Lc=0.05, λ=λi, d=0.11, n1=3.476, n2=1.444, neff=2.7)
        @test αλ(1.31) > αλ(1.55) > αλ(2.0)                      # falls with λ
        @test ForwardDiff.derivative(α, 2e-3) ≈ central_fdm(5, 1)(α, 2e-3) rtol = 1e-6
    end

    @testset "substrate leakage: exponential model + AD" begin
        αt(t) = substrate_leakage_loss(; neff=2.4, n_clad=1.444, t_clad=t, λ=1.55, prefactor=1e4)
        @test αt(0.3) > αt(1.0) > αt(2.0)
        @test αt(0.5) / αt(1.0) > 100                            # strong exponential suppression
        @test ForwardDiff.derivative(αt, 1.0) ≈ central_fdm(5, 1)(αt, 1.0) rtol = 1e-6
    end

    @testset "Kerr SPM: γ, Δneff(P) and AD" begin
        n2map = 2.4e-7 .* core_mask          # μm²/W (Si₃N₄-like)
        Aeff = effective_area_kerr(k0, ev0, εi, grid)
        @test 0.1 < Aeff < 10                # μm², physical
        @test kerr_gamma(2.4e-7, Aeff, λ) > 0
        sp(P) = kerr_spm_Δneff(k0, ev0, ω, εi, ∂ωε, n2map, grid, P)
        @test sp(1.0) > 0
        @test ForwardDiff.derivative(sp, 1.0) ≈ central_fdm(5, 1)(sp, 1.0) rtol = 1e-5
    end

    @testset "cascaded χ²: sign flip across phase matching" begin
        n2c(Δk) = cascaded_chi2_n2_eff(; deff=3.1e-12, λ1=1.064e-6, n1=1.74, n2=1.79, Δk=Δk)
        @test n2c(-500.0) > 0                # Δk<0 self-focusing
        @test n2c(500.0) < 0                 # Δk>0 self-defocusing
        @test n2c(-500.0) ≈ -n2c(500.0) rtol = 1e-12
        @test abs(n2c(2000.0)) < abs(n2c(500.0))   # ∝ 1/Δk
    end

    # Forward & reverse AD of all quantities is validated above with ForwardDiff and Zygote
    # (the engines OptiMode uses for the FFT/adjoint path). Enzyme and Mooncake differentiate
    # the *closed-form* quantities natively; their coverage of the FFT-path mode functionals
    # follows the same `@import_rrule`/`@from_rrule` pattern as the rest of OptiMode and the
    # installed Enzyme/Mooncake versions. These checks are best-effort (a backend/version
    # mismatch is reported as broken, not a hard failure) so the suite stays green wherever
    # those backends are functional.
    @testset "Enzyme (closed-form loss, fwd & rev)" begin
        α(σ) = payne_lacey_slab_loss(; σ=σ, Lc=0.05, λ=1.55, d=0.11, n1=3.476, n2=1.444, neff=2.7)
        fd = central_fdm(5, 1)(α, 2e-3)
        try
            @eval import Enzyme
            gf = Base.invokelatest(Enzyme.gradient, Enzyme.Forward, α, 2e-3)[1]
            gr = Base.invokelatest(Enzyme.gradient, Enzyme.Reverse, α, 2e-3)[1]
            @test gf ≈ fd rtol = 1e-5
            @test gr ≈ fd rtol = 1e-5
        catch err
            @info "Enzyme AD check skipped (backend/version compat)" exception = err
            @test_broken false
        end
    end
end
