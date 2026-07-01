# Shared helpers for the four "paper reproduction" examples:
#
#   • tantala_gvd_black2021.jl              — Black et al., Opt. Lett. 46, 817 (2021)  [χ³]
#   • si3n4_cw_opa_riemensberger2022.jl     — Riemensberger et al., Nature 612, 56 (2022) [χ³]
#   • ppln_reconfigurable_opa_han2026.jl    — Han et al., arXiv:2602.00246 (2026)      [χ²]
#   • pplt_allband_opa_kuznetsov2026.jl     — Kuznetsov et al., arXiv:2605.22704 (2026) [χ²]
#
# Each reproduces, using only OptiMode's mode solver + perturbation tools:
#   (1) the modal dispersion spectra (n_eff, n_g, GVD β₂),
#   (2) the OPA phase-matching / parametric-gain spectra
#       (χ³: degenerate-FWM gain; χ²: QPM SHG/DFG tuning + gain bandwidth),
#   (3) the nonlinear coupling coefficient (χ³: Kerr γ; χ²: normalized efficiency η₀), and
#   (4) the fundamental-mode transverse field profiles.
#
# These run at a moderate grid so the plots generate on a workstation in minutes; the
# geometry sweeps scale to converged grids via ModeSweeps/SLURM (see the sweep examples).
#
# Unit conventions (as elsewhere in OptiMode):
#   ω = 1/λ (µm⁻¹);  |k| = n_eff·ω (µm⁻¹);  physical propagation constant β = 2π|k| (rad/µm).
#   angular optical frequency ω_a = 2πc/λ.  GVD β₂ = d²β/dω_a².

using OptiMode
using OptiMode.DielectricSmoothing: δx, δy
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations: effective_area_kerr, kerr_gamma,
    shg_effective_area, shg_normalized_efficiency
using LinearAlgebra
using Printf

const C_MS   = 299792458.0            # speed of light (m/s)
const C_UM_FS = C_MS * 1e-9           # speed of light (µm/fs) = 0.299792458
const OUTDIR = joinpath(@__DIR__, "paper_reproduction_output")
isdir(OUTDIR) || mkpath(OUTDIR)

# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------

"Rectangular `core` (width `w`, height `h`, µm) centered at (0, `yc`) buried in a background
cladding (the last material column). Returns (shapes, minds) for a fully-clad waveguide."
function buried_core(w, h; yc=0.0)
    core = MaterialShape(Cuboid([0.0, yc], [w, h], [1.0 0.0; 0.0 1.0]), 1)
    return (core,), (1, 2)            # core→mat 1, background→mat 2
end

"Ridge (`w`×`etch`) on an unetched slab (thickness `film-etch`) over a `sub` substrate; air
background. Mirrors the x-cut TFLN geometry used in the package's other thin-film examples.
Returns (shapes, minds) with columns ordered (film-material, substrate, air)."
function ridge_on_slab(w, etch, film; sub_halfW=100.0, sub_halfH=2.0)
    slab = film - etch
    ( MaterialShape(Cuboid([0.0, slab + etch/2], [w, etch], [1.0 0.0; 0.0 1.0]), 1),    # ridge
      MaterialShape(Cuboid([0.0, slab/2], [sub_halfW, slab], [1.0 0.0; 0.0 1.0]), 1),   # slab
      MaterialShape(Cuboid([0.0, -sub_halfH/2], [sub_halfW, sub_halfH], [1.0 0.0; 0.0 1.0]), 2) # substrate
    ), (1, 1, 2, 3)                   # ridge→film, slab→film, substrate→sub, background→air
end

const AIR_COL = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))   # 27-vector for a vacuum column

"Column-wise material dielectric values `matvals(ω)` (27×Nmat) for `mats`, optionally with a
trailing air column."
function matvals_builder(mats; air=false)
    fε, _ = _f_ε_mats(mats, (:ω,))
    air ? (ω -> hcat(fε([ω]), AIR_COL)) : (ω -> fε([ω]))
end

# ---------------------------------------------------------------------------------------
# Mode solving + selection
# ---------------------------------------------------------------------------------------

"Smoothed (ε⁻¹, ∂ωε, ∂²ωε) for `shapes`/`minds` at frequency `ω`."
function diel(shapes, minds, matvals, ω, grid)
    sm = smooth_ε(shapes, matvals(ω), minds, grid)
    (sliceinv_3x3(copy(selectdim(sm, 3, 1))), copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
end

"Fraction of |E|² inside |x|<w/2, |y-yc|<h/2 — distinguishes the guided core mode from
delocalized slab/substrate modes."
function core_confinement(E, grid, w, h; yc=0.0)
    Nx, Ny = size(grid)
    xc = (-grid.Δx/2) .+ (0.5:Nx) .* (grid.Δx/Nx)
    yc_ = (-grid.Δy/2) .+ (0.5:Ny) .* (grid.Δy/Ny)
    I = dropdims(sum(abs2, E; dims=1); dims=1)
    mask = [(abs(x) < w/2 + 0.15 && abs(y - yc) < h/2 + 0.2) ? 1.0 : 0.0 for x in xc, y in yc_]
    sum(I .* mask) / sum(I)
end

"""
    solve_fundamental(shapes, minds, matvals, ω, grid, solver; nev, pol, w, h, yc)
        -> (; k, ev, εi, ∂ωε, ∂²ωε, E, pol_frac, conf, neff)

Solve at `ω`, reconstruct the fields, and select the fundamental mode of the requested
polarization (`:TE` → Eₓ-dominant, `:TM` → Eᵧ-dominant) that is most confined to the core.
"""
function solve_fundamental(shapes, minds, matvals, ω, grid, solver;
        nev=6, pol=:TE, w=1.0, h=0.5, yc=0.0, k_tol=1e-7, eig_tol=1e-7)
    εi, ∂ωε, ∂²ωε = diel(shapes, minds, matvals, ω, grid)
    ε = sliceinv_3x3(copy(εi))
    km, ev = solve_k(ω, copy(εi), grid, solver; nev=nev, k_tol=k_tol, eig_tol=eig_tol)
    Es = [E⃗(km[i], copy(ev[i]), εi, ∂ωε, grid; canonicalize=true, normalized=true) for i in eachindex(ev)]
    rp = [E_relpower_xyz(ε, Es[i]) for i in eachindex(ev)]
    pf = pol === :TE ? [r[1] for r in rp] : [r[2] for r in rp]   # x- or y-fraction
    conf = [core_confinement(Es[i], grid, w, h; yc=yc) for i in eachindex(ev)]
    # rank: right-polarization modes first (birefringence can push several wrong-pol modes
    # above the target fundamental), then most core-confined, then highest polarization purity.
    score = [(pf[i] > 0.5 ? 100.0 : 0.0) + 10 * conf[i] + pf[i] for i in eachindex(ev)]
    i = argmax(score)
    return (; k=km[i], ev=ev[i], εi, ∂ωε, ∂²ωε, E=Es[i], pol_frac=pf[i], conf=conf[i], neff=km[i]/ω)
end

# ---------------------------------------------------------------------------------------
# Dispersion sweep
# ---------------------------------------------------------------------------------------

"GVD β₂ in fs²/mm from OptiMode's `gvd` (= ∂²|k|/∂ω²):  β₂ = gvd/(2π c²)."
gvd_fs2_per_mm(gvd_OM) = 1e3 * gvd_OM / (2π * C_UM_FS^2)

"Dispersion parameter D in ps/(nm·km) from β₂ (fs²/mm):  D = −(2πc/λ²)·β₂."
function D_ps_nm_km(β2_fs2mm, λ_um)
    β2_s2_m = β2_fs2mm * 1e-30 / 1e-3            # fs²/mm → s²/m
    Dm = -(2π * C_MS / (λ_um*1e-6)^2) * β2_s2_m  # s/m²  (= s per m per m)
    return Dm * 1e6 * 1e3 * 1e12 / 1e9           # → ps/(nm·km)
end

"""
    sweep_dispersion(shapes, minds, matvals, λs, grid, solver; pol, w, h, yc, nev)
        -> (; λ, ω, neff, ng, β2, D, kmag)

Solve the fundamental mode across `λs` (µm) and return the modal dispersion spectra:
effective index, group index, GVD β₂ (fs²/mm), D (ps/nm/km), and |k| (µm⁻¹).
"""
function sweep_dispersion(shapes, minds, matvals, λs, grid, solver;
        pol=:TE, w=1.0, h=0.5, yc=0.0, nev=6)
    n = length(λs)
    neff = zeros(n); ng = zeros(n); β2 = zeros(n); D = zeros(n); kmag = zeros(n)
    for (j, λ) in enumerate(λs)
        ω = 1/λ
        m = solve_fundamental(shapes, minds, matvals, ω, grid, solver; nev=nev, pol=pol, w=w, h=h, yc=yc)
        g, gv = ng_gvd(ω, m.k, m.ev, m.εi, m.∂ωε, m.∂²ωε, grid)
        neff[j] = m.neff; ng[j] = g; kmag[j] = m.k
        β2[j] = gvd_fs2_per_mm(gv); D[j] = D_ps_nm_km(β2[j], λ)
        @printf("  λ=%.3f µm  n_eff=%.4f  n_g=%.4f  β₂=%+.1f fs²/mm  (%s %.2f, conf %.2f)\n",
                λ, neff[j], g, β2[j], pol, m.pol_frac, m.conf)
    end
    (; λ=collect(λs), ω=1 ./ λs, neff, ng, β2, D, kmag)
end

# ---------------------------------------------------------------------------------------
# χ³ (Kerr) four-wave-mixing OPA gain
# ---------------------------------------------------------------------------------------

"Signal power gain (dB) of a parametric amplifier: G = 1 + (drive·sinh(gL)/g)² with
g = √(drive² − (κ/2)²) and κ the total wavevector mismatch. Uses the limit sinh(gL)/g → L
as g → 0, so the phase-matched point (g=0) is finite rather than 0·∞."
function parametric_gain_dB(κ, drive, L)
    g = sqrt.(Complex.(drive^2 .- (κ ./ 2).^2))
    sinhc = map(gg -> abs(gg) < 1e-9 ? complex(L) : sinh(gg * L) / gg, g)
    G = 1 .+ (drive .* real.(sinhc)).^2
    10 .* log10.(G)
end

"β(ω_a): physical propagation constant β = 2π|k| (rad/m) vs angular optical frequency
(rad/s), from a dispersion sweep. Used to fit β and its even derivatives β₂, β₄ at the pump."
function beta_of_omega(sw)
    ω_a = 2π * C_MS ./ (sw.λ .* 1e-6)   # angular optical frequency (rad/s)
    β   = 2π .* sw.kmag .* 1e6          # β = 2π|k|, µm⁻¹ → rad/m
    return ω_a, β
end

"Even-order dispersion derivatives (β₂, β₄ in SI: s²/m, s⁴/m) at wavelength `λ0`, from a
degree-5 least-squares fit of β(ω_a). The fit variable is scaled to u = (ω_a−ω₀)/ω₀ (≈0.1)
so the Vandermonde system is well-conditioned; derivatives are then β_n = n!·cₙ/ω₀ⁿ."
function beta_even_derivs(sw, λ0)
    ω_a, β = beta_of_omega(sw)
    ω0 = 2π * C_MS / (λ0 * 1e-6)
    u = (ω_a .- ω0) ./ ω0
    c = hcat((u .^ k for k in 0:5)...) \ β
    β2 = 2 * c[3] / ω0^2                # d²β/dω²  (s²/m)
    β4 = 24 * c[5] / ω0^4              # d⁴β/dω⁴  (s⁴/m)
    (β2, β4)
end

"""
    fwm_gain_spectrum(sw, λp, P, γ; L, Ω_THz)
        -> (; detuning_THz, λs_signal, gain_dB, g, Δβ, β2, β4, bw3_THz)

Degenerate-FWM parametric-gain spectrum (Riemensberger 2022, Eq. 2) for a pump at `λp`
(µm), on-chip power `P` (W), nonlinearity `γ` (W⁻¹m⁻¹), device length `L` (m). Uses the
even-order dispersion at the pump (β₂, β₄ from a local fit of β(ω_a)) so that
Δβ(Ω) = β₂Ω² + β₄Ω⁴/12, g = √((γP)² − (Δβ/2 + γP)²), and signal gain
G = 1 + (γP/g)²·sinh²(gL).
"""
function fwm_gain_spectrum(sw, λp, P, γ; L=1.0, Ω_THz=range(-20, 20; length=401))
    ωp = 2π * C_MS / (λp * 1e-6)
    β2, β4 = beta_even_derivs(sw, λp)   # SI: s²/m, s⁴/m (well-conditioned scaled fit)
    Ω = collect(Ω_THz) .* 2π * 1e12     # angular detuning (rad/s)
    Δβ = β2 .* Ω.^2 .+ (β4/12) .* Ω.^4  # linear phase mismatch (rad/m)
    κ = Δβ .+ 2γ*P                      # total mismatch incl. nonlinear phase
    gain_dB = parametric_gain_dB(κ, γ*P, L)
    λs_signal = 1e6 .* (2π * C_MS) ./ (ωp .+ Ω)   # signal wavelength (µm)
    # 3-dB gain bandwidth about the peak
    gpk = maximum(gain_dB); above = gain_dB .>= (gpk - 3)
    bw = count(above) > 1 ? (maximum(collect(Ω_THz)[above]) - minimum(collect(Ω_THz)[above])) : 0.0
    (; detuning_THz=collect(Ω_THz), λs_signal, gain_dB, Δβ, β2, β4, bw3_THz=bw)
end

# ---------------------------------------------------------------------------------------
# χ² quasi-phase-matching (SHG / DFG) tuning + gain bandwidth
# ---------------------------------------------------------------------------------------

"First-order QPM poling period Λ = λ_FF / (2(n_SH − n_FF)) in µm."
poling_period(neff_FF, neff_SH, λ_FF) = λ_FF / (2 * (neff_SH - neff_FF))

"Even-order dispersion (β₂ in fs²/mm, β₄ in fs⁴/mm) at wavelength `λ0` from a local degree-5
fit of β(ω_a) over a dispersion sweep `sw`."
function dispersion_betas(sw, λ0)
    β2, β4 = beta_even_derivs(sw, λ0)
    (β2 * 1e27, β4 * 1e57)              # (s²/m→fs²/mm, s⁴/m→fs⁴/mm)
end

"""
    chi2_opa_gain_spectrum(sw_FH, λdeg, Γ; L, Ω_THz)
        -> (; detuning_THz, λs_signal, gain_dB, β2, β4, bw3_THz)

Near-degeneracy χ²-OPA parametric gain (SH-pumped, signal/idler about the FH degeneracy at
`λdeg` µm). Unlike the Kerr case the phase mismatch is pump-power-independent:
Δk(Ω) = β₂Ω² + β₄Ω⁴/12 from the FH-band dispersion; with drive Γ = √(η₀ P_SH) (1/m),
g = √(Γ² − (Δk/2)²) and signal gain G = 1 + (Γ/g)²·sinh²(gL)."""
function chi2_opa_gain_spectrum(sw_FH, λdeg, Γ; L=0.01, Ω_THz=range(-100, 100; length=601))
    β2, β4 = beta_even_derivs(sw_FH, λdeg)          # SI: s²/m, s⁴/m
    Ω = collect(Ω_THz) .* 2π * 1e12
    Δk = β2 .* Ω.^2 .+ (β4/12) .* Ω.^4
    gain_dB = parametric_gain_dB(Δk, Γ, L)
    ωdeg = 2π * C_MS / (λdeg * 1e-6)
    λs_signal = 1e6 .* (2π * C_MS) ./ (ωdeg .+ Ω)
    gpk = maximum(gain_dB); above = gain_dB .>= (gpk - 3)
    bw = count(above) > 1 ? (maximum(collect(Ω_THz)[above]) - minimum(collect(Ω_THz)[above])) : 0.0
    (; detuning_THz=collect(Ω_THz), λs_signal, gain_dB, β2=β2*1e27, β4=β4*1e57, bw3_THz=bw)
end

"""
    qpm_mismatch_spectrum(neff_FF_fn, neff_SH_fn, λs_FF, Λ)
        -> (; λ, Δk, ΔkL_over_pi)

SHG quasi-phase-matched wavevector mismatch Δk(λ) = β_SH − 2β_FF − 2π/Λ (rad/mm) across the
fundamental band `λs_FF` (µm), given interpolants n_eff(λ) for the FF and SH modes and poling
period `Λ` (µm). Δk = 0 marks the phase-matched wavelength; the sinc² acceptance sets the
conversion/gain bandwidth.
"""
function qpm_mismatch_spectrum(neff_FF, neff_SH, λs_FF, Λ)
    λ = collect(λs_FF)
    βFF = 2π .* neff_FF ./ λ                 # rad/µm
    βSH = 2π .* neff_SH ./ (λ ./ 2)          # SH at λ/2, rad/µm
    Δk = (βSH .- 2 .* βFF .- 2π/Λ) .* 1e3    # rad/mm
    (; λ, Δk, ΔkL_over_pi=Δk)
end

# ---------------------------------------------------------------------------------------
# Plotting helpers (CairoMakie)
# ---------------------------------------------------------------------------------------

"Normalized |E| map (peak 1) of a (3,Nx,Ny) complex field."
absE_norm(E) = (m = sqrt.(dropdims(sum(abs2, E; dims=1); dims=1)); m ./ maximum(m))

"Pixel-center coordinate vectors (µm) for a grid."
function grid_coords(grid)
    Nx, Ny = size(grid)
    (collect((-grid.Δx/2) .+ (0.5:Nx) .* δx(grid)),
     collect((-grid.Δy/2) .+ (0.5:Ny) .* δy(grid)))
end
