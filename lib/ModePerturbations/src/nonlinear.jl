# χ⁽³⁾ (Kerr SPM/XPM, two-photon absorption) and χ⁽²⁾ (cascaded-χ² effective index, SHG
# normalized-efficiency overlap integral) perturbations of modal properties.
#
# Unit conventions follow the rest of OptiMode: lengths/areas in μm, n₂ in μm²/W, modal
# intensity `I(x,y)` in W/μm² (∫I dA = P in W). SI constants are used only where a result
# is conventionally quoted in SI (cascaded n₂ in m²/W, SHG η₀ in %/W/cm²).
#
# References:
#   Kerr/Aeff/γ: Agrawal, Nonlinear Fiber Optics; Afshar & Monro, Opt. Express 17, 2298 (2009).
#   Si₃N₄ n₂: Ikeda, Opt. Express 16, 12987 (2008); Si n₂/β_TPA: Lin/Painter/Agrawal,
#     Opt. Express 15, 16604 (2007); Bristow, APL 90, 191104 (2007).
#   Cascaded χ²: DeSalvo et al., Opt. Lett. 17, 28 (1992); Stegeman, OQE 28, 1691 (1996).
#   SHG η₀ overlap: Wang et al., Optica 5, 1438 (2018); Luo et al., Optica 5, 1006 (2018).

export effective_area_kerr, kerr_gamma, kerr_spm_Δneff, kerr_xpm_Δneff
export tpa_modal_loss, cm_per_GW_to_μm_per_W, kerr_complex_gamma
export cascaded_chi2_n2_eff, cascaded_chi2_Δneff
export shg_normalized_efficiency, shg_effective_area, shg_overlap_factor

const _c_SI = 2.99792458e8          # m/s
const _ε0_SI = 8.8541878128e-12     # F/m

# --- χ⁽³⁾ Kerr: effective area, γ, SPM/XPM ------------------------------------------------

"""
    effective_area_kerr(k, evec, ε⁻¹, grid) -> Real

Nonlinear effective mode area `A_eff = (∫I dA)² / ∫I² dA` (μm²) from the modal intensity
(z-Poynting flux) profile — the standard SPM/FWM effective area.
"""
function effective_area_kerr(k::Real, evec, ε⁻¹, grid::Grid)
    Sz = poynting_z(k, copy(vec(evec)), ε⁻¹, grid)
    dA = δV(grid)
    return (sum(Sz) * dA)^2 / (sum(abs2, Sz) * dA)
end

"""
    kerr_gamma(n₂, Aeff, λ) -> Real

Waveguide nonlinear parameter `γ = 2π n₂ / (λ A_eff)` in **W⁻¹m⁻¹**, with `n₂` in μm²/W,
`A_eff` in μm², `λ` in μm (the `1e6` converts the native W⁻¹μm⁻¹ to W⁻¹m⁻¹).
"""
kerr_gamma(n₂::Real, Aeff::Real, λ::Real) = 2π * n₂ / (λ * Aeff) * 1e6

"""
    kerr_complex_gamma(n₂, β_TPA, Aeff, λ) -> Complex

Complex nonlinear parameter `γ = (2π n₂)/(λ A_eff) + i β_TPA/(2 A_eff)` (W⁻¹m⁻¹), bundling
the Kerr (real) and two-photon-absorption (imaginary) parts. `β_TPA` in μm/W
(see [`cm_per_GW_to_μm_per_W`](@ref)), `A_eff` in μm².
"""
kerr_complex_gamma(n₂::Real, β_TPA::Real, Aeff::Real, λ::Real) =
    kerr_gamma(n₂, Aeff, λ) + im * (β_TPA / (2 * Aeff)) * 1e6

"""
    kerr_spm_Δneff(k, evec, ω, ε⁻¹, ∂ε_∂ω, n₂_map, grid, P) -> Real

First-order self-phase-modulation effective-index shift at optical power `P` (W): the
modal intensity `I(x,y)` (carrying power `P`) induces `Δn = n₂(x,y)·I(x,y)` and the
Hellmann–Feynman engine returns `Δneff`. This is the *fast* first-order counterpart of
`ModeAnalysis.solve_k_kerr` (no re-solve); for small `P` it matches both that full re-solve
and the textbook `n₂P/A_eff`. AD-compatible in `P` and the `n₂` map.
"""
function kerr_spm_Δneff(k::Real, evec, ω::Real, ε⁻¹, ∂ε_∂ω, n₂_map::AbstractArray, grid::Grid, P::Real)
    I = mode_intensity(k, copy(vec(evec)), ε⁻¹, grid, P)
    ε = sliceinv_3x3(copy(ε⁻¹))
    Δε = Δε_from_Δn(n₂_map .* I, ε)
    return Δneff_perturbation(k, evec, ω, ε⁻¹, Δε, grid)
end

"""
    kerr_xpm_Δneff(k_probe, evec_probe, ω_probe, ε⁻¹, ∂ε_∂ω, n₂_map, grid,
                   k_pump, evec_pump, ε⁻¹_pump, P_pump) -> Real

First-order cross-phase-modulation index shift of a probe mode from a co-propagating pump
mode carrying power `P_pump` (W). The pump intensity induces `Δn = 2 n₂ I_pump` (the
factor-of-2 XPM/SPM ratio) seen by the probe, evaluated with the probe's own field. AD-
compatible in `P_pump` and the `n₂` map.
"""
function kerr_xpm_Δneff(k_probe::Real, evec_probe, ω_probe::Real, ε⁻¹, ∂ε_∂ω,
        n₂_map::AbstractArray, grid::Grid, k_pump::Real, evec_pump, ε⁻¹_pump, P_pump::Real)
    Ipump = mode_intensity(k_pump, copy(vec(evec_pump)), ε⁻¹_pump, grid, P_pump)
    ε = sliceinv_3x3(copy(ε⁻¹))
    Δε = Δε_from_Δn(2 .* n₂_map .* Ipump, ε)
    return Δneff_perturbation(k_probe, evec_probe, ω_probe, ε⁻¹, Δε, grid)
end

"Convert a two-photon-absorption coefficient from cm/GW to μm/W (OptiMode's length/power unit)."
cm_per_GW_to_μm_per_W(β) = β * 1e-5   # cm/GW = 1e-2 m / 1e9 W = 1e-11 m/W = 1e-5 μm/W

"""
    tpa_modal_loss(k, evec, ε⁻¹, ∂ε_∂ω, β_TPA_map, grid, P) -> Real

First-order modal **power**-loss coefficient from two-photon absorption at power `P` (W),
in dB/cm. TPA contributes an intensity-dependent absorption `α_TPA = β_TPA·I`, modeled as
an imaginary index `Δn″ = β_TPA I λ /(4π)` (`κ = α/2k₀`) fed to the loss engine; the
returned modal loss is the field-weighted average over the cross-section. `β_TPA_map` in
μm/W. AD-compatible in `P`.
"""
function tpa_modal_loss(k::Real, evec, ε⁻¹, ∂ε_∂ω, β_TPA_map::AbstractArray, grid::Grid, P::Real, λ::Real)
    I = mode_intensity(k, copy(vec(evec)), ε⁻¹, grid, P)
    Δnpp = (β_TPA_map .* I) .* (λ / (4π))          # imaginary index Δn″ = β_TPA I λ /4π
    ε = sliceinv_3x3(copy(ε⁻¹))
    Δε = Δε_from_Δn(im .* Δnpp, ε)
    α = modal_loss_perturbation(k, evec, ε⁻¹, Δε, grid)
    return np_per_μm_to_dB_per_cm(α)
end

# --- Cascaded χ⁽²⁾ effective Kerr nonlinearity -------------------------------------------

"""
    cascaded_chi2_n2_eff(; deff, λ1, n1, n2, Δk) -> Real

Effective Kerr index from phase-mismatched (cascaded) second-harmonic generation,

```math
n_{2,casc} = -\\frac{2 ω_1 d_{eff}^2}{c^2 ε_0 n_1^2 n_2 Δk}\\quad[\\text{m}^2/\\text{W}],
```

with `ω₁ = 2πc/λ₁`. SI inputs: `deff` in m/V (pm/V × 1e-12), wavelength `λ1` in m, linear
indices `n1` (FH), `n2` (SH), phase mismatch `Δk = k_{2ω} − 2k_ω` in 1/m. Sign reverses
across phase matching: `Δk>0` → self-defocusing (`n₂<0`), `Δk<0` → self-focusing. Matches
the DeSalvo et al. (1992) cascading result. Differentiable in all arguments.
"""
function cascaded_chi2_n2_eff(; deff::Real, λ1::Real, n1::Real, n2::Real, Δk::Real)
    ω1 = 2π * _c_SI / λ1
    return -2 * ω1 * deff^2 / (_c_SI^2 * _ε0_SI * n1^2 * n2 * Δk)
end

"""
    cascaded_chi2_Δneff(n2_casc, I) -> Real

Effective-index shift `Δneff = n₂,casc · I` from the cascaded-χ² Kerr index (SI: `n2_casc`
in m²/W, intensity `I` in W/m²).
"""
cascaded_chi2_Δneff(n2_casc::Real, I::Real) = n2_casc * I

# --- χ⁽²⁾ SHG normalized conversion efficiency (mode-overlap integral) --------------------

# These overlap quantities are invariant to the overall field normalization (numerator and
# denominator scale identically), so the raw canonicalized `E⃗` fields can be used directly.

"""
    shg_effective_area(E_FF, E_SH, grid) -> Real

SHG effective area `A_eff = (A₁² A₂)^{1/3}` (μm²) with
`Aᵢ = (∫|Eᵢ|² dA)³ / |∫ |Eᵢ|² Eᵢ dA|²` (Luo et al., Optica 5, 1006 (2018), Eq. 2). `E_FF`,
`E_SH` are complex `(3, size(grid)...)` fundamental/second-harmonic mode fields; the inner
`∫|Eᵢ|²Eᵢ` is taken over the nonlinear (χ²) region if a mask is folded into the fields,
else over all space.
"""
function shg_effective_area(E_FF::AbstractArray, E_SH::AbstractArray, grid::Grid)
    dA = δV(grid)
    A1 = _mode_area_i(E_FF, dA)
    A2 = _mode_area_i(E_SH, dA)
    return (A1^2 * A2)^(1 / 3)
end

# Aᵢ = (∫|E|² dA)³ / |∫ |E|² E dA|²  — uses the dominant (largest-power) Cartesian component
# of the vector "self-overlap" ∫|E|²E to match the scalar-overlap convention.
function _mode_area_i(E::AbstractArray, dA::Real)
    cols = ntuple(_ -> Colon(), ndims(E) - 1)
    E2 = dropdims(sum(abs2, E; dims=1); dims=1)            # |E|² per pixel
    num = (sum(E2) * dA)^3
    selfov = ntuple(a -> sum(E2 .* view(E, a, cols...)) * dA, 3)   # ∫|E|²E dA (vector)
    den = sum(abs2, selfov)
    return num / den
end

"Cartesian index (1,2,3) of the largest-power component of a vector field, integrated."
function dominant_component(E::AbstractArray)
    cols = ntuple(_ -> Colon(), ndims(E) - 1)
    return argmax(ntuple(a -> sum(abs2, view(E, a, cols...)), 3))
end

"""
    shg_overlap_factor(E_FF, E_SH, grid; chi2_mask=nothing, ff_comp=auto, sh_comp=auto) -> Real

Dimensionless χ² mode-overlap factor `ζ` (Luo et al., Optica 5, 1006 (2018), Eq. 3),

```math
ζ = \\frac{∫_{χ²} (E_{1i}^*)^2 E_{2j}\\,dA}
         {|∫_{χ²}|E_1|^2 E_1\\,dA|^{2/3}\\,|∫_{χ²}|E_2|^2 E_2\\,dA|^{1/3}},
```

coupling two fundamental photons (FF component `i`) to one SH photon (component `j`). `i`,`j`
default to each mode's dominant Cartesian component (`ff_comp`/`sh_comp` to override): for
TE00→TE00 SHG via `d₃₃` both are `Ex` (`i=j=1`); for Luo's type-I intermodal case FF is
`Ex` (`i=1`) and SH `Ey` (`j=2`). `chi2_mask` (1 inside the χ² material) restricts the
numerator integral; defaults to all space. Magnitude `|ζ|` is returned.
"""
function shg_overlap_factor(E_FF::AbstractArray, E_SH::AbstractArray, grid::Grid;
        chi2_mask=nothing, ff_comp::Int=dominant_component(E_FF), sh_comp::Int=dominant_component(E_SH))
    dA = δV(grid)
    cols = ntuple(_ -> Colon(), ndims(E_FF) - 1)
    mask = chi2_mask === nothing ? one(real(eltype(E_FF))) : chi2_mask
    E1i = view(E_FF, ff_comp, cols...)
    E2j = view(E_SH, sh_comp, cols...)
    num = sum((E1i .^ 2) .* conj.(E2j) .* mask) * dA
    E1self = ntuple(a -> sum(dropdims(sum(abs2, E_FF; dims=1); dims=1) .* view(E_FF, a, cols...) .* mask) * dA, 3)
    E2self = ntuple(a -> sum(dropdims(sum(abs2, E_SH; dims=1); dims=1) .* view(E_SH, a, cols...) .* mask) * dA, 3)
    den = sqrt(sum(abs2, E1self))^(2 / 3) * sqrt(sum(abs2, E2self))^(1 / 3)
    return abs(num) / den
end

"""
    shg_normalized_efficiency(E_FF, E_SH, grid; deff, λ1, n1, n2, chi2_mask=nothing) -> Real

Normalized SHG conversion efficiency `η₀` in **%·W⁻¹·cm⁻²** (Luo et al., Optica 5, 1006
(2018), Eq. 2),

```math
η_0 = \\frac{8π^2 ζ^2 d_{eff}^2}{ε_0 c λ_1^2 n_1^2 n_2 A_{eff}},
```

from the fundamental/SH mode fields, the effective nonlinear coefficient `deff` (m/V), the
fundamental wavelength `λ1` (m), and linear indices `n1` (FH), `n2` (SH). `A_eff` and `ζ`
come from [`shg_effective_area`](@ref)/[`shg_overlap_factor`](@ref) (areas converted μm²→m²
internally). Reproduces the few-thousand-%/W/cm² scale of nanophotonic PPLN waveguides
(Wang et al. 2018). Differentiable in `deff`, indices, and (through the fields) geometry.
"""
function shg_normalized_efficiency(E_FF::AbstractArray, E_SH::AbstractArray, grid::Grid;
        deff::Real, λ1::Real, n1::Real, n2::Real, chi2_mask=nothing)
    ζ = shg_overlap_factor(E_FF, E_SH, grid; chi2_mask)
    Aeff = shg_effective_area(E_FF, E_SH, grid) * 1e-12      # μm² → m²
    η = 8π^2 * ζ^2 * deff^2 / (_ε0_SI * _c_SI * λ1^2 * n1^2 * n2 * Aeff)
    return η * 100 * 1e-4   # 1/(W·m²) → %/(W·cm²): ×100 (%) ×1e-4 (m²/cm²)
end
