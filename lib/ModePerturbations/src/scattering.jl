# Radiative scattering loss: sidewall/surface roughness (Payne–Lacey) and substrate
# leakage through a finite lower cladding. Unlike material absorption these are *radiative*
# couplings of the guided mode to a continuum, not a simple Im(ε) perturbation, so they get
# dedicated mode-field functionals rather than the Δε engine.
#
# References:
#   Payne & Lacey, "A theoretical analysis of scattering loss from planar optical
#     waveguides," Opt. Quantum Electron. 26, 977 (1994).
#   Lee et al., Opt. Lett. 26, 1888 (2001) (SOI σ:Lc = 2nm:50nm → ~0.8 dB/cm).
#   Melati, Melloni, Morichetti, Adv. Opt. Photon. 6, 156 (2014) (review).
#   Sridaran & Bhave, Opt. Express 18, 3850 (2010) (substrate leakage e-folding).
#   Bauters et al., Opt. Express 19, 3163 (2011); Blumenthal et al., Proc. IEEE 106, 2209 (2018).

export payne_lacey_slab_loss, roughness_scattering_loss, substrate_leakage_loss
export sidewall_field_overlap

# --- Payne–Lacey slab model (the literature-calibrated, AD-differentiable workhorse) ------

"""
    payne_lacey_slab_loss(; σ, Lc, λ, d, n1, n2, neff, acf=:exp, both_surfaces=true) -> Real

Sidewall/surface roughness scattering loss `α` in **dB/cm** for a symmetric slab waveguide
(core full-thickness `2d`, so `d` is the core **half**-thickness), via the Payne–Lacey
closed form

```math
α = \\frac{σ^2}{\\sqrt2\\,k_0 d^4 n_1}\\,g(V)\\,f(x,γ),\\quad g(V)=\\frac{U^2V^2}{1+W},
```

with `V=d k₀√(n₁²−n₂²)`, `U=d k₀√(n₁²−n_{eff}²)`, `W=d k₀√(n_{eff}²−n₂²)`,
`x=W L_c/d`, `γ=n₂V/(n₁W√Δ)`, `Δ=(n₁²−n₂²)/2n₁²`, and the exponential-ACF spectrum factor
`f(x,γ)=x√(√((1+x²)²+2x²γ²)+1−x²)/√((1+x²)²+2x²γ²)`. All lengths share one unit (μm here:
`σ`, `Lc`, `λ`, `d` in μm). `both_surfaces=false` halves the result (one rough wall).

Loss `∝ σ²`, `∝ (n₁²−n₂²)²` (through `g`), `∝ k₀³` (≈ `λ⁻³`) plus the field/PSD factors —
the magnitude and wavelength scaling reproduce SOI/SiN measurements (Lee 2001, Melati 2014).
Differentiable in every argument (`σ`, `Lc`, `λ`, `d`, indices, `neff`).
"""
function payne_lacey_slab_loss(; σ::Real, Lc::Real, λ::Real, d::Real, n1::Real, n2::Real,
        neff::Real, acf::Symbol=:exp, both_surfaces::Bool=true)
    k0 = 2π / λ
    Δ = (n1^2 - n2^2) / (2n1^2)
    V = d * k0 * sqrt(n1^2 - n2^2)
    U = d * k0 * sqrt(max(n1^2 - neff^2, 0.0))
    W = d * k0 * sqrt(max(neff^2 - n2^2, 0.0))
    g = U^2 * V^2 / (1 + W)
    x = W * Lc / d
    f = _pl_spectrum_factor(x, n2 * V / (n1 * W * sqrt(Δ)), acf)
    α_npμm = σ^2 / (sqrt(2) * k0 * d^4 * n1) * g * f      # nepers/μm
    both_surfaces || (α_npμm *= 0.5)
    return np_per_μm_to_dB_per_cm(α_npμm)
end

# exponential-ACF spectrum factor (Payne–Lacey Eq.); :gaussian uses the standard
# error-function-free approximation x·exp(-x²γ²)·… is not closed form, so we provide the
# exponential ACF (the form used in essentially all SOI/SiN fitting).
function _pl_spectrum_factor(x, γ, acf::Symbol)
    acf === :exp || throw(ArgumentError("only :exp (exponential ACF) closed form is implemented; " *
        "use roughness_scattering_loss for numerical PSDs"))
    r = sqrt((1 + x^2)^2 + 2 * x^2 * γ^2)
    return x * sqrt(r + 1 - x^2) / r
end

# --- Field-based version (uses the real mode field at the etched sidewall) ----------------

"""
    sidewall_field_overlap(k, evec, ε⁻¹, ∂ε_∂ω, grid, boundary_mask, normal_axis) -> Real

Normalized tangential-field intensity integrated over the rough interface selected by
`boundary_mask` (the perimeter pixels of the core sidewall/surface), `∮|E_∥|²ds / ∫∫|E|²dA`
[1/μm], from the energy-normalized mode field. `normal_axis ∈ (1,2)` is the interface
normal direction (so the tangential components are the other two). This is the modal hook
that replaces Payne–Lacey's analytic `φ²(d)` for an arbitrary cross-section.
"""
function sidewall_field_overlap(k::Real, evec, ε⁻¹, ∂ε_∂ω, grid::Grid, boundary_mask::AbstractArray,
        normal_axis::Int)
    E = E⃗(k, copy(vec(evec)), ε⁻¹, ∂ε_∂ω, grid; canonicalize=true, normalized=false)
    tang = filter(!=(normal_axis), (1, 2, 3))
    cols = ntuple(_ -> Colon(), length(size(grid)))
    Et2 = sum(t -> abs2.(view(E, t, cols...)), tang)      # |E_∥|² per pixel
    Eall2 = dropdims(sum(abs2, E; dims=1); dims=1)         # |E|² per pixel
    ds = step_perp(grid, normal_axis)
    dA = δV(grid)
    return (sum(Et2 .* boundary_mask) * ds) / (sum(Eall2) * dA)
end

# along-interface line element (the grid pitch perpendicular to the interface normal, 2D)
step_perp(grid::Grid{2}, normal_axis::Int) = normal_axis == 1 ? δy(grid) : δx(grid)

"""
    roughness_scattering_loss(k, evec, ε⁻¹, ∂ε_∂ω, grid; σ, Lc, n1, n2, λ, boundary_mask,
                              normal_axis, both_surfaces=true) -> Real

Geometry-general Payne–Lacey roughness loss in **dB/cm** using the *actual* modal field at
the rough interface (via [`sidewall_field_overlap`](@ref)) in place of the analytic slab
`φ²(d)`:

```math
α = \\frac{(n₁²−n₂²)^2 k_0^3}{4π\\,n_{eff}}\\,σ^2\\,\\langle|E_∥|^2\\rangle_{interface}\\,
    \\tilde S(β,L_c),
```

with the exponential-ACF angular spectrum `S̃ = ∫_0^π 2L_c/(1+(β−n₂k₀\\cosθ)^2 L_c^2)\\,dθ`
evaluated numerically. `neff = k/ω` is taken from the mode. Differentiable in `σ`, `Lc`,
indices, and (through the field overlap) the geometry. Calibrate against
[`payne_lacey_slab_loss`](@ref) / Lee 2001 for a given platform.
"""
function roughness_scattering_loss(k::Real, evec, ε⁻¹, ∂ε_∂ω, grid::Grid; σ::Real, Lc::Real,
        n1::Real, n2::Real, λ::Real, boundary_mask::AbstractArray, normal_axis::Int,
        both_surfaces::Bool=true)
    k0 = 2π / λ
    neff = k / k0
    β = neff * k0
    overlap = sidewall_field_overlap(k, evec, ε⁻¹, ∂ε_∂ω, grid, boundary_mask, normal_axis)
    S̃ = _exp_acf_angular(β, n2 * k0, Lc)
    α_npμm = (n1^2 - n2^2)^2 * k0^3 / (4π * neff) * σ^2 * overlap * S̃
    both_surfaces || (α_npμm *= 0.5)
    return np_per_μm_to_dB_per_cm(α_npμm)
end

# ∫₀^π R̃(β − n2k0 cosθ)/σ² dθ for an exponential ACF, R̃(Ω)/σ² = 2Lc/(1+Ω²Lc²), via
# fixed midpoint quadrature (smooth integrand; differentiable, no branch on θ).
function _exp_acf_angular(β, n2k0, Lc; nθ::Int=400)
    acc = zero(promote_type(typeof(β), typeof(Lc)))
    dθ = π / nθ
    @inbounds for j in 1:nθ
        θ = (j - 0.5) * dθ
        Ω = β - n2k0 * cos(θ)
        acc += 2 * Lc / (1 + (Ω * Lc)^2)
    end
    return acc * dθ
end

# --- Substrate leakage through the lower cladding ----------------------------------------

"""
    substrate_leakage_loss(; neff, n_clad, t_clad, λ, prefactor) -> Real

Substrate-leakage loss in **dB/cm** of a mode whose evanescent tail tunnels through a
finite lower cladding of thickness `t_clad` into a high-index substrate:

```math
α = A\\,\\exp(-2 γ_c t_{clad}),\\qquad γ_c = k_0\\sqrt{n_{eff}^2 - n_{clad}^2},
```

the standard exponential model (Sridaran & Bhave 2010; Bauters 2011). `prefactor` `A`
(dB/cm) is the substrate-coupling strength at `t_clad→0` (fit per platform, or obtained
from a leaky-mode/complex-`neff` solve; e.g. SOI Si strips e-fold the loss every ≈90 nm of
BOX and need ≥1 μm BOX for <1 dB/cm at 1550 nm). Differentiable in `neff`, `t_clad`,
indices and `A` — pairs with the adjoint `∂neff/∂geometry` for leakage-aware design.
"""
function substrate_leakage_loss(; neff::Real, n_clad::Real, t_clad::Real, λ::Real, prefactor::Real)
    k0 = 2π / λ
    γc = k0 * sqrt(max(neff^2 - n_clad^2, 0.0))
    return prefactor * exp(-2 * γc * t_clad)
end
