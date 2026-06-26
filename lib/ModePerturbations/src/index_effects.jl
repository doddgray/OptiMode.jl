# Index-perturbation effects: thermo-optic tuning and arbitrary user-specified Δn(x,y).
#
# Both reduce to a real dielectric perturbation Δε = 2 n₀ Δn fed to the core engine.
#   • Thermo-optic:  Δn(x,y) = (dn/dT)(x,y) · ΔT, with the per-material thermo-optic
#     coefficient map dn/dT(x,y) (e.g. from `DielectricSmoothing.smooth_scalar` of the
#     material `dn_dT` values), reproducing the resonance-shift relation
#     Δλ/λ = Δneff/n_g with Δneff = Σ_i Γ_i (dn_i/dT) ΔT (Snyder & Love; Arbabi &
#     Goddard, Opt. Lett. 38, 3878 (2013)).
#   • User Δn:  any spatial index-perturbation map (stress, doping, electro-optic, …).

export thermo_optic_Δε, thermo_optic_Δneff, thermo_optic_dneff_dT
export index_perturbation_Δneff, index_perturbation_Δk, confinement_factor
export resonance_shift_dλ_dT

"""
    confinement_factor(k, evec, ε⁻¹, ∂ε_∂ω, region_mask, grid) -> Real

Energy/group-velocity confinement factor of a mode in the spatial region selected by the
boolean/real `region_mask(x,y)` (1 inside, 0 outside),

```math
Γ = \\frac{\\iint_{region} ε |E|^2 dA}{\\iint ε |E|^2 dA},
```

with the dispersive energy-normalized field `E` (so `Γ` is the fraction of modal
electric energy in the region). This is the overlap factor in `Δneff ≈ Γ·Δn` for a
uniform index change confined to the region.
"""
function confinement_factor(k::Real, evec, ε⁻¹, ∂ε_∂ω, region_mask::AbstractArray, grid::Grid)
    E = E⃗(k, copy(vec(evec)), ε⁻¹, ∂ε_∂ω, grid; canonicalize=true, normalized=true)
    ε = sliceinv_3x3(copy(ε⁻¹))
    u = real(_quad_field(E, ε))                      # ε|E|² density per pixel
    return sum(u .* region_mask) / sum(u)
end

# per-pixel ε-weighted energy density Σ_{a,b} conj(E[a]) ε[a,b] E[b]
function _quad_field(E::AbstractArray{<:Number,3}, ε::AbstractArray{<:Number,4})
    @tullio u[ix, iy] := real(conj(E[a, ix, iy]) * ε[a, b, ix, iy] * E[b, ix, iy])
end
function _quad_field(E::AbstractArray{<:Number,4}, ε::AbstractArray{<:Number,5})
    @tullio u[ix, iy, iz] := real(conj(E[a, ix, iy, iz]) * ε[a, b, ix, iy, iz] * E[b, ix, iy, iz])
end

"""
    thermo_optic_Δε(dndT_map, ΔT, ε) -> Array

Dielectric perturbation `Δε = 2 n₀ (dn/dT) ΔT` for a temperature change `ΔT` (K), from a
per-pixel thermo-optic-coefficient map `dndT_map` (1/K) and the unperturbed dielectric
tensor `ε`. `dndT_map` is typically `smooth_scalar(shapes, [dn_dT(mat)...], minds, grid)`.
"""
thermo_optic_Δε(dndT_map::AbstractArray, ΔT::Real, ε::AbstractArray) =
    Δε_from_Δn(dndT_map .* ΔT, ε)

"""
    thermo_optic_Δneff(k, evec, ω, ε⁻¹, dndT_map, ΔT, grid) -> Real

First-order thermo-optic effective-index shift `Δneff(ΔT)`. AD-compatible in `ΔT` and the
thermo-optic-coefficient map.
"""
function thermo_optic_Δneff(k::Real, evec, ω::Real, ε⁻¹, dndT_map::AbstractArray, ΔT::Real, grid::Grid)
    ε = sliceinv_3x3(copy(ε⁻¹))
    Δε = thermo_optic_Δε(dndT_map, ΔT, ε)
    return Δneff_perturbation(k, evec, ω, ε⁻¹, Δε, grid)
end

"""
    thermo_optic_dneff_dT(k, evec, ω, ε⁻¹, dndT_map, grid) -> Real

Thermo-optic coefficient of the *effective index*, `dneff/dT` (1/K) — the `ΔT→0` slope of
[`thermo_optic_Δneff`](@ref), `Σ_i Γ_i (dn_i/dT)` in the confinement-factor picture.
"""
thermo_optic_dneff_dT(k::Real, evec, ω::Real, ε⁻¹, dndT_map::AbstractArray, grid::Grid) =
    thermo_optic_Δneff(k, evec, ω, ε⁻¹, dndT_map, one(eltype(dndT_map)), grid)

"""
    resonance_shift_dλ_dT(dneff_dT, ng, λ) -> Real

Resonator thermo-optic wavelength-shift coefficient `dλ/dT = (λ/n_g)·dneff/dT` (same
length units as `λ`), from the modal `dneff/dT`, group index `n_g`, and resonance
wavelength `λ`. (Thermal expansion is neglected; add `λ·neff·α_L/n_g` if needed.)
"""
resonance_shift_dλ_dT(dneff_dT::Real, ng::Real, λ::Real) = (λ / ng) * dneff_dT

"""
    index_perturbation_Δk(k, evec, ε⁻¹, Δn_map, grid) -> Complex
    index_perturbation_Δneff(k, evec, ω, ε⁻¹, Δn_map, grid) -> Real

First-order modal response to an arbitrary user-specified scalar index-perturbation map
`Δn_map(x,y)` (real for a pure index change; complex `Δn′+iΔn″` to also carry absorption).
`Δk` is complex (`Re/ω → Δneff`, `2 Im → modal loss`).
"""
function index_perturbation_Δk(k::Real, evec, ε⁻¹, Δn_map::AbstractArray, grid::Grid)
    ε = sliceinv_3x3(copy(real(ε⁻¹)))
    Δε = Δε_from_Δn(Δn_map, ε)
    return perturbation_Δk(k, evec, ε⁻¹, Δε, grid)
end

index_perturbation_Δneff(k::Real, evec, ω::Real, ε⁻¹, Δn_map::AbstractArray, grid::Grid) =
    real(index_perturbation_Δk(k, evec, ε⁻¹, Δn_map, grid)) / ω
