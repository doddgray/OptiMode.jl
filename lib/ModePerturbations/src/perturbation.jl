# Core first-order modal perturbation theory.
#
# For a converged guided mode `(k, evec)` at frequency `ω` on a `Grid`, with smoothed
# inverse dielectric tensor `ε⁻¹`, a weak dielectric-tensor perturbation `Δε(x,y)` shifts
# the modal wavenumber by the non-degenerate Hellmann–Feynman first-order amount
#
#     Δk = ⟨E|Δε|E⟩ / ( 2 ⟨ev|∂M̂/∂k|ev⟩ )                                            (1)
#
# where M̂ = [∇× ε⁻¹ ∇×] is the plane-wave Helmholtz operator (eigenvalue ω²) and
# ⟨E|Δε|E⟩ = ⟨ev|∇×ᵀ (ε⁻¹ Δε ε⁻¹) ∇×|ev⟩. The denominator `2⟨ev|∂M̂/∂k|ev⟩ = ∂ω²/∂k` is
# exactly `MaxwellEigenmodes.HMₖH`, and the numerator equals `HMH(ev, ε⁻¹·Δε·ε⁻¹)` for a
# real `Δε`. This is the same frozen-mode sensitivity that the umbrella test-suite
# validates against finite differences (the `te00`/`Lw` construction). Here it is packaged
# as a general perturbation engine, generalized to *complex* `Δε` so that absorptive /
# gain perturbations produce a complex `Δk` whose imaginary part is the modal loss.
#
# Sign / units conventions (μm-based, as everywhere in OptiMode):
#   • `Δε` real  ⇒ `Re(Δk)`  ⇒ `Δneff = Re(Δk)/ω`.
#   • `Δε` with `Im(ε)>0` (lossy material, `exp(-iωt)` convention) ⇒ modal power-loss
#     coefficient `α = 2 Im(Δk)` [1/μm]; `α_dB_per_cm = 4.342944e4 · α`.
# `Δε` may be a full `(3,3,size(grid)...)` tensor field or built from a scalar index map
# with [`Δε_from_Δn`](@ref).

export perturbation_Δk, Δneff_perturbation, modal_loss_perturbation, modal_loss_dB_per_cm
export Δε_from_Δn, perturbation_ng_gvd, np_per_μm_to_dB_per_cm

"Convert a power-loss coefficient `α` in nepers/μm to dB/cm."
np_per_μm_to_dB_per_cm(α) = 4.342944819032518e4 * α   # 10/log(10) [dB/Np] × 1e4 [μm/cm]

"""
    Δε_from_Δn(Δn, ε) -> Array

Isotropic dielectric-tensor perturbation `Δε = 2 n₀ Δn` from a scalar index-shift map
`Δn(x,y)` and the (real) unperturbed dielectric tensor `ε` (used only for the local
background index `n₀ = sqrt(tr(ε)/3)` per pixel). Returns a `(3,3,size...)` tensor field
with `Δn`'s eltype (real for thermo-optic, complex for an absorptive `Δn = Δn′ + iΔn″`).
The quadratic `(Δn)²` term is dropped (first order in `Δn`).
"""
function Δε_from_Δn(Δn::AbstractArray, ε::AbstractArray)
    ND = ndims(Δn)
    cols = ntuple(_ -> Colon(), ND)
    n₀ = sqrt.((view(ε, 1, 1, cols...) .+ view(ε, 2, 2, cols...) .+ view(ε, 3, 3, cols...)) ./ 3)
    Δεdiag = 2 .* n₀ .* Δn
    return _diag_tensor_field(Δεdiag)   # isotropic: Δε[a,b] = δ_ab · Δεdiag (non-mutating)
end

# build a (3,3,dims...) diagonal tensor field from a per-pixel scalar map, without mutation
# (Tullio + a constant identity ⇒ reverse-AD friendly, complex-capable)
const _I3 = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
function _diag_tensor_field(d::AbstractArray{<:Number,2})
    I3 = _I3
    @tullio out[a, b, ix, iy] := I3[a, b] * d[ix, iy]
end
function _diag_tensor_field(d::AbstractArray{<:Number,3})
    I3 = _I3
    @tullio out[a, b, ix, iy, iz] := I3[a, b] * d[ix, iy, iz]
end

# Spectral displacement field D̃ = (k+G)×H̃ (FFT of the transverse-basis curl), the object
# the perturbation numerator contracts. Shares `mag`/`mn` with the caller to avoid recompute.
function _mode_D(evec, mag, mn, grid::Grid{ND}) where {ND}
    evg = reshape(vec(evec), (2, size(grid)...))
    return fft(kx_tc(evg, mn, mag), _fftaxes(grid))
end

"""
    perturbation_Δk(k, evec, ε⁻¹, Δε, grid) -> Complex

First-order modal wavenumber shift `Δk` (Eq. 1) for a dielectric-tensor perturbation
`Δε`. `Δε` may be real (index shift) or complex (absorptive/gain). The returned complex
`Δk` gives the effective-index shift `Re(Δk)/ω` and the modal power loss `2 Im(Δk)`.

End-to-end AD compatible (forward & reverse) with respect to `Δε`, `ε⁻¹`, and any
parameters they depend on, via `HMₖH`'s `rrule` and the natively-differentiable FFT /
Tullio contractions; validated against finite differences.
"""
function perturbation_Δk(k::Real, evec, ε⁻¹::AbstractArray, Δε::AbstractArray, grid::Grid)
    mag, mn = mag_mn(k, grid)
    D = _mode_D(evec, mag, mn, grid)
    A = _tt(ε⁻¹, _tt(Δε, ε⁻¹))                  # ε⁻¹ Δε ε⁻¹ (complex-capable)
    AD = _tv(A, D)
    num = _grid_braket(D, AD) / length(grid)    # = ⟨E|Δε|E⟩ (complex-HMH normalization)
    den = 2 * HMₖH(vec(evec), ε⁻¹, mag, mn)     # = ∂ω²/∂k (real)
    return num / den
end

# ⟨D|AD⟩ summed over the grid, complex-capable (cf. LinearAlgebra.dot but Zygote/Enzyme-clean)
function _grid_braket(D::AbstractArray{<:Number,M}, AD::AbstractArray{<:Number,M}) where {M}
    return sum(conj(D) .* AD)
end

"""
    Δneff_perturbation(k, evec, ω, ε⁻¹, Δε, grid) -> Real

First-order effective-index shift `Δneff = Re(Δk)/ω` from a dielectric perturbation `Δε`.
"""
Δneff_perturbation(k::Real, evec, ω::Real, ε⁻¹, Δε, grid::Grid) =
    _perturbation_re(k, evec, ε⁻¹, Δε, grid) / ω

"""
    modal_loss_perturbation(k, evec, ε⁻¹, Δε, grid) -> Real

First-order modal **power**-loss coefficient `α = 2 Im(Δk)` [nepers/μm] from the
imaginary (absorptive) part of `Δε`. Positive for an absorptive perturbation
(`Im(ε) > 0`). See [`modal_loss_dB_per_cm`](@ref) for engineering units.
"""
modal_loss_perturbation(k::Real, evec, ε⁻¹, Δε, grid::Grid) =
    2 * _perturbation_im(k, evec, ε⁻¹, Δε, grid)

"""
    modal_loss_dB_per_cm(k, evec, ε⁻¹, Δε, grid) -> Real

First-order modal power loss in **dB/cm** from an absorptive perturbation `Δε`.
"""
modal_loss_dB_per_cm(k::Real, evec, ε⁻¹, Δε, grid::Grid) =
    np_per_μm_to_dB_per_cm(modal_loss_perturbation(k, evec, ε⁻¹, Δε, grid))

"""
    perturbation_ng_gvd(ks, evecs, εi_s, Δεs, ωs, grid) -> (; Δneff, Δng, ΔGVD)

First-order shifts of the modal effective index, **group index** and **group-velocity
dispersion** from a (possibly frequency-dependent) perturbation, evaluated on a uniform
3-point frequency stencil `ωs = (ω₀-Δ, ω₀, ω₀+Δ)`. `ks`, `evecs`, `εi_s`, `Δεs` are the
length-3 vectors of the *unperturbed* mode wavenumber, eigenvector, inverse dielectric and
the perturbation `Δε` at those frequencies (e.g. from a dispersion sweep — no perturbed
re-solve is required, which is the whole point of the perturbative method).

Since `k = neff·ω` and the perturbation is additive, `Δk(ω) = k_pert(ω) − k₀(ω)` gives
`Δng = d(Δk)/dω` and `ΔGVD = d²(Δk)/dω²` by central differences. Because each `Δk` is AD-
differentiable in the perturbation parameters and the unperturbed modes are fixed, the
returned `Δng`/`ΔGVD` are themselves end-to-end AD compatible w.r.t. those parameters
(temperature, roughness, power, …).
"""
function perturbation_ng_gvd(ks, evecs, εi_s, Δεs, ωs, grid::Grid)
    @assert length(ωs) == 3 "perturbation_ng_gvd expects a 3-point frequency stencil"
    Δ = (ωs[3] - ωs[1]) / 2
    Δk = ntuple(i -> perturbation_Δk(ks[i], evecs[i], εi_s[i], Δεs[i], grid), 3)
    Δneff = real(Δk[2]) / ωs[2]
    Δng = real(Δk[3] - Δk[1]) / (2Δ)
    ΔGVD = real(Δk[3] - 2Δk[2] + Δk[1]) / (Δ^2)
    return (; Δneff, Δng, ΔGVD)
end
