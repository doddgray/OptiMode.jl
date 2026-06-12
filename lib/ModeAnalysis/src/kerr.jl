# Kerr nonlinearity: first-order power-dependent mode corrections.
#
# A material's intensity-dependent refractive index n(I) = n₀ + n₂·I (n₂ in μm²/W,
# specified per material in MaterialDispersion and mapped onto the grid e.g. with
# `DielectricSmoothing.smooth_scalar`) perturbs the dielectric tensor as
#
#     Δε(x,y) ≈ 2 n₀(x,y) Δn(x,y),   Δn(x,y) = n₂(x,y) · I(x,y),
#
# where I(x,y) is the modal intensity (z-directed Poynting flux density) normalized so
# that ∫ I dA equals the specified optical power P (in W). `solve_k_kerr` applies this
# as a first-order correction: each mode is re-solved with the perturbed dielectric
# tensor computed from *its own* intensity profile, assuming the full power P resides
# in that mode (no cross coupling between modes).

export poynting_z, mode_intensity, kerr_dielectric_perturbation, solve_k_kerr

"""
    poynting_z(k, evec, ε⁻¹, grid) -> Array

The z-component of the time-averaged Poynting vector of a mode, `Re(E × H*)·ẑ`, on the
spatial grid (arbitrary overall scale — normalize with [`mode_intensity`](@ref)).
"""
function poynting_z(k::Real, evec::AbstractVector, ε⁻¹::AbstractArray, grid::Grid{ND}) where {ND}
    fftax = _fftaxes(grid)
    evg = reshape(evec, (2, size(grid)...))
    mag, mn = mag_mn(k, grid)
    # E ∝ ε⁻¹ (k+g)×H̃ with a *real* proportionality factor relative to H ∝ H̃: the i in
    # D = (i/ω)∇×H cancels the i of the spectral curl, so no 1im here (unlike `E⃗`,
    # where only |E|² matters) — otherwise Re(E×H*) loses the E–H phase coherence.
    E = _dot(ε⁻¹, fft(kx_tc(evg, mn, mag), fftax))
    H = fft(tc(evg, mn), fftax)
    Ex = view(E, 1, ntuple(_ -> Colon(), ND)...)
    Ey = view(E, 2, ntuple(_ -> Colon(), ND)...)
    Hx = view(H, 1, ntuple(_ -> Colon(), ND)...)
    Hy = view(H, 2, ntuple(_ -> Colon(), ND)...)
    return real.(Ex .* conj.(Hy) .- Ey .* conj.(Hx))
end

"""
    mode_intensity(k, evec, ε⁻¹, grid, P) -> Array

Modal intensity distribution `I(x,y)` (units of `P` per μm², i.e. W/μm² for `P` in W),
normalized such that `sum(I) * δV(grid) == P` — the mode carries total power `P`.
"""
function mode_intensity(k::Real, evec::AbstractVector, ε⁻¹::AbstractArray, grid::Grid, P::Real)
    Sz = poynting_z(k, evec, ε⁻¹, grid)
    return Sz .* (P / (sum(Sz) * δV(grid)))
end

"""
    kerr_dielectric_perturbation(I, n₂, ε) -> (Δε, Δn)

First-order Kerr dielectric perturbation from an intensity map `I` (W/μm²) and a Kerr
coefficient map `n₂` (μm²/W): `Δn = n₂ .* I` and the (isotropic, diagonal) tensor
perturbation `Δε[a,a] = 2 n₀ Δn` with `n₀ = sqrt(tr(ε)/3)` per pixel.
"""
function kerr_dielectric_perturbation(I::AbstractArray, n₂::AbstractArray, ε::AbstractArray)
    ND = ndims(I)
    Δn = n₂ .* I
    n₀ = sqrt.((view(ε, 1, 1, ntuple(_ -> Colon(), ND)...) .+
                view(ε, 2, 2, ntuple(_ -> Colon(), ND)...) .+
                view(ε, 3, 3, ntuple(_ -> Colon(), ND)...)) ./ 3)
    Δε = zeros(eltype(ε), size(ε))
    for a in 1:3
        view(Δε, a, a, ntuple(_ -> Colon(), ND)...) .= 2 .* n₀ .* Δn
    end
    return Δε, Δn
end

"""
    solve_k_kerr(ω, P, ε⁻¹, ∂ε_∂ω, n₂, grid, solver; nev=1, solver_kwargs...)

Power-dependent mode solve with a first-order Kerr (intensity-dependent index)
correction. After a linear `solve_k`, each band `b` is corrected independently by

1. computing its intensity profile `I_b(x,y)` for total optical power `P` (in W,
   assumed to reside entirely in that mode — no cross coupling between modes),
2. forming the index perturbation `Δn = n₂ .* I_b` (`n₂` map in μm²/W, e.g. from
   `smooth_scalar`; materials without a specified `n₂` contribute 0) and the
   corresponding dielectric perturbation `Δε = 2 n₀ Δn`,
3. re-solving for band `b` with `ε + Δε`.

Returns `(; kmags, evecs, kmags_lin, evecs_lin, dn_max)`: the corrected and linear
wavenumbers/eigenvectors and the peak index shift per band. The power-dependent
effective-index change of band `b` is `(kmags[b] - kmags_lin[b]) / ω`. With `P = 0`
the corrected results equal the linear ones.
"""
function solve_k_kerr(ω::Real, P::Real, ε⁻¹::AbstractArray, ∂ε_∂ω::AbstractArray,
    n₂::AbstractArray, grid::Grid, solver; nev::Int=1, solver_kwargs...)
    kmags_lin, evecs_lin = solve_k(ω, copy(ε⁻¹), grid, solver; nev, solver_kwargs...)
    kmags = copy(kmags_lin)
    evecs = copy.(evecs_lin)
    dn_max = zeros(nev)
    (P > 0 && any(!iszero, n₂)) || return (; kmags, evecs, kmags_lin, evecs_lin, dn_max)
    ε = sliceinv_3x3(ε⁻¹)
    for b in 1:nev
        I = mode_intensity(kmags_lin[b], evecs_lin[b], ε⁻¹, grid, P)
        Δε, Δn = kerr_dielectric_perturbation(I, n₂, ε)
        ε⁻¹_NL = sliceinv_3x3(ε .+ Δε)
        k_NL, ev_NL = solve_k(ω, ε⁻¹_NL, grid, solver; nev=b,
            kguess=kmags_lin[b], solver_kwargs...)
        kmags[b] = k_NL[b]
        evecs[b] = ev_NL[b]
        dn_max[b] = maximum(Δn)
    end
    return (; kmags, evecs, kmags_lin, evecs_lin, dn_max)
end
