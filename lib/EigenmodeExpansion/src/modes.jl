# ──────────────────────────────────────────────────────────────────────────────
#  Per-cell mode solving and MEOW-style modal inner products.
#
#  For each cell the cross-section is smoothed onto the simulation grid and the
#  OptiMode plane-wave eigensolver (`solve_k`, adjoint-differentiable) is run. The
#  E and H of each mode are reconstructed as a *consistent* electromagnetic pair
#  (`E = i ε⁻¹ (k+G)×H`, `H = FFT[transverse H]` — the convention behind
#  `MaxwellEigenmodes.S⃗`) and then power-normalized exactly as MEOW does, so the
#  self-overlap of every mode is 1 and the interface formulas can be used in their
#  `G = I` form.
# ──────────────────────────────────────────────────────────────────────────────

export Mode, inner_product, overlap_matrix, solve_cell_modes, cell_dielectric, mode_fields

using MaterialDispersion: _f_ε_mats
using DielectricSmoothing: smooth_ε, Grid, δV
using MaxwellEigenmodes: solve_k, sliceinv_3x3, mag_mn, kx_tc, tc, ε⁻¹_dot,
    AbstractEigensolver, KrylovKitEigsolve
using FFTW: fft

"""
    Mode

A solved cross-section eigenmode. `k` is the propagation constant β (μm⁻¹),
`neff = k/ω`, and `E`/`H` are the `(3, Nx, Ny)` complex transverse-plus-axial
field arrays on the cross-section `grid`. Modes are power-normalized so that
`inner_product(mode, mode) ≈ 1`.
"""
struct Mode{T,G}
    ω::T
    k::T
    neff::Complex{T}
    E::Array{Complex{T},3}
    H::Array{Complex{T},3}
    grid::G
end

const Modes{T,G} = Vector{Mode{T,G}}

"""
    cell_dielectric(cross_section, materials, ω, grid) -> (ε⁻¹, ∂ε_∂ω)

Smooth a [`CrossSection`](@ref) onto `grid` at frequency `ω`, returning the
inverse dielectric tensor field and its frequency derivative (the inputs the
eigensolver and group-index machinery expect). Differentiable in `ω` and in any
geometry parameters that flow through the cross-section shapes.
"""
function cell_dielectric(cs::CrossSection, materials, ω, grid)
    f_ε, _ = _f_ε_mats(materials, (:ω,))
    mat_vals = f_ε([ω])
    sm = smooth_ε(Tuple(cs.shapes), mat_vals, Tuple(cs.minds), grid)
    ε⁻¹ = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
    ∂ε_∂ω = copy(selectdim(sm, 3, 2))
    return ε⁻¹, ∂ε_∂ω
end

"""
    mode_fields(k, evec, ε⁻¹, grid) -> (E, H)

Reconstruct the real-space electric and magnetic fields of an eigenvector as a
*consistent* electromagnetic pair — `E = i ε⁻¹ (k+G)×H`, `H = FFT[transverse H]`,
the exact convention behind `MaxwellEigenmodes.S⃗` (so `Re(E* × H)·ẑ` is the
physical power flux). Returns `(3, size(grid)...)` complex arrays. Using one
function for both fields avoids the normalization mismatch between the standalone
`E⃗`/`H⃗` helpers.
"""
function mode_fields(k, evec, ε⁻¹, grid::Grid{ND}) where {ND}
    Ns = size(grid)
    mag, mn = mag_mn(k, grid)
    Hₜ = reshape(evec, (2, Ns...))
    D = fft(kx_tc(Hₜ, mn, mag), 2:1+ND)
    E = 1im .* ε⁻¹_dot(D, ε⁻¹)
    H = fft(tc(Hₜ, mn), 2:1+ND)
    return E, H
end

"reconstruct a power-normalized [`Mode`](@ref) from an eigensolver solution"
function build_mode(ω, k, evec, ε⁻¹, ∂ε_∂ω, grid; conjugate::Bool=false)
    E, H = mode_fields(k, evec, ε⁻¹, grid)
    m = Mode(ω, k, complex(k / ω), E, H, grid)
    nrm = sqrt(inner_product(m, m; conjugate))
    return Mode(ω, k, complex(k / ω), E ./ nrm, H ./ nrm, grid)
end

"""
    solve_cell_modes(cell, materials, ω, grid, solver; nev=2, conjugate=false, kwargs...) -> Modes

Solve the `nev` lowest modes of `cell`'s cross-section. `kwargs` are forwarded to
`MaxwellEigenmodes.solve_k` (e.g. `k_tol`, `maxiter`). The returned modes are the
modal basis for that cell in the EME expansion.
"""
function solve_cell_modes(cell::Cell, materials, ω, grid, solver::AbstractEigensolver=KrylovKitEigsolve();
                          nev::Int=2, conjugate::Bool=false, kwargs...)
    ε⁻¹, ∂ε_∂ω = cell_dielectric(cell.cross_section, materials, ω, grid)
    kmags, evecs = solve_k(ω, ε⁻¹, grid, solver; nev, kwargs...)
    return [build_mode(ω, kmags[i], evecs[i], ε⁻¹, ∂ε_∂ω, grid; conjugate) for i in 1:nev]
end

# ── MEOW modal inner product ─────────────────────────────────────────────────

"""
    inner_product(mode1, mode2; conjugate=false) -> Complex

The MEOW modal overlap `½ ∫ (E₁ₓ H₂ᵧ − E₁ᵧ H₂ₓ) dA` over the cross-section — the
ẑ-component of the transverse `E₁ × H₂` flux. With `conjugate=true` the fields of
`mode1` are conjugated (the power-conserving / Lorentz form); the default
unconjugated form matches MEOW's default and its `G = I` interface formulas.
"""
function inner_product(m1::Mode, m2::Mode; conjugate::Bool=false)
    e1x = @view m1.E[1, :, :]
    e1y = @view m1.E[2, :, :]
    h2x = @view m2.H[1, :, :]
    h2y = @view m2.H[2, :, :]
    if conjugate
        integrand = conj.(e1x) .* h2y .- conj.(e1y) .* h2x
    else
        integrand = e1x .* h2y .- e1y .* h2x
    end
    return 0.5 * sum(integrand) * δV(m1.grid)
end

"""
    overlap_matrix(modes1, modes2; conjugate=false) -> Matrix{ComplexF64}

`M[i, j] = inner_product(modes1[i], modes2[j])`, the building block of the
interface S-matrix (used for both self- and cross-overlaps).
"""
function overlap_matrix(modes1, modes2; conjugate::Bool=false)
    M = [inner_product(a, b; conjugate) for a in modes1, b in modes2]
    return M
end
