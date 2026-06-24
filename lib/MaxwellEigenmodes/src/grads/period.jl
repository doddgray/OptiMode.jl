################################################################################
#                                                                              #
#   Adjoint sensitivity analysis for 3D waveguides periodic along the          #
#   propagation axis (ẑ): Bragg / photonic-crystal-defect waveguides.          #
#                                                                              #
#   New degree of freedom relative to the 2D cross-section solver: the         #
#   *absolute spatial period* Λ along ẑ.  In a `Grid{3}` the period is the     #
#   z-extent of the unit cell, Λ ≡ grid.Δz.  At fixed (index-space) ε⁻¹ the    #
#   Helmholtz operator M̂(kz,Λ,ε⁻¹) depends on Λ *only* through the z-components #
#   of the reciprocal-lattice vectors,                                         #
#                                                                              #
#         g_{j,z} = m_j / Λ      (m_j ∈ ℤ, the FFT mode index),                #
#                                                                              #
#   so that the per-plane-wave vector  w⃗_j = k⃗ - g⃗_j  has                      #
#                                                                              #
#         ∂w_{j,z}/∂Λ = -∂g_{j,z}/∂Λ = +g_{j,z}/Λ .                            #
#                                                                              #
#   Every place the existing adjoint forms a kz sensitivity by summing a       #
#   per-plane-wave cotangent  w̄_{j,z}  (i.e. perturbing every plane wave by    #
#   one unit of ẑ), the period sensitivity is the SAME sum weighted by the     #
#   per-plane-wave factor  g_{j,z}/Λ.  This file implements that map for both  #
#   the eigenvalue channel (∂ω²/∂Λ, via a weighted Hellmann–Feynman form) and  #
#   the eigenvector channel (the fixed-k dependence of H on Λ), and assembles  #
#   them — together with the Newton inversion of the fixed-ω problem — into    #
#   `solve_k_periodic` and its reverse-mode rule.                             #
#                                                                              #
################################################################################

export solve_k_periodic, gz_field, HMₖH_weighted, ∂ω²_∂Λ, period_weight

"""
    period_weight(grid::Grid{3}) -> Array{T,3}

Per-plane-wave period-perturbation factor ``g_{j,z}/Λ`` (units μm⁻²) used to convert
a kz sensitivity into a period (Λ ≡ `grid.Δz`) sensitivity. Equal to the z-component
of the reciprocal-lattice vector field divided by the period.
"""
function period_weight(grid::Grid{3,T}) where {T<:Real}
    gz = gz_field(grid)
    return gz ./ grid.Δz
end

"""
    gz_field(grid::Grid{3}) -> Array{T,3}

z-component of the reciprocal-lattice vectors ``g_{j,z} = m_j/Λ`` of the periodic
cell, broadcast to the `(Nx,Ny,Nz)` grid shape (independent of ix,iy). These are the
FFT spatial frequencies along ẑ; the absolute period is `Λ = grid.Δz`.
"""
function gz_field(grid::Grid{3,T}) where {T<:Real}
    gzs = my_fftfreq(grid.Nz, grid.Nz / grid.Δz)   # length Nz, values m/Δz
    gz = Array{T}(undef, grid.Nx, grid.Ny, grid.Nz)
    @inbounds for iz in 1:grid.Nz, iy in 1:grid.Ny, ix in 1:grid.Nx
        gz[ix, iy, iz] = gzs[iz]
    end
    return gz
end

"""
    HMₖH_weighted(H, ε⁻¹, mag, mn, wt) -> Real

Per-plane-wave-weighted Hellmann–Feynman quadratic form. Identical to [`HMₖH`](@ref)
(which evaluates ``⟨H|∂M̂/∂k|H⟩`` by replacing one ``(\\vec k+\\vec G)×`` curl by the
constant ``\\hat z×``), except that the ``\\hat z×`` action is multiplied, plane wave
by plane wave, by the real scalar field `wt`. With `wt = g_z/Λ` this evaluates
``⟨H|∂M̂/∂Λ|H⟩`` because ``∂[(\\vec k+\\vec G)×]/∂Λ = (g_z/Λ)\\,\\hat z×`` for a
z-periodic cell. `wt ≡ 1` recovers `HMₖH`.
"""
function HMₖH_weighted(H::AbstractArray{Complex{T},4}, ε⁻¹, mag, mn, wt::AbstractArray{<:Real,3}) where {T<:Real}
    zxH = zx_tc(H, mn)
    zxHw = _mult(wt, zxH)
    -real(dot(H, kx_ct(ifft(ε⁻¹_dot(fft(zxHw, (2:4)), real(flat(ε⁻¹))), (2:4)), mn, mag)))
end

function HMₖH_weighted(H::AbstractVector{Complex{T}}, ε⁻¹, mag::AbstractArray{<:Real,3},
        mn::AbstractArray{<:Real,5}, wt::AbstractArray{<:Real,3}) where {T<:Real}
    Nx, Ny, Nz = size(mag)
    HMₖH_weighted(reshape(H, (2, Nx, Ny, Nz)), ε⁻¹, mag, mn, wt)
end

"""
    ∂ω²_∂Λ(H, ε⁻¹, mag, mn, grid) -> Real

Eigenvalue-channel period derivative ``∂ω²/∂Λ = ⟨H|∂M̂/∂Λ|H⟩`` for a normalized
eigenvector `H` of the z-periodic Helmholtz operator, evaluated by the weighted
Hellmann–Feynman form [`HMₖH_weighted`](@ref) with weight ``g_z/Λ``. The factor of 2
matches the convention ``∂ω²/∂k = 2\\,⟨H|∂M̂/∂k|H⟩`` used throughout the solver.
"""
function ∂ω²_∂Λ(H, ε⁻¹, mag, mn, grid::Grid{3})
    2 * HMₖH_weighted(H, ε⁻¹, mag, mn, period_weight(grid))
end

"""
    _wbar_z(māg, m̄, n̄, mag, m⃗, n⃗) -> Array{T,ND}

z-component of the per-plane-wave Cartesian cotangent ``\\bar{w}_j`` of the shifted
wavevector ``\\vec w_j = \\vec k - \\vec g_j``, reconstructed from the cotangents of
its derived quantities ``(\\mathrm{mag}_j, \\hat m_j, \\hat n_j)``. Using
``\\hat w_j = \\hat m_j × \\hat n_j`` and ``\\hat m_j,\\hat n_j ⊥ \\vec w_j``,

```math
\\bar{w}_j = \\bar{\\mathrm{mag}}_j\\,\\hat w_j
            - \\frac{\\bar m_j·\\hat w_j}{\\mathrm{mag}_j}\\,\\hat m_j
            - \\frac{\\bar n_j·\\hat w_j}{\\mathrm{mag}_j}\\,\\hat n_j .
```

Summing the z-components reproduces `∇ₖmag_m_n(...; dk̂=ẑ)` (a uniform ẑ shift of
every plane wave); weighting by ``g_{j,z}/Λ`` before summing gives the period
sensitivity. `m̄`, `n̄` are the cotangents of the *unit* vectors (i.e. `kx̄_m .* mag`).
"""
function _wbar_z(māg, m̄, n̄, mag, m⃗, n⃗)
    ŵ = cross.(m⃗, n⃗)                       # = (k⃗-g⃗)/|k⃗-g⃗|  (SVector field)
    wbar = māg .* ŵ .- (dot.(m̄, ŵ) ./ mag) .* m⃗ .- (dot.(n̄, ŵ) ./ mag) .* n⃗
    return getindex.(wbar, 3)
end

"""
    _grid_with_period(grid::Grid{3}, Λ) -> Grid{3}

Copy of `grid` whose z-extent (absolute period) is set to `Λ`, leaving the transverse
cell, all pixel counts, and `Nz` unchanged. The (index-space) dielectric array is held
fixed, so this stretches/compresses the modeled structure along ẑ.
"""
function _grid_with_period(grid::Grid{3,T}, Λ::Real) where {T<:Real}
    Grid(grid.Δx, grid.Δy, T(Λ), grid.Nx, grid.Ny, grid.Nz)
end

"""
    solve_k_periodic(ω, ε⁻¹, Λ, grid, solver; nev=1, kwargs...) -> (kmags, evecs)

Propagation constants and transverse eigenvectors of the first `nev` guided Bloch
modes of a **3D waveguide periodic along ẑ** (Bragg or photonic-crystal-defect
waveguide) at frequency `ω` and **absolute spatial period** `Λ`.

`ε⁻¹` is the smoothed inverse-permittivity field of one period (`(3,3,Nx,Ny,Nz)`),
held fixed in index space; `grid` is a `Grid{3}` supplying the transverse cell and the
pixel counts (its `Δz` is overridden by `Λ`). This is `solve_k` evaluated on the grid
`Grid(Δx, Δy, Λ, Nx, Ny, Nz)`.

A `ChainRulesCore.rrule` provides reverse-mode gradients of `(kmags, evecs)` with
respect to `(ω, ε⁻¹, Λ)` at ≈1 extra eigensolve-equivalent cost. The period gradient
`Λ̄` is obtained from the dependence of the Helmholtz operator on the reciprocal-lattice
z-components `g_z = m/Λ`; see [`∂ω²_∂Λ`](@ref) and `_wbar_z`.
"""
function solve_k_periodic(ω::T, ε⁻¹::AbstractArray{T}, Λ::Real, grid::Grid{3,T},
        solver::AbstractEigensolver; nev=1, kwargs...) where {T<:Real}
    solve_k(ω, ε⁻¹, _grid_with_period(grid, Λ), solver; nev, kwargs...)
end

"""
    _solve_k_period_grads(ω, ε⁻¹, Λ, grid, kmags, evecs, k̄mags, ēvecs, solver; nev)
        -> (ω_bar, ei_bar, Λ_bar)

Reverse-mode (vector-Jacobian product) of [`solve_k_periodic`](@ref): map output
cotangents `(k̄mags, ēvecs)` to input cotangents `(ω_bar, ei_bar, Λ_bar)`.

For each requested band the adjoint is assembled from three physically distinct
channels, exactly mirroring how the existing `solve_k` adjoint treats `ε⁻¹`:

* eigenvector channel (from `ēv`, fixed k): solve the adjoint linear system for `λ⃗`
  and back-propagate through M̂'s dependence on `ε⁻¹` (→ `ei_bar`), on the basis
  `(mag, m̂, n̂)` (→ `k̄ₕ`, the kz sensitivity), and — new here — on the period via the
  same per-plane-wave cotangent weighted by `g_z/Λ` (→ `Λ̄_H`).
* combined kz cotangent `k̄tot = k̄ + k̄ₕ` flows through the Newton inversion of the
  fixed-ω problem (`ω²(k,ε,Λ)=ω²`): `∂k/∂ω = 2ω/∂ω²∂k`, `∂k/∂ε ∝ ∂ω²/∂ε`, and the new
  `∂k/∂Λ = -∂ω²∂Λ / ∂ω²∂k`.

`Λ_bar = Λ̄_H + k̄tot·(-∂ω²∂Λ/∂ω²∂k)`.
"""
function _solve_k_period_grads(ω::T, ε⁻¹::AbstractArray{T}, Λ::Real, grid::Grid{3,T},
        kmags, evecs, k̄mags, ēvecs, solver::AbstractEigensolver; nev=1, maxiter=100) where {T<:Real}
    gridsize = size(grid)
    ei_bar = zero(ε⁻¹)
    ω_bar = zero(ω)
    Λ_bar = zero(float(Λ))

    for (eigind, k̄, ēv, k, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
        ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter)
        ms.H⃗[:, eigind] = copy(ev)
        wt = period_weight(grid)
        λd = similar(ms.M̂.d)
        λẽ = similar(ms.M̂.d)
        ∂ω²∂k = 2 * HMₖH(ev, ms.M̂.ε⁻¹, ms.M̂.mag, ms.M̂.mn)
        ∂ω²∂Λ = 2 * HMₖH_weighted(ev, ms.M̂.ε⁻¹, ms.M̂.mag, ms.M̂.mn, wt)
        ev_grid = reshape(ev, (2, gridsize...))

        k̄ = isa(k̄, AbstractZero) ? zero(T) : T(k̄)

        if !isa(ēv, AbstractZero)
            # --- eigenvector channel (fixed k): solve adjoint linear system ---
            λ⃗ = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=randn(eltype(ev), size(ev)), P̂=ms.P̂)
            λ⃗ -= dot(ev, λ⃗) * ev
            λ = reshape(λ⃗, (2, gridsize...))
            _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms)
            λd = _H2d!(λd, λ, ms)
            ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...)
            # back-propagate to (mag, m̂, n̂) → kz and Λ sensitivities
            λd *= ms.M̂.Ninv
            λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ, λd, ms))
            ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e, ms.M̂.d, ms))
            kx̄_m⃗ = real.(λẽ_sv .* conj.(view(ev_grid, 2, axes(grid)...)) .+ ẽ .* conj.(view(λ, 2, axes(grid)...)))
            kx̄_n⃗ = -real.(λẽ_sv .* conj.(view(ev_grid, 1, axes(grid)...)) .+ ẽ .* conj.(view(λ, 1, axes(grid)...)))
            m⃗ = reinterpret(reshape, SVector{3,T}, ms.M̂.mn[:, 1, axes(grid)...])
            n⃗ = reinterpret(reshape, SVector{3,T}, ms.M̂.mn[:, 2, axes(grid)...])
            māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
            wbar_z = _wbar_z(māg, kx̄_m⃗ .* ms.M̂.mag, kx̄_n⃗ .* ms.M̂.mag, ms.M̂.mag, m⃗, n⃗)
            k̄ₕ = -sum(wbar_z)
            Λ̄_H = -sum(wbar_z .* wt)
        else
            k̄ₕ = zero(T)
            Λ̄_H = zero(float(Λ))
        end

        # --- combined kz cotangent through the Newton (fixed-ω) inversion ---
        k̄tot = k̄ + k̄ₕ
        λ⃗ = (k̄tot / ∂ω²∂k) * ev
        _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms)
        λd = _H2d!(λd, reshape(λ⃗, (2, gridsize...)), ms)
        ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...)
        ω_bar += 2 * ω * k̄tot / ∂ω²∂k
        Λ_bar += Λ̄_H + k̄tot * (-∂ω²∂Λ / ∂ω²∂k)
    end
    return ω_bar, ei_bar, Λ_bar
end

function rrule(::typeof(solve_k_periodic), ω::T, ε⁻¹::AbstractArray{T}, Λ::Real,
        grid::Grid{3,T}, solver::AbstractEigensolver; nev=1, maxiter=100, kwargs...) where {T<:Real}
    g = _grid_with_period(grid, Λ)
    kmags, evecs = solve_k(ω, ε⁻¹, g, solver; nev, maxiter, kwargs...)
    function solve_k_periodic_pullback(ΔΩ)
        k̄mags, ēvecs = ΔΩ
        ω_bar, ei_bar, Λ_bar = _solve_k_period_grads(ω, ε⁻¹, Λ, g, kmags, evecs,
            k̄mags, ēvecs, solver; nev, maxiter)
        return (NoTangent(), ω_bar, ei_bar, Λ_bar, NoTangent(), NoTangent())
    end
    return ((kmags, evecs), solve_k_periodic_pullback)
end
