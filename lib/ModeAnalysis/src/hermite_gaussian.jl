################################################################################
#                                                                              #
#                            hermite_gaussian.jl:                              #
#        Mode classification by least-squares fitting of elliptical            #
#        Hermite–Gaussian (HG) field templates to a mode's transverse          #
#        electric field.                                                       #
#                                                                              #
################################################################################
#
# The functions in `analyze.jl` (`count_E_nodes` / `mode_viable` / `mode_idx`)
# label a guided mode by (i) its dominant polarization axis from the relative
# E-field power and (ii) its Hermite–Gaussian order from counting field-amplitude
# zero crossings (nodes) along the two transverse axes through the field maximum.
# Node counting is fast but brittle: it depends on a `rel_amp_min` threshold, it
# can miscount near mode crossings or when the nodal lines are not axis-aligned,
# and it gives no measure of *how well* the field actually resembles a clean
# Hermite–Gaussian of that order.
#
# This file implements an alternative, threshold-free labeling scheme. For a mode
# field E⃗(x,y) we model the dominant transverse component as a single elliptical
# Hermite–Gaussian
#
#   ψ_{mn}(x,y) = H_m(√2 (x-x₀)/w_x) · H_n(√2 (y-y₀)/w_y) ·
#                 exp[ -((x-x₀)²/w_x² + (y-y₀)²/w_y²) ]
#
# and, for every candidate polarization p ∈ {x (TE), y (TM)} and every order
# (m,n) up to some maximum, optimize the four shape parameters (x₀,y₀,w_x,w_y)
# (the amplitude is eliminated analytically by linear projection) to minimize the
# squared error against the *full* transverse field. The shape parameters are
# initialized at the field's intensity centroid with widths chosen so that the
# template's second moments match the field's (the "matched-variance" guess). The
# mode is labeled by the polarization/order whose optimized fit has the smallest
# residual squared error, which simultaneously yields a normalized fit-quality
# metric in [0,1].
#
# The optimizer is a small self-contained Nelder–Mead simplex (widths optimized in
# log-space to keep them positive) so the package gains no new dependencies.

export hermite_H, hg_template, hg_field_centroid_variance,
    fit_hg_order, hg_mode_label, hg_fit_residuals, hg_label_string

"""
    hermite_H(n, x)

Physicist's Hermite polynomial ``H_n(x)`` evaluated by the stable upward
recurrence ``H_{k+1} = 2x H_k - 2k H_{k-1}`` (``H_0=1``, ``H_1=2x``). Works for a
scalar or a broadcastable array `x`.
"""
function hermite_H(n::Integer, x)
    n == 0 && return one.(x)
    n == 1 && return 2 .* x
    Hkm1 = one.(x)        # H_0
    Hk = 2 .* x           # H_1
    for k in 1:(n-1)
        Hkp1 = 2 .* x .* Hk .- (2k) .* Hkm1
        Hkm1, Hk = Hk, Hkp1
    end
    return Hk
end

# scalar fast path (avoids allocation inside the inner optimization loop)
function hermite_H(n::Integer, x::Real)
    n == 0 && return one(x)
    n == 1 && return 2x
    Hkm1 = one(x)
    Hk = 2x
    for k in 1:(n-1)
        Hkp1 = 2x * Hk - 2k * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end
    return Hk
end

"""
    hg_template(m, n, xs, ys, x₀, y₀, wx, wy) -> Matrix

Real `(length(xs), length(ys))` elliptical Hermite–Gaussian field template of
transverse order `(m,n)`, centered at `(x₀,y₀)` with ``1/e^2`` intensity widths
`(wx, wy)`:

```math
ψ_{mn}(x,y) = H_m\\!\\left(\\frac{\\sqrt2\\,(x-x_0)}{w_x}\\right)
              H_n\\!\\left(\\frac{\\sqrt2\\,(y-y_0)}{w_y}\\right)
              \\exp\\!\\left[-\\frac{(x-x_0)^2}{w_x^2}-\\frac{(y-y_0)^2}{w_y^2}\\right].
```

The template is returned unnormalized; the fitting routines determine its overall
amplitude by linear least squares, so any constant prefactor is irrelevant.
"""
function hg_template(m::Integer, n::Integer, xs::AbstractVector{T}, ys::AbstractVector{T},
        x₀::Real, y₀::Real, wx::Real, wy::Real) where {T<:Real}
    s2 = sqrt(2)
    ψx = [hermite_H(m, s2 * (x - x₀) / wx) * exp(-((x - x₀) / wx)^2) for x in xs]
    ψy = [hermite_H(n, s2 * (y - y₀) / wy) * exp(-((y - y₀) / wy)^2) for y in ys]
    return ψx * ψy'    # outer product, (Nx × Ny)
end

"""
    hg_field_centroid_variance(F, xs, ys) -> (x₀, y₀, σx², σy², power)

Intensity-weighted centroid and per-axis variance of a real 2D field amplitude
`F` (intensity `F.^2`) sampled on the coordinate vectors `xs`, `ys`. Used to seed
the Hermite–Gaussian fit. `power = Σ F²` is returned for convenience.
"""
function hg_field_centroid_variance(F::AbstractMatrix{T}, xs::AbstractVector{T},
        ys::AbstractVector{T}) where {T<:Real}
    I = F .^ 2
    P = sum(I)
    P ≤ 0 && return (zero(T), zero(T), one(T), one(T), zero(T))
    x₀ = sum(I[i, j] * xs[i] for i in eachindex(xs), j in eachindex(ys)) / P
    y₀ = sum(I[i, j] * ys[j] for i in eachindex(xs), j in eachindex(ys)) / P
    σx² = sum(I[i, j] * (xs[i] - x₀)^2 for i in eachindex(xs), j in eachindex(ys)) / P
    σy² = sum(I[i, j] * (ys[j] - y₀)^2 for i in eachindex(xs), j in eachindex(ys)) / P
    return (x₀, y₀, σx², σy², P)
end

# Residual of the best-amplitude fit of template ψ to field f:
#   min_A ‖f − A ψ‖² = ‖f‖² − ⟨f,ψ⟩² / ⟨ψ,ψ⟩.
# Returns (residual, amplitude). `fnorm2 = ‖f‖²` is passed in (constant over the fit).
@inline function _projected_residual(f::AbstractMatrix, ψ::AbstractMatrix, fnorm2::Real)
    fψ = zero(eltype(ψ))
    ψψ = zero(eltype(ψ))
    @inbounds @simd for k in eachindex(f, ψ)
        fψ += f[k] * ψ[k]
        ψψ += ψ[k] * ψ[k]
    end
    ψψ ≤ 0 && return (fnorm2, zero(fψ))
    A = fψ / ψψ
    resid = fnorm2 - fψ^2 / ψψ
    return (max(resid, zero(resid)), A)
end

# Minimal Nelder–Mead simplex minimizer for the 4 shape parameters
# θ = (x₀, y₀, log wx, log wy). Self-contained to avoid an Optim.jl dependency.
function _nelder_mead(cost, θ0::Vector{T}; maxiters::Int=300, xtol::T=T(1e-6),
        ftol::T=T(1e-10), step::T=T(0.25)) where {T<:Real}
    nd = length(θ0)
    # initial simplex
    simplex = [copy(θ0) for _ in 1:(nd+1)]
    for i in 1:nd
        simplex[i+1][i] += step * (abs(θ0[i]) > 1 ? abs(θ0[i]) : one(T))
    end
    fvals = [cost(s) for s in simplex]
    α, γ, ρ, σ = T(1.0), T(2.0), T(0.5), T(0.5)   # reflect, expand, contract, shrink
    for _ in 1:maxiters
        perm = sortperm(fvals)
        simplex, fvals = simplex[perm], fvals[perm]
        # convergence: simplex small in both parameter and function value
        fspread = abs(fvals[end] - fvals[1])
        xspread = maximum(maximum(abs, simplex[i] .- simplex[1]) for i in 2:(nd+1))
        (fspread ≤ ftol * (abs(fvals[1]) + ftol) && xspread ≤ xtol) && break
        # centroid of all but worst
        xc = reduce(.+, simplex[1:nd]) ./ nd
        xr = xc .+ α .* (xc .- simplex[end])      # reflection
        fr = cost(xr)
        if fr < fvals[1]
            xe = xc .+ γ .* (xr .- xc)            # expansion
            fe = cost(xe)
            if fe < fr
                simplex[end], fvals[end] = xe, fe
            else
                simplex[end], fvals[end] = xr, fr
            end
        elseif fr < fvals[nd]
            simplex[end], fvals[end] = xr, fr
        else
            xk = xc .+ ρ .* (simplex[end] .- xc)  # contraction
            fk = cost(xk)
            if fk < fvals[end]
                simplex[end], fvals[end] = xk, fk
            else
                for i in 2:(nd+1)                  # shrink toward best
                    simplex[i] = simplex[1] .+ σ .* (simplex[i] .- simplex[1])
                    fvals[i] = cost(simplex[i])
                end
            end
        end
    end
    i = argmin(fvals)
    return simplex[i], fvals[i]
end

"""
    fit_hg_order(F, xs, ys, m, n; init, maxiters, fnorm2) -> NamedTuple

Fit a single elliptical Hermite–Gaussian of order `(m,n)` to the real field
amplitude `F` sampled on `xs`, `ys`, minimizing the squared error
``\\min_A ‖F - A\\,ψ_{mn}‖²`` over the shape parameters `(x₀,y₀,wx,wy)` (the
amplitude `A` is eliminated by linear projection).

The shape parameters are seeded from `init = (x₀,y₀,σx²,σy²)` (typically the field
[`hg_field_centroid_variance`](@ref)): the centroid sets `(x₀,y₀)` and the widths
are set to `wx = 2√(σx²/(2m+1))`, `wy = 2√(σy²/(2n+1))` so that the template's
transverse second moments initially match the field's ("matched-variance" guess).

Returns `(; residual, rel_residual, amplitude, x₀, y₀, wx, wy)` where
`rel_residual = residual / ‖F‖²` ∈ [0,1] is the normalized misfit.
"""
function fit_hg_order(F::AbstractMatrix{T}, xs::AbstractVector{T}, ys::AbstractVector{T},
        m::Integer, n::Integer; init=nothing, maxiters::Int=300,
        fnorm2::Union{Nothing,T}=nothing) where {T<:Real}
    f2 = fnorm2 === nothing ? sum(abs2, F) : fnorm2
    if init === nothing
        x₀, y₀, σx², σy², _ = hg_field_centroid_variance(F, xs, ys)
    else
        x₀, y₀, σx², σy² = init
    end
    # matched-variance width seed: |H_k·gaussian|² has x-variance (w²/4)(2k+1)
    wx0 = 2 * sqrt(max(σx², eps(T)) / (2m + 1))
    wy0 = 2 * sqrt(max(σy², eps(T)) / (2n + 1))
    θ0 = T[x₀, y₀, log(wx0), log(wy0)]
    cost = function (θ)
        ψ = hg_template(m, n, xs, ys, θ[1], θ[2], exp(θ[3]), exp(θ[4]))
        first(_projected_residual(F, ψ, f2))
    end
    θ, resid = _nelder_mead(cost, θ0; maxiters)
    ψ = hg_template(m, n, xs, ys, θ[1], θ[2], exp(θ[3]), exp(θ[4]))
    _, A = _projected_residual(F, ψ, f2)
    return (; residual=resid, rel_residual=(f2 > 0 ? resid / f2 : zero(T)),
        amplitude=A, x₀=θ[1], y₀=θ[2], wx=exp(θ[3]), wy=exp(θ[4]))
end

# Rotate the global phase of the complex transverse field so that the L2 energy of
# its real part is maximized, then return the real transverse components (Fx, Fy).
# For a guided mode the in-plane components Ex, Ey are nearly co-phased (and ⊥ to the
# π/2-shifted longitudinal Ez), so this recovers the physical standing-wave profile.
function _real_transverse(E::AbstractArray{Complex{T},3}) where {T<:Real}
    Ex = @view E[1, :, :]
    Ey = @view E[2, :, :]
    S = zero(Complex{T})
    for k in eachindex(Ex)
        S += Ex[k]^2 + Ey[k]^2
    end
    φ = -angle(S) / 2
    c = cis(φ)
    Fx = real.(c .* Ex)
    Fy = real.(c .* Ey)
    return Fx, Fy
end

"""
    hg_fit_residuals(E, grid; max_order=4, maxiters=300) -> NamedTuple

Fit elliptical Hermite–Gaussian templates of every transverse order
`(m,n)`, `0 ≤ m,n ≤ max_order`, to both transverse polarizations of a mode field
`E` (a complex `(3, Nx, Ny)` array as returned by [`E⃗`](@ref)) on `grid`.

For a candidate polarization `p ∈ {x (TE), y (TM)}` and order `(m,n)` the fit
models the `p`-component of the real transverse field with one Hermite–Gaussian and
counts all power in the *other* transverse component as residual, so the normalized
error

```math
\\text{err}(p,m,n) = \\frac{\\min_A‖F_p - A\\,ψ_{mn}‖² + ‖F_{q≠p}‖²}{‖F_x‖²+‖F_y‖²}
```

simultaneously measures Hermite–Gaussian order *and* polarization purity.

Returns `(; err_TE, err_TM, fits_TE, fits_TM, orders)` where `err_TE`, `err_TM`
are `(max_order+1) × (max_order+1)` matrices of normalized errors (rows `m`,
columns `n`) and `fits_*` hold the corresponding [`fit_hg_order`](@ref) results.
"""
function hg_fit_residuals(E::AbstractArray{Complex{T},3}, grid::Grid{2};
        max_order::Int=4, maxiters::Int=300) where {T<:Real}
    xs, ys = T.(x(grid)), T.(y(grid))
    Fx, Fy = _real_transverse(E)
    px = sum(abs2, Fx)
    py = sum(abs2, Fy)
    ptot = px + py
    initx = hg_field_centroid_variance(Fx, xs, ys)[1:4]
    inity = hg_field_centroid_variance(Fy, xs, ys)[1:4]
    M = max_order + 1
    err_TE = fill(T(Inf), M, M)
    err_TM = fill(T(Inf), M, M)
    fits_TE = Matrix{Any}(undef, M, M)
    fits_TM = Matrix{Any}(undef, M, M)
    for m in 0:max_order, n in 0:max_order
        fx = fit_hg_order(Fx, xs, ys, m, n; init=initx, maxiters, fnorm2=px)
        fy = fit_hg_order(Fy, xs, ys, m, n; init=inity, maxiters, fnorm2=py)
        fits_TE[m+1, n+1] = fx
        fits_TM[m+1, n+1] = fy
        # cross-polarization power counts as residual so TE/TM is discriminated
        err_TE[m+1, n+1] = ptot > 0 ? (fx.residual + py) / ptot : T(Inf)
        err_TM[m+1, n+1] = ptot > 0 ? (fy.residual + px) / ptot : T(Inf)
    end
    orders = [(m, n) for m in 0:max_order, n in 0:max_order]
    return (; err_TE, err_TM, fits_TE, fits_TM, orders, px, py)
end

"""
    hg_mode_label(E, grid; max_order=4, maxiters=300) -> NamedTuple

Label a guided mode by Hermite–Gaussian fit quality. Fits every transverse order
up to `max_order` in both TE (x) and TM (y) polarizations (see
[`hg_fit_residuals`](@ref)) and returns the polarization/order whose optimized fit
has the smallest normalized squared error:

`(; pol, m, n, rel_error, te_frac, label, fit, residuals)` where

- `pol`   — `:TE` or `:TM` (dominant in-plane polarization of the best fit),
- `m, n`  — best-fit Hermite–Gaussian orders along x and y,
- `rel_error` — normalized squared misfit ∈ [0,1] of the winning template,
- `te_frac` — fraction of real transverse power in the x component (TE-ness),
- `label`  — pretty string, e.g. `"TE₀₀"` (see [`hg_label_string`](@ref)),
- `fit`    — the winning [`fit_hg_order`](@ref) NamedTuple (centroid, widths, …),
- `residuals` — the full [`hg_fit_residuals`](@ref) result for inspection.

This is an alternative to the node-counting [`mode_idx`](@ref)/[`mode_viable`](@ref)
classifier; it is threshold-free and returns a quantitative goodness-of-fit.
"""
function hg_mode_label(E::AbstractArray{Complex{T},3}, grid::Grid{2};
        max_order::Int=4, maxiters::Int=300) where {T<:Real}
    r = hg_fit_residuals(E, grid; max_order, maxiters)
    iTE = argmin(r.err_TE)
    iTM = argmin(r.err_TM)
    eTE = r.err_TE[iTE]
    eTM = r.err_TM[iTM]
    if eTE ≤ eTM
        pol, idx, err, fit = :TE, iTE, eTE, r.fits_TE[iTE]
    else
        pol, idx, err, fit = :TM, iTM, eTM, r.fits_TM[iTM]
    end
    m, n = idx[1] - 1, idx[2] - 1
    te_frac = (r.px + r.py) > 0 ? r.px / (r.px + r.py) : zero(T)
    return (; pol, m, n, rel_error=err, te_frac,
        label=hg_label_string(pol, m, n), fit, residuals=r)
end

const _SUBSCRIPTS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
_subscript(k::Integer) = k < 10 ? string(_SUBSCRIPTS[k+1]) : join(_SUBSCRIPTS[d+1] for d in digits(k) |> reverse)

"""
    hg_label_string(pol, m, n) -> String

Format a mode label such as `"TE₀₀"` or `"TM₂₁"` from a polarization symbol
(`:TE`/`:TM`) and transverse Hermite–Gaussian orders `(m,n)`.
"""
hg_label_string(pol::Symbol, m::Integer, n::Integer) =
    string(pol, _subscript(m), _subscript(n))
