export SellmeierFit, fit_sellmeier, sellmeier_n², sellmeier_n

"""
    SellmeierFit

Result of fitting an `N`-term Sellmeier model

```math
n^2(λ) = A_0 + \\sum_{i=1}^{N} \\frac{B_i\\,λ^2}{λ^2 - C_i}
```

(`λ` in μm, `Cᵢ` in μm²) to an [`IndexDataset`](@ref) over the wavelength range
`λ_range`. Fields: coefficients `A₀`, `B`, `C`; the validity range `λ_range`; fit quality
`rms_error`/`max_error` (in refractive index `n`, over the fit range); the term count
`n_terms`; the temperature `T` and crystal `axis` inherited from the data; an optional
`name`; and the source `dataset`. Evaluate with [`sellmeier_n`](@ref)/[`sellmeier_n²`](@ref);
turn into a `MaterialDispersion.Material` with [`build_material`](@ref).
"""
struct SellmeierFit
    A₀::Float64
    B::Vector{Float64}
    C::Vector{Float64}
    λ_range::Tuple{Float64,Float64}
    rms_error::Float64
    max_error::Float64
    n_terms::Int
    T::Union{Float64,Nothing}
    axis::String
    name::String
    dataset::IndexDataset
end

"""
    sellmeier_n²(fit, λ)  /  sellmeier_n²(A₀, B, C, λ)

Squared refractive index of a Sellmeier model at vacuum wavelength `λ` (μm).
"""
@inline function sellmeier_n²(A₀::Real, B, C, λ::Real)
    λ² = λ^2
    s = float(A₀)
    @inbounds for i in eachindex(B)
        s += B[i] * λ² / (λ² - C[i])
    end
    return s
end
sellmeier_n²(fit::SellmeierFit, λ::Real) = sellmeier_n²(fit.A₀, fit.B, fit.C, λ)

"""
    sellmeier_n(fit, λ)

Refractive index `n(λ) = √(n²(λ))` of a fitted Sellmeier model (`λ` in μm).
"""
sellmeier_n(fit::SellmeierFit, λ::Real) = sqrt(sellmeier_n²(fit, λ))

# Linear warm start: with the poles `C` fixed, n² is linear in [A₀, B₁…B_N], so solve that
# least-squares problem to get a good initial guess before the nonlinear refinement.
function _linear_warmstart(λ::Vector{Float64}, y::Vector{Float64}, C::Vector{Float64})
    M = hcat(ones(length(λ)), [λ[k]^2 / (λ[k]^2 - C[i]) for k in eachindex(λ), i in eachindex(C)])
    coeffs = M \ y
    return coeffs[1], coeffs[2:end]            # A₀, B
end

# `n` geometrically-spaced pole wavelengths in [a, b] (the n==1 case is their geometric
# mean — `range(…; length=1)` rejects distinct endpoints).
_geom_poles(a::Real, b::Real, n::Int) =
    n == 0 ? Float64[] : n == 1 ? [sqrt(a * b)] : exp.(range(log(a), log(b); length=n))

# Default initial poles: spread sqrt(Cᵢ) geometrically, the majority as UV resonances
# below the band and (for ≥2 terms) one or more IR resonances above it.
function _default_poles(n_terms::Int, lo::Real, hi::Real)
    n_ir = n_terms ≥ 2 ? max(1, n_terms ÷ 3) : 0
    n_uv = n_terms - n_ir
    uv = _geom_poles(lo * 0.2, lo * 0.85, n_uv)
    ir = _geom_poles(hi * 1.5, hi * 6, n_ir)
    return sort(vcat(uv, ir)) .^ 2                # → Cᵢ in μm²
end

"""
    fit_sellmeier(ds::IndexDataset; n_terms=2, λ_range=extrema(ds.λ), kwargs...) -> SellmeierFit

Fit an `n_terms`-term Sellmeier model to dataset `ds` over the vacuum-wavelength range
`λ_range` (μm), the desired *range of validity* of the resulting model. A fixed-pole
linear least-squares warm start is followed by Levenberg–Marquardt refinement (LsqFit).

Keywords:
- `n_terms`   — number of Sellmeier resonance terms,
- `λ_range`   — `(λ_lo, λ_hi)` validity range to fit over (defaults to the data extent),
- `p0`        — explicit initial `[A₀, B₁, C₁, …]` (overrides the heuristic),
- `lower`/`upper` — parameter box bounds passed to LsqFit,
- `name`      — model name (string/symbol),
- `plotdir`   — if given (and `Plots` is loaded), a fit-vs-data comparison plot is saved
                there **every time the fit runs** (see [`plot_sellmeier_fit`](@ref)),
- `plotname`  — base filename for that plot,
- `show_plot` — also return/display the plot object.

The returned [`SellmeierFit`](@ref) records the coefficients, the validity range and the
RMS / maximum index error over the fit range.
"""
function fit_sellmeier(ds::IndexDataset; n_terms::Int=2, λ_range=λrange(ds), p0=nothing,
                       lower=nothing, upper=nothing, maxiter::Int=5000, name=nothing,
                       plotdir=nothing, plotname=nothing, show_plot::Bool=false)
    lo, hi = float(λ_range[1]), float(λ_range[2])
    hi > lo || throw(ArgumentError("λ_range must be increasing, got $(λ_range)"))
    m = lo .≤ ds.λ .≤ hi
    np = count(m)
    np ≥ 1 + 2n_terms ||
        throw(ArgumentError("only $np data point(s) in λ_range=$( (lo,hi) ); need ≥ $(1 + 2n_terms) for a $n_terms-term fit"))
    λf = ds.λ[m]
    nf = ds.n[m]
    yf = nf .^ 2

    if p0 === nothing
        C0 = _default_poles(n_terms, lo, hi)
        A0, B0 = _linear_warmstart(λf, yf, C0)
        p0 = Vector{Float64}(undef, 1 + 2n_terms)
        p0[1] = A0
        @inbounds for i in 1:n_terms
            p0[2i] = B0[i]; p0[2i+1] = C0[i]
        end
    else
        p0 = collect(Float64, p0)
        length(p0) == 1 + 2n_terms || throw(ArgumentError("p0 must have length $(1 + 2n_terms)"))
    end

    model(λ, p) = begin
        A₀ = p[1]
        B = @view p[2:2:end]
        C = @view p[3:2:end]
        [sellmeier_n²(A₀, B, C, l) for l in λ]
    end

    kw = (;)
    lower !== nothing && (kw = (; kw..., lower=collect(Float64, lower)))
    upper !== nothing && (kw = (; kw..., upper=collect(Float64, upper)))
    res = curve_fit(model, λf, yf, p0; kw...)
    p = coef(res)
    A₀ = p[1]
    B = collect(Float64, p[2:2:end])
    C = collect(Float64, p[3:2:end])

    npred = [sqrt(max(sellmeier_n²(A₀, B, C, l), 0.0)) for l in λf]
    resid = npred .- nf
    rms = sqrt(mean(abs2, resid))
    mx = maximum(abs, resid)

    nm = name === nothing ? ds.label : string(name)
    sf = SellmeierFit(A₀, B, C, (lo, hi), rms, mx, n_terms, ds.T, ds.axis, nm, ds)

    for c in C
        if lo^2 ≤ c ≤ hi^2
            @warn "Sellmeier pole at λ≈$(round(sqrt(abs(c)); digits=4)) μm lies inside the fit range [$lo, $hi] μm; the model is singular there. Consider fewer terms or a different λ_range."
        end
    end

    if plotdir !== nothing || show_plot
        _emit_fit_plot(sf; dir=plotdir, name=plotname, show=show_plot)
    end
    return sf
end

function Base.show(io::IO, sf::SellmeierFit)
    @printf(io, "SellmeierFit(%s, %d terms, λ∈[%.3f,%.3f] μm, RMS Δn=%.2e, max Δn=%.2e)",
            isempty(sf.name) ? "unnamed" : sf.name, sf.n_terms, sf.λ_range[1], sf.λ_range[2],
            sf.rms_error, sf.max_error)
end
