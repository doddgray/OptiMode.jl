export sellmeier_n²_sym, build_material, fit_thermo_sellmeier, ThermoSellmeierFit, thermo_n²

# ---- symbolic Sellmeier (for MaterialDispersion.Material construction) ----------------

"""
    sellmeier_n²_sym(fit, λ) -> Num

Symbolic squared-index expression `A₀ + Σ Bᵢ λ²/(λ²−Cᵢ)` of a [`SellmeierFit`](@ref) in
the symbolic variable `λ` (a `Symbolics.Num`), ready to assemble into a
`MaterialDispersion.Material`.
"""
function sellmeier_n²_sym(fit::SellmeierFit, λ)
    λ² = λ^2
    s = fit.A₀ + sum(fit.B[i] * λ² / (λ² - fit.C[i]) for i in 1:fit.n_terms)
    return s
end

_default_T₀(fit::SellmeierFit) = fit.T === nothing ? 20.0 : fit.T

# Assemble a Material from a 3-vector of symbolic diagonal n² expressions and defaults.
function _material_from_diag(n²diag, λ, name, color, λ_mid, T₀)
    ε = diagm(collect(n²diag))
    nλ = sqrt.(n²diag)
    models = Dict{Any,Any}(
        :ε  => ε,
        :n  => nλ[1],
        :nₒ => nλ[1],
        :nₑ => nλ[3],
    )
    defaults = Dict{Symbol,Any}(:λ => λ_mid, :ω => inv(λ_mid), :T => T₀)
    return color === nothing ? Material(models, defaults, Symbol(name)) :
                               Material(models, defaults, Symbol(name), color)
end

"""
    build_material(fit::SellmeierFit; name=fit.name, color=nothing, T₀=fit.T) -> Material

Build an **isotropic** `MaterialDispersion.Material` whose dielectric tensor is
`ε = diagm([n², n², n²])` with `n²` the fitted Sellmeier model. The material's default
wavelength is the centre of the fit's validity range; `T₀` sets the default temperature.
"""
function build_material(fit::SellmeierFit; name=nothing, color=nothing, T₀=nothing)
    @variables λ
    n² = sellmeier_n²_sym(fit, λ)
    λ_mid = sum(fit.λ_range) / 2
    nm = name === nothing ? (isempty(fit.name) ? "FittedMaterial" : fit.name) : name
    return _material_from_diag((n², n², n²), λ, nm, color, λ_mid, T₀ === nothing ? _default_T₀(fit) : T₀)
end

"""
    build_material(; o=…, e=…)  /  build_material(; x=…, y=…, z=…)
    build_material(fits::AbstractVector{SellmeierFit}; name, color)

Build an **anisotropic** `MaterialDispersion.Material` from per-axis [`SellmeierFit`](@ref)s
— each axis may come from a different RefractiveIndex.INFO entry or user dataset.

- `build_material(o=fit_o, e=fit_e)` — uniaxial: `ε = diagm([n²ₒ, n²ₒ, n²ₑ])`,
- `build_material(x=fx, y=fy, z=fz)` — biaxial: `ε = diagm([n²ₓ, n²_y, n²_z])`,
- a 3-element vector of fits is taken as `[x, y, z]`.
"""
function build_material(; x=nothing, y=nothing, z=nothing, o=nothing, e=nothing,
                        name=nothing, color=nothing, T₀=nothing)
    @variables λ
    if o !== nothing || e !== nothing
        (o !== nothing && e !== nothing) || throw(ArgumentError("uniaxial build_material needs both `o` and `e` fits"))
        n²o = sellmeier_n²_sym(o, λ); n²e = sellmeier_n²_sym(e, λ)
        fits = (o, e)
        diag = (n²o, n²o, n²e)
    elseif x !== nothing && y !== nothing && z !== nothing
        fits = (x, y, z)
        diag = ntuple(i -> sellmeier_n²_sym(fits[i], λ), 3)
    else
        throw(ArgumentError("provide either (o,e) for uniaxial or (x,y,z) for biaxial materials"))
    end
    λ_mid = sum(first(fits).λ_range) / 2
    nm = name === nothing ? "FittedMaterial_anisotropic" : name
    T0 = T₀ === nothing ? _default_T₀(first(fits)) : T₀
    return _material_from_diag(diag, λ, nm, color, λ_mid, T0)
end

function build_material(fits::AbstractVector{SellmeierFit}; kwargs...)
    length(fits) == 3 || throw(ArgumentError("vector form expects 3 fits [x,y,z]; use keyword form otherwise"))
    return build_material(; x=fits[1], y=fits[2], z=fits[3], kwargs...)
end

# ---- temperature-dependent Sellmeier -------------------------------------------------

"""
    ThermoSellmeierFit

Temperature-dependent Sellmeier model: every Sellmeier coefficient (`A₀`, each `Bᵢ`, each
`Cᵢ`) is itself a polynomial of order `T_poly_order` in `(T − T₀)` (°C). Built by
[`fit_thermo_sellmeier`](@ref) from datasets measured at several temperatures; evaluate
with [`thermo_n²`](@ref) or turn into a `(λ,T)`-dependent `Material` with
[`build_material`](@ref).
"""
struct ThermoSellmeierFit
    A₀::Vector{Float64}            # polynomial coeffs in (T-T₀), low→high order
    B::Vector{Vector{Float64}}
    C::Vector{Vector{Float64}}
    T_poly_order::Int
    n_terms::Int
    λ_range::Tuple{Float64,Float64}
    T_range::Tuple{Float64,Float64}
    T₀::Float64
    rms_error::Float64
    max_error::Float64
    axis::String
    name::String
    fits::Vector{SellmeierFit}
    temperatures::Vector{Float64}
end

_polyval(c::AbstractVector, x) = (s = zero(x) + c[end]; for k in length(c)-1:-1:1; s = s * x + c[k]; end; s)

"""
    fit_thermo_sellmeier(datasets; n_terms=2, λ_range, T_poly_order=1, T₀=nothing, kwargs...) -> ThermoSellmeierFit

Fit a temperature-dependent Sellmeier model from `datasets` — a collection of
[`IndexDataset`](@ref)s each carrying a temperature `T` (°C). A Sellmeier model is fit at
every temperature (warm-started from the previous one for coefficient continuity), then
each coefficient is fit as a degree-`T_poly_order` polynomial in `(T − T₀)`.

`T₀` defaults to the median dataset temperature. The same `λ_range` validity window is
used at all temperatures. `kwargs` are forwarded to [`fit_sellmeier`](@ref) (e.g. `p0`).
"""
function fit_thermo_sellmeier(datasets; n_terms::Int=2, λ_range=nothing, T_poly_order::Int=1,
                              T₀=nothing, name=nothing, plotdir=nothing, kwargs...)
    dss = collect(datasets)
    all(d -> d.T !== nothing, dss) || throw(ArgumentError("every dataset must carry a temperature `T` for thermo fitting"))
    length(dss) ≥ T_poly_order + 1 ||
        throw(ArgumentError("need ≥ $(T_poly_order + 1) temperatures for a degree-$T_poly_order polynomial fit"))
    order = sortperm([d.T::Float64 for d in dss])
    dss = dss[order]
    Ts = Float64[d.T for d in dss]
    λr = λ_range === nothing ? (maximum(d.λ[1] for d in dss), minimum(d.λ[end] for d in dss)) : λ_range
    T0 = T₀ === nothing ? Statistics.median(Ts) : Float64(T₀)

    # per-temperature fits, warm-started for coefficient continuity
    fits = SellmeierFit[]
    p0 = nothing
    for d in dss
        f = fit_sellmeier(d; n_terms, λ_range=λr, p0, kwargs...)
        push!(fits, f)
        p0 = vcat(f.A₀, vcat([[f.B[i], f.C[i]] for i in 1:n_terms]...))
    end

    # fit each coefficient as a polynomial in (T - T₀)
    x = Ts .- T0
    V = hcat([x .^ k for k in 0:T_poly_order]...)
    polyfit(yvals) = V \ yvals
    A₀p = polyfit([f.A₀ for f in fits])
    Bp = [polyfit([f.B[i] for f in fits]) for i in 1:n_terms]
    Cp = [polyfit([f.C[i] for f in fits]) for i in 1:n_terms]

    # accuracy of the assembled thermo model over all data
    resid = Float64[]
    for d in dss
        for k in eachindex(d.λ)
            (λr[1] ≤ d.λ[k] ≤ λr[2]) || continue
            push!(resid, sqrt(max(_thermo_n²(A₀p, Bp, Cp, T0, d.λ[k], d.T), 0.0)) - d.n[k])
        end
    end
    rms = sqrt(mean(abs2, resid)); mx = maximum(abs, resid)
    nm = name === nothing ? (isempty(first(dss).label) ? "ThermoFittedMaterial" : first(dss).label) : string(name)

    tf = ThermoSellmeierFit(A₀p, Bp, Cp, T_poly_order, n_terms, (float(λr[1]), float(λr[2])),
                            (minimum(Ts), maximum(Ts)), T0, rms, mx, first(dss).axis, nm, fits, Ts)
    plotdir !== nothing && _emit_thermo_plot(tf; dir=plotdir)
    return tf
end

_thermo_n²(A₀p, Bp, Cp, T₀, λ, T) = sellmeier_n²(_polyval(A₀p, T - T₀),
    [_polyval(b, T - T₀) for b in Bp], [_polyval(c, T - T₀) for c in Cp], λ)

"""
    thermo_n²(tf::ThermoSellmeierFit, λ, T)

Squared refractive index of a temperature-dependent Sellmeier model at wavelength `λ` (μm)
and temperature `T` (°C).
"""
thermo_n²(tf::ThermoSellmeierFit, λ::Real, T::Real) = _thermo_n²(tf.A₀, tf.B, tf.C, tf.T₀, λ, T)

"""
    sellmeier_n²_sym(tf::ThermoSellmeierFit, λ, T) -> Num

Symbolic `n²(λ,T)` of a temperature-dependent Sellmeier model in symbolic `λ`, `T`.
"""
function sellmeier_n²_sym(tf::ThermoSellmeierFit, λ, T)
    dT = T - tf.T₀
    A₀ = _polyval(tf.A₀, dT)
    λ² = λ^2
    return A₀ + sum(_polyval(tf.B[i], dT) * λ² / (λ² - _polyval(tf.C[i], dT)) for i in 1:tf.n_terms)
end

"""
    build_material(tf::ThermoSellmeierFit; name=tf.name, color=nothing) -> Material

Build a temperature-dependent isotropic `Material` whose dielectric tensor `ε(λ,T)` carries
the full `(λ,T)` Sellmeier dependence; the default temperature is the model's `T₀`.
"""
function build_material(tf::ThermoSellmeierFit; name=nothing, color=nothing)
    @variables λ T
    n² = sellmeier_n²_sym(tf, λ, T)
    λ_mid = sum(tf.λ_range) / 2
    nm = name === nothing ? (isempty(tf.name) ? "ThermoFittedMaterial" : tf.name) : name
    return _material_from_diag((n², n², n²), λ, nm, color, λ_mid, tf.T₀)
end
