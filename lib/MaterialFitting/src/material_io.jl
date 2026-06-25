export save_material_model, load_material_model

"""
    save_material_model(fit, path) -> path

Save a fitted material model — a [`SellmeierFit`](@ref) or [`ThermoSellmeierFit`](@ref),
including its coefficients, validity range, fit-quality metrics and source dataset — to
`path` (a JLD2 file). Reload with [`load_material_model`](@ref) and rebuild the symbolic
`Material` with [`build_material`](@ref).

```julia
fit = fit_sellmeier(ds; n_terms=3, λ_range=(0.25, 2.5))
save_material_model(fit, "SiO2_fit.jld2")
fit2 = load_material_model("SiO2_fit.jld2")
mat  = build_material(fit2)
```
"""
function save_material_model(fit::Union{SellmeierFit,ThermoSellmeierFit}, path::AbstractString)
    JLD2.jldsave(String(path); model=fit, kind=string(nameof(typeof(fit))), mf_format=1)
    return path
end

"""
    load_material_model(path) -> SellmeierFit | ThermoSellmeierFit

Reload a fitted material model saved with [`save_material_model`](@ref).
"""
function load_material_model(path::AbstractString)
    return JLD2.load(String(path), "model")
end
