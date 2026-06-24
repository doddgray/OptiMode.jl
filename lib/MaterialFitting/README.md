# MaterialFitting

Build frequency- and temperature-dependent dielectric `Material` models for the
[OptiMode](../..) suite by fitting **Sellmeier equations** to published refractive-index
data — from [RefractiveIndex.INFO](https://refractiveindex.info) (by URL) or supplied
directly.

```julia
using MaterialFitting, Plots

# from a RefractiveIndex.INFO URL (or a .yml file, or your own (λ, n) arrays):
ds  = refractiveindex_dataset("https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson";
                              λ_range = (0.25, 2.5))
fit = fit_sellmeier(ds; n_terms = 3, λ_range = (0.25, 2.5), plotdir = "fits")  # saves a plot
mat = build_material(fit; name = :SiO₂_fit)        # a MaterialDispersion.Material

# anisotropic (a dataset per axis) and temperature-dependent models:
mat_uniaxial = build_material(; o = fit_o, e = fit_e)
tf  = fit_thermo_sellmeier(datasets_at_several_T; n_terms = 3, λ_range = (0.4, 2.0))
matT = build_material(tf)                           # ε(λ, T)

save_material_model(fit, "SiO2_fit.jld2"); load_material_model("SiO2_fit.jld2")
```

Key functions: `refractiveindex_dataset`, `index_dataset`, `fit_sellmeier`,
`fit_thermo_sellmeier`, `build_material`, `save_material_model` / `load_material_model`,
and (with `Plots` loaded) `plot_sellmeier_fit` — a fit-vs-data comparison plot with the
validity range shaded is saved every time a fit runs. See
[`docs/material_fitting.md`](../../docs/material_fitting.md) and
[`examples/material_fitting_sellmeier.jl`](../../examples/material_fitting_sellmeier.jl).
