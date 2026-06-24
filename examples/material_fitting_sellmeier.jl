# Building dielectric material models with MaterialFitting
# =========================================================
#
# This example shows how to turn published refractive-index data into OptiMode
# `Material` models by fitting Sellmeier equations:
#
#   1. fit an isotropic material from a RefractiveIndex.INFO record (here a bundled
#      fused-silica YAML; the same call accepts a refractiveindex.info URL),
#   2. build an anisotropic (uniaxial) material from separate ordinary/extraordinary
#      datasets,
#   3. fit a temperature-dependent Sellmeier model from data at several temperatures,
#   4. save and reload a fitted model.
#
# A fit-vs-data comparison plot (with the validity range shaded) is saved every time a
# fit runs. Run with an environment that has MaterialFitting + Plots:
#
#   julia --project=… examples/material_fitting_sellmeier.jl

using MaterialFitting
using MaterialDispersion: ε_fn, generate_fn
using Plots

const OUT = joinpath(@__DIR__, "material_fitting_output")
mkpath(OUT)

# ---------------------------------------------------------------------------------------
# 1. Isotropic material from a RefractiveIndex.INFO record
# ---------------------------------------------------------------------------------------
# `refractiveindex_dataset` accepts a refractiveindex.info page URL, a direct .yml URL, or
# a local YAML file. We use the fused-silica record bundled with the package tests; to pull
# it live instead, pass:
#   "https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"
sio2_yaml = joinpath(dirname(pathof(MaterialFitting)), "..", "test", "data", "SiO2_Malitson.yml")
ds_sio2 = refractiveindex_dataset(sio2_yaml; λ_range=(0.25, 2.5), n_points=200)

fit_sio2 = fit_sellmeier(ds_sio2; n_terms=3, λ_range=(0.25, 2.5), name="SiO₂ (Malitson)",
                         plotdir=OUT, plotname="SiO2")
@info "SiO₂ fit" rms=fit_sio2.rms_error max=fit_sio2.max_error n_1p55=sellmeier_n(fit_sio2, 1.55)

mat_sio2 = build_material(fit_sio2; name=:SiO₂_fit)
@info "SiO₂ Material" n_at_1310nm = sqrt(ε_fn(mat_sio2)(1.31)[1, 1])

# ---------------------------------------------------------------------------------------
# 2. Anisotropic (uniaxial) material — different datasets per axis
# ---------------------------------------------------------------------------------------
# Synthetic congruent-LiNbO₃-like ordinary/extraordinary indices (stand-ins for two
# RefractiveIndex.INFO entries, one per axis).
no(λ) = sqrt(1 + 2.6734λ^2/(λ^2 - 0.01764) + 1.2290λ^2/(λ^2 - 0.05914) + 12.614λ^2/(λ^2 - 474.6))
ne(λ) = sqrt(1 + 2.9804λ^2/(λ^2 - 0.02047) + 0.5981λ^2/(λ^2 - 0.0666) + 8.9543λ^2/(λ^2 - 416.08))
λa = collect(range(0.5, 4.0; length=80))
ds_o = index_dataset(λa, no.(λa); axis="o", label="nₒ")
ds_e = index_dataset(λa, ne.(λa); axis="e", label="nₑ")
fit_o = fit_sellmeier(ds_o; n_terms=3, λ_range=(0.5, 4.0), name="LiNbO₃ nₒ", plotdir=OUT, plotname="LiNbO3")
fit_e = fit_sellmeier(ds_e; n_terms=3, λ_range=(0.5, 4.0), name="LiNbO₃ nₑ", plotdir=OUT, plotname="LiNbO3")
mat_ln = build_material(; o=fit_o, e=fit_e, name=:LiNbO₃_fit)
εln = ε_fn(mat_ln)(1.55)
@info "LiNbO₃ Material (uniaxial)" nₒ=sqrt(εln[1,1]) nₑ=sqrt(εln[3,3])

# ---------------------------------------------------------------------------------------
# 3. Temperature-dependent Sellmeier from multi-temperature data
# ---------------------------------------------------------------------------------------
T₀ = 25.0; dndT = 1.0e-5
nT(λ, T) = sqrt(sellmeier_n(fit_sio2, λ)^2 + 2 * sellmeier_n(fit_sio2, λ) * dndT * (T - T₀))
λt = collect(range(0.4, 2.0; length=60))
dsets = [index_dataset(λt, nT.(λt, T); T=T, label="SiO₂") for T in (15.0, 25.0, 35.0, 55.0, 75.0)]
fit_T = fit_thermo_sellmeier(dsets; n_terms=3, λ_range=(0.4, 2.0), T_poly_order=1,
                             name="SiO2_T", plotdir=OUT)
mat_T = build_material(fit_T; name=:SiO₂_T_fit)
fεT = generate_fn(mat_T, :ε, :λ, :T)
@info "SiO₂(T) Material" n_1550_25C=sqrt(fεT(1.55, 25.0)[1,1]) n_1550_75C=sqrt(fεT(1.55, 75.0)[1,1])

# ---------------------------------------------------------------------------------------
# 4. Save / reload
# ---------------------------------------------------------------------------------------
save_material_model(fit_sio2, joinpath(OUT, "SiO2_fit.jld2"))
fit_reloaded = load_material_model(joinpath(OUT, "SiO2_fit.jld2"))
@info "round-trip" same = sellmeier_n(fit_reloaded, 1.55) ≈ sellmeier_n(fit_sio2, 1.55)

println("\nSaved plots and model to: ", OUT)
foreach(f -> println("  ", f), sort(readdir(OUT)))
