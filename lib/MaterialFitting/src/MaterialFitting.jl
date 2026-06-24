################################################################################
#                                                                              #
#                             MaterialFitting.jl:                              #
#        Build frequency- (and temperature-) dependent dielectric material     #
#        models for the OptiMode suite by fitting Sellmeier equations to       #
#        published refractive-index data (RefractiveIndex.INFO or              #
#        user-supplied datasets).                                              #
#                                                                              #
################################################################################
#
# Typical workflow:
#
#   using MaterialFitting, Plots
#   ds  = refractiveindex_dataset("https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson")
#   fit = fit_sellmeier(ds; n_terms = 3, λ_range = (0.25, 2.5), plotdir = "fits")
#   mat = build_material(fit; name = :SiO₂_fit)        # a MaterialDispersion.Material
#   save_material_model(fit, "SiO2_fit.jld2")
#   fit2 = load_material_model("SiO2_fit.jld2")
#
# Anisotropic and temperature-dependent models are built from several datasets;
# see `build_material` and `fit_thermo_sellmeier`.

module MaterialFitting

using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles: readdlm
using Downloads
using LsqFit: curve_fit, coef
using YAML
using JLD2

using MaterialDispersion
using MaterialDispersion: Material, n²_sym_fmt1, ng_model, gvd_model
using Symbolics
using Symbolics: @variables, Num, substitute, value

include("datasets.jl")
include("refractiveindex.jl")
include("sellmeier_fit.jl")
include("build_material.jl")
include("material_io.jl")
include("plots.jl")

end # module MaterialFitting
