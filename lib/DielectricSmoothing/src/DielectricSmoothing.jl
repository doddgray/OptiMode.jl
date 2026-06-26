################################################################################
#                                                                              #
#                            DielectricSmoothing.jl:                           #
#        Sub-pixel (Kottke) smoothing of dielectric tensors on spatial        #
#        grids for the OptiMode differentiable photonics tool suite           #
#                                                                              #
################################################################################

module DielectricSmoothing

##### Imports ######
using LinearAlgebra
using LinearAlgebra: diag
using StaticArrays
using StaticArrays: SVector, SMatrix, MVector, MMatrix, SHermitianCompact

### AD rule infrastructure ###
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, NoTangent, ZeroTangent, AbstractZero
using ForwardDiff

### Geometry ###
using GeometryPrimitives
using GeometryPrimitives: Shape, surfpt_nearby, volfrac

### Material dispersion models ###
using MaterialDispersion
using MaterialDispersion: ε_tensor, εᵥ

## Includes ##
include("grid.jl")
include("shapes.jl")
include("kottke.jl")
include("smooth.jl")
include("smoothing_plan.jl")
include("geometry.jl")

end # module DielectricSmoothing
