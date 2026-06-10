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

### Geometry ###
using GeometryPrimitives
using GeometryPrimitives: Shape, surfpt_nearby, volfrac

### Symbolic codegen (used once at load time to build the Kottke-smoothing kernels) ###
using Symbolics
using Symbolics: Num, scalarize
using SymbolicUtils

### Material dispersion models ###
using MaterialDispersion
using MaterialDispersion: eval_fn_oop, eval_fn_ip, _fj_fjh_sym, ε_tensor, εᵥ

## Includes ##
include("grid.jl")
include("kottke.jl")
include("smooth.jl")
include("geometry.jl")

end # module DielectricSmoothing
