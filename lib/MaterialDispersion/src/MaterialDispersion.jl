################################################################################
#                                                                              #
#                            MaterialDispersion.jl:                            #
#            Symbolic dielectric material dispersion models for the           #
#                 OptiMode differentiable photonics tool suite                 #
#                                                                              #
################################################################################

module MaterialDispersion

##### Imports ######
using LinearAlgebra
using LinearAlgebra: diag
using StaticArrays
using StaticArrays: SVector, SMatrix
using Tullio

### AD rule infrastructure ###
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, NoTangent, ZeroTangent, AbstractZero

### Symbolic modeling ###
using Rotations
using Symbolics
using SymbolicUtils
using Symbolics: Sym, Num, scalarize
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using IterTools: subsets
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

### Colors (for material plotting metadata) ###
using Colors
using Colors: Color, RGB, RGBA, @colorant_str

## Includes ##
include("epsilon.jl")
include("cse.jl")
include("materials.jl")
include("dispersion.jl")

end # module MaterialDispersion
