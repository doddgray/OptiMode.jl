################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                 (an attempt)                                 #
#                                                                              #
################################################################################

module OptiMode

## Imports ##
using LinearAlgebra, LinearMaps, ArrayInterface, StaticArrays, HybridArrays, StructArrays, FFTW, AbstractFFTs, GeometryPrimitives, Roots, ChainRules, Zygote, ForwardDiff, LoopVectorization, Tullio, IterativeSolvers
using StaticArrays: Dynamic, SVector
using Zygote: Buffer, bufferfrom, @ignore, dropgrad
using Plots: plot, heatmap, plot!, heatmap!
# using MaxwellFDM: kottke_avg_param
## Exports ##
export plot_ε, test_shapes, ridge_wg, circ_wg, trap_wg, trap_wg2, plot, heatmap, SHM3

# FFTW settings
FFTW.set_num_threads(4)     # chosen for thread safety when combined with other parallel code, consider increasing

## Abbreviations, aliases, etc. ##
SHM3 = SHermitianCompact{3,Float64,6}   # static Hermitian 3×3 matrix Type, renamed for brevity

## Add methods to external packages ##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

## Includes ##
include("maxwell.jl")
include("plot.jl")
# include("materials.jl")
include("epsilon.jl")
include("geometry.jl")
include("solve.jl")
include("grads.jl")
# include("optimize.jl")


## Definitions ##


## More includes ##


end
