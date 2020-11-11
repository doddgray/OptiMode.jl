################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                 (an attempt)                                 #
#                                                                              #
################################################################################

module OptiMode

## Imports ##
using Plots: plot, heatmap, plot!, heatmap!
using MaxwellFDM: kottke_avg_param
using LinearAlgebra, FFTW, LinearMaps, StaticArrays, GeometryPrimitives, Roots

## Exports ##
export plot_ε, test_shapes, ridge_wg, circ_wg, plot, heatmap, SHM3



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
# include("grads.jl")
# include("optimize.jl")


## Definitions ##


## More includes ##


end
