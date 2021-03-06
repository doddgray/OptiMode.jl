################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                 (an attempt)                                 #
#                                                                              #
################################################################################

module OptiMode

## Imports ##
using LinearAlgebra
using LinearMaps
using ArrayInterface
using StaticArrays
using HybridArrays
using StructArrays
using RecursiveArrayTools
using FFTW
using AbstractFFTs
using GeometryPrimitives
using Roots
using ChainRules
using Zygote
using ForwardDiff
using LoopVectorization
using Tullio
using IterativeSolvers
using Rotations
using Colors: Color, RGB, RGBA, @colorant_str
using RuntimeGeneratedFunctions
# using ModelingToolkit, Unitful #, Latexify
using Symbolics
using SymbolicUtils
using UnicodePlots, AbstractPlotting
# using KrylovKit
using DFTK: LOBPCG
using StaticArrays: Dynamic, SVector
using Zygote: Buffer, bufferfrom, @ignore, dropgrad, forwarddiff
using AbstractPlotting.GeometryBasics

RuntimeGeneratedFunctions.init(@__MODULE__)
# using Plots: plot, heatmap, plot!, heatmap!
# using MaxwellFDM: kottke_avg_param
## Exports ##
# export plot_ε, test_shapes, ridge_wg, circ_wg, trap_wg, trap_wg2, plot, heatmap, SHM3
export SHM3
export plot_data, uplot, uplot!, xlims, ylims

# Import methods that we will overload for custom types
import Base: size, eltype
import LinearAlgebra: mul!

# FFTW settings
FFTW.set_num_threads(1)     # chosen for thread safety when combined with other parallel code, consider increasing

## Abbreviations, aliases, etc. ##
SHM3 = SHermitianCompact{3,Float64,6}   # static Hermitian 3×3 matrix Type, renamed for brevity

## generic plotting methods/aliases
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
uplot(x::AbstractVector, y::AbstractVector; kwargs...) = UnicodePlots.lineplot(x, y ; kwargs...)
uplot!(plt::UnicodePlots.Plot, x::AbstractVector, y::AbstractVector; kwargs...) = UnicodePlots.lineplot!(plt, x, y ; kwargs...)
plot_data(x) = x    # nontrivial methods defined for custom types

xlims(x::Real) = x
xlims(x::Complex) = real(x)
xlims(v::AbstractVector{<:Number}) = [extrema(real(v))...]
xlims(vs::AbstractVector) = [extrema( vcat( xlims.(vs) ) )...]
function xlims(a,b)
	xlims_a, xlims_b = xlims(a), xlims(b)
	[ min(xlims_a[1], xlims_b[1]), max(xlims_a[2], xlims_b[2]) ]
end
xlims(plt::UnicodePlots.Plot) = parse.(Float64,getfield.(unique(plt.decorations),:second))

ylims(x::Real) = x
ylims(x::Complex) = real(x)
ylims(v::AbstractVector{<:Number}) = [extrema(real(v))...]
ylims(vs::AbstractVector) = [extrema( vcat( ylims.(vs) ) )...]
function ylims(fns::AbstractVector{<:Function};xlims=(0.5,1.8),nvals=100)
	xrng = range(xlims[1],xlims[2];length=nvals)
	[ extrema(hcat(map.(fns,(xrng,) )...) )... ]
end
function ylims(a,b)
	ylims_a, ylims_b = ylims(a), ylims(b)
	[ min(ylims_a[1], ylims_b[1]), max(ylims_a[2], ylims_b[2]) ]
end
ylims(plt::UnicodePlots.Plot) = parse.(Float64,getfield.(unique(plt.labels_left),:second))


# function ylims(plt::UnicodePlots.Plot, new ; kwargs...)
# 	ylims_plt = parse.(Float64,getfield.(unique(plt.labels_left),:second))
# 	ylims(ylims_plt,ylims(new))
# end

## Add methods to external packages ##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

## Includes ##

# include("plot.jl")
include("epsilon.jl")
include("materials.jl")
include("grid.jl")
include("geometry.jl")
include("maxwell.jl")
include("constraints.jl")
include("fields.jl")
include("smooth.jl")
include("solve.jl")
include("grads.jl")
include("explicit.jl")
# include("optimize.jl")


## Definitions ##


## More includes ##


end
