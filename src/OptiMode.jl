################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                 (an attempt)                                 #
#                                                                              #
################################################################################

module OptiMode

##### Imports ######

### Linear Algebra Types and Libraries ###
using LinearAlgebra
using LinearAlgebra: diag
using LinearMaps
using ArrayInterface
using StaticArrays
using StaticArrays: Dynamic, SVector
using HybridArrays
# using StructArrays
# using RecursiveArrayTools
# using FillArrays
# using SparseArrays
using FFTW
using AbstractFFTs
using LoopVectorization
using Tullio
using SliceMap

### AD ###
# using ForwardDiff
# using ChainRules: ChainRulesCore, @non_differentiable,  NoTangent, @thunk, @not_implemented, AbstractZero
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, @not_implemented, NoTangent, ZeroTangent, AbstractZero
using Zygote
using Zygote: Buffer, bufferfrom, @ignore, @adjoint, ignore, dropgrad, forwarddiff, Numeric, literal_getproperty, accum
### Materials ###
using Rotations
using Symbolics
using SymbolicUtils
using Symbolics
using Symbolics: Sym, Num, scalarize
using SymbolicUtils: @rule, @acrule, @slots, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using IterTools: subsets
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

### Geometry ###
using GeometryPrimitives
# using GeometryPrimitives: Cylinder

### Iterative Solvers ###
using IterativeSolvers
using KrylovKit
# using DFTK: LOBPCG
using Roots
using PyCall

### Visualization ###
using Colors
using Colors: Color, RGB, RGBA, @colorant_str
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
# using UnicodePlots
# using Makie
# using Makie.GeometryBasics

### I/O ###
using Dates
using HDF5
using DelimitedFiles
using EllipsisNotation
using Printf
using ProgressMeter
# using Distributed

### Utils ###
# using Statistics
# using Interpolations
# using IterTools




## Exports ##
# export plot_ε, test_shapes, ridge_wg, circ_wg, trap_wg, trap_wg2, plot, heatmap, SHM3
# export SHM3
# export plot_data, uplot, uplot!, xlims, ylims

# Import methods that we will overload for custom types
import Base: size, eltype
import LinearAlgebra: mul!
import ChainRulesCore: rrule

# FFTW settings
FFTW.set_num_threads(1)     # chosen for thread safety when combined with other parallel code, consider increasing

# ## Abbreviations, aliases, etc. ##
# SHM3 = SHermitianCompact{3,Float64,6}   # static Hermitian 3×3 matrix Type, renamed for brevity

# ## generic plotting methods/aliases
# uplot(x::AbstractVector, y::AbstractVector; kwargs...) = UnicodePlots.lineplot(x, y ; kwargs...)
# uplot!(plt::UnicodePlots.Plot, x::AbstractVector, y::AbstractVector; kwargs...) = UnicodePlots.lineplot!(plt, x, y ; kwargs...)
# plot_data(x) = x    # nontrivial methods defined for custom types

# xlims(x::Real) = x
# xlims(x::Complex) = real(x)
# xlims(v::AbstractVector{<:Number}) = [extrema(real(v))...]
# xlims(vs::AbstractVector) = [extrema( vcat( xlims.(vs) ) )...]
# function xlims(a,b)
# 	xlims_a, xlims_b = xlims(a), xlims(b)
# 	[ min(xlims_a[1], xlims_b[1]), max(xlims_a[2], xlims_b[2]) ]
# end
# xlims(plt::UnicodePlots.Plot) = parse.(Float64,getfield.(unique(plt.decorations),:second))

# ylims(x::Real) = x
# ylims(x::Complex) = real(x)
# ylims(v::AbstractVector{<:Number}) = [extrema(real(v))...]
# ylims(vs::AbstractVector) = [extrema( vcat( ylims.(vs) ) )...]
# function ylims(fns::AbstractVector{<:Function};xlims=(0.5,1.8),nvals=100)
# 	xrng = range(xlims[1],xlims[2];length=nvals)
# 	[ extrema(hcat(map.(fns,(xrng,) )...) )... ]
# end
# function ylims(a,b)
# 	ylims_a, ylims_b = ylims(a), ylims(b)
# 	[ min(ylims_a[1], ylims_b[1]), max(ylims_a[2], ylims_b[2]) ]
# end
# ylims(plt::UnicodePlots.Plot) = parse.(Float64,getfield.(unique(plt.labels_left),:second))


# function ylims(plt::UnicodePlots.Plot, new ; kwargs...)
# 	ylims_plt = parse.(Float64,getfield.(unique(plt.labels_left),:second))
# 	ylims(ylims_plt,ylims(new))
# end

## Add methods to external packages ##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

## Includes ##

# include("plot.jl")
include("linalg.jl")
include("epsilon.jl")
include("materials.jl")
include("grid.jl")
include("geometry.jl")
include("cse.jl")
include("epsilon_fns.jl")
include("smooth.jl")
include("maxwell.jl")
include("fields.jl")
include("solve.jl")

include("grads.jl")



# include("constraints.jl")

# include("grads.jl")
# include("explicit.jl")
# include("io.jl")
# include("optimize.jl")


# # These empty Python object assignments just set asside pointers for loading Python modules
const pymeep    =   PyNULL()
const pympb     =   PyNULL()
# # const numpy     =   PyNULL()

# using Requires
# # function __init__()
#     @require KrylovKit="0b1a1467-8014-51b9-945f-bf0ae24f4b77" include("solvers/krylovkit.jl")

# 	@require IterativeSolvers="42fd0dbc-a981-5370-80f2-aaf504508153" include("solvers/iterativesolvers.jl")

# 	@require DFTK="acf6eb54-70d9-11e9-0013-234b7a5f5337" include("solvers/dftk.jl")

#     # @require PyCall="acf6eb54-70d9-11e9-0013-234b7a5f5337" begin
#         # using .PyCall
#         # @eval global mp = pyimport("meep")
#         # @eval global mpb = pyimport("meep.mpb")
#         # @eval global np = pyimport("numpy")
function __init__()
    copy!(pymeep, pyimport("meep"))
    copy!(pympb, pyimport("meep.mpb"))
    # copy!(numpy, pyimport("numpy"))
    # wurlitzer = pyimport("wurlitzer")
    py"""
    import numpy
    import h5py
    from numpy import linspace, transpose
    from scipy.interpolate import interp2d
    from meep import Medium, Vector3

    def return_evec(evecs_out):
        fn = lambda ms, band: numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0])
        return fn
        
    def save_evecs(ms,band):
        ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5")
        
    def return_and_save_evecs(evecs_out):
        # fn = lambda ms, band: ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5"); numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0]) )
        def fn(ms,band):
            ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5")
            # numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0])
            # ms.get_eigenvectors(band,1)[:,:,0]
            numpy.copyto(evecs_out[band-1,:,:],numpy.transpose(ms.get_eigenvectors(band,1)[:,:,0]))
        return fn

    def output_dfield_energy(ms,band):
        D = ms.get_dfield(band, bloch_phase=False)
        # U, xr, xi, yr, yi, zr, zi = ms.compute_field_energy()
        # numpy.savetxt(
        #     f"dfield_energy.b{band:02}.csv",
        #     [U, xr, xi, yr, yi, zr, zi],
        #     delimiter=","
        # )
        ms.compute_field_energy()

    # load epsilon data into python closure function `fmat(p)`
    # `fmat(p)` should accept an input point `p` of type meep.Vector3 as a single argument
    # and return a "meep.Material" object with dielectric tensor data for that point
    def matfn_from_file(fpath,Dx,Dy,Nx,Ny):
        x = linspace(-Dx/2., Dx*(0.5 - 1./Nx), Nx)
        y = linspace(-Dy/2., Dy*(0.5 - 1./Ny), Ny)
        with h5py.File(fpath, 'r') as f:
            f_epsxx,f_epsxy,f_epsxz,f_epsyy,f_epsyz,f_epszz = [interp2d(x,y,transpose(f["epsilon."+s])) for s in ["xx","xy","xz","yy","yz","zz"] ]
        matfn = lambda p : Medium(epsilon_diag=Vector3( f_epsxx(p.x,p.y)[0], f_epsyy(p.x,p.y)[0], f_epszz(p.x,p.y)[0] ),epsilon_offdiag=Vector3( f_epsxy(p.x,p.y)[0], f_epsxz(p.x,p.y)[0], f_epsyz(p.x,p.y)[0] ))
        return matfn

    # Transfer Julia epsilon data directly into python closure function `fmat(p)`
    # `fmat(p)` should accept an input point `p` of type meep.Vector3 as a single argument
    # and return a "meep.Material" object with dielectric tensor data for that point
    def matfn(eps,x,y):
        f_epsxx,f_epsxy,f_epsxz,f_epsyy,f_epsyz,f_epszz = [interp2d(x,y,transpose(eps[ix,iy,:,:])) for (ix,iy) in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)] ]
        matfn = lambda p : Medium(epsilon_diag=Vector3( f_epsxx(p.x,p.y)[0], f_epsyy(p.x,p.y)[0], f_epszz(p.x,p.y)[0] ),epsilon_offdiag=Vector3( f_epsxy(p.x,p.y)[0], f_epsxz(p.x,p.y)[0], f_epsyz(p.x,p.y)[0] ))
        return matfn
    """
        
    # include("solvers/mpb.jl")
    # end
end

include("solvers/mpb.jl")
include("solvers/iterativesolvers.jl")
include("solvers/krylovkit.jl")
include("solvers/dftk.jl")
# include("analyze.jl")

## Definitions ##


## More includes ##


end
