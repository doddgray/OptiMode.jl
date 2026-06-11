################################################################################
#                                                                              #
#                            MaxwellEigenmodes.jl:                             #
#         Differentiable electromagnetic eigenmode solver operating on        #
#       smoothed dielectric tensor data on finite-difference spatial grids    #
#                                                                              #
################################################################################

module MaxwellEigenmodes

##### Imports ######

### Linear Algebra Types and Libraries ###
using LinearAlgebra
using LinearAlgebra: diag
using LinearMaps
using StaticArrays
using StaticArrays: SVector, SMatrix
using FFTW
using AbstractFFTs
using Tullio

### AD rule infrastructure ###
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, @not_implemented, NoTangent, ZeroTangent,
    AbstractZero, ignore_derivatives

### Grid types ###
using DielectricSmoothing
using DielectricSmoothing: Grid, g⃗, _fftaxes, N, nₘₐₓ

### Iterative Solvers ###
using IterativeSolvers
using KrylovKit
using Roots

### I/O & Logging ###
using Logging
using Logging: AbstractLogger, NullLogger, with_logger

# Import methods that we will overload for custom types
import Base: size, eltype
import LinearAlgebra: mul!
import ChainRulesCore: rrule

# FFTW settings
FFTW.set_num_threads(1)     # chosen for thread safety when combined with other parallel code, consider increasing

## Add methods to external packages ##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

export k_guess

k_guess(ω,ε⁻¹::AbstractArray{<:Real,4}) = first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:]) for a=1:3]))
k_guess(ω,ε⁻¹::AbstractArray{<:Real,5}) = first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3]))
k_guess(ω,geom) = nₘₐₓ(ω,geom) * ω

## Includes ##
include("logging.jl")
include("linalg.jl")
include("maxwell.jl")
include("fields.jl")
include("solve.jl")
include("grads/StaticArrays.jl")
include("grads/solve.jl")
include("solvers/iterativesolvers.jl")
include("solvers/krylovkit.jl")
include("solvers/dftk.jl")
include("solvers/mpb.jl")

end # module MaxwellEigenmodes
