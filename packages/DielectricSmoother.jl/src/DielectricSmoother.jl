"""
    DielectricSmoother

A Julia package for dielectric tensor smoothing on finite difference spatial grids.

Implements Kottke's interface-averaging method for subpixel smoothing of permittivity
tensors at material interfaces in finite difference time-domain (FDTD) and frequency-
domain (FDFD) simulations. Correct subpixel smoothing is critical for accuracy and
convergence in electromagnetic simulations.

Provides:
- `Grid` type: N-dimensional Cartesian spatial grid with voxel/corner utilities
- `Geometry` type: collection of shapes with associated material data
- Kottke subpixel smoothing (`smooth_ε`, `smooth_ε_single`)
- Precomputed Jacobians/Hessians for the smoothed dielectric tensor
- ChainRulesCore-compatible reverse-mode AD rules
- Extensions for Mooncake.jl, Enzyme.jl, and Reactant.jl
"""
module DielectricSmoother

using LinearAlgebra
using StaticArrays
using StaticArrays: SMatrix, SVector, SHermitianCompact
using GeometryPrimitives
using GeometryPrimitives: Shape, surfpt_nearby, volfrac
using Tullio
using ChainRulesCore
using ChainRulesCore: @non_differentiable, NoTangent, ZeroTangent, @thunk
using Symbolics
using SymbolicUtils
using Symbolics: Num, Sym, scalarize, expand_derivatives, substitute, simplify_fractions
using SymbolicUtils: @acrule
using MaterialModels

import Base: size, length, ndims, eltype, firstindex, lastindex, eachindex,
             getindex, LinearIndices, CartesianIndices, Dims, iterate

include("grid.jl")
include("geometry.jl")
include("epsilon_fns.jl")
include("smooth.jl")

end # module DielectricSmoother
