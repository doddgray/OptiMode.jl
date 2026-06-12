################################################################################
#                                                                              #
#                               ModeAnalysis.jl:                               #
#         Post-processing of electromagnetic eigenmode solver results         #
#       (group index, group velocity dispersion, mode classification, …)      #
#                                                                              #
################################################################################

module ModeAnalysis

##### Imports ######
using LinearAlgebra
using LinearAlgebra: normalize, dot
using StaticArrays
using StaticArrays: SVector, SMatrix, SArray
using FFTW
using Tullio
using EllipsisNotation

### AD rule infrastructure ###
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, NoTangent, ZeroTangent, AbstractZero,
    ignore_derivatives
import ChainRulesCore: rrule
using Zygote   # used (only) to assemble reverse rules for FFT/Tullio-based post-processing

### Mode solver types and field utilities ###
using DielectricSmoothing
using DielectricSmoothing: Grid, _fftaxes, N, δV
using MaxwellEigenmodes
using MaxwellEigenmodes: mag_mn, kx_tc, kx_ct, zx_tc, zx_ct, tc, ct, ε⁻¹_dot, HMH, HMₖH,
    _dot, _outer, _cross, _3dot, herm, ∇ₖmag_mn, eig_adjt,
    HelmholtzMap, HelmholtzPreconditioner

## Includes ##
include("group_index.jl")
include("analyze.jl")
include("kerr.jl")
include("ad_rules.jl")

end # module ModeAnalysis
