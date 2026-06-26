################################################################################
#                                                                              #
#                            ModePerturbations.jl                              #
#      First-order perturbation theory for guided-mode properties of the       #
#      OptiMode differentiable mode solver: weak shifts of effective index,    #
#      group index, GVD and linear/nonlinear loss from material, thermal,      #
#      roughness, substrate-leakage and χ⁽²⁾/χ⁽³⁾ perturbations — all          #
#      end-to-end AD compatible (forward & reverse) and FD-validated.          #
#                                                                              #
################################################################################

module ModePerturbations

using LinearAlgebra
using LinearAlgebra: dot, norm, tr
using StaticArrays
using FFTW
using Tullio
using EllipsisNotation

using ChainRulesCore
using ChainRulesCore: @non_differentiable, NoTangent, ZeroTangent, AbstractZero, ignore_derivatives
import ChainRulesCore: rrule, frule
using Zygote      # used (only) to assemble the reverse rule for the perturbation kernel
using ForwardDiff # used (only) to assemble the forward rule for the perturbation kernel

# Mode-solver types, operators and field utilities
using DielectricSmoothing
using DielectricSmoothing: Grid, _fftaxes, δV, δx, δy, N
using MaxwellEigenmodes
using MaxwellEigenmodes: mag_mn, kx_tc, kx_ct, zx_tc, zx_ct, tc, ct, ε⁻¹_dot,
    HMH, HMₖH, _dot, _outer, sliceinv_3x3, E⃗, S⃗z
using ModeAnalysis
using ModeAnalysis: poynting_z, mode_intensity, group_index

## Includes ##
include("contractions.jl")
include("perturbation.jl")
include("index_effects.jl")
include("scattering.jl")
include("nonlinear.jl")
include("ad_rules.jl")

end # module ModePerturbations
