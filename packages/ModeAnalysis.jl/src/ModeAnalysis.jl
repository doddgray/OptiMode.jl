"""
    ModeAnalysis

Post-processing and analysis of electromagnetic eigenmode solutions.

Given the output of EigenModeSolver (eigenfrequencies ω² and eigenvectors H⃗),
provides tools to:

- Convert between field representations (H⃗ → E⃗ → D⃗ transformations)
- Compute Poynting vector (power flow) S⃗
- Compute group index n_g and group velocity dispersion (GVD) from field data
- Analyze mode spatial profiles (normalization, confinement, nodes)
- Compute overlap integrals and coupling coefficients between modes
- Filter and select modes by polarization, confinement, or other criteria

All post-processing functions are fully differentiable with respect to the
input eigenvectors H⃗ and inverse dielectric tensor ε⁻¹, enabling gradient-
based mode optimization.

Supported AD backends:
- ChainRulesCore (Zygote-compatible)
- Mooncake.jl (via extension)
- Enzyme.jl (via extension)
"""
module ModeAnalysis

using LinearAlgebra
using LinearAlgebra: diag, dot
using StaticArrays
using StaticArrays: SMatrix, SVector
using FFTW
using AbstractFFTs
using Tullio
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, NoTangent, ZeroTangent, AbstractZero
using MaterialModels
using DielectricSmoother
using EigenModeSolver

import Base: size, eltype
import ChainRulesCore: rrule

include("fields.jl")

end # module ModeAnalysis
