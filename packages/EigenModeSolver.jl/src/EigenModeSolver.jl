"""
    EigenModeSolver

A Julia package for solving electromagnetic eigenmodes of dielectric structures.

Implements a plane-wave expansion (FFT-based) Maxwell operator and iterative
eigenvalue solvers to find the resonant electromagnetic modes of a structure
described by its inverse dielectric tensor ε⁻¹(r).

The core algorithm is equivalent to the approach used in the MIT Photonic Bands
(MPB) software, but implemented fully in Julia with support for:
- Automatic differentiation (forward and reverse mode) through the full solve
- Multiple eigensolver backends (KrylovKit, LOBPCG, DFTK)
- Pluggable preconditioners
- Both ω-solve (find ω given k) and k-solve (find k given ω)
- Adjoint method for reverse-mode AD through iterative eigensolvers

Key types:
- `HelmholtzMap`: discretized ∇×ε⁻¹∇× operator
- `HelmholtzPreconditioner`: diagonal preconditioner
- `ModeSolver`: complete solver state (operator + wavevector + eigenvectors)

Key functions:
- `solve_ω²`: find eigenfrequencies ω² for given Bloch wavevector k
- `solve_k`: find wavevector k for given frequency ω (dispersion solve)
- `filter_eigs`: filter eigenmodes by user-specified criteria
"""
module EigenModeSolver

using LinearAlgebra
using LinearAlgebra: mul!, ldiv!
using StaticArrays
using StaticArrays: Dynamic, SVector
using HybridArrays
using FFTW
using AbstractFFTs
using LoopVectorization
using Tullio
using SliceMap
using LinearMaps
using IterativeSolvers
using IterativeSolvers: gmres, lobpcg, lobpcg!
using KrylovKit
using Roots
using Logging
using ProgressMeter
using ChainRulesCore
using ChainRulesCore: @thunk, @non_differentiable, @not_implemented, NoTangent, ZeroTangent, AbstractZero
using Zygote
using Zygote: Buffer, bufferfrom, @ignore, @adjoint, ignore, dropgrad, forwarddiff,
    Numeric, literal_getproperty, accum
using MaterialModels
using DielectricSmoother

import Base: size, eltype
import LinearAlgebra: mul!
import ChainRulesCore: rrule

# FFTW thread settings (single thread for compatibility with external parallelism)
FFTW.set_num_threads(1)

## Add methods to external packages ##
LinearAlgebra.ldiv!(c, A::LinearMaps.LinearMap, b) = mul!(c, A', b)

include("logging.jl")
include("linalg.jl")
include("maxwell.jl")
include("solve.jl")
include("grads/StaticArrays.jl")
include("grads/solve.jl")
include("solvers/krylovkit.jl")
include("solvers/iterativesolvers.jl")
include("solvers/dftk.jl")

end # module EigenModeSolver
