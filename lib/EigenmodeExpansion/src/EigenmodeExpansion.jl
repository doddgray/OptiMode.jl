################################################################################
#                                                                              #
#                           EigenmodeExpansion.jl                              #
#       Differentiable Eigenmode Expansion (EME) on top of OptiMode            #
#                                                                              #
################################################################################
#
# A MEOW/SAX-style eigenmode-expansion solver built on the OptiMode pipeline:
#
#   GDS layout ─▶ 3D structure (layer stack) ─▶ EME cells (cross-sections)
#       ─▶ per-cell modes (solve_k, adjoint-differentiable)
#       ─▶ interface + propagation S-matrices ─▶ SAX-style cascade ─▶ device S
#
# The whole forward computation is built from the OptiMode mode solver and plain
# linear algebra, so it is differentiable in both forward and reverse mode (the
# expensive eigensolve uses `solve_k`'s adjoint `rrule`). It plugs into the
# ModeSweeps SLURM batch/sweep machinery via the ModeSweeps extension.

module EigenmodeExpansion

using LinearAlgebra
using StaticArrays
using ChainRulesCore
using ChainRulesCore: @non_differentiable

using DielectricSmoothing
using MaterialDispersion
using MaxwellEigenmodes
using ModeAnalysis
import GeometryPrimitives

include("gds.jl")
include("structure.jl")
include("cache.jl")
include("modes.jl")
include("smatrix.jl")
include("eme.jl")
include("diagnostics.jl")
include("sweeps.jl")

# Discrete / IO operations are constants for AD: the EME differentiable variables
# are frequency, material data and (via the smoothing stack) geometry parameters,
# while GDS parsing and polygon scan-line intersection are integer/binary.
@non_differentiable read_gds(::Any)
@non_differentiable polygon_transverse_intervals(::Any, ::Any, ::Any, ::Any)

end # module EigenmodeExpansion
