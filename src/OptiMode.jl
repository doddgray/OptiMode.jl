################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                                                              #
################################################################################
#
# OptiMode is now an umbrella package re-exporting four component packages:
#
#   - MaterialDispersion:  symbolic dielectric material dispersion models
#   - DielectricSmoothing: sub-pixel (Kottke) smoothing of dielectric tensors
#                          on finite-difference spatial grids
#   - MaxwellEigenmodes:   plane-wave electromagnetic eigenmode solver with
#                          adjoint-based AD rules
#   - ModeAnalysis:        post-processing of mode solver results
#   - ModeSweeps:          batched/asynchronous (SLURM) mode-simulation sweeps
#
# The component packages live in `lib/` within this repository.

module OptiMode

using Reexport

@reexport using MaterialDispersion
@reexport using DielectricSmoothing
@reexport using MaxwellEigenmodes
@reexport using ModeAnalysis
@reexport using ModeSweeps

end # module OptiMode
