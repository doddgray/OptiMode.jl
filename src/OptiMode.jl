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
#   - ModePerturbations:   first-order perturbation theory for mode properties
#                          (thermo-optic, roughness/substrate loss, χ⁽²⁾/χ⁽³⁾)
#   - ModeSweeps:          batched/asynchronous (SLURM) mode-simulation sweeps
#   - EigenmodeExpansion:  GDS-driven, MEOW/SAX-style eigenmode-expansion (EME)
#                          built on the differentiable mode solver
#
# The component packages live in `lib/` within this repository.

module OptiMode

using Reexport

@reexport using MaterialDispersion
@reexport using DielectricSmoothing
@reexport using MaxwellEigenmodes
@reexport using ModeAnalysis
@reexport using ModePerturbations
@reexport using ModeSweeps
@reexport using EigenmodeExpansion

end # module OptiMode
