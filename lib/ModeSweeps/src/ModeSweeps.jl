################################################################################
#                                                                              #
#                                ModeSweeps.jl:                                #
#       Batched & asynchronous (SLURM) deployment of mode simulations         #
#       with parameter sweeps, status tracking, and tabular result I/O        #
#                                                                              #
################################################################################

module ModeSweeps

using Dates
using Dates: DateTime
using Printf
using TOML
using Serialization
using JSON3
using CSV
using Tables
using HDF5
using LinearAlgebra
using Logging
using ChainRulesCore
using ChainRulesCore: rrule, ZeroTangent, NoTangent

using MaterialDispersion
using DielectricSmoothing
using MaxwellEigenmodes
using ModeAnalysis

export param_grid, SlurmConfig
export deploy_batch, frequency_sweep, load_batch, batch_status, cancel_batch, wait_batch
export gather_batch, save_summary, load_summary, load_fields, plot_paths
export run_task, render_mode_png, write_png
# remote automatic differentiation (forward / backward passes as SLURM tasks)
export deploy_forward, forward_solution, deploy_backward, gradient_result,
    remote_value_and_gradient

include("params.jl")
include("render.jl")
include("batch.jl")
include("worker.jl")
include("adjoint.jl")
include("status.jl")
include("gather.jl")

end # module ModeSweeps
