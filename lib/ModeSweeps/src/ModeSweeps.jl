################################################################################
#                                                                              #
#                                ModeSweeps.jl:                                #
#       Batched & asynchronous (SLURM) deployment of mode simulations         #
#       with parameter sweeps, status tracking, and tabular result I/O        #
#                                                                              #
################################################################################

module ModeSweeps

using Dates
using Printf
using TOML
using Serialization
using JSON3
using CSV
using Tables
using HDF5
using LinearAlgebra
using Logging

using MaterialDispersion
using DielectricSmoothing
using MaxwellEigenmodes
using ModeAnalysis

export param_grid, SlurmConfig
export deploy_batch, frequency_sweep, load_batch, batch_status, cancel_batch
export gather_batch, save_summary, load_summary, load_fields
export run_task

include("params.jl")
include("batch.jl")
include("worker.jl")
include("status.jl")
include("gather.jl")

end # module ModeSweeps
