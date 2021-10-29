#!/bin/bash

#SBATCH -p <list of partition names>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dodd@mit.edu

source /etc/profile
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT
julia test_distributed.jl
# julia --optimize=3 -t12 LN_sweep.jl  #$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

## reference examples
##      github.com/llsc-supercloud/teaching-examples/blob/master/Julia/word_count/JobArray/submit_sbatch.sh
##      discourse.julialang.org/t/running-julia-in-a-slurm-cluster/67614



