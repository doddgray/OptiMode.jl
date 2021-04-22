#!/bin/bash

# #SBATCH -o top5.out-%A-%a
# #SBATCH -a 0-3

# run with: sbatch submit_sbatch.sh
# copied from example in (github.com/llsc-supercloud/teaching-examples)[https://github.com/llsc-supercloud/teaching-examples/blob/master/Julia/word_count/JobArray/submit_sbatch.sh]

# Initialize Modules
source /etc/profile

# Load Julia Module
# module load julia/1.5.2

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

julia --optimize=3 -t12 LN_sweep.jl  #$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
