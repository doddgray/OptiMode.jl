using Distributed

# launch worker processes
addprocs(4)

println("Number of processes: ", nprocs())
println("Number of workers: ", nworkers())

# each worker gets its id, process id and hostname
for i in workers()
    id, pid, host = fetch(@spawnat i (myid(), getpid(), gethostname()))
    println(id, " " , pid, " ", host)
end

# remove the workers
for i in workers()
    rmprocs(i)
end


## reference examples
##      github.com/llsc-supercloud/teaching-examples/blob/master/Julia/word_count/JobArray/submit_sbatch.sh
##      discourse.julialang.org/t/running-julia-in-a-slurm-cluster/67614
