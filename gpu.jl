# CUDA.jl Tutorial

using LinearAlgebra, CUDA, BenchmarkTools, Test

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x
## CPU Kernels

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

@btime sequential_add!($y, $x)
# 148.153 μs (0 allocations: 0 bytes)
@btime parallel_add!($y, $x)
# 38.042 μs (31 allocations: 2.75 KiB)

## GPU Kernels

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

@btime add_broadcast!($y_d, $x_d)
# 147.882 μs (21 allocations: 528 bytes)

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@btime bench_gpu1!($y_d, $x_d)
# 150.796 ms (16 allocations: 368 bytes)

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

@btime bench_gpu2!($y_d, $x_d)
# 1.194 ms (16 allocations: 368 bytes)

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)

fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end

@btime bench_gpu3!($y_d, $x_d)
# 173.399 μs (16 allocations: 368 bytes)


## FoldsCUDA.jl examples

using FoldsCUDA, CUDA, FLoops
using GPUArrays: @allowscalar

xs = CUDA.rand(10^8);
@allowscalar xs[100] = 2;
@allowscalar xs[200] = 2;

@floop CUDAEx() for (x, i) in zip(xs, eachindex(xs))
       @reduce() do (imax = -1; i), (xmax = -Inf32; x)
           if xmax < x
               xmax = x
               imax = i
           end
       end
   end

xmax
# 2.0f0

imax

## Tullio.jl examples

using Tullio
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Zygote
ΔA = Zygote.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CUDA, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] # true

# Reduction over min/max:
Tracker.gradient(x -> (@tullio (max) res := x[i]^3), [1,2,3,-2,-1,3])[1]
