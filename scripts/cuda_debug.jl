using BenchmarkTools, Test, CUDA

a = CUDA.zeros(1024)

function kernel(a)
    i = threadIdx().x
    a[i] += 1
    return
end

@cuda threads=length(a) kernel(a)

##

N = 2^20
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
y_d .+= x_d

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

##
@btime add_broadcast!($y_d, $x_d)

##

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

##

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end
@btime bench_gpu1!($y_d, $x_d)


##

const nx = 1024  # do 1024 x 1024 2D FFT
xc = CuArray{ComplexF64}(CUDA.randn(Float64, nx, nx))
p = plan_fft!( xc )

##
@btime CUDA.@sync(p * x) setup=(
    x=CuArray{ComplexF64}(CUDA.randn(Float64, nx, nx)));

##
for device in CUDA.devices()
    @show capability(device)
end

## 

using AbstractFFTs
using CUDA.CUFFT
##
b = CUDA.rand(ComplexF32,64,64,64)
# pa = plan_fft( a )
@btime fft(b);