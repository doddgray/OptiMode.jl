# CUDA.jl device support for the GPU-capable eigensolver backend (`GPUSolver`).
#
# The solver core in `src/solvers/gpu.jl` is device-generic (broadcast-only kernels,
# AbstractFFTs plans, KrylovKit); this extension only supplies the `:cuda` device
# transfer, after which FFT plans resolve to CUFFT and all arithmetic runs on the GPU.

module MaxwellEigenmodesCUDAExt

using MaxwellEigenmodes
using CUDA

function MaxwellEigenmodes._device_array(::Val{:cuda}, A::AbstractArray)
    CUDA.functional() || error(
        "GPUSolver(device=:cuda): CUDA.jl is loaded but no functional CUDA GPU is available. " *
        "Use GPUSolver(T; device=:cpu) to run the same solver on the CPU.")
    return CuArray(A)
end

end # module
