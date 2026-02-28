"""
    ReactantExt (EigenModeSolver)

Reactant.jl (XLA backend) extension for EigenModeSolver.jl.

Enables compiling and running the HelmholtzMap operator on XLA-compatible
hardware (GPU/TPU) using Reactant's tracing infrastructure.

The core HelmholtzMap operation consists of:
1. Cross product in k-space: k×H (via kx_tc)
2. Forward FFT
3. Inverse permittivity multiplication: ε⁻¹·D
4. Inverse FFT
5. Cross product: k×E (via kx_ct)

Operations 1, 3, 5 are pure array arithmetic (XLA-compatible).
Operations 2, 4 require FFTW→XLA translation (handled via Reactant's FFTW extension).

Usage:
```julia
using EigenModeSolver, Reactant

# Compile the HelmholtzMap application for XLA
ms = ModeSolver(k, ε⁻¹, grid)
compiled_helmholtz = EigenModeSolver.compile_helmholtz_xla(ms)

# Apply compiled operator
H_in = ConcreteRArray(rand(ComplexF32, 2, Nx, Ny))
H_out = compiled_helmholtz(H_in)
```
"""
module ReactantExt

using EigenModeSolver
using EigenModeSolver: HelmholtzMap, ModeSolver, kx_tc!, kx_ct!, ε⁻¹_dot_t!, tc!, ct!
using Reactant
using Reactant: ConcreteRArray, @compile, @jit
using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# HelmholtzMap application: M̂ * H for XLA backend
# This is the inner loop of the iterative eigensolver.
# ──────────────────────────────────────────────────────────────────────────────

"""
    helmholtz_apply_xla(ε⁻¹, mn, mag, H_in, plan_fwd, plan_inv)

XLA-compatible application of the Helmholtz operator to a field H.
All arguments should be ConcreteRArrays or compatible with Reactant tracing.
"""
function helmholtz_apply_xla(ε⁻¹, mn, mag, H_in)
    # This function expresses the Helmholtz operator in pure array operations
    # that Reactant can trace and compile to XLA.
    # The FFT operations will use Reactant's FFT support.

    # Step 1: k×H (transverse to Cartesian, applies wavevector cross product)
    d = similar(H_in, size(H_in)...)
    EigenModeSolver.kx_tc!(d, H_in, mn, mag)

    # Step 2: Forward FFT (spatial → frequency domain)
    d_fft = Reactant.stablehlo.fft(d)  # TODO: use Reactant's FFT when available

    # Step 3: ε⁻¹ · D
    e = similar(d_fft)
    EigenModeSolver.ε⁻¹_dot_t!(e, ε⁻¹, d_fft)

    # Step 4: Inverse FFT
    e_real = Reactant.stablehlo.fft(e, inverse=true)

    # Step 5: k×E (Cartesian to transverse)
    H_out = similar(H_in)
    EigenModeSolver.kx_ct!(H_out, e_real, mn, mag)

    return H_out
end

"""
    compile_helmholtz_xla(ms::ModeSolver)

Compile the HelmholtzMap application for XLA backend using Reactant.
Returns a compiled function that takes H_in as ConcreteRArray and returns H_out.
"""
function compile_helmholtz_xla(ms::ModeSolver{2,T}) where T
    M̂ = ms.M̂
    ε⁻¹_ra = ConcreteRArray(M̂.ε⁻¹)
    mn_ra = ConcreteRArray(M̂.mn)
    mag_ra = ConcreteRArray(M̂.mag)
    H_dummy = ConcreteRArray(zeros(Complex{T}, size(ms.H⃗)))

    return @compile helmholtz_apply_xla(ε⁻¹_ra, mn_ra, mag_ra, H_dummy)
end

end # module ReactantExt
