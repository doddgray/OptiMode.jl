"""
    ReactantExt (ModeAnalysis)

Reactant.jl (XLA backend) extension for ModeAnalysis.jl.

Compiles field post-processing operations for XLA hardware (GPU/TPU).
The E and H field conversions involve FFTs and tensor contractions that
map naturally to XLA operations.

Primary use case: batch computation of Poynting vectors and field overlaps
during optimization loops where the mode solver runs on GPU.
"""
module ReactantExt

using ModeAnalysis
using EigenModeSolver: HelmholtzMap
using Reactant
using Reactant: ConcreteRArray, @compile, @jit
using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# XLA-compatible E⃗ computation from H⃗ and ε⁻¹
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_E_from_H_xla(H, ε⁻¹, mn, mag)

Compute the electric field E from the magnetic field H and inverse dielectric tensor ε⁻¹.
All inputs should be ConcreteRArrays for XLA execution.

Returns E as an array of the same spatial shape as H.
"""
function compute_E_from_H_xla(H, ε⁻¹, mn, mag)
    # H → D: apply k× in reciprocal space, then FFT
    # D → E: multiply by ε⁻¹ in real space, then IFFT
    # This is a simplification — the actual implementation uses
    # the full FFT-based Maxwell operator.
    return ModeAnalysis._H2E(H, ε⁻¹, mn, mag)
end

"""
    compute_poynting_xla(E, H)

Compute the Poynting vector S = (E × H*) / 2 for XLA backend.
"""
function compute_poynting_xla(E, H)
    return ModeAnalysis.S⃗(E, H)
end

"""
    compute_E_overlap_xla(E1, E2, dV)

Compute the spatial overlap integral ∫ E1* · E2 dV between two modes.
"""
function compute_E_overlap_xla(E1, E2, dV)
    return sum(conj(E1) .* E2) * dV
end

"""
    compile_field_analysis(ms; mode_index=1)

Compile field analysis operations for XLA backend.
Returns a named tuple of compiled functions.
"""
function compile_field_analysis(ms)
    M̂ = ms.M̂
    H_dummy = ConcreteRArray(zeros(ComplexF32, 2, M̂.Nx, M̂.Ny))
    ε⁻¹_dummy = ConcreteRArray(M̂.ε⁻¹)
    mn_dummy = ConcreteRArray(M̂.mn)
    mag_dummy = ConcreteRArray(M̂.mag)

    compute_E_compiled = @compile compute_E_from_H_xla(H_dummy, ε⁻¹_dummy, mn_dummy, mag_dummy)

    E_dummy = ConcreteRArray(zeros(ComplexF32, 3, M̂.Nx, M̂.Ny))
    H3_dummy = ConcreteRArray(zeros(ComplexF32, 3, M̂.Nx, M̂.Ny))
    compute_S_compiled = @compile compute_poynting_xla(E_dummy, H3_dummy)

    return (
        compute_E = compute_E_compiled,
        compute_S = compute_S_compiled,
    )
end

end # module ReactantExt
