"""
    ReactantExt (DielectricSmoother)

Reactant.jl (XLA backend) extension for DielectricSmoother.jl.

Enables running the differentiable parts of the dielectric smoothing pipeline
on XLA-compatible hardware (GPU/TPU) using Reactant's tracing infrastructure.

Note: Shape queries (corner_sinds, proc_sinds) involve conditional branching
based on discrete material indices and cannot be XLA-compiled. Only the
differentiable arithmetic parts (τ transforms, rotations, weighted averages)
are XLA-compatible.

The typical usage pattern is:
1. Precompute material indices and volume fractions on CPU
2. Compile the Kottke averaging arithmetic with Reactant for GPU acceleration
"""
module ReactantExt

using DielectricSmoother
using Reactant
using Reactant: ConcreteRArray, @compile, @jit

# ──────────────────────────────────────────────────────────────────────────────
# XLA-compatible Kottke averaging kernel
# Takes precomputed r₁, S matrices, and mat_vals (all as concrete arrays)
# and computes the smoothed dielectric tensor.
# ──────────────────────────────────────────────────────────────────────────────

"""
    kottke_average_xla(r₁, S, ε₁_flat, ε₂_flat)

XLA-compatible Kottke averaging of two dielectric tensors.
Inputs must be plain arrays (not GeometryPrimitives shapes).

- `r₁`: volume fraction of material 1 (scalar)
- `S`: 3×3 rotation matrix (interface frame)
- `ε₁_flat`: flattened 3×3 dielectric tensor of material 1
- `ε₂_flat`: flattened 3×3 dielectric tensor of material 2

Returns the Kottke-averaged dielectric tensor as a 9-element vector.
"""
function kottke_average_xla(r₁, S, ε₁_flat, ε₂_flat)
    ε₁ = reshape(ε₁_flat, (3,3))
    ε₂ = reshape(ε₂_flat, (3,3))
    ε₁ᵣ = DielectricSmoother._rotate(S, ε₁)
    ε₂ᵣ = DielectricSmoother._rotate(S, ε₂)
    εₑᵣ = DielectricSmoother.avg_param_rot(ε₁ᵣ, ε₂ᵣ, r₁)
    εₑ = DielectricSmoother._rotate(S', εₑᵣ)
    return vec(εₑ)
end

"""
    compile_kottke_kernel()

Compile the Kottke averaging kernel for XLA execution using Reactant.
Returns a compiled function that can be called with ConcreteRArray inputs.
"""
function compile_kottke_kernel()
    # Create dummy concrete arrays for tracing
    r₁_dummy = ConcreteRArray(fill(0.5f0, ()))
    S_dummy = ConcreteRArray(randn(Float32, 3, 3))
    ε₁_dummy = ConcreteRArray(randn(Float32, 9))
    ε₂_dummy = ConcreteRArray(randn(Float32, 9))

    return @compile kottke_average_xla(r₁_dummy, S_dummy, ε₁_dummy, ε₂_dummy)
end

"""
    smooth_ε_xla(mat_vals_all, r₁s, Ss, idx_pairs, grid_size)

Batch Kottke smoothing on XLA backend.

- `mat_vals_all`: (27, n_mats) matrix of [ε; ∂ωε; ∂²ωε] for each material
- `r₁s`: volume fractions at each grid point (Nx*Ny or Nx*Ny*Nz vector)
- `Ss`: interface normal matrices at each grid point (3, 3, N) array
- `idx_pairs`: (2, N) matrix of material index pairs at each grid point
- `grid_size`: (Nx, Ny) or (Nx, Ny, Nz) tuple

Returns smoothed dielectric tensor array of shape (3, 3, 3, grid_size...).
"""
function smooth_ε_xla(mat_vals_all, r₁s, Ss, idx_pairs, grid_size)
    N = prod(grid_size)
    results = similar(mat_vals_all, 27, N)

    # Vectorized Kottke averaging over grid
    for i in 1:N
        s1, s2 = idx_pairs[1,i], idx_pairs[2,i]
        if s1 == s2
            results[:,i] = mat_vals_all[:,s1]
        else
            r = r₁s[i]
            S = Ss[:,:,i]
            ε₁ = reshape(mat_vals_all[1:9, s1], (3,3))
            ε₂ = reshape(mat_vals_all[1:9, s2], (3,3))
            ∂ωε₁ = reshape(mat_vals_all[10:18, s1], (3,3))
            ∂ωε₂ = reshape(mat_vals_all[10:18, s2], (3,3))
            ∂²ωε₁ = reshape(mat_vals_all[19:27, s1], (3,3))
            ∂²ωε₂ = reshape(mat_vals_all[19:27, s2], (3,3))
            results[:,i] = DielectricSmoother.εₑ_∂ωεₑ_∂²ωεₑ(r, S, ε₁, ε₂, ∂ωε₁, ∂ωε₂, ∂²ωε₁, ∂²ωε₂)
        end
    end

    return reshape(results, (3, 3, 3, grid_size...))
end

end # module ReactantExt
