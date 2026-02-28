# OptiMode Sub-packages

This directory documents the four Julia packages that OptiMode.jl is refactored into.
Each package is a standalone Julia package located in `/home/user/`.

## Package Structure

```
/home/user/
├── OptiMode.jl/              # Top-level umbrella package (re-exports all)
├── MaterialModels.jl/        # Package 1: Dispersion modeling
├── DielectricSmoother.jl/    # Package 2: Tensor smoothing on FD grids
├── EigenModeSolver.jl/       # Package 3: Electromagnetic eigenmode solving
└── ModeAnalysis.jl/          # Package 4: Post-processing of mode solutions
```

## Dependency Graph

```
MaterialModels.jl
       ↓
DielectricSmoother.jl
       ↓
EigenModeSolver.jl
       ↓
ModeAnalysis.jl
       ↓
OptiMode.jl (re-exports all)
```

## Package Descriptions

### 1. MaterialModels.jl
**Dielectric material dispersion modeling**

Provides symbolic and numerical dispersion models for optical materials.

Key features:
- Sellmeier, Cauchy, and NASA thermo-optic dispersion formulas
- Symbolic differentiation for group index and GVD
- Material rotation for anisotropic crystals
- Jacobian/Hessian generation for frequency-dependent permittivity
- **AD**: ChainRulesCore, Mooncake.jl, Enzyme.jl

### 2. DielectricSmoother.jl
**Dielectric tensor smoothing on finite difference grids**

Implements Kottke's subpixel smoothing method for permittivity tensors at
material interfaces in FDFD/FDTD electromagnetic simulations.

Key features:
- `Grid` type: N-dimensional Cartesian spatial grid
- `Geometry` type: shapes with associated material data
- Kottke τ-transform averaging with correct interface normals
- Precomputed Jacobians/Hessians for the smoothed tensor
- **AD**: ChainRulesCore, Mooncake.jl, Enzyme.jl, Reactant.jl

### 3. EigenModeSolver.jl
**Electromagnetic eigenmode solving**

FFT-based plane-wave expansion (Helmholtz operator) with iterative eigensolvers.

Key features:
- `HelmholtzMap`: ∇×ε⁻¹∇× operator via FFT
- `ModeSolver`: complete solver state
- Multiple backends: KrylovKit, LOBPCG, DFTK
- `solve_ω²`: find frequencies for given k
- `solve_k`: find wavevector for given ω (dispersion)
- Adjoint method for reverse-mode AD through eigensolvers
- **AD**: ChainRulesCore (adjoint), Mooncake.jl, Enzyme.jl, Reactant.jl

### 4. ModeAnalysis.jl
**Post-processing of electromagnetic eigenmode solutions**

Tools for computing physical observables from mode solver output.

Key features:
- H⃗ → D⃗ → E⃗ field conversion pipeline
- Poynting vector (S⃗) computation
- Group index and GVD from field data
- Mode normalization, phase canonicalization
- Overlap integrals, polarization analysis
- **AD**: ChainRulesCore, Mooncake.jl, Enzyme.jl, Reactant.jl

## AD Interface Design

Each package implements three layers of AD support:

### Layer 1: ChainRulesCore (always available)
Standard Julia AD protocol, compatible with Zygote, Diffractor, etc.
Custom `rrule` definitions for non-trivial operations.

### Layer 2: Mooncake.jl (via package extension)
Source-transformation reverse-mode AD. More efficient than Zygote for
code with loops and in-place operations. Custom `rrule!!` definitions for
operations that Mooncake cannot handle automatically.

### Layer 3: Enzyme.jl (via package extension)
Source-transformation forward/reverse AD. Particularly effective for
loop-heavy code and when combined with Reactant.jl for GPU execution.
Custom `EnzymeRules.augmented_primal` / `reverse` for key operations.

### Layer 4: Reactant.jl (EigenModeSolver, ModeAnalysis, DielectricSmoother)
XLA compilation for GPU/TPU execution. Arithmetic-heavy operations
(tensor contractions, FFTs) can be compiled to XLA for acceleration.

## Usage Example

```julia
using OptiMode

# Define geometry
g = Grid(2.0, 2.0, 64, 64)  # 2D grid, 2μm × 2μm, 64×64 points
box = Box(SVector(0.0, 0.0, 0.0), SVector(0.45, 0.22, 0.0),
          Matrix(I, 3, 3), NumMat(silicon))
shapes = (box,)

# Compute dielectric tensor with Kottke smoothing
λ = 1.55  # wavelength in μm
mats = [NumMat(silicon), NumMat(vacuum)]
mat_vals = hcat([vcat(vec(m.fε(λ)), vec(m.fnng(λ)), vec(m.fngvd(λ))) for m in mats]...)
minds = matinds(shapes, mats)
ε_smooth = smooth_ε(shapes, mat_vals, minds, g)

# Solve for eigenmodes
ε⁻¹ = sliceinv_3x3(ε_smooth[1:3, 1:3, :, :])
k = 0.9  # Bloch wavevector in μm⁻¹
ms = ModeSolver(k, ε⁻¹, g)
ω²s, Hvecs = solve_ω²(ms, KrylovKitEigsolve(); nev=2)

# Analyze modes
H = reshape(Hvecs[1], 2, g.Nx, g.Ny)
E = E⃗(H, ms.M̂)
S = S⃗(E, E)
px, py, pz = E_relpower_xyz(E)
println("Mode 1: ω = $(sqrt(ω²s[1])) μm⁻¹, TE fraction = $(px)")

# Gradient-based optimization (using Zygote)
using Zygote
function loss(ε⁻¹)
    ms = ModeSolver(k, ε⁻¹, g)
    ω²s, Hvecs = solve_ω²(ms, KrylovKitEigsolve(); nev=1)
    H = reshape(Hvecs[1], 2, g.Nx, g.Ny)
    E = E⃗(H, ms.M̂)
    px, py, pz = E_relpower_xyz(E)
    return -px  # maximize TE confinement
end
grad = Zygote.gradient(loss, ε⁻¹)
```
