# EigenmodeExpansion.jl

Differentiable **eigenmode expansion** (EME) on top of the OptiMode mode solver,
following the algorithms of the [MEOW](https://github.com/flaport/meow) and
[SAX](https://github.com/flaport/sax) Python packages.

A GDS layout (e.g. from [GDSFactory](https://gdsfactory.github.io/gdsfactory/))
plus a vertical layer stack defines a 3D structure; the structure is sliced
perpendicular to the propagation axis into z-invariant *cells*; each cell's 2D
cross-section is solved with OptiMode's plane-wave eigensolver (`solve_k`); and
the cells are joined by mode-overlap *interface* S-matrices and diagonal
*propagation* S-matrices, then cascaded (Redheffer star product, as SAX does
internally) into the device scattering matrix.

```text
GDS ─▶ Structure (+ LayerStack) ─▶ cells ─▶ per-cell modes (solve_k)
    ─▶ interface + propagation S-matrices ─▶ cascade ─▶ device S-matrix
```

## Quick start

```julia
using OptiMode                       # re-exports EigenmodeExpansion

layout = read_gds("adiabatic_coupler.gds")        # GDSFactory export (flattened)
stack  = LayerStack(
    layers     = [Layer(gds_layer=1, zmin=0.0, zmax=0.3, material=1, patterned=true)],
    materials  = Any[Si₃N₄, SiO₂],   # column 1 = core, column 2 = cladding
    background = 2, prop_axis = :x,
)
structure = Structure(layout, stack; transverse_pad=2.0, vertical_pad=1.0)

res = eme_smatrix(structure, 1/1.55; nev=2, Nx=128, Ny=64, num_cells=30)
power_coupling(res; in_mode=1, out_mode=2)        # cross-coupling |S₂₁[2,1]|²
```

A self-contained, GDSFactory-driven example is in
[`examples/eme_adiabatic_coupler.jl`](../../examples/eme_adiabatic_coupler.jl)
(layout generator: [`examples/gen_adiabatic_coupler_gds.py`](../../examples/gen_adiabatic_coupler_gds.py)).

## Matching MEOW / SAX

- **Mode overlap / normalization** — `inner_product(m1, m2)` is MEOW's
  `½∫(E₁ₓH₂ᵧ − E₁ᵧH₂ₓ)dA`; modes are power-normalized to unit self-overlap.
- **Interface S-matrix** — `interface_smatrix` reproduces MEOW's overlap-matrix
  solve (`A_LR = O_LR + O_RLᵀ`, transmission `= pinv(A)·2I`, reflection from both
  continuity equations averaged), with a Tikhonov-regularized `reg_solve` in
  place of MEOW's truncated-SVD `tsvd_solve` (differentiable).
- **Propagation S-matrix** — `propagation_smatrix` is the diagonal phase
  `exp(2πi·k·L)` (k in cycles/µm).
- **Cascade** — `cascade`/`star` is the Redheffer star product over the linear
  chain `prop₀ ⋆ iface₀₁ ⋆ prop₁ ⋆ … ⋆ prop_{n-1}`, i.e. what SAX's circuit
  backend evaluates.

## Automatic differentiation

The whole forward computation is the OptiMode mode solver plus matmuls and linear
solves, so it is differentiable in **forward** mode (ForwardDiff through the
geometry → smoothing → dielectric stage) and **reverse** mode (Zygote/Mooncake/
Enzyme): the expensive per-cell eigensolve uses `solve_k`'s adjoint-method
`rrule`, and the overlap/interface/propagation/cascade algebra carries
ChainRules-native rules. Discrete GDS parsing and polygon scan-line intersection
are marked non-differentiable. `test/runtests.jl` checks forward-mode dielectric
sensitivities and reverse-mode end-to-end `ε⁻¹` sensitivities against
`FiniteDifferences.jl`.

## SLURM / parameter sweeps

The per-cell mode solves map one-to-one onto ModeSweeps tasks. With `ModeSweeps`
loaded, deploy them as a SLURM array job and re-assemble locally:

```julia
using ModeSweeps
grid  = simulation_grid(structure, 128, 64)
cells = build_cells(structure; num_cells=30)
batch = deploy_eme("examples/eme_coupler_setup.jl", length(cells);
                   ω=1/1.55, nev=2, slurm=SlurmConfig(ssh="me@cluster", remote_dir="/scratch/me/eme"))
res   = gather_eme(batch, cells, structure.stack.materials, 1/1.55, grid; nev=2)
```

Full documentation: [`docs/eigenmode_expansion.md`](../../docs/eigenmode_expansion.md).
