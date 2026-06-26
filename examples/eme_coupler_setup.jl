# ModeSweeps setup script for deploying an EME stack's per-cell mode solves as a
# SLURM batch (one task per (cell, ω)). It is copied into the batch directory and
# `include`d by every worker, so it `using`s what it needs and defines
# `make_problem(p)`; here `p.cell` selects the EME cell and `p.ω` the frequency.
#
# Pair with `deploy_eme(@__FILE__, CELLS; ω, nev=2, ...)` (dedup'd cell × ω tasks;
# `ω` may be a vector) and re-assemble per frequency with
# `gather_eme(batch, CELLS, MATS, ω, GRID)`.

using EigenmodeExpansion
using MaterialDispersion: Si₃N₄, SiO₂

const _GDS = get(ENV, "EME_COUPLER_GDS", joinpath(@__DIR__, "adiabatic_coupler.gds"))

const STACK = LayerStack(
    layers=[Layer(gds_layer=1, zmin=0.0, zmax=0.3, material=1, patterned=true, name="SiN core")],
    materials=Any[Si₃N₄, SiO₂],
    background=2,
    prop_axis=:x,
)
const STRUCTURE = Structure(read_gds(_GDS), STACK; transverse_pad=2.0, vertical_pad=1.0)
const GRID = simulation_grid(STRUCTURE, 128, 64)
const NUM_CELLS = parse(Int, get(ENV, "EME_NUM_CELLS", "30"))
const CELLS = build_cells(STRUCTURE; num_cells=NUM_CELLS)
const MATS = STACK.materials

make_problem(p) = cell_problem(CELLS[p.cell], MATS, p.ω, GRID)
