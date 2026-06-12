# ModeSweeps problem-setup script: Si₃N₄-core rectangular waveguide with Kerr (n₂)
# power-dependent mode corrections.
#
# Because `make_problem` returns an `n₂` map, any parameter set containing an optical
# power `P` (W) triggers a first-order power-corrected solve in the worker, so power
# sweeps deploy exactly like any other batched parameter sweep:
#
#     using ModeSweeps
#     batch = deploy_batch("examples/kerr_power_sweep_setup.jl",
#         param_grid(ω = 1/1.55, P = 0.0:0.5:10.0, w_core = [1.2, 1.6, 2.0]);
#         name = "kerr_power_sweep", nev = 1,
#         slurm = SlurmConfig(time = "0:30:00"))          # or backend = :local
#     # … later, in any Julia session:
#     rows = gather_batch(load_batch("modesweeps_kerr_power_sweep"))
#     # rows have a `dneff_kerr` column: the power-dependent effective-index shift.
#
# Swept parameters: ω (frequency, μm⁻¹), P (power, W), w_core/h_core (core size, μm).

using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid

function make_problem(p)
    λ = 1 / p.ω
    w_core = haskey(p, :w_core) ? p.w_core : 1.60
    h_core = haskey(p, :h_core) ? p.h_core : 0.80
    grid = Grid(4.0, 3.0, 96, 72)

    mats = [Si₃N₄, SiO₂]
    f_ε, _ = _f_ε_mats(mats, (:ω,))
    mat_vals = f_ε([p.ω])
    core = MaterialShape(Cuboid([0.0, 0.0], [w_core, h_core], [1.0 0.0; 0.0 1.0]), 1)
    shapes, minds = (core,), (1, 2)

    sm = smooth_ε(shapes, mat_vals, minds, grid)
    n2_map = smooth_scalar(shapes, [kerr_n2(m, λ) for m in mats], minds, grid)

    return (;
        ε⁻¹=sliceinv_3x3(copy(selectdim(sm, 3, 1))),
        ∂ε_∂ω=copy(selectdim(sm, 3, 2)),
        ∂²ε_∂ω²=copy(selectdim(sm, 3, 3)),
        grid,
        n₂=n2_map,
    )
end
