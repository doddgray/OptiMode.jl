# ModeSweeps problem-setup script: a Si₃N₄ rectangular-ridge waveguide in SiO₂.
#
# A setup script maps one parameter set `p::NamedTuple` to the smoothed inverse
# dielectric tensor and its frequency derivatives on a spatial grid. It is copied into
# the batch directory and `include`d by every worker (inside a fresh module with the
# OptiMode component packages already loaded), so it must `using` whatever it needs and
# define `make_problem`.
#
# By convention `p.ω` is the optical frequency (μm⁻¹); here we also sweep the core width
# `w_top` and (optionally) height `h_core`. Returning `∂²ε_∂ω²` enables group-velocity
# dispersion (GVD) in the summary. Used by `remote_mode_solve.jl` and
# `remote_adjoint_optimization.jl`.

using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid

function make_problem(p)
    w_top = haskey(p, :w_top) ? p.w_top : 1.6
    h_core = haskey(p, :h_core) ? p.h_core : 0.7
    grid = Grid(6.0, 4.0, 128, 96)

    f_ε, _ = _f_ε_mats([Si₃N₄, SiO₂], (:ω,))
    mat_vals = f_ε([p.ω])
    core = MaterialShape(Cuboid([0.0, 0.0], [w_top, h_core], [1.0 0.0; 0.0 1.0]), 1)
    shapes, minds = (core,), (1, 2)

    sm = smooth_ε(shapes, mat_vals, minds, grid)   # slices: ε, ∂ωε, ∂²ωε
    return (;
        ε⁻¹=sliceinv_3x3(copy(selectdim(sm, 3, 1))),
        ∂ε_∂ω=copy(selectdim(sm, 3, 2)),
        ∂²ε_∂ω²=copy(selectdim(sm, 3, 3)),
        grid,
    )
end
