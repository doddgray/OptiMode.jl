# ModeSweeps problem-setup for the dispersion-engineered x-cut PPLN ridge of
#   Jankowski et al., Optica 7, 40 (2020) — used to reproduce Fig. 1(b–e) (poling period,
#   normalized efficiency, group-velocity mismatch Δk′, and GVD) as functions of waveguide
#   geometry by deploying a (top width × etch depth × {ω, 2ω}) sweep on a cluster.
#
# A setup script maps one parameter set `p::NamedTuple` to the smoothed inverse dielectric
# tensor and its frequency derivatives on a spatial grid. It is copied into the batch
# directory and `include`d by every worker in a fresh module (with the OptiMode component
# packages already loaded), so it must be self-contained: it `using`s what it needs and
# defines `make_problem`.
#
# Swept parameters (see the companion deploy script):
#   p.ω      optical frequency (μm⁻¹)        — ω_FF = 1/2.05 and ω_SH = 2/2.05
#   p.w_top  ridge top width (μm)            — 1.80 … 2.00
#   p.etch   etch depth (μm)                 — 0.31 … 0.39   (700-nm film)

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using LinearAlgebra

# x-cut MgO:LiNbO₃ (extraordinary c-axis rotated in-plane along x) + SiO₂; air as 3rd column
const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
const _LN_XCUT = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
const _AIR = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))
const _FEPS, _ = _f_ε_mats([_LN_XCUT, SiO₂], (:ω,))
const _FILM = 0.700                          # starting x-cut TFLN film thickness (μm)

function make_problem(p)
    w_top = haskey(p, :w_top) ? p.w_top : 1.85
    etch = haskey(p, :etch) ? p.etch : 0.34
    slab = _FILM - etch
    # large transverse cell: the weakly-guided 2-µm fundamental needs room to converge.
    # The cell/resolution can be overridden per parameter set (e.g. for a cheaper local
    # demo) via p.Lx/p.Ly/p.Nx/p.Ny; defaults target a converged cluster run.
    Lx = haskey(p, :Lx) ? p.Lx : 12.0
    Ly = haskey(p, :Ly) ? p.Ly : 7.0
    Nx = haskey(p, :Nx) ? p.Nx : 320
    Ny = haskey(p, :Ny) ? p.Ny : 176
    grid = Grid(Lx, Ly, Nx, Ny)

    mat_vals = hcat(_FEPS([p.ω]), _AIR)      # 27×3: LiNbO₃, SiO₂, air
    shapes = (MaterialShape(Cuboid([0.0, slab + etch / 2], [w_top, etch], [1.0 0.0; 0.0 1.0]), 1),
        MaterialShape(Cuboid([0.0, slab / 2], [200.0, slab], [1.0 0.0; 0.0 1.0]), 1),
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2))
    minds = (1, 1, 2, 3)

    sm = smooth_ε(shapes, mat_vals, minds, grid)   # slices: ε, ∂ωε, ∂²ωε
    return (;
        ε⁻¹=sliceinv_3x3(copy(selectdim(sm, 3, 1))),
        ∂ε_∂ω=copy(selectdim(sm, 3, 2)),
        ∂²ε_∂ω²=copy(selectdim(sm, 3, 3)),
        grid,
    )
end
