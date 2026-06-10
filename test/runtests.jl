# Umbrella smoke tests for OptiMode. The substantive unit, gradient-correctness, and
# benchmark suites live with the component packages:
#   lib/MaterialDispersion/test, lib/DielectricSmoothing/test,
#   lib/MaxwellEigenmodes/test,  lib/ModeAnalysis/test
using Test
using OptiMode

@testset "OptiMode umbrella" begin
    # re-exports from all four component packages resolve
    @test OptiMode.SiO₂ isa OptiMode.Material            # MaterialDispersion
    @test OptiMode.Grid(6.0, 4.0, 16, 16) isa OptiMode.Grid  # DielectricSmoothing
    @test OptiMode.KrylovKitEigsolve() isa OptiMode.AbstractEigensolver  # MaxwellEigenmodes
    @test OptiMode.group_index isa Function              # ModeAnalysis

    # minimal end-to-end pipeline: dispersion → smoothing → eigensolve → analysis
    mats = [Si₃N₄, SiO₂]
    f_ε, _ = _f_ε_mats(mats, (:ω,))
    ω = 1 / 1.55
    mat_vals = hcat(f_ε([ω]), vcat(vec([1.0 0 0; 0 1.0 0; 0 0 1.0]), zeros(18)))
    grid = Grid(4.0, 3.0, 16, 12)
    core = MaterialShape(OptiMode.DielectricSmoothing.GeometryPrimitives.Cuboid([0.0, 0.0], [1.6, 0.7], [1.0 0.0; 0.0 1.0]), 1)
    shapes = (core,)
    minds = (1, 2)
    sm = smooth_ε(shapes, mat_vals, minds, grid)
    @test size(sm) == (3, 3, 3, 16, 12)
    epsi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
    kmags, evecs = solve_k(ω, epsi, grid, KrylovKitEigsolve(); nev=1)
    @test sqrt(1.444^2) < kmags[1] / ω < 2.1
    deps_dom = copy(selectdim(sm, 3, 2))
    ng = group_index(kmags[1], evecs[1], ω, epsi, deps_dom, grid)
    @test kmags[1] / ω < ng < 3.0
end
