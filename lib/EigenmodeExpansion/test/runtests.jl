using Test
using LinearAlgebra
using EigenmodeExpansion
using MaterialDispersion: Si₃N₄, SiO₂, _f_ε_mats
using DielectricSmoothing: Grid, smooth_ε, δV
using MaxwellEigenmodes: KrylovKitEigsolve, sliceinv_3x3, solve_k
using ForwardDiff
using FiniteDifferences
import Zygote

# ── adiabatic coupler layout (propagation along x, transverse y) ─────────────
# Two Si₃N₄ strips whose gap narrows linearly along the propagation axis, so the
# modal coupling grows along the device — a textbook adiabatic directional
# coupler. Each strip is a parallelogram polygon (constant width, linearly
# shifting centre line) on GDS layer 1.
function coupler_polygons(; L=10.0, w=0.5, gap0=1.2, gap1=0.3)
    yA(s) = +(gap0 + (gap1 - gap0) * s / L) / 2 + w / 2     # strip-A centre
    yB(s) = -((gap0 + (gap1 - gap0) * s / L) / 2 + w / 2)   # strip-B centre
    # vertices CCW: (x, y); strip width w about the (linear) centre line
    vA = [0.0 L L 0.0; yA(0)+w/2 yA(L)+w/2 yA(L)-w/2 yA(0)-w/2]
    vB = [0.0 L L 0.0; yB(0)+w/2 yB(L)+w/2 yB(L)-w/2 yB(0)-w/2]
    return [GDSPolygon(1, 0, Matrix(vA)), GDSPolygon(1, 0, Matrix(vB))]
end

# Si₃N₄ cores (material column 1), SiO₂ cladding background (column 2), 0.3 µm thick.
const COUPLER_STACK = LayerStack(
    layers=[Layer(gds_layer=1, zmin=0.0, zmax=0.3, material=1, patterned=true, name="SiN")],
    materials=Any[Si₃N₄, SiO₂],
    background=2,
    prop_axis=:x,
)

@testset "EigenmodeExpansion" begin

    @testset "GDS round-trip" begin
        polys = coupler_polygons()
        path = tempname() * ".gds"
        write_gds(path, polys; name="COUPLER")
        layout = read_gds(path)
        @test length(layout.polygons) == 2
        @test all(p -> p.layer == 1, layout.polygons)
        # vertices preserved to database-unit resolution (1 nm default)
        @test isapprox(layout.polygons[1].verts, polys[1].verts; atol=2e-3)
        @test layout.unit_um ≈ 1e-3
    end

    @testset "structure & cells" begin
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        @test st.s_range[1] ≈ 0.0 && st.s_range[2] ≈ 10.0
        cells = build_cells(st; num_cells=4)
        @test length(cells) == 4
        @test sum(c.length for c in cells) ≈ 10.0
        # near the input the gap is wide → two separated cores in the cross-section
        cs0 = cross_section_at(st, 0.5)
        @test length(cs0.shapes) == 2
        @test cs0.minds == [1, 1, 2]            # both cores → col 1, background → col 2
        # smoothing produces a valid dielectric field
        grid = simulation_grid(st, 48, 32)
        ε⁻¹, ∂ε_∂ω = cell_dielectric(cs0, COUPLER_STACK.materials, 1 / 1.55, grid)
        @test size(ε⁻¹) == (3, 3, 48, 32)
        @test all(isfinite, ε⁻¹)
    end

    @testset "EME device S-matrix" begin
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        ω = 1 / 1.55
        res = eme_smatrix(st, ω; nev=2, Nx=64, Ny=32, num_cells=4, k_tol=1e-7)
        @test res.S.nl == 2 && res.S.nr == 2
        T = transmission(res.S)
        @test size(T) == (2, 2)
        # passive: no transmission channel has gain
        @test all(abs.(T) .<= 1.0 + 1e-6)
        # the supermodes carry power across the device
        @test power_coupling(res; in_mode=1, out_mode=1) > 1e-3
        # modes are power-normalized (self-overlap ≈ 1)
        m = res.modes[1][1]
        @test isapprox(inner_product(m, m), 1.0; atol=1e-6)
    end

    @testset "interface & propagation building blocks" begin
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        ω = 1 / 1.55
        grid = simulation_grid(st, 64, 32)
        cells = build_cells(st; num_cells=2)
        modes1 = solve_cell_modes(cells[1], st.stack.materials, ω, grid; nev=2, k_tol=1e-7)
        modes2 = solve_cell_modes(cells[2], st.stack.materials, ω, grid; nev=2, k_tol=1e-7)
        # interface of identical bases ≈ identity transmission, no reflection
        Sid = interface_smatrix(modes1, modes1)
        @test isapprox(transmission(Sid), I(2); atol=2e-2)
        @test maximum(abs, reflection(Sid)) < 2e-2
        # a real interface between distinct cross-sections is passive
        Sif = interface_smatrix(modes1, modes2)
        @test maximum(abs, Sif.S) <= 1.0 + 1e-6
        # propagation matrix is unitary-diagonal (lossless)
        Sp = propagation_smatrix(modes1, 1.0)
        @test all(abs.(diag(transmission(Sp))) .≈ 1.0)
    end

    @testset "AD — forward mode (geometry/material → dielectric)" begin
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        grid = simulation_grid(st, 40, 28)
        cs = cross_section_at(st, 2.0)
        # scalar functional of the smoothed dielectric, differentiated in ω
        f(ω) = sum(abs2, cell_dielectric(cs, st.stack.materials, ω, grid)[1])
        ω0 = 1 / 1.55
        g_fd = central_fdm(5, 1)(f, ω0)
        g_ad = ForwardDiff.derivative(f, ω0)
        @test isapprox(g_ad, g_fd; rtol=1e-4)
    end

    @testset "AD — reverse mode (eigensolve adjoint)" begin
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        grid = simulation_grid(st, 48, 32)
        cs = cross_section_at(st, 3.0)
        ε⁻¹, _ = cell_dielectric(cs, st.stack.materials, 1 / 1.55, grid)
        # n_eff via the solve_k adjoint rrule (the documented reverse path)
        neff(ω) = solve_k(ω, copy(ε⁻¹), grid, KrylovKitEigsolve(); nev=1, k_tol=1e-9)[1][1] / ω
        ω0 = 1 / 1.55
        g_ad = Zygote.gradient(neff, ω0)[1]
        g_fd = central_fdm(5, 1)(neff, ω0)
        @test isapprox(g_ad, g_fd; rtol=5e-3)
    end

    @testset "AD — reverse mode (end-to-end EME, ε⁻¹ sensitivity)" begin
        # Differentiate the device cross-coupling through the FULL EME pipeline —
        # per-cell `solve_k` adjoints + field reconstruction + overlap/interface/
        # propagation S-matrices + Redheffer cascade — w.r.t. a cell's inverse
        # dielectric field. Directional finite-difference check of the full chain.
        path = tempname() * ".gds"
        write_gds(path, coupler_polygons())
        st = Structure(read_gds(path), COUPLER_STACK; transverse_pad=1.5, vertical_pad=1.0)
        grid = simulation_grid(st, 48, 32)
        ω = 1 / 1.55
        cells = build_cells(st; num_cells=2)
        mats = st.stack.materials
        ds = [cell_dielectric(c.cross_section, mats, ω, grid) for c in cells]
        ε⁻¹s = [d[1] for d in ds]
        ∂ε_∂ωs = [d[2] for d in ds]
        Ls = [c.length for c in cells]
        coupling_cross(eis) = power_coupling(
            eme(eis, ∂ε_∂ωs, Ls, ω, grid, KrylovKitEigsolve(); nev=2, k_tol=1e-9);
            in_mode=1, out_mode=2)
        g_ad = Zygote.gradient(coupling_cross, ε⁻¹s)[1]
        @test g_ad !== nothing && all(x -> all(isfinite, x), g_ad)
        # directional FD check along a random perturbation of the first cell's ε⁻¹
        dir = randn(size(ε⁻¹s[1])) .* 1e-3
        ϕ(t) = coupling_cross([ε⁻¹s[1] .+ t .* dir, ε⁻¹s[2]])
        gdir_fd = central_fdm(5, 1)(ϕ, 0.0)
        gdir_ad = sum(g_ad[1] .* dir)
        @test isapprox(gdir_ad, gdir_fd; rtol=5e-2, atol=1e-6)
    end
end
