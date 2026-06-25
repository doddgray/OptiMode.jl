# Eigenmode-expansion (EME) simulation of an adiabatic directional coupler,
# driven end-to-end from a GDSFactory-generated GDS layout.
#
# Pipeline (MEOW/SAX-style, on OptiMode's differentiable mode solver):
#
#   GDSFactory layout ─▶ .gds ─▶ read_gds ─▶ Structure (+ layer stack)
#       ─▶ EME cells ─▶ per-cell modes (solve_k) ─▶ interface/propagation
#       S-matrices ─▶ cascade ─▶ device S-matrix ─▶ bar/cross coupling.
#
# Generate the GDS first with GDSFactory:
#     python examples/gen_adiabatic_coupler_gds.py examples/adiabatic_coupler.gds
# (if the file is absent this script writes an equivalent layout with the
#  built-in GDS writer, so it always runs).

using OptiMode                       # re-exports EigenmodeExpansion
using OptiMode.EigenmodeExpansion
using Printf

const HERE = @__DIR__
const GDS = joinpath(HERE, "adiabatic_coupler.gds")

# ── 1. obtain a GDS layout ───────────────────────────────────────────────────
if !isfile(GDS)
    pyscript = joinpath(HERE, "gen_adiabatic_coupler_gds.py")
    ok = try
        run(`python $pyscript $GDS`); true
    catch
        false
    end
    if !ok
        @info "GDSFactory unavailable; writing an equivalent layout with the built-in GDS writer."
        L, w, g0, g1 = 20.0, 0.5, 1.2, 0.3
        yA(s) = +(g0 + (g1 - g0) * s / L) / 2 + w / 2
        yB(s) = -((g0 + (g1 - g0) * s / L) / 2 + w / 2)
        vA = [0.0 L L 0.0; yA(0)+w/2 yA(L)+w/2 yA(L)-w/2 yA(0)-w/2]
        vB = [0.0 L L 0.0; yB(0)+w/2 yB(L)+w/2 yB(L)-w/2 yB(0)-w/2]
        write_gds(GDS, [GDSPolygon(1, 0, Matrix(vA)), GDSPolygon(1, 0, Matrix(vB))]; name="COUPLER")
    end
end

# ── 2. import GDS and build the 3D structure + EME cells ─────────────────────
layout = read_gds(GDS)
@printf "imported %d polygons on layers %s\n" length(layout.polygons) string(unique(p.layer for p in layout.polygons))

stack = LayerStack(
    layers=[Layer(gds_layer=1, zmin=0.0, zmax=0.3, material=1, patterned=true, name="SiN core")],
    materials=Any[Si₃N₄, SiO₂],     # column 1 = core, column 2 = cladding
    background=2,
    prop_axis=:x,                   # light propagates along the GDS x-axis
)
structure = Structure(layout, stack; transverse_pad=2.0, vertical_pad=1.0)

# ── 3. run EME at 1550 nm ────────────────────────────────────────────────────
λ = 1.55
ω = 1 / λ
res = eme_smatrix(structure, ω; nev=2, Nx=128, Ny=64, num_cells=30, k_tol=1e-8)

println("\nInput-facet supermode effective indices:")
for (i, m) in enumerate(res.modes[1])
    @printf "  mode %d:  n_eff = %.4f\n" i real(m.neff)
end

T = transmission(res.S)
@printf "\nDevice transmission |S21|² (rows=output mode, cols=input mode):\n"
for i in 1:size(T, 1)
    @printf "  [%s]\n" join((@sprintf("%.3f", abs2(T[i, j])) for j in 1:size(T, 2)), "  ")
end
@printf "\nbar coupling  (mode 1 → 1): %.3f\n" power_coupling(res; in_mode=1, out_mode=1)
@printf "cross coupling (mode 1 → 2): %.3f\n" power_coupling(res; in_mode=1, out_mode=2)

# ── 4. wavelength sweep (cross-coupling spectrum) ────────────────────────────
println("\nWavelength sweep:")
for λi in 1.50:0.025:1.60
    r = eme_smatrix(structure, 1 / λi; nev=2, Nx=128, Ny=64, num_cells=30, k_tol=1e-8)
    @printf "  λ = %.3f µm   cross = %.3f   bar = %.3f\n" λi power_coupling(r; in_mode=1, out_mode=2) power_coupling(r; in_mode=1, out_mode=1)
end

# ── 5. remote deployment (SLURM) & parameter sweeps ──────────────────────────
# The per-cell mode solves are the expensive part and map onto ModeSweeps tasks.
# With ModeSweeps loaded, deploy them as a SLURM array job (one task per cell),
# then re-assemble the device S-matrix locally:
#
#     using ModeSweeps
#     grid  = simulation_grid(structure, 128, 64)
#     cells = build_cells(structure; num_cells=30)
#     # eme_coupler_setup.jl defines  make_problem(p) = cell_problem(CELLS[p.cell], MATS, p.ω, GRID)
#     batch = deploy_eme("examples/eme_coupler_setup.jl", length(cells);
#                        ω = 1/1.55, nev = 2,
#                        slurm = SlurmConfig(ssh="me@cluster", remote_dir="/scratch/me/eme"))
#     res   = gather_eme(batch, cells, structure.stack.materials, 1/1.55, grid; nev=2)
#     power_coupling(res; in_mode=1, out_mode=2)
#
# AD: `power_coupling`/`transmission` are differentiable, so a frequency- or
# geometry-gradient of the coupling follows from Zygote (reverse) over `eme`:
#     using Zygote
#     dcross_dω = Zygote.gradient(ω -> power_coupling(
#         eme(cells, structure.stack.materials, ω, grid; nev=2); in_mode=1, out_mode=2), 1/1.55)[1]
