# Forced grid-convergence mode solving for a Si₃N₄ ridge waveguide.
#
# A waveguide effective index computed on a finite finite-difference cell carries two
# discretization errors: (1) *truncation* error from the finite cell — the periodic
# boundaries clip the evanescent cladding fields — controlled by the distance from the
# waveguide center to the grid boundary (Δx/2, Δy/2), and (2) *discretization* error from
# the finite sampling, controlled by the spatial point density (points/μm²).
#
# `solve_k_converged(...; force_convergence=true, force_convergence_settings=...)` drives
# both errors down automatically: it re-runs the whole geometry → sub-pixel smoothing →
# eigensolve pipeline on a sequence of grids, multiplying the point density by
# `resolution_ramp` and the center→boundary distance by `boundary_ramp` each iteration,
# until the mode effective indices stop changing (to within `atol`/`rtol`) between
# successive iterations — or `max_iterations` is reached.
#
# Run with (from the repository root):
#   julia --project=. examples/forced_grid_convergence.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using Printf

# --- structure & materials -------------------------------------------------------
λ = 1.55                  # vacuum wavelength [μm]
ω = 1 / λ                 # frequency [μm⁻¹]
w_core, h_core = 1.60, 0.70

mats = [Si₃N₄, SiO₂]
f_ε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = f_ε([ω])       # per-material (ε, ∂ωε, ∂²ωε) columns at this ω
core = MaterialShape(Cuboid([0.0, 0.0], [w_core, h_core], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)            # background (mind 2) = SiO₂

solver = KrylovKitEigsolve()

# A deliberately coarse, tight starting grid: 3 × 2 μm cell (only ~0.9/0.75 μm of cladding
# beyond the waveguide edges) at ~64 points/μm². Both too small and too coarse to be
# converged — exactly what the forced-convergence loop is meant to fix.
grid0 = Grid(3.0, 2.0, 24, 16)

# --- forced convergence ----------------------------------------------------------
settings = ForceConvergenceSettings(;
    rtol=5e-4,            # relative effective-index tolerance between iterations
    atol=1e-6,            # absolute effective-index tolerance between iterations
    resolution_ramp=1.4,  # ×1.4 point density (points/μm²) per iteration
    boundary_ramp=1.2,    # ×1.2 waveguide-center → boundary distance per iteration
    max_iterations=6,
)

res = solve_k_converged(ω, shapes, mat_vals, minds, grid0, solver;
    nev=1, force_convergence=true, force_convergence_settings=settings)

@printf("\nForced grid convergence of the quasi-TE₀₀ effective index (λ = %.2f μm)\n", λ)
@printf("rtol = %.0e, atol = %.0e, resolution ramp = %.2f×, boundary ramp = %.2f×\n\n",
    settings.rtol, settings.atol, settings.resolution_ramp, settings.boundary_ramp)
@printf("%5s  %12s  %10s  %12s  %12s  %12s\n",
    "iter", "grid", "Nx·Ny", "density", "bnd.dist", "neff")
@printf("%5s  %12s  %10s  %12s  %12s  %12s\n",
    "", "(Nx×Ny)", "[pts]", "[pts/μm²]", "[μm]", "")
for (i, g) in enumerate(res.grid_history)
    density = length(g) / (g.Δx * g.Δy)
    bnd = g.Δx / 2                      # waveguide-center → boundary distance in x
    neff = res.neff_history[i][1]
    @printf("%5d  %5d×%-5d  %10d  %12.1f  %12.3f  %12.8f\n",
        i, g.Nx, g.Ny, length(g), density, bnd, neff)
end

Δfinal = abs(res.neff_history[end][1] - res.neff_history[end-1][1])
@printf("\n%s in %d iteration%s: neff = %.8f on a %d×%d grid (Δneff last step = %.2e)\n",
    res.converged ? "Converged" : "Did NOT converge", res.iterations,
    res.iterations == 1 ? "" : "s", res.neff[1], res.grid.Nx, res.grid.Ny, Δfinal)

# The returned result also carries the smoothed dielectric tensors on the final, most
# refined grid (res.ε⁻¹, res.∂ε_∂ω, res.∂²ε_∂ω²), ready for downstream post-processing
# such as the group index and group-velocity dispersion:
ng, gvd = ng_gvd(ω, res.kmags[1], res.evecs[1], res.ε⁻¹, res.∂ε_∂ω, res.∂²ε_∂ω², res.grid)
@printf("converged-grid group index ng = %.5f, GVD = %.4e\n", ng, gvd)

# The number of iterations and convergence status are recoverable from the output grid
# size alone: a converged run stops as soon as the indices settle, so a larger final grid
# means more refinement was needed. Disabling forced convergence (force_convergence=false)
# performs a single solve on the supplied grid — the drop-in non-converging behavior:
res_single = solve_k_converged(ω, shapes, mat_vals, minds, grid0, solver;
    nev=1, force_convergence=false)
@printf("\nSingle coarse solve (force_convergence=false): neff = %.8f on a %d×%d grid\n",
    res_single.neff[1], res_single.grid.Nx, res_single.grid.Ny)
@printf("→ refinement shifted neff by %.2e\n", abs(res.neff[1] - res_single.neff[1]))
