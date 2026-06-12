# Kerr (n₂) power-dependent mode solves for a Si₃N₄-core rectangular waveguide.
#
# A Si₃N₄ core (w × h = 1.60 × 0.80 μm) embedded in SiO₂ cladding at λ = 1.55 μm.
# The materials carry standard Kerr coefficients from the MaterialDispersion library:
#   Si₃N₄: n₂ = 2.4×10⁻⁷ μm²/W (= 2.4×10⁻¹⁹ m²/W; Ikeda 2008 / Krückel 2017)
#   SiO₂:  n₂ = 2.6×10⁻⁸ μm²/W (= 2.6×10⁻²⁰ m²/W; Agrawal, Nonlinear Fiber Optics)
#
# For each optical power P the fundamental mode is corrected to first order: the modal
# intensity profile I(x,y) (normalized to carry power P) induces Δn = n₂(x,y)·I(x,y),
# and the mode is re-solved with the perturbed dielectric tensor. The resulting
# Δneff(P) is compared with the textbook self-phase-modulation estimate
# Δneff ≈ n₂(core)·P/Aeff.
#
# Run with (from the repository root):
#   julia --project=. examples/kerr_si3n4_waveguide.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using Printf

# --- structure & materials -------------------------------------------------------
λ = 1.55                 # vacuum wavelength [μm]
ω = 1 / λ                # frequency [μm⁻¹]
w_core, h_core = 1.60, 0.80
grid = Grid(4.0, 3.0, 96, 72)

mats = [Si₃N₄, SiO₂]
f_ε, _ = _f_ε_mats(mats, (:ω,))
mat_vals = f_ε([ω])
core = MaterialShape(Cuboid([0.0, 0.0], [w_core, h_core], [1.0 0.0; 0.0 1.0]), 1)
shapes, minds = (core,), (1, 2)          # background (mind 2) = SiO₂

sm = smooth_ε(shapes, mat_vals, minds, grid)
ε⁻¹ = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
∂ε_∂ω = copy(selectdim(sm, 3, 2))

# Kerr coefficient map n₂(x,y) [μm²/W] from the per-material library values
n2_vals = [kerr_n2(m, λ) for m in mats]
n2_map = smooth_scalar(shapes, n2_vals, minds, grid)
@printf("n₂(Si₃N₄) = %.2e μm²/W,  n₂(SiO₂) = %.2e μm²/W\n", n2_vals...)

# --- power sweep -----------------------------------------------------------------
solver = KrylovKitEigsolve()
powers = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]   # [W]

# linear reference quantities for the SPM estimate
res0 = solve_k_kerr(ω, 0.0, ε⁻¹, ∂ε_∂ω, n2_map, grid, solver; nev=1, k_tol=1e-10)
k0 = res0.kmags_lin[1]
ng0 = group_index(k0, res0.evecs_lin[1], ω, ε⁻¹, ∂ε_∂ω, grid)
# standard nonlinear effective area Aeff = (∫I dA)² / ∫I² dA from the intensity profile
I1 = mode_intensity(k0, res0.evecs_lin[1], ε⁻¹, grid, 1.0)
Aeff = 1.0^2 / (sum(abs2, I1) * DielectricSmoothing.δV(grid))
γ = 2π * n2_vals[1] / (λ * Aeff)            # nonlinear parameter [W⁻¹μm⁻¹]
@printf("linear: neff = %.5f, ng = %.4f, Aeff = %.3f μm², γ = %.3e W⁻¹μm⁻¹ (= %.2f W⁻¹m⁻¹)\n\n",
    k0 / ω, ng0, Aeff, γ, γ * 1e6)

@printf("%8s  %12s  %14s  %14s  %12s\n", "P [W]", "neff(P)", "Δneff (solve)", "n₂P/Aeff", "max Δn")
for P in powers
    res = solve_k_kerr(ω, P, ε⁻¹, ∂ε_∂ω, n2_map, grid, solver; nev=1, k_tol=1e-10)
    Δneff = (res.kmags[1] - res.kmags_lin[1]) / ω
    @printf("%8.2f  %12.7f  %14.4e  %14.4e  %12.4e\n",
        P, res.kmags[1] / ω, Δneff, n2_vals[1] * P / Aeff, res.dn_max[1])
end

println("""

Power sweeps over many (P, ω, geometry, …) combinations can equally be deployed as
asynchronous (SLURM) batches — see examples/kerr_power_sweep_setup.jl.""")
