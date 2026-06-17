# Waveguide dispersion for second-harmonic generation in thin-film lithium niobate.
#
# Reproduces the forward (modeling) calculation behind the inverse-design results of
#
#   D. Gray, G. N. West, and R. J. Ram, "Inverse design for waveguide dispersion with a
#   differentiable mode solver," Optics Express 32(17), 30541 (2024).
#   https://doi.org/10.1364/OE.530479
#
# The paper optimizes an etched x-cut TFLN rib waveguide so that the quasi-TE00 modes at
# the fundamental frequency ω and second harmonic 2ω have *matched group indices*. To
# lowest order the quasi-phase-matched (QPM) SHG bandwidth is set by the group-velocity
# mismatch (Eq. 19),
#
#     Δω ∝ |n_g,2ω − n_g,ω|⁻¹,
#
# so driving the modal group-index difference to zero maximizes the conversion bandwidth.
#
# This script builds the rib waveguide from dispersive, anisotropic LiNbO₃ on an SiO₂
# substrate with air cladding, solves the quasi-TE00 mode at ω and 2ω, and computes for
# each the effective index, group index n_g (Eq. 12, from a single mode solution) and
# group-velocity dispersion GVD = ∂n_g/∂ω (single adjoint solve, Supplement 1). It then
# reports the SHG group-velocity mismatch, required poling period, and the resulting QPM
# bandwidth for a 1 cm device, and finally scans the core thickness to locate the
# group-index-matched ("phase-matching") geometry that the inverse-design optimizer of the
# paper converges to — here found by a one-dimensional sweep for illustration.
#
# Run (from the repository root):
#   julia --project=. examples/tfln_shg_dispersion.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------------------
# Materials: x-cut TFLN.  The bundled LiNbO₃ model has its extraordinary (c) axis along z;
# x-cut means the c-axis lies in the waveguide cross-section along the in-plane width (x),
# so the quasi-TE00 mode (dominant Eₓ) sees the extraordinary index nₑ and couples through
# the large d₃₃ nonlinearity.  Rotating the crystal frame by RotY(π/2) maps z → x.
# ---------------------------------------------------------------------------------------
const Ry = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]    # RotY(π/2): c-axis (z) → x
const LiNbO₃_xcut = rotate(LiNbO₃, Ry; name=:LiNbO₃_xcut)

# Material columns for smooth_ε: LiNbO₃ core (1), SiO₂ substrate (2), air cladding (3).
const mats = [LiNbO₃_xcut, SiO₂]
const f_ε, _ = _f_ε_mats(mats, (:ω,))
const air_col = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))   # ε = I, no dispersion

"Stack the per-material dielectric columns (ε, ∂ωε, ∂²ωε) evaluated at frequency `ω`."
mat_vals(ω) = hcat(f_ε([ω]), air_col)                      # 27 × 3  (LiNbO₃, SiO₂, air)

# ---------------------------------------------------------------------------------------
# Geometry: an etched rib.  A LiNbO₃ film of total thickness `t` sits on an SiO₂ substrate;
# the film is partially etched leaving an unetched slab of thickness t·(1−etch) with a
# ridge of top width `w` and height t·etch on top.  Parameterized exactly as in the paper
# (top width, thickness, etch fraction).  Shapes are foreground-ordered (ridge wins over
# slab wins over substrate); points above the film belong to the air background.
# ---------------------------------------------------------------------------------------
function tfln_shapes(w, t, etch)
    # keep both the ridge and the unetched slab at finite height: a zero-height Cuboid is
    # degenerate and stalls the eigensolver, so clamp away from a fully-etched (etch = 1)
    # or unetched (etch = 0) film.
    etch = clamp(etch, 1e-3, 1 - 1e-3)
    h_ridge = t * etch
    h_slab = t * (1 - etch)
    y_film_bot = 0.0                       # film bottom sits at y = 0 on the substrate
    # ridge: centered laterally, on top of the slab
    ridge = MaterialShape(
        Cuboid([0.0, y_film_bot + h_slab + h_ridge / 2], [w, h_ridge], [1.0 0.0; 0.0 1.0]), 1)
    # unetched slab: spans the full grid width (use a wide cuboid)
    slab = MaterialShape(
        Cuboid([0.0, y_film_bot + h_slab / 2], [100.0, h_slab], [1.0 0.0; 0.0 1.0]), 1)
    # SiO₂ substrate: a thick block filling everything below the film
    substrate = MaterialShape(
        Cuboid([0.0, y_film_bot - 50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2)
    return (ridge, slab, substrate)
end
const minds = (1, 1, 2, 3)                 # ridge→LiNbO₃, slab→LiNbO₃, substrate→SiO₂, bg→air

# ---------------------------------------------------------------------------------------
# Mode solving + dispersion for the quasi-TE00 mode at a given frequency.
# ---------------------------------------------------------------------------------------
const solver = KrylovKitEigsolve()

"""
    te00_dispersion(ω, w, t, etch, grid; nev)

Solve the waveguide modes at frequency `ω`, select the most TE-polarized guided mode
(largest fraction of E-field power along x — the quasi-TE00 fundamental for these
geometries), and return its effective index, group index `n_g` (Eq. 12) and
group-velocity dispersion `GVD = ∂n_g/∂ω` (single adjoint solve).
"""
function te00_dispersion(ω, w, t, etch, grid; nev=4)
    shapes = tfln_shapes(w, t, etch)
    sm = smooth_ε(shapes, mat_vals(ω), minds, grid)
    ε = copy(selectdim(sm, 3, 1))
    ε⁻¹ = sliceinv_3x3(copy(ε))
    ∂ωε = copy(selectdim(sm, 3, 2))
    ∂²ωε = copy(selectdim(sm, 3, 3))

    kmags, evecs = solve_k(ω, ε⁻¹, grid, solver; nev=nev, k_tol=1e-9)

    # pick the most TE-polarized guided mode (largest fraction of E-power along x)
    fracs = [E_relpower_xyz(ε, E⃗(kmags[i], copy(evecs[i]), ε⁻¹, ∂ωε, grid;
        canonicalize=true, normalized=true))[1] for i in eachindex(evecs)]
    i = argmax(fracs)
    k, ev = kmags[i], evecs[i]

    ng, gvd, E = ng_gvd_E(ω, k, ev, ε⁻¹, ∂ωε, ∂²ωε, grid)
    neff = k / ω
    return (; neff, ng, gvd, k, te_frac=fracs[i], E)
end

# ---------------------------------------------------------------------------------------
# Forward calculation at a representative design point near the paper's 1.3–1.4 μm range.
# ---------------------------------------------------------------------------------------
λ_f = 1.35                     # fundamental vacuum wavelength [μm]
ω_f = 1 / λ_f                  # fundamental frequency [μm⁻¹]
ω_sh = 2ω_f                    # second-harmonic frequency
w0, t0, etch0 = 1.0, 0.7, 0.5  # top width [μm], film thickness [μm], etch fraction

grid = Grid(8.0, 4.0, 128, 96)  # 8 × 4 μm cell

println("="^78)
@printf("Thin-film LiNbO₃ rib waveguide, x-cut (c-axis ∥ x)\n")
@printf("  cell %.0f×%.0f μm  grid %d×%d  |  w=%.3f μm  t=%.3f μm  etch=%.2f\n",
    8.0, 4.0, size(grid)..., w0, t0, etch0)
@printf("  fundamental λ=%.3f μm (ω=%.4f μm⁻¹),  second harmonic λ=%.4f μm (2ω=%.4f μm⁻¹)\n",
    λ_f, ω_f, λ_f / 2, ω_sh)
println("="^78)

res_f = te00_dispersion(ω_f, w0, t0, etch0, grid)
res_sh = te00_dispersion(ω_sh, w0, t0, etch0, grid)

@printf("\n%-22s %14s %14s\n", "", "fundamental ω", "2nd-harm 2ω")
@printf("%-22s %14.5f %14.5f\n", "n_eff", res_f.neff, res_sh.neff)
@printf("%-22s %14.5f %14.5f\n", "n_g  (group index)", res_f.ng, res_sh.ng)
@printf("%-22s %14.4e %14.4e\n", "GVD  ∂n_g/∂ω", res_f.gvd, res_sh.gvd)
@printf("%-22s %14.2f%% %13.2f%%\n", "TE fraction (Eₓ)", 100res_f.te_frac, 100res_sh.te_frac)

# SHG figures of merit -------------------------------------------------------------------
# In OptiMode's plane-wave convention ω = 1/λ and k = n_eff/λ carry no factor of 2π (see
# the `Grid` `g⃗` docstring); the physical propagation constant is β = 2π·k.
Δng = res_sh.ng - res_f.ng                  # modal group-index mismatch (paper Eq. 19/20)
Δk = res_sh.k - 2res_f.k                     # k(2ω) − 2k(ω) [μm⁻¹];  phase mismatch Δβ = 2π·Δk
Λ_poling = 1 / abs(Δk)                       # QPM poling period 2π/|Δβ| = 1/|Δk| [μm]

# QPM SHG efficiency ∝ sinc²(Δβᵗᵒᵗ·L/2), with Δβᵗᵒᵗ = 2π(k₂ω − 2k_ω) − 2π/Λ cancelled at
# the design frequency.  As the fundamental frequency detunes, the residual mismatch grows
# at the group-velocity-mismatch rate d(Δβᵗᵒᵗ)/dω = 2π·2(n_g,2ω − n_g,ω) = 4π·Δn_g.  sinc²
# falls to half maximum at Δβᵗᵒᵗ·L/2 = ±1.3916, so the FWHM in fundamental frequency
# (ω = 1/λ) for a device of length L is Δω = 2·(2·1.3916)/(L·4π·|Δn_g|).
L = 1e4                                      # 1 cm in μm
Δω_FWHM = (2 * 2 * 1.3916) / (L * 4π * abs(Δng))  # [μm⁻¹]
c = 299.792458                               # speed of light [μm·THz]
Δν_FWHM = c * Δω_FWHM                         # [THz]  (ν = c/λ = c·ω, no 2π)
Δλ_FWHM = λ_f^2 * Δω_FWHM                      # [μm]   (|Δλ| = λ²·|Δ(1/λ)|)

@printf("\nSHG metrics at this geometry:\n")
@printf("  group-index mismatch  Δn_g = n_g,2ω − n_g,ω = %+.4e\n", Δng)
@printf("  phase mismatch        Δβ   = 2π(k₂ω − 2k_ω)  = %+.3f μm⁻¹\n", 2π * Δk)
@printf("  QPM poling period     Λ    = 2π/|Δβ|         = %.3f μm\n", Λ_poling)
@printf("  QPM FWHM bandwidth (L = 1 cm): %.2f THz  (≈ %.1f nm)\n",
    Δν_FWHM, 1e3 * Δλ_FWHM)

# ---------------------------------------------------------------------------------------
# Etch-fraction scan: |Δn_g| is the optimization objective (Eq. 20).  Sweeping one design
# knob shows it is a smooth function of geometry that can be driven toward zero — the
# group-velocity-matched waveguide with maximal SHG bandwidth.  The paper minimizes |Δn_g|
# by gradient descent co-varying width, thickness and etch simultaneously; here a single
# parameter is scanned to expose the trend the optimizer follows.
# ---------------------------------------------------------------------------------------
println("\n" * "="^78)
println("Etch-fraction scan at w = $(w0) μm, t = $(t0) μm: |Δn_g| is the SHG objective")
println("="^78)
@printf("%10s %12s %12s %14s %12s\n", "etch", "n_g(ω)", "n_g(2ω)", "Δn_g", "|Δn_g|²")
etches = 0.3:0.1:0.9
Δngs = Float64[]
for e in etches
    rf = te00_dispersion(ω_f, w0, t0, e, grid)
    rs = te00_dispersion(ω_sh, w0, t0, e, grid)
    d = rs.ng - rf.ng
    push!(Δngs, d)
    @printf("%10.2f %12.5f %12.5f %+14.4e %12.3e\n", e, rf.ng, rs.ng, d, d^2)
end

ibest = argmin(abs.(Δngs))
sgn = sign.(Δngs)
icross = findfirst(i -> sgn[i] != sgn[i+1], 1:length(sgn)-1)
@printf("\nObjective |Δn_g| is minimized in this 1-D scan at etch = %.2f (|Δn_g| = %.3e).\n",
    etches[ibest], abs(Δngs[ibest]))
if icross !== nothing
    e_lo, e_hi = etches[icross], etches[icross+1]
    e_star = e_lo - Δngs[icross] * (e_hi - e_lo) / (Δngs[icross+1] - Δngs[icross])
    @printf("Group-velocity matching (Δn_g = 0) is bracketed in etch ∈ [%.2f, %.2f], ≈ %.3f.\n",
        e_lo, e_hi, e_star)
end
println("""
Driving |Δn_g| → 0 group-velocity matches the quasi-TE00 modes at ω and 2ω, maximizing
the SHG quasi-phase-matching bandwidth (Eq. 19). In Gray et al. (2024) the differentiable
mode solver supplies ∂|Δn_g|²/∂{w,t,etch} so a gradient optimizer reaches |Δn_g| ≲ 1e-3 in
~8 steps, co-varying all three parameters — ~100× fewer mode solves than exhaustive search.
This script performs the forward calculation those gradients are back-propagated through.""")
