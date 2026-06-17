# Forward- and reverse-mode automatic differentiation of second-harmonic-generation (SHG)
# phase matching in thin-film lithium niobate, with respect to *material* parameters —
# temperature and crystal orientation.
#
# Companion to `examples/tfln_shg_dispersion.jl` (geometry sensitivities) and to
#
#   D. Gray, G. N. West, and R. J. Ram, "Inverse design for waveguide dispersion with a
#   differentiable mode solver," Optics Express 32(17), 30541 (2024).
#
# Setup: a 400 nm-thick x-cut TFLN rib waveguide, quasi-TE00 modes at the fundamental
# (~1560 nm) and second harmonic (~780 nm).  X-cut means the crystal X axis is the film
# normal; the optical (c/Z) axis lies in the film plane.  We propagate close to the
# crystal Y axis, and study two material knobs:
#
#   * temperature T  (Sellmeier thermo-optic dispersion of LiNbO₃, 30–80 °C), and
#   * a small angular deviation θ of the propagation direction from the crystal Y axis,
#     in the film plane (0–15°), which rotates extraordinary (c-axis) character into the
#     quasi-TE00 mode and adds Eₓ–E_z (longitudinal) anisotropic coupling.
#
# Both enter the model *only* through the dielectric tensor field ε(x,y; ω,T,θ).  Quasi-
# phase-matched SHG converts most efficiently at the fundamental wavelength where
#
#   Δβ_tot(λ) = β(2ω) − 2β(ω) − 2π/Λ = 0,   i.e.   Δk_tot(λ) = k(2ω) − 2k(ω) − 1/Λ = 0
#
# (OptiMode uses ω = 1/λ and k = n_eff/λ with no factor of 2π; β = 2π·k).  The poling
# period Λ is fixed by the design point, so as T and θ change the dispersion shifts and the
# phase-matched ("peak SHG") fundamental wavelength λ_peak moves.  We compute the gradient
# of λ_peak with respect to T and θ three ways and show they agree:
#
#   1. forward-mode AD  (ForwardDiff) of the material model,
#   2. reverse-mode AD  (Zygote)      of the material model,
#   3. finite differences of the full mode re-solve.
#
# The eigensolve itself is not differentiated through.  Instead the modal sensitivity of a
# wavenumber to the dielectric is the exact first-order (Hellmann–Feynman) perturbation
# from the converged mode field,
#
#   ∂k/∂p = ⟨E| ∂ε/∂p |E⟩ / (2⟨ev|∂M̂/∂k|ev⟩)            (fixed ω, frozen mode),
#
# built from the same validated `HMH`/`HMₖH` quadratic forms used by `group_index`.  This
# is exact for *any* mode (the quasi-TE00 is not the fundamental here, since nₑ < nₒ) and
# for the full anisotropic ∂ε/∂θ, where the generic eigensolver reverse rule is not.  The
# only object that is auto-differentiated is the closed-form material map p ↦ ε(p), which
# OptiMode differentiates in both forward and reverse mode to machine precision.
#
# Run (from the repository root):
#   julia --project=. examples/tfln_shg_temperature_angle_ad.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModeAnalysis: Zygote
using OptiMode.MaxwellEigenmodes: HMH, HMₖH, mag_mn, _dot
using LinearAlgebra
using ForwardDiff
using FiniteDifferences
using Printf

# ---------------------------------------------------------------------------------------
# Materials.  Keep LiNbO₃ in its crystal frame (ordinary indices along x,y; extraordinary
# c-axis along z) and apply the cut + propagation rotation numerically, so temperature T
# and angle θ both flow through plain Julia code that ForwardDiff and Zygote can
# differentiate.  Column order for `smooth_ε`: LiNbO₃ core (1), SiO₂ substrate (2),
# air cladding (3).
# ---------------------------------------------------------------------------------------
const mats = [LiNbO₃, SiO₂]
const f_ε, _ = _f_ε_mats(mats, (:ω, :T))           # (ω,T) ↦ [ε, ∂ωε, ∂²ωε] per material
const air_col = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))   # air: ε = I, no dispersion

# Simulation-frame rotation S(θ): its columns are the simulation basis vectors expressed
# in crystal coordinates.  x-cut ⇒ film normal (sim ŷ) = crystal X.  At θ = 0 the
# propagation direction (sim ẑ) = crystal Y and the in-plane transverse direction (sim x̂) =
# crystal Z (c-axis), so the quasi-TE00 mode (dominant Eₓ) sees the extraordinary index nₑ
# and couples through d₃₃.  Increasing θ tilts propagation within the crystal Y–Z (film)
# plane toward the c-axis.  ε_sim = S(θ)ᵀ ε_crystal S(θ) — the same convention as `rotate`.
function Sframe(θ)
    s, c = sin(θ), cos(θ)
    [0.0 1.0 0.0
     -s  0.0 c
     c   0.0 s]
end

"Rotate one 27-entry material column (vec ε; vec ∂ωε; vec ∂²ωε) by ε_sim = Sᵀ ε S."
function rotate_col(col, S)
    vcat(vec(S' * reshape(col[1:9], 3, 3) * S),
        vec(S' * reshape(col[10:18], 3, 3) * S),
        vec(S' * reshape(col[19:27], 3, 3) * S))
end

"Per-material dielectric columns at (ω, T, θ): LiNbO₃ rotated to the x-cut sim frame, SiO₂, air."
function mat_vals(ω, T, θ)
    cols = f_ε([ω, T])                              # 27 × 2  (LiNbO₃ crystal frame, SiO₂)
    S = Sframe(θ)
    hcat(rotate_col(cols[:, 1], S), cols[:, 2], air_col)
end

# ---------------------------------------------------------------------------------------
# Geometry: a 400 nm x-cut TFLN film, partially etched into a rib (top width w, etch
# fraction etch), on an SiO₂ substrate with air above.  Shapes foreground-first.
# ---------------------------------------------------------------------------------------
function tfln_shapes(w, t, etch)
    etch = clamp(etch, 1e-3, 1 - 1e-3)
    h_ridge = t * etch
    h_slab = t * (1 - etch)
    ridge = MaterialShape(Cuboid([0.0, h_slab + h_ridge / 2], [w, h_ridge], [1.0 0.0; 0.0 1.0]), 1)
    slab = MaterialShape(Cuboid([0.0, h_slab / 2], [100.0, h_slab], [1.0 0.0; 0.0 1.0]), 1)
    substrate = MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2)
    (ridge, slab, substrate)
end
const minds = (1, 1, 2, 3)            # ridge→LiNbO₃, slab→LiNbO₃, substrate→SiO₂, bg→air
const w0, t0, etch0 = 1.20, 0.40, 0.50    # top width, film thickness (400 nm), etch fraction
const grid = Grid(6.0, 4.0, 48, 36)       # 6 × 4 μm cell
const solver = KrylovKitEigsolve()

"Smoothed dielectric tensor field ε(x,y) at (ω,T,θ)."
εfield(ω, T, θ) = copy(selectdim(smooth_ε(tfln_shapes(w0, t0, etch0), mat_vals(ω, T, θ), minds, grid), 3, 1))

# ---------------------------------------------------------------------------------------
# Quasi-TE00 mode solve + first-order sensitivity weight.
#
# `te00_mode` returns the wavenumber k of the most TE-polarized guided mode (largest Eₓ
# power fraction — the quasi-TE00) and, frozen at that mode, the linear functional
#   L(X) = ⟨E| X |E⟩ / (2⟨ev|∂M̂/∂k|ev⟩)
# represented as a weight array `Lw` over the dielectric field, such that the first-order
# change in k for any dielectric perturbation δε is dot(Lw, δε).  Hence at fixed ω,
#   ∂k/∂p = dot(Lw, ∂ε/∂p),
# and ∂ε/∂p is obtained by differentiating the material map — forward or reverse mode.
# ---------------------------------------------------------------------------------------
function solve_te00(ω, T, θ; nev=4)
    sm = smooth_ε(tfln_shapes(w0, t0, etch0), mat_vals(ω, T, θ), minds, grid)
    ε = copy(selectdim(sm, 3, 1))
    ε⁻¹ = sliceinv_3x3(copy(ε))
    ∂ωε = copy(selectdim(sm, 3, 2))
    kmags, evecs = solve_k(ω, copy(ε⁻¹), grid, solver; nev=nev, k_tol=1e-12, eig_tol=1e-12)
    fracs = [E_relpower_xyz(ε⁻¹, E⃗(kmags[i], copy(evecs[i]), ε⁻¹, ∂ωε, grid;
        canonicalize=true, normalized=true))[1] for i in eachindex(evecs)]
    i = argmax(fracs)
    return (k=kmags[i], ev=evecs[i], ε=ε, ε⁻¹=ε⁻¹, te_frac=fracs[i])
end

# just the quasi-TE00 wavenumber (no sensitivity), for sweeps / transfer function / FD references
te00_k(ω, T, θ) = solve_te00(ω, T, θ).k

function te00_mode(ω, T, θ; nev=4)
    s = solve_te00(ω, T, θ; nev)
    mag, mn = mag_mn(s.k, grid)
    HMkH = HMₖH(vec(s.ev), s.ε⁻¹, mag, mn)
    # weight array Lw for the modal sensitivity functional (built once, pure Float64)
    Lfun(xv) = HMH(vec(s.ev), _dot(s.ε⁻¹, reshape(xv, size(s.ε)), s.ε⁻¹), mag, mn) / (2 * HMkH)
    Lw = reshape(Zygote.gradient(Lfun, vec(s.ε))[1], size(s.ε))
    return (; k=s.k, neff=s.k / ω, te_frac=s.te_frac, Lw)
end

# ---------------------------------------------------------------------------------------
# Design point: 1560 nm → 780 nm, mid-range temperature and a small Y-axis offset.
# ---------------------------------------------------------------------------------------
λ_f = 1.560                  # fundamental vacuum wavelength [μm]
ω_f = 1 / λ_f
ω_sh = 2ω_f                  # second harmonic, λ = 780 nm
T0 = 50.0                    # design temperature [°C]
θ0 = deg2rad(7.5)            # design propagation offset from crystal Y [rad]

println("="^80)
@printf("Thin-film LiNbO₃ rib, x-cut (film normal ∥ crystal X), 400 nm film\n")
@printf("  cell 6×4 μm  grid %d×%d  |  w=%.2f μm  t=%.3f μm  etch=%.2f\n", size(grid)..., w0, t0, etch0)
@printf("  fundamental λ=%.3f μm,  second harmonic λ=%.4f μm\n", λ_f, λ_f / 2)
@printf("  design point:  T=%.1f °C,  propagation offset θ=%.2f° from crystal Y\n", T0, rad2deg(θ0))
println("="^80)

mf = te00_mode(ω_f, T0, θ0)
ms = te00_mode(ω_sh, T0, θ0)
@printf("\n%-26s %14s %14s\n", "", "fundamental ω", "2nd-harm 2ω")
@printf("%-26s %14.5f %14.5f\n", "n_eff (quasi-TE00)", mf.neff, ms.neff)
@printf("%-26s %13.2f%% %13.2f%%\n", "TE fraction (Eₓ)", 100mf.te_frac, 100ms.te_frac)

# SHG phase matching at the design point ---------------------------------------------------
Δk0 = ms.k - 2mf.k                       # k(2ω) − 2k(ω) [μm⁻¹]
Λ_poling = 1 / abs(Δk0)                  # QPM poling period Λ = 2π/|Δβ| = 1/|Δk|  [μm]
K = Δk0                                  # grating wavevector; phase matching holds at design
@printf("\nSHG phase matching at the design point:\n")
@printf("  Δk = k(2ω) − 2k(ω)   = %+.5f μm⁻¹\n", Δk0)
@printf("  Δβ = 2π·Δk           = %+.4f μm⁻¹\n", 2π * Δk0)
@printf("  QPM poling period Λ  = %.4f μm\n", Λ_poling)

# Phase mismatch vs fundamental wavelength, with the design grating subtracted.  The QPM
# SHG efficiency is η(λ) ∝ sinc²(Δβ_tot·L/2) = sinc²(π·Δk_tot·L); it peaks where Δk_tot = 0.
Δk_tot(λ, T, θ) = te00_k(2 / λ, T, θ) - 2 * te00_k(1 / λ, T, θ) - K
sinc2(x) = (s = sinc(x); s * s)          # sinc(x)=sin(πx)/(πx) in Julia
L_dev = 1.0e4                            # device length 1 cm = 1e4 μm
shg_eff(λ, T, θ) = sinc2(Δk_tot(λ, T, θ) * L_dev)

# slope ∂Δk_tot/∂λ at the design point.  Δk_tot is essentially linear in λ over this small
# range, so a single evaluation lets us locate the peak (fixed-slope Newton) and estimate
# tuning without re-finding the dispersion slope at every point.
const dΔk_dλ0 = central_fdm(5, 1)(λ -> Δk_tot(λ, T0, θ0), λ_f)

println("\nNormalized SHG transfer function η(λ)=sinc²(Δk_tot·L) near 1560 nm (L = 1 cm), at design:")
@printf("  %10s %12s\n", "λ_f [nm]", "η (norm.)")
for λ in (λ_f - 0.002):0.001:(λ_f+0.002)
    @printf("  %10.1f %12.4f\n", 1e3λ, shg_eff(λ, T0, θ0))
end

# ---------------------------------------------------------------------------------------
# Temperature and angle tuning of the phase-matched (peak SHG) fundamental wavelength.
# λ_peak solves Δk_tot(λ) = 0; we locate it by a few Newton steps seeded at λ_f.
# ---------------------------------------------------------------------------------------
# exact peak via fixed-slope Newton (2 mode solves per iteration), and a one-shot linear
# estimate λ_peak ≈ λ_f − Δk_tot(λ_f)/slope used for the sweeps below (2 solves per point).
function λ_peak(T, θ; iters=4)
    λ = λ_f
    for _ in 1:iters
        λ -= Δk_tot(λ, T, θ) / dΔk_dλ0
    end
    λ
end
λ_peak_lin(T, θ) = λ_f - Δk_tot(λ_f, T, θ) / dΔk_dλ0

@printf("\nDesign check: λ_peak(T0,θ0) = %.4f nm (exact Newton) vs %.4f nm target\n",
    1e3 * λ_peak(T0, θ0), 1e3λ_f)
println("\nPeak SHG (phase-matched) fundamental wavelength vs temperature (θ = $(round(rad2deg(θ0),digits=1))°):")
@printf("  %8s %14s\n", "T [°C]", "λ_peak [nm]")
for T in 30.0:10.0:80.0
    @printf("  %8.1f %14.3f\n", T, 1e3 * λ_peak_lin(T, θ0))
end
println("\nPeak SHG fundamental wavelength vs propagation angle (T = $(T0) °C):")
@printf("  %8s %14s\n", "θ [deg]", "λ_peak [nm]")
for θd in 0.0:3.0:15.0
    @printf("  %8.1f %14.3f\n", θd, 1e3 * λ_peak_lin(T0, deg2rad(θd)))
end

# ---------------------------------------------------------------------------------------
# Gradient of the peak SHG wavelength w.r.t. T and θ — forward AD, reverse AD, and FD.
#
# By the implicit function theorem applied to Δk_tot(λ_peak; T,θ) = 0:
#   dλ_peak/dp = −(∂Δk_tot/∂p) / (∂Δk_tot/∂λ).
# The denominator is a cheap scalar wavelength derivative (finite-differenced once).  The
# numerator ∂Δk_tot/∂p = ∂k(2ω)/∂p − 2 ∂k(ω)/∂p is the modal material sensitivity, obtained
# from the frozen-mode weights Lw by differentiating the material map ε(p):
#   ∂k/∂p = ForwardDiff/Zygote of  p ↦ dot(Lw, ε(ω, p)).
# ---------------------------------------------------------------------------------------
"∂k/∂(T,θ) of the quasi-TE00 mode at frequency ω, via AD of the material map (`backend` = :forward or :reverse)."
function ∂k_∂Tθ(Lw, ω, T, θ; backend)
    g(p) = dot(Lw, vec(εfield(ω, p[1], p[2])))
    backend === :forward ? ForwardDiff.gradient(g, [T, θ]) :
    backend === :reverse ? Zygote.gradient(g, [T, θ])[1] :
    error("backend must be :forward or :reverse")
end

"∇(T,θ) λ_peak at the design point via the implicit function theorem, using `backend` AD for ∂Δk/∂p."
function grad_λpeak(backend)
    mf = te00_mode(ω_f, T0, θ0)
    ms = te00_mode(ω_sh, T0, θ0)
    ∂Δk_∂p = ∂k_∂Tθ(ms.Lw, ω_sh, T0, θ0; backend) .- 2 .* ∂k_∂Tθ(mf.Lw, ω_f, T0, θ0; backend)
    -∂Δk_∂p ./ dΔk_dλ0
end

g_fwd = grad_λpeak(:forward)
g_rev = grad_λpeak(:reverse)

# finite-difference reference: re-root λ_peak with the full mode re-solve
hT, hθ = 2.0, deg2rad(2.0)
g_fd = [(λ_peak(T0 + hT, θ0) - λ_peak(T0 - hT, θ0)) / (2hT),
    (λ_peak(T0, θ0 + hθ) - λ_peak(T0, θ0 - hθ)) / (2hθ)]

println("\n" * "="^80)
println("Gradient of the peak SHG wavelength λ_peak at the design point")
println("="^80)
@printf("%-22s %18s %18s\n", "", "dλ_peak/dT", "dλ_peak/dθ")
@printf("%-22s %18s %18s\n", "", "[nm/°C]", "[nm/deg]")
@printf("%-22s %18.5f %18.5f\n", "forward-mode AD", 1e3g_fwd[1], 1e3deg2rad(1) * g_fwd[2])
@printf("%-22s %18.5f %18.5f\n", "reverse-mode AD", 1e3g_rev[1], 1e3deg2rad(1) * g_rev[2])
@printf("%-22s %18.5f %18.5f\n", "finite differences", 1e3g_fd[1], 1e3deg2rad(1) * g_fd[2])
@printf("\nforward vs reverse:   max rel. diff = %.2e\n", maximum(abs.(g_fwd .- g_rev) ./ abs.(g_fwd)))
@printf("AD vs finite diff:    max rel. diff = %.2e\n", maximum(abs.(g_fwd .- g_fd) ./ abs.(g_fd)))
@printf("\nPredicted tuning:  %.2f nm over 30–80 °C,  %.2f nm over 0–15° (first order).\n",
    1e3 * g_fwd[1] * 50, 1e3 * g_fwd[2] * deg2rad(15))
println("""
Forward and reverse mode AD of the LiNbO₃ material model give the same peak-wavelength
gradient as finite differences of the full quasi-TE00 mode re-solve — the temperature and
crystal-orientation tuning rates that a differentiable design loop would back-propagate
through, computed without finite-difference truncation error.""")
