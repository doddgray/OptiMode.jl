# Shared geometry / mode-solver helpers for reproducing
#   M. Jankowski et al., "Ultrabroadband nonlinear optics in nanophotonic periodically
#   poled lithium niobate waveguides," Optica 7, 40 (2020).  https://doi.org/10.1364/OPTICA.7.000040
#
# Device: x-cut MgO:LiNbO₃, 700-nm starting film, direct-etched ridge of top width `w`
# (µm) and etch depth `etch` (µm) on the remaining (700 nm − etch) slab, SiO₂ lower
# cladding, air above. The quasi-TE₀₀ mode (dominant Eₓ) sees the extraordinary index nₑ,
# so we rotate the bundled LiNbO₃ (c-axis → z) by RotY(π/2) to put the c-axis in-plane
# along x (the x-cut convention used in the package's other TFLN examples).
#
# Physical constants and the OptiMode → physical unit conversions used throughout:
#   • OptiMode frequency  ω = 1/λ  (µm⁻¹);  wavenumber  |k| = neff·ω  (µm⁻¹).
#   • physical β = 2π|k|;  angular ω_a = 2πc/λ.
#   • inverse group velocity 1/v_g = n_g/c   ⇒  GVM Δk′ = (n_g,2ω − n_g,ω)/c.
#   • GVD β₂ = d²β/dω_a² = gvd_OM/(2π c²), with gvd_OM = ∂²|k|/∂ω² from `ng_gvd`.

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using LinearAlgebra

const C_UM_FS = 299792458.0 * 1e-9          # speed of light in µm/fs  (=0.299792458)
const FILM_NM = 700.0                        # starting x-cut TFLN film thickness (nm)
const D33_PMV = 20.5                         # d₃₃ at 2050 nm (pm/V), Jankowski 2020
const DEFF_MV = (2 / π) * D33_PMV * 1e-12    # first-order-QPM d_eff (m/V)

# x-cut LiNbO₃ (extraordinary axis in-plane along x) + SiO₂; air added as a third column.
const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
const LiNbO₃_xcut = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
const _AIR_COL = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))
const _fε_LN, _ = _f_ε_mats([LiNbO₃_xcut, SiO₂], (:ω,))
matvals_LN(ω) = hcat(_fε_LN([ω]), _AIR_COL)     # 27×3 columns: LiNbO₃, SiO₂, air

"Foreground-first shapes for an x-cut TFLN ridge of top width `w` and etch depth `etch` (µm)."
function ppln_shapes(w::Real, etch::Real; film=FILM_NM / 1e3)
    slab = film - etch                       # unetched LN slab thickness (µm)
    (MaterialShape(Cuboid([0.0, slab + etch / 2], [w, etch], [1.0 0.0; 0.0 1.0]), 1),  # ridge (LN)
        MaterialShape(Cuboid([0.0, slab / 2], [200.0, slab], [1.0 0.0; 0.0 1.0]), 1),  # slab  (LN)
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2))    # SiO₂ substrate
end
const PPLN_MINDS = (1, 1, 2, 3)              # ridge→LN, slab→LN, substrate→SiO₂, background→air

"Smoothed (ε⁻¹, ∂ωε, ∂²ωε) dielectric fields for the ridge at frequency `ω`."
function ppln_diel(w, etch, ω, grid)
    sm = smooth_ε(ppln_shapes(w, etch), matvals_LN(ω), PPLN_MINDS, grid)
    (sliceinv_3x3(copy(selectdim(sm, 3, 1))), copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
end

"Fraction of a mode's |E|² energy inside the ridge column (|x|<w, 0<y<film) — distinguishes
the ridge-confined mode from laterally-extended slab modes."
function ridge_confinement(E, grid, w; film=FILM_NM / 1e3)
    Nx, Ny = size(grid)
    xc = (-grid.Δx / 2) .+ (0.5:Nx) .* (grid.Δx / Nx)
    yc = (-grid.Δy / 2) .+ (0.5:Ny) .* (grid.Δy / Ny)
    I = dropdims(sum(abs2, E; dims=1); dims=1)
    mask = [(abs(x) < w && -0.05 < y < film + 0.15) ? 1.0 : 0.0 for x in xc, y in yc]
    return sum(I .* mask) / sum(I)
end

"""
    solve_te00(w, etch, ω, grid, solver; nev=6) -> (; k, ev, εi, ∂ωε, ∂²ωε, te_frac, conf, E)

Ridge quasi-TE₀₀ mode of the ridge at frequency `ω`: among the solved bands we keep the
Eₓ-dominant (quasi-TE) ones and pick the one most confined to the ridge column. This
rejects both the higher-index quasi-TM modes (Eᵧ-dominant, since n_o > n_e in LiNbO₃) and
the laterally-extended slab modes (TE-polarized but delocalized) that this thick-slab,
lateral-leakage geometry supports.
"""
function solve_te00(w, etch, ω, grid, solver; nev=6)
    εi, ∂ωε, ∂²ωε = ppln_diel(w, etch, ω, grid)
    ε = sliceinv_3x3(copy(εi))
    km, ev = solve_k(ω, copy(εi), grid, solver; nev=nev, k_tol=1e-10, eig_tol=1e-10)
    Es = [E⃗(km[i], copy(ev[i]), εi, ∂ωε, grid; canonicalize=true, normalized=true) for i in eachindex(ev)]
    te = [E_relpower_xyz(ε, Es[i])[1] for i in eachindex(ev)]
    conf = [ridge_confinement(Es[i], grid, w) for i in eachindex(ev)]
    score = [te[i] > 0.5 ? conf[i] : -1.0 for i in eachindex(ev)]   # quasi-TE, most ridge-confined
    i = argmax(score)
    return (; k=km[i], ev=ev[i], εi, ∂ωε, ∂²ωε, te_frac=te[i], conf=conf[i], E=Es[i])
end

"Group-velocity mismatch Δk′ = (n_g,2ω − n_g,ω)/c in fs/mm."
gvm_fs_per_mm(ng_FF, ng_SH) = 1e3 * (ng_SH - ng_FF) / C_UM_FS

"Group-velocity dispersion β₂ = gvd_OM/(2π c²) in fs²/mm, from the OptiMode `ng_gvd` output."
gvd_fs2_per_mm(gvd_OM) = 1e3 * gvd_OM / (2π * C_UM_FS^2)

"Required first-order QPM poling period Λ = λ/(2(n_2ω − n_ω)) in µm."
poling_period(neff_FF, neff_SH, λ_FF) = λ_FF / (2 * (neff_SH - neff_FF))

"""
    shg_eta0_eq1(neff_FF, neff_SH, λ_FF, Aeff_um2; deff=DEFF_MV) -> Real

Normalized SHG efficiency η₀ (%/W·cm²) from Jankowski 2020 Eq. (1),
`η₀ = 2 ω_a² d_eff² / (n_ω² n_2ω ε₀ c³ A_eff)`, with `A_eff` in µm² (→ m² internally).
"""
function shg_eta0_eq1(neff_FF, neff_SH, λ_FF, Aeff_um2; deff=DEFF_MV)
    c = 299792458.0
    ε0 = 8.8541878128e-12
    ω_a = 2π * c / (λ_FF * 1e-6)
    Aeff = Aeff_um2 * 1e-12
    η = 2 * ω_a^2 * deff^2 / (neff_FF^2 * neff_SH * ε0 * c^3 * Aeff)   # 1/(W·m²)
    return η * 100 * 1e-4                                              # → %/(W·cm²)
end
