# Hermite–Gaussian mode labeling vs. the node-counting classifier.
#
# OptiMode labels a guided mode by its polarization and transverse order. The original
# classifier (`count_E_nodes` / `mode_idx` in ModeAnalysis/analyze.jl) finds the dominant
# polarization from the relative E-field power and the mode order by counting field-
# amplitude zero crossings (nodes) along the two transverse axes through the field peak.
# Node counting is fast but threshold-dependent (`rel_amp_min`), can miscount near mode
# crossings or for non-axis-aligned nodal lines, and gives no measure of how cleanly a
# field actually resembles a Hermite–Gaussian.
#
# `hg_mode_label` (ModeAnalysis/hermite_gaussian.jl) is an alternative: it fits an
# elliptical Hermite–Gaussian template
#
#   ψ_{mn}(x,y) = H_m(√2(x−x₀)/w_x) H_n(√2(y−y₀)/w_y) exp[−(x−x₀)²/w_x² − (y−y₀)²/w_y²]
#
# to the dominant transverse field for every order (m,n) and polarization (TE≈x, TM≈y),
# seeding the shape parameters at the field's intensity centroid with matched transverse
# variances, optimizing the four shape parameters per candidate (the amplitude is solved
# analytically), and labeling the mode by the lowest-residual fit. It is threshold-free
# and returns a normalized goodness-of-fit `rel_error ∈ [0,1]`.
#
# This script compares the two classifiers on Si₃N₄- and x-cut-LiNbO₃-core multimode
# waveguides with realistic, dispersive (and, for LiNbO₃, anisotropic) materials.
#
#   julia --project=. examples/hermite_gaussian_mode_labeling.jl
#
# Note on the node count: `count_E_nodes` returns Σ|Δ sign|, i.e. *twice* the Hermite–
# Gaussian order (each zero crossing flips the field sign, contributing 2), so the order
# it implies is node_count ÷ 2 — that is what we compare against the HG-fit order.

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModeAnalysis: hg_mode_label, Eperp_max
using LinearAlgebra

const _SUB = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
sub(k) = _SUB[k+1]

# --- smoothed dielectric fields (ε, ε⁻¹, ∂ωε) for a core/cladding waveguide geometry ---
# `matvals` is the (27 × n_materials) column matrix from `_f_ε_mats`, evaluated up front
# at the design frequency (so the runtime-generated dispersion function is called from a
# later world age than where it was built — no `invokelatest` needed).
function build_eps(matvals, geom, minds, grid)
    sm = smooth_ε(geom, matvals, minds, grid)
    ε = copy(selectdim(sm, 3, 1))
    ∂ωε = copy(selectdim(sm, 3, 2))
    ε⁻¹ = sliceinv_3x3(copy(ε))
    return ε, ε⁻¹, ∂ωε
end

# cladding (background) index, for flagging genuinely guided modes (n_eff > n_clad)
nclad(ε) = sqrt(maximum(ε[a, a, 1, 1] for a in 1:3))

# original node-counting label (order = raw node count ÷ 2; see header note)
function node_label(E, ε)
    En = E ./ Eperp_max(E)
    pol = argmax(E_relpower_xyz(ε, En))
    nodes = count_E_nodes(En, ε, pol; rel_amp_min=0.1)
    return (pol == 1 ? :TE : :TM, Int(nodes[1]) ÷ 2, Int(nodes[2]) ÷ 2)
end

function compare_classifiers(name, matvals, geom, minds, grid, ω; nev=8, max_order=4)
    println("\n", "="^86)
    println(name, "   (grid ", size(grid), ", λ = ", round(1 / ω, digits=3), " μm)")
    println("="^86)
    ε, ε⁻¹, ∂ωε = build_eps(matvals, geom, minds, grid)
    nc = nclad(ε)
    kmags, evecs = solve_k(ω, copy(ε⁻¹), grid, KrylovKitEigsolve(); nev, k_tol=1e-11, eig_tol=1e-11)
    println(rpad("idx", 4), rpad("neff", 9), rpad("guided", 8), rpad("node÷2", 9),
        rpad("HG-fit", 9), rpad("rel_err", 10), rpad("te_frac", 9), "agree")
    println("-"^86)
    nagree = nguided = 0
    for i in eachindex(evecs)
        neff = kmags[i] / ω
        guided = neff > nc + 1e-3
        E = E⃗(kmags[i], copy(evecs[i]), ε⁻¹, ∂ωε, grid; canonicalize=true, normalized=true)
        op, om, on = node_label(E, ε)
        nw = hg_mode_label(E, grid; max_order)
        agree = (nw.pol, nw.m, nw.n) == (op, om, on)
        guided && (nguided += 1; nagree += agree)
        println(rpad(i, 4), rpad(round(neff, digits=4), 9), rpad(guided ? "yes" : "—", 8),
            rpad(string(op, sub(om), sub(on)), 9), rpad(nw.label, 9),
            rpad(round(nw.rel_error, sigdigits=3), 10), rpad(round(nw.te_frac, digits=3), 9),
            guided ? (agree ? "✓" : "✗") : "(radiation)")
    end
    println("-"^86)
    println("guided-mode agreement (node÷2 vs HG-fit): ", nagree, "/", nguided)
end

const ω = 1 / 1.55
const air_col = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))   # ε = I, no dispersion

# ===================== Si₃N₄ multimode waveguide =====================
# 2.5 × 0.7 μm Si₃N₄ core in SiO₂; supports several quasi-TE/TM orders at 1.55 μm.
let
    fε, _ = _f_ε_mats([Si₃N₄, SiO₂], (:ω,))
    matvals = hcat(fε([ω]), air_col)
    grid = Grid(7.0, 4.0, 112, 64)
    w, h = 2.5, 0.7
    geom = (
        MaterialShape(Cuboid([0.0, 0.0], [w, h], [1.0 0.0; 0.0 1.0]), 1),                # Si₃N₄ core
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2),      # SiO₂ surround
    )
    compare_classifiers("Si₃N₄ core ($(w)×$(h) μm) in SiO₂", matvals, geom, (1, 2, 2), grid, ω)
end

# ===================== x-cut LiNbO₃ ridge waveguide =====================
# 1.4 × 0.6 μm x-cut TFLN ridge on SiO₂, air above (anisotropic, dispersive, multimode).
let
    Ry = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]              # c-axis in-plane along x
    LiNbO₃_xcut = rotate(LiNbO₃, Ry; name=:LiNbO₃_xcut)
    fε, _ = _f_ε_mats([LiNbO₃_xcut, SiO₂], (:ω,))
    matvals = hcat(fε([ω]), air_col)
    grid = Grid(7.0, 4.0, 112, 64)
    w, t = 1.4, 0.6
    geom = (
        MaterialShape(Cuboid([0.0, t / 2], [w, t], [1.0 0.0; 0.0 1.0]), 1),              # LiNbO₃ ridge
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2),      # SiO₂ substrate
    )
    compare_classifiers("x-cut LiNbO₃ ridge ($(w)×$(t) μm) on SiO₂", matvals, geom, (1, 2, 3), grid, ω)
end
