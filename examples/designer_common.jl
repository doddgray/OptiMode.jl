# Shared machinery for the AD-driven *designer* scripts. Each designer applies the modeling
# workflow of a reproduction example to a new material stack and new wavelength target and uses
# OptiMode's automatic differentiation to optimize the waveguide geometry for that target.
#
# The geometry gradient is computed by pure Enzyme reverse-mode across the whole pipeline —
# geometry parameters → shape construction → Kottke smoothing → (FFT-free) dielectric fields →
# `solve_k`'s adjoint eigensolve → mode quantity — in one `Enzyme.gradient(ReverseWithPrimal, …)`
# call. This became possible once `DielectricSmoothingEnzymeExt.jl` stopped marking
# `surfpt_nearby`/`volfrac`/`_interface_geometry` as `EnzymeRules.inactive`: those functions
# carry the actual geometry dependence of the smoothed dielectric, so marking them inactive
# (previously done to dodge an old Enzyme Union-type limitation) silently zeroed out geometry
# gradients. With that fix, Enzyme reproduces ForwardDiff/finite-difference geometry gradients
# exactly, in both homogeneous- and heterogeneous-shape-type tuples, at the same steady-state
# cost as the previous ForwardDiff-Jacobian ∘ Zygote-adjoint hybrid scheme of Gray, West & Ram,
# "Inverse design for waveguide dispersion with a differentiable mode solver," Opt. Express 32,
# 30541 (2024) (≈ one forward + one adjoint solve per gradient) — but as a single AD backend
# instead of two composed ones.

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode: solve_k, sliceinv_3x3, smooth_ε, E⃗, E_relpower_xyz, group_index, ng_gvd
using LinearAlgebra
using Enzyme
using Printf

# ---------------------------------------------------------------------------------------
# Enzyme geometry gradient
# ---------------------------------------------------------------------------------------

"Smoothed (ε⁻¹, ∂ωε, ∂²ωε) for a parametric geometry `geomfn(p)` at frequency `om`."
function diel_p(geomfn, matvals, minds, grid, p, om)
    sm = smooth_ε(geomfn(p), matvals(om), minds, grid)
    (sliceinv_3x3(copy(selectdim(sm, 3, 1))), copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
end

# Runtime activity: `matvals(om)` (`matvals_builder(...; air=true)`) `hcat`s a computed material
# column with the literal constant `AIR_COL`; Enzyme's static activity analysis cannot prove
# that array (mixing constant and active memory in one `hcat`) is safe without this, and raises
# `EnzymeRuntimeActivityError` — the same fix `ad_backend_benchmarks.jl` already uses.
const _REVWP = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)

"""
    geom_value_grad(fobj, geomfn, matvals, minds, grid, p, om) -> (value, ∇ₚ value)

Value and geometry gradient of a scalar mode functional `fobj(ε⁻¹, ∂ωε)` (e.g. n_eff, n_g), by
Enzyme reverse-mode automatic differentiation of the whole geometry→dielectric→eigensolve
pipeline, in one pass (`ReverseWithPrimal` returns the primal value alongside the gradient)."""
function geom_value_grad(fobj, geomfn, matvals, minds, grid, p, om)
    F(q) = (d = diel_p(geomfn, matvals, minds, grid, q, om); fobj(d[1], d[2]))
    g = Enzyme.gradient(_REVWP, F, p)
    (g.val, g.derivs[1])
end

"Fundamental effective index n_eff(ε⁻¹) at frequency `om` (Enzyme-differentiable via solve_k's
adjoint rule)."
neff_of(ei, om, grid, solver; k_tol=1e-9) = solve_k(om, ei, grid, solver; nev=1, k_tol=k_tol)[1][1] / om

"Fundamental group index n_g(ε⁻¹, ∂ωε) at `om`."
function ng_of(ei, de, om, grid, solver; k_tol=1e-9)
    k, ev = solve_k(om, ei, grid, solver; nev=1, k_tol=k_tol)
    group_index(k[1], ev[1], om, ei, de, grid)
end

"""
    gvd_value_grad(geomfn, matvals, minds, grid, p, om, solver; Δ) -> (β₂ fs²/mm, ∇ₚ β₂)

GVD β₂ and its geometry gradient. `ng_gvd`'s hand-rolled adjoint is not itself reverse-mode
differentiable, so — following the 2024 paper — the high-dimensional geometry gradient stays
exact AD on n_g and only the scalar ω-derivative (GVD = ∂n_g/∂ω ⇒ β₂) is finite-differenced."""
function gvd_value_grad(geomfn, matvals, minds, grid, p, om, solver; Δ=1e-3)
    grad_ng(o) = geom_value_grad((ei, de) -> ng_of(ei, de, o, grid, solver), geomfn, matvals, minds, grid, p, o)[2]
    # β₂ geometry gradient = ∂/∂ω(∇ₚ n_g), mapped to fs²/mm (gvd_fs2_per_mm is linear)
    dβ2 = gvd_fs2_per_mm.((grad_ng(om + Δ) .- grad_ng(om - Δ)) ./ (2Δ))
    # β₂ value from ng_gvd (gvd_OM = ∂²|k|/∂ω²)
    e, d, dd = diel_p(geomfn, matvals, minds, grid, p, om)
    k, ev = solve_k(om, e, grid, solver; nev=1, k_tol=1e-9)
    β2 = gvd_fs2_per_mm(ng_gvd(om, k[1], ev[1], e, d, dd, grid)[2])
    (β2, dβ2)
end

# ---------------------------------------------------------------------------------------
# Simple projected-gradient / Adam optimizer
# ---------------------------------------------------------------------------------------

"""
    optimize_design(loss_and_grad, p0; lo, hi, iters, lr, verbose) -> (; p, history)

Minimize a scalar loss with Adam, projecting each step into the box [lo, hi]. `loss_and_grad(p)`
returns `(loss, ∇ₚ loss)`. Returns the optimized parameters and the loss history."""
function optimize_design(loss_and_grad, p0; lo, hi, iters=30, lr=0.02, verbose=true)
    p = copy(float.(p0)); m = zero(p); v = zero(p); β1, β2, ϵ = 0.9, 0.999, 1e-8
    history = Float64[]; best_p = copy(p); best_L = Inf
    for t in 1:iters
        L, g = loss_and_grad(p)
        push!(history, L)
        L < best_L && (best_L = L; best_p = copy(p))     # keep the best-so-far design
        verbose && (@printf("  iter %2d  loss=%.4e  p=%s  |g|=%.2e\n", t, L, string(round.(p; digits=4)), norm(g)); flush(stdout))
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* g .^ 2
        m̂ = m ./ (1 - β1^t); v̂ = v ./ (1 - β2^t)
        p .= clamp.(p .- lr .* m̂ ./ (sqrt.(v̂) .+ ϵ), lo, hi)
    end
    (; p=best_p, loss=best_L, history)
end
