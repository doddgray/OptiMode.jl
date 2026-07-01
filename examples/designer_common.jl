# Shared machinery for the AD-driven *designer* scripts. Each designer applies the modeling
# workflow of a reproduction example to a new material stack and new wavelength target and uses
# OptiMode's automatic differentiation to optimize the waveguide geometry for that target.
#
# The geometry gradient is the hybrid forward/reverse scheme of Gray, West & Ram, "Inverse
# design for waveguide dispersion with a differentiable mode solver," Opt. Express 32, 30541
# (2024): ForwardDiff carries the geometry parameters through shape construction + Kottke
# smoothing to the (FFT-free) dielectric fields (exact, thanks to the parametric-eltype
# GeometryPrimitives fork), and Zygote's reverse adjoint of the eigensolve (`solve_k`'s rrule)
# supplies the cotangents of the mode quantity w.r.t. those fields; the two are chained. A mode
# quantity gradient costs ≈ one forward + one adjoint solve, so gradient-based optimization runs
# on a modest grid in seconds.

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode: solve_k, sliceinv_3x3, smooth_ε, E⃗, E_relpower_xyz, group_index, ng_gvd
using LinearAlgebra
using ForwardDiff, Zygote
using Printf

# ---------------------------------------------------------------------------------------
# Hybrid geometry gradient
# ---------------------------------------------------------------------------------------

"Smoothed (ε⁻¹, ∂ωε, ∂²ωε) for a parametric geometry `geomfn(p)` at frequency `om`."
function diel_p(geomfn, matvals, minds, grid, p, om)
    sm = smooth_ε(geomfn(p), matvals(om), minds, grid)
    (sliceinv_3x3(copy(selectdim(sm, 3, 1))), copy(selectdim(sm, 3, 2)), copy(selectdim(sm, 3, 3)))
end

"""
    geom_value_grad(fobj, geomfn, matvals, minds, grid, p, om) -> (value, ∇ₚ value)

Value and geometry gradient of a scalar mode functional `fobj(ε⁻¹, ∂ωε)` (e.g. n_eff, n_g),
via ForwardDiff geometry→dielectric Jacobian ∘ Zygote reverse-adjoint of the eigensolve."""
function geom_value_grad(fobj, geomfn, matvals, minds, grid, p, om)
    ei0, de0, _ = diel_p(geomfn, matvals, minds, grid, p, om)
    N = length(vec(ei0))
    J = ForwardDiff.jacobian(q -> (d = diel_p(geomfn, matvals, minds, grid, q, om); vcat(vec(d[1]), vec(d[2]))), p)
    val, back = Zygote.pullback(fobj, copy(ei0), copy(de0))
    ḡei, ḡde = back(one(val))
    ḡei === nothing && (ḡei = zero(ei0))       # objective may not depend on ∂ωε (n_eff) …
    ḡde === nothing && (ḡde = zero(de0))       # … or on ε⁻¹
    grad = J[1:N, :]' * vec(ḡei) .+ J[N+1:2N, :]' * vec(ḡde)
    (val, grad)
end

"Fundamental effective index n_eff(ε⁻¹) at frequency `om` (Zygote-differentiable via solve_k's
adjoint). `warm` seeds nothing; kept simple for AD."
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
