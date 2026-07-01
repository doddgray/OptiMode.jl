# Remote, SLURM-managed automatic differentiation of the mode solver.
#
# OptiMode's eigensolve `solve_k` carries an adjoint-method `rrule`: given output
# cotangents it returns input cotangents `(ω̄, ε̄⁻¹)` for ≈ one extra eigensolve,
# independent of the number of parameters. ModeSweeps runs the **forward (primal)** and
# **backward (adjoint)** passes as *separate SLURM tasks* that exchange state through the
# cluster's shared filesystem — so an adjoint-based inverse-design loop can offload both
# the value and the gradient evaluation to the cluster, even across Julia sessions.
#
#     julia --project=. examples/remote_adjoint_optimization.jl
#
# As elsewhere, flip `REMOTE` to use ssh+rsync on a real cluster; `:local` runs it all in
# background processes on one machine.

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using ModeSweeps
using Enzyme
using LinearAlgebra: dot

const SETUP = joinpath(@__DIR__, "ridge_wg_setup.jl")
include(SETUP)                       # make_problem(p) — also used locally below

const REMOTE = false
slurm = REMOTE ?
        SlurmConfig(time="0:20:00", partition="general", ssh="me@login.cluster.edu",
            remote_dir="/scratch/me/ad_ridge") : SlurmConfig()
backend = REMOTE ? :slurm : :local

# ----------------------------------------------------------------------------------
# 1. One remote forward + backward pass: ∂neff/∂(ω, ε⁻¹)
# ----------------------------------------------------------------------------------
p = (; ω=1 / 1.55, w_top=1.6, h_core=0.7)

# Forward pass as its own SLURM task. Besides the usual summary, it persists everything
# the adjoint needs — (ω, ε⁻¹, ∂ε_∂ω, grid, kmags, evecs, solver) — to the shared FS.
fwd = deploy_forward(SETUP, [p]; nev=1, name="adjoint_fwd",
    solver_kwargs=(; k_tol=1e-10), backend, slurm, blocking=true)
sol = forward_solution(fwd)                    # (; ω, kmags, evecs, ε⁻¹, ∂ε_∂ω, grid)
neff = sol.kmags[1] / sol.ω
@info "forward pass" neff host = "(see summary)"

# Backward pass as a *separate* SLURM task. Our objective is L = neff = kmag/ω, so the
# cotangent w.r.t. the per-band kmags is k̄ = ∂L/∂kmag = 1/ω.
bwd = deploy_backward(fwd, [(; k̄=[1 / sol.ω])]; name="adjoint_bwd",
    backend, slurm, blocking=true)
g = gradient_result(bwd)                       # (; ω_bar, ε⁻¹_bar)
@info "backward pass" ∂neff_∂ω = g.ω_bar ‖∂neff_∂ε⁻¹‖ = sqrt(sum(abs2, g.ε⁻¹_bar))

# The same in one blocking call:
r = remote_value_and_gradient(SETUP, p, [1 / p.ω]; nev=1, name="adjoint_oneshot",
    solver_kwargs=(; k_tol=1e-10), backend, slurm)
@info "one-shot value+gradient" neff = r.kmags[1] / p.ω ω_bar = r.ω_bar

# ----------------------------------------------------------------------------------
# 2. Geometry gradient by the standard hybrid adjoint pattern
# ----------------------------------------------------------------------------------
# `ε⁻¹_bar` is the sensitivity of the objective to every inverse-dielectric tensor entry
# at every pixel. Chaining it with the (cheap, FFT-free) forward-mode geometry Jacobian
# ∂ε⁻¹/∂(geometry) gives the exact geometry gradient — the eigensolve and its adjoint
# stay on the cluster, only the smoothing Jacobian is differentiated locally (with Enzyme,
# now that geometry-parameter gradients through `smooth_ε` are Enzyme-differentiable).
#
#   ∂neff/∂w_top = vec(ε⁻¹_bar) · ∂vec(ε⁻¹)/∂w_top
#
# `make_problem` itself calls `_f_ε_mats` (SymbolicUtils-backed) on every invocation, and
# SymbolicUtils' internal expression cache is not Enzyme-differentiable (forward mode has no
# rule for its `IdDict`-based memoization) — so unlike the designer scripts (which build the
# material-value function *once*, outside the differentiated closure, via `matvals_builder`),
# here we mirror `make_problem`'s geometry with the materials function precomputed the same way
# before differentiating, keeping only the genuinely geometry-dependent part (shape → smooth_ε)
# inside the Enzyme trace.
const _fε_ridge, _ = _f_ε_mats([Si₃N₄, SiO₂], (:ω,))
function geom_to_epsinv(w)
    mat_vals = _fε_ridge([p.ω])
    core = MaterialShape(Cuboid([0.0, 0.0], [w, p.h_core], [1.0 0.0; 0.0 1.0]), 1)
    sm = smooth_ε((core,), mat_vals, (1, 2), Grid(6.0, 4.0, 128, 96))
    vec(sliceinv_3x3(copy(selectdim(sm, 3, 1))))
end
J = Enzyme.jacobian(Enzyme.Forward, geom_to_epsinv, p.w_top)[1]         # ∂vec(ε⁻¹)/∂w_top
∂neff_∂w_top = dot(vec(g.ε⁻¹_bar), J)
@info "geometry gradient (remote adjoint × local forward Jacobian)" ∂neff_∂w_top

# Finite-difference sanity check of the geometry gradient (re-solves locally):
function neff_local(w)
    pr = make_problem((; p.ω, w_top=w, p.h_core))
    k, _ = solve_k(p.ω, copy(pr.ε⁻¹), pr.grid, KrylovKitEigsolve(); nev=1, k_tol=1e-10)
    return k[1] / p.ω
end
δ = 1e-3
fd = (neff_local(p.w_top + δ) - neff_local(p.w_top - δ)) / (2δ)
@info "vs. finite difference" fd rel_error = abs(∂neff_∂w_top - fd) / abs(fd)

# ----------------------------------------------------------------------------------
# 3. Batched gradients (one forward/backward task per design point)
# ----------------------------------------------------------------------------------
# `deploy_forward`/`deploy_backward` take whole parameter lists, so a population of
# designs (e.g. a generation in an optimizer, or a stencil of finite-difference points)
# is one array job each. Each backward task reads its matching forward state and writes
# its own (ω̄, ε̄⁻¹).
designs = [(; ω=1 / 1.55, w_top=w, h_core=0.7) for w in [1.4, 1.6, 1.8, 2.0]]
fwds = deploy_forward(SETUP, designs; nev=1, name="pop_fwd", backend, slurm, blocking=true)
cots = [(; k̄=[1 / forward_solution(fwds, i).ω]) for i in 1:length(designs)]   # ∂neff/∂kmag
bwds = deploy_backward(fwds, cots; name="pop_bwd", backend, slurm, blocking=true)
grads = [gradient_result(bwds, i) for i in 1:length(designs)]
for (d, gi) in zip(designs, grads)
    println("w_top=", d.w_top, "  ∂neff/∂ω=", round(gi.ω_bar, digits=4),
        "  ‖∂neff/∂ε⁻¹‖=", round(sqrt(sum(abs2, gi.ε⁻¹_bar)), digits=4))
end
