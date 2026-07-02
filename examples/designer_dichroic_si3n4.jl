# AD-driven DESIGNER — spectrally-selective EME dichroic filter on a NEW stack and NEW target.
#
# Applies the mode-crossing / dichroic-filter design methodology of Magden et al., "Transmissive
# silicon photonic dichroic filters with spectrally selective waveguides," Nat. Commun. 9, 3009
# (2018) — reproduced on Si SOI in dichroic_filter_magden2018.jl — to a user-controlled
# **Si₃N₄-on-SiO₂** stack and a NEW cutoff target: **λ_C = 1000 nm**.
#
# Dissimilar guides are made exactly as in the paper's Fig. 1b cross-section (not by thickness
# contrast): a solid strip **WGA** (width w_A) next to a sub-wavelength segmented **WGB** — three
# solid rail segments of width w_B separated by gaps g_B, same material and thickness as WGA.
# "This idea of mimicking a continuous waveguide using segments smaller than the wavelength is
# similar to a sub-wavelength grating... we independently engineer the effective and group
# indices... by controlling the widths and separations of individual segments" (§Coupled-mode
# description). The segmented WGB behaves like a lower-index, lower-group-index waveguide than
# the solid WGA — the paper states this is "essential for a good extinction ratio between the
# two output ports," since phase matching at a single λ_C together with a *large* group-index
# mismatch elsewhere is what gives a sharp, well-defined cutoff (§Power roll-off).
#
# ── Design sign convention (from the paper) ──────────────────────────────────────────────────
# Fig. 2a: "δ(λ < λ_C) > 0 and δ(λ > λ_C) < 0" for δ = β_A − β_B. Since dβ/dλ = −2π n_g/λ² < 0,
# a *decreasing* δ(λ) (as here) means d(β_A−β_B)/dλ ∝ −(n_gA − n_gB) < 0, i.e. **n_gA > n_gB** —
# consistent with "WGB possesses spectral characteristics equivalent to a waveguide with a
# smaller refractive index core, resulting in a smaller group index" (§Coupled-mode description).
# We therefore reward designs with n_gA − n_gB > 0, matching the paper's sign.
#
# ── Stage 1: phase-matching + maximum group-index mismatch (AD, Enzyme) ──────────────────────
# Design DOF p = (w_A, w_B, g_B). Minimise
#   L₁(p) = (n_A(λ_C,p) − n_B(λ_C,p))² − β·(n_gA(λ_C,p) − n_gB(λ_C,p))
# with **OptiMode's automatic differentiation**: each of n_A, n_B, n_gA, n_gB and their p-gradients
# come from a single Enzyme reverse-mode pass across the geometry→dielectric→eigensolve pipeline
# (`geom_value_grad`, designer_common.jl). Minimizing L₁ drives the crossing to λ_C while, among
# the crossing-satisfying designs, preferring the largest group-index mismatch in the paper's
# sign — i.e. the sharpest, most clearly-defined single-wavelength cutoff.
#
# ── Stage 2: adiabatic taper length (EME) ────────────────────────────────────────────────────
# With the optimized cross-section fixed, build the actual two-waveguide coupled structure and
# an EME cascade of `cfg.n_cells` cells over which the WGA–WGB edge gap widens from a strongly-
# coupled interaction value to a well-separated output value (the paper's "section ③"). We
# search for the *shortest* taper length whose EME-computed quasi-even → quasi-odd mode mixing
# (`power_coupling`, the exact non-adiabatic loss) stays below a target at both a short-pass and
# a long-pass test wavelength — i.e. the shortest adiabatic (low-loss) transition, following the
# paper's own Fig. 3b–d methodology of choosing section lengths from transmission-vs-length
# curves.
#
# Settings (see examples/README.md): --n-freqs (post-optimization crossing sweep, default 11),
# --n-cells (EME taper cell count, default 6), --resolution-scale / --domain-scale (grid).
#
# Run:  julia --project=. examples/designer_dichroic_si3n4.jl   (needs CairoMakie)
#       julia --project=. examples/designer_dichroic_si3n4.jl --resolution-scale=1.5

include(joinpath(@__DIR__, "designer_common.jl"))
include(joinpath(@__DIR__, "eme_reproductions_common.jl"))
using CairoMakie

cfg = example_settings(n_freqs=11, n_cells=6)
solver = KrylovKitEigsolve()
λC_target = 1.00                              # NEW cutoff target (µm)
t = 0.40                                       # uniform Si₃N₄ thickness for WGA & WGB (µm)
mats = [Si₃N₄, SiO₂]
mv = matvals_builder(mats; air=false)         # Si₃N₄ cores buried in SiO₂
grid = mk_grid(cfg, 6.0, 3.0, 48, 30)
om_C = 1 / λC_target

# --- geometry: solid WGA vs. 3-segment (rail) WGB, both isolated for stage 1 ------------------
rail(cx, w) = MaterialShape(Cuboid([cx, 0.0], [w, t], [1.0 0.0; 0.0 1.0]), 1)
wgA_shapes(p) = (rail(0.0, p[1]),)                                    # solid WGA, width p[1]
wgB_shapes(p) = (w = p[2]; g = p[3]; (rail(-(w + g), w), rail(0.0, w), rail(w + g, w)))  # 3-rail WGB
mindsA = (1, 2); mindsB = (1, 1, 1, 2)

nA_of(p, λ) = neff_of(diel_p(wgA_shapes, mv, mindsA, grid, p, 1/λ)[1], 1/λ, grid, solver)
nB_of(p, λ) = neff_of(diel_p(wgB_shapes, mv, mindsB, grid, p, 1/λ)[1], 1/λ, grid, solver)

# --- Stage 1: AD-optimize p=(w_A, w_B, g_B) for crossing at λ_C + max n_gA−n_gB ---------------
# A single soft-penalty loss r² − β·Δn_g has a structural equilibrium bias: near the crossing
# (r→0) the r² gradient vanishes while the constant −β·∇Δn_g term does not, so a fixed-β run
# settles at some small but nonzero r rather than the crossing itself. We instead warm-start in
# two phases — (1a) minimize the crossing condition alone (β=0) to lock onto λ_C precisely, then
# (1b) a short, small-step refinement of the *combined* loss from that crossing, which nudges
# Δn_g in the paper's sign without re-opening the gap. Both phases share the same AD machinery.
const β_gvm = 0.002
lo1, hi1 = [0.20, 0.10, 0.10], [0.40, 0.18, 0.18]
"Loss and Enzyme gradient of the pure crossing condition L(p) = (n_A−n_B)² at λ_C."
function loss_grad_crossing(p)
    nA, gnA = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), wgA_shapes, mv, mindsA, grid, p, om_C)
    nB, gnB = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), wgB_shapes, mv, mindsB, grid, p, om_C)
    r = nA - nB
    (r^2, 2r .* (gnA .- gnB))
end
"Loss and Enzyme gradient of L₁(p) = (n_A−n_B)² − β·(n_gA−n_gB) at λ_C."
function loss_grad_stage1(p)
    nA, gnA   = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), wgA_shapes, mv, mindsA, grid, p, om_C)
    nB, gnB   = geom_value_grad((ei, de) -> neff_of(ei, om_C, grid, solver), wgB_shapes, mv, mindsB, grid, p, om_C)
    ngA, gngA = geom_value_grad((ei, de) -> ng_of(ei, de, om_C, grid, solver), wgA_shapes, mv, mindsA, grid, p, om_C)
    ngB, gngB = geom_value_grad((ei, de) -> ng_of(ei, de, om_C, grid, solver), wgB_shapes, mv, mindsB, grid, p, om_C)
    r = nA - nB
    L = r^2 - β_gvm * (ngA - ngB)
    g = 2r .* (gnA .- gnB) .- β_gvm .* (gngA .- gngB)
    (L, g)
end

p0 = [0.28, 0.14, 0.12]                      # start: w_A, w_B(seg), g_B(rail gap) [µm]
@printf("== dichroic designer: Si₃N₄ solid-WGA / segmented-WGB, cutoff target λ_C=%.0f nm ==\n", 1e3λC_target)
@printf("start (w_A,w_B,g_B)=(%.3f,%.3f,%.3f) µm: n_A=%.4f n_B=%.4f\n",
        p0..., nA_of(p0, λC_target), nB_of(p0, λC_target))
println("-- phase 1a: lock the crossing (β=0) --")
res0 = optimize_design(loss_grad_crossing, p0; lo=lo1, hi=hi1, iters=40, lr=0.01)
println("-- phase 1b: refine for max Δn_g near the crossing (small steps) --")
res1 = optimize_design(loss_grad_stage1, res0.p; lo=lo1, hi=hi1, iters=12, lr=0.002)
p★ = res1.p
w_A★, w_B★, g_B★ = p★
nA★, ngA★ = nA_of(p★, λC_target), ng_of(diel_p(wgA_shapes, mv, mindsA, grid, p★, om_C)[1:2]..., om_C, grid, solver)
nB★, ngB★ = nB_of(p★, λC_target), ng_of(diel_p(wgB_shapes, mv, mindsB, grid, p★, om_C)[1:2]..., om_C, grid, solver)
@printf("optimized (w_A,w_B,g_B)=(%.3f,%.3f,%.3f) µm: n_A=%.4f n_B=%.4f  n_gA-n_gB=%+.3f (paper sign: >0)\n",
        p★..., nA★, nB★, ngA★ - ngB★)

# --- dispersion crossing before / after --------------------------------------------------------
λs = collect(range(0.80, 1.30; length=cfg.n_freqs))
nA0, nB0 = [nA_of(p0, λ) for λ in λs], [nB_of(p0, λ) for λ in λs]
nA1, nB1 = [nA_of(p★, λ) for λ in λs], [nB_of(p★, λ) for λ in λs]
λC0, λC★ = crossing_wavelength(λs, nA0, nB0), crossing_wavelength(λs, nA1, nB1)
@printf("cutoff: start λ_C=%s µm → optimized λ_C=%s µm (target %.3f)\n",
        isnan(λC0) ? "—" : string(round(λC0, digits=3)), isnan(λC★) ? "—" : string(round(λC★, digits=3)), λC_target)

# --- Stage 2: EME adiabatic taper — shortest length keeping mode-mixing loss low --------------
println("== Stage 2: EME adiabatic taper length search (WGA–WGB coupling gap) ==")
g_start, g_end = 0.30, 2.00                    # coupled (interaction) → separated (output) gap [µm]

"Coupled 4-shape (WGA + 3-segment WGB) cross-section at WGA–WGB edge-to-edge gap `gap`."
function coupled_shapes(gap)
    xA = -(gap / 2 + w_A★ / 2)
    xB = gap / 2 + w_B★ / 2
    (rail(xA, w_A★), rail(xB, w_B★), rail(xB + w_B★ + g_B★, w_B★), rail(xB + 2 * (w_B★ + g_B★), w_B★))
end
minds_AB = (1, 1, 1, 1, 2)

"Non-adiabatic loss (quasi-even → quasi-odd mode mixing) of a K-cell, length-`L` gap taper at ω."
function adiabatic_loss(L, ω; K=cfg.n_cells)
    edges = collect(range(0.0, L; length=K + 1))
    gaps = collect(range(g_start, g_end; length=K))
    cells = [Cell(i, (edges[i]+edges[i+1])/2, edges[i+1]-edges[i],
                  CrossSection(collect(MaterialShape, coupled_shapes(gaps[i])), collect(Int, minds_AB)))
             for i in 1:K]
    res = eme(cells, mats, ω, grid, solver; nev=2, k_tol=1e-7)
    power_coupling(res; in_mode=1, out_mode=2)
end

λ_short, λ_long = λC_target - 0.15, λC_target + 0.15
# Loss decreases with L as expected for an adiabatic transition, but is not perfectly monotonic
# at very long L (nev=2 mode tracking becomes less reliable over many cells); `findfirst` below
# only needs the *first* L meeting tolerance, so this doesn't affect the reported L★.
Ls = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
losses = Float64[]
for L in Ls
    ℓ = max(adiabatic_loss(L, 1/λ_short), adiabatic_loss(L, 1/λ_long))
    push!(losses, ℓ)
    @printf("  L=%6.1f µm  max(mode-mixing loss)=%.4e\n", L, ℓ)
    flush(stdout)
end
adiabatic_tol = 0.01
i_ok = findfirst(≤(adiabatic_tol), losses)
L★ = i_ok === nothing ? Ls[end] : Ls[i_ok]
@printf("shortest adiabatic taper (loss ≤ %.0e): L★ = %.1f µm%s\n",
        adiabatic_tol, L★, i_ok === nothing ? "  (tolerance not reached within Ls — using longest tested)" : "")

# --- plots ------------------------------------------------------------------------------------
fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="Adam iteration", ylabel="loss  (log scale)", yscale=log10,
    title=@sprintf("Stage 1: AD-optimized crossing + max Δn_g (Si₃N₄, %.0f nm)", 1e3λC_target))
lines!(ax1, 1:length(res0.history), max.(res0.history, 1e-12), color=:dodgerblue, linewidth=2, label="1a: (n_A−n_B)²")
lines!(ax1, length(res0.history) .+ (1:length(res1.history)), max.(abs.(res1.history), 1e-12),
    color=:crimson, linewidth=2, label="1b: |(n_A−n_B)² − β·Δn_g|  (loss is negative ⇒ Δn_g improving)")
vlines!(ax1, [length(res0.history) + 0.5], color=:gray, linestyle=:dot)
axislegend(ax1, position=:rt, labelsize=9)
ax2 = Axis(fig1[1, 2], xlabel="wavelength (µm)", ylabel="effective index",
    title="WGA/WGB mode crossing: start vs optimized")
lines!(ax2, λs, nB0, color=:black, linewidth=2, linestyle=:dash, label="WGB start")
lines!(ax2, λs, nA0, color=:dodgerblue, linewidth=2, linestyle=:dash, label="WGA start")
lines!(ax2, λs, nB1, color=:gray30, linewidth=2, label="WGB optimized")
lines!(ax2, λs, nA1, color=:crimson, linewidth=2, label="WGA optimized")
vlines!(ax2, [λC_target], color=:gray, linestyle=:dot)
axislegend(ax2, position=:rt, labelsize=10)
save(joinpath(OUTDIR, "designer_dichroic_si3n4_stage1.png"), fig1)

fig2 = Figure(size=(560, 380))
ax3 = Axis(fig2[1, 1], xlabel="taper length L (µm)", ylabel="quasi-even → quasi-odd mode mixing",
    title="Stage 2: adiabatic taper length search (EME)", xscale=log10, yscale=log10)
scatterlines!(ax3, Ls, max.(losses, 1e-12), color=:seagreen, linewidth=2, marker=:circle)
hlines!(ax3, [adiabatic_tol], color=:gray, linestyle=:dash, label=@sprintf("tolerance %.0e", adiabatic_tol))
vlines!(ax3, [L★], color=:crimson, linestyle=:dot, label=@sprintf("L★=%.0f µm", L★))
axislegend(ax3, position=:rt)
save(joinpath(OUTDIR, "designer_dichroic_si3n4_stage2.png"), fig2)

println("saved: designer_dichroic_si3n4_stage1.png, designer_dichroic_si3n4_stage2.png → ", OUTDIR)
