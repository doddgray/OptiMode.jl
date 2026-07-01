# AD-driven DESIGNER — χ² quasi-phase-matching on a NEW stack and NEW wavelength target.
#
# Applies the χ² OPA/SHG modeling workflow of the reproduction examples
# (pplt_allband_opa_kuznetsov2026.jl, ppln_reconfigurable_opa_han2026.jl) to a user-controlled
# stack — an **MgO:LiNbO₃** rib on SiO₂ (air-clad), c-axis out of plane so the guided mode is
# in-plane isotropic — and a new target: **SHG quasi-phase-matching at a 1310-nm fundamental**
# (655-nm second harmonic) with a fabrication-preferred poling period Λ_target.
#
# The design degree of freedom is the rib top width w. First-order QPM needs
#     n_SH(w) − n_FF(w) = λ_FF / (2 Λ_target).
# We minimise  L(w) = (Δn(w) − λ_FF/(2Λ_target))²  with **OptiMode's automatic differentiation**:
# dΔn/dw = dn_SH/dw − dn_FF/dw is the hybrid ForwardDiff(geometry)∘Zygote(adjoint eigensolve)
# gradient (designer_common.jl), driving an Adam optimizer. No finite-difference geometry sweep.
#
# Run:  julia --project=. examples/designer_qpm_mgoln_1310.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "designer_common.jl"))
using OptiMode.MaterialDispersion: MgO_LiNbO₃
using CairoMakie

solver = KrylovKitEigsolve()
λF = 1.31; λS = λF/2                          # NEW target: 1310 nm FH → 655 nm SH
Λ_target = 4.0                                # fabrication-preferred poling period (µm)
target_Δn = λF / (2 * Λ_target)              # required n_SH − n_FF
film, slab, etch = 0.70, 0.30, 0.40          # 700-nm MgO:LiNbO₃, 400-nm etch (user stack)

mats = [MgO_LiNbO₃, SiO₂]
mv = matvals_builder(mats; air=true)         # MgO:LiNbO₃ core, SiO₂ box, air background
grid = Grid(6.0, 3.6, 40, 30)                # small grid → fast AD optimization
# rib of top width w[1] on the unetched slab, over a SiO₂ substrate; air above
geomfn(w) = (
    MaterialShape(Cuboid([0.0, slab + etch/2], [w[1], etch], [1.0 0.0; 0.0 1.0]), 1),
    MaterialShape(Cuboid([0.0, slab/2], [100.0, slab], [1.0 0.0; 0.0 1.0]), 1),
    MaterialShape(Cuboid([0.0, -1.0], [100.0, 2.0], [1.0 0.0; 0.0 1.0]), 2),
)
minds = (1, 1, 2, 3)

Δn_of(w) = neff_of(diel_p(geomfn, mv, minds, grid, w, 1/λS)[1], 1/λS, grid, solver) -
           neff_of(diel_p(geomfn, mv, minds, grid, w, 1/λF)[1], 1/λF, grid, solver)

"Loss and AD gradient of L(w) = (Δn(w) − target)²."
function loss_grad(w)
    nF, gF = geom_value_grad((ei, de) -> neff_of(ei, 1/λF, grid, solver), geomfn, mv, minds, grid, w, 1/λF)
    nS, gS = geom_value_grad((ei, de) -> neff_of(ei, 1/λS, grid, solver), geomfn, mv, minds, grid, w, 1/λS)
    Δn = nS - nF; r = Δn - target_Δn
    (r^2, 2r .* (gS .- gF))
end

w0 = [1.20]
@printf("== χ² QPM designer: MgO:LiNbO₃ rib, SHG %.0f→%.0f nm, target Λ=%.1f µm (Δn=%.4f) ==\n",
        1e3λF, 1e3λS, Λ_target, target_Δn)
@printf("start w=%.3f µm: Δn=%.4f  (Λ=%.3f µm)\n", w0[1], Δn_of(w0), λF/(2*Δn_of(w0)))
res = optimize_design(loss_grad, w0; lo=[0.6], hi=[2.2], iters=18, lr=0.03)
w★ = res.p[1]
Δn★ = Δn_of([w★]); Λ★ = λF / (2Δn★)
@printf("optimized w=%.3f µm: Δn=%.4f  → Λ=%.3f µm (target %.1f)\n", w★, Δn★, Λ★, Λ_target)

# --- Δn(w) landscape + poling period, marking start/optimum ------------------------------
ws = collect(range(0.7, 2.1; length=15))
Δns = [Δn_of([w]) for w in ws]
Λs = [λF/(2d) for d in Δns]

fig = Figure(size=(920, 340))
ax1 = Axis(fig[1, 1], xlabel="Adam iteration", ylabel="loss (Δn − target)²",
    title="AD-optimized QPM design (MgO:LiNbO₃, 1310 nm)", yscale=log10)
lines!(ax1, 1:length(res.history), res.history, color=:crimson, linewidth=2)
scatter!(ax1, [length(res.history)], [res.history[end]], color=:crimson, markersize=9)
ax2 = Axis(fig[1, 2], xlabel="rib top width w (µm)", ylabel="required poling period Λ (µm)",
    title=@sprintf("width → QPM period  (w★=%.3f µm, Λ★=%.2f µm)", w★, Λ★))
lines!(ax2, ws, Λs, color=:seagreen, linewidth=2)
hlines!(ax2, [Λ_target], color=:gray, linestyle=:dash, label=@sprintf("target Λ=%.1f µm", Λ_target))
scatter!(ax2, [w0[1]], [λF/(2*Δn_of(w0))], color=:black, markersize=9, label="start")
scatter!(ax2, [w★], [Λ★], color=:crimson, markersize=11, label="optimized")
axislegend(ax2, position=:rt)
save(joinpath(OUTDIR, "designer_qpm_mgoln_1310.png"), fig)
println("saved: designer_qpm_mgoln_1310.png → ", OUTDIR)
