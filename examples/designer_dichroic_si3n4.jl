# AD-driven DESIGNER — spectrally-selective EME dichroic filter on a NEW stack and NEW target.
#
# Applies the mode-crossing / dichroic-filter workflow of the reproduction example
# (dichroic_filter_magden2018.jl) to a user-controlled **Si₃N₄-on-SiO₂** stack (instead of Si
# SOI) and a NEW cutoff target in the near-IR: **λ_C = 1000 nm**. Dissimilar guides are made by
# thickness contrast (as in the MEOW fork's dichroic_designer_si3n4_thickness): a thick-narrow
# WGA (t_A = 400 nm) coupled to a thin-wide WGB (t_B = 200 nm). Their n_eff(λ) cross once — the
# dichroic cutoff — and the crossing wavelength is tuned by the WGA width.
#
# The design DOF is the WGA width w_A. We minimise  L(w_A) = (n_A(λ_C, w_A) − n_B(λ_C))²  with
# **OptiMode's automatic differentiation**: dn_A/dw_A is the hybrid ForwardDiff(geometry) ∘
# Zygote(adjoint eigensolve) gradient, driving Adam. The optimum places β_A = β_B exactly at
# the 1000-nm target.
#
# Run:  julia --project=. examples/designer_dichroic_si3n4.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "designer_common.jl"))
using CairoMakie

solver = KrylovKitEigsolve()
λC_target = 1.00                              # NEW cutoff target (µm)
tA, tB = 0.40, 0.20                           # WGA thick / WGB thin Si₃N₄ (thickness-contrast)
wB = 1.10                                      # WGB fixed width (thin, wide)
mats = [Si₃N₄, SiO₂]
mv = matvals_builder(mats; air=false)         # Si₃N₄ cores buried in SiO₂
grid = Grid(5.0, 3.0, 40, 30)

wgA(w) = (MaterialShape(Cuboid([0.0, 0.0], [w[1], tA], [1.0 0.0; 0.0 1.0]), 1),)
wgB     = (MaterialShape(Cuboid([0.0, 0.0], [wB, tB], [1.0 0.0; 0.0 1.0]), 1),)
mindsA = (1, 2); mindsB = (1, 2)

nA_of(w, λ) = neff_of(diel_p(wgA, mv, mindsA, grid, w, 1/λ)[1], 1/λ, grid, solver)
nB_of(λ)    = neff_of(diel_p((_)->wgB, mv, mindsB, grid, [0.0], 1/λ)[1], 1/λ, grid, solver)

nB_target = nB_of(λC_target)
"Loss and AD gradient of L(w_A) = (n_A(λ_C, w_A) − n_B(λ_C))²."
function loss_grad(w)
    nA, gA = geom_value_grad((ei, de) -> neff_of(ei, 1/λC_target, grid, solver), wgA, mv, mindsA, grid, w, 1/λC_target)
    r = nA - nB_target
    (r^2, 2r .* gA)
end

w0 = [0.45]
@printf("== dichroic designer: Si₃N₄ thickness-contrast coupler, cutoff target λ_C=%.0f nm ==\n", 1e3λC_target)
@printf("WGB (t=%.0f nm, w=%.2f µm): n_B(λ_C)=%.4f\n", 1e3tB, wB, nB_target)
@printf("start w_A=%.3f µm: n_A(λ_C)=%.4f\n", w0[1], nA_of(w0, λC_target))
res = optimize_design(loss_grad, w0; lo=[0.30], hi=[1.20], iters=18, lr=0.02)
wA★ = res.p[1]
@printf("optimized w_A=%.3f µm: n_A(λ_C)=%.4f  (n_B=%.4f)\n", wA★, nA_of([wA★], λC_target), nB_target)

# --- dispersion crossing before / after + cutoff ---------------------------------------
λs = collect(range(0.80, 1.30; length=11))
nB = [nB_of(λ) for λ in λs]
nA0 = [nA_of(w0, λ) for λ in λs]
nA★ = [nA_of([wA★], λ) for λ in λs]
xcross(nA_) = (d = nA_ .- nB; for i in 1:length(λs)-1; sign(d[i]) != sign(d[i+1]) && return λs[i]-d[i]*(λs[i+1]-λs[i])/(d[i+1]-d[i]); end; NaN)
λC0, λC★ = xcross(nA0), xcross(nA★)
@printf("cutoff: start λ_C=%s µm → optimized λ_C=%s µm (target %.3f)\n",
        isnan(λC0) ? "—" : string(round(λC0,digits=3)), isnan(λC★) ? "—" : string(round(λC★,digits=3)), λC_target)

fig = Figure(size=(920, 340))
ax1 = Axis(fig[1, 1], xlabel="Adam iteration", ylabel="loss (n_A − n_B)²",
    title=@sprintf("AD-optimized dichroic cutoff (Si₃N₄, %.0f nm)", 1e3λC_target), yscale=log10)
lines!(ax1, 1:length(res.history), res.history, color=:crimson, linewidth=2)
ax2 = Axis(fig[1, 2], xlabel="wavelength (µm)", ylabel="effective index",
    title="WGA/WGB mode crossing: start vs optimized")
lines!(ax2, λs, nB, color=:black, linewidth=2, label="WGB (fixed)")
lines!(ax2, λs, nA0, color=:dodgerblue, linewidth=2, linestyle=:dash, label="WGA start")
lines!(ax2, λs, nA★, color=:crimson, linewidth=2, label="WGA optimized")
vlines!(ax2, [λC_target], color=:gray, linestyle=:dot)
axislegend(ax2, position=:rt)
save(joinpath(OUTDIR, "designer_dichroic_si3n4.png"), fig)
println("saved: designer_dichroic_si3n4.png → ", OUTDIR)
