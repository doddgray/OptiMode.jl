# AD-driven DESIGNER — χ³ dispersion engineering on a NEW stack and NEW wavelength target.
#
# Applies the modal-dispersion / GVD workflow of the reproduction examples
# (tantala_gvd_black2021.jl, si3n4_cw_opa_riemensberger2022.jl) to a user-controlled
# **air-clad tantala (Ta₂O₅) core on SiO₂** stack and a NEW target: a **zero-dispersion point
# (β₂ = 0) at 1.3 µm** — the anomalous-GVD edge a χ³ OPA / soliton comb needs, shifted well
# below the reproduction example's ~1.05-µm ZDW.
#
# The design DOF is the core cross-section (width, height) p = (w, h). We minimise
# L(p) = β₂(p; 1.3 µm)²  with **OptiMode's automatic differentiation**: the geometry gradient of
# β₂ keeps the high-dimensional geometry sensitivity as exact AD (hybrid ForwardDiff∘Zygote on
# n_g) and finite-differences only the scalar ω-derivative (β₂ = ∂n_g/∂ω), per Gray, West & Ram
# (2024). Adam then drives β₂ → 0 at 1.3 µm and the ZDW onto the target.
#
# Run:  julia --project=. examples/designer_dispersion_tantala_1p3um.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "designer_common.jl"))
using CairoMakie

solver = KrylovKitEigsolve()
λ0 = 1.30                                      # NEW target: zero-GVD at 1.30 µm
mats = [Ta₂O₅, SiO₂]
mv = matvals_builder(mats; air=true)           # tantala core on SiO₂ substrate, air-clad
grid = Grid(5.0, 4.0, 40, 32)
# tantala core (w, h) sitting on a SiO₂ substrate (core bottom at y=0), air top + sides
geomfn(p) = (MaterialShape(Cuboid([0.0, p[2]/2], [p[1], p[2]], [1.0 0.0; 0.0 1.0]), 1),
             MaterialShape(Cuboid([0.0, -1.5], [100.0, 3.0], [1.0 0.0; 0.0 1.0]), 2))
minds = (1, 2, 3)

"β₂ (fs²/mm) and its geometry gradient at the target wavelength."
β2_grad(p) = gvd_value_grad(geomfn, mv, minds, grid, p, 1/λ0, solver)
"Loss L(p) = β₂² and its gradient 2β₂·∇β₂."
function loss_grad(p)
    β2, gβ2 = β2_grad(p)
    (β2^2, 2β2 .* gβ2)
end

p0 = [0.9, 0.7]
@printf("== χ³ dispersion designer: Ta₂O₅ on SiO₂ (air-clad), zero-GVD (β₂=0) target at %.2f µm ==\n", λ0)
@printf("start (w,h)=(%.3f,%.3f) µm: β₂=%+.1f fs²/mm\n", p0[1], p0[2], β2_grad(p0)[1])
res = optimize_design(loss_grad, p0; lo=[0.5, 0.4], hi=[2.5, 1.2], iters=75, lr=0.06)
p★ = res.p
@printf("optimized (w,h)=(%.3f,%.3f) µm: β₂=%+.1f fs²/mm (target 0)\n", p★[1], p★[2], β2_grad(p★)[1])

# --- β₂(λ) before / after: the ZDW moves to the target ---------------------------------
λs = collect(range(1.0, 1.7; length=9))
disp(p) = [gvd_value_grad(geomfn, mv, minds, grid, p, 1/λ, solver)[1] for λ in λs]
β2_0 = disp(p0); β2_★ = disp(p★)
zdw(b) = (for i in 1:length(λs)-1; sign(b[i]) != sign(b[i+1]) && return λs[i]-b[i]*(λs[i+1]-λs[i])/(b[i+1]-b[i]); end; NaN)
z0, z★ = zdw(β2_0), zdw(β2_★)
@printf("zero-dispersion wavelength: start %s µm → optimized %s µm (target %.2f)\n",
        isnan(z0) ? "—" : string(round(z0,digits=3)), isnan(z★) ? "—" : string(round(z★,digits=3)), λ0)

fig = Figure(size=(920, 340))
ax1 = Axis(fig[1, 1], xlabel="Adam iteration", ylabel="loss β₂²  (fs²/mm)²",
    title=@sprintf("AD-optimized zero-GVD design (Ta₂O₅, %.2f µm)", λ0), yscale=log10)
lines!(ax1, 1:length(res.history), res.history, color=:crimson, linewidth=2)
ax2 = Axis(fig[1, 2], xlabel="wavelength (µm)", ylabel="GVD β₂ (fs²/mm)",
    title=@sprintf("β₂(λ): start vs optimized  (w,h)=(%.2f,%.2f)", p★[1], p★[2]))
lines!(ax2, λs, β2_0, color=:dodgerblue, linewidth=2, linestyle=:dash, label="start")
lines!(ax2, λs, β2_★, color=:crimson, linewidth=2, label="optimized")
hlines!(ax2, [0.0], color=:gray, linestyle=:dash); vlines!(ax2, [λ0], color=:gray, linestyle=:dot)
axislegend(ax2, position=:rt)
save(joinpath(OUTDIR, "designer_dispersion_tantala_1p3um.png"), fig)
println("saved: designer_dispersion_tantala_1p3um.png → ", OUTDIR)
