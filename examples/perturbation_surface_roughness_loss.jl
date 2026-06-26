# Sidewall-roughness scattering loss by the Payne–Lacey model.
#
# Reproduces the canonical experimental facts of roughness-limited loss in high-index-
# contrast waveguides (Payne & Lacey, Opt. Quantum Electron. 26, 977 (1994);
# Lee et al., Opt. Lett. 26, 1888 (2001); Melati et al., Adv. Opt. Photon. 6, 156 (2014)):
#   • α ∝ σ²        (loss scales with roughness variance)
#   • α ∝ (n₁²−n₂²)² (index-contrast dependence)
#   • α ∝ ~λ⁻³      (strong wavelength dependence; steeper for high contrast)
# For an SOI slab (n₁=3.476, n₂=1.444) with AFM-measured σ:Lc = 2 nm:50 nm the model
# gives loss in the dB/cm range characteristic of as-fabricated Si waveguides.
#
# Run:  julia --project=. examples/perturbation_surface_roughness_loss.jl

using OptiMode.ModePerturbations: payne_lacey_slab_loss
using Printf
using CairoMakie

n1, n2, neff, d, Lc = 3.476, 1.444, 2.7, 0.11, 0.05    # SOI 220 nm slab, μm
σs = (0.5:0.1:5.0) .* 1e-3                              # rms roughness 0.5–5 nm
λs = 1.2:0.05:2.0

α_vs_σ(λ) = [payne_lacey_slab_loss(; σ=σ, Lc=Lc, λ=λ, d=d, n1=n1, n2=n2, neff=neff) for σ in σs]
α_vs_λ(σ) = [payne_lacey_slab_loss(; σ=σ, Lc=Lc, λ=λ, d=d, n1=n1, n2=n2, neff=neff) for λ in λs]

α2 = payne_lacey_slab_loss(; σ=2e-3, Lc=Lc, λ=1.55, d=d, n1=n1, n2=n2, neff=neff)
@printf("SOI slab, σ=2 nm, Lc=50 nm, 1550 nm:  α = %.2f dB/cm\n", α2)
@printf("σ² scaling: α(4 nm)/α(2 nm) = %.3f (expect 4.0)\n",
    payne_lacey_slab_loss(; σ=4e-3, Lc=Lc, λ=1.55, d=d, n1=n1, n2=n2, neff=neff) / α2)

fig = Figure(size=(760, 320))
ax1 = Axis(fig[1, 1], xlabel="rms roughness σ (nm)", ylabel="loss α (dB/cm)",
    title="Payne–Lacey: α ∝ σ²  (SOI, 1550 nm)")
lines!(ax1, collect(σs) .* 1e3, α_vs_σ(1.55), color=:dodgerblue)
scatter!(ax1, [2.0], [α2], color=:black, markersize=10)
text!(ax1, 2.1, α2, text=@sprintf("σ=2 nm → %.1f dB/cm", α2), align=(:left, :center))

ax2 = Axis(fig[1, 2], xlabel="wavelength λ (μm)", ylabel="loss α (dB/cm)",
    title="wavelength dependence (σ=2 nm)", yscale=log10)
lines!(ax2, collect(λs), α_vs_λ(2e-3), color=:seagreen, label="Payne–Lacey")
lines!(ax2, collect(λs), α_vs_λ(2e-3)[1] .* (λs[1] ./ λs) .^ 3, color=:grey,
    linestyle=:dash, label="∝ λ⁻³")
axislegend(ax2)
out = joinpath(@__DIR__, "perturbation_output", "surface_roughness_payne_lacey.png")
save(out, fig)
println("saved ", out)
