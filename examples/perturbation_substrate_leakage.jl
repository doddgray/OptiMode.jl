# Substrate-leakage loss: tunneling of a guided mode through a finite lower cladding into a
# high-index substrate, α = A·exp(−2γ_c t_clad) with γ_c = k₀√(neff²−n_clad²).
#
# Reproduces the buried-oxide design rule for SOI (Sridaran & Bhave, Opt. Express 18, 3850
# (2010); Bauters et al., Opt. Express 19, 3163 (2011)): thin BOX (<0.5 μm) gives >100
# dB/cm leakage while ≥1 μm BOX suppresses it below 1 dB/cm, with a loss e-folding length
# of ~90 nm of oxide for a Si strip at 1550 nm.
#
# Run:  julia --project=. examples/perturbation_substrate_leakage.jl

using OptiMode.ModePerturbations: substrate_leakage_loss
using Printf
using CairoMakie

λ, neff, n_clad, A = 1.55, 2.4, 1.444, 1.0e4    # SOI strip; prefactor A from a leaky solve/fit
t = 0.1:0.02:2.0
α = [substrate_leakage_loss(; neff=neff, n_clad=n_clad, t_clad=ti, λ=λ, prefactor=A) for ti in t]
γc = 2π / λ * sqrt(neff^2 - n_clad^2)
@printf("loss e-folding length 1/(2γ_c) = %.0f nm (Sridaran ~90 nm)\n", 1e3 / (2γc))
@printf("t_BOX=0.3 μm: %.1f dB/cm   t_BOX=1.0 μm: %.3f dB/cm\n",
    substrate_leakage_loss(; neff, n_clad, t_clad=0.3, λ, prefactor=A),
    substrate_leakage_loss(; neff, n_clad, t_clad=1.0, λ, prefactor=A))

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="lower-cladding (BOX) thickness t (μm)",
    ylabel="substrate-leakage loss (dB/cm)", yscale=log10,
    title="Substrate leakage ∝ exp(−2γ_c t)  (SOI strip, 1550 nm)")
lines!(ax, collect(t), α, color=:darkorange)
hlines!(ax, [1.0], color=:grey, linestyle=:dash)
text!(ax, 1.3, 2.0, text="1 dB/cm", color=:grey)
vlines!(ax, [1.0], color=:grey, linestyle=:dot)
out = joinpath(@__DIR__, "perturbation_output", "substrate_leakage_SOI.png")
save(out, fig)
println("saved ", out)
