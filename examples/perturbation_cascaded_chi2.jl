# Cascaded-χ⁽²⁾ effective Kerr nonlinearity: the self-focusing/defocusing index induced by
# phase-mismatched second-harmonic generation.
#
# Reproduces the DeSalvo et al. measurement (Opt. Lett. 17, 28 (1992)) in KTP at 1064 nm:
# the effective nonlinear index n₂,casc = −2ω₁d_eff²/(c²ε₀n₁²n₂Δk) changes sign across phase
# matching (Δk>0 self-defocusing, Δk<0 self-focusing) and reaches |n₂,eff| ≈ 2×10⁻¹⁴ cm²/W
# near phase matching. (Stegeman, Opt. Quantum Electron. 28, 1691 (1996).)
#
# Run:  julia --project=. examples/perturbation_cascaded_chi2.jl

using OptiMode.ModePerturbations: cascaded_chi2_n2_eff
using Printf
using CairoMakie

deff, λ1, n1, n2 = 3.1e-12, 1.064e-6, 1.74, 1.79      # KTP type-II, DeSalvo 1992
Δks = vcat(-3000.0:50.0:-100.0, 100.0:50.0:3000.0)    # phase mismatch (1/m)
n2c = [cascaded_chi2_n2_eff(; deff=deff, λ1=λ1, n1=n1, n2=n2, Δk=Δk) * 1e4 for Δk in Δks]  # m²/W→cm²/W

@printf("Δk=+500 1/m: n₂,casc = %+.2e cm²/W (self-defocusing)\n",
    cascaded_chi2_n2_eff(; deff, λ1, n1, n2, Δk=500.0) * 1e4)
@printf("Δk=−500 1/m: n₂,casc = %+.2e cm²/W (self-focusing)\n",
    cascaded_chi2_n2_eff(; deff, λ1, n1, n2, Δk=-500.0) * 1e4)

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="phase mismatch Δk = k₂ᵥ − 2kᵥ (1/m)",
    ylabel="effective n₂,casc (cm²/W)",
    title="Cascaded χ²: sign flips across phase matching (KTP, 1064 nm)")
lines!(ax, Δks[Δks .< 0], n2c[Δks .< 0], color=:crimson, label="Δk<0: self-focusing (n₂>0)")
lines!(ax, Δks[Δks .> 0], n2c[Δks .> 0], color=:navy, label="Δk>0: self-defocusing (n₂<0)")
hlines!(ax, [0.0], color=:black, linewidth=0.5)
hlines!(ax, [2e-14, -2e-14], color=:grey, linestyle=:dot)
text!(ax, -2800, 2.3e-14, text="DeSalvo 1992 peak ±2×10⁻¹⁴ cm²/W", align=(:left, :bottom), fontsize=11)
axislegend(ax, position=:rt)
ylims!(ax, -6e-14, 6e-14)
out = joinpath(@__DIR__, "perturbation_output", "cascaded_chi2_KTP.png")
save(out, fig)
println("saved ", out)
