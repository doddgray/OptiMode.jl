# χ⁽²⁾ second-harmonic-generation normalized efficiency of a thin-film lithium niobate
# (TFLN) waveguide, from the nonlinear mode-overlap integral of the fundamental (1550 nm)
# and second-harmonic (775 nm) TE₀₀ modes.
#
# Reproduces the few-thousand-%/W/cm² scale of nanophotonic PPLN-on-insulator waveguides
# (Wang et al., Optica 5, 1438 (2018): 2600 %/W/cm² measured, ~5000 theory; overlap
# formalism of Luo et al., Optica 5, 1006 (2018)). Uses d_eff = (2/π)·d₃₃ ≈ 17 pm/V for
# first-order quasi-phase-matched TE₀₀→TE₀₀ SHG via d₃₃ (x-cut, extraordinary in-plane).
#
# Run:  julia --project=. examples/perturbation_shg_efficiency.jl

using OptiMode
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid
using OptiMode.ModePerturbations: shg_normalized_efficiency, shg_effective_area,
    shg_overlap_factor
using LinearAlgebra, Printf
using CairoMakie

solver = KrylovKitEigsolve()
Ry = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]               # x-cut: extraordinary axis in-plane
LN = rotate(LiNbO₃, Ry; name=:LN_xcut)
mats = [LN, SiO₂]
fε, _ = _f_ε_mats(mats, (:ω,))
air = vcat(vec(Matrix(1.0I, 3, 3)), zeros(18))
matvals(om) = hcat(fε([om]), air)
grid = Grid(5.0, 4.0, 110, 88)
mindsS = (1, 2, 3)
deff = 2 / π * 27e-12                                       # (2/π)·d₃₃, first-order QPM

function te00(om, geom)
    sm = smooth_ε(geom, matvals(om), mindsS, grid)
    ei = sliceinv_3x3(copy(selectdim(sm, 3, 1))); de = copy(selectdim(sm, 3, 2))
    ε = sliceinv_3x3(copy(ei))
    kk, ee = solve_k(om, copy(ei), grid, solver; nev=4, k_tol=1e-11)
    fr = [E_relpower_xyz(ε, E⃗(kk[i], copy(ee[i]), ei, de, grid; canonicalize=true, normalized=true))[1]
          for i in eachindex(ee)]
    i = argmax(fr)
    return kk[i], ee[i], ei, de
end

λF = 1.55; ωF = 1 / λF; ωSH = 2ωF
widths = 0.9:0.1:1.6
η0s = Float64[]
for w in widths
    t = 0.6
    geom = (MaterialShape(Cuboid([0.0, t / 2], [w, t], [1.0 0.0; 0.0 1.0]), 1),
        MaterialShape(Cuboid([0.0, -50.0], [200.0, 100.0], [1.0 0.0; 0.0 1.0]), 2))
    kF, evF, eiF, deF = te00(ωF, geom)
    kS, evS, eiS, deS = te00(ωSH, geom)
    EF = E⃗(kF, copy(evF), eiF, deF, grid; canonicalize=true, normalized=true)
    ES = E⃗(kS, copy(evS), eiS, deS, grid; canonicalize=true, normalized=true)
    mask = smooth_scalar(geom, [1.0, 0.0, 0.0], mindsS, grid)
    η0 = shg_normalized_efficiency(EF, ES, grid; deff=deff, λ1=λF * 1e-6,
        n1=kF / ωF, n2=kS / ωSH, chi2_mask=mask)
    push!(η0s, η0)
    @printf("w=%.2f μm: neff(FF)=%.3f neff(SH)=%.3f  η₀=%.0f %%/W/cm²\n", w, kF/ωF, kS/ωSH, η0)
end

fig = Figure(size=(560, 360))
ax = Axis(fig[1, 1], xlabel="waveguide width w (μm)", ylabel="η₀ (%/W/cm²)",
    title="TFLN PPLN SHG normalized efficiency (1550→775 nm)")
scatterlines!(ax, collect(widths), η0s, color=:teal)
hlines!(ax, [2600.0], color=:grey, linestyle=:dash)
text!(ax, 0.92, 2750, text="Wang 2018 measured 2600 %/W/cm²", color=:grey, fontsize=11)
out = joinpath(@__DIR__, "perturbation_output", "shg_efficiency_TFLN.png")
save(out, fig)
println("saved ", out)
