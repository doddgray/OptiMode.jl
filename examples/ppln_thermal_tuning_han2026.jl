# Electro-thermal QPM tuning of the reconfigurable PPLN χ² OPA of
#
#   G. Han et al., "On-chip electrically reconfigurable octave-bandwidth optical amplification
#   from visible to near-infrared," arXiv:2602.00246 (2026).
#
# Companion to ppln_reconfigurable_opa_han2026.jl. The paper's key knob (Fig. 1e) is *local
# electro-thermal tuning of the quasi-phase-matching condition*: heating the x-cut TFLN
# waveguide changes the LiNbO₃ index (thermo-optic dn/dT), which shifts the QPM phase-matched
# wavelength for a fixed poling period Λ, extending the gain coverage. This script reproduces,
# with OptiMode's temperature-dependent LiNbO₃ model + mode solver:
#   (1) the FH-band QPM phase mismatch Δk(λ) at several ΔT (the curves translate with heating);
#   (2) the phase-matched FH wavelength λ_PM vs ΔT (the electro-thermal tuning curve);
#   (3) the effective-index thermo-optic coefficients dn_eff/dT of the FH and SH modes.
#
# Settings (see examples/README.md): --n-freqs (dispersion-sweep points, default 3), --n-dense
# (Δk(λ) curve resolution, default 300), --resolution-scale / --domain-scale (grid).
#
# Run:  julia --project=. examples/ppln_thermal_tuning_han2026.jl   (needs CairoMakie)
#       julia --project=. examples/ppln_thermal_tuning_han2026.jl --n-freqs=7 --resolution-scale=1.5

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))
using OptiMode: rotate
using OptiMode.MaterialDispersion: LiNbO₃
using CairoMakie

cfg = example_settings(n_freqs=3, n_dense=300)
solver = KrylovKitEigsolve()
λF = 1.064; λS = λF/2
w, etch, film = 1.40, 0.35, 0.70
T0 = 25.0                                   # reference temperature (°C)

const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
LiNbO₃_xcut = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
mvT = matvals_builder_T([LiNbO₃_xcut, SiO₂]; air=true)   # (ω, T) → material values
mv_at(T) = (ω -> mvT(ω, T))                               # freeze temperature for a mode solve
shapes, minds = ridge_on_slab(w, etch, film)
slab = film - etch
grid = mk_grid(cfg, 6.0, 4.0, 96, 64)

# --- FH & SH band dispersion at the reference temperature -------------------------------
println("== TFLN QPM thermal tuning: dispersion at T₀ = $(T0) °C ==")
λFH = range(1.02, 1.12; length=cfg.n_freqs)
swF = sweep_dispersion(shapes, minds, mv_at(T0), λFH, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=6)
swS = sweep_dispersion(shapes, minds, mv_at(T0), λFH ./ 2, grid, solver; pol=:TE, w=w, h=film, yc=slab, nev=9)
mF0 = solve_fundamental(shapes, minds, mv_at(T0), 1/λF, grid, solver; nev=6, pol=:TE, w=w, h=film, yc=slab)
mS0 = solve_fundamental(shapes, minds, mv_at(T0), 1/λS, grid, solver; nev=9, pol=:TE, w=w, h=film, yc=slab)
Λ = poling_period(mF0.neff, mS0.neff, λF)
@printf("design: n_FF=%.4f  n_SH=%.4f  Λ=%.3f µm\n", mF0.neff, mS0.neff, Λ)

# --- effective-index thermo-optic coefficients dn_eff/dT --------------------------------
ΔTref = 60.0
mF1 = solve_fundamental(shapes, minds, mv_at(T0+ΔTref), 1/λF, grid, solver; nev=6, pol=:TE, w=w, h=film, yc=slab)
mS1 = solve_fundamental(shapes, minds, mv_at(T0+ΔTref), 1/λS, grid, solver; nev=9, pol=:TE, w=w, h=film, yc=slab)
dnFF_dT = (mF1.neff - mF0.neff) / ΔTref
dnSH_dT = (mS1.neff - mS0.neff) / ΔTref
@printf("dn_eff/dT:  FH = %.2e /K,  SH = %.2e /K\n", dnFF_dT, dnSH_dT)

# --- QPM phase-mismatch Δk(λ; ΔT) and phase-matched wavelength λ_PM(ΔT) -----------------
# Δk(λ) = (4π/λ)(n_SH(λ/2) − n_FF(λ)) − 2π/Λ, with n(λ,T) ≈ n(λ,T₀) + (dn/dT)·ΔT.
nFF(λ, ΔT) = interp1(swF.λ, swF.neff, λ) + dnFF_dT*ΔT
nSH(λ, ΔT) = interp1(swS.λ, swS.neff, λ/2) + dnSH_dT*ΔT
Δk(λ, ΔT) = (4π/λ)*(nSH(λ, ΔT) - nFF(λ, ΔT)) - 2π/Λ     # rad/µm
function λpm(ΔT; lo=λFH[1], hi=λFH[end])
    f(λ) = Δk(λ, ΔT)
    fa, fb = f(lo), f(hi)
    sign(fa) == sign(fb) && return NaN
    for _ in 1:60
        m = (lo+hi)/2; sign(f(m)) == sign(fa) ? (lo=m; fa=f(m)) : (hi=m)
    end
    (lo+hi)/2
end

ΔTs = 0.0:20.0:120.0
λpms = [λpm(ΔT) for ΔT in ΔTs]
for (ΔT, λp) in zip(ΔTs, λpms)
    @printf("  ΔT=%3.0f K → λ_PM = %s nm\n", ΔT, isnan(λp) ? "—" : string(round(1e3λp; digits=1)))
end
dλ_dT = (λpms[end]-λpms[1])/(ΔTs[end]-ΔTs[1])
@printf("thermal tuning slope dλ_PM/dT ≈ %.2f nm/K\n", 1e3*dλ_dT)

# --- plots ------------------------------------------------------------------------------
λdense = range(λFH[1], λFH[end]; length=cfg.n_dense)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="FH wavelength (µm)", ylabel="QPM Δk (rad/mm)",
    title=@sprintf("QPM phase mismatch vs ΔT (Λ=%.2f µm)", Λ))
for (i, ΔT) in enumerate((0.0, 40.0, 80.0, 120.0))
    lines!(ax1, λdense, [1e3*Δk(λ, ΔT) for λ in λdense], linewidth=2,
        color=get(cgrad(:thermal), i/4), label=@sprintf("ΔT=%.0f K", ΔT))
end
hlines!(ax1, [0.0], color=:gray, linestyle=:dash); axislegend(ax1, position=:rt)
ax1b = Axis(fig1[1, 2], xlabel="ΔT (K)", ylabel="phase-matched λ_FH (nm)",
    title=@sprintf("electro-thermal QPM tuning (%.2f nm/K)", 1e3*dλ_dT))
scatterlines!(ax1b, collect(ΔTs), 1e3 .* λpms, color=:crimson, linewidth=2, markersize=8)
save(joinpath(OUTDIR, "ppln_thermal_tuning.png"), fig1)

println("saved: ppln_thermal_tuning.png → ", OUTDIR)
