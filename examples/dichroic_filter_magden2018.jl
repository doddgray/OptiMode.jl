# EME reproduction of the transmissive silicon dichroic filter of
#
#   E. S. Magden, N. Li, M. Raval, C. V. Poulton, A. Ruocco, N. Singh, D. Vermeulen,
#   E. P. Ippen, L. A. Kolodziejski, M. R. Watts, "Transmissive silicon photonic dichroic
#   filters with spectrally selective waveguides," Nature Communications 9, 3009 (2018).
#
# Device (Fig. 1b): a 220-nm SOI, SiO₂-clad spectrally-selective coupler — a solid silicon
# strip WGA (width 318 nm) evanescently coupled across a 750-nm gap to a sub-wavelength
# segmented WGB (3 Si ridges of 250 nm separated by 100 nm gaps). WGA and WGB are phase-matched
# (β_A = β_B) at a single cutoff wavelength λ_C; an adiabatic transition then routes λ < λ_C to
# WGA (short-pass) and λ > λ_C to WGB (long-pass), giving an octave-wide single-cutoff filter.
#
# Following the MEOW-fork example (examples/papers/magden2018_dichroic.py) but with OptiMode:
#   (1) MODE SOLVING + DISPERSION: isolated n_eff(λ) of WGA and WGB across > 1 octave.
#   (2) MODE-CROSSING: the cutoff λ_C where the WGA/WGB dispersion curves cross (β_A = β_B).
#   (3) DENSE TRANSMISSION SPECTRUM: the coupled quasi-even supermode's WGA power fraction —
#       the ideal adiabatic mode-evolution short-pass (WGA) / long-pass (WGB) response.
#   (4) supermode |E| profiles below / at / above cutoff (Fig. 1d: field shifts WGA → WGB).
#
# Settings (see examples/README.md): --n-freqs (dispersion/crossing sweep points, default 6),
# --n-dense (interpolated transmission-curve resolution, default 400), --resolution-scale /
# --domain-scale (grid).
#
# Run:  julia --project=. examples/dichroic_filter_magden2018.jl   (needs CairoMakie)
#       julia --project=. examples/dichroic_filter_magden2018.jl --n-freqs=10 --resolution-scale=1.5

include(joinpath(@__DIR__, "eme_reproductions_common.jl"))
using CairoMakie

cfg = example_settings(n_freqs=6, n_dense=400)
solver = KrylovKitEigsolve()
H = 0.22                                   # 220-nm SOI silicon thickness
wA = 0.318                                  # WGA solid-Si strip width
wB_seg, gB = 0.25, 0.10                     # WGB: 3 sub-wavelength segments of 250 nm / 100 nm gaps
wB = 3*wB_seg + 2*gB                         # total WGB extent = 0.95 µm
edge_gap = 0.75                              # WGA–WGB edge-to-edge coupling gap
xA = -(edge_gap/2 + wA/2)                    # centre WGA / WGB about x = 0
xB = +(edge_gap/2 + wB/2)
x_split = (xA + wA/2 + xB - wB/2)/2          # WGA|WGB divider (gap midpoint)
mats = [silicon, SiO₂]
mv = matvals_builder(mats; air=false)        # Si core, SiO₂ background (buried)
grid = mk_grid(cfg, 4.4, 2.2, 124, 62)

rect(cx, w) = MaterialShape(Cuboid([cx, 0.0], [w, H], [1.0 0.0; 0.0 1.0]), 1)
shapes_A = (rect(xA, wA),)                                        # WGA alone
shapes_B = ntuple(k -> rect(xB - wB/2 + wB_seg/2 + (k-1)*(wB_seg+gB), wB_seg), 3)   # WGB SWG alone
shapes_AB = (shapes_A..., shapes_B...)                            # coupled cross-section
minds_A = (1, 2); minds_B = (1, 1, 1, 2); minds_AB = (1, 1, 1, 1, 2)

# --- (1,2) isolated dispersion + mode crossing (cutoff) ---------------------------------
println("== Si dichroic filter: isolated WGA / WGB dispersion (>1 octave) ==")
λs = collect(range(1.40, 2.60; length=cfg.n_freqs))   # 1.35–2.65 µm ≈ 0.97 octave (moderate grid; scale via SLURM)
nA = zero(λs); nB = zero(λs); fracA = zero(λs)
for (j, λ) in enumerate(λs)
    ω = 1/λ
    nA[j] = supermodes(shapes_A, minds_A, mv, ω, grid, solver; nev=2)[1].neff
    nB[j] = supermodes(shapes_B, minds_B, mv, ω, grid, solver; nev=3)[1].neff
    sup = supermodes(shapes_AB, minds_AB, mv, ω, grid, solver; nev=4)
    fracA[j] = port_fraction(sup[1].E, grid, x_split)   # quasi-even supermode WGA fraction
    @printf("  λ=%.3f µm  n_A=%.4f  n_B=%.4f  (WGA frac of even supermode %.2f)\n", λ, nA[j], nB[j], fracA[j])
end
λC = crossing_wavelength(λs, nA, nB)
@printf("cutoff λ_C (β_A = β_B): %s µm\n", isnan(λC) ? "none in range" : string(round(λC; digits=3)))

# --- (3) dense >1-octave transmission spectrum -----------------------------------------
λdense = collect(range(λs[1], λs[end]; length=cfg.n_dense))
T_short, T_long = dichroic_spectrum(λs, fracA, λdense)

# --- (4) supermode profiles below / at / above cutoff ----------------------------------
λpix = isnan(λC) ? λs[cld(length(λs),2)] : λC
λ_show = (λpix - 0.35, λpix, λpix + 0.35)
Efields = [supermodes(shapes_AB, minds_AB, mv, 1/λ, grid, solver; nev=4)[1].E for λ in λ_show]

# --- plots ------------------------------------------------------------------------------
xc, yc = grid_coords(grid)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="effective index",
    title="isolated WGA / WGB dispersion + mode crossing")
lines!(ax1, λs, nA, color=:dodgerblue, linewidth=2, label="WGA (Si strip)")
lines!(ax1, λs, nB, color=:crimson, linewidth=2, label="WGB (SWG)")
isnan(λC) || vlines!(ax1, [λC], color=:gray, linestyle=:dash)
axislegend(ax1, position=:rt)
ax1b = Axis(fig1[1, 2], xlabel="wavelength (µm)", ylabel="transmission",
    title=isnan(λC) ? "dichroic filter response" : @sprintf("dichroic filter response (λ_C=%.0f nm)", 1e3λC))
lines!(ax1b, λdense, T_short, color=:dodgerblue, linewidth=2, label="short-pass → WGA")
lines!(ax1b, λdense, T_long, color=:crimson, linewidth=2, label="long-pass → WGB")
isnan(λC) || vlines!(ax1b, [λC], color=:gray, linestyle=:dash)
axislegend(ax1b, position=:rc)
save(joinpath(OUTDIR, "magden_dichroic_dispersion_transmission.png"), fig1)

fig2 = Figure(size=(1100, 320))
for (col, (E, λ)) in enumerate(zip(Efields, λ_show))
    lbl = col == 1 ? "λ < λ_C (in WGA)" : col == 2 ? "λ ≈ λ_C (hybrid)" : "λ > λ_C (in WGB)"
    ax = Axis(fig2[1, col], xlabel="x (µm)", ylabel="y (µm)", aspect=DataAspect(),
        title=@sprintf("%s  (%.0f nm)", lbl, 1e3λ))
    hm = heatmap!(ax, xc, yc, absE_norm(E), colormap=:turbo)
    for (cx, w) in ((xA, wA),); lines!(ax, [cx-w/2,cx+w/2,cx+w/2,cx-w/2,cx-w/2], [-H/2,-H/2,H/2,H/2,-H/2], color=:white, linewidth=1); end
    for k in 1:3; cx = xB - wB/2 + wB_seg/2 + (k-1)*(wB_seg+gB); lines!(ax, [cx-wB_seg/2,cx+wB_seg/2,cx+wB_seg/2,cx-wB_seg/2,cx-wB_seg/2], [-H/2,-H/2,H/2,H/2,-H/2], color=:white, linewidth=0.8); end
    xlims!(ax, -1.6, 1.9); ylims!(ax, -0.9, 0.9)
end
Colorbar(fig2[1, 4], colormap=:turbo, label="|E| (norm.)")
save(joinpath(OUTDIR, "magden_dichroic_supermodes.png"), fig2)

println("saved: magden_dichroic_dispersion_transmission.png, magden_dichroic_supermodes.png → ", OUTDIR)
