# EME reproduction of the ultra-broadband TFLN wavelength combiner of
#
#   R. Kwolek, P. Thapalia, A. Tripathi, …, S. Fathpour, R. Nehra, "Ultra-broadband, Low-loss
#   Wavelength Combiners and Filters: Novel Designs and Experiments in Thin-film Lithium
#   Niobate," arXiv:2603.27034 (2026).
#
# Device: two coupled x-cut TFLN rib waveguides (300-nm film, ~100-nm etch → 200-nm slab,
# ~1.2 µm top width, 65° sidewall) forming a quasi-adiabatic directional coupler that combines
# the telecom fundamental (FH, 1550 nm) and its second harmonic (SH, 775 nm) — an octave-scale
# separation — into/out of separate ports. The FAQUAD design opens the gap g(z) along z so the
# supermodes evolve adiabatically at FH while staying weakly coupled at SH.
#
# Following the MEOW-fork example (examples/papers/kwolek2026_faquad.py) but with OptiMode:
#   (1) MODE SOLVING + DISPERSION: even/odd supermode indices of the two-ridge coupler across
#       > 1 octave (0.70–1.65 µm), and the coupling length L_c(λ)=λ/(2Δn_super).
#   (2) DENSE >1-OCTAVE TRANSMISSION SPECTRUM: directional-coupler bar/cross transmission
#       T_cross(λ)=sin²(πL·Δn_super(λ)/λ) — strong FH coupling (→ cross) vs weak SH coupling
#       (→ bar) gives the wavelength combiner response.
#   (3) EME validation: cascade the uniform coupler with OptiMode's `eme` (cross-section dedup
#       makes this cheap) and confirm the supermode S-matrix.
#   (4) even/odd supermode |E| profiles at FH and SH.
#
# Run:  julia --project=. examples/tfln_combiner_kwolek2026.jl   (needs CairoMakie)

include(joinpath(@__DIR__, "eme_reproductions_common.jl"))
using CairoMakie

solver = KrylovKitEigsolve()
film, slab, w = 0.30, 0.20, 1.20            # 300-nm film, 200-nm slab (100-nm etch), 1.2-µm width
etch = film - slab                           # 0.10 µm rib height
g_c = 0.80                                   # coupling gap (G_M = 0.8 µm, FAQUAD design)
λF, λS = 1.55, 0.775                          # FH / SH

const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]
LiNbO₃_xcut = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
mats = [LiNbO₃_xcut, SiO₂, Vacuum]
mv = matvals_builder(mats; air=false)        # LN, SiO₂ box, Vacuum background (air top)
grid = Grid(7.0, 3.0, 168, 90)

rib(cx) = MaterialShape(Cuboid([cx, slab + etch/2], [w, etch], [1.0 0.0; 0.0 1.0]), 1)
slabsh  = MaterialShape(Cuboid([0.0, slab/2], [200.0, slab], [1.0 0.0; 0.0 1.0]), 1)
box     = MaterialShape(Cuboid([0.0, -1.0], [200.0, 2.0], [1.0 0.0; 0.0 1.0]), 2)
xL, xR  = -(g_c/2 + w/2), +(g_c/2 + w/2)     # left / right rib centres
coupler = (rib(xL), rib(xR), slabsh, box)
minds   = (1, 1, 1, 2, 3)

# --- (1) even/odd supermode dispersion over > 1 octave ---------------------------------
println("== TFLN combiner: even/odd supermode dispersion (0.70–1.65 µm) ==")
λs = collect(range(0.72, 1.62; length=11))    # 1.17 octave
Δn = zero(λs); Lc = zero(λs)
for (j, λ) in enumerate(λs)
    sup = supermodes(coupler, minds, mv, 1/λ, grid, solver; nev=4)   # sorted by n_eff desc
    ne, no = sup[1].neff, sup[2].neff                                 # even (higher) / odd
    Δn[j] = ne - no
    Lc[j] = λ / (2*abs(Δn[j]))                                        # half-beat coupling length (µm)
    @printf("  λ=%.3f µm  n_even=%.4f  n_odd=%.4f  Δn=%.2e  L_c=%.1f µm\n", λ, ne, no, Δn[j], Lc[j])
end

# --- (2) dense directional-coupler combiner transmission -------------------------------
Lc_FH = interp1(λs, Lc, λF)                   # set device length = one coupling length at FH
L = Lc_FH                                       # → full cross-over at FH
λdense = collect(range(λs[1], λs[end]; length=400))
Δn_d = [interp1(λs, Δn, λ) for λ in λdense]
T_cross = [sin(π * L * abs(Δn_d[i]) / λdense[i])^2 for i in eachindex(λdense)]
T_bar = 1 .- T_cross
@printf("device length L = L_c(FH) = %.1f µm;  T_cross(FH)=%.2f  T_cross(SH)=%.2f\n",
        L, sin(π*L*abs(interp1(λs,Δn,λF))/λF)^2, sin(π*L*abs(interp1(λs,Δn,λS))/λS)^2)

# --- (3) EME validation of the uniform coupler (dedup-cheap) ---------------------------
println("== EME validation (uniform coupler, cross-section dedup) ==")
Ncell = 10
s_edges = collect(range(0.0, L; length=Ncell+1))
cell_shapes = fill(coupler, Ncell)
for λ in (λF, λS)
    res = eme_transmission(cell_shapes, fill(minds, Ncell), mats, 1/λ, grid, s_edges, solver; nev=4)
    # bar/cross from the two supermode propagation phases k_even, k_odd
    ke, ko = res.modes[1][1].k, res.modes[1][2].k
    tc = sin(π * L * abs(ke - ko))^2
    @printf("  λ=%.3f µm  EME supermodes k_even=%.4f k_odd=%.4f → T_cross=%.2f\n", λ, ke, ko, tc)
end

# --- (4) supermode profiles at FH and SH ------------------------------------------------
xc, yc = grid_coords(grid)
supF = supermodes(coupler, minds, mv, 1/λF, grid, solver; nev=4)
supS = supermodes(coupler, minds, mv, 1/λS, grid, solver; nev=4)

fig1 = Figure(size=(920, 340))
ax1 = Axis(fig1[1, 1], xlabel="wavelength (µm)", ylabel="supermode split Δn = n_even − n_odd",
    title="TFLN coupler supermode dispersion", yscale=log10)
lines!(ax1, λs, abs.(Δn), color=:purple, linewidth=2)
vlines!(ax1, [λS, λF], color=:gray, linestyle=:dash)
text!(ax1, λS, minimum(abs.(Δn)); text="SH", align=(:center,:bottom)); text!(ax1, λF, minimum(abs.(Δn)); text="FH", align=(:center,:bottom))
ax1b = Axis(fig1[1, 2], xlabel="wavelength (µm)", ylabel="transmission",
    title=@sprintf("combiner response (L=%.0f µm)", L))
lines!(ax1b, λdense, T_cross, color=:crimson, linewidth=2, label="cross port")
lines!(ax1b, λdense, T_bar, color=:dodgerblue, linewidth=2, label="bar port")
vlines!(ax1b, [λS, λF], color=:gray, linestyle=:dash); axislegend(ax1b, position=:rc)
save(joinpath(OUTDIR, "kwolek_combiner_dispersion_transmission.png"), fig1)

fig2 = Figure(size=(1000, 620))
for (row, (sup, tag)) in enumerate(((supF, "FH 1550 nm"), (supS, "SH 775 nm")))
    for (col, lbl) in enumerate(("even supermode", "odd supermode"))
        ax = Axis(fig2[row, col], xlabel="x (µm)", ylabel="y (µm)", aspect=DataAspect(),
            title="$tag — $lbl")
        hm = heatmap!(ax, xc, yc, absE_norm(sup[col].E), colormap=:turbo)
        for cx in (xL, xR); lines!(ax, [cx-w/2,cx+w/2,cx+w/2,cx-w/2,cx-w/2], [slab,slab,slab+etch,slab+etch,slab], color=:white, linewidth=0.8); end
        lines!(ax, [xc[1], xc[end]], [slab, slab], color=:white, linewidth=0.5)
        xlims!(ax, -2.6, 2.6); ylims!(ax, -0.4, 1.0)
    end
end
save(joinpath(OUTDIR, "kwolek_combiner_supermodes.png"), fig2)

println("saved: kwolek_combiner_dispersion_transmission.png, kwolek_combiner_supermodes.png → ", OUTDIR)
