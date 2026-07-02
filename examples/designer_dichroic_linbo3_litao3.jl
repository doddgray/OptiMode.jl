# AD+EME DESIGNER — spectrally-selective dichroic filter on X-cut TFLN/TFLT rib waveguides,
# swept over cutoff wavelength and over three (full-thickness, slab-thickness) film stacks.
#
# Same mode-crossing / dichroic-filter methodology as designer_dichroic_si3n4.jl (Magden et al.,
# Nat. Commun. 9, 3009 (2018)) and the same shared two-stage AD+EME driver
# (dichroic_designer_common.jl), applied to a **rib-on-slab** waveguide family instead of a
# buried strip: X-cut thin-film LiNbO₃ (TFLN) and X-cut thin-film LiTaO₃ (TFLT), fully
# encapsulated in SiO₂, with a 65° sidewall (`GeometryPrimitives.Trapezoid`, the same rib profile
# used descriptively — but not previously actually built with sloped sidewalls — in
# tfln_combiner_kwolek2026.jl).
#
# Film stacks (full thickness / slab thickness, all a 300-nm etch): 400/100, 500/150, 600/200 nm.
# WGA is a single solid-core rib of top-etch base width w_A atop the shared slab; WGB is the same
# paper-Fig.-1b sub-wavelength segmentation as the Si₃N₄ designer — three rib segments of base
# width w_B, gap g_B — atop the *same* continuous slab (so WGA and WGB differ only in their
# above-slab core geometry, exactly mirroring the paper's "same material and thickness" WGB
# construction). X-cut orientation (extraordinary axis in-plane along x) is the standard
# TFLN/TFLT convention used throughout this package's other TFLN/TFLT examples (see
# tfln_shg_dispersion.jl, pplt_allband_opa_kuznetsov2026.jl): `rotate(LiNbO₃, Ry)`.
#
# Same design sign convention as designer_dichroic_si3n4.jl (n_gA > n_gB at the crossing) and
# the same two-phase AD warm start / EME adiabatic-taper-length search / dense-spectrum / TE00
# field-profile pipeline — see dichroic_designer_common.jl and designer_dichroic_si3n4.jl headers
# for the physics. Unlike the Si₃N₄ designer's thin-core regime, TFLN/TFLT ribs at these
# thicknesses (400–600 nm) are comparable in scale to this package's other TFLN rib examples
# (~1–2 µm top width), so the design-space bounds below are NOT thickness-dependent.
#
# Settings (see examples/README.md): --n-freqs (dispersion/spectrum sweep nodes, default 11),
# --n-dense (dense spectrum resolution, default 400), --resolution-scale/--domain-scale/--n-cells
# or the bundled --quality=low|medium|high|ultra preset (grid + EME taper cell count).
#
# Run:  julia --project=. examples/designer_dichroic_linbo3_litao3.jl        (needs CairoMakie)
#       julia --project=. examples/designer_dichroic_linbo3_litao3.jl --quality=high

include(joinpath(@__DIR__, "dichroic_designer_common.jl"))
using OptiMode: rotate
using OptiMode.DielectricSmoothing.GeometryPrimitives: Trapezoid
using CairoMakie

cfg = example_settings(n_freqs=11, n_dense=400)
solver = KrylovKitEigsolve()
λC_grid = collect(1.00:0.05:1.50)                       # 11 targets, 1000–1500 nm
stacks_nm = ((400, 100), (500, 150), (600, 200))         # (full thickness, slab thickness)
θ_sidewall = deg2rad(65.0)                                # 65° sidewall (base-to-wall angle)
SUBW = 60.0                                                # slab/substrate half-plane extent (µm), ≫ domain

const _RY = [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0]      # RotY(π/2): c-axis (z) → in-plane x
LiNbO₃_xcut = rotate(LiNbO₃, _RY; name=:LiNbO₃_xcut)
LiTaO₃_xcut = rotate(LiTaO₃, _RY; name=:LiTaO₃_xcut)
# `mv` built at top level (once per material) — see DichroicGeometry docstring for why.
mv_LN = matvals_builder([LiNbO₃_xcut, SiO₂]; air=false)
mv_LT = matvals_builder([LiTaO₃_xcut, SiO₂]; air=false)

"Design-space starting point + box bounds (µm) for a TFLN/TFLT rib: (w_A, w_B, g_B), base
(bottom-of-etch) widths/gap — thickness-independent at these (400–600 nm) film thicknesses."
tfln_geometry_defaults() = (p0=[1.20, 0.60, 0.30], lo=[0.80, 0.40, 0.20], hi=[1.80, 1.00, 0.60])

"Build the DichroicGeometry (rib-on-slab WGA / 3-rib segmented WGB, 65° sidewalls) for one
`(core_material, full_thickness_nm, slab_thickness_nm)` TFLN/TFLT stack."
function tfln_geometry(core_mat, mv, tag_prefix, full_nm, slab_nm, cfg)
    full, slab = 1e-3 * full_nm, 1e-3 * slab_nm
    etch = full - slab
    mats = (core_mat, SiO₂)
    ridge(cx, w_base) = MaterialShape(Trapezoid([cx, slab], w_base, etch, θ_sidewall), 1)
    slab_shape = MaterialShape(Cuboid([0.0, slab/2], [SUBW, slab], [1.0 0.0; 0.0 1.0]), 1)
    wgA_shapes(p) = (ridge(0.0, p[1]), slab_shape)
    wgB_shapes(p) = (w = p[2]; g = p[3]; (ridge(-(w + g), w), ridge(0.0, w), ridge(w + g, w), slab_shape))
    function coupled_shapes(p, gap)
        w_A, w_B, g_B = p
        xA, xB = -(gap/2 + w_A/2), gap/2 + w_B/2
        (ridge(xA, w_A), ridge(xB, w_B), ridge(xB + w_B + g_B, w_B), ridge(xB + 2*(w_B + g_B), w_B), slab_shape)
    end
    d = tfln_geometry_defaults()
    grid = mk_grid(cfg, 7.0, 3.0, 112, 58)
    DichroicGeometry(; tag=@sprintf("%s_full%03d_slab%03dnm", tag_prefix, full_nm, slab_nm), mats, mv,
        wgA_shapes, wgB_shapes, coupled_shapes,
        mindsA=(1, 1, 2), mindsB=(1, 1, 1, 1, 2), mindsAB=(1, 1, 1, 1, 1, 2), grid,
        p0=d.p0, lo=d.lo, hi=d.hi, g_start=0.40, g_end=2.50)
end

println("== X-cut TFLN/TFLT dichroic designer: λ_C ∈ [", λC_grid[1], ", ", λC_grid[end],
        "] µm × stacks ", stacks_nm, " (65° sidewall) ==")
all_results = Dict{String,Any}()
for (mat, mv, tag) in ((LiNbO₃_xcut, mv_LN, "LiNbO3"), (LiTaO₃_xcut, mv_LT, "LiTaO3"))
    for (full_nm, slab_nm) in stacks_nm
        @printf("\n---- %s full=%d nm slab=%d nm (etch=%d nm) ----\n", tag, full_nm, slab_nm, full_nm - slab_nm)
        geo = tfln_geometry(mat, mv, tag, full_nm, slab_nm, cfg)
        all_results[geo.tag] = run_dichroic_sweep(geo, λC_grid, cfg, solver; xlims=(0.6, 1.9))
    end
end
println("\ndone: 2 materials × ", length(stacks_nm), " stacks × ", length(λC_grid), " cutoff targets → ",
        2 * length(stacks_nm) * length(λC_grid), " cases. Summary grids + CSVs → ", OUTDIR)
