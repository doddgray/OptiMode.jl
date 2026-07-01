# Shared helpers for the two EME (eigenmode-expansion) "paper reproduction" examples:
#
#   • dichroic_filter_magden2018.jl   — Magden et al., Nat. Commun. 9, 3009 (2018)  [Si SOI]
#   • tfln_combiner_kwolek2026.jl      — Kwolek et al., arXiv:2603.27034 (2026)      [TFLN]
#
# Both are adiabatic two-waveguide mode-evolution devices (a silicon dichroic filter and a
# TFLN FH/SH wavelength combiner). Following the approach of the corresponding examples in the
# MEOW fork (examples/papers/magden2018_dichroic.py, kwolek2026_faquad.py) but using OptiMode
# for the mode solving, the mode-crossing / dispersion analysis, and the EME:
#
#   (1) MODE SOLVING + DISPERSION: isolated-waveguide n_eff(λ) of the two guides WGA, WGB.
#   (2) MODE-CROSSING: the filter cutoff λ_C is the wavelength where β_A = β_B (the guides are
#       phase-matched); the dispersion curves cross there.
#   (3) DENSE >1-OCTAVE TRANSMISSION SPECTRUM: solve the *coupled* two-waveguide supermodes and
#       track the quasi-even supermode; for an adiabatic transition its power evolves from WGA
#       (below cutoff) to WGB (above cutoff), so its per-port power fraction is the ideal
#       filter response T_A(λ)=short-pass / T_B(λ)=long-pass. Evaluated densely by interpolation.
#   (4) FULL EME (validation): cascade the actual tapered device with OptiMode's `eme` and read
#       the bar/cross transmission at a few wavelengths.
#
# Grids are moderate so the plots generate on a workstation; the dense EME sweep deploys to a
# cluster via `deploy_eme`/`gather_eme` (as the MEOW fork's *_slurm.py designers do).

include(joinpath(@__DIR__, "paper_reproductions_common.jl"))   # Grid, solve_fundamental, matvals_builder, grid_coords, absE_norm, OUTDIR, C_MS
using OptiMode.EigenmodeExpansion: CrossSection, Cell, eme, power_coupling
using OptiMode.MaterialDispersion: Vacuum
using OptiMode: E⃗, E_relpower_xyz, solve_k, sliceinv_3x3, smooth_ε, MaterialShape
using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid

# ---------------------------------------------------------------------------------------
# Coupled-supermode solve on a two-waveguide cross-section
# ---------------------------------------------------------------------------------------

"""
    supermodes(shapes, minds, matvals, ω, grid, solver; nev, pol) -> Vector of (; neff, k, E)

Solve the lowest `nev` modes of a (coupled) cross-section and return the requested-polarization
modes sorted by descending effective index (the quasi-even supermode has the highest n_eff)."""
function supermodes(shapes, minds, matvals, ω, grid, solver; nev=4, pol=:TE)
    sm = smooth_ε(shapes, matvals(ω), minds, grid)
    εi = sliceinv_3x3(copy(selectdim(sm, 3, 1)))
    ∂ωε = copy(selectdim(sm, 3, 2))
    ε = sliceinv_3x3(copy(εi))
    km, ev = solve_k(ω, copy(εi), grid, solver; nev=nev, k_tol=1e-7, eig_tol=1e-7)
    Es = [E⃗(km[i], copy(ev[i]), εi, ∂ωε, grid; canonicalize=true, normalized=true) for i in eachindex(ev)]
    pf = [E_relpower_xyz(ε, Es[i])[pol === :TE ? 1 : 2] for i in eachindex(ev)]
    keep = [i for i in eachindex(ev) if pf[i] > 0.5]
    isempty(keep) && (keep = collect(eachindex(ev)))
    sort!(keep; by=i -> -km[i])                       # highest n_eff first (quasi-even)
    [(; neff=km[i]/ω, k=km[i], E=Es[i]) for i in keep]
end

"Power fraction of field `E` on the WGA side (x < x_split µm)."
function port_fraction(E, grid, x_split)
    Nx, Ny = size(grid)
    xc = (-grid.Δx/2) .+ (0.5:Nx) .* (grid.Δx/Nx)
    I = dropdims(sum(abs2, E; dims=1); dims=1)
    mask = [x < x_split ? 1.0 : 0.0 for x in xc, _ in 1:Ny]
    sum(I .* mask) / sum(I)
end

# ---------------------------------------------------------------------------------------
# Mode-crossing (cutoff) + dense adiabatic transmission spectrum
# ---------------------------------------------------------------------------------------

"Linear-interpolate `y(xq)` from samples (x, y) (x monotonically increasing)."
function interp1(x, y, xq)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    j = searchsortedlast(x, xq)
    t = (xq - x[j]) / (x[j+1] - x[j])
    y[j] * (1 - t) + y[j+1] * t
end

"First wavelength where the isolated-guide index difference nA−nB crosses zero (the βA=βB
phase-matching cutoff λ_C). Returns NaN if no crossing in range."
function crossing_wavelength(λ, nA, nB)
    d = nA .- nB
    for i in 1:length(λ)-1
        if sign(d[i]) != sign(d[i+1])
            return λ[i] - d[i]*(λ[i+1]-λ[i])/(d[i+1]-d[i])
        end
    end
    NaN
end

"""
    dichroic_spectrum(λ_nodes, fracA_nodes, λ_dense) -> (T_A, T_B)

Dense short-/long-pass transmission by interpolating the quasi-even supermode's WGA power
fraction (the ideal adiabatic mode-evolution filter response) onto `λ_dense`."""
function dichroic_spectrum(λ_nodes, fracA_nodes, λ_dense)
    T_A = [clamp(interp1(λ_nodes, fracA_nodes, λ), 0.0, 1.0) for λ in λ_dense]
    (T_A, 1 .- T_A)
end

# ---------------------------------------------------------------------------------------
# Full EME: build tapered-coupler cells and read bar/cross transmission
# ---------------------------------------------------------------------------------------

"""
    eme_transmission(cell_shapes, minds, materials, ω, grid, s_edges, solver; nev, in_mode, x_split)
        -> (; T, modes_out, S)

Cascade the device whose cell cross-sections are `cell_shapes[i]` (a tuple of MaterialShape),
each spanning `[s_edges[i], s_edges[i+1]]` µm, with OptiMode's `eme`. Returns the device
result and the per-output-mode transmission from input `in_mode`."""
function eme_transmission(cell_shapes, minds, materials, ω, grid, s_edges, solver; nev=4)
    cells = Cell[]
    for i in eachindex(cell_shapes)
        sc = (s_edges[i] + s_edges[i+1]) / 2
        L = s_edges[i+1] - s_edges[i]
        push!(cells, Cell(i, sc, L, CrossSection(collect(cell_shapes[i]), collect(minds))))
    end
    res = eme(cells, materials, ω, grid, solver; nev=nev, k_tol=1e-7)
    res
end
