# ──────────────────────────────────────────────────────────────────────────────
#  Top-level eigenmode-expansion driver.
#
#  Solve every cell's modal basis, build the per-cell propagation S-matrices and
#  the inter-cell interface S-matrices, and cascade them into the device S-matrix
#  — the same pipeline as MEOW (modes → interface/propagation matrices → SAX
#  cascade), expressed with OptiMode's differentiable mode solver.
# ──────────────────────────────────────────────────────────────────────────────

export EMEResult, eme, eme_smatrix, power_coupling

using MaxwellEigenmodes: AbstractEigensolver, KrylovKitEigsolve, solve_k

"""
    EMEResult

Result of an EME run: the device scattering matrix `S` (left-facet modes →
right-facet modes), the per-cell modal bases `modes`, the cell lengths, and `ω`.
"""
struct EMEResult{T,G}
    S::SMat
    modes::Vector{Modes{T,G}}
    lengths::Vector{Float64}
    ω::T
end

"""
    eme(cells, materials, ω, grid, solver=KrylovKitEigsolve();
        nev=2, conjugate=false, reg=1e-9, reciprocity=true, kwargs...) -> EMEResult

Run eigenmode expansion over a vector of [`Cell`](@ref)s. Each cell's cross-section
is solved for `nev` modes; adjacent cells are coupled through MEOW interface
S-matrices and joined by diagonal propagation S-matrices, then cascaded.
`kwargs` are forwarded to `solve_k`.
"""
function eme(cells::AbstractVector{Cell}, materials, ω, grid,
             solver::AbstractEigensolver=KrylovKitEigsolve();
             nev::Int=2, conjugate::Bool=false, reg::Real=1e-9,
             reciprocity::Bool=true, kwargs...)
    modes = [solve_cell_modes(c, materials, ω, grid, solver; nev, conjugate, kwargs...) for c in cells]
    S = _assemble(modes, [c.length for c in cells]; conjugate, reg, reciprocity)
    G = typeof(grid)
    T = typeof(float(ω))
    return EMEResult{T,G}(S, modes, [c.length for c in cells], T(ω))
end

"""
    eme(ε⁻¹s, ∂ε_∂ωs, lengths, ω, grid, solver=KrylovKitEigsolve();
        nev=2, conjugate=false, reg=1e-9, reciprocity=true, kwargs...) -> EMEResult

Run EME directly from precomputed per-cell dielectric fields `ε⁻¹s`/`∂ε_∂ωs`
(one entry per cell). This is the reverse-mode-AD-friendly entry point: each cell
solve carries the `solve_k` adjoint `rrule`, and the overlap/interface/cascade
algebra is plain differentiable linear algebra, so a `Zygote.gradient` of any
scalar of the result w.r.t. the `ε⁻¹s` (or `ω`) flows end-to-end through the
whole device.
"""
function eme(ε⁻¹s::AbstractVector{<:AbstractArray}, ∂ε_∂ωs::AbstractVector{<:AbstractArray},
             lengths::AbstractVector, ω, grid,
             solver::AbstractEigensolver=KrylovKitEigsolve();
             nev::Int=2, conjugate::Bool=false, reg::Real=1e-9,
             reciprocity::Bool=true, kwargs...)
    modes = map(eachindex(ε⁻¹s)) do i
        kmags, evecs = solve_k(ω, ε⁻¹s[i], grid, solver; nev, kwargs...)
        [build_mode(ω, kmags[j], evecs[j], ε⁻¹s[i], ∂ε_∂ωs[i], grid; conjugate) for j in 1:nev]
    end
    S = _assemble(modes, collect(Float64, lengths); conjugate, reg, reciprocity)
    G = typeof(grid)
    T = typeof(float(ω))
    return EMEResult{T,G}(S, modes, collect(Float64, lengths), T(ω))
end

"""
    eme_smatrix(structure, ω; nev=2, Nx=128, Ny=96, num_cells=20,
                solver=KrylovKitEigsolve(), kwargs...) -> EMEResult

Convenience entry point: build the simulation grid and cells from a
[`Structure`](@ref), then run [`eme`](@ref).
"""
function eme_smatrix(st::Structure, ω; nev::Int=2, Nx::Int=128, Ny::Int=96,
                     num_cells::Int=20, solver::AbstractEigensolver=KrylovKitEigsolve(),
                     conjugate::Bool=false, reg::Real=1e-9, reciprocity::Bool=true, kwargs...)
    grid = simulation_grid(st, Nx, Ny)
    cells = build_cells(st; num_cells)
    return eme(cells, st.stack.materials, ω, grid, solver; nev, conjugate, reg, reciprocity, kwargs...)
end

"""
assemble the device S-matrix from per-cell modes and lengths by folding the chain
`prop₀ ⋆ iface₀₁ ⋆ prop₁ ⋆ …`. Written as a mutation-free fold (no `push!`) so it
is reverse-mode (Zygote) differentiable.
"""
function _assemble(modes, lengths; conjugate::Bool=false, reg::Real=1e-9, reciprocity::Bool=true)
    n = length(modes)
    n >= 1 || throw(ArgumentError("need at least one cell"))
    acc = propagation_smatrix(modes[1], lengths[1])
    for i in 1:(n - 1)
        iface = interface_smatrix(modes[i], modes[i+1]; conjugate, reg, reciprocity)
        prop = propagation_smatrix(modes[i+1], lengths[i+1])
        acc = star(star(acc, iface), prop)
    end
    return acc
end

"""
    power_coupling(result_or_S; in_mode=1, out_mode=1) -> Real

`|S₂₁[out_mode, in_mode]|²` — the fraction of input power launched in
`in_mode` (left facet) that emerges in `out_mode` (right facet). A convenient
differentiable scalar objective for sweeps and inverse design.
"""
power_coupling(r::EMEResult; kwargs...) = power_coupling(r.S; kwargs...)
function power_coupling(S::SMat; in_mode::Int=1, out_mode::Int=1)
    t = transmission(S)
    return abs2(t[out_mode, in_mode])
end
