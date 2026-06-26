# ──────────────────────────────────────────────────────────────────────────────
#  Convergence / accuracy diagnostics.
#
#  EME accuracy rests on two coupled choices: the number of cells (z-discretisation)
#  and the number of modes per cell `nev` (modal-basis truncation). A truncated
#  basis shows up as a *non-passive* raw interface S-matrix (singular values σ > 1) —
#  the very thing `enforce_passivity` masks. Since the passivity SVD is not
#  reverse-mode-AD friendly (differentiate with `passivity=:none`), the right recipe
#  is to raise `nev` until the raw interfaces are passive on their own, then the
#  `:none` adjoint matches the physical primal. These helpers measure both.
# ──────────────────────────────────────────────────────────────────────────────

export passivity_report, nev_convergence

"""
    passivity_report(modes; conjugate=false, reg=1e-9) -> NamedTuple
    passivity_report(result::EMEResult; …) -> NamedTuple

Quantify modal-truncation error by how far each *raw* (un-enforced) interface S-matrix
exceeds passivity. For every adjacent pair of cells the interface S-matrix is rebuilt with
`passivity=:none` and its largest singular value taken; the report returns

- `per_interface :: Vector{Float64}` — `max(σ) - 1` for each interface (0 ⇒ passive),
- `max_excess :: Float64` — the worst interface,
- `passive :: Bool` — whether `max_excess ≤ tol` (`tol = 1e-6`).

A large `max_excess` means `nev` is too small (or cells too coarse) and the passivity
enforcement is doing real work — so a `passivity=:none` adjoint would not match the
enforced primal. Drive `max_excess → 0` (raise `nev`) for trustworthy gradients.
"""
function passivity_report(modes::AbstractVector; conjugate::Bool=false, reg::Real=1e-9,
                          tol::Real=1e-6)
    n = length(modes)
    n ≥ 2 || return (; per_interface=Float64[], max_excess=0.0, passive=true)
    per = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        S = interface_smatrix(modes[i], modes[i+1]; conjugate, reg, reciprocity=false, passivity=:none)
        per[i] = maximum(svdvals(S.S)) - 1.0
    end
    mx = maximum(per)
    return (; per_interface=per, max_excess=mx, passive=(mx ≤ tol))
end

passivity_report(r::EMEResult; kwargs...) = passivity_report(r.modes; kwargs...)

"""
    nev_convergence(cells, materials, ω, grid, solver=KrylovKitEigsolve();
                    nevs=1:4, objective=power_coupling, conjugate=false, reg=1e-9,
                    reciprocity=true, kwargs...) -> NamedTuple

Sweep the modal-basis size and report convergence of a scalar `objective(result)` (default
[`power_coupling`](@ref)). For each `nev ∈ nevs` an EME run is performed (with `dedup` so
repeated cross-sections cost nothing extra) and the report returns

- `nevs :: Vector{Int}`,
- `values :: Vector{Float64}` — `objective` at each `nev`,
- `deltas :: Vector{Float64}` — `|value[j] − value[j-1]|` (first entry `NaN`),
- `max_excess :: Vector{Float64}` — the [`passivity_report`](@ref) worst interface at each
  `nev` (should fall toward 0 as the basis converges).

Pick the smallest `nev` whose `deltas` and `max_excess` are both below your tolerance.
`kwargs` are forwarded to `solve_k`.
"""
function nev_convergence(cells::AbstractVector{Cell}, materials, ω, grid,
                         solver::AbstractEigensolver=KrylovKitEigsolve();
                         nevs=1:4, objective=power_coupling, conjugate::Bool=false,
                         reg::Real=1e-9, reciprocity::Bool=true, kwargs...)
    nv = collect(Int, nevs)
    values = Vector{Float64}(undef, length(nv))
    excess = Vector{Float64}(undef, length(nv))
    for (j, nev) in enumerate(nv)
        r = eme(cells, materials, ω, grid, solver; nev, conjugate, reg, reciprocity,
                passivity=:invert, kwargs...)
        values[j] = float(objective(r))
        excess[j] = passivity_report(r; conjugate, reg).max_excess
    end
    deltas = [j == 1 ? NaN : abs(values[j] - values[j-1]) for j in eachindex(values)]
    return (; nevs=nv, values, deltas, max_excess=excess)
end

"""
    nev_convergence(structure, ω; nevs=1:4, Nx=128, Ny=96, num_cells=20,
                    solver=KrylovKitEigsolve(), objective=power_coupling, kwargs...)

Convenience method building the grid and cells from a [`Structure`](@ref) (cf.
[`eme_smatrix`](@ref)) before sweeping `nev`.
"""
function nev_convergence(st::Structure, ω; nevs=1:4, Nx::Int=128, Ny::Int=96,
                         num_cells::Int=20, solver::AbstractEigensolver=KrylovKitEigsolve(),
                         objective=power_coupling, kwargs...)
    grid = simulation_grid(st, Nx, Ny)
    cells = build_cells(st; num_cells)
    return nev_convergence(cells, st.stack.materials, ω, grid, solver; nevs, objective, kwargs...)
end
