# Worker side: `run_task(batchdir, i)` executes one task of a deployed batch. Invoked by
# the generated `runtask.jl` inside each SLURM array task (or local process).
#
# A batch has a `kind`:
#   "solve"    (default) — solve for the requested bands of one parameter set and write
#                          the per-band summary (+ optional fields / PNGs).
#   "forward"  — AD forward pass: solve and additionally persist everything the adjoint
#                needs (ε⁻¹, ∂ε_∂ω, the converged k & eigenvectors) to the shared
#                filesystem, for a later `"backward"` task (see adjoint.jl).
#   "backward" — AD backward pass: load a forward task's saved state plus output
#                cotangents and run the `solve_k` adjoint, writing the input cotangents.
#
# Per "solve"/"forward" task it writes, atomically:
#   tasks/task_NNNNNN.json     per-band summary (neff, ng, GVD, Aeff, polarization, mode
#                              label, timing, host, …)
#   tasks/task_NNNNNN.h5       full mode-field data (only when save_fields=true)
#   tasks/task_NNNNNN_bMM.png  annotated mode-field image per band (when save_plots=true)
#   tasks/task_NNNNNN.done     completion marker (or .failed with the error text)

"""
    run_task(batchdir, i::Int)

Run task `i` (1-based) of the batch deployed in `batchdir`. Dispatches on the batch
`kind` (`"solve"`, `"forward"`, `"backward"`); for the default solve, it builds the
problem via the batch's `setup.jl` (`make_problem(p)`), solves the requested bands,
computes the per-band summary quantities (and optional fields/PNGs), and writes results
into `batchdir/tasks/`.
"""
function run_task(batchdir::AbstractString, i::Int)
    batch = load_batch(batchdir)
    kind = get(batch.manifest, "kind", "solve")
    if kind == "forward"
        return run_forward_task(batch, i)
    elseif kind == "backward"
        return run_backward_task(batch, i)
    else
        return _run_solve_task(batch, i)
    end
end

"per-task execution metadata (timing + scheduler/host identity)"
function _task_meta(t0::Float64, started::DateTime)
    return Dict{String,Any}(
        "host" => gethostname(),
        "node" => get(ENV, "SLURMD_NODENAME", gethostname()),
        "slurm_job" => get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "")),
        "slurm_task" => get(ENV, "SLURM_ARRAY_TASK_ID", ""),
        "started" => string(started),
        "finished" => string(now()),
        "runtime_s" => round(time() - t0; digits=3),
    )
end

"load `make_problem` from the batch's setup.jl inside a fresh module with the stack loaded"
function _load_setup(batchdir::AbstractString)
    m = Module(:SweepSetup)
    Core.eval(m, :(using ModeSweeps, MaterialDispersion, DielectricSmoothing,
        MaxwellEigenmodes, ModeAnalysis, LinearAlgebra))
    Base.include(m, joinpath(batchdir, "setup.jl"))
    return m
end

function _run_solve_task(batch::BatchInfo, i::Int)
    t0 = time()
    started = now()
    p = batch.params[i]
    base = _task_base(batch, i)
    rm(base * ".done"; force=true)
    rm(base * ".failed"; force=true)
    try
        nev = batch.manifest["nev"]::Int
        save_fields = batch.manifest["save_fields"]::Bool
        save_plots = get(batch.manifest, "save_plots", false)::Bool
        mode_labels = get(batch.manifest, "mode_labels", true)::Bool
        label_max_order = get(batch.manifest, "label_max_order", 4)
        solver_kwargs = NamedTuple(Symbol(k) => v for (k, v) in batch.manifest["solver_kwargs"])

        m = _load_setup(batch.dir)
        prob = Base.invokelatest(getfield(m, :make_problem), p)
        solver = Core.eval(m, Meta.parse(batch.manifest["solver"]::String))

        need_E = save_fields || save_plots
        summary, fields, Es = Base.invokelatest(_solve_and_summarize, p, prob, solver, nev,
            solver_kwargs, need_E, mode_labels, Int(label_max_order))
        summary["task"] = i
        summary["params"] = Dict(String(k) => v for (k, v) in pairs(p))
        merge!(summary, _task_meta(t0, started))

        _atomic_write(base * ".json", JSON3.write(summary))
        save_fields && _write_fields(base * ".h5", fields)
        save_plots && _write_plots(batch, i, summary, Es, prob.grid)
        _atomic_write(base * ".done", "")
        return summary
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        _atomic_write(base * ".failed", msg)
        rethrow()
    end
end

function _atomic_write(path::AbstractString, content::AbstractString)
    tmp = path * ".tmp"
    write(tmp, content)
    mv(tmp, path; force=true)
    return path
end

"solve one parameter set and compute per-band summary quantities (+ optional E fields)"
function _solve_and_summarize(p::NamedTuple, prob::NamedTuple, solver, nev::Int,
    solver_kwargs::NamedTuple, need_E::Bool, mode_labels::Bool, label_max_order::Int;
    allow_kerr::Bool=true)
    haskey(p, :ω) || throw(ArgumentError("parameter sets must include the frequency `ω`"))
    ω = Float64(p.ω)
    ε⁻¹ = prob.ε⁻¹
    ∂ε_∂ω = prob.∂ε_∂ω
    grid = prob.grid
    ∂²ε_∂ω² = hasproperty(prob, :∂²ε_∂ω²) ? prob.∂²ε_∂ω² : nothing
    ε = sliceinv_3x3(ε⁻¹)

    # Optional first-order Kerr correction: active when the problem supplies an n₂ map
    # (μm²/W; see DielectricSmoothing.smooth_scalar) and the parameter set an optical
    # power `P` (W). Each band is corrected assuming the full power in that mode.
    P = haskey(p, :P) ? Float64(p.P) : 0.0
    local kmags, evecs, kmags_lin, dn_max
    if allow_kerr && P > 0 && hasproperty(prob, :n₂)
        res = solve_k_kerr(ω, P, ε⁻¹, ∂ε_∂ω, prob.n₂, grid, solver; nev, solver_kwargs...)
        kmags, evecs = res.kmags, res.evecs
        kmags_lin, dn_max = res.kmags_lin, res.dn_max
    else
        kmags, evecs = solve_k(ω, copy(ε⁻¹), grid, solver; nev, solver_kwargs...)
        kmags_lin, dn_max = kmags, zeros(length(kmags))
    end

    bands = Vector{Dict{String,Any}}(undef, length(kmags))
    Es = need_E ? Vector{Array{ComplexF64}}(undef, length(kmags)) : nothing
    for (b, (kmag, ev)) in enumerate(zip(kmags, evecs))
        neff = kmag / ω
        local ng, gvd
        if ∂²ε_∂ω² !== nothing && grid isa Grid{2}
            ng, gvd = ng_gvd(ω, kmag, ev, ε⁻¹, ∂ε_∂ω, ∂²ε_∂ω², grid)
        else
            ng = group_index(kmag, ev, ω, ε⁻¹, ∂ε_∂ω, grid)
            gvd = NaN
        end
        # power-normalized, phase-canonicalized E-field for effective area & polarization
        E = E⃗(kmag, copy(ev), ε⁻¹, ∂ε_∂ω, grid; canonicalize=true, normalized=true)
        Aeff = 𝓐(neff, ng, E)
        relpwr = E_relpower_xyz(ε, E)
        bd = Dict{String,Any}(
            "band" => b,
            "kmag" => kmag,
            "neff" => neff,
            "ng" => ng,
            "gvd" => gvd,
            "Aeff" => Aeff,
            "pol_x" => relpwr[1],
            "pol_y" => relpwr[2],
            "pol_z" => relpwr[3],
            "pol_axis" => argmax(relpwr),
            "dneff_kerr" => (kmag - kmags_lin[b]) / ω,
            "dn_max" => dn_max[b],
        )
        _add_mode_label!(bd, E, grid, mode_labels, label_max_order)
        bands[b] = bd
        need_E && (Es[b] = E)
    end
    summary = Dict{String,Any}("omega" => ω, "bands" => bands, "nev" => length(kmags))
    fields = need_E ? (; ω, kmags, evecs, Es, grid) : nothing
    return summary, fields, Es
end

"classify a mode by Hermite–Gaussian fit (TE/TM, transverse order) and store it on `bd`"
function _add_mode_label!(bd::Dict{String,Any}, E, grid, mode_labels::Bool, max_order::Int)
    label, mp, mm, mn, rel, tef = "", "", 0, 0, NaN, NaN
    if mode_labels && grid isa Grid{2}
        try
            nw = hg_mode_label(E, grid; max_order)
            label, mp, mm, mn = nw.label, String(nw.pol), nw.m, nw.n
            rel, tef = nw.rel_error, nw.te_frac
        catch err
            @warn "mode labeling failed; leaving label blank" exception = err
        end
    end
    bd["label"] = label
    bd["mode_pol"] = mp
    bd["mode_m"] = mm
    bd["mode_n"] = mn
    bd["hg_rel_error"] = rel
    bd["te_frac"] = tef
    return bd
end

# ---------------------------------------------------------------------------------
# field PNGs
# ---------------------------------------------------------------------------------

"ASCII mode label (e.g. TE00) for the PNG header — the 5×7 font has no subscripts"
function _ascii_label(bd)
    pol = get(bd, "mode_pol", "")
    isempty(pol) && return ""
    return string(pol, get(bd, "mode_m", 0), get(bd, "mode_n", 0))
end

"header lines (summary annotations) for the PNG of one band"
function _png_header(summary, bd)
    ω = get(summary, "omega", NaN)
    λ = ω > 0 ? 1 / ω : NaN
    f3(x) = (x isa Real && isfinite(x)) ? string(round(x; sigdigits=4)) : "n/a"
    lines = String[
        "TASK $(get(summary,"task","?"))  BAND $(get(bd,"band","?"))  $(_ascii_label(bd))",
        "LAMBDA=$(f3(λ)) UM  NEFF=$(f3(get(bd,"neff",NaN)))  NG=$(f3(get(bd,"ng",NaN)))",
        "GVD=$(f3(get(bd,"gvd",NaN)))  AEFF=$(f3(get(bd,"Aeff",NaN))) UM2",
        "POL X/Y/Z = $(f3(get(bd,"pol_x",NaN)))/$(f3(get(bd,"pol_y",NaN)))/$(f3(get(bd,"pol_z",NaN)))",
        "HOST $(get(summary,"node",get(summary,"host","?")))  T=$(f3(get(summary,"runtime_s",NaN)))S",
    ]
    return lines
end

"render & save one annotated PNG per band into the batch's tasks/ directory"
function _write_plots(batch::BatchInfo, i::Int, summary, Es, grid)
    Es === nothing && return nothing
    base = _task_base(batch, i)
    for (b, bd) in enumerate(summary["bands"])
        b <= length(Es) || break
        path = base * "_b" * @sprintf("%02i", b) * ".png"
        try
            render_mode_png(path, Es[b], grid, _png_header(summary, bd))
        catch err
            @warn "field PNG rendering failed for task $i band $b" exception = err
        end
    end
    return nothing
end

"write full mode-field data for one task to HDF5"
function _write_fields(path::AbstractString, fields::NamedTuple)
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        f["omega"] = fields.ω
        f["kmags"] = collect(fields.kmags)
        evm = hcat(fields.evecs...)            # (2N, nev)
        f["evecs_re"] = real.(evm)
        f["evecs_im"] = imag.(evm)
        Earr = cat(fields.Es...; dims=ndims(first(fields.Es)) + 1)  # (3, Ns..., nev)
        f["E_re"] = real.(Earr)
        f["E_im"] = imag.(Earr)
        g = fields.grid
        f["grid_Δ"] = ndims(g) == 2 ? [g.Δx, g.Δy] : [g.Δx, g.Δy, g.Δz]
        f["grid_N"] = collect(size(g))
    end
    mv(tmp, path; force=true)
    return path
end
