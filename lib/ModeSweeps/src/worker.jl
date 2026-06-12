# Worker side: `run_task(batchdir, i)` executes one parameter set of a deployed batch.
# Invoked by the generated `runtask.jl` inside each SLURM array task (or local process).
#
# Per task it writes, atomically:
#   tasks/task_NNNNNN.json   per-band summary (neff, ng, GVD, effective area, polarization)
#   tasks/task_NNNNNN.h5     full mode-field data (only when the batch was deployed
#                            with save_fields=true)
#   tasks/task_NNNNNN.done   completion marker (or .failed with the error text)

"""
    run_task(batchdir, i::Int)

Run task `i` (1-based) of the batch deployed in `batchdir`: build the problem via the
batch's `setup.jl` (`make_problem(p)`), solve for the requested bands, compute the
per-band summary quantities, and write results into `batchdir/tasks/`.
"""
function run_task(batchdir::AbstractString, i::Int)
    t0 = time()
    batch = load_batch(batchdir)
    p = batch.params[i]
    base = _task_base(batch, i)
    rm(base * ".done"; force=true)
    rm(base * ".failed"; force=true)
    try
        nev = batch.manifest["nev"]::Int
        save_fields = batch.manifest["save_fields"]::Bool
        solver_kwargs = NamedTuple(Symbol(k) => v for (k, v) in batch.manifest["solver_kwargs"])

        # build the problem & solver inside a fresh module with the stack pre-loaded
        m = Module(:SweepSetup)
        Core.eval(m, :(using ModeSweeps, MaterialDispersion, DielectricSmoothing,
            MaxwellEigenmodes, ModeAnalysis, LinearAlgebra))
        Base.include(m, joinpath(batchdir, "setup.jl"))
        prob = Base.invokelatest(getfield(m, :make_problem), p)
        solver = Core.eval(m, Meta.parse(batch.manifest["solver"]::String))

        summary, fields = Base.invokelatest(_solve_and_summarize, p, prob, solver, nev, solver_kwargs, save_fields)
        summary["task"] = i
        summary["params"] = Dict(String(k) => v for (k, v) in pairs(p))
        summary["runtime_s"] = round(time() - t0; digits=3)
        summary["finished"] = string(now())

        _atomic_write(base * ".json", JSON3.write(summary))
        save_fields && _write_fields(base * ".h5", fields)
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

"solve one parameter set and compute per-band summary quantities (+ optional field data)"
function _solve_and_summarize(p::NamedTuple, prob::NamedTuple, solver, nev::Int,
    solver_kwargs::NamedTuple, save_fields::Bool)
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
    if P > 0 && hasproperty(prob, :n₂)
        res = solve_k_kerr(ω, P, ε⁻¹, ∂ε_∂ω, prob.n₂, grid, solver; nev, solver_kwargs...)
        kmags, evecs = res.kmags, res.evecs
        kmags_lin, dn_max = res.kmags_lin, res.dn_max
    else
        kmags, evecs = solve_k(ω, copy(ε⁻¹), grid, solver; nev, solver_kwargs...)
        kmags_lin, dn_max = kmags, zeros(length(kmags))
    end

    bands = Vector{Dict{String,Any}}(undef, length(kmags))
    Es = save_fields ? Vector{Array{ComplexF64}}(undef, length(kmags)) : nothing
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
        bands[b] = Dict{String,Any}(
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
        save_fields && (Es[b] = E)
    end
    summary = Dict{String,Any}("omega" => ω, "bands" => bands, "nev" => length(kmags))
    fields = save_fields ? (; ω, kmags, evecs, Es, grid) : nothing
    return summary, fields
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
