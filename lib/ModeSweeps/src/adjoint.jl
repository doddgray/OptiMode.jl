# Remote, SLURM-friendly automatic differentiation of the mode solver.
#
# The expensive eigensolve `solve_k` carries an adjoint-method `rrule` (in
# MaxwellEigenmodes): given output cotangents `(k̄mags, ēvecs)` it returns input
# cotangents `(ω̄, ε̄⁻¹)` for ≈ one extra eigensolve, independent of the number of
# parameters. We expose this across the SLURM boundary by splitting the forward (primal)
# and backward (adjoint) passes into *separate tasks* that communicate through the
# cluster's shared filesystem:
#
#   forward task  (kind="forward")  : solve_k(ω, ε⁻¹, grid, solver) → (kmags, evecs),
#                                     persist (ω, ε⁻¹, ∂ε_∂ω, grid, kmags, evecs, solver)
#                                     to  tasks/task_NNNNNN.fwd.h5  (the adjoint inputs),
#                                     plus the usual summary / fields / PNGs.
#   backward task (kind="backward") : load that forward state from the (shared) forward
#                                     batch directory + the user-supplied output
#                                     cotangents (tasks/task_NNNNNN.ct.h5), run the
#                                     `solve_k` adjoint, and write the input cotangents
#                                     (ω̄, ε̄⁻¹) to  tasks/task_NNNNNN.bwd.h5.
#
# Because all state lives on the shared filesystem, the forward and backward passes can
# run as different SLURM jobs (different nodes, different times, even different Julia
# sessions / machines driving them), exactly like ordinary sweeps. This makes the outer
# loop of an adjoint-based inverse design (value on the forward pass, gradient on the
# backward pass) deployable to the cluster.
#
# Cotangent convention. `solve_k`'s pullback is exact for `k̄mags` cotangents (the usual
# case: objectives built from neff = kmag/ω, ng, GVD, … are functions of `kmag`), giving
# ∂/∂ω and ∂/∂ε⁻¹. `ēvecs` cotangents are phase-sensitive (they must match the exact
# eigenvector phase of the forward solve); they are supported here by persisting and
# reusing the forward eigenvectors, but objectives that depend on the field are usually
# better expressed through the field post-processing rules (`group_index`, …) whose
# adjoints already reduce to ε⁻¹/∂ε_∂ω cotangents.

# ---------------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------------

function run_forward_task(batch::BatchInfo, i::Int)
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
        label_max_order = Int(get(batch.manifest, "label_max_order", 4))
        solver_str = batch.manifest["solver"]::String
        solver_kwargs = NamedTuple(Symbol(k) => v for (k, v) in batch.manifest["solver_kwargs"])

        m = _load_setup(batch.dir)
        prob = Base.invokelatest(getfield(m, :make_problem), p)
        solver = Core.eval(m, Meta.parse(solver_str))

        # forward AD pass uses the plain (linear) solve_k that the adjoint rrule covers
        summary, fields, Es = Base.invokelatest(_solve_and_summarize, p, prob, solver, nev,
            solver_kwargs, true, mode_labels, label_max_order; allow_kerr=false)
        summary["task"] = i
        summary["params"] = Dict(String(k) => v for (k, v) in pairs(p))
        summary["kind"] = "forward"
        merge!(summary, _task_meta(t0, started))

        _write_forward_state(base * ".fwd.h5", prob, fields, solver_str, nev)
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

function run_backward_task(batch::BatchInfo, i::Int)
    t0 = time()
    started = now()
    base = _task_base(batch, i)
    rm(base * ".done"; force=true)
    rm(base * ".failed"; force=true)
    try
        fwd_dir = batch.manifest["forward_dir"]::String
        fwd_base = joinpath(fwd_dir, "tasks", "task_" * _task_id_str(i))
        st = _load_forward_state(fwd_base * ".fwd.h5")
        ct = _load_cotangents(base * ".ct.h5", st.nev)

        mod = Module(:SweepSolver)
        Core.eval(mod, :(using MaxwellEigenmodes))
        solver = Core.eval(mod, Meta.parse(st.solver))

        ω_bar, ε⁻¹_bar = Base.invokelatest(_remote_pullback, st.ω, st.ε⁻¹, st.grid, solver,
            st.nev, ct.k̄mags, ct.ēvecs)
        _write_backward_state(base * ".bwd.h5", ω_bar, ε⁻¹_bar)

        summary = Dict{String,Any}("task" => i, "kind" => "backward",
            "omega_bar" => ω_bar, "eps_inv_bar_norm" => sqrt(sum(abs2, ε⁻¹_bar)),
            "forward_dir" => fwd_dir)
        merge!(summary, _task_meta(t0, started))
        _atomic_write(base * ".json", JSON3.write(summary))
        _atomic_write(base * ".done", "")
        return summary
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        _atomic_write(base * ".failed", msg)
        rethrow()
    end
end

"run the `solve_k` adjoint `rrule` for the saved forward state and output cotangents"
function _remote_pullback(ω, ε⁻¹, grid, solver, nev::Int, k̄mags, ēvecs)
    _, pb = rrule(solve_k, ω, copy(ε⁻¹), grid, solver; nev)
    Δk = Any[k̄mags === nothing || !isfinite(k̄mags[j]) ? ZeroTangent() : Float64(k̄mags[j]) for j in 1:nev]
    Δev = ēvecs === nothing ? Any[ZeroTangent() for _ in 1:nev] : Any[ēvecs[j] for j in 1:nev]
    res = pb((Δk, Δev))
    return (Float64(res[2]), Array{Float64}(res[3]))    # (ω̄, ε̄⁻¹)
end

# ---- forward-state / cotangent / backward-state I/O on the shared filesystem ----

function _write_forward_state(path, prob::NamedTuple, fields::NamedTuple, solver::String, nev::Int)
    g = prob.grid
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        f["omega"] = Float64(fields.ω)
        f["nev"] = nev
        f["solver"] = solver
        f["eps_inv"] = Array{Float64}(real.(prob.ε⁻¹))
        f["deps_dom"] = Array{Float64}(real.(prob.∂ε_∂ω))
        f["kmags"] = collect(Float64.(fields.kmags))
        evm = hcat(fields.evecs...)
        f["evecs_re"] = real.(evm)
        f["evecs_im"] = imag.(evm)
        f["grid_Δ"] = ndims(g) == 2 ? [g.Δx, g.Δy] : [g.Δx, g.Δy, g.Δz]
        f["grid_N"] = collect(size(g))
    end
    mv(tmp, path; force=true)
    return path
end

function _grid_from(Δ, N)
    return length(N) == 2 ? Grid(Float64(Δ[1]), Float64(Δ[2]), Int(N[1]), Int(N[2])) :
           Grid(Float64(Δ[1]), Float64(Δ[2]), Float64(Δ[3]), Int(N[1]), Int(N[2]), Int(N[3]))
end

function _load_forward_state(path)
    isfile(path) || error("forward state not found at $path (was the forward batch run?)")
    return h5open(path, "r") do f
        nev = Int(read(f, "nev"))
        evm = read(f, "evecs_re") .+ im .* read(f, "evecs_im")
        (; ω=read(f, "omega"), nev, solver=read(f, "solver"),
            ε⁻¹=read(f, "eps_inv"), ∂ε_∂ω=read(f, "deps_dom"),
            kmags=read(f, "kmags"), evecs=[evm[:, j] for j in 1:nev],
            grid=_grid_from(read(f, "grid_Δ"), read(f, "grid_N")))
    end
end

"write a per-task output-cotangent file `(k̄mags[, ēvecs])` for a backward batch"
function _write_cotangents(path, ct, nev::Int)
    kbar = hasproperty(ct, :k̄) && ct.k̄ !== nothing ? collect(Float64.(ct.k̄)) : fill(NaN, nev)
    length(kbar) == nev || throw(ArgumentError("k̄ must have length nev=$nev, got $(length(kbar))"))
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        f["kbar"] = kbar
        if hasproperty(ct, :ēv) && ct.ēv !== nothing
            evm = hcat(ct.ēv...)
            f["evbar_re"] = real.(evm)
            f["evbar_im"] = imag.(evm)
        end
    end
    mv(tmp, path; force=true)
    return path
end

function _load_cotangents(path, nev::Int)
    isfile(path) || error("cotangent file not found at $path")
    return h5open(path, "r") do f
        kbar = haskey(f, "kbar") ? read(f, "kbar") : nothing
        evbar = nothing
        if haskey(f, "evbar_re")
            evm = read(f, "evbar_re") .+ im .* read(f, "evbar_im")
            evbar = [evm[:, j] for j in 1:nev]
        end
        (; k̄mags=kbar, ēvecs=evbar)
    end
end

function _write_backward_state(path, ω_bar, ε⁻¹_bar)
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        f["omega_bar"] = Float64(ω_bar)
        f["eps_inv_bar"] = Array{Float64}(ε⁻¹_bar)
    end
    mv(tmp, path; force=true)
    return path
end

# ---------------------------------------------------------------------------------
# user side
# ---------------------------------------------------------------------------------

"cluster-side path of a batch directory (remote_dir in ssh mode, else the local dir)"
_cluster_dir(b::BatchInfo) = let r = get(b.manifest, "remote_dir", "")
    isempty(r) ? b.dir : r
end

"""
    deploy_forward(setup_file, params; nev=1, solver=…, slurm=…, backend=…, blocking=false, kwargs...) -> BatchInfo

Deploy an AD **forward pass**: one task per parameter set runs the (linear) `solve_k`
and, besides the usual summary (and optional fields/PNGs), persists everything the
adjoint needs — `(ω, ε⁻¹, ∂ε_∂ω, grid, kmags, evecs, solver)` — to the shared
filesystem (`tasks/task_NNNNNN.fwd.h5`). Pair it with [`deploy_backward`](@ref).

Accepts the same keywords as [`deploy_batch`](@ref) (`nev`, `solver`, `solver_kwargs`,
`save_fields`, `save_plots`, `mode_labels`, `backend`, `slurm`, `name`, `dir`, `submit`,
`blocking`). Retrieve the primal solution with [`forward_solution`](@ref).
"""
function deploy_forward(setup_file::AbstractString, params; name::AbstractString="ad_forward",
    kwargs...)
    return deploy_batch(setup_file, params; name, kind="forward", kwargs...)
end

"""
    forward_solution(batch, i=1) -> (; ω, kmags, evecs, ε⁻¹, ∂ε_∂ω, grid)

Load the persisted forward-pass state of task `i` of a forward batch (the primal
solution plus the adjoint inputs). In ssh mode the `.fwd.h5` file is fetched first.
"""
function forward_solution(batch::BatchInfo, i::Int=1)
    _maybe_fetch_markers!(batch; fields=true)
    return _load_forward_state(_task_base(batch, i) * ".fwd.h5")
end

"""
    deploy_backward(forward_batch, cotangents; name="ad_backward", slurm=…, backend=…, blocking=false, kwargs...) -> BatchInfo

Deploy an AD **backward pass** for a completed (or running) `forward_batch`. `cotangents`
is a vector with one entry per forward task — each a `NamedTuple` `(; k̄, ēv=nothing)`
giving the output cotangents (`k̄` w.r.t. the per-band `kmags`, length `nev`; optional
`ēv` w.r.t. the eigenvectors). Each backward task loads its forward state from the
forward batch's (shared) directory, runs the `solve_k` adjoint, and writes the input
cotangents `(ω̄, ε̄⁻¹)` to `tasks/task_NNNNNN.bwd.h5`. Retrieve them with
[`gradient_result`](@ref).

The backward batch reuses the forward `solver`/`nev`; `backend`, `slurm`, `name`, `dir`,
`submit`, `blocking` behave as in [`deploy_batch`](@ref).
"""
function deploy_backward(forward_batch::BatchInfo, cotangents::AbstractVector;
    name::AbstractString="ad_backward", dir::AbstractString="",
    backend::Symbol=:slurm, slurm::SlurmConfig=SlurmConfig(), submit::Bool=true,
    blocking::Bool=false)
    nfwd = length(forward_batch)
    length(cotangents) == nfwd || throw(ArgumentError(
        "expected one cotangent per forward task ($nfwd), got $(length(cotangents))"))
    nev = forward_batch.manifest["nev"]::Int
    fwd_cluster_dir = _cluster_dir(forward_batch)

    # placeholder setup.jl (the backward worker reads forward state, never make_problem)
    setup = joinpath(tempdir(), "modesweeps_backward_setup.jl")
    write(setup, "make_problem(p) = error(\"backward batches have no setup\")\n")

    prepare = function (bw::BatchInfo)
        bw.manifest["forward_dir"] = fwd_cluster_dir
        _write_manifest(bw.dir, bw.manifest)
        for (i, ct) in enumerate(cotangents)
            _write_cotangents(_task_base(bw, i) * ".ct.h5", ct, nev)
        end
    end

    return deploy_batch(setup, forward_batch.params; name, dir, nev,
        solver=forward_batch.manifest["solver"]::String, backend, slurm, submit, blocking,
        kind="backward", prepare)
end

"""
    gradient_result(backward_batch, i=1) -> (; ω_bar, ε⁻¹_bar)

Load the input cotangents `(ω̄, ε̄⁻¹)` produced by task `i` of a backward batch — the
gradient of the differentiated objective w.r.t. the optical frequency and the
inverse-dielectric field. In ssh mode the `.bwd.h5` file is fetched first.
"""
function gradient_result(batch::BatchInfo, i::Int=1)
    _maybe_fetch_markers!(batch; fields=true)
    path = _task_base(batch, i) * ".bwd.h5"
    isfile(path) || error("no backward result for task $i at $path (has the backward " *
                          "batch finished? in ssh mode, results are fetched automatically)")
    return h5open(path, "r") do f
        (; ω_bar=read(f, "omega_bar"), ε⁻¹_bar=read(f, "eps_inv_bar"))
    end
end

"""
    remote_value_and_gradient(setup_file, p, k̄; ēv=nothing, nev=1, name="ad", kwargs...) -> NamedTuple

Blocking convenience that runs a remote forward pass and then a remote backward pass for
a single parameter set `p`, returning `(; kmags, evecs, ω_bar, ε⁻¹_bar, forward, backward)`
— the primal solution and the gradient of an objective whose cotangent w.r.t. the
per-band `kmags` is `k̄` (length `nev`). Both passes run as SLURM (or `:local`) tasks
that exchange data through the shared filesystem; this call blocks until each completes.

`kwargs` (e.g. `solver`, `solver_kwargs`, `backend`, `slurm`, `dir`) are forwarded to
the forward deployment; the backward pass reuses the forward solver and SLURM config.
"""
function remote_value_and_gradient(setup_file::AbstractString, p::NamedTuple, k̄;
    ēv=nothing, nev::Int=1, name::AbstractString="ad", dir::AbstractString="",
    backend::Symbol=:slurm, slurm::SlurmConfig=SlurmConfig(), kwargs...)
    fdir = isempty(dir) ? "" : dir * "_fwd"
    bdir = isempty(dir) ? "" : dir * "_bwd"
    fwd = deploy_forward(setup_file, [p]; nev, name=name * "_fwd", dir=fdir,
        backend, slurm, blocking=true, kwargs...)
    sol = forward_solution(fwd, 1)
    bw = deploy_backward(fwd, [(; k̄, ēv)]; name=name * "_bwd", dir=bdir,
        backend, slurm, blocking=true)
    g = gradient_result(bw, 1)
    return (; sol.kmags, sol.evecs, g.ω_bar, g.ε⁻¹_bar, forward=fwd, backward=bw)
end
