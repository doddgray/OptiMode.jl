# Gathering results: assemble per-task summaries into one flat table (one row per
# task × band, with the swept parameters as columns), save it in common tabular
# formats, and re-load it for analysis. Full mode-field data is loaded on demand.

const _SUMMARY_COLS = (:kmag, :neff, :ng, :gvd, :Aeff, :pol_x, :pol_y, :pol_z, :pol_axis)

"""
    gather_batch(batch; partial=true, save=true, formats=(:csv,:tsv,:json), fetch_fields=false)

Collect the results of a deployed batch into a flat summary table: a
`Vector{NamedTuple}` (a Tables.jl row table — pass it to `DataFrame`, `CSV.write`, …)
with one row per task × band, containing the swept parameters, `task`, `band`,
`status`, and the summary quantities `kmag`, `neff` (effective index), `ng` (group
index), `gvd`, `Aeff` (effective area) and the polarization fractions
`pol_x`/`pol_y`/`pol_z` (+ dominant `pol_axis`), plus the Kerr columns `dneff_kerr`
(power-dependent effective-index shift) and `dn_max` (peak index perturbation) for
power-corrected solves (zero for linear solves).

- `partial=true`: gather whatever is finished so far (works while the batch is still
  running); rows for unfinished tasks are omitted and failed tasks get a row per band
  index filled with `NaN` and `status="failed"`. With `partial=false` an error is
  thrown unless all tasks completed successfully.
- `save=true`: also write the table to `<batchdir>/summary.<fmt>` for each format in
  `formats` (supported: `:csv`, `:tsv`, `:json`); reload with [`load_summary`](@ref).
- `fetch_fields=true`: in ssh mode, also transfer the (large) per-task HDF5 field
  files from the cluster, not just the summaries.

Use [`load_fields`](@ref)`(batch, i)` for the full mode-field data of one task
(batches deployed with `save_fields=true`).
"""
function gather_batch(batch::BatchInfo; partial::Bool=true, save::Bool=true,
    formats=(:csv, :tsv, :json), fetch_fields::Bool=false)
    _maybe_fetch_markers!(batch; fields=fetch_fields)
    st = batch_status(batch; verbose=false)
    if !partial && (st.done < st.total)
        error("batch incomplete ($(st.done)/$(st.total) done, $(st.failed) failed); " *
              "pass partial=true to gather partial results")
    end
    nev = batch.manifest["nev"]::Int
    pkeys = Tuple(Symbol.(batch.manifest["param_keys"]))
    rows = NamedTuple[]
    for i in 1:length(batch)
        base = _task_base(batch, i)
        pvals = NamedTuple{pkeys}(Tuple(batch.params[i][k] for k in pkeys))
        if isfile(base * ".done") && isfile(base * ".json")
            summary = JSON3.read(read(base * ".json", String))
            for bd in summary.bands
                push!(rows, merge((; task=i), pvals, (;
                    band=Int(bd.band), status="done",
                    kmag=Float64(bd.kmag), neff=Float64(bd.neff), ng=Float64(bd.ng),
                    gvd=Float64(bd.gvd), Aeff=Float64(bd.Aeff),
                    pol_x=Float64(bd.pol_x), pol_y=Float64(bd.pol_y), pol_z=Float64(bd.pol_z),
                    pol_axis=Int(bd.pol_axis),
                    dneff_kerr=haskey(bd, :dneff_kerr) ? Float64(bd.dneff_kerr) : 0.0,
                    dn_max=haskey(bd, :dn_max) ? Float64(bd.dn_max) : 0.0,
                )))
            end
        elseif isfile(base * ".failed")
            for b in 1:nev
                push!(rows, merge((; task=i), pvals, (;
                    band=b, status="failed",
                    kmag=NaN, neff=NaN, ng=NaN, gvd=NaN, Aeff=NaN,
                    pol_x=NaN, pol_y=NaN, pol_z=NaN, pol_axis=0,
                    dneff_kerr=NaN, dn_max=NaN,
                )))
            end
        end
    end
    if save && !isempty(rows)
        save_summary(rows, joinpath(batch.dir, "summary"); formats)
    end
    return rows
end

"""
    save_summary(rows, basepath; formats=(:csv,:tsv,:json)) -> Vector{String}

Write a gathered summary table to `basepath.<ext>` in each requested tabular format
(`:csv`, `:tsv`, `:json`). Returns the written paths.
"""
function save_summary(rows, basepath::AbstractString; formats=(:csv, :tsv, :json))
    paths = String[]
    tbl = Tables.columntable(rows)
    for fmt in formats
        if fmt === :csv
            p = basepath * ".csv"
            CSV.write(p, tbl)
        elseif fmt === :tsv
            p = basepath * ".tsv"
            CSV.write(p, tbl; delim='\t')
        elseif fmt === :json
            p = basepath * ".json"
            write(p, JSON3.write(rows))
        else
            throw(ArgumentError("unsupported summary format :$fmt (use :csv, :tsv or :json)"))
        end
        push!(paths, p)
    end
    return paths
end

"""
    load_summary(path) -> Vector{NamedTuple}

Re-load a summary table written by [`save_summary`](@ref)/[`gather_batch`](@ref); the
format is inferred from the file extension (`.csv`, `.tsv`, `.json`). Returns a
Tables.jl-compatible row table.
"""
function load_summary(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".csv"
        return Tables.rowtable(CSV.File(path))
    elseif ext == ".tsv"
        return Tables.rowtable(CSV.File(path; delim='\t'))
    elseif ext == ".json"
        raw = JSON3.read(read(path, String))
        return [NamedTuple{Tuple(Symbol.(keys(o)))}(Tuple(values(o))) for o in raw]
    else
        throw(ArgumentError("unsupported summary file extension $ext"))
    end
end

"""
    load_fields(batch, i) -> NamedTuple

Load the full mode-field data of task `i` from a batch deployed with
`save_fields=true`: `(; ω, kmags, evecs, Es, grid_Δ, grid_N)` where `evecs` is a
`Vector` of flat eigenvectors and `Es` a `Vector` of `(3, Ns...)` complex E-field
arrays (one per band). In ssh mode, fetch the field files first with
`gather_batch(batch; fetch_fields=true)`.
"""
function load_fields(batch::BatchInfo, i::Int)
    path = _task_base(batch, i) * ".h5"
    isfile(path) || error("no field data for task $i at $path — was the batch deployed " *
                          "with save_fields=true (and, in ssh mode, gathered with fetch_fields=true)?")
    return h5open(path, "r") do f
        ω = read(f, "omega")
        kmags = read(f, "kmags")
        evm = read(f, "evecs_re") .+ im .* read(f, "evecs_im")
        Earr = read(f, "E_re") .+ im .* read(f, "E_im")
        nev = length(kmags)
        evecs = [evm[:, b] for b in 1:nev]
        Es = [collect(selectdim(Earr, ndims(Earr), b)) for b in 1:nev]
        (; ω, kmags, evecs, Es, grid_Δ=read(f, "grid_Δ"), grid_N=read(f, "grid_N"))
    end
end
