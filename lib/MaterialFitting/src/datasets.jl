export IndexDataset, index_dataset

"""
    IndexDataset(λ, n; T=nothing, axis="", label="", source="")

Refractive-index data for a single crystal axis at a single temperature: vacuum
wavelengths `λ` (μm, strictly increasing) and the corresponding real refractive index
`n`. Optional metadata:

- `T`        — temperature in °C (`nothing` if unspecified / room temperature),
- `axis`     — axis label for anisotropic materials (`"o"`, `"e"`, `"x"`, `"y"`, `"z"`, …),
- `label`    — human-readable label used in plot legends,
- `source`   — provenance string (a RefractiveIndex.INFO URL, a file path, `"user"`, …).

This is the common currency of the package: data pulled from RefractiveIndex.INFO
(tabulated or evaluated from a published formula) and user-supplied data are both
represented as `IndexDataset`s and fed to [`fit_sellmeier`](@ref).
"""
struct IndexDataset
    λ::Vector{Float64}
    n::Vector{Float64}
    T::Union{Float64,Nothing}
    axis::String
    label::String
    source::String
end

function IndexDataset(λ::AbstractVector{<:Real}, n::AbstractVector{<:Real};
                      T=nothing, axis::AbstractString="", label::AbstractString="",
                      source::AbstractString="user")
    length(λ) == length(n) || throw(ArgumentError("λ and n must have equal length (got $(length(λ)) and $(length(n)))"))
    length(λ) ≥ 2 || throw(ArgumentError("need at least 2 data points to fit a dispersion model"))
    p = sortperm(λ)
    λs = Float64.(collect(λ))[p]
    ns = Float64.(collect(n))[p]
    all(>(0), diff(λs)) || throw(ArgumentError("wavelengths must be distinct"))
    all(>(0), ns) || throw(ArgumentError("refractive indices must be positive"))
    Tv = T === nothing ? nothing : Float64(T)
    return IndexDataset(λs, ns, Tv, String(axis), String(label), String(source))
end

"""
    index_dataset(λ, n; kwargs...)

Convenience constructor for an [`IndexDataset`](@ref) from user-supplied wavelength (μm)
and refractive-index vectors. Accepts the same keyword metadata as the type.
"""
index_dataset(λ, n; kwargs...) = IndexDataset(λ, n; kwargs...)

"""
    index_dataset(path::AbstractString; delim=nothing, λcol=1, ncol=2, header=false, kwargs...)

Load an [`IndexDataset`](@ref) from a two-column (λ μm, n) text/CSV file. `delim`
defaults to whitespace; set e.g. `delim=','` for CSV. `λcol`/`ncol` select columns and
`header=true` skips the first row.
"""
function index_dataset(path::AbstractString; delim=nothing, λcol::Int=1, ncol::Int=2,
                       header::Bool=false, source=path, kwargs...)
    raw = delim === nothing ? readdlm(path; skipstart=header ? 1 : 0) :
                              readdlm(path, delim; skipstart=header ? 1 : 0)
    λ = Float64.(raw[:, λcol])
    n = Float64.(raw[:, ncol])
    return IndexDataset(λ, n; source=String(source), kwargs...)
end

Base.length(ds::IndexDataset) = length(ds.λ)
λrange(ds::IndexDataset) = (first(ds.λ), last(ds.λ))

function Base.show(io::IO, ds::IndexDataset)
    lo, hi = λrange(ds)
    tstr = ds.T === nothing ? "" : @sprintf(", T=%.1f°C", ds.T)
    axstr = isempty(ds.axis) ? "" : ", axis=$(ds.axis)"
    @printf(io, "IndexDataset(%d pts, λ∈[%.3f,%.3f] μm%s%s)", length(ds), lo, hi, tstr, axstr)
end
