export refractiveindex_dataset, RefractiveIndexEntry, parse_refractiveindex_yaml,
    refractiveindex_url_to_yaml_urls

#### RefractiveIndex.INFO database interface #################################################
#
# RefractiveIndex.INFO stores each material entry as a YAML record with one or more `DATA`
# blocks. Each block is either *tabulated* (whitespace-separated `λ n [k]` rows, λ in μm)
# or a *formula* (`formula 1` … `formula 9`) with a `coefficients` string and a
# `wavelength_range`. We parse the record, evaluate whichever block covers the requested
# range, and return an [`IndexDataset`](@ref) of `(λ, n)` samples — unifying tabulated data
# and published Sellmeier/Cauchy/… fits into a single representation that
# [`fit_sellmeier`](@ref) can refit to the desired number of terms and validity range.

"""
    RefractiveIndexEntry

Parsed RefractiveIndex.INFO record. `blocks` is a vector of `NamedTuple`s, one per `DATA`
entry, each with fields `kind` (`:tabulated` or `:formula`), `formula` (the integer
formula id, or `0`), `coefficients::Vector{Float64}`, `λ::Vector{Float64}`,
`n::Vector{Float64}` (for tabulated blocks), and `λrange::Tuple{Float64,Float64}`.
`references`/`comments` carry the record's bibliographic metadata.
"""
struct RefractiveIndexEntry
    blocks::Vector{NamedTuple}
    references::String
    comments::String
    source::String
end

# ---- formula evaluators (RefractiveIndex.INFO formula 1–9), λ in μm -------------------
# Each returns the real refractive index n(λ). `c` is the coefficient vector.

@inline _padded(c, i) = i ≤ length(c) ? c[i] : 0.0

function _ri_formula(id::Int, c::Vector{Float64}, λ::Real)
    λ² = λ^2
    if id == 1                      # Sellmeier (Cᵢ as wavelengths, μm)
        s = 1.0 + _padded(c, 1)
        @inbounds for k in 2:2:length(c)-1
            s += c[k] * λ² / (λ² - c[k+1]^2)
        end
        return sqrt(s)
    elseif id == 2                  # Sellmeier-2 (Cᵢ as wavelengths², μm²)
        s = 1.0 + _padded(c, 1)
        @inbounds for k in 2:2:length(c)-1
            s += c[k] * λ² / (λ² - c[k+1])
        end
        return sqrt(s)
    elseif id == 3                  # polynomial: n² = c₁ + Σ cₖ λ^{cₖ₊₁}
        s = _padded(c, 1)
        @inbounds for k in 2:2:length(c)-1
            s += c[k] * λ^c[k+1]
        end
        return sqrt(s)
    elseif id == 4                  # RefractiveIndex.INFO formula 4
        s = _padded(c, 1)
        s += _padded(c,2) * λ^_padded(c,3) / (λ² - _padded(c,4)^_padded(c,5))
        s += _padded(c,6) * λ^_padded(c,7) / (λ² - _padded(c,8)^_padded(c,9))
        @inbounds for k in 10:2:length(c)-1
            s += c[k] * λ^c[k+1]
        end
        return sqrt(s)
    elseif id == 5                  # Cauchy: n = c₁ + Σ cₖ λ^{cₖ₊₁}
        s = _padded(c, 1)
        @inbounds for k in 2:2:length(c)-1
            s += c[k] * λ^c[k+1]
        end
        return s
    elseif id == 6                  # gases: n = 1 + c₁ + Σ cₖ /(cₖ₊₁ − λ⁻²)
        s = 1.0 + _padded(c, 1)
        @inbounds for k in 2:2:length(c)-1
            s += c[k] / (c[k+1] - λ^(-2))
        end
        return s
    elseif id == 7                  # Herzberger
        s = _padded(c,1) + _padded(c,2)/(λ² - 0.028) + _padded(c,3)/(λ² - 0.028)^2
        s += _padded(c,4)*λ² + _padded(c,5)*λ^4 + _padded(c,6)*λ^6
        return s
    elseif id == 8                  # (n²−1)/(n²+2) form
        b = _padded(c,1) + _padded(c,2)*λ²/(λ² - _padded(c,3)) + _padded(c,4)*λ²
        return sqrt((1 + 2b) / (1 - b))
    elseif id == 9                  # exciton
        s = _padded(c,1) + _padded(c,2)/(λ² - _padded(c,3)) +
            _padded(c,4)*(λ - _padded(c,5)) / ((λ - _padded(c,5))^2 + _padded(c,6))
        return sqrt(s)
    else
        throw(ArgumentError("unsupported RefractiveIndex.INFO formula id $id"))
    end
end

# ---- YAML parsing --------------------------------------------------------------------

_floats(s::AbstractString) = parse.(Float64, split(strip(s)))

function _parse_block(d::AbstractDict)
    typ = String(get(d, "type", ""))
    if startswith(typ, "tabulated")
        rows = [_floats(l) for l in split(strip(String(d["data"])), '\n') if !isempty(strip(l))]
        λ = Float64[r[1] for r in rows]
        n = Float64[r[2] for r in rows]            # column 2 is n (k, if present, is col 3)
        p = sortperm(λ)
        return (; kind=:tabulated, formula=0, coefficients=Float64[], λ=λ[p], n=n[p],
                  λrange=(minimum(λ), maximum(λ)))
    elseif startswith(typ, "formula")
        id = parse(Int, strip(replace(typ, "formula" => "")))
        c = _floats(String(d["coefficients"]))
        wr = haskey(d, "wavelength_range") ? _floats(String(d["wavelength_range"])) : [0.0, Inf]
        return (; kind=:formula, formula=id, coefficients=c, λ=Float64[], n=Float64[],
                  λrange=(wr[1], wr[2]))
    else
        return nothing
    end
end

"""
    parse_refractiveindex_yaml(text; source="") -> RefractiveIndexEntry

Parse the YAML text of a RefractiveIndex.INFO record into a [`RefractiveIndexEntry`](@ref).
"""
function parse_refractiveindex_yaml(text::AbstractString; source::AbstractString="")
    doc = YAML.load(text)
    raw = get(doc, "DATA", nothing)
    raw === nothing && throw(ArgumentError("YAML record has no DATA block"))
    rawvec = raw isa AbstractVector ? raw : [raw]
    blocks = NamedTuple[]
    for d in rawvec
        b = _parse_block(d)
        b === nothing || push!(blocks, b)
    end
    isempty(blocks) && throw(ArgumentError("no usable tabulated/formula DATA blocks found"))
    refs = String(get(doc, "REFERENCES", ""))
    coms = String(get(doc, "COMMENTS", ""))
    return RefractiveIndexEntry(blocks, refs, coms, String(source))
end

# ---- URL handling --------------------------------------------------------------------

"""
    refractiveindex_url_to_yaml_urls(url) -> Vector{String}

Map a refractiveindex.info page URL (`…/?shelf=main&book=SiO2&page=Malitson`) to candidate
raw-YAML URLs in the public database (the site mirror and the GitHub mirror, current and
legacy layouts). A direct `.yml`/`.yaml` URL is returned unchanged.
"""
function refractiveindex_url_to_yaml_urls(url::AbstractString)
    u = String(url)
    (endswith(u, ".yml") || endswith(u, ".yaml")) && return [u]
    q = occursin('?', u) ? split(u, '?'; limit=2)[2] : ""
    params = Dict{String,String}()
    for kv in split(q, '&'; keepempty=false)
        p = split(kv, '='; limit=2)
        length(p) == 2 && (params[p[1]] = p[2])
    end
    shelf = get(params, "shelf", ""); book = get(params, "book", ""); page = get(params, "page", "")
    (isempty(shelf) || isempty(book) || isempty(page)) &&
        throw(ArgumentError("could not extract shelf/book/page from URL: $u (pass a direct .yml URL or a local file path instead)"))
    rel = "$shelf/$book/$page.yml"
    return [
        "https://refractiveindex.info/database/data-nk/$rel",
        "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/$rel",
        "https://refractiveindex.info/database/data/$rel",
        "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/$rel",
    ]
end

_is_url(s::AbstractString) = startswith(s, "http://") || startswith(s, "https://")

"""
    fetch_refractiveindex_yaml(url) -> (text, resolved_url)

Download the YAML record for a RefractiveIndex.INFO page or direct YAML URL, trying the
candidate database locations in order. Requires network access.
"""
function fetch_refractiveindex_yaml(url::AbstractString)
    candidates = refractiveindex_url_to_yaml_urls(url)
    errs = String[]
    for c in candidates
        try
            io = IOBuffer()
            Downloads.download(c, io)
            return (String(take!(io)), c)
        catch e
            push!(errs, "$(c) → $(sprint(showerror, e))")
        end
    end
    error("failed to fetch RefractiveIndex.INFO data for $url; tried:\n  " * join(errs, "\n  "))
end

"""
    refractiveindex_entry(src) -> RefractiveIndexEntry

Load a RefractiveIndex.INFO record from a page URL, a direct `.yml` URL, or a local YAML
file path.
"""
function refractiveindex_entry(src::AbstractString)
    if _is_url(src)
        text, resolved = fetch_refractiveindex_yaml(src)
        return parse_refractiveindex_yaml(text; source=resolved)
    else
        return parse_refractiveindex_yaml(read(src, String); source=src)
    end
end

# ---- entry → IndexDataset ------------------------------------------------------------

"""
    refractiveindex_dataset(src; n_points=300, λ_range=nothing, axis="", T=nothing,
                            label=nothing, block=nothing) -> IndexDataset

Build an [`IndexDataset`](@ref) from a RefractiveIndex.INFO entry. `src` may be a page URL
(`https://refractiveindex.info/?shelf=…&book=…&page=…`), a direct `.yml` URL, or a local
YAML file path.

The first `DATA` block covering `λ_range` is used (override with `block`); tabulated data
is returned directly (restricted to `λ_range`), a formula block is evaluated at `n_points`
wavelengths spanning `λ_range` (default: the block's own validity range). Use `axis`/`T`
to tag the dataset for anisotropic/temperature-dependent model building.
"""
function refractiveindex_dataset(src::AbstractString; n_points::Int=300, λ_range=nothing,
                                 axis::AbstractString="", T=nothing, label=nothing, block=nothing)
    entry = src isa RefractiveIndexEntry ? src : refractiveindex_entry(src)
    return refractiveindex_dataset(entry; n_points, λ_range, axis, T, label, block)
end

function refractiveindex_dataset(entry::RefractiveIndexEntry; n_points::Int=300, λ_range=nothing,
                                 axis::AbstractString="", T=nothing, label=nothing, block=nothing)
    blk = if block !== nothing
        entry.blocks[block]
    elseif λ_range !== nothing
        idx = findfirst(b -> b.λrange[1] ≤ λ_range[1] && λ_range[2] ≤ b.λrange[2], entry.blocks)
        entry.blocks[idx === nothing ? 1 : idx]
    else
        entry.blocks[1]
    end
    lbl = label === nothing ? _entry_label(entry) : String(label)
    if blk.kind == :tabulated
        λ, n = blk.λ, blk.n
        if λ_range !== nothing
            m = λ_range[1] .≤ λ .≤ λ_range[2]
            count(m) ≥ 2 || throw(ArgumentError("requested λ_range $(λ_range) contains <2 tabulated points"))
            λ, n = λ[m], n[m]
        end
        return IndexDataset(λ, n; T, axis, label=lbl, source=entry.source)
    else
        lo, hi = λ_range === nothing ? blk.λrange : λ_range
        (isfinite(lo) && isfinite(hi) && hi > lo) ||
            throw(ArgumentError("formula block needs a finite λ_range; pass λ_range=(lo,hi)"))
        λ = collect(range(lo, hi; length=n_points))
        n = [_ri_formula(blk.formula, blk.coefficients, l) for l in λ]
        return IndexDataset(λ, n; T, axis, label=lbl, source=entry.source)
    end
end

function _entry_label(entry::RefractiveIndexEntry)
    s = entry.source
    m = match(r"([^/]+)/([^/]+)\.ya?ml", s)
    m === nothing ? "RefractiveIndex.INFO" : "$(m.captures[1]) ($(m.captures[2]))"
end
