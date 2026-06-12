# Parameter-sweep specification: Cartesian products of varied parameters, or explicit
# lists of parameter sets. A parameter set is a `NamedTuple` of (typically numeric)
# values; by convention the optical frequency is the parameter `ω` (in μm⁻¹).

"""
    param_grid(; kwargs...) -> Vector{<:NamedTuple}

Build the Cartesian product of parameter values. Each keyword may be a scalar (held
fixed) or an iterable of values (swept). The *first* keyword varies fastest, so e.g.

    param_grid(ω = 0.6:0.01:0.7, w_top = [1.4, 1.7, 2.0])

produces one contiguous frequency sweep per `w_top` value — convenient for combined
frequency × geometry/material sweeps.

A pre-built list of parameter sets (`Vector` of `NamedTuple`s) can be passed to
[`deploy_batch`](@ref) directly instead.
"""
function param_grid(; kwargs...)
    keys_ = collect(keys(kwargs))
    vals = [v isa AbstractString ? [v] : (v isa Union{AbstractArray,Tuple,AbstractRange} ? collect(v) : [v]) for v in values(values(kwargs))]
    out = NamedTuple[]
    for combo in Iterators.product(vals...)
        push!(out, NamedTuple{Tuple(keys_)}(combo))
    end
    return out
end

"normalize a parameter specification to a `Vector{NamedTuple}`"
_normalize_params(params::AbstractVector{<:NamedTuple}) = collect(params)
_normalize_params(params::NamedTuple) = param_grid(; params...)

# JSON round-trip for parameter sets (portable across sessions/machines).
# NamedTuples are written directly so key order is preserved; note JSON cannot
# distinguish `1.0` from `1` on read, so `deploy_batch` additionally stores an exact
# Julia-serialized copy (`params.jls`) which `load_batch` prefers when available.
function _params_to_json(params::AbstractVector{<:NamedTuple})
    return JSON3.write(params)
end

function _params_from_json(str::AbstractString)
    raw = JSON3.read(str)
    return [NamedTuple{Tuple(Symbol.(keys(obj)))}(Tuple(values(obj))) for obj in raw]
end
