# Reactant.jl interface for MaterialDispersion.
#
# The generated dispersion functions are pure scalar/array Julia programs, so they can be
# traced and compiled by Reactant (e.g. to evaluate and differentiate dielectric tensors
# for batches of frequency/temperature points on CPU/GPU/TPU via XLA). This extension
# provides a convenience wrapper that compiles a generated dispersion function for a
# sample parameter vector.

module MaterialDispersionReactantExt

using MaterialDispersion
using Reactant

"""
    reactant_compile_dispersion(f, p0::AbstractVector{<:Real})

Compile the generated dispersion function `f` (e.g. from `_f_ε_mats`) with Reactant for
parameter vectors shaped like `p0`. Returns `(f_compiled, p_ra)` where `p_ra` is the
Reactant array corresponding to `p0`; call `f_compiled(Reactant.to_rarray(p))` thereafter.
"""
function MaterialDispersion.reactant_compile_dispersion(f, p0::AbstractVector{<:Real})
    p_ra = Reactant.to_rarray(collect(float.(p0)))
    f_compiled = Reactant.@compile f(p_ra)
    return f_compiled, p_ra
end

end # module
