# Enzyme.jl interface for MaterialDispersion.
#
# The numeric dispersion functions returned by `_f_ε_mats`/`generate_fn`/`ε_fn` are plain
# generated Julia code which Enzyme differentiates natively in forward and reverse mode.
# Here we mark the symbolic model-manipulation and code-generation layer inactive so that
# programs which build dispersion functions inline remain differentiable.

module MaterialDispersionEnzymeExt

using MaterialDispersion
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(MaterialDispersion.generate_fn), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaterialDispersion.generate_array_fn), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaterialDispersion.get_model), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaterialDispersion.has_model), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaterialDispersion.material_name), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(MaterialDispersion.unique_axes), args...; kwargs...) = nothing

end # module
