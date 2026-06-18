# Enzyme.jl interface for EigenmodeExpansion.
#
# The EME forward computation is matmuls + linear solves + the OptiMode mode
# solver (whose Enzyme rule is imported in MaxwellEigenmodes' own extension), so
# Enzyme differentiates it natively. We only need to mark the discrete GDS/cell
# bookkeeping as inactive so Enzyme does not attempt to differentiate it.

module EigenmodeExpansionEnzymeExt

using EigenmodeExpansion
using Enzyme
using Enzyme: EnzymeRules

EnzymeRules.inactive(::typeof(EigenmodeExpansion.read_gds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(EigenmodeExpansion.write_gds), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(EigenmodeExpansion.polygon_transverse_intervals), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(EigenmodeExpansion.cross_section_at), args...; kwargs...) = nothing
EnzymeRules.inactive(::typeof(EigenmodeExpansion.build_cells), args...; kwargs...) = nothing

end # module
