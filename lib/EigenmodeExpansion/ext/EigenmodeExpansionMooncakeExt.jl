# Mooncake.jl interface for EigenmodeExpansion.
#
# The differentiable EME pipeline (smoothing, mode solve, overlaps, S-matrix
# linear algebra) differentiates through the per-package rules already bridged to
# Mooncake; here we only mark the discrete GDS/cell bookkeeping as
# zero-derivative.

module EigenmodeExpansionMooncakeExt

using EigenmodeExpansion
using Mooncake
using Mooncake: @zero_adjoint, MinimalCtx

@zero_adjoint MinimalCtx Tuple{typeof(EigenmodeExpansion.read_gds),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(EigenmodeExpansion.write_gds),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(EigenmodeExpansion.polygon_transverse_intervals),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(EigenmodeExpansion.cross_section_at),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(EigenmodeExpansion.build_cells),Vararg}

end # module
