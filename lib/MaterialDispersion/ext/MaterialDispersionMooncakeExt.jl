# Mooncake.jl interface for MaterialDispersion.
#
# The numeric dispersion functions returned by `_f_ε_mats`/`generate_fn`/`ε_fn` are plain
# generated Julia code (scalar arithmetic + array construction), which Mooncake
# differentiates natively in both forward-over-reverse and reverse mode. What we add here
# are zero-derivative markers for the symbolic model-manipulation and code-generation
# layer, so user programs that build dispersion functions inline remain differentiable.

module MaterialDispersionMooncakeExt

using MaterialDispersion
using MaterialDispersion: AbstractMaterial
using Mooncake
using Mooncake: @zero_adjoint, MinimalCtx

# Symbolic model construction & code generation: not differentiable, treat as constants.
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.generate_fn),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.generate_array_fn),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.get_model),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.has_model),AbstractMaterial,Symbol}
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.material_name),Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(MaterialDispersion.unique_axes),AbstractMaterial}

end # module
