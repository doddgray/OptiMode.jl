# Dispersion models for NLO-relevant materials adapted from
# https://github.com/doddgray/optics_modeling/blob/master/nlo/NLO_tools.py

################################################################################
#            Temperature Dependent Index, Group Index and GVD models           #
#                        for phase-matching calculations                       #
################################################################################
# ng(fₙ::Function) = λ -> ( (n,n_pb) = Zygote.pullback(fₙ,λ); ( n - λ * n_pb(1)[1] ) )
export λ, Material, materials, n, ng, gvd, ε, ε⁻¹ # , fn, fng, fgvd, fε, fε⁻¹,

c = Unitful.c0      # Unitful.jl speed of light
@parameters λ, T
Dλ = Differential(λ)
DT = Differential(T)
ng(n_sym::Num) = n_sym - λ * expand_derivatives(Dλ(n_sym))
gvd(n_sym::Num) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym))) # gvd = uconvert( ( 1 / ( 2π * c^2) ) * _gvd(lm_um,T_C)u"μm", u"fs^2 / mm" )

struct Material{T}
	ε::SMatrix{3,3,T,9}
	fε::Function
end

# Material(x::AbstractArray) = Material(ε_tensor(x))
# Material(x::Number) = Material(ε_tensor(x))
# Material(x::Tuple) = Material(ε_tensor(x))
Material(x,f) = Material(ε_tensor(x),f)
Material(mat::Material) = mat	# maybe
# import Base.convert
# convert(::Type{Material}, x) = Material(x)

materials(shapes::Vector{<:GeometryPrimitives.Shape}) = Zygote.@ignore(unique(Material.(getfield.(shapes,:data)))) # unique!(getfield.(shapes,:data))


#### Fixed index Material constructors and methods ####

# Material(n::T where T<:Real) = Material{T}(ε_tensor(n))							# isotropic fixed index Material
# Material(ns::Tuple{T,T,T} where T<:Real) = Material{T}(ε_tensor(ns))			# anisotropic fixed index Material
# Material(ε::AbstractMatrix{T} where T<:Real) = Material{T}(ε_tensor(ε))			# fixed dielectric tensor material

ε(mat::Material) = mat.ε
ε⁻¹(mat::Material) = inv(mat.ε)
n(mat::Material) = sqrt.(diag(mat.ε))
n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])


#### Symbolic Material model methods ####
ng(mat::Material) = ng.(n(mat))
ng(mat::Material,axind::Int) = ng(n(mat,axind))
gvd(mat::Material) = gvd.(n(mat))
gvd(mat::Material,axind::Int) = gvd(n(mat,axind))

# ### methods to generate numerical functions for symbolic material dispersions ###
# # non-mutating numerical function generators #
# fε(mat::Material{Num}) = eval(build_function(ε(mat),λ)[1])
# fε⁻¹(mat::Material{Num}) = eval(build_function(ε⁻¹(mat),λ)[1])
# fn(mat::Material{Num}) = eval(build_function(n(mat),λ)[1])
# fn(mat::Material{Num},axind::Int) = eval(build_function(n(mat,axind),λ)[1])
# fng(mat::Material{Num}) = eval(build_function(ng(mat),λ)[1])
# fng(mat::Material{Num},axind::Int) = eval(build_function(ng(mat,axind),λ)[1])
# fgvd(mat::Material{Num}) = eval(build_function(gvd(mat),λ)[1])
# fgvd(mat::Material{Num},axind::Int) = eval(build_function(gvd(mat,axind),λ)[1])
# # mutating numerical function generators #
# fε!(mat::Material{Num}) = eval(build_function(ε(mat),λ)[2])
# fε⁻¹!(mat::Material{Num}) = eval(build_function(ε⁻¹(mat),λ)[2])
# fn!(mat::Material{Num}) = eval(build_function(n(mat),λ)[2])
# fn!(mat::Material{Num},axind::Int) = eval(build_function(n(mat,axind),λ)[2])
# fng!(mat::Material{Num}) = eval(build_function(ng(mat),λ)[2])
# fng!(mat::Material{Num},axind::Int) = eval(build_function(ng(mat,axind),λ)[2])
# fgvd!(mat::Material{Num}) = eval(build_function(gvd(mat),λ)[2])
# fgvd!(mat::Material{Num},axind::Int) = eval(build_function(gvd(mat,axind),λ)[2])


# # Fallback methods for fixed-index Materials
# ng(mat::Material{T} where T<:Real) = zeros(T,3)
# ng(mat::Material{T},axind::Int) where T<:Real = zero(T)
# gvd(mat::Material{T} where T<:Real) = zeros(T,3)
# gvd(mat::Material{T},axind::Int) where T<:Real = zero(T)
#
# fε(mat::Material) = x->ε(mat)
# fε⁻¹(mat::Material) = x->ε⁻¹(mat)
# fn(mat::Material) = x->n(mat)
# fn(mat::Material,axind::Int) = x->n(mat,axind)
# fng(mat::Material) = x->ng(mat)
# fng(mat::Material,axind::Int) = x->ng(mat,axind)
# fgvd(mat::Material) = x->gvd(mat)
# fgvd(mat::Material,axind::Int) = x->gvd(mat,axind)


################################################################################
#                                Load Materials                                #
################################################################################
include("material_lib/MgO_LiNbO3.jl")
include("material_lib/SiO2.jl")
include("material_lib/Si3N4.jl")
# include("material_lib/LiB3O5.jl")
# include("material_lib/silicon.jl")
# include("material_lib/GaAs.jl")
# include("material_lib/MgF2.jl")
# include("material_lib/HfO2.jl")
