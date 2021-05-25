# Dispersion models for NLO-relevant materials adapted from
# https://github.com/doddgray/optics_modeling/blob/master/nlo/NLO_tools.py

"""
################################################################################
#																			   #
#							Physical Material Models						   #
#																			   #
################################################################################
"""

# ng(fₙ::Function) = λ -> ( (n,n_pb) = Zygote.pullback(fₙ,λ); ( n - λ * n_pb(1)[1] ) )
export λ, Material, materials, n, ng, gvd, ε, ε⁻¹, unique_axes, plot_data, nn̂g, nĝvd, rotate #, fn, fng, fgvd, fε, fε⁻¹,

c = Unitful.c0      # Unitful.jl speed of light
@parameters λ, T
Dλ = Differential(λ)
DT = Differential(T)
ng(n_sym::Num) = n_sym - λ * expand_derivatives(Dλ(n_sym))
gvd(n_sym::Num) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym))) # gvd = uconvert( ( 1 / ( 2π * c^2) ) * _gvd(lm_um,T_C)u"μm", u"fs^2 / mm" )

struct Material{T}
	ε::T
	fε::Function
	fng::Function
	fgvd::Function
	# fχ⁽²⁾::Function
end

# Material(x::AbstractArray) = Material(ε_tensor(x))
# Material(x::Number) = Material(ε_tensor(x))
# Material(x::Tuple) = Material(ε_tensor(x))


# Material(x,fe,fng) = Material(ε_tensor(x),fe,fng)
# Material(x) = Material(ε_tensor(x),lm->ε_tensor(x),lm->sqrt.(ε_tensor(x)))
# Material(mat::Material) = mat	# maybe


Material(x) = Material(lm->ε_tensor(x),lm->sqrt.(ε_tensor(x)))
Material(mat::Material) = mat	# maybe



# import Base.convert
# convert(::Type{Material}, x) = Material(x)

materials(shapes::Vector{<:GeometryPrimitives.Shape}) = Zygote.@ignore(unique(Material.(getfield.(shapes,:data)))) # unique!(getfield.(shapes,:data))


#### Fixed index Material constructors and methods ####

# Material(n::T where T<:Real) = Material{T}(ε_tensor(n))							# isotropic fixed index Material
# Material(ns::Tuple{T,T,T} where T<:Real) = Material{T}(ε_tensor(ns))			# anisotropic fixed index Material
# Material(ε::AbstractMatrix{T} where T<:Real) = Material{T}(ε_tensor(ε))			# fixed dielectric tensor material

# ε(mat::Material) = mat.ε
# ε⁻¹(mat::Material) = inv(mat.ε)
# n(mat::Material) = sqrt.(diag(mat.ε))
# n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])
nn̂g(mat::Material,lm::Real) = getfield(mat,:fng)(lm) * sqrt.(getfield(mat,:fε)(lm))
nĝvd(mat::Material,lm::Real) = getfield(mat,:fgvd)(lm) * sqrt.(getfield(mat,:fε)(lm))

#### Symbolic Material model methods ####
# ng(mat::Material) = ng.(n(mat))
# ng(mat::Material,axind::Int) = ng(n(mat,axind))
# gvd(mat::Material) = gvd.(n(mat))
# gvd(mat::Material,axind::Int) = gvd(n(mat,axind))

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
# ng(mat::Material{<:Real}) = n(mat)
# ng(mat::Material{<:Real},axind::Int) = n(mat,axind)
# function gvd(mat::Material{T})  where T<:Real
# 	zeros(T,3)
# end
# function gvd(mat::Material{T},axind::Int) where T<:Real
# 	zero(T)
# end

# fε(mat::Material) = x->ε(mat)
# fε⁻¹(mat::Material) = x->ε⁻¹(mat)
# fn(mat::Material) = x->n(mat)
# fn(mat::Material,axind::Int) = x->n(mat,axind)
# fng(mat::Material) = x->ng(mat)
# fng(mat::Material,axind::Int) = x->ng(mat,axind)
# fgvd(mat::Material) = x->gvd(mat)
# fgvd(mat::Material,axind::Int) = x->gvd(mat,axind)



"""
################################################################################
#																			   #
#							   	  Rotations							   		   #
#																			   #
################################################################################
"""

function rotate(χ::AbstractArray{T,2},𝓡::TR) where {T<:Real, TR<:StaticMatrix{3,3}}
	@tullio χᵣ[i,j] := 𝓡[a,i] * 𝓡[b,j] * χ[a,b]  fastmath=true
end

function rotate(χ::AbstractArray{T,3},𝓡::TR) where {T<:Real, TR<:StaticMatrix{3,3}}
	@tullio χᵣ[i,j,k] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * χ[a,b,c]  fastmath=true
end

function rotate(χ::AbstractArray{T,4},𝓡::TR) where {T<:Real, TR<:StaticMatrix{3,3}}
	@tullio χᵣ[i,j,k,l] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * 𝓡[d,l] * χ[a,b,c,d]  fastmath=true
end

# struct RotatedMaterial{T,TR}
# 	parent::Material{T}
# 	fε::Function
# 	fng::Function
# 	fgvd::Function
# end

"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

function unique_axes(mat::Material)
	e11,e22,e33 = diag(getfield(mat,:ε))
	if isequal(e11,e22)
		isequal(e11,e33) ? (return ( [1,], [""] )) : (return ( [1,3], ["₁,₂","₃"] )) # 1 == 2 == 3 (isotropic) : 1 == 2 != 3 (uniaxial)
	elseif isequal(e22,e33)
		return ( [1,2], ["₁","₂,₃"] )	# 1 != 2 == 3 (uniaxial)
	else
		isequal(e11,e33) ? (return ( [1,2], ["₁,₃","₂"] )) : (return ( [1,2,3], ["₁","₂","₃"] )) # 1 == 3 != 2 (uniaxial) : 1 != 2 != 3 (biaxial)
	end
end



"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""

function plot_data(mats::AbstractVector{<:Material})
	fes = getfield.(mats,:fε)
	axind_axstr_unq = unique_axes.(mats)
	axind_unq = getindex.(axind_axstr_unq,1)
	axstr_unq = getindex.(axind_axstr_unq,2)
	fns = vcat(map((ff,as)->[(x->sqrt(ff(x)[a,a])) for a in as ], fes, axind_unq)...)
	mat_names = chop.(String.(nameof.(fes)),head=2,tail=0)	# remove "ε_" from function names
	names = "n" .* vcat([.*(axstr_unq[i], " (", mat_names[i],")") for i=1:length(mats)]...) # "n, n_i or n_i,j (Material)" for all unique axes and materials
	return fns, names
end
plot_data(mat::Material) = plot_data([mat,])
plot_data(mats::NTuple{N,<:Material} where N) = plot_data([mats...])

function uplot(x::Union{Material, AbstractVector{<:Material}, NTuple{N,<:Material} };
		xlim=[0.5,1.8], xlabel="λ [μm]", ylabel="n", kwargs...)  where N
	fns, name = plot_data(x)
	UnicodePlots.lineplot(fns, xlim[1], xlim[2];
	 	xlim,
		ylim=map((a,b)->a(b,digits=1),(floor,ceil),ylims(fns;xlims=xlim)),
		name,
		xlabel,
		ylabel,
		kwargs...
		)
end

function uplot!(plt::UnicodePlots.Plot,x::Union{Material, AbstractVector{<:Material}, NTuple{N,<:Material} };
		xlim=[0.5,1.8], xlabel="λ [μm]", ylabel="n")  where N
	fns, name = plot_data(x)
	UnicodePlots.lineplot!(plt, fns; name ) #, xlim[1], xlim[2];
	 	# xlim,
		# ylim=round.( ylims(plt,ylims(fns;xlims=xlim)) ,digits=1),
		# name,
		# xlabel,
		# ylabel,
		# )
end
#
import Base: show
Base.show(io::IO, ::MIME"text/plain", mat::Material) = uplot(mat;title="MIME version") #print(io, "Examplary instance of Material\n", m.x, " ± ", m.y)
Base.show(io::IO, mat::Material) = uplot(mat) #print(io, m.x, '(', m.y, ')')

################################################################################
#                                Load Materials                                #
################################################################################
include("material_lib/MgO_LiNbO3.jl")
include("material_lib/SiO2.jl")
include("material_lib/Si3N4.jl")
include("material_lib/αAl2O3.jl")
# include("material_lib/LiB3O5.jl")
# include("material_lib/silicon.jl")
# include("material_lib/GaAs.jl")
# include("material_lib/MgF2.jl")
# include("material_lib/HfO2.jl")
