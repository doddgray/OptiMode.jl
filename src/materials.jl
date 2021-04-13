using Symbolics: get_variables, make_array, SerialForm, Func, toexpr, _build_and_inject_function, @__MODULE__, MultithreadedForm, tosymbol, Sym
using SymbolicUtils.Code: MakeArray
# using Rotations

export AbstractMaterial, Material, RotatedMaterial, get_model, generate_fn, Δₘ_factors, Δₘ
export rotate, mult, unique_axes, plot_data, nn̂g, nĝvd, nn̂g_model, nn̂g_fn, nĝvd_model, nĝvd_fn, ε_fn
export n²_sym_fmt1, n_sym_cauchy

# RuntimeGeneratedFunctions.init(@__MODULE__)

import Symbolics: substitute, simplify
Symbolics.substitute(A::AbstractArray{Num},d::Dict) = Symbolics.substitute.(A,(d,))
Symbolics.simplify(A::AbstractArray{Num}) = Symbolics.simplify.(A)

# add Symbolics.get_variables for arrays of `Num`s
import Symbolics.get_variables
function Symbolics.get_variables(A::AbstractArray{Num})
	unique(vcat(get_variables.(A)...))
end

# # add minimal Unitful+Symbolics interoperability
# import Base:*
# *(x::Unitful.AbstractQuantity,y::Num) =  Quantity(x.val*y, unit(x))
# *(y::Num,x::Unitful.AbstractQuantity) = x*y

# adjoint/rrule for SymbolicUtils.Code.create_array
# https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/278/files
function ChainRulesCore.rrule(::typeof(SymbolicUtils.Code.create_array), A::Type{<:AbstractArray}, T, u::Val{j}, d::Val{dims}, elems...) where {dims, j}
  y = SymbolicUtils.Code.create_array(A, T, u, d, elems...)
  function create_array_pullback(Δ)
    dx = Δ
    (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dx..., ntuple(_ -> DoesNotExist(), length(elems) - prod(dims) + j)...)
  end
  y, create_array_pullback
end

function generate_array_fn(args::Vector{Num},A::AbstractMatrix; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = _build_and_inject_function(expr_module,toexpr(Func(args,[],make_array(parallel,args,A,Matrix))))
end

function generate_array_fn(arg::Num,A::AbstractMatrix; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = generate_array_fn([arg,], A; expr_module, parallel)
end

function generate_array_fn(args::Vector{Num},A::SArray; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = _build_and_inject_function(expr_module,toexpr(Func(args,[],make_array(parallel,args,A,SArray))))
end

function generate_array_fn(arg::Num,A::SArray; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = generate_array_fn([arg,], A; expr_module, parallel)
end


function generate_array_fn(args::Vector{Num},A::TA; expr_module=@__MODULE__(), parallel=SerialForm()) where TA<:AbstractArray
	return fn = _build_and_inject_function(expr_module,toexpr(Func(args,[],make_array(parallel,args,A,TA))))
end

function generate_array_fn(arg::Num,A::TA; expr_module=@__MODULE__(), parallel=SerialForm()) where TA<:AbstractArray
	return fn = generate_array_fn([arg,], A; expr_module, parallel)
end

"""
################################################################################
#																			   #
#							   	   Materials							   	   #
#																			   #
################################################################################
"""

abstract type AbstractMaterial end

struct Material <: AbstractMaterial
	models::Dict
	defaults::Dict
	name::Symbol
end

import Base: nameof
Base.nameof(mat::AbstractMaterial) = getfield(mat, :name)

material_name(x::Real) = Symbol("Const_Material_$x")
material_name(x::AbstractVector) = Symbol("Const_Material_$(x[1])_$(x[2])_$(x[3])")
material_name(x::AbstractMatrix) = Symbol("Const_Material_$(x[1,1])_$(x[2,2])_$(x[3,3])")

Material(x) = Material(Dict([ε_tensor(x),]),Dict([]),material_name(x))
Material(mat::AbstractMaterial) = mat



function get_model(mat::AbstractMaterial,model_name::Symbol,args...)
	model = mat.models[model_name]
	missing_var_defaults = filter(x->!in(first(x),tosymbol.(args)),mat.defaults)
	subs =  Dict([(Sym{Real}(k),v) for (k,v) in missing_var_defaults])
	if typeof(model)<:AbstractArray
		model_subs = substitute.(model, (subs,))
	else
		model_subs = substitute(model, subs)
	end
	return model_subs
end

function generate_fn(mat::AbstractMaterial,model_name::Symbol,args...)
	model = get_model(mat,model_name,args...)
	if typeof(model)<:AbstractArray
		fn = generate_array_fn([Num(Sym{Real}(arg)) for arg in args],model)
	else
		fn = build_function(model,args...;expression=Val{false})
	end
	return fn
end

"""
################################################################################
#																			   #
#							   	  Rotations							   		   #
#																			   #
################################################################################
"""

struct RotatedMaterial{TM,TR} <: AbstractMaterial
	parent::TM
	rotation::TR
	rotation_defaults::Dict
	name::Symbol
end

function mult(χ::AbstractArray{T,3},v₁::AbstractVector,v₂::AbstractVector) where T<:Real
	@tullio v₃[i] := χ[i,j,k] * v₁[j] * v₂[k]
end

function mult(χ::AbstractArray{T,4},v₁::AbstractVector,v₂::AbstractVector,v₃::AbstractVector) where T<:Real
	@tullio v₄[i] := χ[i,j,k,l] * v₁[j] * v₂[k] * v₃[l]
end

function rotate(χ::AbstractMatrix,𝓡::AbstractMatrix)
	@tullio χᵣ[i,j] := 𝓡[a,i] * 𝓡[b,j] * χ[a,b]  fastmath=true
end

function rotate(χ::AbstractArray{T,3},𝓡::AbstractMatrix) where {T<:Real}
	@tullio χᵣ[i,j,k] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * χ[a,b,c]  fastmath=true
end

function rotate(χ::AbstractArray{T,4},𝓡::TR) where {T<:Real, TR<:StaticMatrix{3,3}}
	@tullio χᵣ[i,j,k,l] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * 𝓡[d,l] * χ[a,b,c,d]  fastmath=true
end

rotate(χ::Real,𝓡::StaticMatrix{3,3}) = χ

function rotate(mat::TM,𝓡::TR;name=nothing) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	if eltype(𝓡)<:Num
		vars = get_variables(𝓡)
		defs = Dict{Symbol,Real}([ tosymbol(var) => 0.0 for var in vars])
	else
		defs = Dict{Symbol,Real}([])
	end
	if isnothing(name)
		name = Symbol(String(mat.name)*"_Rotated")
	end
	RotatedMaterial{TM,TR}(mat,𝓡,defs,name)
end

function rotate(mat::TM,𝓡::TR,defs::Dict;name=nothing) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	if isnothing(name)
		name = Symbol(String(mat.name)*"_Rotated")
	end
	RotatedMaterial{TM,TR}(mat,𝓡,defs,name)
end

function get_model(mat::RotatedMaterial,model_name::Symbol,args...)
	model = rotate(mat.parent.models[model_name],mat.rotation)
	defs = merge(mat.parent.defaults,mat.rotation_defaults)
	missing_var_defaults = filter(x->!in(first(x),tosymbol.(args)),defs)
	subs =  Dict([(Sym{Real}(k),v) for (k,v) in missing_var_defaults])
	if typeof(model)<:AbstractArray
		model_subs = substitute.(model, (subs,))
	else
		model_subs = substitute(model, subs)
	end
	return model_subs
end

"""
################################################################################
#																			   #
#					  Dispersion (group index, GVD) models					   #
#																			   #
################################################################################
"""

function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    λ² = λ^2
    A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
end

function n_sym_cauchy( λ ; A=1, B=0, C=0, B₂=0, kwargs...)
    A   +   B / λ^2    +   C / λ^4
end

# Miller's Delta scaling
function Δₘ_factors(λs,ε_sym)
	λ = Num(first(get_variables(sum(ε_sym))))
	diagε_m1 = Vector(diag(ε_sym)) .- 1
	mapreduce(lm->substitute.( diagε_m1, ([λ=>lm],)), .*, λs)
end

function Δₘ(λs::AbstractVector,ε_sym, λᵣs::AbstractVector, χᵣ::AbstractArray{T,3}) where T
	dm = Δₘ_factors(λs,ε_sym) ./ Δₘ_factors(λᵣs,ε_sym)
	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k] fastmath=true
end

# Symbolic Differentiation
function ng_model(n_model::Num, λ::Num)
	Dλ = Differential(λ)
	return n_model - ( λ * expand_derivatives(Dλ(n_model)) )
end

function gvd_model(n_model::Num, λ::Num)
	Dλ = Differential(λ)
	return λ^3 * expand_derivatives(Dλ(Dλ(n_model)))
end

ng_model(n_model::AbstractArray{Num}, λ::Num) = ng_model.(n_model,(λ,))
gvd_model(n_model::AbstractArray{Num}, λ::Num) = gvd_model.(n_model,(λ,))

function ng_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	n_model = sqrt.(get_model(mat,:ε,symbol))
	return ng_model(n_model,λ)
end

function gvd_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	n_model = sqrt.(get_model(mat,:ε,symbol))
	return gvd_model(n_model,λ)
end

function nn̂g_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	n_model = sqrt.(get_model(mat,:ε,symbol))
	return ng_model(n_model,λ) .* n_model
end

function nĝvd_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	n_model = sqrt.(get_model(mat,:ε,symbol))
	return gvd_model(n_model,λ) .* n_model
end

ε_fn(mat::AbstractMaterial) = generate_array_fn([Num(Sym{Real}(:λ)) ,],get_model(mat,:ε,:λ))

nn̂g_fn(mat::AbstractMaterial) = generate_array_fn([Num(Sym{Real}(:λ)) ,],nn̂g_model(mat))
nĝvd_fn(mat::AbstractMaterial) = generate_array_fn([Num(Sym{Real}(:λ)) ,],nĝvd_model(mat))

nn̂g(mat::AbstractMaterial,lm::Real) = nn̂g_fn(mat)(lm)
nĝvd(mat::AbstractMaterial,lm::Real) = nĝvd_fn(mat)(lm)

"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

function unique_axes(mat::AbstractMaterial)
	e11,e22,e33 = diag(get_model(mat,:ε,:λ))
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

function plot_data(mats::AbstractVector{<:AbstractMaterial})
	# fes = getfield.(mats,:fε)
	fes = generate_fn.(mats,(:ε,),(:λ,))
	axind_axstr_unq = unique_axes.(mats)
	axind_unq = getindex.(axind_axstr_unq,1)
	axstr_unq = getindex.(axind_axstr_unq,2)
	fns = vcat(map((ff,as)->[(x->sqrt(ff(x)[a,a])) for a in as ], fes, axind_unq)...)
	# mat_names = chop.(String.(nameof.(fes)),head=2,tail=0)	# remove "ε_" from function names
	mat_names = String.(nameof.(mats))
	names = "n" .* vcat([.*(axstr_unq[i], " (", mat_names[i],")") for i=1:length(mats)]...) # "n, n_i or n_i,j (Material)" for all unique axes and materials
	return fns, names
end
plot_data(mat::AbstractMaterial) = plot_data([mat,])
plot_data(mats::NTuple{N,<:AbstractMaterial} where N) = plot_data([mats...])

function uplot(x::Union{AbstractMaterial, AbstractVector{<:AbstractMaterial}, NTuple{N,<:AbstractMaterial} };
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
include("material_lib/LiB3O5.jl")
# include("material_lib/silicon.jl")
# include("material_lib/GaAs.jl")
# include("material_lib/MgF2.jl")
# include("material_lib/HfO2.jl")
