using Symbolics: get_variables, make_array, SerialForm, Func, toexpr, _build_and_inject_function, @__MODULE__, MultithreadedForm, tosymbol, Sym
using SymbolicUtils.Code: MakeArray
# using Rotations

export AbstractMaterial, Material, RotatedMaterial, get_model, generate_fn, Δₘ_factors, Δₘ
export rotate, mult, unique_axes, plot_data, nn̂g, nĝvd, nn̂g_model, nn̂g_fn, nĝvd_model, nĝvd_fn, ε_fn
export n²_sym_fmt1, n_sym_cauchy, has_model, χ⁽²⁾_fn, material_name, plot_model!

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

@non_differentiable generate_array_fn(arg,A)

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
	color::Color
end
# constructor adding random color when color is not specified
Material(models::Dict,defaults::Dict,name::Symbol) = Material(models,defaults,name,RGBA(rand(3)...,1.0))


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

function generate_fn(mat::AbstractMaterial,model_name::Symbol,args...; expr_module=@__MODULE__(), parallel=SerialForm())
	model = get_model(mat,model_name,args...)
	if typeof(model)<:AbstractArray
		fn = generate_array_fn([Num(Sym{Real}(arg)) for arg in args],model; expr_module, parallel)
	else
		fn = build_function(model,args...;expression=Val{false})
	end
	return fn
end

function has_model(mat::AbstractMaterial,model_name::Symbol)
	return haskey(mat.models,model_name)
end

@non_differentiable generate_fn(mat::AbstractMaterial,model_name::Symbol,args...)

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
	color::Color
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

function rotate(mat::TM,𝓡::TR;name=nothing,color=mat.color) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	if eltype(𝓡)<:Num
		vars = get_variables(𝓡)
		defs = Dict{Symbol,Real}([ tosymbol(var) => 0.0 for var in vars])
	else
		defs = Dict{Symbol,Real}([])
	end
	if isnothing(name)
		name = Symbol(String(mat.name)*"_Rotated")
	end
	RotatedMaterial{TM,TR}(mat,𝓡,defs,name,color)
end

function rotate(mat::TM,𝓡::TR,defs::Dict;name=nothing,color=mat.color) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	if isnothing(name)
		name = Symbol(String(mat.name)*"_Rotated")
	end
	RotatedMaterial{TM,TR}(mat,𝓡,defs,name,color)
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

has_model(mat::RotatedMaterial,model_name::Symbol) = haskey(mat.parent.models,model_name)

# material_name(mat::RotatedMaterial) = material_name(mat.parent)
# Base.nameof(mat::RotatedMaterial) = getfield(mat.parent, :name)
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
nn̂g_fn(mat::AbstractMaterial) =  generate_array_fn([Num(Sym{Real}(:λ)) ,],nn̂g_model(mat))
nĝvd_fn(mat::AbstractMaterial) =  generate_array_fn([Num(Sym{Real}(:λ)) ,],nĝvd_model(mat))

function χ⁽²⁾_fn(mat::AbstractMaterial)
	if has_model(mat,:χ⁽²⁾)
		return generate_array_fn([Num(Sym{Real}(:λs₁)), Num(Sym{Real}(:λs₂)), Num(Sym{Real}(:λs₃))],get_model(mat,:χ⁽²⁾,:λs₁,:λs₂,:λs₃))
	else
		return (lm1,lm2,lm3) -> zero(SArray{Tuple{3,3,3}})
	end
end


nn̂g(mat::AbstractMaterial,lm::Real) = SMatrix{3,3}(nn̂g_fn(mat)(lm))
nĝvd(mat::AbstractMaterial,lm::Real) = SMatrix{3,3}(nĝvd_fn(mat)(lm))

"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

function unique_axes(mat::AbstractMaterial;model=:ε)
	e11,e22,e33 = diag(get_model(mat,model,:λ))
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

function plot_data(mats_in::AbstractVector{<:AbstractMaterial};model=:n)
	if isequal(model,:n)
		mats = filter(x->has_model(x,:ε),mats_in)
		fes = generate_fn.(mats,(:ε,),(:λ,))
		axind_axstr_unq = unique_axes.(mats)
		axind_unq = getindex.(axind_axstr_unq,1)
		axstr_unq = getindex.(axind_axstr_unq,2)
		fns = vcat(map((ff,as)->[(x->sqrt(ff(x)[a,a])) for a in as ], fes, axind_unq)...)
		mat_names = String.(nameof.(mats))
		names = "n" .* vcat([.*(axstr_unq[i], " (", mat_names[i],")") for i=1:length(mats)]...) # "n, n_i or n_i,j (Material)" for all unique axes and materials
	else
		mats = filter(x->has_model(x,model),mats_in)
		fgs = generate_fn.(mats,(model,),(:λ,))
		axind_axstr_unq = unique_axes.(mats)
		axind_unq = getindex.(axind_axstr_unq,1)
		axstr_unq = getindex.(axind_axstr_unq,2)
		fns = vcat(map((ff,as)->[(x->sqrt(ff(x)[a,a])) for a in as ], fgs, axind_unq)...)
		mat_names = String.(nameof.(mats))
		names = String(model) .* vcat([.*(axstr_unq[i], " (", mat_names[i],")") for i=1:length(mats)]...)
	end
	colors = vcat( [ [ mat.color for i=1:ll ] for (mat,ll) in zip(mats,length.(getindex.(axind_axstr_unq,(1,)))) ]...)
	all_linestyles	=	[nothing,:dash,:dot,:dashdot,:dashdotdot]
	linestyles  =	vcat( [ getindex.((all_linestyles,),1:ll) for ll in length.(getindex.(axind_axstr_unq,(1,))) ]... )
	return fns, names, colors, linestyles
end
plot_data(mat::AbstractMaterial ; model=:n) = plot_data([mat,]; model)
plot_data(mats::NTuple{N,<:AbstractMaterial} where N ; model=:n) = plot_data([mats...]; model)



function uplot(x::Union{AbstractMaterial, AbstractVector{<:AbstractMaterial}, NTuple{N,<:AbstractMaterial} };
		xlim=[0.5,1.8], xlabel="λ [μm]", ylabel="n", kwargs...)  where N
	fns, name, colors, styles = plot_data(x)
	UnicodePlots.lineplot(fns, xlim[1], xlim[2];
	 	xlim,
		ylim=map((a,b)->a(b,digits=1),(floor,ceil),ylims(fns;xlims=xlim)),
		name,
		xlabel,
		ylabel,
		width=75,
		height=35,
		kwargs...
		)
end

function uplot!(plt::UnicodePlots.Plot,x::Union{Material, AbstractVector{<:Material}, NTuple{N,<:Material} };
		xlim=[0.5,1.8], xlabel="λ [μm]", ylabel="n")  where N
	fns, name, colors, styles = plot_data(x)
	UnicodePlots.lineplot!(plt, fns; name ) #, xlim[1], xlim[2];
	 	# xlim,
		# ylim=round.( ylims(plt,ylims(fns;xlims=xlim)) ,digits=1),
		# name,
		# xlabel,
		# ylabel,
		# )
end

function plot_model!(ax, mats::AbstractVector{<:AbstractMaterial};model=:n,xrange=nothing)
	if isnothing(xrange)
		xmin = ax.limits[].origin[1]
		xmax = xmin + ax.limits[].widths[1]
	end
	lns = [lines!(ax, xmin..xmax, fn; label=lbl, color=clr, linestyle=ls) for (fn,lbl,clr,ls) in zip(plot_data(mats; model)...)]
end
plot_model(ax, mat::AbstractMaterial ; model=:n, xrange=nothing) = plot_model([mat,]; model, xrange)
plot_model(ax, mats::NTuple{N,<:AbstractMaterial} where N ; model=:n, xrange=nothing) = plot_model([mats...]; model, xrange)

import Base: show
Base.show(io::IO, ::MIME"text/plain", mat::AbstractMaterial) = uplot(mat) #print(io, "Examplary instance of Material\n", m.x, " ± ", m.y)
Base.show(io::IO, mat::AbstractMaterial) = uplot(mat) #print(io, m.x, '(', m.y, ')')
Base.show(io, ::MIME"text/plain", mat::AbstractMaterial) = uplot(mat)
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
