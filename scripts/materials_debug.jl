using Symbolics
using SymbolicUtils
using RuntimeGeneratedFunctions
# using ModelingToolkit
using Unitful, UnitfulEquivalences
using LinearAlgebra
using StaticArrays
using SparseArrays
using Tullio
using Symbolics: get_variables
using ChainRules
using Zygote
using ForwardDiff
using Symbolics: destructure_arg, DestructuredArgs, make_array, _make_array, SerialForm, Func, toexpr, _build_and_inject_function, @__MODULE__, MultithreadedForm, tosymbol, Sym
using SymbolicUtils.Code: MakeArray
using Rotations
import Symbolics: substitute, simplify
Symbolics.substitute(A::AbstractArray{Num},d::Dict) = Symbolics.substitute.(A,(d,))
Symbolics.simplify(A::AbstractArray{Num}) = Symbolics.simplify.(A)

# short term manual fix for bug in SymbolicUtils.Code.create_array(::Type{<:Matrix},...) methods
# see https://github.com/JuliaSymbolics/SymbolicUtils.jl/issues/270
# import SymbolicUtils.Code: create_array
# @inline function SymbolicUtils.Code.create_array(::Type{<:Matrix}, ::Nothing, ::Val{dims}, elems...) where dims
# 	rows = Tuple(repeat([dims[2],],dims[1]))
# 	Base.hvcat(rows, elems...)
# end
# @inline function SymbolicUtils.Code.create_array(::Type{<:Matrix}, T, ::Val{dims}, elems...) where dims
# 	rows = Tuple(repeat([dims[2],],dims[1]))
#     Base.typed_hvcat(T, rows, elems...)
# end

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

# add Symbolics.get_variables for arrays of `Num`s
import Symbolics.get_variables
function Symbolics.get_variables(A::AbstractArray{Num})
	unique(vcat(get_variables.(A)...))
end


RuntimeGeneratedFunctions.init(@__MODULE__)

# add minimal Unitful+Symbolics interoperability
import Base:*
*(x::Unitful.AbstractQuantity,y::Num) =  Quantity(x.val*y, unit(x))
*(y::Num,x::Unitful.AbstractQuantity) = x*y

# rrules
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (NO_FIELDS, dv) )

function ChainRulesCore.rrule(::typeof(SparseArrays.SparseMatrixCSC),
    m::Integer, n::Integer, pp::Vector, ii::Vector, Av::Vector)
    A = SparseMatrixCSC(m,n,pp,ii,Av)
    function SparseMatrixCSC_pullback(dA::AbstractMatrix)
        dAv = Vector{eltype(dA)}(undef, length(Av))
        for j = 1:n, p = pp[j]:pp[j+1]-1
            dAv[p] = dA[ii[p],j]
        end
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dAv)
    end
    function SparseMatrixCSC_pullback(dA::SparseMatrixCSC)
        @assert getproperty.(Ref(A), (:m,:n,:colptr,:rowval)) == getproperty.(Ref(dA), (:m,:n,:colptr,:rowval))
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dA.nzval)
    end

    return A, SparseMatrixCSC_pullback
end

function mult(χ::AbstractArray{T,3},v₁::AbstractVector,v₂::AbstractVector) where T<:Real
	@tullio v₃[i] := χ[i,j,k] * v₁[j] * v₂[k]
end

function mult(χ::AbstractArray{T,4},v₁::AbstractVector,v₂::AbstractVector,v₃::AbstractVector) where T<:Real
	@tullio v₄[i] := χ[i,j,k,l] * v₁[j] * v₂[k] * v₃[l]
end

function Δₘ_factors(λs,ε_sym)
	λ = Num(first(get_variables(sum(ε_sym))))
	diagε_m1 = Vector(diag(ε_sym)) .- 1
	mapreduce(lm->substitute.( diagε_m1, ([λ=>lm],)), .*, λs)
end

function Δₘ(λs::AbstractVector,ε_sym, λᵣs::AbstractVector, χᵣ::AbstractArray{T,3}) where T
	dm = Δₘ_factors(λs,ε_sym) ./ Δₘ_factors(λᵣs,ε_sym)
	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k] fastmath=true
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
end

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

function rotate(mat::TM,𝓡::TR) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	if eltype(𝓡)<:Num
		vars = get_variables(𝓡)
		defs = Dict{Symbol,Real}([ tosymbol(var) => 0.0 for var in vars])
	else
		defs = Dict{Symbol,Real}([])
	end
	RotatedMaterial{TM,TR}(mat,𝓡,defs)
end

function rotate(mat::TM,𝓡::TR,defs::Dict) where {TM<:AbstractMaterial,TR<:AbstractMatrix}
	RotatedMaterial{TM,TR}(mat,𝓡,defs)
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

##

function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

pₑ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)

pᵪ₂ = (
	d₃₃ =   20.3,    #   pm/V
	d₃₁ =   -4.1,    #   pm/V
	d₂₂ =   2.1,     #   pm/V
	λs  =  [1.313, 1.313, 1.313/2.0]
)

function make_LN(;pₒ=pₒ,pₑ=pₑ,pᵪ₂=pᵪ₂)
	@variables λ, T, λs[1:3]
	nₒ² = n²_MgO_LiNbO₃_sym(λ, T; pₒ...)
	nₑ² = n²_MgO_LiNbO₃_sym(λ, T; pₑ...)
	ε 	= diagm([nₒ², nₒ², nₑ²])
	d₃₃, d₃₁, d₂₂, λᵣs = pᵪ₂
	χ⁽²⁾ᵣ = cat(
		[ 	0.0	 	-d₂₂ 	d₃₁			#	xxx, xxy and xxz
		 	-d₂₂	0.0 	0.0			#	xyx, xyy and xyz
			d₃₁	 	0.0		0.0		],	#	xzx, xzy and xzz
		[ 	-d₂₂	0.0 	0.0			#	yxx, yxy and yxz
			0.0	 	d₂₂ 	d₃₁			#	yyx, yyy and yyz
			0.0	 	d₃₁		0.0		],	#	yzx, yzy and yzz
		[ 	d₃₁	 	0.0 	0.0			#	zxx, zxy and zxz
			0.0	 	d₃₁ 	0.0			#	zyx, zyy and zyz
			0.0	 	0.0 	d₃₃		],	#	zzx, zzy and zzz
		 dims = 3
	)
	nₒ = sqrt(nₒ²)
	ngₒ = ng_model(nₒ,λ)
	gvdₒ = gvd_model(nₒ,λ)
	nₑ = sqrt(nₑ²)
	ngₑ = ng_model(nₑ,λ)
	gvdₑ = gvd_model(nₑ,λ)
	models = Dict([
		:nₒ		=>	nₒ,
		:ngₒ	=>	ngₒ,
		:gvdₒ	=>	gvdₒ,
		:nₑ		=>	nₑ,
		:ngₑ	=>	ngₑ,
		:gvdₑ	=>	gvdₑ,
		:ng		=>	diagm([ngₒ, ngₒ, ngₑ]),
		:gvd	=>	diagm([gvdₒ, gvdₒ, gvdₑ]),
		:ε 		=> 	ε,
		:χ⁽²⁾	=>	SArray{Tuple{3,3,3}}(Δₘ(λs,ε, λᵣs, χ⁽²⁾ᵣ)),
	])
	defaults =	Dict([
		:λ		=>		0.8,	# μm
		:T		=>		24.5,	# °C
		:λs₁	=>		1.064,	# μm
		:λs₂	=>		1.064,	# μm
		:λs₃	=>		0.532,	# μm

	])
	Material(models, defaults)
end

LN = make_LN()

LN.models


LN.models[:ε]
get_model(LN,:ε, :λ)
feps1 = generate_fn(LN,:ε, :λ)
feps1(0.8)
feps1(0.7)
Zygote.gradient(x->sum(feps1(x)), 0.7)
ForwardDiff.derivative(x->sum(feps1(x)), 0.7)
Zygote.hessian(x->sum(feps1(x)), 0.7)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->sum(feps1(x)),y), 0.7)
ForwardDiff.derivative(feps1,0.7)
ForwardDiff.derivative(x->ForwardDiff.derivative(feps1,x),0.7)

get_model(LN,:χ⁽²⁾, :λs₁, :λs₂, :λs₃, :T)
get_model(LN,:χ⁽²⁾)
get_model(LN,:χ⁽²⁾, :λs₁, :λs₂, :λs₃)
fchi21 = generate_fn(LN, :χ⁽²⁾, :λs₁, :λs₂, :λs₃)
Zygote.gradient(x->sum(fchi21(2x,2x,x)), 0.7)
ForwardDiff.derivative(x->sum(fchi21(2x,2x,x)), 0.7)
Zygote.hessian(x->sum(fchi21(2x,2x,x)), 0.7)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->sum(fchi21(2x,2x,x)),y), 0.7)

LN.models[:nₒ]
fno = generate_fn(LN,:nₒ, :λ)
fno(0.8)
fno''(0.8)
Zygote.hessian(fno,0.8)

fno'.([0.7,0.8,0.9])
##
𝓡 = MRP(RotY(π/2)) |> Matrix
LNr = rotate(LN,𝓡)
𝓡2 = MRP(RotY(π/4)) |> Matrix
LNr2 = rotate(LN,𝓡2)
epsmr1 = get_model(LNr,:ε)
epsmr2 = get_model(LNr2,:ε)

epsmr1 = get_model(LNr,:ε,:λ)
epsmr2 = get_model(LNr2,:ε,:λ)
epsmr2 = get_model(LNr2,:ε,:λ,:T)

@variables θ
𝓡3 = RotY(θ) |> Matrix
LNr3 = rotate(LN,𝓡3)
LNr3.rotation

feps1 = generate_fn(LNr,:ε, :λ)
feps1(0.8)

chi2r1 = get_model(LNr,:χ⁽²⁾, :λs₁, :λs₂, :λs₃)
get_model(LNr,:χ⁽²⁾)[1,1,1]
fchi2r1 = generate_fn(LNr,:χ⁽²⁾, :λs₁, :λs₂, :λs₃)

fchi2r1(1.6,1.6,0.8)[1,1,1]
fchi2r1(1.064,1.064,0.532)[1,1,1]

Zygote.gradient(x->fchi2r1(x,x,x/2)[1,1,1],1.064)
ForwardDiff.derivative(x->fchi2r1(x,x,x/2)[1,1,1],1.064)
Zygote.hessian(x->fchi2r1(x,x,x/2)[1,1,1],1.064)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->fchi2r1(x,x,x/2)[1,1,1],y),1.064)

feps3 = generate_fn(LNr3,:ε, :λ, :θ)

feps3(0.8,0.0)




feps3(0.8,π/2.0)






feps3

##



ng_fn(mat::AbstractMaterial; symbol=:λ)


ng_model(LNr)
@variables λ
Dλ = Differential(λ)
Dλ |> typeof
# ng(n_sym::Num) = n_sym - λ * expand_derivatives(Dλ(n_sym))
# gvd(n_sym::Num) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym)))

nom = get_model(LN,:nₒ,:λ)
nem = get_model(LN,:nₑ,:λ)
typeof(nom)
Dλ(nom)|>expand_derivatives

ns = [nom, nom, nem]
Dλ.(ns)|>expand_derivatives
##

𝓡 = MRP(RotY(π/2))
𝓡
typeof(𝓡)<:AbstractMatrix
LN.models[:ε]


@tullio χᵣ[i,j] := 𝓡[a,i] * 𝓡[b,j] * (LN.models[:ε])[a,b]  fastmath=true

rotate((LN.models[:ε]),Matrix{3,3}(𝓡))
rotate((get_model(LN,:χ⁽²⁾)),Matrix{3,3}(𝓡))


##
ε_syms = tosymbol.(ε_vars)
ε_syms2 = tosymbol.(get_variables(LN.models[:ε]))
ε_syms[1] |> tosymbol

getindex.((LN.defaults,),ε_syms)

@variables x,y
z = 2x - y^2

substitute(z,Dict([Sym{Real}(:y)=>0.3]))

Num(:x)


##
function make_LN(;pₒ=pₒ,pₑ=pₑ,pᵪ₂=pᵪ₂)
	@variables λ, T
	nₒ² = n²_MgO_LiNbO₃_sym(λ, T; pₒ...)
	nₑ² = n²_MgO_LiNbO₃_sym(λ, T; pₑ...)
	ε 	= diagm([nₒ², nₒ², nₑ²])
	Material(
		ε,							# dielectric tensor
		[λ, T],						# variables
		[pₒ,pₑ,pᵪ₂],				# parameters
		[(0.0,8.0),(-20.0,300.0)],	# domains
		Dict([λ=>0.8,T=>pₒ.T₀]),	# Defaults
	)
end

LN = make_LN()
feps_LN = generate_ε(LN)
feps_LN(0.8)
feps_LN.([0.8,0.85,0.9])
ForwardDiff.derivative(feps_LN,0.7)
ForwardDiff.derivative(x->ForwardDiff.derivative(feps_LN,x),0.7)
Zygote.gradient(x->sum(feps_LN(x)),0.7)
ForwardDiff.derivative(x->sum(feps_LN(x)),0.7)

fchi2_LN = generate_χ⁽²⁾(LN)

fchi2_LN(1.0,1.0,0.5)
fchi2_LN(1.0,1.0,0.5)[3,3,3]

Zygote.gradient(x->sum(fchi2_LN(x,x,x/2)),1.0)
Zygote.hessian(x->sum(fchi2_LN(x,x,x/2)),1.0)
ForwardDiff.derivative(x->sum(fchi2_LN(x,x,x/2)),1.0)
ForwardDiff.derivative(x->fchi2_LN(x,x,x/2),1.0)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->sum(fchi2_LN(x,x,x/2)),y),1.0)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->fchi2_LN(x,x,x/2),y),1.0)
##

@variables λ, T, εᵣ[1:3,1:3], χ⁽²⁾

name 			=	:MgO_LiNbO₃
props 			=	[	:εᵣ, 		:χ⁽²⁾ 	]
prop_units 		= 	[	NoUnits, 	pm/V	]
params 			= 	[	:λ, 		:T    	]
param_units 	= 	[	μm, 		°C		]
param_defaults 	= 	[	0.8μm,		20°C 	]



prop_models		=	Dict([
						:εᵣ 	=> 	[ 	2.0*λ	0.0		0.0
						 				0.0		2.1*λ	0.0
										0.0		0.0		2.2*λ	],

						:χ⁽²⁾ => 	cat( [ 	4.0*λ	0.0		0.0
								 			0.0		4.1*λ	0.0
											0.0		0.0		2.2*λ	],
										[ 	4.0*λ+T	0.0		0.0
								 			0.0		4.1*λ	0.0
											0.0		0.0		2.2*λ	],
										dims=3,
										),
])

using Symbolics: get_variables, tosymbol

get_variables.(prop_models)

Dict( zip(keys(prop_models), tosymbol.(vars) for vars in (get_variables.(values(prop_models))) ) )



import Symbolics.get_variables
function Symbolics.get_variables(A::AbstractArray{Num})
	unique(vcat(get_variables.(A)...))
end


get_variables.(values(prop_models))

rhss(value.(εᵣ))

using Symbolics: rhss, lhss, value

# LN = Material(
# 	[	εᵣ	~	[	]
#
# 	]
#
# )


##
## try out Symbolics+Unitful (+FieldMetadata?)
@variables λ

λ

lm = 1.0u"μm"

uconvert(u"cm", lm)

ustrip(u"nm",lm)

lm - λ*1.0u"μm"

λ.val

using SymbolicUtils: symtype, @syms, getmetadata
symtype(λ.val)

typeof(λ.val)
tosymbol(λ)
getmetadata(λ.val)

u"cm" |> typeof <: Unitful.Units

lm
u"cm"(lm)
u"Hz"(lm,Spectral())

uconvert(u"Hz",lm,Spectral())

using Unitful.DefaultSymbols

using FieldMetadata
@metadata describe ""

@describe mutable struct Described
   a::Int     | "an Int with a description"
   b::Float64 | "a Float with a description"
end

d = Described(1, 1.0)

describe(d,:a)

using Parameters
@metadata describe "" String
@metadata bounds (0, 1) Tuple

@bounds @describe @with_kw struct WithKeyword{T}
    a::T = 3 | (0, 100) | "a field with a range, description and default"
    b::T = 5 | (2, 9)   | "another field with a range, description and default"
end

k = WithKeyword()

describe(k, :b)

bounds(k,:b)

##

d₃₃, d₃₁, d₂₂, λᵣs = LN.parameters[3]
chi2 = cat(
	[ 	0.0	 	-d₂₂ 	d₃₁			#	xxx, xxy and xxz
		-d₂₂	0.0 	0.0			#	xyx, xyy and xyz
		d₃₁	 	0.0		0.0		],	#	xzx, xzy and xzz
	[ 	-d₂₂	0.0 	0.0			#	yxx, yxy and yxz
		0.0	 	d₂₂ 	d₃₁			#	yyx, yyy and yyz
		0.0	 	d₃₁		0.0		],	#	yzx, yzy and yzz
	[ 	d₃₁	 	0.0 	0.0			#	zxx, zxy and zxz
		0.0	 	d₃₁ 	0.0			#	zyx, zyy and zyz
		0.0	 	0.0 	d₃₃		],	#	zzx, zzy and zzz
	 dims = 3
)

SArray{Tuple{3,3,3}}(chi2)


function gen_A()
	@variables a
	A	= 	[	a  			0.2
		 		-1.0*a   	2.0*a	]
	fA1 = build_function(A1,λ₁)[1]|>eval
end

##

@variables a
eps_LN = calculate_ε(a,LN)
f1 = generate_array_fn([a,],eps_LN)
f1(0.9)



f2 = _build_and_inject_function(Main,toexpr(Func([a,],[],make_array(SerialForm(),[a,],eps_LN,Matrix))))
f2(0.9)


f3 = _build_and_inject_function(Main,toexpr(Func([a,],[],MakeArray(eps_LN::Matrix,Matrix))))
f3(0.9)

using SymbolicUtils.Code: create_array

f4 = _build_and_inject_function(Main,toexpr(Func([a,],[],create_array(Matrix,nothing, Val{(3,3,3)}(),A...))))
create_array(Matrix, nothing, Val{(3, 3, 3)}(), A...)
f4(0.9)


sum()
Zygote.gradient(x->sum(f4(x)),0.9)


make_array(SerialForm(),[a,],eps_LN,Matrix)




using SymbolicUtils.Code: MakeArray

toexpr(Func([a,],[],make_array(SerialForm(),[a,],eps_LN,Matrix)))




make_array(SerialForm(),[a,],eps_LN,Matrix)))

MakeArray(eps_LN,Matrix)

## MBE
using Symbolics: @variables
using SymbolicUtils.Code: MakeArray, toexpr, create_array
@variables a
A	= 	[	a  		0		0
			0   	2*a		0
			0		0		3*a		]


A_expr = toexpr(MakeArray(A,Matrix))
eval(A_expr)

# ERROR: ArgumentError: argument count 9 does not match specified shape (2, 3)
# Stacktrace:
# {Number})
#    @ Base ./abstractarray.jl:1938
#  [2] hvcat(::Tuple{Int64, Int64}, ::Symbolics.Num, ::Vararg{Number})
#    @ Base ./abstractarray.jl:1925
#  [3] create_array(::Type{Matrix{T} where T}, ::Nothing, ::Val{(3, 3)}, ::Symbolics.Num, ::Vararg{Any})
#    @ SymbolicUtils.Code ~/.julia/packages/SymbolicUtils/fikCA/src/code.jl:411
#  [4] top-level scope
#    @ ~/.julia/packages/SymbolicUtils/fikCA/src/code.jl:374
#  [5] eval
# ...

##
using StaticArrays
using ChainRulesCore
using Zygote
using Symbolics: Num, make_array, SerialForm, Func, toexpr, _build_and_inject_function, @variables
using RuntimeGeneratedFunctions: @__MODULE__, init
init(@__MODULE__)
# generic rrules for StaticArray construction, might also want these for SVector and SMatrix
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (NO_FIELDS, dv) )

# hacky methods bypassing Symbolics.build_function to explicitly specify `SArray` in `make_array`
function generate_array_fn(args::Vector{Num},A::SArray; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = _build_and_inject_function(expr_module,toexpr(Func(args,[],make_array(parallel,args,A,SArray))))
end

function generate_array_fn(arg::Num,A::SArray; expr_module=@__MODULE__(), parallel=SerialForm())
	return fn = generate_array_fn([arg,], A; expr_module, parallel)
end

function generate_A_fn()
	@variables a
	A	= 		cat(	[	a  		0		0
							0   	2*a		0
							0		0		3*a		],

						[	a^2  	0		0
							0   	sin(a)	0
							0		0		3.3		],
						dims=3,
			)
	return f_A = generate_array_fn(a,SArray{Tuple{3,3,2}}(A))
end
f_A = generate_A_fn()
f_A(0.3)


Zygote.gradient(x->sum(f_A(x)),0.3)

##
# M1 = [ 1.0a 2.0
# 	   3.0	4.0	]

fA1 = build_function(As,a)[1]|>eval
# fA2 = build_function(A,[M1,a])[1]|>eval
fA1(0.5)
Zygote.gradient(x->sum(fA1(x)),0.5)

args = [a,]
dargs = map(arg -> destructure_arg(arg, !checkbounds), [args...])
i = findfirst(x->x isa DestructuredArgs, dargs) # nothing

make_array(SerialForm(),[a,],A,Array)

make_array(SerialForm(),[a,],A,Matrix)

make_array(MultithreadedForm(),[a,],A,Matrix)

fA4 = _build_and_inject_function(Main,toexpr(Func([a,],[],make_array(SerialForm(),[a,],A,SMatrix)))) # == oop_expr2

fA5 = _build_and_inject_function(Main,toexpr(Func([a,],[],make_array(SerialForm(),[a,],A,SArray))))

fA5(0.7)
Zygote.refresh()
Zygote.gradient(x->sum(fA4(x)),0.7)
ForwardDiff.derivative(x->sum(fA5(x)),0.7)
ForwardDiff.derivative(fA4,0.7)

Zygote.gradient(x->sum(fA5(x)),0.7)
ForwardDiff.derivative(x->sum(fA5(x)),0.7)
ForwardDiff.derivative(fA5,0.7)

oop_expr1 = Func(dargs,[],make_array(SerialForm(),[a,],A,Array))

oop_expr2 = Func(dargs,[],make_array(SerialForm(),[a,],A,Matrix))

oop_expr3 = Func(dargs,[],make_array(SerialForm(),[a,],A,Matrix))

expr1 = oop_expr1 |> toexpr

expr2 = oop_expr2 |> toexpr

expr3 = oop_expr2 |> toexpr






MA1 = make_array(SerialForm(),[a,],A,Matrix)

function toexpr2(a::MakeArray, st)
    similarto = toexpr(a.similarto, st)
    T = similarto isa Type ? similarto : :(typeof($similarto))
    elT = a.output_eltype
    quote
        $create_array($T,
                     $elT,
                     Val{$(size(a.elems))}(),
                     $(map(x->toexpr(x, st), a.elems)...),)
    end
end

toexpr2(MA1,Matrix{Float64}())



expr2.head

expr2.args[1].head

expr_mod = @__MODULE__()

# using RuntimeGeneratedFunctions: _tagname,
# using RuntimeGeneratedFunctions


# expr1
(SymbolicUtils.Code.create_array)(Array, nothing, Val{(3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
(SymbolicUtils.Code.create_array)(Matrix{T} where T, nothing, Val{(3, 3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
#expr2 ^

create_array(Matrix, nothing, Val{(3, 3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
create_array(Matrix, nothing, Val{(3, 3, 3)}(), A...)


# expr1
(create_array2)(Array, nothing, Val{(3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
(create_array2)(Matrix, nothing, Val{(3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
#expr2 ^

els1 = (a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))
hvcat((3,3,3),els1...)

@inline function create_array2(::Type{<:Matrix}, ::Nothing, ::Val{dims}, elems...) where dims
	println("")
	println("dims:")
	@show dims
	Base.hvcat(dims, elems...)
end

(SymbolicUtils.Code.create_array)(Matrix{T} where T, nothing, Val{(3, 3)}(), a, (*)(-1.0, a), 0.0, 0.2, (*)(2.0, a), 1.5, (*)(0.5, a), (*)(-2.3, a), (^)(a, 2))

module_tag = getproperty(@__MODULE__(), _tagname)

fA2 = _build_and_inject_function(Main,expr1)
fA3 = _build_and_inject_function(Main,expr2)

fA2(0.8)
fA3(0.8)

Zygote.gradient(x->sum(fA2(x)),0.8)
ForwardDiff.derivative(x->sum(fA2(x)),0.8)

##
eps_sym1 = calculate_ε(λ₁,LN)


using StaticArrays

#substitute.(A1, λ₁=>0.21)

fA1 = build_function(A1,λ₁)[1]|>eval
Zygote.gradient(x->sum(fA1(x)),0.5)

fA2 = build_function(A1::Matrix{Num},λ₁)[1]|>eval
Zygote.gradient(x->sum(fA2(x)),0.5)

A1s = SMatrix{2,2}(A1)
fA3 = build_function(A1s,λ₁)[1]|>eval
Zygote.gradient(x->sum(fA3(x)),0.5)

(SymbolicUtils.Code.create_array)(Array, nothing, Val{(2, 2)}(), (*)(3.2, λ₁), (*)(-1.2, λ₁), 0.2, λ₁)

(SymbolicUtils.Code.create_array)(Matrix, nothing, Val{(2, 2)}(), (*)(3.2, λ₁), (*)(-1.2, λ₁), 0.2, λ₁)

foo2(x) = (SymbolicUtils.Code.create_array)(Matrix, nothing, Val{(2, 2)}(), (*)(3.2, x), (*)(-1.2, x), 0.2, x)
foo2(3.3)
Zygote.gradient(x->sum(foo2(x)),3.3)
foo3(x) = (SymbolicUtils.Code.create_array)(Array, nothing, Val{(2, 2)}(), (*)(3.2, x), (*)(-1.2, x), 0.2, x)
foo3(3.3)
Zygote.gradient(x->sum(foo3(x)),3.3)



using SymbolicUtils
using SymbolicUtils.Code
toexpr.(eps_sym1)

eps_code1 = build_function(eps_sym1,λ₁)[1]



##
@variables λ, T
c = Unitful.c0      # Unitful.jl speed of light
Dλ = Differential(λ)
DT = Differential(T)
ng(n_sym::Num) = n_sym - λ * expand_derivatives(Dλ(n_sym))
gvd(n_sym::Num) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym))) # gvd = uconvert( ( 1 / ( 2π * c^2) ) * _gvd(lm_um,T_C)u"μm", u"fs^2 / mm" )

##


s1 = λ + 3T

typeof(s1)

Symbolics.toexpr(s1)

nₒ_MgO_LiNbO₃_sym

simplify(nₒ_MgO_LiNbO₃_sym)

ngo = ng(nₒ_MgO_LiNbO₃_sym)

simplify(ngo)

gvdo = gvd(nₒ_MgO_LiNbO₃_sym)

epsLN = sparse(Diagonal([nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₑ²_MgO_LiNbO₃_sym]))
χ⁽²⁾_MgO_LiNbO₃_sym(λ)
χ⁽²⁾_MgO_LiNbO₃_sym(0.8)

feps_oop = build_function(epsLN,λ, expression=Val{false})[1]
feps_ip = build_function(epsLN,λ, expression=Val{false})[2]

using ChainRules, Zygote, ForwardDiff


feps_oop(0.8)[1,1]
Zygote.refresh()
Zygote.gradient(x->feps_oop(x)[1,1],0.8)[1]
Zygote.hessian(x->feps_oop(x)[1,1],0.8)[1]
ForwardDiff.derivative(feps_oop,0.8)
ForwardDiff.derivative(x->ForwardDiff.derivative(feps_oop,x),0.8)
ForwardDiff.derivative(y->ForwardDiff.derivative(x->ForwardDiff.derivative(feps_oop,x),y),0.8)
##

LN2 = make_LN()
LN3 = make_LN()

LN4 = make_LN()


LN_defs = [var=>LN.defaults[var] for var in LN.variables[2:end]]
substitute.(LN.ε,LN_defs)





f_χ⁽²⁾_LN = eval(generate_χ⁽²⁾(LN4))
expression = Val{true}
f_chiLN = generate_χ⁽²⁾(LN)
f_chiLN2 = generate_χ⁽²⁾(LN2)
f_chiLN4 = generate_χ⁽²⁾(LN3)
f_chiLN5 = generate_χ⁽²⁾(LN2)
f_chiLN(1.064,1.064,0.532)
f_chiLN(1.064,1.064,0.532)[3,3,3]
Zygote.gradient(x->f_χ⁽²⁾_LN(x,x,x/2)[3,3,3],1.1)

Zygote.gradient(x->f_chiLN2(x,x,x/2)[3,3,3],1.1)
Zygote.gradient(x->f_chiLN5(x,x,x/2)[3,3,3],1.1)

@variables λ₁, λ₂, λ₃
χ⁽²⁾_sym = calculate_χ⁽²⁾([λ₁, λ₂, λ₃],LN)


get_variables(sum(χ⁽²⁾_sym))[1]



get_variables(sum(χ⁽²⁾_sym))[1] |> typeof
get_variables(sum(χ⁽²⁾_sym))[1] |> Symbolics.arguments
Num(get_variables(sum(χ⁽²⁾_sym))[1]) === λ₁
λ₁ |> typeof
λ₁ |> Symbolics.istree
χ⁽²⁾_sym |> Symbolics.istree

χ⁽²⁾_sym = calculate_χ⁽²⁾([λ₁, λ₂, λ₃],LN)
χ⁽²⁾_sym = calculate_χ⁽²⁾([1.064,1.064,0.532],LN)[3,3,3]

f_χ⁽²⁾_LN(1.064,1.064,0.532)[3,3,3]

substitute.(χ⁽²⁾_sym,[λ₁=>1.064, λ₂=>1.064, λ₃=>1.064/2])

Dict(zip(get_variables(sum(χ⁽²⁾_sym)),[1.064,1.064,0.532]))

LN.parameters[3][:d₃₃]
d₃₃3, d₃₁3, d₂₂3, λsᵣ3 = LN.parameters[3]
d₃₃3
d₃₁3
Symbolics.get_variables(sum(LN.ε))
Symbolics.get_varnumber(sum(LN.ε))
Symbolics.tosymbol.(Symbolics.get_variables(sum(LN.ε)))

##
pₑ_MgO_LiNbO₃ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ_MgO_LiNbO₃ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)

@variables λ, T

nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ_MgO_LiNbO₃_λT_sym = sqrt(nₒ²_MgO_LiNbO₃_λT_sym)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
nₒ_MgO_LiNbO₃_sym = sqrt(nₒ²_MgO_LiNbO₃_sym)

nₑ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₑ_MgO_LiNbO₃...)
nₑ_MgO_LiNbO₃_λT_sym = sqrt(nₑ²_MgO_LiNbO₃_λT_sym)
nₑ²_MgO_LiNbO₃_sym = substitute(nₑ²_MgO_LiNbO₃_λT_sym,[T=>pₑ_MgO_LiNbO₃.T₀])
nₑ_MgO_LiNbO₃_sym = sqrt(nₑ²_MgO_LiNbO₃_sym)


# 	Values to use for green-pumped processes
#    d₃₃ = 27

d₃₃ =   20.3    #   pm/V
d₃₁ =   -4.1    #   pm/V
d₂₂ =   2.1     #   pm/V

χ⁽²⁾ᵣ_MgO_LiNbO₃ = cat(
	[ 	0.0	 	-d₂₂ 	d₃₁			#	xxx, xxy and xxz
	 	-d₂₂	0.0 	0.0			#	xyx, xyy and xyz
		d₃₁	 	0.0		0.0		],	#	xzx, xzy and xzz
	[ 	-d₂₂	0.0 	0.0			#	yxx, yxy and yxz
		0.0	 	d₂₂ 	d₃₁			#	yyx, yyy and yyz
		0.0	 	d₃₁		0.0		],	#	yzx, yzy and yzz
	[ 	d₃₁	 	0.0 	0.0			#	zxx, zxy and zxz
		0.0	 	d₃₁ 	0.0			#	zyx, zyy and zyz
		0.0	 	0.0 	d₃₃		],	#	zzx, zzy and zzz
	 dims = 3
)
λs_χ⁽²⁾ᵣ_MgO_LiNbO₃ = [1.313,1.313,1.313/2.0]

nₒ²ᵣs = [ substitute(nₒ²_MgO_LiNbO₃_sym,[λ=>lm]).val for lm in λs_χ⁽²⁾ᵣ_MgO_LiNbO₃ ]
nₑ²ᵣs = [ substitute(nₑ²_MgO_LiNbO₃_sym,[λ=>lm]).val for lm in λs_χ⁽²⁾ᵣ_MgO_LiNbO₃ ]
εᵣs = [sparse(Diagonal([nosq, nosq, nesq])) for (nosq, nesq) in zip(nₒ²ᵣs,nₑ²ᵣs)]

ε_LN = sparse(Diagonal([nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₑ²_MgO_LiNbO₃_sym]))
[eps->(eps-1) for eps in diag(ε_LN)]

@variables λ₁, λ₂
λ₃ = λ₁ + λ₂

[ substitute.( diag(ε_LN - 1I), ([λ=>lm],)) for lm in (λ₁,λ₂,λ₃) ]



reduce(.*,[ substitute.( diag(ε_LN - 1I), ([λ=>lm],)) for lm in (λ₁,λ₂,λ₃) ] )
diagε_m1 = Vector(diag(ε_LN)) .- 1
map(lm->substitute.( diagε_m1, ([λ=>lm],)), (λ₁,λ₂,λ₃) )
mapreduce(lm->substitute.( diagε_m1, ([λ=>lm],)), .*, (λ₁,λ₂,λ₃) )



Δₘ_factors([λ₁,λ₂,λ₃],ε_LN)
Δₘ_factors([1.313,1.313,1.313/2.0],ε_LN)

χᵣ = χ⁽²⁾ᵣ_MgO_LiNbO₃
dm = Δₘ_factors([λ₁,λ₂,λ₃],ε_LN) ./ Δₘ_factors([1.313,1.313,1.313/2.0],ε_LN)
@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k] fastmath=true

dm = Δₘ_factors([1.0,1.0,0.5],ε_LN) ./ Δₘ_factors([1.313,1.313,1.313/2.0],ε_LN)
@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k]
χ⁽²⁾ᵣ_MgO_LiNbO₃
χ⁽²⁾_MgO_LiNbO = Δₘ([λ₁,λ₂,λ₃],ε_LN,[1.313,1.313,1.313/2.0],χ⁽²⁾ᵣ_MgO_LiNbO₃)

Δₘ([1.0,1.0,0.5],ε_LN,[1.313,1.313,1.313/2.0],χ⁽²⁾ᵣ_MgO_LiNbO₃)
Δₘ_factors(,ε_LN)


substitute( diag(ε_LN - 1I), [λ=>λ₁])

Δₘ(λs, εs, λs_χ⁽²⁾ᵣ_MgO_LiNbO₃, εᵣs,χ⁽²⁾ᵣ_MgO_LiNbO₃)

χ⁽²⁾_MgO_LiNbO₃_sym(λ::Real) =  χ⁽²⁾_MgO_LiNbO₃_sym([λ,λ,λ/2])

using Tullio
function mult(χ::AbstractArray{T,3},v₁::AbstractVector,v₂::AbstractVector) where T<:Real
	@tullio v₃[i] := χ[i,j,k] * v₁[j] * v₂[k]
end

function mult(χ::AbstractArray{T,4},v₁::AbstractVector,v₂::AbstractVector,v₃::AbstractVector) where T<:Real
	@tullio v₄[i] := χ[i,j,k,l] * v₁[j] * v₂[k] * v₃[l]
end

function Δₘ(λs::AbstractVector, fε::Function, λᵣs::AbstractVector, χᵣ::AbstractArray{T,3}) where T
	dm = flat(map( (lm,lmr) -> (diag(fε(lm)).-1.) ./ (diag(fε(lmr)).-1.), λs, λᵣs ))
	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i,1] * dm[j,2] * dm[k,3] fastmath=true
end

function Δₘ(λs::AbstractVector,ε_sym, λᵣs::AbstractVector, χᵣ::AbstractArray{T,3}) where T
	# dm = Symbolics.value.(Δₘ_factors(λs,ε_sym) ./ Δₘ_factors(λᵣs,ε_sym))
	dm = Δₘ_factors(λs,ε_sym) ./ Δₘ_factors(λᵣs,ε_sym)
	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k] fastmath=true
end




##
struct Material{T}
	ε::T
	fε::Function
	fng::Function
	fgvd::Function
	# fχ⁽²⁾::Function
end

##

ngLN_sym = sparse(Diagonal(ng.([nₒ_MgO_LiNbO₃_sym,nₒ_MgO_LiNbO₃_sym,nₑ_MgO_LiNbO₃_sym])))

Num |> supertypes

ngLN_expr = build_function(ng(nₒ_MgO_LiNbO₃_sym), λ) #expresssion = false, parallel=Symbolics.MultithreadedForm())

ngLN = eval(build_function(ngLN_sym,
	λ;
	expresssion = Val{false}, )[1])
	# parallel=Symbolics.MultithreadedForm())[1])
ngLNfast = build_function(ngLN_sym,
	λ;
	expresssion = Val{false})[2] |> eval


ngLN


ngLN(0.9)

using ChainRules, Zygote, ForwardDiff

ForwardDiff.derivative(x->ngLN(x),0.9)
ForwardDiff.derivative(x->sum(ngLN(x)),0.9)
Zygote.gradient(x->sum(ngLN(x)),0.9)

struct Material{T}
	ε::T
	fε::Function
	fng::Function
	fgvd::Function
end

substitute(nₒ²_MgO_LiNbO₃_sym,[λ=>0.8])

@variables θ
using Rotations, StaticArrays, SparseArrays

R1 = RotY(0.1)

Matrix(RotY(θ))





SMatrix{3,3}(R1)
