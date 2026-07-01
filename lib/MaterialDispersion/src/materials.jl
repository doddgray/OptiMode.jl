using Symbolics: get_variables, make_array, SerialForm, Func, toexpr, _build_and_inject_function, @__MODULE__, MultithreadedForm, tosymbol, Sym, wrap, unwrap, MakeTuple, substitute, value
# using SymbolicUtils: @rule, @acrule, RuleSet, numerators, denominators, get_pvar2sym, get_sym2term, unpolyize, numerators, denominators #, toexpr
using SymbolicUtils: Term
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using SymbolicUtils.Code: toexpr, MakeArray 
export AbstractMaterial, Material, RotatedMaterial, get_model, generate_fn, Δₘ_factors, Δₘ
export rotate, unique_axes, nn̂g, nĝvd, nn̂g_model, nn̂g_fn, nĝvd_model, nĝvd_fn, ε_fn
export n²_sym_fmt1, n_sym_cauchy, has_model, χ⁽²⁾_fn, material_name, n_model, ng_model, gvd_model
export kerr_n2, with_kerr_n2, set_kerr_n2!
export NumMat 


get_array_vars(A) = mapreduce(x->wrap.(get_variables(x)),union,A)

"""
    generate_array_fn(args, model::AbstractArray)

Generate a fast, out-of-place function evaluating the symbolic array `model` as a
function of the symbolic arguments `args`.
"""
function generate_array_fn(args::AbstractVector, model::AbstractArray; expr_module=@__MODULE__(), parallel=SerialForm())
	return first(build_function(model, args...; expression=Val{false}))
end

"""
################################################################################
#																			   #
#							   	   Materials							   	   #
#																			   #
################################################################################
"""

"""
    AbstractMaterial

Supertype of all material descriptions ([`Material`](@ref), [`RotatedMaterial`](@ref),
[`NumMat`](@ref)). A material is fundamentally a collection of *symbolic models* —
Symbolics.jl expressions in variables such as the vacuum wavelength `λ` (μm), frequency
`ω` (μm⁻¹) and temperature `T` (°C) — from which fast numeric functions and exact
symbolic derivatives are generated on demand.
"""
abstract type AbstractMaterial end

"""
    Material(models::Dict, defaults::Dict, name::Symbol, color)

A dispersive optical material described by symbolic models.

`models` maps model names to symbolic expressions (or constants):

| key      | meaning                                              | typical form |
|----------|------------------------------------------------------|--------------|
| `:ε`     | 3×3 relative-permittivity tensor ``ε(λ, T, …)``      | diagonal of squared Sellmeier indices |
| `:χ⁽²⁾`  | second-order susceptibility tensor ``χ^{(2)}_{ijk}`` | 3×3×3 constant/symbolic array |
| `:n₂`    | Kerr coefficient (μm²/W), see [`kerr_n2`](@ref)      | constant or expression in `λ`/`ω` |

`defaults` maps variable symbols (e.g. `:T`) to values substituted when a generated
function does not expose that variable as an argument. All wavelengths are vacuum
wavelengths in μm, frequencies are `ω = 1/λ` in μm⁻¹ (i.e. units with ``c = 1``).

Materials in the bundled library (`Si₃N₄`, `SiO₂`, `LiNbO₃`, `MgO_LiNbO₃`, `Si`, `Ge`,
`LiB₃O₅`, `αAl₂O₃`, …) are constructed this way; see `src/material_lib/`.

See also [`get_model`](@ref), [`generate_fn`](@ref), [`ε_fn`](@ref), [`rotate`](@ref).
"""
struct Material <: AbstractMaterial
	models::Dict
	defaults::Dict
	name::Symbol
	color::Color
end

"""
    NumMat(mat::AbstractMaterial)
    NumMat(ε)

Purely *numeric* material: pre-generated functions of wavelength for the dielectric
tensor (`fε`), the group-index-weighted tensor ``n \\hat{n}_g`` (`fnng`), the
GVD-weighted tensor (`fngvd`) and ``χ^{(2)}`` (`fχ⁽²⁾`), captured from a symbolic
[`Material`](@ref) (or from constant dielectric data). Useful when material models must
be passed across processes or stored without the symbolic toolchain.
"""
struct NumMat{T,F1,F2,F3,F4,TC} # <: AbstractMaterial
	ε::T
	fε::F1
	fnng::F2
	fngvd::F3
	fχ⁽²⁾::F4
	name::Symbol
	color::TC
end

function NumMat(mat::AbstractMaterial;expr_module=@__MODULE__())
	eps_model = ε_model_λ(mat)
	feps = ε_fn(mat)
	fnng = generate_fn(mat,nn̂g_model(mat),:λ; expr_module)
	fngvd = generate_fn(mat,nĝvd_model(mat),:λ; expr_module)
	fchi2 = χ⁽²⁾_fn(mat)
	return NumMat(eps_model,feps,fnng,fngvd,fchi2,nameof(mat),mat.color)
end
Material(nmat::NumMat) = nmat
get_model(nmat::NumMat,epssymb,args...) = nmat.ε
ε_fn(mat::NumMat) = mat.fε
nn̂g_fn(mat::NumMat) =  mat.fnng
nĝvd_fn(mat::NumMat) = mat.fngvd
χ⁽²⁾_fn(mat::NumMat) = mat.fχ⁽²⁾

function NumMat(eps_in;color=RGB(0,0,0))
	constant_epsilon = ε_tensor(eps_in)
	eps_model = constant_epsilon
	feps = x->constant_epsilon
	fnng = x->constant_epsilon
	fngvd = x->zero(constant_epsilon)
	fchi2 = (x1,x2,x3)->zeros(eltype(constant_epsilon),3,3,3)
	return NumMat(eps_model,feps,fnng,fngvd,fchi2,material_name(eps_in),color)
end


# constructor adding random color when color is not specified
Material(models::Dict,defaults::Dict,name::Symbol) = Material(models,defaults,name,RGBA(rand(3)...,1.0))


import Base: nameof
Base.nameof(mat::AbstractMaterial) = getfield(mat, :name)
Base.nameof(mat::NumMat) = getfield(mat, :name)

material_name(x::Real) = Symbol("Const_Material_$x")
material_name(x::AbstractVector) = Symbol("Const_Material_$(x[1])_$(x[2])_$(x[3])")
material_name(x::AbstractMatrix) = Symbol("Const_Material_$(x[1,1])_$(x[2,2])_$(x[3,3])")

Material(x) = Material(Dict([ε_tensor(x),]),Dict([]),material_name(x))
Material(mat::AbstractMaterial) = mat



"""
    get_model(mat::AbstractMaterial, model_name::Symbol, args...) -> Num / Array{Num}

Return the symbolic model `model_name` of `mat` with every variable *not* listed in
`args` replaced by its default value from `mat.defaults`. The result is a symbolic
expression (or array of expressions) in the free variables `args`, ready for
[`generate_fn`](@ref) or symbolic differentiation.

```julia
get_model(Si₃N₄, :ε, :λ)        # 3×3 symbolic ε(λ)
get_model(MgO_LiNbO₃, :ε, :λ, :T)  # 3×3 symbolic ε(λ, T)
```

For a [`RotatedMaterial`](@ref) the parent model is fetched first and then rotated.
"""
function get_model(mat::AbstractMaterial,model_name::Symbol,args...)
	model = mat.models[model_name]
	missing_var_defaults = filter(x->!in(first(x),tosymbol.(args)),mat.defaults)
	# subs =  Dict([(Sym{Real}(k),v) for (k,v) in missing_var_defaults])
	subs =  Dict([(Sym{Real}(k),v) for (k,v) in missing_var_defaults])
	if typeof(model)<:AbstractArray
		model_subs = substitute.(model, (subs,))
	else
		model_subs = substitute(model, subs)
	end
	return model_subs
end

function get_model(mat::AbstractMaterial,fn_model::Tuple{TF,Symbol},args...) where TF<:Function
	first(fn_model)(get_model(mat,fn_model[2],args...))
end

"""
    generate_fn(mat::AbstractMaterial, model_name::Symbol, args...) -> Function
    generate_fn(mat::AbstractMaterial, model, args...) -> Function

Build a fast numeric function evaluating the symbolic model `model_name` of `mat` (or a
symbolic expression `model` directly) at the free variables `args`, e.g.

```julia
fε_λT = generate_fn(MgO_LiNbO₃, :ε, :λ, :T)
fε_λT(1.55, 35.0)                # 3×3 ε at λ = 1.55 μm, T = 35 °C
fng = generate_fn(SiO₂, ng_model(SiO₂), :λ)
```

Variables not listed in `args` are fixed at their `mat.defaults` values. Code is
generated by `Symbolics.build_function` (out-of-place variant).
"""
function generate_fn(mat::AbstractMaterial,model_name::Symbol,args...; expr_module=@__MODULE__(), parallel=SerialForm())
	model = get_model(mat,model_name,args...)
	if typeof(model)<:AbstractArray
		# fn = generate_array_fn([Num(Sym{Real}(arg)) for arg in args],model; expr_module, parallel)
		fn = build_function(model,[Num(Sym{Real}(arg)) for arg in args]...;expression=Val{false})[1]
	else
		fn = build_function(model,[Num(Sym{Real}(arg)) for arg in args]...;expression=Val{false})
	end
	return fn
end

function generate_fn(mat::AbstractMaterial,model,args...; expr_module=@__MODULE__(), parallel=SerialForm())
	# model = get_model(mat,model_name,args...)
	if typeof(model)<:AbstractArray
		# fn = generate_array_fn([Num(Sym{Real}(arg)) for arg in args],model; expr_module, parallel)
		fn = build_function(model,[Num(Sym{Real}(arg)) for arg in args]...;expression=Val{false})[1]
	else
		fn = build_function(model,[Num(Sym{Real}(arg)) for arg in args]...;expression=Val{false})
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

"""
    RotatedMaterial(parent, rotation, rotation_defaults, name, color)

A material whose tensor models are those of `parent` expressed in a rotated frame.
Rank-2 tensors transform as

```math
ε^{rot}_{ij} = \\mathcal{R}_{ai}\\,\\mathcal{R}_{bj}\\,ε_{ab}
```

(and rank-3 ``χ^{(2)}`` with three factors of ``\\mathcal{R}``), where ``\\mathcal{R}``
may itself be symbolic (e.g. parameterized by an angle with a default in
`rotation_defaults`). Construct with [`rotate`](@ref). Scalar models (e.g. the Kerr
coefficient `:n₂`) are rotation-invariant and pass through to the parent.
"""
struct RotatedMaterial{TM,TR} <: AbstractMaterial
	parent::TM
	rotation::TR
	rotation_defaults::Dict
	name::Symbol
	color::Color
end

"""
    rotate(mat::AbstractMaterial, 𝓡::AbstractMatrix; name, color) -> RotatedMaterial
    rotate(χ::AbstractMatrix, 𝓡)   /   rotate(χ::AbstractArray{<:Any,3}, 𝓡)

Rotate a material (or a bare rank-2/rank-3 susceptibility tensor) by the 3×3 rotation
matrix `𝓡`: ``χ^{rot}_{ij} = \\mathcal{R}_{ai}\\mathcal{R}_{bj} χ_{ab}`` and
``χ^{rot}_{ijk} = \\mathcal{R}_{ai}\\mathcal{R}_{bj}\\mathcal{R}_{ck} χ_{abc}``.
Used e.g. to model rotated-axis (X/Y/Z-cut) nonlinear crystals:

```julia
using Rotations
LN_xcut = rotate(LiNbO₃, Matrix(RotZ(π/2)); name=:LiNbO₃_rot)
```
"""
function rotate(χ::AbstractMatrix,𝓡::AbstractMatrix)
	@tullio χᵣ[i,j] := 𝓡[a,i] * 𝓡[b,j] * χ[a,b]  fastmath=true
end

function rotate(χ::AbstractArray{T,3},𝓡::AbstractMatrix) where {T<:Real}
	@tullio χᵣ[i,j,k] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * χ[a,b,c]  fastmath=true
end

function rotate(χ::AbstractArray{T,4},𝓡::TR) where {T<:Real, TR<:StaticMatrix{3,3}}
	@tullio χᵣ[i,j,k,l] := 𝓡[a,i] * 𝓡[b,j] * 𝓡[c,k] * 𝓡[d,l] * χ[a,b,c,d]  fastmath=true
end

# rotate(χ::Real,𝓡::StaticMatrix{3,3}) = χ

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
	# model = rotate(mat.parent.models[model_name],mat.rotation)
	model = rotate(get_model(mat.parent,model_name,args...),mat.rotation)
	# defs = merge(mat.parent.defaults,mat.rotation_defaults)
	# missing_var_defaults = filter(x->!in(first(x),tosymbol.(args)),defs)
	missing_var_defaults = filter(x->!in(first(x),tosymbol.(args)),mat.rotation_defaults)
	subs =  Dict([(Sym{Real}(k),v) for (k,v) in missing_var_defaults])
	if typeof(model)<:AbstractArray
		model_subs = substitute.(model, (subs,))
	else
		model_subs = substitute(model, subs)
	end
	return model_subs
end

# get_model(mat::RotatedMaterial,model_name::Symbol,model_fn::Function,args...) = model_fn(get_model(mat,model_name,args...))
function get_model(mat::RotatedMaterial,fn_model::Tuple{TF,Symbol},args...) where TF<:Function
	first(fn_model)(get_model(mat,fn_model[2],args...))
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

"""
    n²_sym_fmt1(λ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0)

Three-term Sellmeier model for the squared refractive index,

```math
n^2(λ) = A_0 + \\frac{B_1 λ^2}{λ^2 - C_1} + \\frac{B_2 λ^2}{λ^2 - C_2}
             + \\frac{B_3 λ^2}{λ^2 - C_3},
```

with `λ` the vacuum wavelength in μm and the `Cᵢ` in μm². Each term models an optical
resonance of strength `Bᵢ` at wavelength ``\\sqrt{C_i}``. Returns a symbolic
expression when `λ` is symbolic — this is the standard building block for the `:ε`
models in the material library.
"""
function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    λ² = λ^2
    A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
end

function n²_sym_fmt1_ω( ω ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    A₀  + B₁ / ( 1 - C₁*ω^2 ) + B₂ / ( 1 - C₂*ω^2 ) + B₃ / ( 1 - C₃*ω^2 )
end

"""
    n_sym_cauchy(λ; A=1, B=0, C=0)

Cauchy model for the refractive index,
``n(λ) = A + B/λ^2 + C/λ^4`` (`λ` in μm); a useful fit form far from material
resonances.
"""
function n_sym_cauchy( λ ; A=1, B=0, C=0, B₂=0, kwargs...)
    A   +   B / λ^2    +   C / λ^4
end

function n_sym_cauchy_ω( ω ; A=1, B=0, C=0, B₂=0, kwargs...)
    A   +   B * ω^2    +   C * ω^4
end

"""
Dispersive thermo-optic Sellmeier format based on:
	Frey, Leviton and Madison, "Temperature-dependent refractive index of silicon and germanium"
	https://arxiv.org/pdf/physics/0606168.pdf

in work from NASA Goddard using their Cryogenic High-Accuracy Refraction Measuring System (CHARMS).

The squared index of refraction n² is approximated in a Sellmeier form 

	n² = 1 + ∑ᵢ ( Sᵢ * λ² ) / ( λ² - λᵢ² )

with temperature-dependent coefficients Sᵢ and λᵢ representing the strengths and vacuum 
wavelengths of optical resonances, respectively. Sᵢ and λᵢ are both calcualted as fourth-order
polynomials in absolute temperature `T` (in deg. Kelvin). Model parameters are supplied as
n × 5 matrices Sᵢⱼ and λᵢⱼ, where n is the number of Sellmeier terms. Sᵢ and λᵢ are 
calculated as dot products

	Sᵢ	=	Sᵢⱼ ⋅ [1, T, T^2, T^3, T^4]
	λᵢ	=	λᵢⱼ ⋅ [1, T, T^2, T^3, T^4]

In the referenced paper three-term Sellemeier forms are used, and thus Sᵢⱼ and λᵢⱼ of the form

	Sᵢⱼ	= 	[	S₀₁		S₁₁		S₁₂		S₁₃		S₁₄
				S₀₂		S₂₁		S₂₂		S₂₃		S₂₄
				S₀₃		S₃₁		S₃₂		S₃₃		S₃₄		]

	λᵢⱼ	= 	[	λ₀₁		λ₁₁		λ₁₂		λ₁₃		λ₁₄
				λ₀₂		λ₂₁		λ₂₂		λ₂₃		λ₂₄
				λ₀₃		λ₃₁		λ₃₂		λ₃₃		λ₃₄		]

is provided for silicon and germanium in Tables 5 and 10, respectively.
"""
function n²_sym_NASA( λ, T ; Sᵢⱼ=zeros(3,5), λᵢⱼ=zeros(3,5), kwargs...)
    λ² 	= 	λ^2
	# Tₖ	=	T + 273.15
	# T_pows	=	[1, Tₖ, Tₖ^2, Tₖ^3, Tₖ^4]
	T_pows	=	[1.0, T, T^2, T^3, T^4]
	Sᵢ	=	Sᵢⱼ * T_pows
	λᵢ	=	λᵢⱼ * T_pows
	return 1 + sum( s_lm->((first(s_lm) * λ²)/(λ²-last(s_lm)^2)), zip(Sᵢ, λᵢ) )		# <--- nominal
	# return sum( s_lm->((first(s_lm)^2 * λ²)/(λ²-last(s_lm))), zip(Sᵢ, λᵢ) )
end

"""
Dispersive thermo-optic Sellmeier format based on:
	Frey, Leviton and Madison, "Temperature-dependent refractive index of silicon and germanium"
	https://arxiv.org/pdf/physics/0606168.pdf

in work from NASA Goddard using their Cryogenic High-Accuracy Refraction Measuring System (CHARMS).

The squared index of refraction n² is approximated in a Sellmeier form 

	n² = 1 + ∑ᵢ  Sᵢ / ( 1 - (ω * λᵢ)² )

with temperature-dependent coefficients Sᵢ and λᵢ representing the strengths and vacuum 
wavelengths of optical resonances, respectively. Sᵢ and λᵢ are both calcualted as fourth-order
polynomials in absolute temperature `T` (in deg. Kelvin). Model parameters are supplied as
n × 5 matrices Sᵢⱼ and λᵢⱼ, where n is the number of Sellmeier terms. Sᵢ and λᵢ are 
calculated as dot products

	Sᵢ	=	Sᵢⱼ ⋅ [1, T, T^2, T^3, T^4]
	λᵢ	=	λᵢⱼ ⋅ [1, T, T^2, T^3, T^4]

In the referenced paper three-term Sellemeier forms are used, and thus Sᵢⱼ and λᵢⱼ of the form

	Sᵢⱼ	= 	[	S₀₁		S₁₁		S₁₂		S₁₃		S₁₄
				S₀₂		S₂₁		S₂₂		S₂₃		S₂₄
				S₀₃		S₃₁		S₃₂		S₃₃		S₃₄		]

	λᵢⱼ	= 	[	λ₀₁		λ₁₁		λ₁₂		λ₁₃		λ₁₄
				λ₀₂		λ₂₁		λ₂₂		λ₂₃		λ₂₄
				λ₀₃		λ₃₁		λ₃₂		λ₃₃		λ₃₄		]

is provided for silicon and germanium in Tables 5 and 10, respectively.
"""
function n²_sym_NASA_ω( ω, T ; Sᵢⱼ=zeros(3,5), λᵢⱼ=zeros(3,5), kwargs...)
	# Tₖ	=	T + 273.15
	# T_pows	=	[1, Tₖ, Tₖ^2, Tₖ^3, Tₖ^4]
	T_pows	=	[1, T, T^2, T^3, T^4]
	Sᵢ	=	Sᵢⱼ * T_pows
	λᵢ	=	λᵢⱼ * T_pows
	# return 1 + sum( (s,lm)->((s^2)/(1-(lm*ω)^2)), zip(Sᵢ, λᵢ) )
	return 1 + sum( s_lm->(first(s_lm)/(1-(last(s_lm)*ω)^2)), zip(Sᵢ, λᵢ) )
	# return 1 + sum( s_lm->((first(s_lm)^2)/(1-last(s_lm)*ω^2)), zip(Sᵢ, λᵢ) )
end


# Miller's Delta scaling
"""
    Δₘ_factors(λs, ε_sym)

Per-axis products of the linear-susceptibility factors ``\\prod_p χ^{(1)}_{ii}(λ_p) =
\\prod_p (ε_{ii}(λ_p) - 1)`` over the wavelengths `λs`, evaluated from the symbolic
dielectric model `ε_sym`. The building block of Miller's rule (see [`Δₘ`](@ref)).
"""
function Δₘ_factors(λs,ε_sym)
	λ = Num(first(get_variables(sum(ε_sym))))
	diagε_m1 = Vector(diag(ε_sym)) .- 1
	# mapreduce(lm->substitute.( diagε_m1, ([λ=>lm],)), .*, λs)
	mapreduce(i->substitute.( diagε_m1, [λ=>λs[i]]), .*, 1:length(λs))
end

"""
    Δₘ(λs, ε_sym, λᵣs, χᵣ)

Miller's-rule wavelength scaling of a second-order susceptibility tensor: given
``χ^{(2)}`` data `χᵣ` measured at reference wavelengths `λᵣs`, estimate it at new
wavelengths `λs` using the (empirically near-constant) Miller delta

```math
χ^{(2)}_{ijk}(λ_1,λ_2,λ_3) ≈ χ^{(2),ref}_{ijk}\\,
\\frac{χ^{(1)}_{ii}(λ_1)\\,χ^{(1)}_{jj}(λ_2)\\,χ^{(1)}_{kk}(λ_3)}
     {χ^{(1)}_{ii}(λ^r_1)\\,χ^{(1)}_{jj}(λ^r_2)\\,χ^{(1)}_{kk}(λ^r_3)},
\\qquad χ^{(1)}_{ii}(λ) = ε_{ii}(λ) - 1.
```
"""
function Δₘ(λs::AbstractVector,ε_sym, λᵣs::AbstractVector, χᵣ::AbstractArray{T,3}) where T
	dm = Δₘ_factors(λs,ε_sym) ./ Δₘ_factors(λᵣs,ε_sym)
	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i] * dm[j] * dm[k] fastmath=true
end

# Symbolic Differentiation
"""
    ng_model(n_model::Num, λ::Num)
    ng_model(mat::AbstractMaterial; symbol=:λ)

Symbolic group-index model obtained by exact differentiation of the refractive-index
model ``n(λ)``:

```math
n_g(λ) = \\frac{∂k}{∂ω} = n - λ \\frac{dn}{dλ},
```

the index governing the propagation speed of pulse envelopes, ``v_g = c/n_g``. The
material method differentiates ``\\sqrt{ε_{ii}}`` element-wise.
"""
function ng_model(n_model::Num, λ::Num)
	Dλ = Differential(λ)
	return n_model - ( λ * expand_derivatives(Dλ(n_model),true) )
end

"""
    gvd_model(n_model::Num, λ::Num)
    gvd_model(mat::AbstractMaterial; symbol=:λ)

Symbolic group-velocity-dispersion model obtained by exact differentiation of the
refractive-index model:

```math
\\mathrm{GVD}(λ) = \\frac{∂n_g}{∂ω} = \\frac{∂^2 k}{∂ω^2} = λ^3 \\frac{d^2n}{dλ^2}
```

in units with ``c = 1`` (lengths in μm, so GVD in μm; multiply by ``1/(2πc^2)`` in SI
to obtain ``β_2`` in fs²/mm).
"""
function gvd_model(n_model::Num, λ::Num)
	Dλ = Differential(λ)
	return λ^3 * expand_derivatives(Dλ(Dλ(n_model)),true)
end

ng_model(n_model::AbstractArray{Num}, λ::Num) = ng_model.(n_model,(λ,))
gvd_model(n_model::AbstractArray{Num}, λ::Num) = gvd_model.(n_model,(λ,))

function ng_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	n_model = sqrt.(ε_model_λ(mat))
	return ng_model(n_model,λ)
end

function gvd_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	n_model = sqrt.(ε_model_λ(mat))
	return gvd_model(n_model,λ)
end

"""
    nn̂g_model(mat::AbstractMaterial; symbol=:λ)
    nn̂g_model(ε_model::AbstractMatrix{Num}; symbol=:λ)

Symbolic model of the *group-index-weighted* dielectric tensor

```math
(n\\hat{n}_g)_{ij} = \\frac{∂(ω\\,ε_{ij})}{∂ω} = ε_{ij} + ω\\frac{∂ε_{ij}}{∂ω}
                   = -λ^2 \\frac{d}{dλ}\\!\\left(\\frac{ε_{ij}}{λ}\\right),
```

which for an isotropic medium reduces to ``n·n_g`` on the diagonal. This tensor enters
electromagnetic energy-density/power normalization and perturbation theory for
dispersive media.
"""
function nn̂g_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	ε_model = ε_model_λ(mat)
	# ω∂ε∂ω_model =   -1 * λ .* expand_derivatives.(Dλ.(ε_model),(true,))
	# return ω∂ε∂ω_model ./ 2
	∂∂ω_ωε_model =   (-1 * λ^2) .* expand_derivatives.(Dλ.(ε_model./λ),(true,))
	return ∂∂ω_ωε_model
end

"""
    nĝvd_model(mat::AbstractMaterial; symbol=:λ)
    nĝvd_model(ε_model::AbstractMatrix{Num}; symbol=:λ)

Symbolic model of the second frequency derivative of ``ω ε``,

```math
\\frac{∂^2(ω\\,ε_{ij})}{∂ω^2} = -λ^2 \\frac{d}{dλ}\\,(n\\hat{n}_g)_{ij},
```

the tensor generalization of ``d(n\\,n_g)/dω`` used in group-velocity-dispersion
calculations for dispersive media. See [`nn̂g_model`](@ref).
"""
function nĝvd_model(mat::AbstractMaterial; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	# ∂ε∂ω_model = nn̂g_model(mat; symbol) .* (2 / λ)
	# ω∂²ε∂ω²_model =   -1 * λ .* expand_derivatives.(Dλ.(∂ε∂ω_model),(true,))
	# return (∂ε∂ω_model .+ ω∂²ε∂ω²_model) ./ 2
	nng_model = nn̂g_model(mat; symbol)
	∂²∂ω²_ωε_model =   (-1 * λ^2) .* expand_derivatives.(Dλ.(nng_model),(true,))
	return ∂²∂ω²_ωε_model
end

function nn̂g_model(ε_model::AbstractMatrix{Num}; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	# ω∂ε∂ω_model =   -1 * λ .* expand_derivatives.(Dλ.(ε_model),(true,))
	# return ω∂ε∂ω_model ./ 2
	∂∂ω_ωε_model =   (-1 * λ^2) .* expand_derivatives.(Dλ.(ε_model./λ),(true,))
	return ∂∂ω_ωε_model
end

function nĝvd_model(ε_model::AbstractMatrix{Num}; symbol=:λ)
	λ = Num(Sym{Real}(symbol))
	Dλ = Differential(λ)
	# ∂ε∂ω_model = nn̂g_model(ε_model; symbol) .* (2 / λ)
	# ω∂²ε∂ω²_model =   -1 * λ .* expand_derivatives.(Dλ.(∂ε∂ω_model),(true,))
	# return (∂ε∂ω_model .+ ω∂²ε∂ω²_model) ./ 2
	nng_model = nn̂g_model(ε_model; symbol)
	∂²∂ω²_ωε_model =   (-1 * λ^2) .* expand_derivatives.(Dλ.(nng_model),(true,))
	return ∂²∂ω²_ωε_model
end

# generate_fn(mat::AbstractMaterial,model_name::Symbol,args...; expr_module=@__MODULE__(), parallel=SerialForm())

"""
    ε_model_λ(mat)

Return the symbolic dielectric tensor model of `mat` as a function of the free
variable `λ` only. Material models may be stored in terms of frequency `ω` and/or
vacuum wavelength `λ`; both are left free here and `ω` is then substituted by `1/λ`.
"""
function ε_model_λ(mat::AbstractMaterial)
	λ = Num(Sym{Real}(:λ))
	ω = Num(Sym{Real}(:ω))
	model = get_model(mat,:ε,:λ,:ω)
	return substitute.(model, (Dict(ω => 1/λ),))
end

"""
    ε_fn(mat::AbstractMaterial) -> Function

Generate a fast function `λ -> ε` mapping vacuum wavelength (μm) to the material's 3×3
relative-permittivity tensor, with all other model variables (e.g. temperature) at
their defaults.

```julia
n_SiN_1550 = sqrt(ε_fn(Si₃N₄)(1.55)[1,1])   # ≈ 1.996
```
"""
ε_fn(mat::AbstractMaterial) = generate_array_fn([Num(Sym{Real}(:λ)) ,],ε_model_λ(mat))
"""
    nn̂g_fn(mat) -> Function

Fast function `λ -> ∂(ωε)/∂ω` (3×3 group-index-weighted dielectric tensor); see
[`nn̂g_model`](@ref).
"""
nn̂g_fn(mat::AbstractMaterial) =  generate_array_fn([Num(Sym{Real}(:λ)) ,],nn̂g_model(mat))
"""
    nĝvd_fn(mat) -> Function

Fast function `λ -> ∂²(ωε)/∂ω²` (3×3); see [`nĝvd_model`](@ref).
"""
nĝvd_fn(mat::AbstractMaterial) =  generate_array_fn([Num(Sym{Real}(:λ)) ,],nĝvd_model(mat))



"""
    χ⁽²⁾_fn(mat) -> Function

Generate a function `(λ₁, λ₂, λ₃) -> χ⁽²⁾` returning the material's 3×3×3 second-order
susceptibility tensor at the three interacting vacuum wavelengths (μm), or the zero
tensor for materials without a `:χ⁽²⁾` model. Wavelength dependence, when present, is
modeled by Miller's-rule scaling ([`Δₘ`](@ref)) of reference data.
"""
function χ⁽²⁾_fn(mat::AbstractMaterial;expr_module=@__MODULE__())
	if has_model(mat,:χ⁽²⁾)
		@variables λs[1:3]
		fn = generate_array_fn(λs,get_model(mat,:χ⁽²⁾,:λs); expr_module)
		# return generate_array_fn([Num(Sym{Real}(:λs₁)), Num(Sym{Real}(:λs₂)), Num(Sym{Real}(:λs₃))],get_model(mat,:χ⁽²⁾,:λs₁,:λs₂,:λs₃); expr_module)
		# return generate_fn(mat,get_model(mat,:χ⁽²⁾,:λs₁,:λs₂,:λs₃),Num(Sym{Real}(:λs₁)), Num(Sym{Real}(:λs₂)), Num(Sym{Real}(:λs₃)); expr_module, parallel=SerialForm())
		# return generate_fn(mat,:χ⁽²⁾,Num(Sym{Real}(:λs₁)), Num(Sym{Real}(:λs₂)), Num(Sym{Real}(:λs₃)); expr_module, parallel=SerialForm())
		return (lm1,lm2,lm3) -> fn([lm1,lm2,lm3])
	else
		return (lm1,lm2,lm3) -> zero(SArray{Tuple{3,3,3}})
	end
end


nn̂g(mat::AbstractMaterial,lm::Real) = SMatrix{3,3}(nn̂g_fn(mat)(lm))
nĝvd(mat::AbstractMaterial,lm::Real) = SMatrix{3,3}(nĝvd_fn(mat)(lm))

"""
################################################################################
#                                                                              #
#            Kerr nonlinearity: intensity-dependent refractive index           #
#                                                                              #
################################################################################

Materials may carry an intensity-dependent refractive-index coefficient `n₂`
(`n(I) = n₀ + n₂ I`) in units of μm²/W, stored as the `:n₂` model entry — either a
constant `Real` or a symbolic expression in the vacuum wavelength `λ` (μm) and/or
frequency `ω` (μm⁻¹). Materials without an `:n₂` model are linear (`n₂ = 0`).
"""

"""
    kerr_n2(mat, λ=1.55) -> Float64

Intensity-dependent refractive-index coefficient `n₂` of a material in μm²/W at vacuum
wavelength `λ` (μm). Returns `0.0` for materials without a specified `:n₂` model
(including raw dielectric data and `NumMat`s built without one).
"""
function kerr_n2(mat::Material, λ::Real=1.55)
	haskey(mat.models, :n₂) || return 0.0
	model = mat.models[:n₂]
	# Num <: Real, so explicitly exclude symbolic models from the constant fast path
	(model isa Number && !(model isa Num)) && return Float64(model)
	m = get_model(mat, :n₂, :λ, :ω)
	λv = Num(Sym{Real}(:λ))
	ωv = Num(Sym{Real}(:ω))
	val = substitute(m, Dict(λv => λ, ωv => 1 / λ))
	return Float64(Symbolics.value(val))
end
kerr_n2(mat::RotatedMaterial, λ::Real=1.55) = kerr_n2(mat.parent, λ)  # n₂ is scalar: rotation-invariant
kerr_n2(::Any, λ::Real=1.55) = 0.0

"""
    with_kerr_n2(mat, n2) -> Material

Return a copy of `mat` with the Kerr coefficient model `:n₂` set to `n2` — a constant
in μm²/W or a symbolic expression in `λ` (μm) / `ω` (μm⁻¹).
"""
with_kerr_n2(mat::Material, n2) =
	Material(merge(mat.models, Dict{Any,Any}(:n₂ => n2)), mat.defaults, mat.name, mat.color)

"""
    set_kerr_n2!(mat, n2) -> Material

In-place version of [`with_kerr_n2`](@ref) (the `models` Dict is mutated).
"""
set_kerr_n2!(mat::Material, n2) = (mat.models[:n₂] = n2; mat)

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

################################################################################
#                                Load Materials                                #
################################################################################
include("material_lib/vacuum.jl")
include("material_lib/LiNbO3.jl")
include("material_lib/LiNbO3_MgO.jl")
include("material_lib/LiTaO3.jl")
include("material_lib/SiO2.jl")
include("material_lib/Si3N4.jl")
include("material_lib/Ta2O5.jl")
include("material_lib/αAl2O3.jl")
include("material_lib/LiB3O5.jl")
include("material_lib/silicon.jl")
include("material_lib/germanium.jl")
# include("material_lib/GaAs.jl")
# include("material_lib/MgF2.jl")
# include("material_lib/HfO2.jl")
