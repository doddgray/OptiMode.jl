using ForwardDiff
using Zygote: @adjoint, Numeric, literal_getproperty, accum
using ChainRules: Thunk, @non_differentiable
export sum2, jacobian, ε⁻¹_bar!, ε⁻¹_bar, ∂ω²∂k_adj, Mₖᵀ_plus_Mₖ, ∂²ω²∂k²
export ∇ₖmag_m_n, ∇HMₖH, ∇M̂, ∇solve_k, ∇solve_k!, solve_adj!

### ForwardDiff Comoplex number support
# ref: https://github.com/JuliaLang/julia/pull/36030
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/455
Base.float(d::ForwardDiff.Dual{T}) where T = ForwardDiff.Dual{T}(float(d.value), d.partials)
Base.prevfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(prevfloat(float(d.value)), d.partials)
Base.nextfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(nextfloat(float(d.value)), d.partials)
function Base.ldexp(x::T, e::Integer) where T<:ForwardDiff.Dual
    if e >=0
        x * (1<<e)
    else
        x / (1<<-e)
    end
end

### ForwardDiff FFT support
# ref: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495/files
# https://discourse.julialang.org/t/forwarddiff-and-zygote-cannot-automatically-differentiate-ad-function-from-c-n-to-r-that-uses-fft/52440/18
ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) =
    Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual(real(val), map(real, parts)),
            ForwardDiff.Dual(imag(val), map(imag, parts)),
        )
    end
end

# used with the ForwardDiff+FFTW code above, this Zygote.extract method
# enables Zygote.hessian to work on real->real functions that internally use
# FFTs (and thus complex numbers)
import Zygote: extract
function Zygote.extract(xs::AbstractArray{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N}
  J = similar(xs, complex(V), N, length(xs))
  for i = 1:length(xs), j = 1:N
    J[j, i] = xs[i].re.partials.values[j] + im * xs[i].im.partials.values[j]
  end
  x0 = ForwardDiff.value.(xs)
  return x0, J
end

####
# Example code for defining custom ForwardDiff rules, copied from YingboMa's gist:
# https://gist.github.com/YingboMa/c22dcf8239a62e01b27ac679dfe5d4c5
# using ForwardDiff
# goo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
# foo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
# function foo(u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
#     # unpack: AoS -> SoA
#     vs = ForwardDiff.value.(u)
#     # you can play with the dimension here, sometimes it makes sense to transpose
#     ps = mapreduce(ForwardDiff.partials, hcat, u)
#     # get f(vs)
#     val = foo(vs)
#     # get J(f, vs) * ps (cheating). Write your custom rule here
#     jvp = ForwardDiff.jacobian(goo, vs) * ps
#     # pack: SoA -> AoS
#     return map(val, eachrow(jvp)) do v, p
#         ForwardDiff.Dual{T}(v, p...) # T is the tag
#     end
# end
# ForwardDiff.gradient(u->sum(cumsum(foo(u))), [1, 2, 3]) == ForwardDiff.gradient(u->sum(cumsum(goo(u))), [1, 2, 3])
####

# AD rules for StaticArrays Constructors
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:HybridArray}, x::AbstractArray) = ( T(x), dv -> (NO_FIELDS, dv) )

# AD rules for reinterpreting back and forth between N-D arrays of SVectors and (N+1)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T2},N2}) where {T1,N1,T2,N2}
	# return ( reinterpret(reshape,T1,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SVector{N1,T1}, Δ ) ) )
	function reinterpret_reshape_SV_pullback(Δ)
		return (NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret(reshape,SVector{N1,eltype(Δ)},Δ))
	end
	( reinterpret(reshape,T1,A), reinterpret_reshape_SV_pullback )
end
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
	return ( reinterpret(reshape,type,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ ) ) )
end

# need adjoint for constructor:
# Base.ReinterpretArray{Float64, 2, SVector{3, Float64}, Matrix{SVector{3, Float64}}, false}. Gradient is of type FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
import Base: ReinterpretArray
function ChainRulesCore.rrule(T::Type{ReinterpretArray{T1, N1, SVector{N2, T2}, Array{SVector{N3, T3},N4}, IsReshaped}}, x::AbstractArray)  where {T1, N1, N2, T2, N3, T3, N4, IsReshaped}
	function ReinterpretArray_SV_pullback(Δ)
		if IsReshaped
			Δr = reinterpret(reshape,SVector{N2,eltype(Δ)},Δ)
		else
			Δr = reinterpret(SVector{N2,eltype(Δ)},Δ)
		end
		return (NO_FIELDS, Δr)
	end
	( T(x), ReinterpretArray_SV_pullback )
end

# AD rules for reinterpreting back and forth between N-D arrays of SMatrices and (N+2)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SMatrix{N1,N2,T2,N3},N4}) where {T1,T2,N1,N2,N3,N4}
	# @show A
	# @show eltype(A)
	# @show type
	# @show size(reinterpret(reshape,T1,A))
	# @show N1*N2
	# function f_pb(Δ)
	# 	@show eltype(Δ)
	# 	@show size(Δ)
	# 	# @show Δ
	# 	@show typeof(Δ)
	# 	return ( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SMatrix{N1,N2,T1,N3}, Δ ) )
	# end
	# return ( reinterpret(reshape,T1,A), Δ->f_pb(Δ) )
	return ( reinterpret(reshape,T1,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SMatrix{N1,N2,T1,N3}, Δ ) ) )
end
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SMatrix{N1,N2,T1,N3}},A::AbstractArray{T1}) where {T1,T2,N1,N2,N3}
	@show type
	@show eltype(A)
	return ( reinterpret(reshape,type,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ ) ) )
end

# AD rules for fast norms of types SVector{2,T} and SVector{2,3}

function _norm2_back_SV2r(x::SVector{2,T}, y, Δy) where T<:Real
    ∂x = Vector{T}(undef,2)
    ∂x .= x .* (real(Δy) * pinv(y))
    return reinterpret(SVector{2,T},∂x)[1]
end

function _norm2_back_SV3r(x::SVector{3,T}, y, Δy) where T<:Real
    ∂x = Vector{T}(undef,3)
    ∂x .= x .* (real(Δy) * pinv(y))
    return reinterpret(SVector{3,T},∂x)[1]
end

function _norm2_back_SV2r(x::SVector{2,T}, y, Δy) where T<:Complex
    ∂x = Vector{T}(undef,2)
    ∂x .= conj.(x) .* (real(Δy) * pinv(y))
    return reinterpret(SVector{2,T},∂x)[1]
end

function _norm2_back_SV3r(x::SVector{3,T}, y, Δy) where T<:Complex
    ∂x = Vector{T}(undef,3)
    ∂x .= conj.(x) .* (real(Δy) * pinv(y))
    return reinterpret(SVector{3,T},∂x)[1]
end

function ChainRulesCore.rrule(::typeof(norm), x::SVector{3,T}) where T<:Real
	y = LinearAlgebra.norm(x)
	function norm_pb(Δy)
		∂x = Thunk() do
			_norm2_back_SV3r(x, y, Δy)
		end
		return ( NO_FIELDS, ∂x )
	end
	norm_pb(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pb
end

function ChainRulesCore.rrule(::typeof(norm), x::SVector{2,T}) where T<:Real
	y = LinearAlgebra.norm(x)
	function norm_pb(Δy)
		∂x = Thunk() do
			_norm2_back_SV2r(x, y, Δy)
		end
		return ( NO_FIELDS, ∂x )
	end
	norm_pb(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pb
end

function ChainRulesCore.rrule(::typeof(norm), x::SVector{3,T}) where T<:Complex
	y = LinearAlgebra.norm(x)
	function norm_pb(Δy)
		∂x = Thunk() do
			_norm2_back_SV3c(x, y, Δy)
		end
		return ( NO_FIELDS, ∂x )
	end
	norm_pb(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pb
end

function ChainRulesCore.rrule(::typeof(norm), x::SVector{2,T}) where T<:Complex
	y = LinearAlgebra.norm(x)
	function norm_pb(Δy)
		∂x = Thunk() do
			_norm2_back_SV2c(x, y, Δy)
		end
		return ( NO_FIELDS, ∂x )
	end
	norm_pb(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pb
end



@non_differentiable KDTree(::Any)
@non_differentiable g⃗(::Any)
@non_differentiable _fftaxes(::Any)
# Examples of how to assert type stability for broadcasting custom types (see https://github.com/FluxML/Zygote.jl/issues/318 )
# Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T<:Real, V<:Real3Vector} = Real3Vector(Vector{T}(undef,3))
# Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T, V<:Real3Vector} = Array{T}(undef, size(bc))

@adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
@adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end


function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end


# now-removed Zygote trick to improve stability of `norm` pullback
# found referenced here: https://github.com/JuliaDiff/ChainRules.jl/issues/338
function Zygote._pullback(cx::Zygote.AContext, ::typeof(norm), x::AbstractArray, p::Real = 2)
  fallback = (x, p) -> sum(abs.(x).^p .+ eps(0f0)) ^ (one(eltype(x)) / p) # avoid d(sqrt(x))/dx == Inf at 0
  Zygote._pullback(cx, fallback, x, p)
end

"""
jacobian(f,x) : stolen from https://github.com/FluxML/Zygote.jl/pull/747/files

Construct the Jacobian of `f` where `x` is a real-valued array
and `f(x)` is also a real-valued array.
"""
function jacobian(f,x)
    y,back  = Zygote.pullback(f,x)
    k  = length(y)
    n  = length(x)
    J  = Matrix{eltype(y)}(undef,k,n)
    e_i = fill!(similar(y), 0)
    @inbounds for i = 1:k
        e_i[i] = oneunit(eltype(x))
        J[i,:] = back(e_i)[1]
        e_i[i] = zero(eltype(x))
    end
    (J,)
end

### Zygote StructArrays rules from https://github.com/cossio/ZygoteStructArrays.jl
@adjoint function (::Type{SA})(t::Tuple) where {SA<:StructArray}
    sa = SA(t)
    back(Δ::NamedTuple) = (values(Δ),)
    function back(Δ::AbstractArray{<:NamedTuple})
        nt = (; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...)
        return back(nt)
    end
    return sa, back
end

@adjoint function (::Type{SA})(t::NamedTuple) where {SA<:StructArray}
    sa = SA(t)
    back(Δ::NamedTuple) = (NamedTuple{propertynames(sa)}(Δ),)
    function back(Δ::AbstractArray)
        back((; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...))
    end
    return sa, back
end

@adjoint function (::Type{SA})(a::A) where {T,SA<:StructArray,A<:AbstractArray{T}}
    sa = SA(a)
    function back(Δsa)
        Δa = [(; (p => Δsa[p][i] for p in propertynames(Δsa))...) for i in eachindex(a)]
        return (Δa,)
    end
    return sa, back
end

# Must special-case for Complex (#1)
@adjoint function (::Type{SA})(a::A) where {T<:Complex,SA<:StructArray,A<:AbstractArray{T}}
    sa = SA(a)
    function back(Δsa) # dsa -> da
        Δa = [Complex(Δsa.re[i], Δsa.im[i]) for i in eachindex(a)]
        (Δa,)
    end
    return sa, back
end

@adjoint function literal_getproperty(sa::StructArray, ::Val{key}) where {key}
    key::Symbol
    result = getproperty(sa, key)
    function back(Δ::AbstractArray)
        nt = (; (k => zero(v) for (k,v) in pairs(fieldarrays(sa)))...)
        return (Base.setindex(nt, Δ, key), nothing)
    end
    return result, back
end

@adjoint Base.getindex(sa::StructArray, i...) = sa[i...], Δ -> ∇getindex(sa,i,Δ)
@adjoint Base.view(sa::StructArray, i...) = view(sa, i...), Δ -> ∇getindex(sa,i,Δ)
function ∇getindex(sa::StructArray, i, Δ::NamedTuple)
    dsa = (; (k => ∇getindex(v,i,Δ[k]) for (k,v) in pairs(fieldarrays(sa)))...)
    di = map(_ -> nothing, i)
    return (dsa, map(_ -> nothing, i)...)
end
# based on
# https://github.com/FluxML/Zygote.jl/blob/64c02dccc698292c548c334a15ce2100a11403e2/src/lib/array.jl#L41
∇getindex(a::AbstractArray, i, Δ::Nothing) = nothing
function ∇getindex(a::AbstractArray, i, Δ)
    if i isa NTuple{<:Any, Integer}
        da = Zygote._zero(a, typeof(Δ))
        da[i...] = Δ
    else
        da = Zygote._zero(a, eltype(Δ))
        dav = view(da, i...)
        dav .= Zygote.accum.(dav, Zygote._droplike(Δ, dav))
    end
    return da
end

@adjoint function (::Type{NT})(t::Tuple) where {K,NT<:NamedTuple{K}}
    nt = NT(t)
    back(Δ::NamedTuple) = (values(NT(Δ)),)
    return nt, back
end

# # https://github.com/FluxML/Zygote.jl/issues/680
# @adjoint function (T::Type{<:Complex})(re, im)
# 	back(Δ::Complex) = (nothing, real(Δ), imag(Δ))
# 	back(Δ::NamedTuple) = (nothing, Δ.re, Δ.im)
# 	T(re, im), back
# end



#### AD Rules for Iterative eigensolves of Helmholtz Operator


"""
Inversion/Conjugate-transposition equalities of Maxwell operator components
----------------------------------------------------------------------------

FFT operators:
--------------
	If 𝓕⁻¹ === `bfft` (symmetric, unnormalized case)

	(1a)	(𝓕)' 	= 	𝓕⁻¹

	(2a)	(𝓕⁻¹)' = 	𝓕

	If 𝓕⁻¹ === `ifft` (asymmetric, normalized case)

	(1a)	(𝓕)' 	= 	𝓕⁻¹ * N		( N := Nx * Ny * Nz )

	(2a)	(𝓕⁻¹)' = 	𝓕	 / N

Combined Curl+Basis-Change operators:
--------------------------------------
	(3)	( [(k⃗+g⃗)×]cₜ )' 	= 	-[(k⃗+g⃗)×]ₜc

	(4)	( [(k⃗+g⃗)×]ₜc )' 	= 	-[(k⃗+g⃗)×]cₜ

Combined Cross+Basis-Change operators:
--------------------------------------
	(3)	( [(ẑ)×]cₜ )' 	= 	[(ẑ)×]ₜc

	(4)	( [(ẑ)×]ₜc )' 	= 	[(ẑ)×]cₜ


--------------
"""

"""
Calculate k̄ contribution from M̄ₖ, where M̄ₖ is backpropagated from ⟨H|M̂ₖ|H⟩

Consider Mₖ as composed of three parts:

	(1) Mₖ	= 	[(k⃗+g⃗)×]cₜ  ⋅  [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]   ⋅  [ẑ×]ₜc
				------------	-----------------	  -------
  			= 		 A				    B                C

where the "cₜ" and "ₜc" labels on the first and third components denote
Cartesian-to-Transverse and Transverse-to-Cartesian coordinate transformations,
respectively.

From Giles, we know that if D = A B C, then

	(2)	Ā 	=	D̄ Cᵀ Bᵀ

	(3)	B̄ 	=	Aᵀ D̄ Cᵀ

	(4) C̄	=	Bᵀ Aᵀ D̄		(to the bone)

We also know that M̄ₖ corresponding to the gradient of ⟨H|M̂ₖ|H⟩ will be

	(5) M̄ₖ	=	|H*⟩⟨H|

where `*` denote complex conjugation.
Equations (2)-(5) give us formulae for the gradients back-propagated to the three
parameterized operators composing Mₖ

	(6) 	[k+g ×]̄		 = 	 |H⟩⟨H|	 ⋅  [[ẑ×]ₜc]ᵀ  ⋅  [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]ᵀ

							= 	|H⟩⟨ ( [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]  ⋅  [ẑ×]ₜc ⋅ H ) |

	(7)	[ 𝓕 nn̂g⁻¹ 𝓕⁻¹ ]̄	   = 	 [[k+g ×]cₜ]ᵀ  ⋅  |H⟩⟨H| ⋅  [[ẑ×]ₜc]ᵀ

						 	= 	-[k+g ×]ₜc  ⋅  |H⟩⟨H| ⋅  [ẑ×]cₜ

							=	-| [k+g ×]ₜc  ⋅ H ⟩⟨ [ẑ×]ₜc ⋅ H |

	(8)  ⇒ 	[ nn̂g⁻¹ ]̄	 	  =   -| 𝓕 ⋅ [k+g ×]ₜc  ⋅ H ⟩⟨ 𝓕 ⋅ [ẑ×]ₜc ⋅ H |

	(9)			[ẑ ×]̄	 	  =   [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]ᵀ ⋅ [[k+g×]cₜ]ᵀ ⋅ |H⟩⟨H|

							= 	-| [ 𝓕  nn̂g⁻¹ 𝓕⁻¹ ]  ⋅  [k+g×]cₜ ⋅ H ⟩⟨H|

where `[ẑ ×]` operators are still parameterized by k⃗ because they involve
m⃗ & n⃗ orthonormal polarization basis vectors, which are determined by k⃗+g⃗
and thus k⃗-dependent.

Our [(k⃗+g⃗)×]cₜ and [ẑ ×]ₜc operators act locally in reciprocal space with the
following local structures

	(10) [ ( k⃗+g⃗[ix,iy,iz] ) × ]ₜc  =		[	-n⃗₁	m⃗₁
											  -n⃗₂	  m⃗₂
											  -n⃗₃	  m⃗₃	]

									=	  [	   m⃗     n⃗  	]  ⋅  [  0   -1
																	1 	 0	]


	(11) [ ( k⃗+g⃗[ix,iy,iz] ) × ]cₜ  =			 [	   n⃗₁	  n⃗₂	  n⃗₃
										  			-m⃗₁   -m⃗₂	  -m⃗₃	  ]

									=	[  0   -1 		⋅	[   m⃗ᵀ
										   1 	0  ]			n⃗ᵀ		]

										=	-(	[ ( k⃗+g⃗[ix,iy,iz] ) × ]ₜc	)ᵀ

	(12) [ ẑ × ]ₜc	 	 =	[	 -m⃗₂	-n⃗₂
								 m⃗₁	n⃗₁
								 0	    0	 ]

						=	[	0 	-1	  0		 		[	m⃗₁	 n⃗₁
								1 	 0	  0	 		⋅		m⃗₂	 n⃗₂
								0 	 0	  0	  ]				m⃗₃	 n⃗₃	]

	(13) [ ẑ × ]cₜ	 	 =	   [	 -m⃗₂	m⃗₁	 	0
									-n⃗₂	n⃗₁		0		]

						=	  [   m⃗ᵀ				[	0	 1	 0
								  n⃗ᵀ	]		⋅		-1	  0	  0
								  						0	 0	 0	]

						=	  (  [ ẑ × ]ₜc  )ᵀ
"""

function update_k_pb(M̂::HelmholtzMap{T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n), mag_m_n_pb = Zygote.pullback(k⃗) do x
		mag_m_n(x,dropgrad(M̂.g⃗))
	end
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
	M̂.m⃗ = m #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = n #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.k⃗ = k⃗
	return (mag, m, n), mag_m_n_pb
end

update_k_pb(M̂::HelmholtzMap{T},kz::T) where T<:Real = update_k_pb(M̂,SVector{3,T}(0.,0.,kz))

# 3D
function ε⁻¹_bar!(eī, d⃗, λ⃗d, Nx, Ny, Nz)
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
	eīf = flat(eī)
	@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
		q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	end
	return eī
end

# 2D
function ε⁻¹_bar!(eī, d⃗, λ⃗d, Nx, Ny)
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
	eīf = flat(eī)
	@avx for iy=1:Ny,ix=1:Nx
		q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
			eīf[a2+1,a2,ix,iy] = eīf[a2,a2+1,ix,iy]
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
		eīf[3,1,ix,iy] = eīf[1,3,ix,iy]
	end
	return eī # inv( (eps' + eps) / 2)

	# eīM = Matrix.(eī)
	# for iy=1:Ny,ix=1:Nx
	# 	q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
	# 	for a=1:3 # loop over diagonal elements: {11, 22, 33}
	# 		eīM[ix,iy][a,a] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
	# 	end
	# 	for a2=1:2 # loop over first off diagonal
	# 		eīM[ix,iy][a2,a2+1] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
	# 	end
	# 	# a = 1, set 1,3 and 3,1, second off-diagonal
	# 	eīM[ix,iy][1,3] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	# end
	# ēM = inv.(eīM)
	# eīMH = inv.( ( ēM .+ ēM' ) ./ 2 )
	# eī .= SMatrix{3,3}.( eīMH  ) # SMatrix{3,3}.(eīM)
	# return eī
end

# 3D
function ε⁻¹_bar(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny, Nz) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = Buffer(Array{Float64,1}([2., 2.]),3,3,Nx,Ny,Nz) # bufferfrom(zero(T),3,3,Nx,Ny,Nz)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny,Nz))
	@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
		q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	end
	# eī = unflat(copy(eīf))
	eī = reinterpret(reshape,SMatrix{3,3,T,9},reshape(copy(eīf),9,Nx,Ny,Nz))
	return eī
end

# 2D
function ε⁻¹_bar(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = Buffer(Array{Float64,1}([2., 2.]),3,3,Nx,Ny) # bufferfrom(zero(T),3,3,Nx,Ny)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny))
	@avx for iy=1:Ny,ix=1:Nx
		q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
			eīf[a2+1,a2,ix,iy] = eīf[a2,a2+1,ix,iy]
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
		eīf[3,1,ix,iy] = eīf[1,3,ix,iy]
	end
	eī = reinterpret(reshape,SMatrix{3,3,T,9},reshape(copy(eīf),9,Nx,Ny))
	return eī # inv( (eps' + eps) / 2)
end

uplot(ch::IterativeSolvers.ConvergenceHistory; kwargs...) = lineplot(log10.(ch.data[:resnorm]); name="log10(resnorm)", kwargs...)

function solve_adj!(ms::ModeSolver,H̄,eigind::Int)
	ms.adj_itr = bicgstabl_iterator!(
		ms.adj_itr.x,	# recycle previous soln as initial guess
		ms.M̂ - real(ms.ω²[eigind])*I, # A
		H̄[:,eigind] - ms.H⃗[:,eigind] * dot(ms.H⃗[:,eigind],H̄[:,eigind]), # b,
		3;	# l = number of GMRES iterations per CG iteration
		Pl = ms.P̂) # left preconditioner
	for (iteration, item) = enumerate(ms.adj_itr) end # iterate until convergence or until (iters > max_iters || mvps > max_mvps)
	copyto!(ms.λ⃗,ms.adj_itr.x) # copy soln. to ms.λ⃗ where other contributions/corrections can be accumulated
	# λ₀, ch = bicgstabl(
	# 	ms.adj_itr.x,	# recycle previous soln as initial guess
	# 	ms.M̂ - real(ms.ω²[eigind])*I, # A
	# 	H̄[:,eigind] - ms.H⃗[:,eigind] * dot(ms.H⃗[:,eigind],H̄[:,eigind]), # b,
	# 	3;	# l = number of GMRES iterations per CG iteration
	# 	Pl = ms.P̂, # left preconditioner
	# 	reltol = 1e-10,
	# 	log=true,
	# 	)
	# copyto!(ms.λ⃗,λ₀) # copy soln. to ms.λ⃗ where other contributions/corrections can be accumulated
	# println("\t\tAdjoint Problem for kz = $( ms.M̂.k⃗[3] ) ###########")
	# println("\t\t\tadj converged?: $ch")
	# println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")
	# uplot(ch;name="log10( adj. prob. res. )")
	return ms.λ⃗
end

function solve_adj!(λ⃗,M̂::HelmholtzMap,H̄,ω²,H⃗,eigind::Int;log=false)
	res = bicgstabl(
		# ms.adj_itr.x,	# recycle previous soln as initial guess
		M̂ - real(ω²[eigind])*I, # A
		H̄[:,eigind] - H⃗[:,eigind] * dot(H⃗[:,eigind],H̄[:,eigind]), # b,
		2;	# l = number of GMRES iterations per CG iteration
		# Pl = HelmholtzPreconditioner(M̂), # left preconditioner
		log,
		)
	if log
		copyto!(λ⃗,res[1])
		ch = res[2]
	else
		copyto!(λ⃗,res)
	end
	# println("#########  Adjoint Problem for kz = $( ms.M̂.k⃗[3] ) ###########")
	# uplot(ch;name="log10( adj. prob. res. )")
	# println("\t\t\tadj converged?: $ch")
	# println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")
	return λ⃗
end


using LinearMaps: ⊗
export eig_adjt, linsolve

function linsolve(Â, b⃗; x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	# x⃗ = isnothing(x⃗₀) ? randn(eltype(b⃗),first(size(b⃗))) : copy(x⃗₀)
	# return bicgstabl!(x⃗, Â, b⃗, 3; Pl=P̂, max_mv_products=5000)
	bicgstabl(Â, b⃗, 2; Pl=P̂, max_mv_products=3000)
end

function ChainRulesCore.rrule(::typeof(linsolve), Â, b⃗;
		x⃗₀=nothing, P̂=IterativeSolvers.Identity())
	x⃗ = linsolve(Â, b⃗; x⃗₀, P̂)
	function linsolve_pullback(x̄)
		λ⃗ = linsolve(Â', vec(x̄))
		Ā = (-λ⃗) ⊗ x⃗'
		return (NO_FIELDS, Ā, λ⃗)
	end
	return (x⃗, linsolve_pullback)
end

"""
	eig_adjt(A, α, x⃗, ᾱ, x̄; λ⃗₀, P̂)

Compute the adjoint vector `λ⃗` for a single eigenvalue/eigenvector pair (`α`,`x⃗`) of `Â` and
sensitivities (`ᾱ`,`x̄`). It is assumed (but not checked) that ``Â ⋅ x⃗ = α x⃗``. `λ⃗` is the
sum of two components,

	``λ⃗ = λ⃗ₐ + λ⃗ₓ``

where ``λ⃗ₐ = ᾱ x⃗`` and ``λ⃗ₓ`` correspond to `ᾱ` and `x̄`, respectively. When `x̄` is non-zero
``λ⃗ₓ`` is computed by iteratively solving

	``(Â - αÎ) ⋅ λ⃗ₓ = x̄ - (x⃗ ⋅ x̄)``

An inital guess can be supplied for `λ⃗ₓ` via the keyword argument `λ⃗₀`, otherwise a random
vector is used. A preconditioner `P̂` can also be supplied to improve convergeance.
"""
function eig_adjt(Â, α, x⃗, ᾱ, x̄; λ⃗₀=nothing, P̂=IterativeSolvers.Identity())
	if iszero(x̄)
		λ⃗ = iszero(ᾱ)	? zero(x⃗) : ᾱ * x⃗
 	else
		λ⃗ₓ₀ = linsolve(
			Â - α*I,
		 	x̄ - x⃗ * dot(x⃗,x̄);
			P̂,
			x⃗₀=λ⃗₀,
		)
		# λ⃗ -= x⃗ * dot(x⃗,λ⃗)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		# λ⃗ += ᾱ * x⃗
		λ⃗ₓ = λ⃗ₓ₀  - x⃗ * dot(x⃗,λ⃗ₓ₀)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		λ⃗ = λ⃗ₓ + ᾱ * x⃗
	end

	return λ⃗
end

function ChainRulesCore.rrule(::typeof(eig_adjt), Â, α, x⃗, ᾱ, x̄; λ⃗₀=nothing, P̂=IterativeSolvers.Identity())
	# primal
	if iszero(x̄)
		λ⃗ = iszero(ᾱ)	? zero(x⃗) : ᾱ * x⃗
 	else
		λ⃗ₓ = linsolve(
			Â - α*I,
		 	x̄ - x⃗ * dot(x⃗,x̄);
			P̂,
			x⃗₀=λ⃗₀,
		)
		# λ⃗ₓ = λ⃗ₓ₀ - x⃗ * dot(x⃗,λ⃗ₓ₀)	# re-orthogonalize λ⃗ w.r.t. x⃗, correcting for round-off err.
		λ⃗ = λ⃗ₓ + ᾱ * x⃗
	end

	# define pullback
	function eig_adjt_pullback(lm̄)
		if lm̄ isa AbstractZero
			return (NO_FIELDS, Zero(), Zero(), Zero(), Zero(), Zero())
		else
			if iszero(x̄)
				if iszero(ᾱ)
					return (NO_FIELDS, Zero(), Zero(), Zero(), Zero(), Zero())
				else
					x⃗_bar = ᾱ * lm̄
					ᾱ_bar = dot(lm̄,x⃗)
					return (NO_FIELDS, Zero(), Zero(), x⃗_bar, ᾱ_bar, Zero())
				end
		 	else	# λ⃗ₓ exists, must solve 2ⁿᵈ-order adjoint problem
				ξ⃗ = linsolve(
					Â - α*I,
				 	lm̄ - x⃗ * dot(x⃗,lm̄);
					P̂,
				)
				println("")
				println("ξ⃗ err: $(sum(abs2,(Â - α*I)*ξ⃗-(lm̄ - x⃗ * dot(x⃗,lm̄))))")
				A_bar = -ξ⃗ ⊗ λ⃗ₓ'
				α_bar = dot(ξ⃗,λ⃗ₓ)
				x⃗_bar = dot(x̄,x⃗) * ξ⃗ + dot(ξ⃗,x⃗) * x̄  + ᾱ * lm̄
				ᾱ_bar = dot(lm̄,x⃗)
				x̄_bar = ξ⃗ - (x⃗ * dot(x⃗,ξ⃗))
			end
			return (NO_FIELDS, A_bar, α_bar, x⃗_bar, ᾱ_bar, x̄_bar)
		end
	end

	# return primal and pullback
	return (λ⃗, eig_adjt_pullback)
end

function ∇solve_ω²(ΔΩ,Ω,k,ε⁻¹,grid)
	@show ω̄sq, H̄ = ΔΩ
	@show ω², H⃗ = Ω
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	H = reshape(H⃗[:,eigind],(2,Ns...))
	g⃗s = g⃗(dropgrad(grid))
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
	λd = similar(M̂.d)
	λẽ = similar(M̂.d)
	ẽ = similar(M̂.d)
	ε⁻¹_bar = similar(ε⁻¹)
	if typeof(ω̄sq)==ChainRulesCore.Zero
		ω̄sq = 0.
	end
	if typeof(H̄) != ChainRulesCore.Zero
		λ⃗ = solve_adj!(M̂,H̄,ω²,H⃗,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
		λ⃗ -= (ω̄sq + dot(H⃗[:,eigind],λ⃗)) * H⃗[:,eigind]
	else
		λ⃗ = -ω̄sq * H⃗[:,eigind]
	end
	λ = reshape(λ⃗,(2,Ns...))
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(ε⁻¹_bar, vec(M̂.d), vec(λd), Ns...)
	# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
	λd *=  M̂.Ninv
	λẽ .= reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  , M̂ ) )
	ẽ .= reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d, M̂) )
	kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
	kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
	māg .= dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
	k̄ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	return (NO_FIELDS, ChainRulesCore.Zero(), k̄ , ε⁻¹_bar)
end

function ChainRulesCore.rrule(::typeof(solve_ω²), k::Union{T,SVector{3,T}},shapes::Vector{<:Shape},grid::Grid{ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	println("using new rrule")
	ms = @ignore(ModeSolver(k, shapes, grid)) # ; nev, eigind, maxiter, tol, log))
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	ω²H⃗ = solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
    solve_ω²_pullback(ΔΩ) = ∇solve_ω²(ΔΩ,ω²H⃗,k,ε⁻¹,grid)
    return (ω²H⃗, solve_ω²_pullback)
end

function ChainRulesCore.rrule(::typeof(solve_ω²), ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ω²,H⃗ = solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
		mag_m_n(x,dropgrad(ms.M̂.g⃗))
	end
    function solve_ω²_pullback(ΔΩ)
		ω̄sq, H̄ = ΔΩ
		Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
		Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
		H = reshape(H⃗[:,eigind],(2,Ns...))
		# mn2 = vcat(reshape(ms.M̂.m,(1,3,Ns...)),reshape(ms.M̂.n,(1,3,Ns...)))
		if typeof(ω̄sq)==ChainRulesCore.Zero
			ω̄sq = 0.
		end
		if typeof(H̄) != ChainRulesCore.Zero
			solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
			ms.λ⃗ -= (ω̄sq[eigind] + dot(H⃗[:,eigind],ms.λ⃗)) * H⃗[:,eigind]
		else
			ms.λ⃗ = -ω̄sq[eigind] * H⃗[:,eigind]
		end
		λ = reshape(ms.λ⃗,(2,Ns...))
		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd), Ns...)
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		ms.λd *=  ms.M̂.Ninv
		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
		ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
		ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
		ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
		k̄ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
		# if !(typeof(k)<:SVector)
		# 	k̄_kx = k̄_kx[3]
		# end
		return (NO_FIELDS, ChainRulesCore.Zero(), k̄ , ms.ε⁻¹_bar)
    end
    return ((ω², H⃗), solve_ω²_pullback)
end


"""
function mapping |H⟩ ⤇ ( (∂M/∂k)ᵀ + ∂M/∂k )|H⟩
"""
function Mₖᵀ_plus_Mₖ(H⃗::AbstractVector{Complex{T}},k,ε⁻¹,grid) where T<:Real
	Ns = size(grid)
	# g⃗s = g⃗(grid)
	# mag,m⃗,n⃗ = mag_m_n(k,g⃗s)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	H = reshape(H⃗,(2,Ns...))
	mn = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	X = -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(flat(ε⁻¹))), (2:3) ), mn, mag )
	Y = zx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(flat(ε⁻¹))), (2:3)), mn )
	vec(X + Y)
end

"""
solve the adjoint sensitivity problem corresponding to ∂ω²∂k = <H|∂M/∂k|H>
"""
function ∂ω²∂k_adj(M̂::HelmholtzMap,ω²,H⃗,H̄;eigind=1,log=false)
	res = bicgstabl(
		M̂ - real(ω²[eigind])*I, # A
		H̄ - H⃗[:,eigind] * dot(H⃗[:,eigind],H̄), # b,
		3;	# l = number of GMRES iterations per CG iteration
		# Pl = HelmholtzPreconditioner(M̂), # left preconditioner
		log,
		)
end

"""
(māg,m̄,n̄) → k̄ map
"""
function ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	return +( k̄_mag, k̄_m, k̄_n )
end

function ∇ₖmag_m_n(ΔΩ,Ω;dk̂=SVector(0.,0.,1.))
	māg,m̄,n̄ = ΔΩ
	mag,m⃗,n⃗ = Ω
	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	return +( k̄_mag, k̄_m, k̄_n )
end

function ChainRulesCore.rrule(::typeof(mag_m_n),k⃗::SVector{3,T},g⃗::AbstractArray{SVector{3,T}}) where T <: Real
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	n_buf = Buffer(g⃗,size(g⃗))
	m_buf = Buffer(g⃗,size(g⃗))
	kpg_buf = Buffer(g⃗,size(g⃗))
	mag_buf = Buffer(zeros(T,size(g⃗)),size(g⃗))
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg_buf[i] = k⃗ - g⃗[i]
		@inbounds mag_buf[i] = norm(kpg_buf[i])
		@inbounds n_buf[i] =   ( ( abs2(kpg_buf[i][1]) + abs2(kpg_buf[i][2]) ) > 0. ) ?  normalize( cross( ẑ, kpg_buf[i] ) ) : ŷ
		@inbounds m_buf[i] =  normalize( cross( n_buf[i], kpg_buf[i] )  )
	end
	mag_m⃗_n⃗ = (copy(mag_buf), copy(m_buf), copy(n_buf))
	kp⃗g = copy(kpg_buf)
	mag_m_n_pullback(ΔΩ) = let Ω=mag_m⃗_n⃗, kp⃗g=kp⃗g, dk̂=normalize(k⃗)
		māg,m̄,n̄ = ΔΩ
		mag,m⃗,n⃗ = Ω
		ê_over_mag = cross.( kp⃗g, (dk̂,) ) ./ mag.^2
		k̄ = sum( māg .* dot.( kp⃗g, (dk̂,) ) ./ mag )
		k̄ -= sum( dot.( m̄ , cross.(m⃗, ê_over_mag ) ) )
		k̄ -= sum( dot.( n̄ , cross.(n⃗, ê_over_mag ) ) )
		return ( NO_FIELDS, k̄*dk̂, ChainRulesCore.Zero() )
	end
    return (mag_m⃗_n⃗ , mag_m_n_pullback)
end

"""
pull back sensitivity w.r.t. ∂ω²∂k = 2⟨H|∂M/∂k|H⟩ to corresponding
k̄ (scalar) and nn̄g⁻¹ (tensor field) sensitivities
"""
function ∇HMₖH(k,H⃗::AbstractArray{Complex{T}},nng⁻¹,grid;eigind=1) where T<:Real
	# Setup
	local 	zxtc_to_mn = SMatrix{3,3}(	[	0 	-1	  0
											1 	 0	  0
											0 	 0	  0	  ]	)

	local 	kxtc_to_mn = SMatrix{2,2}(	[	0 	-1
											1 	 0	  ]	)

	g⃗s, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.@ignore begin
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		g⃗s = g⃗(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		return (g⃗s,Ninv,Ns,𝓕,𝓕⁻¹)
	end
	mag, m⃗, n⃗  = mag_m_n(k,g⃗s)
	H = reshape(H⃗[:,eigind],(2,Ns...))
	Hsv = reinterpret(reshape, SVector{2,Complex{T}}, H )

	#TODO: Banish this quadruply re(shaped,interpreted) m,n,mns format back to hell
	# mns = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))

	### calculate k̄ contribution from M̄ₖ ( from ⟨H|M̂ₖ|H⟩ )
	Ā₁		=	conj.(Hsv)
	Ā₂ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * zx_tc(H * Ninv,mns) , real(flat(nng⁻¹))),
		)
	Ā 	= 	Ā₁  .*  transpose.( Ā₂ )
	m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
	m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
	n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
	māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)

	# B̄₁ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{T}},
	# 	# 𝓕  *  kxtcsp	 *	vec(H),
	# 	𝓕 * kx_tc( conj.(H) ,mns,mag),
	# 	)
	# B̄₂ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{T}},
	# 	# 𝓕  *  zxtcsp	 *	vec(H),
	# 	𝓕 * zx_tc( H * Ninv ,mns),
	# 	)
	# B̄ 	= 	real.( Hermitian.(  B̄₁  .*  transpose.( B̄₂ )  ) )

	B̄₁ = 𝓕 * kx_tc( conj.(H) ,mns,mag)
	B̄₂ = 𝓕 * zx_tc( H * Ninv ,mns)
	@tullio B̄[a,b,i,j] := B̄₁[a,i,j] * B̄₂[b,i,j] + B̄₁[b,i,j] * B̄₂[a,i,j]   #/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2

	C̄₁ = reinterpret(
		reshape,
		SVector{3,Complex{T}},
		# reshape(
		# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
		# 	(3,size(gr)...),
		# 	),
		𝓕⁻¹ * ε⁻¹_dot(  𝓕 * -kx_tc( H * Ninv, mns, mag) , real(flat(nng⁻¹))),
		)
	C̄₂ =   conj.(Hsv)
	C̄ 	= 	C̄₁  .*  transpose.( C̄₂ )
	m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
	m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
	n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )

	# Accumulate gradients and pull back
	nngī 	=  real(B̄)/2 #( B̄ .+ transpose.(B̄) ) ./ 2
	k̄	 	= ∇ₖmag_m_n(
						māg_Ā, 				# māg total
						m̄_Ā.*mag .+ m̄_C̄, 	  # m̄  total
						n̄_Ā.*mag .+ n̄_C̄,	  # n̄  total
						mag,
						m⃗,
						n⃗,
					)
	# H̄ = Mₖᵀ_plus_Mₖ(H⃗[:,eigind],k,nng⁻¹,grid)
	# # X = -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mns), (2:3) ), real(flat(ε⁻¹))), (2:3) ), mns, mag )
	# # Y = zx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mns,mag), (2:3) ), real(flat(ε⁻¹))), (2:3)), mns )
	nngif = real(flat(nng⁻¹))
	X = -kx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_tc(H,mns)		, nngif), mns, mag )
	Y =  zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nngif), mns )
	H̄ = vec(X + Y) * Ninv
	return k̄, H̄, nngī
	# return k̄, H̄, reinterpret(SMatrix{3,3,Float64,9},reshape( nngī ,9*128,128))
end

# nng = inv.(nnginv)
# ε = inv.(ε⁻¹)
# ∂ε∂ω_man = (2/ω) * (nng .- ε)
# ∂ei∂ω_man = copy(flat(-(ε⁻¹.^2) .* ∂ε∂ω_man ))
# ∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * ε⁻¹.^2 .* (  inv.(nng⁻¹) .- inv.(ε⁻¹) ) #(2.0/ω) * ε⁻¹ .* (  ε⁻¹ .* inv.(nng⁻¹) - I )
∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * (  ε⁻¹.^2 .* inv.(nng⁻¹) .- ε⁻¹ )
# function ∂nng∂ω_man_LN(om)
# 	 ng = ng_MgO_LiNbO₃(inv(om))[1,1]
# 	 n = sqrt(ε_MgO_LiNbO₃(inv(om))[1,1])
# 	 gvd = gvd_MgO_LiNbO₃(inv(om))[1,1]  #/ (2π)
# 	 # om = 1/om
# 	 om*(ng^2 - n*ng) + n * gvd
# end
∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω) = -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd) # (1.0/ω) * (nng⁻¹ .- ε⁻¹ ) .- (  ngvd .* (nng⁻¹).^2  )

"""
solve the adjoint sensitivity problem corresponding to ∂ω²∂k = <H|∂M/∂k|H>
"""
# function ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,H⃗,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
function ∂²ω²∂k²(ω,geom,k,H⃗,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	# nng⁻¹, nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
	# ε⁻¹, epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)
	ngvd = ngvdₛ(ω,geom,grid)
	nng⁻¹, nnginv_pb = Zygote.pullback(x->nngₛ⁻¹(x,geom,grid),ω)
	ε⁻¹, epsi_pb = Zygote.pullback(x->εₛ⁻¹(x,geom,grid),ω)
	# nng⁻¹, nnginv_pb = Zygote._pullback(Zygote.Context(),x->nngₛ⁻¹(x,dropgrad(geom),dropgrad(grid)),ω)
	# ε⁻¹, epsi_pb = Zygote._pullback(Zygote.Context(),x->εₛ⁻¹(x,dropgrad(geom),dropgrad(grid)),ω)
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	∂ω²∂k_nd = 2 * H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,H⃗[:,eigind]),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)
	nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī2)) ) ./ 2
	eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
	om̄₂ = nnginv_pb(nngī_herm)[1] #nngī2)
	om̄₃ = epsi_pb(eī_herm)[1] #eī₁)
	println("")
	println("om̄₁: $(om̄₁)")
	println("om̄₂: $(om̄₂)")
	println("om̄₃: $(om̄₃)")

	# om̄₂2 = sum(flat(nngī_herm	.* 	∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω)))
	# om̄₃2 = sum(flat(eī_herm	.* 	∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω)))
	om̄₂2 = dot(vec(flat(nngī_herm)), 	vec(flat(∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω))))
	om̄₃2 = dot(vec(flat(eī_herm)), 	vec(flat(∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω))))
	println("om̄₂2: $(om̄₂2)")
	println("om̄₃2: $(om̄₃2)")
	println("om̄₁ + om̄₂2 + om̄₃2: $(om̄₁ + om̄₂2 + om̄₃2)")

	# om̄₂4 = dot(inv.(vec(flat(nngī_herm))), 	vec(flat(∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω))))
	# om̄₃4 = dot(inv.(vec(flat(eī_herm))), 	vec(flat(∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω))))
	# println("om̄₂4: $(om̄₂4)")
	# println("om̄₃4: $(om̄₃4)")

	om̄ = om̄₁ + om̄₂ + om̄₃
	println("om̄: $(om̄)")
	return om̄
end

function ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,H⃗,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	ω² = ω^2
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	∂ω²∂k_nd = 2 * H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,H⃗[:,eigind]),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)
	nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	nngī_herm = (nngī2 .+ adjoint.(nngī2) ) / 2
	eī_herm = (eī₁ .+ adjoint.(eī₁) ) / 2
	return om̄₁, eī_herm, nngī_herm
end

function ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid::Grid{ND,T}) where {ND,T<:Real}
	Nranges, Ninv, Ns, 𝓕, 𝓕⁻¹ = Zygote.@ignore begin
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		# g⃗s = g⃗(grid)
		Nranges		=		eachindex(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
		return (Nranges,Ninv,Ns,𝓕,𝓕⁻¹)
	end
	mag, m⃗, n⃗  = mag_m_n(k,grid)
	# mns = vcat(reshape(flat(m⃗),(1,3,Ns...)),reshape(flat(n⃗),(1,3,Ns...)))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
	H = reshape(H⃗,(2,Ns...))
	λ = reshape(λ⃗,(2,Ns...))
	d 	= 	𝓕 * kx_tc( H , mns, mag ) * Ninv
	λd 	= 	𝓕 * kx_tc( λ , mns, mag )
	eī	 = 	 ε⁻¹_bar(vec(d), vec(λd), Ns...)
	eif = flat(ε⁻¹)
	# eif = reshape(reinterpret(reshape,T,ε⁻¹),3,3,Ns...) #flat(ε⁻¹)
	# eif = reshape(reinterpret(T,ε⁻¹),3,3,Ns...)
	λẽ  =   𝓕⁻¹ * ε⁻¹_dot(λd * Ninv, real(eif)) #flat(ε⁻¹)) # _d2ẽ!(λẽ , λd  ,M̂ )
	ẽ 	 =   𝓕⁻¹ * ε⁻¹_dot(d        , real(eif)) #flat(ε⁻¹)) # _d2ẽ!(M̂.e,M̂.d,M̂)
	λẽ_sv  = reinterpret(reshape, SVector{3,Complex{T}}, λẽ )
	ẽ_sv 	= reinterpret(reshape, SVector{3,Complex{T}}, ẽ )
	m̄_kx = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ_sv .* conj.(view(λ,2,Nranges...)) )	#NB: m̄_kx and n̄_kx would actually
	n̄_kx =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ_sv .* conj.(view(λ,1,Nranges...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
	māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
	k̄		= ∇ₖmag_m_n(
				māg_kx, 		# māg total
				m̄_kx.*mag, 	# m̄  total
				n̄_kx.*mag,	  	# n̄  total
				mag, m⃗, n⃗,
			)
	return k̄, eī
end

function ChainRulesCore.rrule(::typeof(HelmholtzMap), kz::T, ε⁻¹, grid::Grid; shift=0.) where {T<:Real}
	function HelmholtzMap_pullback(M̄)
		if M̄ isa AbstractZero
			k̄	= Zero()
			eī = Zero()
		else
			λ⃗ = -M̄.maps[1].lmap
			H⃗ = M̄.maps[2].lmap'
			k̄, eī = ∇M̂(kz,ε⁻¹,λ⃗,H⃗,grid)
		end

		return (NO_FIELDS, k̄, eī, Zero())
	end
	return HelmholtzMap(kz, ε⁻¹, grid; shift), HelmholtzMap_pullback
end


# function ChainRulesCore.rrule(T::Type{<:LinearMaps.LinearCombination{Complex{T1}}},As::Tuple{<:HelmholtzMap,<:LinearMaps.UniformScalingMap}) where T1<:Real
# 	function LinComb_Helmholtz_USM_pullback(M̄)
# 		# return (NO_FIELDS, (M̄, M̄))
# 		return (NO_FIELDS, Composite{Tuple{LinearMap,LinearMap}}(M̄, M̄))
# 	end
# 	return LinearMaps.LinearCombination{Complex{T1}}(As), LinComb_Helmholtz_USM_pullback
# end
#
# function ChainRulesCore.rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,N::Int) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/N, M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NO_FIELDS, ᾱ, Zero())
# 	end
# 	return LinearMaps.UniformScalingMap(α,N), USM_pullback
# end
#
# function ChainRulesCore.rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,N::Int,N2::Int) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/N, M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NO_FIELDS, ᾱ, Zero(), Zero())
# 	end
# 	return LinearMaps.UniformScalingMap(α,N,N2), USM_pullback
# end
#
# function ChainRulesCore.rrule(T::Type{<:LinearMaps.UniformScalingMap},α::T1,Ns::Tuple{<:Int,<:Int}) where T1
# 	function USM_pullback(M̄)
# 		# ᾱ = dot(M̄.maps[1].lmap/first(Ns), M̄.maps[2].lmap')
# 		ᾱ = mean( M̄.maps[1].lmap .* M̄.maps[2].lmap' )
# 		return (NO_FIELDS, ᾱ, DoesNotExist())
# 	end
# 	return LinearMaps.UniformScalingMap(α,Ns), USM_pullback
# end

function ∇solve_k(ΔΩ, Ω, ∂ω²∂k, ω, ε⁻¹, grid::Grid{ND,T}; eigind=1, λ⃗₀=nothing) where {ND,T<:Real}
	k̄ₖ, H̄ = ΔΩ
	k, H⃗ = Ω
	Ninv, Ns, 𝓕 = Zygote.@ignore begin
		Ninv 		= 		1. / N(grid)
		Ns			=		size(grid)
		d0 = randn(Complex{T}, (3,Ns...))
		𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
		return (Ninv,Ns,𝓕)
	end
	M̂ = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	g⃗s = g⃗(dropgrad(grid))
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	# mn = vcat(reshape(flat(m⃗),(1,3,Ns...)),reshape(flat(n⃗),(1,3,Ns...)))
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)))
	mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
	if !iszero(H̄)
		# solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
		λ⃗	= eig_adjt(
				M̂,								 # Â
				ω^2, 							# α
				H⃗[:,eigind], 					 # x⃗
				0.0, 							# ᾱ
				H̄;								 # x̄
				λ⃗₀,
				P̂	= HelmholtzPreconditioner(M̂),
			)
		k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗[:,eigind],grid)
	else
		eīₕ 	= zero(ε⁻¹) #fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
		k̄ₕ 	= 0.0
	end
	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	# println("")
	# println("k̄ₖ = $(k̄ₖ)")
	# println("k̄ₕ = $(k̄ₕ)")
	# println("k̄ₖ + k̄ₕ = $(k̄ₖ+k̄ₕ)")
	λ⃗ₖ	 = ( (k̄ₖ + k̄ₕ ) / ∂ω²∂k[eigind] ) * H⃗[:,eigind]
	H 	= reshape(H⃗[:,eigind],(2,Ns...))
	λₖ  = reshape(λ⃗ₖ, (2,Ns...))
	d	= 	𝓕 * kx_tc( H  , mns, mag ) * Ninv
	λdₖ	=	𝓕 * kx_tc( λₖ , mns, mag )
	eīₖ = ε⁻¹_bar(vec(d), vec(λdₖ), Ns...)
	ω̄  =  2ω * (k̄ₖ + k̄ₕ ) / ∂ω²∂k[eigind]
	# println("ω = $ω")
	# println("ωbar = $(ω̄ )")
	# println("∂ω²∂k[eigind]: $(∂ω²∂k[eigind])")
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
	return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , eīₖ + eīₕ)
end

function ∇solve_k!(ΔΩ, Ω, ∂ω²∂k, ω::T, ε⁻¹, grid; eigind=1) where T<:Real
	k̄, H̄ = ΔΩ
	# println("k̄ = $(k̄)")
	k, H⃗ = Ω
	M̂ = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	H = reshape(H⃗[:,eigind],(2,Ns...))
	g⃗s = g⃗(dropgrad(grid))
	mag = M̂.mag
	n⃗ 	 = M̂.n⃗
	m⃗ 	= M̂.m⃗
	λd = similar(M̂.d)
	λẽ = similar(M̂.d)
	ẽ = similar(M̂.d)
	eīₕ = similar(ε⁻¹)
	eīₖ = similar(ε⁻¹)
	λ⃗ = similar(H⃗[:,eigind])
	λ = reshape(λ⃗,(2,Ns...))
	if typeof(k̄)==ChainRulesCore.Zero
		k̄ = 0.
	end
	if typeof(H̄) != ChainRulesCore.Zero
		solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
		λ⃗ -= dot(H⃗[:,eigind],λ⃗) * H⃗[:,eigind]
		d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
		λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
		ε⁻¹_bar!(eīₕ, vec(M̂.d), vec(λd), Ns...)
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λd *=  M̂.Ninv
		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,M̂ ) )
		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d,M̂) )
		m̄_kx = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )	#NB: m̄_kx and n̄_kx would actually
		n̄_kx =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )	# be these quantities mulitplied by mag, I do that later because māg is calc'd with m̄/mag & n̄/mag
		māg_kx = dot.(n⃗, n̄_kx) + dot.(m⃗, m̄_kx)
		k̄ₕ	= -∇ₖmag_m_n(māg_kx, 		# māg total
						m̄_kx.*mag, 	# m̄  total
						n̄_kx.*mag,	  	# n̄  total
						mag, m⃗, n⃗ )
	else
		eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
		k̄ₕ = 0.0
	end
	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k[eigind] ) * H⃗[:,eigind] )
	λ = reshape(λ⃗,(2,Ns...))
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(eīₖ, vec(M̂.d), vec(λd),Ns...)
	ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k[eigind]
	# println("k̄ₕ = $(k̄ₕ)")
	# println("k̄ + k̄ₕ = $(k̄+k̄ₕ)")
	# println("ω = $ω")
	# println("ωbar = $(ω̄ )")
	# println("∂ω²∂k[eigind]: $(∂ω²∂k[eigind])")
	# eī = eīₖ + eīₕ
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
	return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , eīₖ + eīₕ)
end

function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	kH⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
    solve_k_pullback(ΔΩ) = let kH⃗=kH⃗, ∂ω²∂k=ms.∂ω²∂k, ω=ω, ε⁻¹=ε⁻¹, grid=ms.grid, eigind=eigind
		∇solve_k!(ΔΩ,kH⃗,∂ω²∂k,ω,ε⁻¹,grid;eigind)
	end
    return (kH⃗, solve_k_pullback)
end

# 	println("#########  ∂ω²/∂k Adjoint Problem for kz = $( M̂.k⃗[3] ) ###########")
# 	uplot(ch;name="log10( adj. prob. res. )")
# 	println("\t\t\tadj converged?: $ch")
# 	println("\t\t\titrs, mvps: $(ch.iters), $(ch.mvps)")




# function ChainRulesCore.rrule(::typeof(solve_k), ω::T,geom::Vector{<:Shape},gr::Grid{ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
#
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)
#
# 	Srvol,proc_sinds,mat_inds = @ignore begin
# 		xyz = x⃗(gr)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 		xyzc = x⃗c(gr)
# 		ms = ModeSolver(kguess(ω,geom), geom, gr))
# 		corner_sinds!(ms.corner_sinds,geom,xyz,xyzc))
# 		proc_sinds!(ms.sinds_proc,ms.corner_sinds))
# 		Srvol(x) = let psinds=ms.sinds_proc, xyz=xyz, vxlmin=vxl_min(xyzc), vxlmax=vxl_max(xyzc)
# 			S_rvol(sinds_proc,xyz,vxlmin,vxlmax,x)
# 		end
# 		eism(om,x) =
# 		(Srvol, ms.sinds_proc)
# 	end
# 	# Srvol = S_rvol(proc_sinds,xyz,vxl_min(xyzc),vxl_max(xyzc),shapes)
# 	ε⁻¹ = εₛ⁻¹(ω,geom;ms=dropgrad(ms))
# 	kH⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
#     solve_k_pullback(ΔΩ) = let kH⃗=kH⃗, ∂ω²∂k=ms.∂ω²∂k, ω=ω, ε⁻¹=ε⁻¹, grid=ms.grid, eigind=eigind
# 		∇solve_k(ΔΩ,kH⃗,∂ω²∂k,ω,ε⁻¹,grid;eigind)
# 	end
#     return (kH⃗, solve_k_pullback)
# end


# function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	k, H⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
# 	# k, H⃗ = copy.(solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)) # ,ω²_tol)	 # returned data are refs to fields in ms struct. copy to preserve result for (possibly delayed) pullback closure.
# 	g⃗ = copy(ms.M̂.g⃗)
# 	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
# 		mag_m_n(x,dropgrad(g⃗))
# 	end
# 	∂ω²∂k = copy(ms.∂ω²∂k[eigind])
# 	Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 	Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
# 	# println("\tsolve_k:")
# 	# println("\t\tω² (target): $(ω^2)")
# 	# println("\t\tω² (soln): $(ms.ω²[eigind])")
# 	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
# 	# println("\t\tk: $k")
# 	# println("\t\t∂ω²∂k: $∂ω²∂k")
# 	omsq_soln = ms.ω²[eigind]
# 	ε⁻¹_copy = copy(ε⁻¹)
# 	k_copy = copy(k)
# 	H⃗ = copy(H⃗)
#     function solve_k_pullback(ΔΩ)
# 		k̄, H̄ = ΔΩ
# 		# println("\tsolve_k_pullback:")
# 		# println("k̄ (bar): $k̄")
# 		update_k!(ms,k_copy)
# 		update_ε⁻¹(ms,ε⁻¹_copy) #ε⁻¹)
# 		ms.ω²[eigind] = omsq_soln # ω^2
# 		ms.∂ω²∂k[eigind] = ∂ω²∂k
# 		copyto!(ms.H⃗, H⃗)
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
# 		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		# Nranges = eachindex(ms.grid)
#
# 		H = reshape(H⃗,(2,Ns...))
# 	    if typeof(k̄)==ChainRulesCore.Zero
# 			k̄ = 0.
# 		end
# 		if typeof(H̄) != ChainRulesCore.Zero
# 			solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
# 			# solve_adj!(ms,H̄,ω^2,H⃗,eigind)
# 			ms.λ⃗ -= dot(H⃗[:,eigind],ms.λ⃗) * H⃗[:,eigind]
# 			λ = reshape(ms.λ⃗,(2,Ns...))
# 			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 			λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 			ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd), Ns...)
# 			eīₕ = copy(ms.ε⁻¹_bar)
# 			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 			ms.λd *=  ms.M̂.Ninv
# 			λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
# 			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 			ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 			ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 			ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
# 			k̄ₕ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		else
# 			eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
# 			k̄ₕ = 0.0
# 		end
# 		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 		copyto!(ms.λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * H⃗[:,eigind] )
# 		λ = reshape(ms.λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd),Ns...)
# 		eīₖ = copy(ms.ε⁻¹_bar)
# 		ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
# 		ε⁻¹_bar = eīₖ + eīₕ
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , ε⁻¹_bar)
#     end
#     return ((k, H⃗), solve_k_pullback)
# end


# function ChainRulesCore.rrule(::typeof(solve_n), ms::ModeSolver{ND,T},ω::T,geom::Vector{<:Shape};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	ε⁻¹, ei_pb = εₛ⁻¹(ω,geom;ms) # make_εₛ⁻¹(ω,shapes,dropgrad(ms))
# 	nnginv, nng_pb = nngₛ⁻¹(ω,geom;ms)
# 	(k,H⃗), solk_pb = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
# 	g⃗ = copy(ms.M̂.g⃗)
# 	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
# 		mag_m_n(x,dropgrad(g⃗))
# 	end
#
# 	ng, ng_pb = Zygote.pullback(ω) do ω, H⃗, nnginv, mag, m⃗, n⃗
# 		ω / H_Mₖ_H(H⃗[:,eigind],nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
# 	end
# 	∂ω²∂k = 2ω * inv(ng)
#
# 	Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 	Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
# 	# println("\tsolve_k:")
# 	# println("\t\tω² (target): $(ω^2)")
# 	# println("\t\tω² (soln): $(ms.ω²[eigind])")
# 	# println("\t\tΔω² (soln): $(real(ω^2 - ms.ω²[eigind]))")
# 	# println("\t\tk: $k")
# 	# println("\t\t∂ω²∂k: $∂ω²∂k")
# 	omsq_soln = ms.ω²[eigind]
# 	ε⁻¹_copy = copy(ε⁻¹)
# 	k_copy = copy(k)
# 	H⃗ = copy(H⃗)
#     function solve_k_pullback(ΔΩ)
# 		k̄, H̄ = ΔΩ
# 		# println("\tsolve_k_pullback:")
# 		# println("k̄ (bar): $k̄")
# 		update_k!(ms,k_copy)
# 		update_ε⁻¹(ms,ε⁻¹_copy) #ε⁻¹)
# 		ms.ω²[eigind] = omsq_soln # ω^2
# 		ms.∂ω²∂k[eigind] = ∂ω²∂k
# 		copyto!(ms.H⃗, H⃗)
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
# 		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		# Nranges = eachindex(ms.grid)
#
# 		H = reshape(H⃗,(2,Ns...))
# 	    if typeof(k̄)==ChainRulesCore.Zero
# 			k̄ = 0.
# 		end
# 		if typeof(H̄) != ChainRulesCore.Zero
# 			solve_adj!(ms,H̄,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
# 			# solve_adj!(ms,H̄,ω^2,H⃗,eigind)
# 			ms.λ⃗ -= dot(H⃗[:,eigind],ms.λ⃗) * H⃗[:,eigind]
# 			λ = reshape(ms.λ⃗,(2,Ns...))
# 			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 			λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 			ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd), Ns...)
# 			eīₕ = copy(ms.ε⁻¹_bar)
# 			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 			ms.λd *=  ms.M̂.Ninv
# 			λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
# 			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 			ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 			ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 			ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
# 			k̄ₕ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		else
# 			eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
# 			k̄ₕ = 0.0
# 		end
# 		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 		copyto!(ms.λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * H⃗[:,eigind] )
# 		λ = reshape(ms.λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd),Ns...)
# 		eīₖ = copy(ms.ε⁻¹_bar)
# 		ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
# 		ε⁻¹_bar = eīₖ + eīₕ
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , ε⁻¹_bar)
#     end
#     return ((k, H⃗), solve_k_pullback)
# end


# function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	k, H⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log) # ,ω²_tol)
# 	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
# 		mag_m_n(x,dropgrad(ms.M̂.g⃗))
# 	end
#     function solve_k_pullback(ΔΩ)
# 		k̄, H̄ = ΔΩ
# 		# @show k̄
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		# Nx,Ny,Nz = ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz
# 		Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		Nranges = eachindex(ms.grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
# 		H = reshape(H⃗,(2,Ns...))
# 		# mn2 = vcat(reshape(ms.M̂.m,(1,3,Ns...)),reshape(ms.M̂.n,(1,3,Ns...)))
# 	    if typeof(k̄)==ChainRulesCore.Zero
# 			k̄ = 0.
# 		end
# 		if typeof(H̄) != ChainRulesCore.Zero
# 			solve_adj!(ms,H̄,eigind) 											 # overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = H̄ - dot(H⃗,H̄)*H⃗
# 			# ms.λ⃗ += ( k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] - dot(H⃗,ms.λ⃗) ) * H⃗[:,eigind]
# 			ms.λ⃗ += ( k̄ / ms.∂ω²∂k[eigind] - dot(H⃗,ms.λ⃗) ) * H⃗[:,eigind]
# 		else
# 			# ms.λ⃗ = ( k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] ) * H⃗[:,eigind]
# 			ms.λ⃗ =   k̄ / ms.∂ω²∂k[eigind] * H⃗[:,eigind]
# 		end
# 		λ = reshape(ms.λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(ms.λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(ms.λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		ε⁻¹_bar!(ms.ε⁻¹_bar, vec(ms.M̂.d), vec(ms.λd),Ns...)
# 		# λ -= ( 2k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] ) * H
# 		# ms.λd -= ( ( 2k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] ) * ms.M̂.N ) * ms.M̂.d
# 		# ms.λd *=  ms.M̂.Ninv
# 		λ -= ( 2k̄ / ms.∂ω²∂k[eigind] ) * H
# 		ms.λd -= ( ( 2k̄ / ms.∂ω²∂k[eigind] ) * ms.M̂.N ) * ms.M̂.d
# 		ms.λd *=  ms.M̂.Ninv
# 		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.λẽ , ms.λd  ,ms ) )
# 		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 		ms.kx̄_m⃗ .= real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 		ms.kx̄_n⃗ .=  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 		ms.māg .= dot.(n⃗, ms.kx̄_n⃗) + dot.(m⃗, ms.kx̄_m⃗)
# 		k̄_kx = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NO_FIELDS, ChainRulesCore.Zero(), ms.ω̄  , ms.ε⁻¹_bar)
#     end
#     return ((k, H⃗), solve_k_pullback)
# end



########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
# # old
#
# function ChainRulesCore.rrule(::typeof(solve_ω²), ms::ModeSolver{T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{T,5};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
# 	(mag, m⃗, n⃗), mag_m_n_pb = update_k_pb(ms.M̂,k)
# 	Ω = solve_ω²(ms,ε⁻¹; nev, eigind, maxiter, tol, log)
#     function solve_ω²_pullback(ΔΩ) # ω̄ ₖ)
#         ω², H⃗ = Ω
# 		ω̄sq, H̄ = ΔΩ
# 		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
# 		H = reshape(H⃗,(2,Nx,Ny,Nz))
# 		mn2 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,Nx,Ny,Nz)))
# 	    if typeof(ω̄sq)==ChainRulesCore.Zero
# 			ω̄sq = 0.
# 		end
# 		if typeof(H̄)==ChainRulesCore.Zero
# 			λ⃗ =  -ω̄sq * H⃗
# 		else
# 			λ⃗₀ = IterativeSolvers.bicgstabl(
# 											ms.M̂-ω²*I, # A
# 											H̄ - H⃗ * dot(H⃗,H̄), # b,
# 											3,  # "l"
# 											)
# 			λ⃗ = λ⃗₀ - (ω̄sq + dot(H⃗,λ⃗₀)) * H⃗  # (P * λ⃗₀) + ω̄sq * H⃗ # λ⃗₀ + ω̄sq * H⃗
# 		end
# 		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
# 		d =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
# 		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
# 		d⃗ = vec( d )
# 		λ⃗d = vec( λd )
# 		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 		λẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) / (Nx * Ny * Nz)) )
# 		ẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹)) ) # pre-scales needed to compensate fft/
# 		kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,:,:,:)) .+ ẽ .* conj.(view(λ,2,:,:,:)) )
# 		kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,:,:,:)) .+ ẽ .* conj.(view(λ,1,:,:,:)) )
# 		māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# 		# m̄ = kx̄_m⃗ .* mag
# 		# n̄ = kx̄_n⃗ .* mag
# 		k̄ = mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]
# 		if !(typeof(k)<:SVector)
# 			k̄ = k̄[3]
# 		end
# 		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
# 		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
# 		ε⁻¹_bar = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(zeros(Float64,(3,3,Nx,Ny,Nz)))
# 		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
# 	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
# 	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
# 	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
# 	        end
# 	        for a2=1:2 # loop over first off diagonal
# 	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
# 	        end
# 	        # a = 1, set 1,3 and 3,1, second off-diagonal
# 	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
# 	    end
# 		return (NO_FIELDS, ChainRulesCore.Zero(), k̄, ε⁻¹_bar)
#     end
#     return (Ω, solve_ω²_pullback)
# end
#
# function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{T},ω::T,ε⁻¹::AbstractArray{T,5};
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
# 	k, H⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log ,ω²_tol)
# 	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
# 		mag_m_n(x,dropgrad(ms.M̂.g⃗))
# 	end
#     function solve_k_pullback(ΔΩ)
# 		k̄, H̄ = ΔΩ
# 		Nx,Ny,Nz = ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz
# 		H = reshape(H⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz))
# 		mn2 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
# 	    if typeof(k̄)==ChainRulesCore.Zero
# 			k̄ = 0.
# 		end
# 		ω̄sq_eff = -k̄ / ms.∂ω²∂k[eigind] - ms.∂ω²∂k[eigind]
# 		if typeof(H̄)==ChainRulesCore.Zero
# 			λ⃗ =  ω̄sq_eff * H⃗
# 		else
# 			λ⃗₀ = IterativeSolvers.bicgstabl(
# 											ms.M̂-(ω^2)*I, # A
# 											H̄ - H⃗ * dot(H⃗,H̄), # b,
# 											3,  # "l"
# 											)
# 			λ⃗ = λ⃗₀ - ( ω̄sq_eff  + dot(H⃗,λ⃗₀) ) * H⃗
# 		end
# 		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
# 		d =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv # ms.M̂.𝓕 * kx_tc( H , mn2, mag )  / (Nx * Ny * Nz)
# 		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag )
# 		d⃗ = vec( d )
# 		λ⃗d = vec( λd )
# 		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
# 		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
# 		ε⁻¹_bar = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(zeros(Float64,(3,3,Nx,Ny,Nz)))
# 		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
# 	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
# 	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
# 	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
# 	        end
# 	        for a2=1:2 # loop over first off diagonal
# 	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
# 	        end
# 	        # a = 1, set 1,3 and 3,1, second off-diagonal
# 	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
# 	    end
# 		λ -= ( 2k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] ) * H  # now λ⃗ = λ⃗₀ - ( k̄ / ms.∂ω²∂k[eigind] + ms.∂ω²∂k[eigind] + dot(H⃗,λ⃗₀) ) * H⃗
# 		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag )
# 		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 		λẽ = reinterpret(reshape,SVector{3,Complex{T}}, ( ms.M̂.𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) * ms.M̂.Ninv ) ) # reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) / (Nx * Ny * Nz)) )
# 		ẽ = reinterpret(reshape,SVector{3,Complex{T}}, ( ms.M̂.𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹)) )
# 		kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,:,:,:)) .+ ẽ .* conj.(view(λ,2,:,:,:)) )
# 		kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,:,:,:)) .+ ẽ .* conj.(view(λ,1,:,:,:)) )
# 		māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# 		k̄_kx = mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , ε⁻¹_bar)
#     end
#     return ((k, H⃗), solve_k_pullback)
# end
#
# function ChainRulesCore.rrule(::typeof(solve_ω²), k::T, ε⁻¹::Array{T,5},Δx::T,Δy::T,Δz::T;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
#     Ω = solve_ω²(k,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
#     function solve_ω²_pullback(ΔΩ) # ω̄ ₖ)
#         H⃗, ω² = Ω
# 		H̄, ω̄sq = ΔΩ
# 		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
# 		H = reshape(H⃗[:,eigind],(2,Nx,Ny,Nz))
# 		(mag, mn), magmn_pb = Zygote.pullback(k) do k
# 		    # calc_kpg(k,make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
# 			calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
# 		end
# 	    if typeof(ω̄sq)==ChainRulesCore.Zero
# 			ω̄sq = 0.
# 		end
# 		𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
# 		𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
# 		if typeof(H̄)==ChainRulesCore.Zero
# 			λ⃗ =  -ω̄sq * H⃗[:,eigind]
# 		else
# 			λ⃗₀ = IterativeSolvers.bicgstabl(
# 											M̂_old(ε⁻¹,mn,mag,𝓕,𝓕⁻¹)-ω²[eigind]*I, # A
# 											H̄[:,eigind] - H⃗[:,eigind] * dot(H⃗[:,eigind],H̄[:,eigind]), # b,
# 											3,  # "l"
# 											)
# 			λ⃗ = λ⃗₀ - (ω̄sq + dot(H⃗[:,eigind],λ⃗₀)) * H⃗[:,eigind]  # (P * λ⃗₀) + ω̄sq * H⃗[:,eigind] # λ⃗₀ + ω̄sq * H⃗[:,eigind]
# 		end
# 		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
# 		d =  𝓕 * kx_t2c( H , mn, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
# 		λd = 𝓕 * kx_t2c( λ, mn, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
# 		d⃗ = vec( d )
# 		λ⃗d = vec( λd )
# 		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 		λẽ = vec( 𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) )
# 		ẽ = vec( 𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹) * (Nx * Ny * Nz) ) # pre-scales needed to compensate fft/ifft normalization asymmetry. If bfft is used, this will need to be adjusted
# 		λẽ_3v = reinterpret(SVector{3,ComplexF64},λẽ)
# 		ẽ_3v = reinterpret(SVector{3,ComplexF64},ẽ)
# 		λ_2v = reinterpret(SVector{2,ComplexF64},λ⃗)
# 		H_2v = reinterpret(SVector{2,ComplexF64},H⃗[:,eigind])
# 		kx̄ = reshape( reinterpret(Float64, -real.( λẽ_3v .* adjoint.(conj.(H_2v)) + ẽ_3v .* adjoint.(conj.(λ_2v)) ) ), (3,2,Nx,Ny,Nz) )
# 		@tullio māg[ix,iy,iz] := mn[a,2,ix,iy,iz] * kx̄[a,1,ix,iy,iz] - mn[a,1,ix,iy,iz] * kx̄[a,2,ix,iy,iz]
# 		mn̄_signs = [-1 ; 1]
# 		@tullio mn̄[a,b,ix,iy,iz] := kx̄[a,3-b,ix,iy,iz] * mag[ix,iy,iz] * mn̄_signs[b] nograd=mn̄_signs
# 		k̄ = magmn_pb((māg,mn̄))[1]
# 		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
# 		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
# 		ε⁻¹_bar = zeros(Float64,(3,3,Nx,Ny,Nz))
# 		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
# 	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
# 	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
# 	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
# 	        end
# 	        for a2=1:2 # loop over first off diagonal
# 	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
# 	        end
# 	        # a = 1, set 1,3 and 3,1, second off-diagonal
# 	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
# 	    end
# 		return (NO_FIELDS, k̄, ε⁻¹_bar,ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
#     end
#     return (Ω, solve_ω²_pullback)
# end
