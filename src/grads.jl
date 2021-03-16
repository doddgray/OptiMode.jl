using ForwardDiff
using Zygote: @adjoint, Numeric, literal_getproperty, accum
using ChainRules: Thunk, @non_differentiable
export sum2, jacobian, ε⁻¹_bar!, ∂ω²∂k_adj, Mₖᵀ_plus_Mₖ

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
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:HybridArray}, x::AbstractArray) = ( T(x), dv -> (nothing, dv) )

# AD rules for reinterpreting back and forth between N-D arrays of SVectors and (N+1)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T1},N2}) where {T1,N1,N2}
	return ( reinterpret(reshape,T1,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SVector{N1,T1}, Δ ) ) )
end
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
	return ( reinterpret(reshape,type,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ ) ) )
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
		3;	# l = number of GMRES iterations per CG iteration
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


# function ChainRulesCore.rrule(::typeof(bicstabl),M̂,b,l;kwargs...)
# 	x = bicstabl(A, b, l; kwargs...)
# 	bicstabl_pb(x̄) =
# 	λ = bicstabl(A', x̄, l; kwargs...)
# 	return λ₀
# end

"""
function mapping |H⟩ ⤇ ( (∂M/∂k)ᵀ + ∂M/∂k )|H⟩
"""
function Mₖᵀ_plus_Mₖ(H⃗::AbstractVector{Complex{T}},ε⁻¹,mag,m,n) where T<:Real
	H = reshape(H⃗,(2,size(mag)...))
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	X = zx_tc(H,mn) + kx_tc(H,mn,mag)
	Y = ifft( ε⁻¹_dot( fft( X, (2:3) ), real(flat(ε⁻¹))), (2:3))
	-(kx_ct(Y,mn,mag) + zx_ct(Y,mn))
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
solve the adjoint sensitivity problem corresponding to ∂ω²∂k = <H|∂M/∂k|H>
"""
function ∂²ω²∂k²(M̂::HelmholtzMap{T},ω²,H⃗,ε⁻¹,mag,m,n;eigind=1,log=false) where T<:Real
	H̄ = vec(Mₖᵀ_plus_Mₖ(H⃗[:,eigind],ε⁻¹,mag,m,n))
	adj_res = ∂ω²∂k_adj(M̂,ω²,H⃗,H̄;eigind,log)
	λ⃗₀ = !log ? adj_res : adj_res[1]
	λ⃗ = λ⃗₀ + H⃗[:,eigind]
	Ns = size(M̂.mag)
	Nranges = eachindex(mag)
	H = reshape(H⃗[:,eigind],(2,Ns...))
	λ = reshape(λ⃗,(2,Ns...))
	mn = vcat(reshape(M̂.m,(1,size(M̂.m)...)),reshape(M̂.n,(1,size(M̂.m)...)))
	zxh = M̂.𝓕 * zx_tc(H,mn)  * M̂.Ninv
	λd =  M̂.𝓕 * kx_tc(λ,mn,mag)
	eī = similar(ε⁻¹)
	ε⁻¹_bar!(ε⁻¹_bar, vec(zxh), vec(λd), Ns...)
	#TODO replace iffts below with pre-planned ifft carried by M̂
	λẽf = ifft( ε⁻¹_dot( (λd * ms.M̂.Ninv), real(flat(ε⁻¹))), (2:3))
	ẽf = ifft( ε⁻¹_dot( zxh, real(flat(ε⁻¹))), (2:3))
	λẽ = reinterpret(reshape, SVector{3,Complex{T}}, λẽf )
	ẽ = reinterpret(reshape, SVector{3,Complex{T}}, ẽf )
	# scaling by mag or √mag may differ from normal case here, as one of the kx
	# operators has been replaced by ẑx, so two of the four terms in the next two
	# lines are a factor of mag smaller at each point in recip. space?
	kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
	kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
	māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
	# almost there! need to replace this pullback with a Zygote compatible fn.
	k̄ = -mag_m_n_pb(( ms.māg, ms.kx̄_m⃗.*mag, ms.kx̄_n⃗.*mag ))[1]	# should equal ∂/∂k(2 * ∂ω²/∂k) = 2∂²ω²/∂k²
end



function ∇solve_k(ΔΩ, Ω::Tuple{T,Matrix{Complex{T}}}, ∂ω²∂k::Vector{T}, ω, ε⁻¹, grid; eigind=1) where T<:Real
	k̄, H̄ = ΔΩ
	k, H⃗ = Ω
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
		ε⁻¹_bar!(ε⁻¹_bar, vec(M̂.d), vec(λd), Ns...)
		eīₕ = copy(ε⁻¹_bar)
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λd *=  M̂.Ninv
		λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,M̂ ) )
		ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d,M̂) )
		kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
		kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
		māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
		k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
	else
		eīₕ = fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
		k̄ₕ = 0.0
	end
	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k[eigind] ) * H⃗[:,eigind] )
	λ = reshape(λ⃗,(2,Ns...))
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(ε⁻¹_bar, vec(M̂.d), vec(λd),Ns...)
	eīₖ = copy(ε⁻¹_bar)
	ω̄  =  2ω * (k̄ + k̄ₕ ) / ∂ω²∂k[eigind] #2ω * k̄ₖ / ∂ω²∂k[eigind]
	ε⁻¹_bar = eīₖ + eīₕ
	# if !(typeof(k)<:SVector)
	# 	k̄_kx = k̄_kx[3]
	# end
	# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
	return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , ε⁻¹_bar)
end

function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	kH⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log)
    solve_k_pullback(ΔΩ) = let kH⃗=kH⃗, ∂ω²∂k=ms.∂ω²∂k, ω=ω, ε⁻¹=ε⁻¹, grid=ms.grid, eigind=eigind
		∇solve_k(ΔΩ,kH⃗,∂ω²∂k,ω,ε⁻¹,grid;eigind)
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
