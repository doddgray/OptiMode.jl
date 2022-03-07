# export sum2, jacobian, ε⁻¹_bar!, ε⁻¹_bar, ∂ω²∂k_adj, Mₖᵀ_plus_Mₖ, ∂²ω²∂k², herm
# export ∇ₖmag_m_n, ∇HMₖH, ∇M̂, ∇solve_k, ∇solve_k!, solve_adj!, neff_ng_gvd, ∂ε⁻¹_∂ω, ∂nng⁻¹_∂ω
# export ∇ₖmag_mn

@non_differentiable KDTree(::Any)
@non_differentiable g⃗(::Any)
@non_differentiable _fftaxes(::Any)

# include("grad_lib/ForwardDiff.jl")
# include("grad_lib/StaticArrays.jl")
# include("grad/solve.jl")


##### Begin newly commented section



# # Examples of how to assert type stability for broadcasting custom types (see https://github.com/FluxML/Zygote.jl/issues/318 )
# # Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T<:Real, V<:Real3Vector} = Real3Vector(Vector{T}(undef,3))
# # Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T, V<:Real3Vector} = Array{T}(undef, size(bc))

# @adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
# _ndims(::Base.HasShape{d}) where {d} = d
# _ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
# @adjoint function Iterators.product(xs...)
#                     d = 1
#                     Iterators.product(xs...), dy -> ntuple(length(xs)) do n
#                         nd = _ndims(xs[n])
#                         dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
#                         d += nd
#                         func = sum(y->y[n], dy; dims=dims)
#                         ax = axes(xs[n])
#                         reshape(func, ax)
#                     end
#                 end


# function sum2(op,arr)
#     return sum(op,arr)
# end

# function sum2adj( Δ, op, arr )
#     n = length(arr)
#     g = x->Δ*Zygote.gradient(op,x)[1]
#     return ( nothing, map(g,arr))
# end

# @adjoint function sum2(op,arr)
#     return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
# end


# # now-removed Zygote trick to improve stability of `norm` pullback
# # found referenced here: https://github.com/JuliaDiff/ChainRules.jl/issues/338
# function Zygote._pullback(cx::Zygote.AContext, ::typeof(norm), x::AbstractArray, p::Real = 2)
#   fallback = (x, p) -> sum(abs.(x).^p .+ eps(0f0)) ^ (one(eltype(x)) / p) # avoid d(sqrt(x))/dx == Inf at 0
#   Zygote._pullback(cx, fallback, x, p)
# end

# """
# jacobian(f,x) : stolen from https://github.com/FluxML/Zygote.jl/pull/747/files
#
# Construct the Jacobian of `f` where `x` is a real-valued array
# and `f(x)` is also a real-valued array.
# """
# function jacobian(f,x)
#     y,back  = Zygote.pullback(f,x)
#     k  = length(y)
#     n  = length(x)
#     J  = Matrix{eltype(y)}(undef,k,n)
#     e_i = fill!(similar(y), 0)
#     @inbounds for i = 1:k
#         e_i[i] = oneunit(eltype(x))
#         J[i,:] = back(e_i)[1]
#         e_i[i] = zero(eltype(x))
#     end
#     (J,)
# end

##### end newly commented section





### Zygote StructArrays rules from https://github.com/cossio/ZygoteStructArrays.jl
# @adjoint function (::Type{SA})(t::Tuple) where {SA<:StructArray}
#     sa = SA(t)
#     back(Δ::NamedTuple) = (values(Δ),)
#     function back(Δ::AbstractArray{<:NamedTuple})
#         nt = (; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...)
#         return back(nt)
#     end
#     return sa, back
# end
#
# @adjoint function (::Type{SA})(t::NamedTuple) where {SA<:StructArray}
#     sa = SA(t)
#     back(Δ::NamedTuple) = (NamedTuple{propertynames(sa)}(Δ),)
#     function back(Δ::AbstractArray)
#         back((; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...))
#     end
#     return sa, back
# end
#
# @adjoint function (::Type{SA})(a::A) where {T,SA<:StructArray,A<:AbstractArray{T}}
#     sa = SA(a)
#     function back(Δsa)
#         Δa = [(; (p => Δsa[p][i] for p in propertynames(Δsa))...) for i in eachindex(a)]
#         return (Δa,)
#     end
#     return sa, back
# end
#
# # Must special-case for Complex (#1)
# @adjoint function (::Type{SA})(a::A) where {T<:Complex,SA<:StructArray,A<:AbstractArray{T}}
#     sa = SA(a)
#     function back(Δsa) # dsa -> da
#         Δa = [Complex(Δsa.re[i], Δsa.im[i]) for i in eachindex(a)]
#         (Δa,)
#     end
#     return sa, back
# end
#
# @adjoint function literal_getproperty(sa::StructArray, ::Val{key}) where {key}
#     key::Symbol
#     result = getproperty(sa, key)
#     function back(Δ::AbstractArray)
#         nt = (; (k => zero(v) for (k,v) in pairs(fieldarrays(sa)))...)
#         return (Base.setindex(nt, Δ, key), nothing)
#     end
#     return result, back
# end
#
# @adjoint Base.getindex(sa::StructArray, i...) = sa[i...], Δ -> ∇getindex(sa,i,Δ)
# @adjoint Base.view(sa::StructArray, i...) = view(sa, i...), Δ -> ∇getindex(sa,i,Δ)
# function ∇getindex(sa::StructArray, i, Δ::NamedTuple)
#     dsa = (; (k => ∇getindex(v,i,Δ[k]) for (k,v) in pairs(fieldarrays(sa)))...)
#     di = map(_ -> nothing, i)
#     return (dsa, map(_ -> nothing, i)...)
# end
# # based on
# # https://github.com/FluxML/Zygote.jl/blob/64c02dccc698292c548c334a15ce2100a11403e2/src/lib/array.jl#L41
# ∇getindex(a::AbstractArray, i, Δ::Nothing) = nothing
# function ∇getindex(a::AbstractArray, i, Δ)
#     if i isa NTuple{<:Any, Integer}
#         da = Zygote._zero(a, typeof(Δ))
#         da[i...] = Δ
#     else
#         da = Zygote._zero(a, eltype(Δ))
#         dav = view(da, i...)
#         dav .= Zygote.accum.(dav, Zygote._droplike(Δ, dav))
#     end
#     return da
# end
#
# @adjoint function (::Type{NT})(t::Tuple) where {K,NT<:NamedTuple{K}}
#     nt = NT(t)
#     back(Δ::NamedTuple) = (values(NT(Δ)),)
#     return nt, back
# end

# # https://github.com/FluxML/Zygote.jl/issues/680
# @adjoint function (T::Type{<:Complex})(re, im)
# 	back(Δ::Complex) = (nothing, real(Δ), imag(Δ))
# 	back(Δ::NamedTuple) = (nothing, Δ.re, Δ.im)
# 	T(re, im), back
# end



