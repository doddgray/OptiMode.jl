using Revise
using LinearAlgebra, StaticArrays, GeometryPrimitives, ArrayInterface, ChainRules, ForwardDiff, Zygote, FiniteDifferences, BenchmarkTools, LoopVectorization, Tullio
using Test, ChainRulesTestUtils
using Zygote: @ignore, dropgrad, Context, _pullback, @adjoint
using ForwardDiff: Dual, Tag
using ChainRules: @non_differentiable

ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (nothing, dv) )

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

function _norm2_back_SV2c(x::SVector{2,T}, y, Δy) where T<:Complex
    ∂x = Vector{T}(undef,2)
    ∂x .= conj.(x) .* (real(Δy) * pinv(y))
    return reinterpret(SVector{2,T},∂x)[1]
end

function _norm2_back_SV3c(x::SVector{3,T}, y, Δy) where T<:Complex
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

# #### Polygon fn defs
# function v_pgn(wₜₒₚ::T1,t_core::T1,θ::T1,edge_gap::T1,n_core::T1,n_subs::T1,Δx::T2,Δy::T2) where {T1<:Real,T2<:Real}
# 	t_subs = (Δy -t_core - edge_gap )/2.
# 	c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
# 	ε_core = ε_tensor(n_core)
# 	ε_subs = ε_tensor(n_subs)
# 	wt_half = wₜₒₚ / 2
# 	wb_half = wt_half + ( t_core * tan(θ) )
# 	@show tc_half = t_core / 2
# 	verts =   [   	wt_half     -wt_half     -wb_half    wb_half
# 					tc_half     tc_half    -tc_half      -tc_half    ]'
#     return SMatrix{4,2}(verts)
# end
# v_pgn(p::AbstractVector) = v_pgn(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
# f_onbnd_poly(pin,abs∆xe) =  (p = Zygote.@ignore pin; onbnd = Zygote.@ignore ( abs∆xe .≤ Base.rtoldefault(Float64) * maximum(abs.((-)(bounds(p)...))) ); onbnd)
# f_isout_poly(p,∆xe) =  (isout = Zygote.@ignore ( (∆xe.>0) .| f_onbnd_poly(p,abs.(∆xe))); isout)
# function l_bnds_poly(v::SMatrix{K,2,T})::SVector{2,T} where {K,T<:Real}
# 	@tullio (min) res[b] :=  (v)[a,b]
# end
# function u_bnds_poly(v::SMatrix{K,2,T})::SVector{2,T} where {K,T<:Real}
# 	@tullio (max) res[b] :=  (v)[a,b]
# end
# bounds2(s::Polygon) = l_bnds_poly(s.v), u_bnds_poly(s.v)
# function _Δxe_poly(x::SVector{2,T},v::SMatrix{K,2,T},n::SMatrix{K,2,T})::SVector{K,T}  where {K,T<:Real}
# 	@tullio out[k] := n[k,a] * ( x[a] - v[k,a] ) # edge line eqns for a K-point Polygon{K} `s`
# end
# function _Δxe_poly2(x::SVector{2},v::SMatrix{K,2},n::SMatrix{K,2})  where {K}   #
# 	@tullio out[k] := n[k,a] * ( x[a] - v[k,a] ) # edge line eqns for a K-point Polygon{K} `s`
# end
# Δxe_poly(x::AbstractVector{<:Real},s::Polygon) = _Δxe_poly2(x,s.v,s.n)
# function f_onbnd2(Δxe::SVector{K},s::Polygon{K})::SVector{K} where K
# 	map(x->abs(x)≤rmin,Δxe)
# end
# function f_onbnd2(x::SVector{2},s::Polygon{K})::SVector{K} where K
# 	map(x->abs(x)≤rmin,Δxe_poly(x,s))
# end
# function _Δx_poly(x::SVector{2},v::SMatrix{K,2},n::SMatrix{K,2},kmax::Int)::SVector{2}  where {K} #,T<:Real}
# 	# @tullio Δx[i] := n[$kmax,a] * ( v[$kmax,a] - x[a] * n[$kmax,i]	# works but gradient doesn't vectorize
# 	( @tullio Δx_factor := n[$kmax,a] * ( v[$kmax,a] - x[a] ) )  * SVector(n[kmax,1],n[kmax,2])
# end
# function _Δx_poly3(x::SVector{2},v::SMatrix{K,2},n::SMatrix{K,2},kmax::Int)  where {K} #,T<:Real}
# 	@tullio Δx[i] := n[$kmax,a] * ( v[$kmax,a] - x[a] ) * n[$kmax,i]	# works but gradient doesn't vectorize
# 	# ( @tullio Δx_factor := n[$kmax,a] * ( v[$kmax,a] - x[a] ) )  * SVector(n[kmax,1],n[kmax,2])
# end
# Δx_poly(x::AbstractVector{<:Real},s::Polygon)::SVector{2} = _Δx_poly(x,s.v,s.n,argmax(_Δxe_poly(x,s.v,s.n)))
#
#
# function f_Δxe(n::AbstractArray{<:Real},x::AbstractVector{<:Real},v::AbstractArray{<:Real})::AbstractVector{<:Real}
# 	# edge line eqns for a K-point Polygon{K} `s`:
# 	# ∆xe::SVector{K} = sum(s.n .* (x' .- s.v), dims=Val(2))[:,1]
# 	@tullio out[k] := n[k,a] * ( x[a] - v[k,a] )
# end
# function f_sz(n::AbstractArray,x::AbstractVector,v::AbstractArray)::AbstractVector
# 	# Determine if x is outside of edges, inclusive.
# 	# sz = abs.((-)(bounds(s)...))  # SVector{2}
# 	# onbnd = abs∆xe .≤ Base.rtoldefault(Float64) * max(sz.data...)  # SVector{K}
# 	@tullio sz (-) := n[k,a] * ( x[a] - v[k,a] )
# end
# @non_differentiable f_onbnd_poly(::Any,::Any)
# @non_differentiable f_isout_poly(::Any,::Any)
#
# function _∆x_poly(x::SVector{2},v::SMatrix{K,2},n::SMatrix{K,2},kmax::Int)::SVector{2}  where {K} #,T<:Real}
# 	# @tullio Δx[i] := n[$kmax,a] * ( v[$kmax,a] - x[a] * n[$kmax,i]	# works but gradient doesn't vectorize
# 	( @tullio Δx_factor := n[$kmax,a] * ( v[$kmax,a] - x[a] ) )  * SVector(n[kmax,1],n[kmax,2])
# end
# ∆x_poly(x::AbstractVector{<:Real},s::Polygon)::SVector{2} = _∆x_poly(x,s.v,s.n,argmax(_∆xe_poly(x,s.v,s.n)))
#
# function surfpt_nearby2(x::SVector{2,T1}, s::Polygon{K,K2,D,T2})::Tuple{SVector{2, <:Real}, SVector{2, <:Real}} where {K,K2,D,T1<:Real,T2<:Real}
#     #∆xe::SVector{K,T} = @inbounds sum(s.n .* (x' .- s.v), dims=2)[1:K,1]  # Calculate the signed distances from x to edge lines.
# 	∆xe = Δxe_poly(x,s) ##∆xe_poly(x,s)
# 	# abs∆xe::SVector{K,T} = abs.(∆xe)
#     onbnd = Zygote.@ignore( f_onbnd_poly(s, abs.(∆xe) ) )# abs∆xe .≤ Base.rtoldefault(Float64) * max(sz.data...)  # SVector{K}
#     # isout::BitVector = Zygote.@ignore(  )#(∆xe.>0) .| onbnd  # SVector{K}
#     cout = Zygote.@ignore( count(f_isout_poly(s,∆xe)) )
#     if cout == 2  # x is outside two edges
# 		# println("case 1")
# 		∆xv = ones(4)*x' - s.v #x' .- s.v::SMatrix{K,2,T,K2}
#         l∆xv = map(sqrt,mapreduce(abs2,+,∆xv,dims=2))# sum(abs2,Δxv;dims=2) # hypot.(∆xv[1:K,1], ∆xv[1:K,2])
#         imin = 2 #argmin(l∆xv)
#         surf = @inbounds SVector(s.v[imin,1],s.v[imin,2])
#         imin₋₁ =1 # mod1(imin-1,K)
#         if onbnd[imin] && onbnd[imin₋₁]  # x is very close to vertex imin
# 			# println("case 1.1")
# 			nout = SVector( normalize( [ s.n[imin,1]+s.n[imin₋₁,1], s.v[imin,2]+s.n[imin₋₁,2] ]) )
# 			# nout = reinterpret(SVector{2,T}, normalize( [ s.n[imin,1]+s.n[imin₋₁,1],s.v[imin,2]+s.n[imin₋₁,2] ]))  #  s.n[imin,:] + s.n[imin₋₁,:]
#         else
# 			# println("case 1.2")
# 			nout = SVector( normalize( [ x[1] - s.v[imin,1] , x[2] - s.v[imin,2] ] ) )
#             # nout = reinterpret(SVector{2,T}, normalize( [ x[1] - s.v[imin,1] , x[2] - s.v[imin,2] ] ) )[1]
#         end
#         # nout = normalize(nout)
#     else  # cout ≤ 1 or cout ≥ 3
# 		# println("case 2")
#         imax::Int = Zygote.@ignore(argmax(∆xe))
#         # vmax = SVector{2,T}(s.v[imax,1], s.v[imax,2])
# 		# nout  = SVector{2,T}(s.n[imax,1], s.n[imax,2])
#         # ∆x = (nout⋅(vmax-x)) .* nout
#         # surf = x + ∆x
# 		surf = x + _∆x_poly(x,s.v,s.n,imax) # ∆x = (nmax⋅(vmax-x)) .* nmax
#         nout = @inbounds SVector(s.n[imax,1],s.n[imax,2] )
# 		# vmax = SVector( 1.009770432073105, -0.35 )  #::SVector{2,T} = SVector{2,T}(s.v[imax,1], s.v[imax,2])
# 		# nout = SVector( 0.9749279121818236, 0.2225209339563144 )# = SVector{2,T}(s.n[imax,1], s.n[imax,2])
#         # ∆x = SVector( 0.006711245692134241, 0.0015317980342586456 ) #::SVector{2,T} = (nout⋅(vmax-x)) .* nout
#         # surf = SVector( 0.9067112456921342, 0.10153179803425866 ) # = x + ∆x
#         # nout = nmax
#     end
#     return surf, nout::SVector
# end
#
#
#
# f_onbnd(bin,absdin) =  (b = Zygote.@ignore bin; absd = Zygote.@ignore absdin; abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r)  # basically b.r .≈ absd but faster
#
# f_isout(b,absd) =  (isout = Zygote.@ignore ((b.r.<absd) .| f_onbnd(b,absd) ); isout)
# # isout(bin) =  (b = Zygote.@ignore bin; (b.r.<absd) .| (abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r))
# f_signs(d) =  (signs = Zygote.@ignore (copysign.(1.0,d)'); signs)
#
#
# function surfpt_nearby2(x::AbstractVector{<:Real}, b::GeometryPrimitives.Box{2,4,D,T}) where {D,T<:Real}
#     ax = inv(b.p)
#     n0 = b.p ./  [ sqrt(b.p[1,1]^2 + b.p[1,2]^2) sqrt(b.p[2,1]^2 + b.p[2,2]^2)  ]
#     d = Array(b.p * (x - b.c))
#     cosθ = diag(n0*ax)
#
#     n = n0 .* copysign.(1.0,d)' #f_signs(d)
#     absd = abs.(d)
#     ∆ = (b.r .- absd) .* cosθ
#     # onbnd = abs.(dropgrad(b.r).-dropgrad(absd)) .≤ Base.rtoldefault(Float64) .* dropgrad(b.r)  # basically b.r .≈ absd but faster
#     # isout = (dropgrad(b.r).<dropgrad(absd)) .| dropgrad(onbnd)
#     onbnd = f_onbnd(b,absd)
#     isout = f_isout(b,absd)
#     projbnd =  all(.!isout .| onbnd)
#     # onbnd_float =  reinterpret(T,onbnd) #Float64.(onbnd)
#     # isout_float =  reinterpret(T,isout) #Float64.(isout)
#     # if ( ( abs(b.r[1]-absd[1]) ≤ Base.rtoldefault(Float64) * b.r[1] ) | (b.r[1]<absd[1]) ) | ( ( abs(b.r[2]-absd[2]) ≤ Base.rtoldefault(Float64) * b.r[2] ) | ( b.r[2] < absd[2] ) )
#     if count(isout) == 0
#         # = not(isout[1]) &  not(isout[2]) case, point is inside box
#         l∆x, i = findmin(∆)  # find closest face
#         nout = n[i,:]
#         ∆x = l∆x * nout
#     else
#         ∆x = n' * (∆ .* isout)
#         # = isout[1] | isout[2] case 1: at least one dimension outside or on boundry
#         if all(.!isout .| onbnd)
#             nout0 = n' * onbnd
#         else
#             nout0 = -∆x
#         end
#         nout = nout0 / norm(nout0)
#     end
#
#     return SVector{2}(x+∆x), SVector{2}(nout)
# end


@non_differentiable KDTree(::Any)
@non_differentiable copysign(::Any,::Any)
ChainRules.refresh_rules()
Zygote.refresh()

const εᵥ = I
function ε_tensor(n::T)::SMatrix{3,3,T,9} where T<:Real
    n² = n^2
    ε =     [	n²      0. 	    0.
                0. 	    n² 	    0.
                0. 	    0. 	    n²  ]
end
function ridge_wg(wₜₒₚ::T1,t_core::T1,θ::T1,edge_gap::T1,n_core::T1,n_subs::T1,Δx::T2,Δy::T2)::Vector{<:GeometryPrimitives.Shape} where {T1<:Real,T2<:Real}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
    verts =   [   wt_half     -wt_half     -wb_half    wb_half
                    tc_half     tc_half    -tc_half      -tc_half    ]'
    core = GeometryPrimitives.Polygon(                                                                # Instantiate 2D polygon, here a trapazoid
                    SMatrix{4,2}(verts),                                                                    # v: polygon vertices in counter-clockwise order
                    ε_core,                                                                            # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]
    b_subs = GeometryPrimitives.Box(                                                        # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],                    # c: center
                    [Δx - edge_gap, t_subs ],        # r: "radii" (half span of each axis)
                    ax,                                    # axes: box axes
                    ε_subs,                                                # data: any type, data associated with box shape
                )
    return [core,b_subs]
end
p = [
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]
Δx,Δy,Δz,Nx,Ny,Nz = 6., 4., 1., 128, 128, 1
rwg(p) = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
shapes = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
s = shapes[1]


function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end


import Base: findfirst
function Base.findfirst(p::SVector{N}, s::Vector{S},backup::Shape{N})::Shape{N} where {N,S<:Shape{N}}
    for i in eachindex(s)
        b = bounds(s[i])
        if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
            return s[i]
        end
    end
    return backup
end

function Base.findfirst(p::SVector{N}, kd::KDTree{N},backup::Shape{N})::Shape{N} where {N}
    if isempty(kd.s)
        if p[kd.ix] ≤ kd.x
            return findfirst(p, kd.left, backup)
        else
            return findfirst(p, kd.right, backup)
        end
    else
        return findfirst(p, kd.s, backup)
    end
end

Base.findfirst(p::Vector{<:Real}, kd::KDTree{N}, backup) where {N} = findfirst(SVector{N}(p), kd, backup)
Base.findfirst(p::Vector{<:Real}, s::Vector{<:Shape{N}}, backup) where {N} = findfirst(SVector{N}(p), s, backup)

function findfirst2(p::SVector{N}, s::Vector{S},bu) where {N,S<:Shape{N}}
    for i in eachindex(s)
        b = @inbounds bounds(s[i])
        if @inbounds all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
            return @inbounds s[i]
        end
    end
    return bu
end

function findfirst2(p::SVector{N}, kd::KDTree{N},bu) where {N}
    if isempty(kd.s)
        if p[kd.ix] ≤ kd.x
            return findfirst2(p, kd.left,bu)
        else
            return findfirst2(p, kd.right,bu)
        end
    else
        return findfirst2(p, kd.s,bu)
    end
end


function avg_param2(param1, param2, n12, rvol1)
    n = normalize(n12) #n12 / norm(n12) #sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end


function make_ei(shapes::Vector{<:Shape};Δx::T,Δy::T,Δz::T,Nx::Int,Ny::Int,Nz::Int,
	 	δx::T=Δx/Nx, δy::T=Δy/Ny, δz::T=Δz/Nz, x=( ( δx .* (0:(Nx-1))) .- Δx/2. ),
		y=( ( δy .* (0:(Ny-1))) .- Δy/2. ), z=( ( δz .* (0:(Nz-1))) .- Δz/2. ) ) where T<:Real
    tree = Zygote.@ignore(KDTree(shapes))
    eibuf = Zygote.Buffer(randn(T,2),3,3,Nx,Ny,Nz)
	# map((a,b)->εₛ(shapes,a,b;tree,δx,δy),x,y)
	for i ∈ eachindex(x), j ∈ eachindex(y)
	# 	# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		eps = εₛ(shapes,dropgrad(x)[i],dropgrad(y)[j];tree=dropgrad(tree),δx=dropgrad(δx),δy=dropgrad(δy))
		epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        eibuf[:,:,i,j,1] = epsi # inv(εₛ(shapes,x[i],y[j];tree,δx,δy)) # epsi #(epsi' + epsi) / 2
    end
    return real(copy(eibuf)) #HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
end

function make_ei2(shapes::Vector{<:Shape},x,y)
    tree = KDTree(shapes)
    # eibuf = Zygote.Buffer(x,3,3,Nx,Ny,Nz)
	eibuf = Zygote.Buffer(zeros(SMatrix{3,3,Float64,9},2),Nx,Ny,Nz)
	# map((a,b)->εₛ(shapes,a,b;tree,δx,δy),x,y)
	for i ∈ eachindex(x), j ∈ eachindex(y)
	# 	# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        eibuf[i,j,1] = inv(εₛ(shapes,x[i],y[j];tree,δx,δy)::SMatrix{3,3,Float64,9}) # epsi #(epsi' + epsi) / 2
    end
    return real(copy(eibuf)) #HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
end

function make_ei3(eibuf,shapes::Vector{<:Shape},x,y)
    tree = KDTree(shapes)
    # eibuf = Zygote.Buffer(x,3,3,Nx,Ny,Nz)
	# eibuf = Zygote.Buffer(zeros(SMatrix{3,3,Float64,9},2),Nx,Ny,Nz)
	# map((a,b)->inv(εₛ(shapes,a,b;tree,δx,δy)),x,y)
	for i ∈ eachindex(x), j ∈ eachindex(y)
	# 	# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        @inbounds eibuf[i,j,1] = inv(εₛ(shapes,x[i],y[j];tree,δx,δy)::SMatrix{3,3}) # epsi #(epsi' + epsi) / 2
    end
    return eibuf #HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
end



make_ei2_fwd(shapes,x,y) = Zygote.forwarddiff(shapes) do shapes
	make_ei2(shapes,x,y)
end


## Polygon testing

function foo1(p)
	s = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.,4.)[1]
	s.n[2,1] * s.v[3,2]
end
function foo2(p)
	s = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.,4.)[1]
	xy = SVector(0.9,0.1)
	spn = surfpt_nearby(xy,s)
	spn[1][1] * spn[2][2]
end
function foo3(p::Vector)
	s = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.,4.)[1]::Polygon{4,8}
	xy = SVector(0.9,0.1)
	spn = surfpt_nearby2(xy,s)
	spn[1][1] * spn[2][2]
end

function foo4(p::Vector)
	s = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.,4.)[2]
	xy = SVector(0.9,0.1)
	spn = surfpt_nearby(xy,s)
	spn[1][1] * spn[2][2]
end

function foo5(p::Vector)
	s = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.,4.)[2]::Box{2,4}
	xy = SVector(0.9,0.1)
	spn = surfpt_nearby2(xy,s)
	spn[1][1] * spn[2][2]
end


using OptiMode: εₛ, make_εₛ⁻¹

function goo1(p)
	s::Vector{<:Shape2} = rwg(p)
	εₛ(s,0.9,0.1;tree=KDTree(s),δx=Δx/Nx,δy=Δy/Ny)
end

function goo2(p)
	s::Vector{<:Shape2} = rwg(p)
	εₛ(s,0.9,0.1;tree=KDTree(s),δx=Δx/Nx,δy=Δy/Ny)
end


function eip(shapes::AbstractVector{<:GeometryPrimitives.Shape{2,4,D}},x::Real,y::Real;tree::KDTree,δx::Real,δy::Real,npix_sm::Int=1) where D
    s1 = @ignore(findfirst(SVector(x+δx/2.,y+δy/2.),tree))
    s2 = @ignore(findfirst(SVector(x+δx/2.,y-δy/2.),tree))
    s3 = @ignore(findfirst(SVector(x-δx/2.,y-δy/2.),tree))
    s4 = @ignore(findfirst(SVector(x-δx/2.,y+δy/2.),tree))

    ε1 = isnothing(s1) ? εᵥ : s1.data
    ε2 = isnothing(s2) ? εᵥ : s2.data
    ε3 = isnothing(s3) ? εᵥ : s3.data
    ε4 = isnothing(s4) ? εᵥ : s4.data
	return (ε1,ε2,ε3,ε4)
end

function eip2(shapes::AbstractVector{<:GeometryPrimitives.Shape{2,4,D}},x::Real,y::Real;tree::KDTree,δx::Real,δy::Real,npix_sm::Int=1) where D
	mapres1 = map((op1,op2)->findfirst([op1(x,δx),op2(y,δy)],tree),(+,+,-,-),(+,-,-,+))
	mapres2 = map(x->( isnothing(x) ? εᵥ : x.data ), mapres1 )
	return mapres2
end

##
println("@btime foo1, ∇foo1")
@show foo1(p)
@btime foo1($p)
@show Zygote.gradient(foo1,p)
@btime Zygote.gradient(foo1,$p)

println("@btime foo2, ∇foo2")
@show foo2(p)
@btime foo2($p)
@show Zygote.gradient(foo2,p)
@btime Zygote.gradient(foo2,$p)

println("@btime foo3, ∇foo3")
@show foo3(p)
@btime foo3($p)
@show Zygote.gradient(foo3,p)
@btime Zygote.gradient(foo3,$p)

## polygon step by step
K=4
@show v = v_pgn(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
@show w = v .- mean(v, dims=1)  # v in center-of-mass coordinates
@show ϕ = mod.(atan.(w[:,2], w[:,1]), 2π)  # SVector{K}: angle of vertices between 0 and 2π; `%` does not work for negative angle
if !issorted(ϕ)	# TODO: make sort_verts shuffling fn with AD rules, currently unsorted verts would break differentiability
    # Do this only when ϕ is not sorted, because the following uses allocations.
    ind = MVector{K}(sortperm(ϕ))  # sortperm(::SVector) currently returns Vector, not MVector
    v = v[ind,:]  # SVector{K}: sorted v
end
# Calculate the increases in angle between neighboring edges.
# ∆v = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,:]-v[end,:]))  # SMatrix{K,2}: edge directions
@show ∆v = vcat(diff(v, dims=1), transpose(v[1,:]-v[end,:]))
@show ∆z = ∆v[:,1] + im * ∆v[:,2]  # SVector{K}: edge directions as complex numbers
@show icurr = ntuple(identity, Val(K-1))
@show inext = ntuple(x->x+1, Val(K-1))
@show ∆ϕ = angle.(∆z[SVector(inext)] ./ ∆z[SVector(icurr)])  # angle returns value between -π and π
@show  n0 = [∆v[:,2] -∆v[:,1]]  # outward normal directions to edges
@show  n = n0 ./ hypot.(n0[:,1],n0[:,2])  # normalize

function foo(v)
	@show w = v .- mean(v, dims=1)  # v in center-of-mass coordinates
	@show ϕ = mod.(atan.(w[:,2], w[:,1]), 2π)  # SVector{K}: angle of vertices between 0 and 2π; `%` does not work for negative angle
	if !issorted(ϕ)	# TODO: make sort_verts shuffling fn with AD rules, currently unsorted verts would break differentiability
	    # Do this only when ϕ is not sorted, because the following uses allocations.
	    ind = MVector{K}(sortperm(ϕ))  # sortperm(::SVector) currently returns Vector, not MVector
	    v = v[ind,:]  # SVector{K}: sorted v
	end

	# Calculate the increases in angle between neighboring edges.
	# ∆v = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,:]-v[end,:]))  # SMatrix{K,2}: edge directions
	@show ∆v = vcat(diff(v, dims=1), transpose(v[1,:]-v[end,:]))
	@show ∆z = ∆v[:,1] + im * ∆v[:,2]  # SVector{K}: edge directions as complex numbers
	@show icurr = ntuple(identity, Val(K-1))
	@show inext = ntuple(x->x+1, Val(K-1))
	@show ∆ϕ = angle.(∆z[SVector(inext)] ./ ∆z[SVector(icurr)])  # angle returns value between -π and π
	@show  n0 = [∆v[:,2] -∆v[:,1]]  # outward normal directions to edges
	@show  n = n0 ./ hypot.(n0[:,1],n0[:,2])  # normalize
end
## box step by step
# spnby
T = Float64
@show x = SVector(2.8,-1.1)
@show b = rwg(p)[2]
@show ax = inv(b.p)  # axes: columns are unit vectors
@show n = (b.p ./ sqrt.(sum(abs2,b.p,dims=2)[:,1]))  # b.p normalized in row direction
@show cosθ = sum(ax.*n', dims=1)[1,:]  # equivalent to diag(n*ax)
# cosθ = diag(n*ax)  # faster than SVector(ntuple(i -> ax[:,i]⋅n[i,:], Val(N)))
# @assert all(cosθ .≥ 0)

@show d = b.p * (x - b.c)
@show n = n .* copysign.(1.0,d)  # operation returns SMatrix (reason for leaving n untransposed)
@show absd = abs.(d)
@show onbnd = abs.(b.r.-absd) .≤ Base.rtoldefault(T) .* b.r  # basically b.r .≈ absd but faster
@show isout = (b.r.<absd) .| onbnd
@show ∆ = (b.r .- absd) .* cosθ  # entries can be negative
if count(isout) == 0  # x strictly inside box; ∆ all positive
	@show l∆x, i = findmin(∆)  # find closest face
	@show nout = n[i,:]
	@show ∆x = l∆x * nout
else  # x outside box or on boundary in one or multiple directions
	@show ∆x = n' * (∆ .* isout)  # project out .!isout directions
	@show nout = all(.!isout .| onbnd) ? n'*onbnd : -∆x
	@show nout = normalize(nout)
end

# bounds
signmatrix(b::Box{1}) = SMatrix{1,1}(1)
signmatrix(b::Box{2}) = SMatrix{2,2}(1,1, -1,1)
signmatrix(b::Box{3}) = SMatrix{3,4}(1,1,1, -1,1,1, 1,-1,1, 1,1,-1)
@show A = inv(b.p) .* b.r'
@show m = maximum(abs.(A * signmatrix(b)), dims=2)[:,1]
## box step by step
K=4
@show v = v_pgn(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
@show w = v .- mean(v, dims=1)  # v in center-of-mass coordinates
@show ϕ = mod.(atan.(w[:,2], w[:,1]), 2π)  # SVector{K}: angle of vertices between 0 and 2π; `%` does not work for negative angle
if !issorted(ϕ)	# TODO: make sort_verts shuffling fn with AD rules, currently unsorted verts would break differentiability
    # Do this only when ϕ is not sorted, because the following uses allocations.
    ind = MVector{K}(sortperm(ϕ))  # sortperm(::SVector) currently returns Vector, not MVector
    v = v[ind,:]  # SVector{K}: sorted v
end
# Calculate the increases in angle between neighboring edges.
# ∆v = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,:]-v[end,:]))  # SMatrix{K,2}: edge directions
@show ∆v = vcat(diff(v, dims=1), transpose(v[1,:]-v[end,:]))
@show ∆z = ∆v[:,1] + im * ∆v[:,2]  # SVector{K}: edge directions as complex numbers
@show icurr = ntuple(identity, Val(K-1))
@show inext = ntuple(x->x+1, Val(K-1))
@show ∆ϕ = angle.(∆z[SVector(inext)] ./ ∆z[SVector(icurr)])  # angle returns value between -π and π
@show  n0 = [∆v[:,2] -∆v[:,1]]  # outward normal directions to edges
@show  n = n0 ./ hypot.(n0[:,1],n0[:,2])  # normalize



##

@tullio ∆z2[i] := ∆v[i,j] + im * ∆v[i,j]

∆v *  SMatrix{2,1}(1. + 0im, 0 + 1im)

∆v2 = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,1]-v[end,1],v[1,2]-v[end,2]))

@tullio n[i,j] = Δv

∆v2 ≈ ∆v
∆z2 ≈ ∆z
n0 = [∆v[:,2] -∆v[:,1]]  # outward normal directions to edges
n = n0 ./ hypot.(n0[:,1],n0[:,2])  # normalize

function mydiff(v::SMatrix{K,2,T}) where {K,T}
	SMatrix{K,2,T}(vcat(diff(v, dims=1), [v[1,1]-v[K,1]   v[1,2]-v[K,2]]))
end

circdiff0(v) = vcat(diff(v, dims=1), transpose(v[1,:]-v[end,:]))
dcircdiff0(v) = Zygote.gradient(x->circdiff0(x)[1,1],v)
circdiff0(v)
@btime circdiff0($v)
dcircdiff0(v)
@btime dcircdiff0($v)

circdiff1(v) = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,1]-v[end,1],v[1,2]-v[end,2]))
dcircdiff1(v) = Zygote.gradient(x->circdiff1(x)[1,1],v)
circdiff1(v)
@btime circdiff1($v)
dcircdiff1(v)
@btime dcircdiff1($v)
Zygote.gradient(x->circdiff1(x[1,1]),v)

circdiff2(v) = v - circshift(v,1)
dcircdiff2(v) = Zygote.gradient(x->circdiff2(x)[1,1],v)
circdiff2(v)
@btime circdiff2($v)
dcircdiff2(v)
@btime dcircdiff2($v)



diff(vcat(v,v),dims=1)[1:K,1:2] ≈ vcat(diff(v, dims=Val(1)), SMatrix{1,2}(v[1,:]-v[end,:]))
diff(vcat(v,v),dims=1)[1:K,1:2] ≈ vcat(diff(v, dims=1), SMatrix{1,2}(v[1,1]-v[end,1],v[1,2]-v[end,2]))
f1(v) = ( K=size(v)[1]; sum(abs2,diff(vcat(v,v),dims=1)[1:K,1:2]) )
f2(v) = sum(abs2,vcat(diff(v, dims=1), SMatrix{1,2}(v[1,1]-v[end,1],v[1,2]-v[end,2])))
f3(v) = sum(abs2,vcat(diff(v, dims=Val(1)), SMatrix{1,2}(v[1,:]-v[end,:])))
f1(v)
f2(v)
f3(v)
Zygote.gradient(f1,v)
Zygote.gradient(f2,v)
Zygote.gradient(f3,v)

##
function surfpt_nearby2(x::AbstractVector, s::GeometryPrimitives.Sphere{2})
    nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
                    normalize(x-s.c)
    return s.c+s.r*nout, nout
end








@btime bounds($s) # 771.861 ns (11 allocations: 848 bytes)
@btime bounds2($s) # 35.546 ns (2 allocations: 64 bytes)



# match original surfpt_nearby(::Polygon) code with vectorized, type-stable, differentiable fns
x = SVector(0.9,0.1)
x = SVector(1.9,1.1)
s = rwg(p)[1]	# point near edge and polygon to use for testing
# original
∆xe = sum(s.n .* (x' .- s.v), dims=Val(2))[:,1]  # SVector{K}: values of equations of edge lines
abs∆xe = abs.(∆xe)  # SVector{K}
# Determine if x is outside of edges, inclusive.
sz = abs.((-)(bounds(s)...))  # SVector{2}
onbnd = abs∆xe .≤ Base.rtoldefault(Float64) * max(sz.data...)  # SVector{K}
isout = (∆xe.>0) .| onbnd  # SVector{K}
cout = count(isout)

∆xv = x' .- s.v
l∆xv = hypot.(∆xv[:,1], ∆xv[:,2])
imin = argmin(l∆xv)
surf = s.v[imin,:]
imin₋₁ = mod1(imin-1,length(l∆xv))


imax = argmax(∆xe)
vmax, nmax = s.v[imax,:], s.n[imax,:]

∆x = (nmax⋅(vmax-x)) .* nmax




surf = x + ∆x
nout = nmax


∆xv = ones(4)*x' - s.v #x' .- s.v::SMatrix{K,2,T,K2}
l∆xv = map(sqrt,mapreduce(abs2,+,∆xv,dims=2))
sum(abs2,Δxv;dims=2) # hypot.(∆xv[1:K,1], ∆xv[1:K,2])
imin = argmin(l∆xv)
surf = @inbounds SVector{2,Float64}(s.v[imin,1],s.v[imin,2])
imin₋₁ = mod1(imin-1,K)


# new
∆xe = f_Δxe(s.n,x,s.v)  # SVector{K}: values of equations of edge lines
# abs∆xe = abs.(∆xe)  # SVector{K}
# Determine if x is outside of edges, inclusive.
# sz = abs.((-)(bounds(s)...))  # SVector{2}
# onbnd = abs∆xe .≤ Base.rtoldefault(Float64) * max(sz.data...)  # SVector{K}
# isout = (∆xe.>0) .| onbnd  # SVector{K}




function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end

function avg_param(param1, param2, n12, rvol1)
    n = n12 / norm(n12) #sqrt(sum2(abs2,n12))

    # Pick a vector that is not along n.
    if any(n .== 0)
    	htemp1 = (n .== 0)
    else
    	htemp1 = SVector(1., 0. , 0.)
    end

    # Create two vectors that are normal to n and normal to each other.
    htemp2 = n × htemp1
    h = htemp2 / norm(htemp2) #sqrt(sum2(abs2,htemp2))
    vtemp = n × h
    v = vtemp / norm(vtemp) #sqrt(sum2(abs2,vtemp))
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary

    τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform

    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average

    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

function εₛ(shapes::AbstractVector{<:GeometryPrimitives.Shape{2,4,D}},x::Real,y::Real;tree::KDTree,δx::Real,δy::Real,npix_sm::Int=1)::D where D
    # x1,y1 = x+npix_sm*δx/2.,y+npix_sm*δy/2.
    # x2,y2 = x+npix_sm*δx/2.,y-npix_sm*δy/2.
    # x3,y3 = x-npix_sm*δx/2.,y-npix_sm*δy/2.
    # x4,y4 = x-npix_sm*δx/2.,y+npix_sm*δy/2.
    # x1 = x+δx/2.
    # y1 = y+δy/2.
    # x2 = x+δx/2.
    # y2 = y-δy/2.
    # x3 = x-δx/2.
    # y3 = y-δy/2.
    # x4 = x-δx/2.
    # y4 = y+δy/2.

    s1 = @ignore(findfirst([x+δx/2.,y+δy/2.],tree))
    s2 = @ignore(findfirst([x+δx/2.,y-δy/2.],tree))
    s3 = @ignore(findfirst([x-δx/2.,y-δy/2.],tree))
    s4 = @ignore(findfirst([x-δx/2.,y+δy/2.],tree))

    ε1 = isnothing(s1) ? εᵥ : s1.data
    ε2 = isnothing(s2) ? εᵥ : s2.data
    ε3 = isnothing(s3) ? εᵥ : s3.data
    ε4 = isnothing(s4) ? εᵥ : s4.data

    if (ε1==ε2==ε3==ε4)
        return ε1
    else
        sinds = @ignore ( [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]] )
        n_unique = @ignore( length(unique(sinds)) )
        if n_unique==2
            s_fg = @ignore(shapes[minimum(dropgrad(sinds))])
            r₀,nout = surfpt_nearby2([x; y], s_fg)
            # bndry_pxl[i,j] = 1
            # nouts[i,j,:] = nout
            # vxl = (SVector{2}(x-δx/2.,y-δy/2.), SVector{2}(x+δx/2.,y+δy/2.))
            rvol = volfrac((SVector{2}(x-δx/2.,y-δy/2.), SVector{2}(x+δx/2.,y+δy/2.)),nout,r₀)
            sind_bg = @ignore(maximum(dropgrad(sinds))) #max(sinds...)
            ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
            # return avg_param(
            #         s_fg.data,
            #         ε_bg,
            #         [nout[1];nout[2];0],
            #         rvol,)
            return s_fg.data.^2 + 8*ε_bg
        else
            return (ε1+ε2+ε3+ε4)/4.
        end
    end
end

@adjoint function Broadcast.broadcasted(::typeof(εₛ), x::Union{T,AbstractArray{<:T}}) where {T<:Real}
    y, back = Zygote.broadcast_forward(εₛ, x)
    y, ȳ -> (nothing, back(ȳ)...)
end

function foo(p::Vector{<:Real},x::Real,y::Real;Δx::Real,Δy::Real,Nx::Int,Ny::Int)
    s::Vector{<:Shape2} = rwg(p)
    εₛ(s,x,y;tree=KDTree(s),δx=Δx/Nx,δy=Δy/Ny)
end
foop(x) = foo(x,0.9,0.1;Δx=6.,Δy=4.,Nx=128,Ny=128)[1,1]

function goo1(p::Vector{<:Real},x::Real,y::Real)
    s::Shape2 = rwg(p)[1]
    r₀,nout = surfpt_nearby2([x; y], s)
	return r₀[1] * r₀[2] * nout[1] * nout[2]
end
function goo2(p::Vector{<:Real},x::Real,y::Real)::Real
    s::Shape2 = rwg(p)[2]
    r₀,nout = surfpt_nearby2([x; y], s)
	return r₀[1] * r₀[2]
end
goop1(p) = goo1(p,0.9,0.1)
goop2(p::Vector{<:Real})::Real = goo2(p,0.9,0.1)

function hoo(p::Vector{<:Real},x::Real,y::Real)
    s::Vector{<:Shape2} = rwg(p)[1]
    r₀,nout = surfpt_nearby2([x; y], s)
	volfrac((SVector{2}(x-δx/2.,y-δy/2.), SVector{2}(x+δx/2.,y+δy/2.)),nout,r₀)
end
hoop(p) = goo1(p,0.9,0.1)

pgon = rwg(p)[1] # polygon

function goo(p)
	local xy = SVector(0.9,0.1)
	pgon::Polygon = rwg(p)[1]
	(r,n) = surfpt_nearby(xy,pgon)
	return r[1]
end

dgoo(p) = Zygote.gradient(goo,p)

goo(p)
dgoo(p)
@btime goo($p)


_,spn_pb = Zygote._pullback(Context(),(x,y)->( (r,n) = surfpt_nearby(x,y) ; return (r[1] * r[2] * n[1] * n[2])), SVector(0.9,0.1),pgon)

es_pb(1)
es_pb2(1)

foop(p)
goop1(p)
goop2(p)
hoop(p)

ForwardDiff.gradient(goop2,p)

p̄_FD3 = FiniteDifferences.grad(central_fdm(3,1),foop,p)[1]
@test Zygote.gradient(foop,p)[1] ≈ p̄_FD3
@test ForwardDiff.gradient(foop,p) ≈ p̄_FD3

@btime Zygote.gradient(foop,$p)[1]
@btime ForwardDiff.gradient(foop,$p)

@btime Zygote.gradient(goop1,$p)[1]
@btime ForwardDiff.gradient(goop1,$p)
@btime Zygote.gradient(goop2,$p)[1]
@btime ForwardDiff.gradient(goop2,$p)
@btime Zygote.gradient(hoop,$p)[1]
@btime ForwardDiff.gradient(hoop,$p)

##
M = randn(Float64,3,3)
Ms = SMatrix{3,3,Float64,9}(M)

##
a,b,c = randn(),randn(),randn()
v = [a,b,c]
vs = SVector(a,b,c)
norm(v)
norm(vs)
Zygote.gradient(norm,v)[1]
Zygote.gradient(norm,vs)[1]
@btime Zygote.gradient(norm,$v)[1]
@btime Zygote.gradient(norm,$vs)[1]

vsr = reinterpret(reshape,vs)

_, norm_pb = Zygote._pullback(Context(),norm,v)
_, norm_pb = Zygote._pullback(Context(),norm,vs)


normalize(v)
normalize(vs)
Zygote.gradient(x->normalize(x)[1],v)[1]
Zygote.gradient(x->normalize(x)[1],vs)[1]

@btime normalize($v)
@btime normalize($vs)
@btime Zygote.gradient(x->normalize(x)[1],$v)[1]
@btime Zygote.gradient(x->normalize(x)[1],$vs)[1]
