#### AD rules for StaticArrays Constructors ####

# SArray and subtypes  (already defined in my fork of GeometryPrimitives)
# ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
# ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
# ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
# ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
# ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
# ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )

# MArray and subtypes
ChainRulesCore.rrule(T::Type{<:MArray}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MArray}, x::AbstractArray) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:MMatrix}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MMatrix}, x::AbstractMatrix) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
ChainRulesCore.rrule(T::Type{<:MVector}, xs::Number...) = ( T(xs...), dv -> (ChainRulesCore.NoTangent(), dv...) )
ChainRulesCore.rrule(T::Type{<:MVector}, x::AbstractVector) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )
# ChainRulesCore.rrule(T::Type{<:HybridArray}, x::AbstractArray) = ( T(x), dv -> (ChainRulesCore.NoTangent(), dv) )

################# rrules for `reinterpret` with StaticArrays types #########################

# # AD rules for reinterpreting back and forth between N-D arrays of SVectors and (N+1)-D arrays
# function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T2},N2}) where {T1,N1,T2,N2}
# 	# return ( reinterpret(reshape,T1,A), Δ->( NoTangent(), ZeroTangent(), ZeroTangent(), reinterpret( reshape,SVector{N1,T1}, Δ ) ) )
# 	function reinterpret_reshape_SV_pullback(Δ)
# 		return (ChainRulescore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret(reshape,SVector{N1,eltype(Δ)},Δ))
# 	end
# 	( reinterpret(reshape,T1,A), reinterpret_reshape_SV_pullback )
# end
# function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
# 	return ( reinterpret(reshape,type,A), Δ->( ChainRulescore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape, eltype(A), Δ ) ) )
# end

# # need adjoint for constructor:
# # Base.ReinterpretArray{Float64, 2, SVector{3, Float64}, Matrix{SVector{3, Float64}}, false}. Gradient is of type FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
# import Base: ReinterpretArray
# function ChainRulesCore.rrule(T::Type{ReinterpretArray{T1, N1, SVector{N2, T2}, Array{SVector{N3, T3},N4}, IsReshaped}}, x::AbstractArray)  where {T1, N1, N2, T2, N3, T3, N4, IsReshaped}
# 	function ReinterpretArray_SV_pullback(Δ)
# 		if IsReshaped
# 			Δr = reinterpret(reshape,SVector{N2,eltype(Δ)},Δ)
# 		else
# 			Δr = reinterpret(SVector{N2,eltype(Δ)},Δ)
# 		end
# 		return (ChainRulescore.NoTangent(), Δr)
# 	end
# 	( T(x), ReinterpretArray_SV_pullback )
# end

# # AD rules for reinterpreting back and forth between N-D arrays of SMatrices and (N+2)-D arrays
# function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SMatrix{N1,N2,T2,N3},N4}) where {T1<:Real,T2,N1,N2,N3,N4}
# 	# @show A
# 	# @show eltype(A)
# 	# @show type
# 	# @show size(reinterpret(reshape,T1,A))
# 	# @show N1*N2
# 	# function f_pb(Δ)
# 	# 	@show eltype(Δ)
# 	# 	@show size(Δ)
# 	# 	# @show Δ
# 	# 	@show typeof(Δ)
# 	# 	return ( ChainRulescore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape,SMatrix{N1,N2,T1,N3}, Δ ) )
# 	# end
# 	# return ( reinterpret(reshape,T1,A), Δ->f_pb(Δ) )
# 	return ( reinterpret(reshape,T1,A), Δ->( ChainRulescore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape,SMatrix{N1,N2,T1,N3}, real(Δ) ) ) )
# end

# function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SMatrix{N1,N2,T1,N3}},A::AbstractArray{T1}) where {T1,T2,N1,N2,N3}
# 	# @show type
# 	# @show eltype(A)
# 	return ( reinterpret(reshape,type,A), Δ->( ChainRulescore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape, eltype(A), Δ ) ) )
# end

# # adjoint for constructor Base.ReinterpretArray{SMatrix{3, 3, Float64, 9}, 1, Float64, Vector{Float64}, false}.
# # Gradient is of type FillArrays.Fill{FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, 1, Tuple{Base.OneTo{Int64}}}

# function ChainRulesCore.rrule(::typeof(reinterpret), ::typeof(reshape), ::Type{R}, A::AbstractArray{T}) where {N1, N2, T, R <: SMatrix{N1,N2,T}}
#     function pullback(Ā)
#         ∂A = mapreduce(v -> v isa R ? v : zero(R), vcat, Ā; init = similar(A, 0))
#         return (ChainRulescore.NoTangent(), DoesNotExist(), DoesNotExist(), reshape(∂A, size(A)))
#     end
#     return (reinterpret(reshape, R, A), pullback)
# end

# # function rrule(T::Type{::R3, x::R2) where {T1, N1, N2, T2, R1<:SMatrix{N1,N2,T1}, R2<:AbstractArray{T2}, R3<:ReinterpretArray{R1}}}
# # 	function ReinterpretArray_SM_pullback(Δ)
# # 		Δr = reshape(reinterpret(T1,collect(Δ)),size(x))
# # 		# if IsReshaped
# # 		# 	Δr = reshape(reinterpret(T4,collect(Δ)),size(x))
# # 		# else
# # 		# 	Δr = reshape(reinterpret(T4,collect(Δ)),size(x))
# # 		# end
# # 		return (NoTangent(), Δr)
# # 	end
# # 	( T(x), ReinterpretArray_SM_pullback )
# # end

################# end rrules for `reinterpret` with StaticArrays  #################3


# # AD rules for fast norms of types SVector{2,T} and SVector{2,3}

# function _norm2_back_SV2r(x::SVector{2,T}, y, Δy) where T<:Real
#     ∂x = Vector{T}(undef,2)
#     ∂x .= x .* (real(Δy) * pinv(y))
#     return reinterpret(SVector{2,T},∂x)[1]
# end

# function _norm2_back_SV3r(x::SVector{3,T}, y, Δy) where T<:Real
#     ∂x = Vector{T}(undef,3)
#     ∂x .= x .* (real(Δy) * pinv(y))
#     return reinterpret(SVector{3,T},∂x)[1]
# end

# function _norm2_back_SV2r(x::SVector{2,T}, y, Δy) where T<:Complex
#     ∂x = Vector{T}(undef,2)
#     ∂x .= conj.(x) .* (real(Δy) * pinv(y))
#     return reinterpret(SVector{2,T},∂x)[1]
# end

# function _norm2_back_SV3r(x::SVector{3,T}, y, Δy) where T<:Complex
#     ∂x = Vector{T}(undef,3)
#     ∂x .= conj.(x) .* (real(Δy) * pinv(y))
#     return reinterpret(SVector{3,T},∂x)[1]
# end

# function rrule(::typeof(norm), x::SVector{3,T}) where T<:Real
# 	y = LinearAlgebra.norm(x)
# 	function norm_pb(Δy)
# 		∂x = Thunk() do
# 			_norm2_back_SV3r(x, y, Δy)
# 		end
# 		return ( NoTangent(), ∂x )
# 	end
# 	norm_pb(::ZeroTangent) = (NoTangent(), ZeroTangent())
#     return y, norm_pb
# end

# function rrule(::typeof(norm), x::SVector{2,T}) where T<:Real
# 	y = LinearAlgebra.norm(x)
# 	function norm_pb(Δy)
# 		∂x = Thunk() do
# 			_norm2_back_SV2r(x, y, Δy)
# 		end
# 		return ( NoTangent(), ∂x )
# 	end
# 	norm_pb(::ZeroTangent) = (NoTangent(), ZeroTangent())
#     return y, norm_pb
# end

# function rrule(::typeof(norm), x::SVector{3,T}) where T<:Complex
# 	y = LinearAlgebra.norm(x)
# 	function norm_pb(Δy)
# 		∂x = Thunk() do
# 			_norm2_back_SV3c(x, y, Δy)
# 		end
# 		return ( NoTangent(), ∂x )
# 	end
# 	norm_pb(::ZeroTangent) = (NoTangent(), ZeroTangent())
#     return y, norm_pb
# end

# function rrule(::typeof(norm), x::SVector{2,T}) where T<:Complex
# 	y = LinearAlgebra.norm(x)
# 	function norm_pb(Δy)
# 		∂x = Thunk() do
# 			_norm2_back_SV2c(x, y, Δy)
# 		end
# 		return ( NoTangent(), ∂x )
# 	end
# 	norm_pb(::ZeroTangent) = (NoTangent(), ZeroTangent())
#     return y, norm_pb
# end