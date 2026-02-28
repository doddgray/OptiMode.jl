################# rrules for `reinterpret` with StaticArrays types #########################

# AD rules for reinterpreting back and forth between N-D arrays of SVectors and (N+1)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T2},N2}) where {T1,N1,T2,N2}
	function reinterpret_reshape_SV_pullback(Δ)
		return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret(reshape,SVector{N1,eltype(Δ)},Δ))
	end
	( reinterpret(reshape,T1,A), reinterpret_reshape_SV_pullback )
end
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
	return ( reinterpret(reshape,type,A), Δ->( ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape, eltype(A), Δ ) ) )
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
		return (ChainRulesCore.NoTangent(), Δr)
	end
	( T(x), ReinterpretArray_SV_pullback )
end

# AD rules for reinterpreting back and forth between N-D arrays of SMatrices and (N+2)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SMatrix{N1,N2,T2,N3},N4}) where {T1<:Real,T2,N1,N2,N3,N4}
	return ( reinterpret(reshape,T1,A), Δ->( ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape,SMatrix{N1,N2,T1,N3}, real(Δ) ) ) )
end

function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SMatrix{N1,N2,T1,N3}},A::AbstractArray{T1}) where {T1,T2,N1,N2,N3}
	# @show type
	# @show eltype(A)
	return ( reinterpret(reshape,type,A), Δ->( ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent(), reinterpret( reshape, eltype(A), Δ ) ) )
end

# adjoint for constructor Base.ReinterpretArray{SMatrix{3, 3, Float64, 9}, 1, Float64, Vector{Float64}, false}.
# Gradient is of type FillArrays.Fill{FillArrays.Fill{Float64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, 1, Tuple{Base.OneTo{Int64}}}

function ChainRulesCore.rrule(::typeof(reinterpret), ::typeof(reshape), ::Type{R}, A::AbstractArray{T}) where {N1, N2, T, R <: SMatrix{N1,N2,T}}
    function pullback(Ā)
        ∂A = mapreduce(v -> v isa R ? v : zero(R), vcat, Ā; init = similar(A, 0))
        return (ChainRulesCore.NoTangent(), DoesNotExist(), DoesNotExist(), reshape(∂A, size(A)))
    end
    return (reinterpret(reshape, R, A), pullback)
end
