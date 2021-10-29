export _mult, _dot
export _cross, _cross_x, _cross_y, _cross_z
export _sum_cross, _sum_cross_x, _sum_cross_y, _sum_cross_z
export _outer, _expect
export slice_inv


"""
local multiplication, avoiding broadcast for AD performance
"""
# scalar-scalar muliplication
function _mult(s₁::AbstractArray{T1,2},s₂::AbstractArray{T2,2}) where {T1,T2}
	@tullio s₃[ix,iy] := s₁[ix,iy] * s₂[ix,iy]
end
function _mult(s₁::AbstractArray{T1,3},s₂::AbstractArray{T2,3}) where {T1,T2}
	@tullio s₃[ix,iy,iz] := s₁[ix,iy,iz] * s₂[ix,iy,iz]
end
# scalar-vector muliplication
function _mult(s::AbstractArray{T1,2},v₁::AbstractArray{T2,3}) where {T1,T2}
	@tullio v₂[a,ix,iy] := s[ix,iy] * v₁[a,ix,iy]
end
function _mult(v₁::AbstractArray{T2,3},s::AbstractArray{T1,2}) where {T1,T2}
	@tullio v₂[a,ix,iy] := s[ix,iy] * v₁[a,ix,iy]
end
function _mult(s::AbstractArray{T1,3},v₁::AbstractArray{T2,4}) where {T1,T2}
	@tullio v₂[a,ix,iy,iz] := s[ix,iy,iz] * v₁[a,ix,iy,iz]
end
function _mult(v₁::AbstractArray{T2,4},s::AbstractArray{T1,3}) where {T1,T2}
	@tullio v₂[a,ix,iy,iz] := s[ix,iy,iz] * v₁[a,ix,iy,iz]
end
# scalar-matrix muliplication
function _mult(s::AbstractArray{T1,2},v₁::AbstractArray{T2,4}) where {T1,T2}
	@tullio v₂[a,b,ix,iy] := s[ix,iy] * v₁[a,b,ix,iy]
end
function _mult(v₁::AbstractArray{T2,4},s::AbstractArray{T1,2}) where {T1,T2}
	@tullio v₂[a,b,ix,iy] := s[ix,iy] * v₁[a,b,ix,iy]
end
function _mult(s::AbstractArray{T1,3},v₁::AbstractArray{T2,5}) where {T1,T2}
	@tullio v₂[a,b,ix,iy,iz] := s[ix,iy,iz] * v₁[a,b,ix,iy,iz]
end
function _mult(v₁::AbstractArray{T2,5},s::AbstractArray{T1,3}) where {T1,T2}
	@tullio v₂[a,b,ix,iy,iz] := s[ix,iy,iz] * v₁[a,b,ix,iy,iz]
end

"""
dot products in the first few dimensions
"""
# first-order (linear) tensor-vector muliplication
function _dot(χ::AbstractArray{T,2},v₁::AbstractVector) where T<:Real
	@tullio v₂[i] := χ[i,j] * v₁[j]
end

function _dot(χ::AbstractArray{T,4},v₁::AbstractArray{Complex{T},3}) where T<:Real
	@tullio v₂[i,ix,iy] := χ[i,j,ix,iy] * v₁[j,ix,iy]
end

function _dot(χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},4}) where T<:Real
	@tullio v₂[i,ix,iy,iz] := χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz]
end

# first-order (linear) vector-tensor-vector muliplication (three element dot product)
function _3dot(v₂::AbstractVector,χ::AbstractArray{T,2},v₁::AbstractVector) where T<:Real
	@tullio out := conj(v₂)[i] * χ[i,j] * v₁[j]
end

function _3dot(v₂::AbstractArray{Complex{T},3},χ::AbstractArray{T,4},v₁::AbstractArray{Complex{T},3}) where T<:Real
	@tullio out[i,ix,iy] := conj(v₂)[i,ix,iy] * χ[i,j,ix,iy] * v₁[j,ix,iy]
end

function _3dot(v₂::AbstractArray{Complex{T},4},χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},4}) where T<:Real
	@tullio out[i,ix,iy,iz] := conj(v₂)[i,ix,iy,iz] * χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz]
end

# expectation value/inner product of vector field over tensor 
function _expect(χ::AbstractArray{T,2},v₁::AbstractVector) where T<:Real
	@tullio out := conj(v₁)[i] * χ[i,j] * v₁[j]
end

function _expect(χ::AbstractArray{T,4},v₁::AbstractArray{Complex{T},3}) where T<:Real
	@tullio out := conj(v₁)[i,ix,iy] * χ[i,j,ix,iy] * v₁[j,ix,iy] 
end

function _expect(χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},4}) where T<:Real
	@tullio out := conj(v₁)[i,ix,iy,iz] * χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz]
end

# first-order (linear) tensor-tensor muliplication

function _dot(A::AbstractArray{T,4},B::AbstractArray{T,4}) where T<:Real
	@tullio C[a,b,ix,iy] := A[a,j,ix,iy] * B[j,b,ix,iy]
end

function _dot(A::AbstractArray{T,4},B::AbstractArray{T,4},C::AbstractArray{T,4}) where T<:Real
	@tullio D[a,b,ix,iy] := A[a,j,ix,iy] * B[j,k,ix,iy] * C[k,b,ix,iy]
end

function _dot(A::AbstractArray{T,5},B::AbstractArray{T,5}) where T<:Real
	@tullio C[a,b,ix,iy,iz] := A[a,j,ix,iy,iz] * B[j,b,ix,iy,iz]
end

function _dot(A::AbstractArray{T,5},B::AbstractArray{T,5},C::AbstractArray{T,5}) where T<:Real
	@tullio D[a,b,ix,iy,iz] := A[a,j,ix,iy,iz] * B[j,k,ix,iy,iz] * C[k,b,ix,iy,iz]
end


# second-order nonlinear tensor muliplication
function _dot(χ::AbstractArray{T,3},v₁::AbstractVector,v₂::AbstractVector) where T<:Real
	@tullio v₃[i] := χ[i,j,k] * v₁[j] * v₂[k]
end

function _dot(χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},3},v₂::AbstractArray{Complex{T},3}) where T<:Real
	@tullio v₃[i,ix,iy] := χ[i,j,k,ix,iy] * v₁[j,ix,iy] * v₂[k,ix,iy]
end

function _dot(χ::AbstractArray{T,6},v₁::AbstractArray{Complex{T},4},v₂::AbstractArray{Complex{T},4}) where T<:Real
	@tullio v₃[i,ix,iy,iz] := χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz] * v₂[k,ix,iy,iz]
end

# third-order nonlinear tensor muliplication
function _dot(χ::AbstractArray{T,4},v₁::AbstractVector,v₂::AbstractVector,v₃::AbstractVector) where T<:Real
	@tullio v₄[i] := χ[i,j,k,l] * v₁[j] * v₂[k] * v₃[l]
end


"""
Tensor utilities
"""

function slice_inv(A::Array{T,4}) where {T}
    B = reinterpret(SArray{Tuple{3,3},T,2,9}, vec(A))
    C = map(inv, B)
    reshape(reinterpret(T, B), size(A))
end

function slice_inv(A::Array{T,5}) where {T}
    B = reinterpret(SArray{Tuple{3,3},T,2,9}, vec(A))
    C = map(inv, B)
    reshape(reinterpret(T, B), size(A))
end

"""
Spatially local linear algebra operations for flat (non-nested) arrays of 3-vectors and 3x3 tensors
"""
function _cross_x(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_x[ix,iy] := v1[2,ix,iy] * v2[3,ix,iy] - v1[3,ix,iy] * v2[2,ix,iy]
end
function _cross_y(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_y[ix,iy] := v1[1,ix,iy] * v2[3,ix,iy] - v1[3,ix,iy] * v2[1,ix,iy]
end
function _cross_z(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_z[ix,iy] := v1[1,ix,iy] * v2[2,ix,iy] - v1[2,ix,iy] * v2[1,ix,iy]
end
function _cross(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy] := v1[mod(i-2),ix,iy] * v2[mod(i-1),ix,iy] - v1[mod(i-1),ix,iy] * v2[mod(i-2),ix,iy] (i in 1:3)
end
function _cross(v1::TA,v2::TV) where {TA<:AbstractArray{<:Number,3},TV<:AbstractVector{<:Number}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy] := v1[mod(i-2),ix,iy] * v2[mod(i-1)] - v1[mod(i-1),ix,iy] * v2[mod(i-2)] (i in 1:3)
end
function _cross(v1::TV,v2::TA) where {TA<:AbstractArray{<:Number,3},TV<:AbstractVector{<:Number}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy] := v1[mod(i-2)] * v2[mod(i-1),ix,iy] - v1[mod(i-1)] * v2[mod(i-2),ix,iy] (i in 1:3)
end
function _sum_cross_x(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_x := v1[2,ix,iy] * v2[3,ix,iy] - v1[3,ix,iy] * v2[2,ix,iy]
end
function _sum_cross_y(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_y := v1[1,ix,iy] * v2[3,ix,iy] - v1[3,ix,iy] * v2[1,ix,iy]
end
function _sum_cross_z(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_z := v1[1,ix,iy] * v2[2,ix,iy] - v1[2,ix,iy] * v2[1,ix,iy]
end
function _sum_cross(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3[i] := v1[mod(i-2),ix,iy] * v2[mod(i-1),ix,iy] - v1[mod(i-1),ix,iy] * v2[mod(i-2),ix,iy] (i in 1:3)
end
function _cross_x(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_x[ix,iy,iz] := v1[2,ix,iy,iz] * v2[3,ix,iy,iz] - v1[3,ix,iy,iz] * v2[2,ix,iy,iz]
end
function _cross_y(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_y[ix,iy,iz] := v1[1,ix,iy,iz] * v2[3,ix,iy,iz] - v1[3,ix,iy,iz] * v2[1,ix,iy,iz]
end
function _cross_z(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3_z[ix,iy,iz] := v1[1,ix,iy,iz] * v2[2,ix,iy,iz] - v1[2,ix,iy,iz] * v2[1,ix,iy,iz]
end
function _cross(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy,iz] := v1[mod(i-2),ix,iy,iz] * v2[mod(i-1),ix,iy,iz] - v1[mod(i-1),ix,iy,iz] * v2[mod(i-2),ix,iy,iz] (i in 1:3)
end
function _cross(v1::TA,v2::TV) where {TA<:AbstractArray{<:Number,4},TV<:AbstractVector{<:Number}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy,iz] := v1[mod(i-2),ix,iy,iz] * v2[mod(i-1)] - v1[mod(i-1),ix,iy,iz] * v2[mod(i-2)] (i in 1:3)
end
function _cross(v1::TV,v2::TA) where {TA<:AbstractArray{<:Number,4},TV<:AbstractVector{<:Number}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio v3[i,ix,iy,iz] := v1[mod(i-2)] * v2[mod(i-1),ix,iy,iz] - v1[mod(i-1)] * v2[mod(i-2),ix,iy,iz] (i in 1:3)
end
function _sum_cross_x(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_x := v1[2,ix,iy,iz] * v2[3,ix,iy,iz] - v1[3,ix,iy,iz] * v2[2,ix,iy,iz]
end
function _sum_cross_y(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_y := v1[1,ix,iy,iz] * v2[3,ix,iy,iz] - v1[3,ix,iy,iz] * v2[1,ix,iy,iz]
end
function _sum_cross_z(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3_z := v1[1,ix,iy,iz] * v2[2,ix,iy,iz] - v1[2,ix,iy,iz] * v2[1,ix,iy,iz]
end
function _sum_cross(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @assert size(v1,1)==3
        @assert size(v2,1)==3
        @tullio sum_v3[i] := v1[mod(i-2),ix,iy,iz] * v2[mod(i-1),ix,iy,iz] - v1[mod(i-1),ix,iy,iz] * v2[mod(i-2),ix,iy,iz] (i in 1:3)
end

function rrule(::typeof(_cross),v₁,v₂)
        v₃ = _cross(v₁,v₂)
        function _cross_pullback(v̄₃)
                return NoTangent(), @thunk(_cross(conj(v₂),v̄₃)), @thunk(_cross(v̄₃,conj(v₁)))
        end
        return v₃, _cross_pullback
end

function rrule(::typeof(_sum_cross),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃ = _sum_cross(v₁,v₂)
        function _sum_cross_pullback(sum_v̄₃)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat(sum_v̄₃,outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat(sum_v̄₃,outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃, _sum_cross_pullback
end

function rrule(::typeof(_sum_cross_x),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_x = _sum_cross_x(v₁,v₂)
        function _sum_cross_x_pullback(sum_v̄₃_x)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([sum_v̄₃_x,0.,0.],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([sum_v̄₃_x,0.,0.],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_x, _sum_cross_x_pullback
end

function rrule(::typeof(_sum_cross_y),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_y = _sum_cross_y(v₁,v₂)
        function _sum_cross_y_pullback(sum_v̄₃_y)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([0.,sum_v̄₃_y,0.],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([0.,sum_v̄₃_y,0.],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_y, _sum_cross_y_pullback
end

function rrule(::typeof(_sum_cross_z),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_z = _sum_cross_z(v₁,v₂)
        function _sum_cross_z_pullback(sum_v̄₃_z)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([0.,0.,sum_v̄₃_z],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([0.,0.,sum_v̄₃_z],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_z, _sum_cross_z_pullback
end

# Outer products
function _outer(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
		@tullio A[i,j,ix,iy] := v1[i,ix,iy] * conj(v2)[j,ix,iy]
		# v2_star = conj(v2)
		# @tullio A[i,j,ix,iy] := v1[i,ix,iy] * v2_star[j,ix,iy]
end

function _outer(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,4},TA2<:AbstractArray{<:Number,4}}
        @tullio A[i,j,ix,iy,iz] := v1[i,ix,iy,iz] * conj(v2)[j,ix,iy,iz]
		# v2_star = conj(v2)
		# @tullio A[i,j,ix,iy,iz] := v1[i,ix,iy,iz] * v2_star[j,ix,iy,iz]
end
