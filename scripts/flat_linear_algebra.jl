# define cross product of fields of 3-vectors of type TA<:AbstractArray with size(TA,1)==3
using Tullio
import LinearAlgebra: cross
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

using ChainRulesCore: @thunk, NoTangent
using FillArrays
function ChainRulesCore.rrule(::typeof(_cross),v₁,v₂)
        v₃ = _cross(v₁,v₂)
        function _cross_pullback(v̄₃)
                return NoTangent(), @thunk(_cross(conj(v₂),v̄₃)), @thunk(_cross(v̄₃,conj(v₁)))
        end
        return v₃, _cross_pullback
end

function ChainRulesCore.rrule(::typeof(_sum_cross),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃ = _sum_cross(v₁,v₂)
        function _sum_cross_pullback(sum_v̄₃)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat(sum_v̄₃,outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat(sum_v̄₃,outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃, _sum_cross_pullback
end

function ChainRulesCore.rrule(::typeof(_sum_cross_x),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_x = _sum_cross_x(v₁,v₂)
        function _sum_cross_x_pullback(sum_v̄₃_x)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([sum_v̄₃_x,0.,0.],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([sum_v̄₃_x,0.,0.],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_x, _sum_cross_x_pullback
end

function ChainRulesCore.rrule(::typeof(_sum_cross_y),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_y = _sum_cross_y(v₁,v₂)
        function _sum_cross_y_pullback(sum_v̄₃_y)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([0.,sum_v̄₃_y,0.],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([0.,sum_v̄₃_y,0.],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_y, _sum_cross_y_pullback
end

function ChainRulesCore.rrule(::typeof(_sum_cross_z),v₁::AbstractArray{<:Number,N},v₂::AbstractArray{<:Number,N}) where N
        sum_v₃_z = _sum_cross_z(v₁,v₂)
        function _sum_cross_z_pullback(sum_v̄₃_z)
                return NoTangent(), @thunk(_cross(conj(v₂),repeat([0.,0.,sum_v̄₃_z],outer=(1,size(v₂)[2:N]...)))) , @thunk(_cross(repeat([0.,0.,sum_v̄₃_z],outer=(1,size(v₁)[2:N]...)),conj(v₁)))
        end
        return sum_v₃_z, _sum_cross_z_pullback
end

## tests
v₁ = randn(Float64,3,20,20)
v₂ = randn(Float64,3,20,20)
v₃, v₃_pb = pullback(_cross,v₁,v₂)
Δ = randn(Float64,3,20,20)
v₃_pb(Δ)

sum_v₃, sum_v₃_pb = pullback((vv1,vv2)->sum(_cross(vv1,vv2)),v₁,v₂)
Δ = 1.0
v̄₁,v̄₂ = sum_v₃_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->sum(_cross(vv1,vv2)),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂

sum_v₃, sum_v₃_pb = pullback(_sum_cross,v₁,v₂)
Δ = [1.0, 1.0, 1.0]
v̄₁,v̄₂ = sum_v₃_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->sum(_sum_cross(vv1,vv2)),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂


sum_v₃_z, sum_v₃_z_pb = pullback(_sum_cross_z,v₁,v₂)
Δ = 1.0
v̄₁,v̄₂ = sum_v₃_z_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->_sum_cross_z(vv1,vv2),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂

##
v₁ = randn(ComplexF64,3,20,20)
v₂ = randn(ComplexF64,3,20,20)
v₃, v₃_pb = pullback(_cross,v₁,v₂)
Δ = randn(ComplexF64,3,20,20)
v₃_pb(Δ)

sum_v₃, sum_v₃_pb = pullback((vv1,vv2)->abs2(sum(_cross(vv1,vv2))),v₁,v₂)
Δ = Complex(1.0)
v̄₁,v̄₂ = sum_v₃_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->abs2(sum(_cross(vv1,vv2))),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂

# sum_v₃, sum_v₃_pb = pullback(_sum_cross,v₁,v₂)
sum_v₃, sum_v₃_pb = pullback((vv1,vv2)->abs2(sum(_sum_cross(vv1,vv2))),v₁,v₂)
Δ = Complex(1.0) # Complex.([1.0, 1.0, 1.0])
v̄₁,v̄₂ = sum_v₃_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->abs2(sum(_sum_cross(vv1,vv2))),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂

# sum_v₃_z, sum_v₃_z_pb = pullback(_sum_cross_z,v₁,v₂)
sum_v₃_z, sum_v₃_z_pb = pullback((vv1,vv2)->abs2(_sum_cross_z(vv1,vv2)),v₁,v₂)
Δ = Complex(1.0)
v̄₁,v̄₂ = sum_v₃_z_pb(Δ)
v̄₁_FD, v̄₂_FD   =	FiniteDifferences.grad(central_fdm(3,1),(vv1,vv2)->abs2(_sum_cross_z(vv1,vv2)),v₁,v₂)
@assert v̄₁_FD ≈ v̄₁
@assert v̄₂_FD ≈ v̄₂
