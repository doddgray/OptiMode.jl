export HelmholtzMap, HelmholtzPreconditioner, ModeSolver, update_k, update_k!, replan_ffts!
export update_ε⁻¹, mag_m_n, mag_m_n!, mag_mn, mag_mn!, kx_ct, kx_tc, zx_tc, zx_ct, ∇ₖmag_m_n, ∇ₖmag_mn
export ε⁻¹_dot, ε⁻¹_dot_t, _M!, _P!, kx_ct!, kx_tc!, zx_tc!, kxinv_ct!
export kxinv_tc!, ed_approx!, HMₖH, HMH, tc, ct, eid!

# x̂ = SVector(1.,0.,0.)
# ŷ = SVector(0.,1.,0.)
# ẑ = SVector(0.,0.,1.)

"""
################################################################################
#																			   #
#			Function Definitions Implementing Non-Mutating Operators		   #
#																			   #
################################################################################
"""

# 3D

"""
    tc: v⃗ (transverse vector) → a⃗ (cartesian vector)
"""
function tc(H::AbstractArray{T,4},mn) where T<:Union{Real,Complex}
    @tullio h[a,ix,iy,iz] := H[b,ix,iy,iz] * mn[a,b,ix,iy,iz]
end

"""
    ct: a⃗ (cartesian vector) → v⃗ (transverse vector)
"""
function ct(h::AbstractArray{T,4},mn) where T<:Union{Real,Complex}
    @tullio H[a,ix,iy,iz] := h[b,ix,iy,iz] * mn[b,a,ix,iy,iz]
end

"""
    kx_tc: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_tc(H::AbstractArray{T,4},mn,mag) where T
	kxscales = [1.; -1.]
	kxinds = [2; 1]
    @tullio d[a,ix,iy,iz] := kxscales[b] * H[kxinds[b],ix,iy,iz] * mn[a,b,ix,iy,iz] * mag[ix,iy,iz] nograd=(kxscales,kxinds) # fastmath=false
	return d # -1im * d
end

"""
    kx_c2t: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_ct(e⃗::AbstractArray{T,4},mn,mag) where T
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[b,ix,iy,iz] := kxscales[b] * e⃗[a,ix,iy,iz] * mn[a,kxinds[b],ix,iy,iz] * mag[ix,iy,iz] nograd=(kxinds,kxscales) # fastmath=false
	return H # -1im * H
end

"""
    zx_t2c: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_tc(H::AbstractArray{T,4},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,ix,iy,iz] := zxscales[a] * H[b,ix,iy,iz] * mn[zxinds[a],b,ix,iy,iz] nograd=(zxscales,zxinds) # fastmath=false
	return zxH #-zxH # -1im * zxH
end

"""
    zx_c2t: v⃗ (transverse vector) = ẑ × a⃗ (cartesian vector)
"""
function zx_ct(e⃗::AbstractArray{T,4},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxe⃗[b,ix,iy,iz] := zxscales[a] * e⃗[a,ix,iy,iz] * mn[zxinds[a],b,ix,iy,iz] nograd=(zxscales,zxinds)  # fastmath=false
	return zxe⃗ #-zxe⃗ # -1im * zxe⃗
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗::AbstractArray{T,4},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,ix,iy,iz] :=  ε⁻¹[a,b,ix,iy,iz] * fft(d⃗,(2:4))[b,ix,iy,iz]  #fastmath=false
	return ifft(e⃗,(2:4))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗::AbstractArray{T,4},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,ix,iy,iz] :=  ε⁻¹[a,b,ix,iy,iz] * d⃗[b,ix,iy,iz]  #fastmath=false
end

function HMₖH(H::AbstractArray{Complex{T},4},ε⁻¹,mag,m,n)::T where T<:Real
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	-real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:4) ), real(flat(ε⁻¹))), (2:4)),mn,mag) ) )
end

function HMₖH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,3},m::AbstractArray{T,4},n::AbstractArray{T,4})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	HMₖH(Ha,ε⁻¹,mag,m,n)
end

function HMₖH(H::AbstractArray{Complex{T},4},ε⁻¹,mag,mn)::T where T<:Real
	-real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:4) ), real(flat(ε⁻¹))), (2:4)),mn,mag) ) )
end

function HMₖH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,3},mn::AbstractArray{T,5})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	HMₖH(Ha,ε⁻¹,mag,mn)
end

function HMH(H::AbstractArray{Complex{T},4},ε⁻¹,mag,m,n)::T where T<:Real
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	-real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:4) ), real(flat(ε⁻¹))), (2:4)),mn,mag) ) )
end

function HMH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,3},m::AbstractArray{T,4},n::AbstractArray{T,4})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	HMH(Ha,ε⁻¹,mag,m,n)
end

function HMH(H::AbstractArray{Complex{T},4},ε⁻¹,mag,mn)::T where T<:Real
	real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:4) ), real(flat(ε⁻¹))), (2:4)),mn,mag) ) )
end

function HMH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,3},m::AbstractArray{T,5})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	HMH(Ha,ε⁻¹,mag,mn)
end

"""
    tc: v⃗ (transverse vector) → a⃗ (cartesian vector)
"""
function tc(H::AbstractArray{T,3},mn) where T<:Union{Real,Complex}
    @tullio h[a,ix,iy] := H[b,ix,iy] * mn[a,b,ix,iy]
end

"""
    ct: a⃗ (cartesian vector) → v⃗ (transverse vector)
"""
function ct(h::AbstractArray{T,3},mn) where T<:Union{Real,Complex}
    @tullio H[a,ix,iy] := h[b,ix,iy] * mn[b,a,ix,iy]
end

"""
    kx_tc: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_tc(H::AbstractArray{T,3},mn,mag) where T
	kxscales = [1.; -1.]
	kxinds = [2; 1]
    @tullio d[a,ix,iy] := kxscales[b] * H[kxinds[b],ix,iy] * mn[a,b,ix,iy] * mag[ix,iy] nograd=(kxscales,kxinds) # fastmath=false
	# return d # -1im * d
	return d
end

"""
    kx_ct: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_ct(e⃗::AbstractArray{T,3},mn,mag) where T
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[b,ix,iy] := kxscales[b] * e⃗[a,ix,iy] * mn[a,kxinds[b],ix,iy] * mag[ix,iy] nograd=(kxinds,kxscales) # fastmath=false
	# return H # -1im * H
	return H
end

"""
    zx_tc: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_tc(H::AbstractArray{T,3},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,ix,iy] := zxscales[a] * H[b,ix,iy] * mn[zxinds[a],b,ix,iy] nograd=(zxscales,zxinds) # fastmath=false
	# return zxH #-zxH # -1im * zxH
	return  zxH
end

"""
    zx_ct: v⃗ (transverse vector) = ẑ × a⃗ (cartesian vector)
"""
function zx_ct(e⃗::AbstractArray{T,3},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxe⃗[b,ix,iy] := zxscales[a] * e⃗[a,ix,iy] * mn[zxinds[a],b,ix,iy] nograd=(zxscales,zxinds)  # fastmath=false
	# return zxe⃗ #-zxe⃗ # -1im * zxe⃗
	return  zxe⃗
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗::AbstractArray{T,3},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,ix,iy] :=  ε⁻¹[a,b,ix,iy] * fft(d⃗,(2:4))[b,ix,iy]  #fastmath=false
	return ifft(e⃗,(2:4))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗::AbstractArray{T,3},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,ix,iy] :=  ε⁻¹[a,b,ix,iy] * d⃗[b,ix,iy]  #fastmath=false
end

function HMₖH(H::AbstractArray{Complex{T},3},ε⁻¹,mag,m,n)::T where T<:Real
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	-real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) ) )
end

function HMₖH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,2},m::AbstractArray{T,3},n::AbstractArray{T,3})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(H,(2,Nx,Ny))
	HMₖH(Ha,ε⁻¹,mag,m,n)
end

function HMₖH(H::AbstractArray{Complex{T},3},ε⁻¹,mag,mn)::T where T<:Real
	-real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) ) )
end

function HMₖH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,2},mn::AbstractArray{T,4})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(H,(2,Nx,Ny))
	HMₖH(Ha,ε⁻¹,mag,mn)
end

function HMH(H::AbstractArray{Complex{T},3},ε⁻¹,mag,m,n)::T where T<:Real
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) ) )
end

function HMH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,2},m::AbstractArray{T,3},n::AbstractArray{T,3})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(H,(2,Nx,Ny))
	HMH(Ha,ε⁻¹,mag,m,n)
end

function HMH(H::AbstractArray{Complex{T},3},ε⁻¹,mag,mn)::T where T<:Real
	real( dot(H, kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) ) )
end

function HMH(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,2},mn::AbstractArray{T,4})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(H,(2,Nx,Ny))
	HMH(Ha,ε⁻¹,mag,mn)
end

"""
################################################################################
#																			   #
#			  Function Definitions Implementing Mutating Operators			   #
#																			   #
################################################################################
"""

# 3D _M! and _P! subroutines

function kx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},mn::AbstractArray{T,5},mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    # @assert size(Y) === size(X)
    # @assert size(d,4) == 3
    # @assert size(H,4) === 2
    @inbounds @fastmath for k ∈ axes(d,4), j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
	# @inbounds @fastmath for i ∈ axes(d,1), j ∈ axes(d,2), k ∈ axes(d,3), l in 0:0
		# scale = -mag[i,j,k]
		d[1+l,i,j,k] = ( H[1,i,j,k] * mn[1+l,2,i,j,k] - H[2,i,j,k] * mn[1+l,1,i,j,k] ) * -mag[i,j,k]
        d[2+l,i,j,k] = ( H[1,i,j,k] * mn[2+l,2,i,j,k] - H[2,i,j,k] * mn[2+l,1,i,j,k] ) * -mag[i,j,k]
        d[3+l,i,j,k] = ( H[1,i,j,k] * mn[3+l,2,i,j,k] - H[2,i,j,k] * mn[3+l,1,i,j,k] ) * -mag[i,j,k]
    end
    return d
end

function zx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},mn::AbstractArray{T,5})::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(d,4), j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
		d[1+l,i,j,k] = -H[1,i,j,k] * mn[2+l,1,i,j,k] - H[2,i,j,k] * mn[2+l,2,i,j,k]
        d[2+l,i,j,k] =  H[1,i,j,k] * mn[1+l,1,i,j,k] + H[2,i,j,k] * mn[1+l,2,i,j,k]
    end
    return d
end

function kx_ct!(H::AbstractArray{Complex{T},4},e::AbstractArray{Complex{T},4},mn::AbstractArray{T,5},mag::AbstractArray{T,3},Ninv::T)::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(H,4), j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = mag[i,j,k] * Ninv
        H[1+l,i,j,k] =  (	e[1+l,i,j,k] * mn[1+l,2,i,j,k] + e[2+l,i,j,k] * mn[2+l,2,i,j,k] + e[3+l,i,j,k] * mn[3+l,2,i,j,k]	) * -scale  # -mag[i,j,k] * Ninv
		H[2+l,i,j,k] =  (	e[1+l,i,j,k] * mn[1+l,1,i,j,k] + e[2+l,i,j,k] * mn[2+l,1,i,j,k] + e[3+l,i,j,k] * mn[3+l,1,i,j,k]	) * scale   # mag[i,j,k] * Ninv
    end
    return H
end

function eid!(e::AbstractArray{Complex{T},4},ε⁻¹,d::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0, h in 0:0
        e[1+h,i,j,k] =  ε⁻¹[1+h,1+l,i,j,k]*d[1+l,i,j,k] + ε⁻¹[2+h,1+l,i,j,k]*d[2+l,i,j,k] + ε⁻¹[3+h,1+l,i,j,k]*d[3+l,i,j,k]
        e[2+h,i,j,k] =  ε⁻¹[1+h,2+l,i,j,k]*d[1+l,i,j,k] + ε⁻¹[2+h,2+l,i,j,k]*d[2+l,i,j,k] + ε⁻¹[3+h,2+l,i,j,k]*d[3+l,i,j,k]
        e[3+h,i,j,k] =  ε⁻¹[1+h,3+l,i,j,k]*d[1+l,i,j,k] + ε⁻¹[2+h,3+l,i,j,k]*d[2+l,i,j,k] + ε⁻¹[3+h,3+l,i,j,k]*d[3+l,i,j,k]
    end
    return e
end

function eid!(e::AbstractArray{Complex{T},4},ε⁻¹::AbstractArray{TA,3},d::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where {T<:Real,TA<:SMatrix{3,3}}
    er = reinterpret(reshape,SVector{3,Complex{T}},e)
	dr = reinterpret(reshape,SVector{3,Complex{T}},d)
	map!(*,er,ε⁻¹,dr)
	# map!(*,er,ε⁻¹,er)
    return e
end

function kxinv_tc!(e::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},mn::AbstractArray{T,5},inv_mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
		e[1+l,i,j,k] = ( H[1,i,j,k] * mn[1+l,2,i,j,k] - H[2,i,j,k] * mn[1+l,1,i,j,k] ) * inv_mag[i,j,k]
        e[2+l,i,j,k] = ( H[1,i,j,k] * mn[2+l,2,i,j,k] - H[2,i,j,k] * mn[2+l,1,i,j,k] ) * inv_mag[i,j,k]
        e[3+l,i,j,k] = ( H[1,i,j,k] * mn[3+l,2,i,j,k] - H[2,i,j,k] * mn[3+l,1,i,j,k] ) * inv_mag[i,j,k]
    end
    return e
end

function kxinv_ct!(H::AbstractArray{Complex{T},4},d::AbstractArray{Complex{T},4},mn::AbstractArray{T,5},inv_mag::AbstractArray{T,3},N::T)::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(H,4), j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = inv_mag[i,j,k] * N
        H[1+l,i,j,k] =  (	d[1+l,i,j,k] * mn[1+l,2,i,j,k] + d[2+l,i,j,k] * mn[2+l,2,i,j,k] + d[3+l,i,j,k] * mn[3+l,2,i,j,k]	) * scale # inv_mag[i,j,k] * N
		H[2+l,i,j,k] =  (	d[1+l,i,j,k] * mn[1+l,1,i,j,k] + d[2+l,i,j,k] * mn[2+l,1,i,j,k] + d[3+l,i,j,k] * mn[3+l,1,i,j,k]	) * -scale # inv_mag[i,j,k] * N
    end
    return H
end

function ed_approx!(d::AbstractArray{Complex{T},4},ε_ave::AbstractArray{T,3},e::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @inbounds @fastmath for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
        d[1+l,i,j,k] =  ε_ave[i,j,k]*e[1+l,i,j,k]
        d[2+l,i,j,k] =  ε_ave[i,j,k]*e[2+l,i,j,k]
        d[3+l,i,j,k] =  ε_ave[i,j,k]*e[3+l,i,j,k]
    end
    return d
end

# 2D _M! and _P! subroutines

function kx_tc!(d::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},mn::AbstractArray{T,4},mag::AbstractArray{T,2})::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
	# @inbounds @fastmath for i ∈ axes(d,1), j ∈ axes(d,2), l in 0:0
		# scale = -mag[i,j,k]
		d[1+l,i,j] = ( H[1,i,j] * mn[1+l,2,i,j] - H[2,i,j] * mn[1+l,1,i,j] ) * -mag[i,j]
        d[2+l,i,j] = ( H[1,i,j] * mn[2+l,2,i,j] - H[2,i,j] * mn[2+l,1,i,j] ) * -mag[i,j]
        d[3+l,i,j] = ( H[1,i,j] * mn[3+l,2,i,j] - H[2,i,j] * mn[3+l,1,i,j] ) * -mag[i,j]
    end
    return d
end

function zx_tc!(d::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},mn::AbstractArray{T,4})::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
		d[1+l,i,j] = -H[1,i,j] * mn[2+l,1,i,j] - H[2,i,j] * mn[2+l,2,i,j]
        d[2+l,i,j] =  H[1,i,j] * mn[1+l,1,i,j] + H[2,i,j] * mn[1+l,2,i,j]
    end
    return d
end

function kx_ct!(H::AbstractArray{Complex{T},3},e::AbstractArray{Complex{T},3},mn::AbstractArray{T,4},mag::AbstractArray{T,2},Ninv::T)::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = mag[i,j] * Ninv
        H[1+l,i,j] =  (	e[1+l,i,j] * mn[1+l,2,i,j] + e[2+l,i,j] * mn[2+l,2,i,j] + e[3+l,i,j] * mn[3+l,2,i,j]	) * -scale  # -mag[i,j] * Ninv
		H[2+l,i,j] =  (	e[1+l,i,j] * mn[1+l,1,i,j] + e[2+l,i,j] * mn[2+l,1,i,j] + e[3+l,i,j] * mn[3+l,1,i,j]	) * scale   # mag[i,j] * Ninv
    end
    return H
end

function eid!(e::AbstractArray{Complex{T},3},ε⁻¹,d::AbstractArray{Complex{T},3}) where T<:Real
    @inbounds @fastmath for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0, h in 0:0
        e[1+h,i,j] =  ε⁻¹[1+h,1+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,1+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,1+l,i,j]*d[3+l,i,j]
        e[2+h,i,j] =  ε⁻¹[1+h,2+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,2+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,2+l,i,j]*d[3+l,i,j]
        e[3+h,i,j] =  ε⁻¹[1+h,3+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,3+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,3+l,i,j]*d[3+l,i,j]
    end
    return e
end

function eid!(e::AbstractArray{Complex{T},3},ε⁻¹::AbstractArray{TA,2},d::AbstractArray{Complex{T},3})::AbstractArray{Complex{T},3} where {T<:Real,TA<:SMatrix{3,3}}
    er = reinterpret(reshape,SVector{3,Complex{T}},e)
	dr = reinterpret(reshape,SVector{3,Complex{T}},d)
	map!(*,er,ε⁻¹,dr)
	# map!(*,er,ε⁻¹,er)
    return e
end

function kxinv_tc!(e::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},mn::AbstractArray{T,4},inv_mag::AbstractArray{T,2})::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
		e[1+l,i,j] = ( H[1,i,j] * mn[1+l,2,i,j] - H[2,i,j] * mn[1+l,1,i,j] ) * inv_mag[i,j]
        e[2+l,i,j] = ( H[1,i,j] * mn[2+l,2,i,j] - H[2,i,j] * mn[2+l,1,i,j] ) * inv_mag[i,j]
        e[3+l,i,j] = ( H[1,i,j] * mn[3+l,2,i,j] - H[2,i,j] * mn[3+l,1,i,j] ) * inv_mag[i,j]
    end
    return e
end

function kxinv_ct!(H::AbstractArray{Complex{T},3},d::AbstractArray{Complex{T},3},mn::AbstractArray{T,4},inv_mag::AbstractArray{T,2},N::T)::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = inv_mag[i,j] * N
        H[1+l,i,j] =  (	d[1+l,i,j] * mn[1+l,2,i,j] + d[2+l,i,j] * mn[2+l,2,i,j] + d[3+l,i,j] * mn[3+l,2,i,j]	) * scale # inv_mag[i,j] * N
		H[2+l,i,j] =  (	d[1+l,i,j] * mn[1+l,1,i,j] + d[2+l,i,j] * mn[2+l,1,i,j] + d[3+l,i,j] * mn[3+l,1,i,j]	) * -scale # inv_mag[i,j] * N
    end
    return H
end

function ed_approx!(d::AbstractArray{Complex{T},3},ε_ave::AbstractArray{T,2},e::AbstractArray{Complex{T},3})::AbstractArray{Complex{T},3} where T<:Real
    @inbounds @fastmath for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
        d[1+l,i,j] =  ε_ave[i,j]*e[1+l,i,j]
        d[2+l,i,j] =  ε_ave[i,j]*e[2+l,i,j]
        d[3+l,i,j] =  ε_ave[i,j]*e[3+l,i,j]
    end
    return d
end

# _M! and _P!

function _P!(Hout::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N}, ε_ave::AbstractArray{T},
	mn::AbstractArray{T}, inv_mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},N} where {T<:Real,N}
	kxinv_tc!(e,Hin,mn,inv_mag);
	mul!(e,𝓕⁻¹!,e);
    ed_approx!(d,ε_ave,e);
    mul!(d,𝓕!,d);
    kxinv_ct!(Hout,d,mn,inv_mag,Ninv)
end

function _M!(Hout::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N}, ε⁻¹,
	mn::AbstractArray{T}, mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},N} where {T<:Real,N}
    kx_tc!(d,Hin,mn,mag);
    mul!(d,𝓕!,d);
    eid!(e,ε⁻¹,d);
    mul!(e,𝓕⁻¹!,e);
    kx_ct!(Hout,e,mn,mag,Ninv)
end

"""
################################################################################
#																			   #
#			  Utility Function Definitions Needed for Constructors 			   #
#																			   #
################################################################################
"""

function mag_m_n!(mag,m,n,k⃗::SVector{3,T},g⃗) where T <: Real
	# for iz ∈ axes(g⃗,3), iy ∈ axes(g⃗,2), ix ∈ axes(g⃗,1) #, l in 0:0
	local ẑ = SVector{3}(0,0,1)
	local ŷ = SVector{3}(0,1,0)
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		# @inbounds n[i] =  !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return mag,m,n
end

mag_m_n!(mag,m,n,kz::T,g⃗) where T <: Real = mag_m_n!(mag,m,n,SVector{3,T}(0.,0.,kz),g⃗)

function mag_m_n(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
	# for iz ∈ axes(g⃗s,3), iy ∈ axes(g⃗s,2), ix ∈ axes(g⃗s,1) #, l in 0:0
    T = promote_type(T1,T2)
	local ẑ = SVector{3,T}(0.,0.,1.)
	local ŷ = SVector{3,T}(0.,1.,0.)
	n = Array{SVector{3,T}}(undef,size(g⃗s))
	m = Array{SVector{3,T}}(undef,size(g⃗s))
	mag = Array{T}(undef,size(g⃗s))
	@fastmath @inbounds for i ∈ eachindex(g⃗s)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		# @inbounds n[i] =   !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return mag, m, n
end

function mag_m_n(kmag::T,g⃗::AbstractArray{SVector{3,T2}};k̂=SVector(0.,0.,1.)) where {T<:Real,T2<:Real}
	k⃗ = kmag * k̂
	mag_m_n(k⃗,g⃗)
end

mag_m_n(k::Real,grid::Grid) = mag_m_n(k, g⃗(grid))


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

function rrule(::typeof(mag_m_n),k⃗::SVector{3,T},g⃗::AbstractArray{SVector{3,T}}) where T <: Real
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	n_buf = similar(g⃗)
	m_buf = similar(g⃗)
	kpg_buf = similar(g⃗)
	mag_buf = Array{T}(undef,size(g⃗))
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg_buf[i] = k⃗ - g⃗[i]
		@inbounds mag_buf[i] = norm(kpg_buf[i])
		# @inbounds n_buf[i] =   ( ( abs2(kpg_buf[i][1]) + abs2(kpg_buf[i][2]) ) > 0. ) ?  normalize( cross( ẑ, kpg_buf[i] ) ) : SVector(-1.,0.,0.) # ŷ
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
		return ( NoTangent(), k̄*dk̂, ZeroTangent() )
	end
    return (mag_m⃗_n⃗ , mag_m_n_pullback)
end

function mag_mn(k::SVector{3,T1},g::Grid{ND,T2}) where {T1<:Real,ND,T2<:Real}
	mag, m⃗, n⃗ = mag_m_n(k,g⃗(g))
	# mn = copy(reshape(reinterpret(T1,hcat.(m⃗,n⃗)),(3,2,size(g)...)))
	# return mag, HybridArray{Tuple{3,2,Dynamic(),Dynamic(),Dynamic()},T1}(copy(reshape(reinterpret(T1,hcat.(m⃗,n⃗)),(3,2,size(g)...))))
	m = reshape(reinterpret(reshape,T1,m⃗), (3,1,size(g)...))
    n = reshape(reinterpret(reshape,T1,n⃗), (3,1,size(g)...))
	mn = hcat(m,n)
	return mag, mn
end

function mag_mn(k::SVector{3,T1},g::AbstractArray{SVector{3,T2},3}) where {T1<:Real,T2<:Real}
	mag, m⃗, n⃗ = mag_m_n(k,g)
	m = reshape(reinterpret(reshape,T1,m⃗), (3,1,size(g)...))
    n = reshape(reinterpret(reshape,T1,n⃗), (3,1,size(g)...))
	mn = hcat(m,n)
	return mag, mn
end

function mag_mn(k::SVector{3,T1},g::AbstractArray{SVector{3,T2},2}) where {T1<:Real,T2<:Real}
	mag, m⃗, n⃗ = mag_m_n(k,g)
	m = reshape(reinterpret(reshape,T1,m⃗), (3,1,size(g)...))
    n = reshape(reinterpret(reshape,T1,n⃗), (3,1,size(g)...))
	mn = hcat(m,n)
	return mag, mn
end

function mag_mn(kmag::T,g::TG;k̂=SVector(0.,0.,1.)) where {T<:Real,TG}
	return mag_mn(kmag*k̂,g)
end

function mag_mn!(mag,mn::AbstractArray{T1,NDp2},k⃗::SVector{3,T2},g⃗) where {T1<:Real,T2<:Real,NDp2}
	local ẑ = SVector{3}(0.,0.,1.)
	local ŷ = SVector{3}(0.,1.,0.)
	kpg = zero(k⃗)
	@fastmath @inbounds for I ∈ CartesianIndices(g⃗)
		@inbounds kpg = k⃗ - g⃗[I]
		@inbounds mag[I] = norm(kpg)
		n = ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		m = normalize( cross( n, kpg ) )
		@inbounds mn[1:3,2,I] .= n
		@inbounds mn[1:3,1,I] .= m
	end
	# return mag,m,n
	return mag, mn
end

mag_mn!(mag,mn,kmag::T,g⃗;k̂=SVector(0.,0.,1.)) where T <: Real = mag_mn!(mag,mn,k̂*kmag,g⃗)

"""
(māg,m̄n̄) → k̄ map

assumes mn and mn̄ have axes/sizes:
dim_idx=1:3, mn_idx=1:2, x_idx=1:Nx, y_idx=1:Ny
"""
function ∇ₖmag_mn(māg::AbstractArray{T1,2},mn̄,mag::AbstractArray{T2,2},mn;dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Number}
	m = view(mn,:,1,:,:)
	n = view(mn,:,2,:,:)
	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄)[i,imn,ix,iy] * mn[mod(i-2),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄)[i,imn,ix,iy] * mn[mod(i-1),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

"""
(māg,m̄n̄) → k̄ map

assumes mn and mn̄ have axes/sizes:
dim_idx=1:3, mn_idx=1:2, x_idx=1:Nx, y_idx=1:Ny, z_idx=1:Nz
"""
function ∇ₖmag_mn(māg::AbstractArray{T1,3},mn̄,mag::AbstractArray{T2,3},mn;dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Number}
	m = view(mn,:,1,:,:,:)
	n = view(mn,:,2,:,:,:)
	@tullio kp̂g_over_mag[i,ix,iy,iz] := m[mod(i-2),ix,iy,iz] * n[mod(i-1),ix,iy,iz] / mag[ix,iy,iz] - m[mod(i-1),ix,iy,iz] * n[mod(i-2),ix,iy,iz] / mag[ix,iy,iz] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy,iz] * mag[ix,iy,iz] * kp̂g_over_mag[j,ix,iy,iz] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄)[i,imn,ix,iy,iz] * mn[mod(i-2),imn,ix,iy,iz] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy,iz] + conj(mn̄)[i,imn,ix,iy,iz] * mn[mod(i-1),imn,ix,iy,iz] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy,iz] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

"""
(māg,m̄n̄) → k̄ map

assumes mn and mn̄ have axes/sizes:
dim_idx=1:3, mn_idx=1:2, x_idx=1:Nx, y_idx=1:Ny

Method with `kpg` (= k⃗ .+ g⃗(grid)) input for pullback performance
"""
function ∇ₖmag_mn(māg::AbstractArray{T1,2},mn̄,mag::AbstractArray{T2,2},mn,kpg;dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Number}
	@tullio kp̂g_over_mag[i,ix,iy] := kpg[i,ix,iy] / mag[ix,iy] 
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄)[i,imn,ix,iy] * mn[mod(i-2),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄)[i,imn,ix,iy] * mn[mod(i-1),imn,ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

"""
(māg,m̄n̄) → k̄ map

assumes mn and mn̄ have axes/sizes:
dim_idx=1:3, mn_idx=1:2, x_idx=1:Nx, y_idx=1:Ny, z_idx=1:Nz

Method with `kpg` (= k⃗ .+ g⃗(grid)) input for pullback performance
"""
function ∇ₖmag_mn(māg::AbstractArray{T1,3},mn̄,mag::AbstractArray{T2,3},mn,kpg;dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Number}
	@tullio kp̂g_over_mag[i,ix,iy,iz] := kpg[i,ix,iy,iz]  / mag[ix,iy,iz] 
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy,iz] * mag[ix,iy,iz] * kp̂g_over_mag[j,ix,iy,iz] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄)[i,imn,ix,iy,iz] * mn[mod(i-2),imn,ix,iy,iz] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy,iz] + conj(mn̄)[i,imn,ix,iy,iz] * mn[mod(i-1),imn,ix,iy,iz] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy,iz] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

function rrule(::typeof(mag_mn),k⃗::SVector{3,T1},g::AbstractArray{<:SVector{3,T2}};dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Real}
	local ẑ = SVector{3}(0.,0.,1.)
	local ŷ = SVector{3}(0.,1.,0.)
	grid_size = size(g)
	m_buf = Array{SVector{3,T1}}(undef,grid_size)
	n_buf = Array{SVector{3,T1}}(undef,grid_size)
	kpg_buf = Array{SVector{3,T1}}(undef,grid_size)
	mag_buf = Array{T1}(undef,grid_size)
	@fastmath @inbounds for i ∈ eachindex(g)
		@inbounds kpg_buf[i] = k⃗ - g[i]
		@inbounds mag_buf[i] = norm(kpg_buf[i])
		@inbounds n_buf[i] =   ( ( abs2(kpg_buf[i][1]) + abs2(kpg_buf[i][2]) ) > 0. ) ?  normalize( cross( ẑ, kpg_buf[i] ) ) :  ŷ
		@inbounds m_buf[i] =  normalize( cross( n_buf[i], kpg_buf[i] )  )
	end
	mag = copy(mag_buf)
	m⃗	=	copy(m_buf)
	n⃗	= copy(n_buf)
	# kp⃗g = copy(kpg_buf)
	m = reshape(reinterpret(reshape,T1,m⃗), (3,1,grid_size...))
	n = reshape(reinterpret(reshape,T1,n⃗), (3,1,grid_size...))
	# kpg = reshape(reinterpret(reshape,T1,kp⃗g), (3,grid_size...))
	mn = hcat(m,n)
	
	mag_mn_pullback(ΔΩ) = let mag=mag, mn=mn, dk̂=dk̂ # , kpg=kpg
		māg,mn̄ = ΔΩ
		# k̄ = ∇ₖmag_mn(māg,mn̄,mag,mn,kpg;dk̂)
		k̄ = ∇ₖmag_mn(māg,mn̄,mag,mn;dk̂)
		return ( NoTangent(), k̄*dk̂, ZeroTangent() )
	end
    return ((mag, mn) , mag_mn_pullback)
end

function rrule(::typeof(mag_mn),kmag::T1,g::AbstractArray{<:SVector{3,T2}};dk̂=SVector{3}(0.,0.,1.)) where {T1<:Real,T2<:Real}
	local ẑ = SVector{3}(0.,0.,1.)
	local ŷ = SVector{3}(0.,1.,0.)
	k⃗ = kmag * dk̂
	grid_size = size(g)
	m_buf = Array{SVector{3,T1}}(undef,grid_size)
	n_buf = Array{SVector{3,T1}}(undef,grid_size)
	kpg_buf = Array{SVector{3,T1}}(undef,grid_size)
	mag_buf = Array{T1}(undef,grid_size)
	@fastmath @inbounds for i ∈ eachindex(g)
		@inbounds kpg_buf[i] = k⃗ - g[i]
		@inbounds mag_buf[i] = norm(kpg_buf[i])
		@inbounds n_buf[i] =   ( ( abs2(kpg_buf[i][1]) + abs2(kpg_buf[i][2]) ) > 0. ) ?  normalize( cross( ẑ, kpg_buf[i] ) ) :  ŷ
		@inbounds m_buf[i] =  normalize( cross( n_buf[i], kpg_buf[i] )  )
	end
	mag = copy(mag_buf)
	m⃗	=	copy(m_buf)
	n⃗	= copy(n_buf)
	# kp⃗g = copy(kpg_buf)
	m = reshape(reinterpret(reshape,T1,m⃗), (3,1,grid_size...))
	n = reshape(reinterpret(reshape,T1,n⃗), (3,1,grid_size...))
	# kpg = reshape(reinterpret(reshape,T1,kp⃗g), (3,grid_size...))
	mn = hcat(m,n)
	
	mag_mn_pullback(ΔΩ) = let mag=mag, mn=mn, dk̂=dk̂ # , kpg=kpg
		māg,mn̄ = ΔΩ
		# k̄ = ∇ₖmag_mn(māg,mn̄,mag,mn,kpg;dk̂)
		k̄ = ∇ₖmag_mn(māg,mn̄,mag,mn;dk̂)
		return ( NoTangent(), k̄, ZeroTangent() )
	end
    return ((mag, mn) , mag_mn_pullback)
end

"""
################################################################################
#																			   #
#							  Struct Definitions 							   #
#																			   #
################################################################################
"""

mutable struct HelmholtzMap{ND,T,NDp1,NDp2} <: LinearMap{T}
    k⃗::SVector{3,T}
	Nx::Int
	Ny::Int
	Nz::Int
	N::Int
	Ninv::T
	g⃗::Array{SVector{3,T},ND}
	mag::Array{T,ND}
	mn::Array{T,NDp2}
    e::Array{Complex{T},NDp1}
    d::Array{Complex{T},NDp1}
    𝓕!::FFTW.cFFTWPlan{Complex{T}, -1, true, NDp1, UnitRange{Int64}}
	𝓕⁻¹!::FFTW.cFFTWPlan{Complex{T}, 1, true, NDp1, UnitRange{Int64}}
	𝓕::FFTW.cFFTWPlan{Complex{T}, -1, false, NDp1, UnitRange{Int64}}
	𝓕⁻¹::FFTW.cFFTWPlan{Complex{T}, 1, false, NDp1, UnitRange{Int64}}
	ε⁻¹::Array{T,NDp2}
	ε_ave::Array{T,ND}  # for preconditioner
	inv_mag::Array{T,ND} # for preconditioner
	shift::T
end

mutable struct HelmholtzPreconditioner{ND,T,NDp1,NDp2} <: LinearMap{T}
	M̂::HelmholtzMap{ND,T,NDp1,NDp2}
end

mutable struct ModeSolver{ND,T,NDp1,NDp2}
	grid::Grid{ND,T}
	M̂::HelmholtzMap{ND,T,NDp1,NDp2}
	P̂::HelmholtzPreconditioner{ND,T,NDp1,NDp2}
	H⃗::Matrix{Complex{T}}
	ω²::Vector{Complex{T}}
	∂ω²∂k::Vector{T}
end

"""
################################################################################
#																			   #
#							  Constructor Methods 							   #
#																			   #
################################################################################
"""

function HelmholtzMap(k⃗::AbstractVector{T}, ε⁻¹, gr::Grid{3,T}; shift=0. ) where {ND,T<:Real}
	g⃗s = g⃗(gr)
	# mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
	mag, mn = mag_mn(k⃗,g⃗s)
	d0 = randn(Complex{T}, (3,size(gr)...))
	fftax = _fftaxes(gr)
	return HelmholtzMap{3,T,4,5}(
			SVector{3,T}(k⃗),
			gr.Nx,
			gr.Ny,
			gr.Nz,
			N(gr),
			1. / N(gr),
			g⃗s,
			mag,
			mn,
			copy(d0),
			copy(d0),
			plan_fft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			plan_fft(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			ε⁻¹,
			[ 3. * inv(sum(diag(ε⁻¹[:,:,I]))) for I in eachindex(gr)], #[ 3. * inv(sum(diag(einv))) for einv in ε⁻¹],
			[ inv(mm) for mm in mag ], # inverse |k⃗+g⃗| magnitudes for precond. ops
			shift,
		)
end

function HelmholtzMap(k⃗::AbstractVector{T}, ε⁻¹, gr::Grid{2,T}; shift=0. ) where {ND,T<:Real}
	g⃗s = g⃗(gr)
	# mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
	mag, mn = mag_mn(k⃗,g⃗s)
	d0 = randn(Complex{T}, (3,size(gr)...))
	fftax = _fftaxes(gr)
	return HelmholtzMap{2,T,3,4}(
			SVector{3,T}(k⃗),
			gr.Nx,
			gr.Ny,
			gr.Nz,
			N(gr),
			1. / N(gr),
			g⃗s,
			mag,
			mn,
			copy(d0),
			copy(d0),
			plan_fft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			plan_fft(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			ε⁻¹,
			[ 3. * inv(sum(diag(ε⁻¹[:,:,I]))) for I in eachindex(gr)], # [ 3. * inv(sum(diag(einv))) for einv in ε⁻¹],
			[ inv(mm) for mm in mag ], # inverse |k⃗+g⃗| magnitudes for precond. ops
			shift,
		)
end

function HelmholtzMap(kz::T, ε⁻¹, gr::Grid; shift=0.) where {T<:Real}
	HelmholtzMap(SVector{3,T}(0.,0.,kz), ε⁻¹, gr::Grid; shift)
end

# function ModeSolver(k⃗::SVector{3,T}, geom::Geometry, grid::Grid{ND}; nev=1, tol=1e-8, maxiter=3000, ω₀=1/1.55, constraint=nothing,) where {ND,T<:Real}
function ModeSolver(k⃗::SVector{3,T}, ε⁻¹, grid::Grid{2}; nev=1, tol=1e-8, maxiter=3000, ω₀=1/1.55, constraint=nothing,) where {T<:Real}

	M̂ = HelmholtzMap(k⃗, ε⁻¹, grid)
	P̂ = HelmholtzPreconditioner(M̂)

	ModeSolver{2,T,3,4}(
		# geom,
		# mats,
		grid,
		M̂,
		P̂,
		# eigs_itr,
		randn(Complex{T},2*N(grid),nev), #eigs_itr.XBlocks.block,
		zeros(Complex{T},nev),
		zeros(T,nev),
	)
end


function ModeSolver(k⃗::SVector{3,T}, ε⁻¹, grid::Grid{3}; nev=1, tol=1e-8, maxiter=3000, ω₀=1/1.55, constraint=nothing,) where {T<:Real}
	# run inital smoothing sub-processes
	# ε⁻¹ = εₛ⁻¹( (1. / ω₀), geom, grid)
	M̂ = HelmholtzMap(k⃗, ε⁻¹, grid)
	P̂ = HelmholtzPreconditioner(M̂)

	ModeSolver{3,T,4,5}(
		# geom,
		# mats,
		grid,
		M̂,
		P̂,
		# eigs_itr,
		randn(Complex{T},2*N(grid),nev), #eigs_itr.XBlocks.block,
		zeros(Complex{T},nev),
		zeros(T,nev),
	)
end

function ModeSolver(kz::T, ε⁻¹, grid::Grid{ND}; nev=1, tol=1e-8, maxiter=3000,constraint=nothing,) where {ND,T<:Real}
	ModeSolver(SVector{3,T}(0.,0.,kz), ε⁻¹, grid; nev, tol, maxiter, constraint)
end


"""
################################################################################
#																			   #
#							  	Struct Methods 								   #
#																			   #
################################################################################
"""

function (M̂::HelmholtzMap{2,T,3,4})(Hout::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3}) where T<:Real
	_M!(Hout,Hin,M̂.e,M̂.d,M̂.ε⁻¹,M̂.mn,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv)
end

function (M̂::HelmholtzMap{3,T,4,5})(Hout::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4}) where T<:Real
	_M!(Hout,Hin,M̂.e,M̂.d,M̂.ε⁻¹,M̂.mn,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv)
end

function (M̂::HelmholtzMap{2,T,3,4})(Hout::AbstractVector{Complex{T}}, Hin::AbstractVector{Complex{T}}) where T<:Real
	@inbounds Hin_arr = reshape(Hin,(2,M̂.Nx,M̂.Ny))
	@inbounds Hout_arr = reshape(Hout,(2,M̂.Nx,M̂.Ny))
	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.mn,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv) )
end

function (M̂::HelmholtzMap{3,T,4,5})(Hout::AbstractVector{Complex{T}}, Hin::AbstractVector{Complex{T}}) where T<:Real
	@inbounds Hin_arr = reshape(Hin,(2,M̂.Nx,M̂.Ny,M̂.Nz))
	@inbounds Hout_arr = reshape(Hout,(2,M̂.Nx,M̂.Ny,M̂.Nz))
	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.mn,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv) )
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractArray{T,3}, Hin::AbstractArray{T,3}) where T<:Union{Real, Complex}
	_P!(Hout,Hin,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.mn,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv)
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractArray{T,4}, Hin::AbstractArray{T,4}) where T<:Union{Real, Complex}
	_P!(Hout,Hin,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.mn,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv)
end

function (P̂::HelmholtzPreconditioner{2})(Hout::AbstractVector{T}, Hin::AbstractVector{T}) where T<:Union{Real, Complex}
	@inbounds Hin_arr = reshape(Hin,(2,P̂.M̂.Nx,P̂.M̂.Ny))
	@inbounds Hout_arr = reshape(Hout,(2,P̂.M̂.Nx,P̂.M̂.Ny))
	vec( _P!(Hout_arr,Hin_arr,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.mn,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv) )
end

function (P̂::HelmholtzPreconditioner{3})(Hout::AbstractVector{T}, Hin::AbstractVector{T}) where T<:Union{Real, Complex}
	@inbounds Hin_arr = reshape(Hin,(2,P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz))
	@inbounds Hout_arr = reshape(Hout,(2,P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz))
	vec( _P!(Hout_arr,Hin_arr,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.mn,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv) )
end

function Base.:(*)(M̂::HelmholtzMap, x::AbstractVector)
    #length(x) == A.N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(M̂), eltype(x)), 2*M̂.N)
    M̂(y, x)
end

function _unsafe_mul!(y::AbstractVecOrMat, M̂::HelmholtzMap, x::AbstractVector)
    M̂(y, x)
end

function Base.:(*)(P̂::HelmholtzPreconditioner, x::AbstractVector)
    #length(x) == A.N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(P̂.M̂), eltype(x)), 2*P̂.M̂.N)
    P̂(y, x)
end

function _unsafe_mul!(y::AbstractVecOrMat, P̂::HelmholtzPreconditioner, x::AbstractVector)
    P̂(y, x)
end

# replan_ffts! methods (ModeSolver FFT plans should be re-created for backwards pass during AD)
function replan_ffts!(ms::ModeSolver{3,T}) where T<:Real
	ms.M̂.𝓕! = plan_fft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹! = plan_bfft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕 = plan_fft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹ = plan_bfft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
end

function replan_ffts!(ms::ModeSolver{2,T}) where T<:Real
	ms.M̂.𝓕! = plan_fft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹! = plan_bfft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕 = plan_fft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹ = plan_bfft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
end

# Update k methods

function update_k(M̂::HelmholtzMap{ND,T,NDp1,NDp2},k⃗::SVector{3,T2}) where {ND,T<:Real,NDp1,NDp2,T2<:Real}
	mag, mn = mag_mn(k⃗,M̂.g⃗)
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
	M̂.mn = mn
	M̂.k⃗ = k⃗
end

function update_k(M̂::HelmholtzMap{ND,T1,NDp1,NDp2},kz::T2) where {ND,T1<:Real,NDp1,NDp2,T2<:Real}
	update_k(M̂,SVector{3,T}(0.,0.,kz))
end

update_k(ms::ModeSolver,k) = update_k(ms.M̂,k)

function update_k!(M̂::HelmholtzMap{ND,T1,NDp1,NDp2},k⃗::SVector{3,T2}) where {ND,T1<:Real,NDp1,NDp2,T2<:Real}
	# mag_m_n!(M̂.mag,M̂.m⃗,M̂.n⃗,k⃗,M̂.g⃗)
	M̂.k⃗ = k⃗
	mag_mn!(M̂.mag,M̂.mn,M̂.k⃗,M̂.g⃗)
	M̂.inv_mag = [inv(mm) for mm in M̂.mag]

end

function update_k!(M̂::HelmholtzMap{ND,T1,NDp1,NDp2},kz::T2) where {ND,T1<:Real,NDp1,NDp2,T2<:Real}
	update_k!(M̂,SVector{3,T2}(0.,0.,kz))
end

update_k!(ms::ModeSolver,k) = update_k!(ms.M̂,k)

# Update ε⁻¹ methods

function update_ε⁻¹(M̂::HelmholtzMap{2,T,NDp1,NDp2},ε⁻¹) where {T<:Real,NDp1,NDp2}
	@assert size(M̂.ε⁻¹) == size(ε⁻¹)
	M̂.ε⁻¹ = ε⁻¹
	M̂.ε_ave = [ 3. * inv(sum(diag(ε⁻¹[:,:,ix,iy]))) for ix in axes(ε⁻¹,3), iy in axes(ε⁻¹,4)]
end

function update_ε⁻¹(M̂::HelmholtzMap{3,T,NDp1,NDp2},ε⁻¹) where {T<:Real,NDp1,NDp2}
	@assert size(M̂.ε⁻¹) == size(ε⁻¹)
	M̂.ε⁻¹ = ε⁻¹
	M̂.ε_ave = [ 3. * inv(sum(diag(ε⁻¹[:,:,ix,iy,iz]))) for ix in axes(ε⁻¹,3), iy in axes(ε⁻¹,4), iz in axes(ε⁻¹,5)]
end

function update_ε⁻¹(ms::ModeSolver{2,T},ε⁻¹) where {T<:Real}
	@assert size(ms.M̂.ε⁻¹) == size(ε⁻¹)
	ms.M̂.ε⁻¹ = ε⁻¹
	ms.M̂.ε_ave = [ 3. * inv(sum(diag(ε⁻¹[:,:,ix,iy]))) for ix in 1:ms.grid.Nx, iy in 1:ms.grid.Ny]
end

function update_ε⁻¹(ms::ModeSolver{3,T},ε⁻¹) where {T<:Real}
	@assert size(ms.M̂.ε⁻¹) == size(ε⁻¹)
	ms.M̂.ε⁻¹ = ε⁻¹
	ms.M̂.ε_ave = [ 3. * inv(sum(diag(ε⁻¹[:,:,ix,iy,iz]))) for ix in 1:ms.grid.Nx, iy in 1:ms.grid.Ny, iz in 1:ms.grid.Nz]
end

# property methods
Base.size(A::HelmholtzMap) = (2*A.N, 2*A.N)
Base.size(A::HelmholtzMap,d::Int) = 2*A.N
Base.eltype(A::HelmholtzMap{ND,T,NDp1,NDp2}) where {ND,T<:Real,NDp1,NDp2}  = Complex{T}
LinearAlgebra.issymmetric(A::HelmholtzMap) = false # A._issymmetric
LinearAlgebra.ishermitian(A::HelmholtzMap) = true # A._ishermitian
LinearAlgebra.isposdef(A::HelmholtzMap)    = true # A._isposdef
ismutating(A::HelmholtzMap) = true # A._ismutating
#_ismutating(f) = first(methods(f)).nargs == 3
Base.size(A::HelmholtzPreconditioner) = (2*A.M̂.N, 2*A.M̂.N)
Base.size(A::HelmholtzPreconditioner,d::Int) = 2*A.M̂.N
Base.eltype(A::HelmholtzPreconditioner) = eltype(A.M̂)
LinearAlgebra.issymmetric(A::HelmholtzPreconditioner) = true # A._issymmetric
LinearAlgebra.ishermitian(A::HelmholtzPreconditioner) = true # A._ishermitian
LinearAlgebra.isposdef(A::HelmholtzPreconditioner)    = true # A._isposdef
ismutating(A::HelmholtzPreconditioner) = true # A._ismutating

import Base: *, transpose, adjoint
function Base.:(*)(M::HelmholtzMap,X::Matrix)
	#if isequal(size(M),size(X)) # size check?
	ncolsX = size(X)[2]
	# @assert ncolsX == size(M)[1]
	Y = similar(X)
	for i in 1:ncolsX
		@views Y[:,i] = M * X[:,i]
	end
	return Y
end


function LinearAlgebra.mul!(y::AbstractVecOrMat, M̂::HelmholtzMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, M̂, x)
	M̂(y, x)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, P̂::HelmholtzPreconditioner, x::AbstractVector)
    LinearMaps.check_dim_mul(y, P̂, x)
	P̂(y, x)
end

Base.adjoint(A::HelmholtzMap) = A
Base.transpose(P̂::HelmholtzPreconditioner) = P̂
LinearAlgebra.ldiv!(c,P̂::HelmholtzPreconditioner,b) = mul!(c,P̂,b) # P̂(c, b) #
LinearAlgebra.ldiv!(P̂::HelmholtzPreconditioner,b) = mul!(b,P̂,b)



