export HelmholtzMap, HelmholtzPreconditioner, ModeSolver, update_k, update_k!
export update_ε⁻¹, ε⁻¹_ω, mag_m_n, mag_m_n2, mag_m_n!, kx_ct, kx_tc, zx_tc, zx_ct
export ε⁻¹_dot, ε⁻¹_dot_t, _M!, _P!, kx_ct!, kx_tc!, zx_tc!, kxinv_ct!
export kxinv_tc!, ε⁻¹_dot!, ε_dot_approx!, H_Mₖ_H, tc, ct

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
    @tullio h[a,i,j,k] := H[b,i,j,k] * mn[a,b,i,j,k]
end

"""
    ct: a⃗ (cartesian vector) → v⃗ (transverse vector)
"""
function ct(h::AbstractArray{T,4},mn) where T<:Union{Real,Complex}
    @tullio H[a,i,j,k] := h[b,i,j,k] * mn[b,a,i,j,k]
end

"""
    kx_tc: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_tc(H::AbstractArray{T,4},mn,mag) where T
	kxscales = [-1.; 1.]
	kxinds = [2; 1]
    @tullio d[a,i,j,k] := kxscales[b] * H[kxinds[b],i,j,k] * mn[b,a,i,j,k] * mag[i,j,k] nograd=(kxscales,kxinds) # fastmath=false
	# @tullio d[a,i,j,k] := H[2,i,j,k] * m[a,i,j,k] * mag[i,j,k] - H[1,i,j,k] * n[a,i,j,k] * mag[i,j,k]  # nograd=(kxscales,kxinds) fastmath=false
end

"""
    kx_c2t: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_ct(e⃗::AbstractArray{T,4},mn,mag) where T
	# mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[b,i,j,k] := kxscales[b] * e⃗[a,i,j,k] * mn[kxinds[b],a,i,j,k] * mag[i,j,k] nograd=(kxinds,kxscales) # fastmath=false
end

"""
    zx_t2c: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_tc(H::AbstractArray{T,4},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,i,j,k] := zxscales[a] * H[b,i,j,k] * mn[b,zxinds[a],i,j,k] nograd=(zxscales,zxinds) # fastmath=false
end

"""
    zx_c2t: v⃗ (transverse vector) = ẑ × a⃗ (cartesian vector)
"""
function zx_ct(e⃗::AbstractArray{T,4},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxe⃗[b,i,j,k] := zxscales[a] * e⃗[a,i,j,k] * mn[b,zxinds[a],i,j,k] nograd=(zxscales,zxinds) # fastmath=false
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗::AbstractArray{T,4},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * fft(d⃗,(2:4))[b,i,j,k]  #fastmath=false
	return ifft(e⃗,(2:4))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗::AbstractArray{T,4},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * d⃗[b,i,j,k]  #fastmath=false
end

function H_Mₖ_H(H::AbstractArray{Complex{T},4},ε⁻¹,mag,m,n)::T where T<:Real
	# kxinds = [2; 1]
	# kxscales = [-1.; 1.]
	# ,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * temp[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# @tullio out := conj.(H)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_dot_t(zx_t2c(H,mn),ε⁻¹)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# return abs(out[1])
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	real( dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:4) ), real(flat(ε⁻¹))), (2:4)),mn,mag) ) )
end

function H_Mₖ_H(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,3},m::AbstractArray{T,4},n::AbstractArray{T,4})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	H_Mₖ_H(Ha,ε⁻¹,mag,m,n)
end


# 2D

"""
    tc: v⃗ (transverse vector) → a⃗ (cartesian vector)
"""
function tc(H::AbstractArray{T,3},mn) where T<:Union{Real,Complex}
    @tullio h[a,i,j] := H[b,i,j] * mn[b,a,i,j]
end

"""
    ct: a⃗ (cartesian vector) → v⃗ (transverse vector)
"""
function ct(h::AbstractArray{T,3},mn) where T<:Union{Real,Complex}
    @tullio H[a,i,j] := h[b,i,j] * mn[a,b,i,j]
end

"""
    kx_tc: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_tc(H::AbstractArray{T,3},mn,mag) where T
	kxscales = [-1.; 1.]
	kxinds = [2; 1]
    @tullio d[a,i,j] := kxscales[b] * H[kxinds[b],i,j] * mn[b,a,i,j] * mag[i,j] nograd=(kxscales,kxinds) # fastmath=false
	# @tullio d[a,i,j] := H[2,i,j] * m[a,i,j] * mag[i,j] - H[1,i,j] * n[a,i,j] * mag[i,j]  # nograd=(kxscales,kxinds) fastmath=false
end

"""
    kx_c2t: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_ct(e⃗::AbstractArray{T,3},mn,mag) where T
	# mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[b,i,j] := kxscales[b] * e⃗[a,i,j] * mn[kxinds[b],a,i,j] * mag[i,j] nograd=(kxinds,kxscales) # fastmath=false
end

"""
    zx_t2c: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_tc(H::AbstractArray{T,3},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,i,j] := zxscales[a] * H[b,i,j] * mn[b,zxinds[a],i,j] nograd=(zxscales,zxinds) # fastmath=false
end

"""
    zx_c2t: v⃗ (transverse vector) = ẑ × a⃗ (cartesian vector)
"""
function zx_ct(e⃗::AbstractArray{T,3},mn) where T
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxe⃗[b,i,j] := zxscales[a] * e⃗[a,i,j] * mn[b,zxinds[a],i,j] nograd=(zxscales,zxinds) # fastmath=false
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗::AbstractArray{T,3},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,i,j] :=  ε⁻¹[a,b,i,j] * fft(d⃗,(2:4))[b,i,j]  #fastmath=false
	return ifft(e⃗,(2:4))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗::AbstractArray{T,3},ε⁻¹) where T
	# eif = flat(ε⁻¹)
	@tullio e⃗[a,i,j] :=  ε⁻¹[a,b,i,j] * d⃗[b,i,j]  #fastmath=false
end

function H_Mₖ_H(H::AbstractArray{Complex{T},3},ε⁻¹,mag,m,n)::T where T<:Real
	# kxinds = [2; 1]
	# kxscales = [-1.; 1.]
	# ,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * temp[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# @tullio out := conj.(H)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_dot_t(zx_t2c(H,mn),ε⁻¹)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# return abs(out[1])
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	real( dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(flat(ε⁻¹))), (2:3)),mn,mag) ) )
end

function H_Mₖ_H(H::AbstractVector{Complex{T}},ε⁻¹,mag::AbstractArray{T,2},m::AbstractArray{T,3},n::AbstractArray{T,3})::T where T<:Real
	Nx,Ny = size(mag)
	Ha = reshape(H,(2,Nx,Ny))
	H_Mₖ_H(Ha,ε⁻¹,mag,m,n)
end


"""
################################################################################
#																			   #
#			  Function Definitions Implementing Mutating Operators			   #
#																			   #
################################################################################
"""

# 3D _M! and _P! subroutines

function kx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    # @assert size(Y) === size(X)
    # @assert size(d,4) == 3
    # @assert size(H,4) === 2
    @avx for k ∈ axes(d,4), j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
	# @avx for i ∈ axes(d,1), j ∈ axes(d,2), k ∈ axes(d,3), l in 0:0
		# scale = -mag[i,j,k]
		d[1+l,i,j,k] = ( H[1,i,j,k] * n[1+l,i,j,k] - H[2,i,j,k] * m[1+l,i,j,k] ) * -mag[i,j,k]
        d[2+l,i,j,k] = ( H[1,i,j,k] * n[2+l,i,j,k] - H[2,i,j,k] * m[2+l,i,j,k] ) * -mag[i,j,k]
        d[3+l,i,j,k] = ( H[1,i,j,k] * n[3+l,i,j,k] - H[2,i,j,k] * m[3+l,i,j,k] ) * -mag[i,j,k]
    end
    return d
end

function zx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(d,4), j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
		d[1+l,i,j,k] = -H[1,i,j,k] * m[2+l,i,j,k] - H[2,i,j,k] * n[2+l,i,j,k]
        d[2+l,i,j,k] =  H[1,i,j,k] * m[1+l,i,j,k] + H[2,i,j,k] * n[1+l,i,j,k]
    end
    return d
end

function kx_ct!(H::AbstractArray{Complex{T},4},e::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},mag::AbstractArray{T,3},Ninv::T)::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(H,4), j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = mag[i,j,k] * Ninv
        H[1+l,i,j,k] =  (	e[1+l,i,j,k] * n[1+l,i,j,k] + e[2+l,i,j,k] * n[2+l,i,j,k] + e[3+l,i,j,k] * n[3+l,i,j,k]	) * -scale  # -mag[i,j,k] * Ninv
		H[2+l,i,j,k] =  (	e[1+l,i,j,k] * m[1+l,i,j,k] + e[2+l,i,j,k] * m[2+l,i,j,k] + e[3+l,i,j,k] * m[3+l,i,j,k]	) * scale   # mag[i,j,k] * Ninv
    end
    return H
end

function eid!(e::AbstractArray{Complex{T},4},ε⁻¹,d::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0, h in 0:0
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

function kxinv_tc!(e::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},inv_mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
		e[1+l,i,j,k] = ( H[1,i,j,k] * n[1+l,i,j,k] - H[2,i,j,k] * m[1+l,i,j,k] ) * inv_mag[i,j,k]
        e[2+l,i,j,k] = ( H[1,i,j,k] * n[2+l,i,j,k] - H[2,i,j,k] * m[2+l,i,j,k] ) * inv_mag[i,j,k]
        e[3+l,i,j,k] = ( H[1,i,j,k] * n[3+l,i,j,k] - H[2,i,j,k] * m[3+l,i,j,k] ) * inv_mag[i,j,k]
    end
    return e
end

function kxinv_ct!(H::AbstractArray{Complex{T},4},d::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},inv_mag::AbstractArray{T,3},N::T)::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(H,4), j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = inv_mag[i,j,k] * N
        H[1+l,i,j,k] =  (	d[1+l,i,j,k] * n[1+l,i,j,k] + d[2+l,i,j,k] * n[2+l,i,j,k] + d[3+l,i,j,k] * n[3+l,i,j,k]	) * scale # inv_mag[i,j,k] * N
		H[2+l,i,j,k] =  (	d[1+l,i,j,k] * m[1+l,i,j,k] + d[2+l,i,j,k] * m[2+l,i,j,k] + d[3+l,i,j,k] * m[3+l,i,j,k]	) * -scale # inv_mag[i,j,k] * N
    end
    return H
end

function ed_approx!(d::AbstractArray{Complex{T},4},ε_ave::AbstractArray{T,3},e::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,4), j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
        d[1+l,i,j,k] =  ε_ave[i,j,k]*e[1+l,i,j,k]
        d[2+l,i,j,k] =  ε_ave[i,j,k]*e[2+l,i,j,k]
        d[3+l,i,j,k] =  ε_ave[i,j,k]*e[3+l,i,j,k]
    end
    return d
end

# 2D _M! and _P! subroutines

function kx_tc!(d::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},m::AbstractArray{T,3},n::AbstractArray{T,3},mag::AbstractArray{T,2})::AbstractArray{Complex{T},3} where T<:Real
    # @assert size(Y) === size(X)
    # @assert size(d,4) == 3
    # @assert size(H,4) === 2
    @avx for j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
	# @avx for i ∈ axes(d,1), j ∈ axes(d,2), l in 0:0
		# scale = -mag[i,j,k]
		d[1+l,i,j] = ( H[1,i,j] * n[1+l,i,j] - H[2,i,j] * m[1+l,i,j] ) * -mag[i,j]
        d[2+l,i,j] = ( H[1,i,j] * n[2+l,i,j] - H[2,i,j] * m[2+l,i,j] ) * -mag[i,j]
        d[3+l,i,j] = ( H[1,i,j] * n[3+l,i,j] - H[2,i,j] * m[3+l,i,j] ) * -mag[i,j]
    end
    return d
end

function zx_tc!(d::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},m::AbstractArray{T,3},n::AbstractArray{T,3})::AbstractArray{Complex{T},3} where T<:Real
    @avx for j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
		d[1+l,i,j] = -H[1,i,j] * m[2+l,i,j] - H[2,i,j] * n[2+l,i,j]
        d[2+l,i,j] =  H[1,i,j] * m[1+l,i,j] + H[2,i,j] * n[1+l,i,j]
    end
    return d
end

function kx_ct!(H::AbstractArray{Complex{T},3},e::AbstractArray{Complex{T},3},m::AbstractArray{T,3},n::AbstractArray{T,3},mag::AbstractArray{T,2},Ninv::T)::AbstractArray{Complex{T},3} where T<:Real
    @avx for j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = mag[i,j] * Ninv
        H[1+l,i,j] =  (	e[1+l,i,j] * n[1+l,i,j] + e[2+l,i,j] * n[2+l,i,j] + e[3+l,i,j] * n[3+l,i,j]	) * -scale  # -mag[i,j] * Ninv
		H[2+l,i,j] =  (	e[1+l,i,j] * m[1+l,i,j] + e[2+l,i,j] * m[2+l,i,j] + e[3+l,i,j] * m[3+l,i,j]	) * scale   # mag[i,j] * Ninv
    end
    return H
end

# function eid!(e::AbstractArray{Complex{T},3},ε⁻¹,d::AbstractArray{Complex{T},3})::AbstractArray{Complex{T},3} where T<:Real
#     @avx for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0, h in 0:0
#         e[1+h,i,j] =  ε⁻¹[1+h,1+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,1+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,1+l,i,j]*d[3+l,i,j]
#         e[2+h,i,j] =  ε⁻¹[1+h,2+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,2+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,2+l,i,j]*d[3+l,i,j]
#         e[3+h,i,j] =  ε⁻¹[1+h,3+l,i,j]*d[1+l,i,j] + ε⁻¹[2+h,3+l,i,j]*d[2+l,i,j] + ε⁻¹[3+h,3+l,i,j]*d[3+l,i,j]
#     end
#     return e
# end

function eid!(e::AbstractArray{Complex{T},3},ε⁻¹::AbstractArray{TA,2},d::AbstractArray{Complex{T},3})::AbstractArray{Complex{T},3} where {T<:Real,TA<:SMatrix{3,3}}
    er = reinterpret(reshape,SVector{3,Complex{T}},e)
	dr = reinterpret(reshape,SVector{3,Complex{T}},d)
	map!(*,er,ε⁻¹,dr)
	# map!(*,er,ε⁻¹,er)
    return e
end

function kxinv_tc!(e::AbstractArray{Complex{T},3},H::AbstractArray{Complex{T},3},m::AbstractArray{T,3},n::AbstractArray{T,3},inv_mag::AbstractArray{T,2})::AbstractArray{Complex{T},3} where T<:Real
    @avx for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
		e[1+l,i,j] = ( H[1,i,j] * n[1+l,i,j] - H[2,i,j] * m[1+l,i,j] ) * inv_mag[i,j]
        e[2+l,i,j] = ( H[1,i,j] * n[2+l,i,j] - H[2,i,j] * m[2+l,i,j] ) * inv_mag[i,j]
        e[3+l,i,j] = ( H[1,i,j] * n[3+l,i,j] - H[2,i,j] * m[3+l,i,j] ) * inv_mag[i,j]
    end
    return e
end

function kxinv_ct!(H::AbstractArray{Complex{T},3},d::AbstractArray{Complex{T},3},m::AbstractArray{T,3},n::AbstractArray{T,3},inv_mag::AbstractArray{T,2},N::T)::AbstractArray{Complex{T},3} where T<:Real
    @avx for j ∈ axes(H,3), i ∈ axes(H,2), l in 0:0
        scale = inv_mag[i,j] * N
        H[1+l,i,j] =  (	d[1+l,i,j] * n[1+l,i,j] + d[2+l,i,j] * n[2+l,i,j] + d[3+l,i,j] * n[3+l,i,j]	) * scale # inv_mag[i,j] * N
		H[2+l,i,j] =  (	d[1+l,i,j] * m[1+l,i,j] + d[2+l,i,j] * m[2+l,i,j] + d[3+l,i,j] * m[3+l,i,j]	) * -scale # inv_mag[i,j] * N
    end
    return H
end

function ed_approx!(d::AbstractArray{Complex{T},3},ε_ave::AbstractArray{T,2},e::AbstractArray{Complex{T},3})::AbstractArray{Complex{T},3} where T<:Real
    @avx for j ∈ axes(e,3), i ∈ axes(e,2), l in 0:0
        d[1+l,i,j] =  ε_ave[i,j]*e[1+l,i,j]
        d[2+l,i,j] =  ε_ave[i,j]*e[2+l,i,j]
        d[3+l,i,j] =  ε_ave[i,j]*e[3+l,i,j]
    end
    return d
end

# _M! and _P!

function _P!(Hout::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N}, ε_ave::AbstractArray{T},
	m::AbstractArray{T,N}, n::AbstractArray{T,N}, inv_mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},N} where {T<:Real,N}
	kxinv_tc!(e,Hin,m,n,inv_mag);
	mul!(e.data,𝓕⁻¹!,e.data);
    ed_approx!(d,ε_ave,e);
    mul!(d.data,𝓕!,d.data);
    kxinv_ct!(Hout,d,m,n,inv_mag,Ninv)
end

function _M!(Hout::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N}, ε⁻¹,
	m::AbstractArray{T,N}, n::AbstractArray{T,N}, mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},N} where {T<:Real,N}
    kx_tc!(d,Hin,m,n,mag);
    mul!(d.data,𝓕!,d.data);
    eid!(e,ε⁻¹,d);
    mul!(e.data,𝓕⁻¹!,e.data);
    kx_ct!(Hout,e,m,n,mag,Ninv)
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
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : SVector(-1.,0.,0.) #ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return mag, m, n
end

mag_m_n!(mag,m,n,kz::T,g⃗) where T <: Real = mag_m_n!(mag,m,n,SVector{3,T}(0.,0.,kz),g⃗)


function mag_m_n2(k⃗::SVector{3,T},g⃗::AbstractArray) where T <: Real
	g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗[1:2,:,:,:];dims=1)[1,:,:,:] .> 0. );
	g⃗ₜ_zero_mask! = Zygote.@ignore( .!(g⃗ₜ_zero_mask) );
	local ŷ = [0.; 1. ;0.]
	local zxinds = [2; 1; 3]
	local zxscales = [-1; 1. ;0.]
	local xinds1 = [2; 3; 1]
	local xinds2 = [3; 1; 2]
	@tullio kpg[ix,iy,iz] := k⃗[a] - g⃗[a,ix,iy,iz] fastmath=false
	@tullio mag[ix,iy,iz] := sqrt <| kpg[a,ix,iy,iz]^2 fastmath=false
	@tullio nt[ix,iy,iz,a] := zxscales[a] * kpg[zxinds[a],ix,iy,iz] * g⃗ₜ_zero_mask[ix,iy,iz] + ŷ[a] * g⃗ₜ_zero_mask![ix,iy,iz]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
	@tullio nmag[ix,iy,iz] := sqrt <| nt[a,ix,iy,iz]^2 fastmath=false
	@tullio n[a,ix,iy,iz] := nt[a,ix,iy,iz] / nmag[ix,iy,iz] fastmath=false
	@tullio mt[a,ix,iy,iz] := n[xinds1[a],ix,iy,iz] * kpg[xinds2[a],ix,iy,iz] - kpg[xinds1[a],ix,iy,iz] * n[xinds2[a],ix,iy,iz] nograd=(xinds1,xinds2) fastmath=false
	@tullio mmag[ix,iy,iz] := sqrt <| mt[a,ix,iy,iz]^2 fastmath=false
	@tullio m[a,ix,iy,iz] := mt[a,ix,iy,iz] / mmag[ix,iy,iz] fastmath=false
	return mag, m, n
end

function mag_m_n2(kz::T,g⃗::AbstractArray) where T <: Real
	mag_m_n2(SVector{3,T}(0.,0.,kz),g⃗)
end

function mag_m_n(k⃗::SVector{3,T},g⃗::AbstractArray{SVector{3,T2}}) where {T<:Real,T2<:Real}
	# for iz ∈ axes(g⃗,3), iy ∈ axes(g⃗,2), ix ∈ axes(g⃗,1) #, l in 0:0
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	n = Buffer(g⃗,size(g⃗))
	m = Buffer(g⃗,size(g⃗))
	mag = Buffer(zeros(T,size(g⃗)),size(g⃗))
	# n = bufferfrom(zeros(SVector{3,T},size(g⃗)))
	# m = bufferfrom(zeros(SVector{3,T},size(g⃗)))
	# mag = bufferfrom(zeros(T,size(g⃗)))
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =   ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : SVector(-1.,0.,0.) #ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return copy(mag), copy(m), copy(n) # HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(m))), HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(n)))
end

function mag_m_n(kz::T,g⃗::AbstractArray{SVector{3,T2}}) where {T<:Real,T2<:Real}
	mag_m_n(SVector{3,T}(0.,0.,kz),g⃗)
end

mag_m_n(k::Real,grid::Grid) = mag_m_n(k, g⃗(grid))

"""
################################################################################
#																			   #
#							  Struct Definitions 							   #
#																			   #
################################################################################
"""

mutable struct HelmholtzMap{ND,T} <: LinearMap{T}
    k⃗::SVector{3,T}
	Nx::Int
	Ny::Int
	Nz::Int
	N::Int
	Ninv::T
	g⃗::Array{SVector{3, T}, ND}
	mag::Array{T,ND} #HybridArray{Tuple{Nx,Ny,Nz},T,3,3,Array{T,3}}
    m⃗::Array{SVector{3, T}, ND} # HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}
	n⃗::Array{SVector{3, T}, ND} # HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}
	m::HybridArray # Base.ReinterpretArray{T,4}
	n::HybridArray # Base.ReinterpretArray{T,4}
    e::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
    d::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
    𝓕!::FFTW.cFFTWPlan
	𝓕⁻¹!::FFTW.cFFTWPlan #AbstractFFTs.ScaledPlan
	𝓕::FFTW.cFFTWPlan
	𝓕⁻¹::FFTW.cFFTWPlan #AbstractFFTs.ScaledPlan
	ε⁻¹::Array{SMatrix{3,3,T,9},ND} #HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3,3},T,5,5,Array{T,5}}
	ε_ave::Array{T,ND}  # for preconditioner
	inv_mag::Array{T,ND} # for preconditioner
	shift::T
end

mutable struct HelmholtzPreconditioner{ND,T} <: LinearMap{T}
	M̂::HelmholtzMap{ND,T}
end

mutable struct ModeSolver{ND,T}
	geom::Geometry
	materials::Vector{<:AbstractMaterial}
	grid::Grid{ND,T}
	M̂::HelmholtzMap{ND,T}
	P̂::HelmholtzPreconditioner{ND,T}
	eigs_itr::IterativeSolvers.LOBPCGIterator
	H⃗::Matrix{Complex{T}}
	ω²::Vector{Complex{T}}
	∂ω²∂k::Vector{T}
	λ⃗::Vector{Complex{T}}
	b⃗::Vector{Complex{T}}
	λd::HybridArray
	λẽ::HybridArray
	ε⁻¹_bar::Array{SMatrix{3,3,T,9}, ND}
	kx̄_m⃗::Array{SVector{3, T}, ND}
	kx̄_n⃗::Array{SVector{3, T}, ND}
	māg::Array{T,ND}
	k̄_kx::SVector{3,T}
	ω̄::T
	adj_itr::IterativeSolvers.BiCGStabIterable
	corner_sinds::Array{Int,ND}
	sinds_proc::Array #{NTuple{8,Int},ND}
	Srvol::Array{Tuple{SMatrix{3,3,T,9},T},ND}
	minds::Vector{Int}
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
	mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
	d0 = randn(Complex{T}, (3,size(gr)...))
	fftax = _fftaxes(gr)
	return HelmholtzMap{3,T}(
			SVector{3,T}(k⃗),
			gr.Nx,
			gr.Ny,
			gr.Nz,
			N(gr),
			1. / N(gr),
			g⃗s,
			mag,
			m⃗,
			n⃗,
			HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)),
			HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)),
		    HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Complex{T}}(d0),# (Array{T}(undef,(Nx,Ny,Nz,3))),
		    HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Complex{T}}(d0),# (Array{T}(undef,(Nx,Ny,Nz,3))),
			plan_fft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			plan_fft(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			ε⁻¹,
			[ 3. * inv(sum(diag(einv))) for einv in ε⁻¹],
			[ inv(mm) for mm in mag ], # inverse |k⃗+g⃗| magnitudes for precond. ops
			shift,
		)
end

function HelmholtzMap(k⃗::AbstractVector{T}, ε⁻¹, gr::Grid{2,T}; shift=0. ) where {ND,T<:Real}
	g⃗s = g⃗(gr)
	mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
	d0 = randn(Complex{T}, (3,size(gr)...))
	fftax = _fftaxes(gr)
	return HelmholtzMap{2,T}(
			SVector{3,T}(k⃗),
			gr.Nx,
			gr.Ny,
			gr.Nz,
			N(gr),
			1. / N(gr),
			g⃗s,
			mag,
			m⃗,
			n⃗,
			HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,m⃗)),
			HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,n⃗)),
		    HybridArray{Tuple{3,Dynamic(),Dynamic()},Complex{T}}(d0),# (Array{T}(undef,(Nx,Ny,Nz,3))),
		    HybridArray{Tuple{3,Dynamic(),Dynamic()},Complex{T}}(d0),# (Array{T}(undef,(Nx,Ny,Nz,3))),
			plan_fft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			plan_fft(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
			plan_bfft(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
			ε⁻¹,
			[ 3. * inv(sum(diag(einv))) for einv in ε⁻¹],
			[ inv(mm) for mm in mag ], # inverse |k⃗+g⃗| magnitudes for precond. ops
			shift,
		)
end

function HelmholtzMap(kz::T, ε⁻¹, gr::Grid; shift=0.) where {T<:Real}
	HelmholtzMap(SVector{3,T}(0.,0.,kz), ε⁻¹, gr::Grid; shift)
end

function ModeSolver(k⃗::SVector{3,T}, geom::Geometry, gr::Grid{ND}; nev=1, tol=1e-8, maxiter=3000, ω₀=1/1.55, constraint=nothing,) where {ND,T<:Real}
	# run inital smoothing sub-processes
	# ε⁻¹ = εₛ⁻¹( (1. / ω₀), geom, gr)
	sinds,sinds_proc,Srvol,mats,minds,ε⁻¹ = _εₛ⁻¹_init( (1. / ω₀), geom.shapes, gr)
	M̂ = HelmholtzMap(k⃗, ε⁻¹, gr)
	P̂ = HelmholtzPreconditioner(M̂)
	eigs_itr = LOBPCGIterator(M̂,false,randn(eltype(M̂),(size(M̂)[1],nev)),P̂,constraint)
	λ⃗ = randn(Complex{T},2*M̂.N)
	b⃗ = similar(λ⃗)
	adj_itr = bicgstabl_iterator!(λ⃗, M̂ - ( 1. * I ), b⃗, 2;		# last entry is `l`::Int = # of GMRES iterations
                             Pl = Identity(),
                             max_mv_products = size(M̂, 2),
                             abstol = zero(T),
                             reltol = sqrt(eps(T)),
                             initial_zero = false)
	ModeSolver{ND,T}(
		geom,
		mats,
		gr,
		M̂,
		P̂,
		eigs_itr,
		eigs_itr.XBlocks.block,
		eigs_itr.λ,
		zeros(T,nev),
		λ⃗,
		b⃗,
		similar(M̂.d),							# λ⃗d
		similar(M̂.e),							# λ⃗ẽ
		similar(M̂.ε⁻¹),						# ε⁻¹_bar
		similar(M̂.m⃗),							 # kx̄_m⃗
		similar(M̂.n⃗),							# kx̄_n⃗
		similar(M̂.mag),						# māg
		zero(SVector{3,Float64}),				# k̄_kx
		0.,										# ω̄
		adj_itr,
		sinds,
		sinds_proc,
		Srvol,
		minds,
	)
end

function ModeSolver(kz::T, geom::Geometry, gr::Grid{ND}; nev=1, tol=1e-8, maxiter=3000,constraint=nothing,) where {ND,T<:Real}
	ModeSolver(SVector{3,T}(0.,0.,kz), geom, gr; nev, tol, maxiter, constraint)
end

"""
################################################################################
#																			   #
#							  	Struct Methods 								   #
#																			   #
################################################################################
"""

function (M̂::HelmholtzMap{2,T})(Hout::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3}) where T<:Real
	_M!(Hout,Hin,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m,M̂.n,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv)
end

function (M̂::HelmholtzMap{3,T})(Hout::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4}) where T<:Real
	_M!(Hout,Hin,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m,M̂.n,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv)
end

function (M̂::HelmholtzMap{2,T})(Hout::AbstractVector{Complex{T}}, Hin::AbstractVector{Complex{T}}) where T<:Real
	@inbounds Hin_arr = reshape(Hin,(2,M̂.Nx,M̂.Ny))
	@inbounds Hout_arr = reshape(Hout,(2,M̂.Nx,M̂.Ny))
	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m,M̂.n,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv) )
end

function (M̂::HelmholtzMap{3,T})(Hout::AbstractVector{Complex{T}}, Hin::AbstractVector{Complex{T}}) where T<:Real
	@inbounds Hin_arr = reshape(Hin,(2,M̂.Nx,M̂.Ny,M̂.Nz))
	@inbounds Hout_arr = reshape(Hout,(2,M̂.Nx,M̂.Ny,M̂.Nz))
	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m,M̂.n,M̂.mag,M̂.𝓕!,M̂.𝓕⁻¹!,M̂.Ninv) )
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractArray{T,3}, Hin::AbstractArray{T,3}) where T<:Union{Real, Complex}
	_P!(Hout,Hin,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m,P̂.M̂.n,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv)
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractArray{T,4}, Hin::AbstractArray{T,4}) where T<:Union{Real, Complex}
	_P!(Hout,Hin,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m,P̂.M̂.n,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv)
end

function (P̂::HelmholtzPreconditioner{2})(Hout::AbstractVector{T}, Hin::AbstractVector{T}) where T<:Union{Real, Complex}
	@inbounds Hin_arr = reshape(Hin,(2,P̂.M̂.Nx,P̂.M̂.Ny))
	@inbounds Hout_arr = reshape(Hout,(2,P̂.M̂.Nx,P̂.M̂.Ny))
	vec( _P!(Hout_arr,Hin_arr,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m,P̂.M̂.n,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv) )
end

function (P̂::HelmholtzPreconditioner{3})(Hout::AbstractVector{T}, Hin::AbstractVector{T}) where T<:Union{Real, Complex}
	@inbounds Hin_arr = reshape(Hin,(2,P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz))
	@inbounds Hout_arr = reshape(Hout,(2,P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz))
	vec( _P!(Hout_arr,Hin_arr,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m,P̂.M̂.n,P̂.M̂.inv_mag,P̂.M̂.𝓕!,P̂.M̂.𝓕⁻¹!,P̂.M̂.Ninv) )
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

# Update k methods

function update_k(M̂::HelmholtzMap{2,T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n) = mag_m_n(k⃗,M̂.g⃗)
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
    M̂.m⃗ = m #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = n #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.k⃗ = k⃗
end

function update_k(M̂::HelmholtzMap{3,T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n) = mag_m_n(k⃗,M̂.g⃗)
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
    M̂.m⃗ = m #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = n #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.k⃗ = k⃗
end


function update_k(M̂::HelmholtzMap{ND,T},kz::T) where {ND,T<:Real}
	update_k(M̂,SVector{3,T}(0.,0.,kz))
end

update_k(ms::ModeSolver,k) = update_k(ms.M̂,k)

function update_k!(M̂::HelmholtzMap{2,T},k⃗::SVector{3,T}) where T<:Real
	mag_m_n!(M̂.mag,M̂.m⃗,M̂.n⃗,k⃗,M̂.g⃗)
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.inv_mag = [inv(mm) for mm in M̂.mag]
	M̂.k⃗ = k⃗
end

function update_k!(M̂::HelmholtzMap{3,T},k⃗::SVector{3,T}) where T<:Real
	mag_m_n!(M̂.mag,M̂.m⃗,M̂.n⃗,k⃗,M̂.g⃗)
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.inv_mag = [inv(mm) for mm in M̂.mag]
	M̂.k⃗ = k⃗
end

function update_k!(M̂::HelmholtzMap{ND,T},kz::T) where {ND,T<:Real}
	update_k!(M̂,SVector{3,T}(0.,0.,kz))
end

update_k!(ms::ModeSolver,k) = update_k!(ms.M̂,k)

# Update ε⁻¹ methods

function update_ε⁻¹(M̂::HelmholtzMap{ND,T},ε⁻¹) where {ND,T<:Real}
	@assert size(M̂.ε⁻¹) == size(ε⁻¹)
	M̂.ε⁻¹ = ε⁻¹
end

function update_ε⁻¹(ms::ModeSolver{ND,T},ε⁻¹) where {ND,T<:Real}
	@assert size(ms.M̂.ε⁻¹) == size(ε⁻¹)
	ms.M̂.ε⁻¹ = ε⁻¹
end

# function S_rvol(geom;ms::ModeSolver)
# 	es = vcat(εs(ms.geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end



# property methods
Base.size(A::HelmholtzMap) = (2*A.N, 2*A.N)
Base.size(A::HelmholtzMap,d::Int) = 2*A.N
Base.eltype(A::HelmholtzMap{ND,T}) where {ND,T<:Real}  = Complex{T}
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


mag_m_n!(M̂::HelmholtzMap,k) = mag_m_n!(M̂.mag,M̂.m⃗,M̂.n⃗,M̂.g⃗,k)
mag_m_n!(ms::ModeSolver,k) = mag_m_n!(ms.M̂.mag,ms.M̂.m⃗,ms.M̂.n⃗,ms.M̂.g⃗,k)
