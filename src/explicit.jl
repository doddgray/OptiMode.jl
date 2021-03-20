export kx_tc_sp_coo, kx_tc_sp, kx_ct_sp, zx_tc_sp_coo, zx_tc_sp, zx_ct_sp, ε⁻¹_sp_coo,  ε⁻¹_sp, M̂_sp, M̂ₖ_sp

using SparseArrays
using Zygote: Buffer
function ChainRulesCore.rrule(::typeof(SparseArrays.SparseMatrixCSC),
    m::Integer, n::Integer, pp::Vector, ii::Vector, Av::Vector)
    A = SparseMatrixCSC(m,n,pp,ii,Av)
    function SparseMatrixCSC_pullback(dA)
        # Pick out the entries in `dA` corresponding to nonzeros in `A`
        dAv = Vector{eltype(dA)}(undef, length(Av))
        for j = 1:n, p = pp[j]:pp[j+1]-1
            dAv[p] = dA[ii[p],j]
        end
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dAv)
    end

    return A, SparseMatrixCSC_pullback
end

function kx_tc_sp_coo(mag::AbstractArray{T,2},mn) where T<:Real
    Nx, Ny = size(mag)
    I = Int32[]
    J = Int32[]
    V = Float64[]
    # kxt2c_matrix_buf = Zygote.bufferfrom(zeros(Complex{T},(3*Nx*Ny),(2*Nx*Ny)))
    for ix=1:Nx,iy=1:Ny,a=1:3 #,b=1:2
        q =  Nx * (iy - 1) + ix

        # reference from kcross_t2c!:
        #       ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[1,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[1,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        #       ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[2,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[2,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        #       ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[3,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[3,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        # which implements d⃗ = k×ₜ₂c ⋅ H⃗
        # Here we want to explicitly define the matrix k×ₜ₂c
        # the general indexing scheme:
        # kxt2c_matrix_buf[ (3*q-2)+a-1 ,(2*q-1) + (b-1) ] <==> mn[a,b,ix,iy,iz], mag[ix,iy,iz]
        # b = 1  ( m⃗ )
        # kxt2c_matrix_buf[(3*q-2)+a-1,(2*q-1)] = mn[a,2,ix,iy] * mag[ix,iy]
		push!(I,(3*q-2)+a-1)
		push!(J,2*q-1)
		push!(V,mn[2,a,ix,iy] * mag[ix,iy])
	        # b = 2  ( n⃗ )
	        # kxt2c_matrix_buf[(3*q-2)+a-1,(2*q-1)+1] = mn[a,1,ix,iy] * -mag[ix,iy]
		push!(I,(3*q-2)+a-1)
		push!(J,(2*q-1)+1)
		push!(V,mn[1,a,ix,iy] * -mag[ix,iy])
    end
    # return copy(kxt2c_matrix_buf)
    return sparse(I,J,V)
end

function kx_tc_sp(mag::AbstractArray{T,2},mn::AbstractArray{T,4}) where T<:Real
    Nx, Ny = size(mag)
    NN = length(mag)
    V = Buffer(mag,0)
    for iy=1:Ny, ix=1:Nx
    	push!(V,(mn[2,:,ix,iy] * mag[ix,iy])...)
    	push!(V,(mn[1,:,ix,iy] * -mag[ix,iy])...)
    end
    SparseMatrixCSC(
    	3*NN,	# m
    	2*NN,	# n
    	collect(Int32,1:3:6*NN+3), # colptr
    	convert(Vector{Int32},repeat(0:NN-1,inner=6).*3 .+ repeat([1,2,3,1,2,3],NN)), # rowval
    	copy(V),	# nzval
    )
end

function kx_tc_sp(mag::AbstractArray{T,2},m⃗::AbstractArray{SVector{3,T},2},n⃗::AbstractArray{SVector{3,T},2}) where T<:Real
    Nx, Ny = size(mag)
    NN = length(mag)
    V = Buffer(mag,0)
    for iy=1:Ny, ix=1:Nx
    	push!(V,(n⃗[ix,iy] * mag[ix,iy])...)
    	push!(V,(m⃗[ix,iy] * -mag[ix,iy])...)
    end
    SparseMatrixCSC(
    	3*NN,	# m
    	2*NN,	# n
    	collect(Int32,1:3:6*NN+3), # colptr
    	convert(Vector{Int32},repeat(0:NN-1,inner=6).*3 .+ repeat([1,2,3,1,2,3],NN)), # rowval
    	copy(V),	# nzval
    )
end

function kx_tc_sp(k,grid::Grid{ND,T}) where {ND,T<:Real}
    mag,m⃗,n⃗ = mag_m_n(k,g⃗(grid))
    Ns = size(grid)
    NN = N(grid)
    V = Buffer(mag,0)
    for iy=1:Ns[2], ix=1:Ns[1]
    	push!(V,(n⃗[ix,iy] * mag[ix,iy])...)
    	push!(V,(m⃗[ix,iy] * -mag[ix,iy])...)
    end
    SparseMatrixCSC(
    	3*NN,	# m
    	2*NN,	# n
    	collect(Int32,1:3:6*NN+3), # colptr
    	convert(Vector{Int32},repeat(0:NN-1,inner=6).*3 .+ repeat([1,2,3,1,2,3],NN)), # rowval
    	copy(V),	# nzval
    )
end

kx_ct_sp(k,grid::Grid) = -kx_tc_sp(k,grid::Grid)'

function zx_tc_sp_coo(mag::AbstractArray{T,2},mn) where T<:Real
    Nx, Ny = size(mn)[3:4]
    NN = Nx * Ny
    I = Int32[]
    J = Int32[]
    V = Float64[]
    for ix=1:Nx,iy=1:Ny
        q =  Nx * (iy - 1) + ix
		# reference from zcross_t2c!:
		#          ds.d[1,i,j,k] = -Hin[1,i,j,k] * ds.mn[2,1,i,j,k] - Hin[2,i,j,k] * ds.mn[2,2,i,j,k]
		#          ds.d[2,i,j,k] =  Hin[1,i,j,k] * ds.mn[1,1,i,j,k] + Hin[2,i,j,k] * ds.mn[1,2,i,j,k]
		#          ds.d[3,i,j,k] = 0
		# which implements d⃗ = z×ₜ₂c ⋅ H⃗
		# Here we want to explicitly define the matrix z×ₜ₂c
		# the general indexing scheme:
		# zxt2c_matrix_buf[ (3*q-2)+a-1 ,(2*q-1) + (b-1) ] <==> mn[a,b,ix,iy,iz]
		# a = 1  ( x̂ ), b = 1  ( m⃗ )
		# zxt2c_matrix_buf[(3*q-2),(2*q-1)] = -mn[2,1,ix,iy,iz]
		push!(I,3*q-2)
		push!(J,2*q-1)
		push!(V,-mn[1,2,ix,iy])
		# a = 1  ( x̂ ), b = 2  ( n⃗ )
		# zxt2c_matrix_buf[(3*q-2),2*q] = -mn[2,2,ix,iy,iz]
		push!(I,3*q-2)
		push!(J,2*q)
		push!(V,-mn[2,2,ix,iy])
		# a = 2  ( ŷ ), b = 1  ( m⃗ )
		# zxt2c_matrix_buf[(3*q-2)+1,(2*q-1)] = mn[1,1,ix,iy,iz]
		push!(I,3*q-1)
		push!(J,2*q-1)
		push!(V,mn[1,1,ix,iy])
		# a = 2  ( ŷ ), b = 2  ( n⃗ )
		# zxt2c_matrix_buf[(3*q-2)+1,2*q] = mn[1,2,ix,iy,iz]
		push!(I,3*q-1)
		push!(J,2*q)
		push!(V,mn[2,1,ix,iy])
    end
    return sparse(I,J,V,3*NN,2*NN)
end

function zx_tc_sp(mag::AbstractArray{T,2},m⃗::AbstractArray{SVector{3,T},2},n⃗::AbstractArray{SVector{3,T},2}) where T<:Real
    Nx, Ny = size(mag)
    NN = length(mag)
    V = Buffer(mag,0)
    for iy=1:Ny, ix=1:Nx
    	push!(V, -m⃗[ix,iy][2] )
		push!(V, m⃗[ix,iy][1] )
		push!(V, -n⃗[ix,iy][2] )
		push!(V, n⃗[ix,iy][1] )
    end
    SparseMatrixCSC(
    	3*NN,	# m
    	2*NN,	# n
		collect(Int32,1:2:4*NN+2), # colptr
		convert(Vector{Int32},repeat(0:NN-1,inner=4).*3 .+ repeat([1,2,1,2],NN)), # rowval
    	copy(V),	# nzval
    )
end

function zx_tc_sp(k,grid::Grid{ND,T}) where {ND,T<:Real}
    mag,m⃗,n⃗ = mag_m_n(k,g⃗(grid))
    Ns = size(grid)
    NN = N(grid)
    V = Buffer(mag,0)
	for iy=1:Ns[2], ix=1:Ns[1]
    	push!(V, -m⃗[ix,iy][2] )
		push!(V, m⃗[ix,iy][1] )
		push!(V, -n⃗[ix,iy][2] )
		push!(V, n⃗[ix,iy][1] )
    end
    SparseMatrixCSC(
    	3*NN,	# m
    	2*NN,	# n
		collect(Int32,1:2:4*NN+2), # colptr
		convert(Vector{Int32},repeat(0:NN-1,inner=4).*3 .+ repeat([1,2,1,2],NN)), # rowval
    	copy(V),	# nzval
    )
end

zx_ct_sp(k,grid::Grid) = zx_tc_sp(k,grid::Grid)'

function ε⁻¹_sp_coo(ω,geom::AbstractVector{<:Shape},grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
	ei = εₛ⁻¹(ω,geom,grid) #wl doesnt affect anything for constant indices  #make_εₛ⁻¹(shapes,grid)
	Ns = size(grid)
    NN = N(grid)
    I = Int32[]
    J = Int32[]
    V = Float64[]
	for i=1:Ns[1],j=1:Ns[2],a=1:3,b=1:3
		q = (Ny * (j-1) + i)
		eiv = ei[i,j][a,b]
		# if !iszero(eiv)
			# ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
		push!(I,(3*q-2)+a-1)
		push!(J,(3*q-2)+b-1)
		push!(V,eiv)
		# end
	end
	return sparse(I,J,V,3*NN,3*NN)
end

function ε⁻¹_sp(ω,geom::AbstractVector{<:Shape},grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
	ei = εₛ⁻¹(ω,geom,grid) #wl doesnt affect anything for constant indices  #make_εₛ⁻¹(shapes,grid)
	Ns = size(grid)
    NN = N(grid)
    V = Buffer([3.2],0)
	for iy=1:Ns[2], ix=1:Ns[1] #,a=1:3,b=1:3
		push!(V,ei[ix,iy]...) #[a,b])
	end
	SparseMatrixCSC(
    	3*NN,	# m
    	3*NN,	# n
    	collect(Int32,1:3:9*NN+3), # colptr
    	convert(Vector{Int32},repeat(0:NN-1,inner=9).*3 .+ repeat([1,2,3,1,2,3,1,2,3],NN)), # rowval
    	copy(V),	# nzval
    )
end

function M̂_sp(ω,k,geom,grid::Grid{2})
	Ns = size(grid)
	kxtcsp = kx_tc_sp(k,grid)
	eisp = ε⁻¹_sp(ω,geom,grid)
	𝓕 = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=false,ismutating=false)
    end
    𝓕⁻¹ = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(ifft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=false,ismutating=false)
    end
	kxtcsp' * 𝓕⁻¹ * eisp * 𝓕 * kxtcsp
end

function M̂ₖ_sp(ω,k,geom,grid::Grid{2})
	Ns = size(grid)
	kxtcsp = kx_tc_sp(k,grid)
	zxtcsp = zx_tc_sp(k,grid)
	eisp = ε⁻¹_sp(ω,geom,grid)
	𝓕 = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=false,ismutating=false)
    end
    𝓕⁻¹ = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(ifft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=false,ismutating=false)
    end
	kxtcsp' * 𝓕⁻¹ * eisp * 𝓕 * zxtcsp
end


# ## set discretization parameters and generate explicit dense matrices
# Δx          =   6.                    # μm
# Δy          =   4.                    # μm
# Δz          =   1.
# Nx          =   16
# Ny          =   16
# Nz          =   1
# kz          =   p0[1] #1.45
# # ω           =   1 / λ
# p = p0 #[kz,w,t_core,θ,n_core,n_subs,edge_gap] #,Δx,Δy,Δz,Nx,Ny,Nz]
# eid = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
# g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# ds = MaxwellData(p[1],g)
# ei = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
# # eii = similar(ei); [ (eii[a,b,i,j,k] = inv(ei[:,:,i,j,k])[a,b]) for a=1:3,b=1:3,i=1:Nx,j=1:Ny,k=1:Nz ] # eii = epsilon tensor field (eii for epsilon_inverse_inverse, yea it's dumb)
# Mop = M̂!(ei,ds)
# Mop2 = M̂(ei,ds)
# Mₖop = M̂ₖ(ei,ds.mn,ds.kpg_mag,ds.𝓕,ds.𝓕⁻¹)
# M = Matrix(Mop)
# dMdk = Matrix(Mₖop)
# mag,mn = calc_kpg(p[1],OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
# eid = ei_dot_rwg(p0)
#
# make_M(eid,mag,mn) ≈ M
# make_Mₖ(eid,mag,mn) ≈ -dMdk
# make_Mₖ_eidot(p,eid) ≈ -dMdk
# make_Mₖ(p0) ≈ -dMdk
#
# 𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
# 𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
#
# ##
# # M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)
# # function M(H,ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)
# #     kx_c2t( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_t2c(H,mn,kpg_mag), ε⁻¹), mn,kpg_mag)
# # end
# kxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# kxt2c = Matrix(kxt2c_op)
# F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# # F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# F = Matrix(F_op)
# einv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec( ε⁻¹_dot( reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)), ei ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# einv = Matrix(einv_op)
# Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕⁻¹*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# # Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(bfft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# Finv = Matrix(Finv_op)
# kxc2t_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(2,ds.Nx,ds.Ny,ds.Nz),*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# kxc2t = Matrix(kxc2t_op)
# zxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( zx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# zxt2c = Matrix(zxt2c_op)
#
# @assert -kxc2t * Finv * einv * F * kxt2c ≈ M
# @assert kxc2t * Finv * einv * F * zxt2c ≈ dMdk # wrong sign?
# @assert make_M(p;Δx,Δy,Δz,Nx,Ny,Nz) ≈ M
# @assert make_M_eidot(p,eid;Δx,Δy,Δz,Nx,Ny,Nz) ≈ M
# @assert ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz) ≈ einv
# # if Finv is ifft
# @assert F' ≈  Finv * ( size(F)[1]/3 )
# @assert Finv' * ( size(F)[1]/3 ) ≈  F
# # # if Finv is bfft
# # @assert F' ≈ Finv
# # @assert Finv' ≈  F
# @assert kxc2t' ≈ -kxt2c
# @assert kxt2c' ≈ -kxc2t
#
# # ix = 8
# # iy = 4
# # q = Nx * (iy - 1) + ix
# # 3q-2:3q+3 # 3q-2:3q-2+6-1
# # 2q-1:2q+2 # 2q-1:2q-1+4-1
# #
# # real(kxt2c[3q-2:3q+3,2q-1:2q+2])
# @assert kxt2c_matrix(p0) ≈ kxt2c
# @assert kxt2c_matrix(mag,mn) ≈ kxt2c
# @assert zxt2c_matrix(mn) ≈ zxt2c
