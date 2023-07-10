export kx_tc_sp_coo, kx_tc_sp, kx_ct_sp, zx_tc_sp_coo, zx_tc_sp, zx_ct_sp, ε⁻¹_sp_coo
export ε⁻¹_sp, nng⁻¹_sp, M̂_sp, M̂ₖ_sp, 𝓕_dense, 𝓕⁻¹_dense

using SparseArrays
using Zygote: Buffer


function ChainRulesCore.rrule(::typeof(SparseArrays.SparseMatrixCSC),
    m::Integer, n::Integer, pp::Vector, ii::Vector, Av::Vector)
    A = SparseMatrixCSC(m,n,pp,ii,Av)
    function SparseMatrixCSC_pullback(dA::AbstractMatrix)
        dAv = Vector{eltype(dA)}(undef, length(Av))
        for j = 1:n, p = pp[j]:pp[j+1]-1
            dAv[p] = dA[ii[p],j]
        end
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dAv)
    end
    function SparseMatrixCSC_pullback(dA::SparseMatrixCSC)
        @assert getproperty.(Ref(A), (:m,:n,:colptr,:rowval)) == getproperty.(Ref(dA), (:m,:n,:colptr,:rowval))
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), dA.nzval)
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

function ε⁻¹_sp_coo(ω,geom::Geometry,grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
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

function ε⁻¹_sp(ω,geom::Geometry,grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
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

function nng⁻¹_sp(ω,geom::Geometry,grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
	nnginv = nngₛ⁻¹(ω,geom,grid)
	Ns = size(grid)
    NN = N(grid)
    V = Buffer([3.2],0)
	for iy=1:Ns[2], ix=1:Ns[1] #,a=1:3,b=1:3
		push!(V,nnginv[ix,iy]...) #[a,b])
	end
	SparseMatrixCSC(
    	3*NN,	# m
    	3*NN,	# n
    	collect(Int32,1:3:9*NN+3), # colptr
    	convert(Vector{Int32},repeat(0:NN-1,inner=9).*3 .+ repeat([1,2,3,1,2,3,1,2,3],NN)), # rowval
    	copy(V),	# nzval
    )
end

function nng⁻¹_sp(nnginv,grid::Grid) #(ω,geom::Vector{<:Shape},grid::Grid{ND,T})
	Ns = size(grid)
    NN = N(grid)
    V = Buffer([3.2],0)
	for iy=1:Ns[2], ix=1:Ns[1] #,a=1:3,b=1:3
		push!(V,nnginv[ix,iy]...) #[a,b])
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
	Ninv = 1. / N(grid)
	kxtcsp = kx_tc_sp(k,grid)
	eisp = ε⁻¹_sp(ω,geom,grid)
	𝓕 = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
    𝓕⁻¹ = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(bfft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
	Ninv * kxtcsp' * 𝓕⁻¹ * eisp * 𝓕 * kxtcsp
end

function M̂ₖ_sp(ω,k,geom::Geometry,grid::Grid{2})
	Ns = size(grid)
	Ninv = 1. / N(grid)
	kxtcsp = kx_tc_sp(k,grid)
	zxtcsp = zx_tc_sp(k,grid)
	eisp = nng⁻¹_sp(ω,geom,grid) #ε⁻¹_sp(ω,geom,grid)
	𝓕 = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
    𝓕⁻¹ = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(bfft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
	Ninv * kxtcsp' * 𝓕⁻¹ * eisp * 𝓕 * zxtcsp
end

function M̂ₖ_sp(k,nnginv,grid::Grid{2})
	Ns = size(grid)
	Ninv = 1. / N(grid)
	kxtcsp = kx_tc_sp(k,grid)
	zxtcsp = zx_tc_sp(k,grid)
	nnginvsp = nng⁻¹_sp(nnginv,grid)
	𝓕 = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
    𝓕⁻¹ = Zygote.ignore() do
        LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(bfft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
    end
	Ninv * kxtcsp' * 𝓕⁻¹ * nnginvsp * 𝓕 * zxtcsp
end

𝓕_dense(grid::Grid{3}) = Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,grid.Nx,grid.Ny,grid.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=false))
𝓕⁻¹_dense(grid::Grid{3}) = Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(bfft(reshape(d,(3,grid.Nx,grid.Ny,grid.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=false))

𝓕_dense(grid::Grid{2}) = Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,grid.Nx,grid.Ny)),(2:3)))::AbstractArray{ComplexF64,1},*(3,grid.Nx,grid.Ny),ishermitian=false,ismutating=false))
𝓕⁻¹_dense(grid::Grid{2}) = Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(bfft(reshape(d,(3,grid.Nx,grid.Ny)),(2:3)))::AbstractArray{ComplexF64,1},*(3,grid.Nx,grid.Ny),ishermitian=false,ismutating=false))

