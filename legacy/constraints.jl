# Parity operators based on MPB code:
# https://github.com/NanoComp/mpb/blob/master/src/maxwell/maxwell_constraints.c
using LoopVectorization

export 𝓟x!, 𝓟x̄!, 𝓟x, 𝓟x̄

function _𝓟x!(Ho::AbstractArray{Complex{T},3},Hi::AbstractArray{Complex{T},3},grid::Grid{2}) where T<:Real
    Nx,Ny = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iy ∈ 1:Ny
		for ix ∈ 1:(Nx÷2), l in 0:0 #1:((Nx÷2)-1), l in 0:0
			ix1 = ix
			ix2 = (ix > 1 ? Nx - (ix-2) : 1)
			temp1 		= 0.5*( Hi[1+l,ix1,iy] - Hi[1+l,ix2,iy] )
			temp2 		= 0.5*( Hi[2+l,ix1,iy] + Hi[2+l,ix2,iy] )
			Ho[1+l,ix1,iy] 	= temp1
			Ho[2+l,ix1,iy] 	= temp2
			Ho[1+l,ix2,iy] = -temp1
			Ho[2+l,ix2,iy] = temp2
		end
	end
	return Ho
end

function _𝓟x̄!(Ho::AbstractArray{Complex{T},3},Hi::AbstractArray{Complex{T},3},grid::Grid{2}) where T<:Real
    Nx,Ny = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iy ∈ 1:Ny
		for ix ∈ 1:(Nx÷2), l in 0:0 # 1:((Nx÷2)-1), l in 0:0
			ix1 = ix
			ix2 = (ix > 1 ? Nx - (ix-2) : 1)
			temp1 		= 0.5*( Hi[1+l,ix1,iy] + Hi[1+l,ix2,iy] )
			temp2 		= 0.5*( Hi[2+l,ix1,iy] - Hi[2+l,ix2,iy] )
			Ho[1+l,ix1,iy] 	= temp1
			Ho[2+l,ix1,iy] 	= temp2
			Ho[1+l,ix2,iy] = temp1
			Ho[2+l,ix2,iy] = -temp2
		end
	end
	return Ho
end

function _𝓟x!(Ho::AbstractArray{Complex{T},4},Hi::AbstractArray{Complex{T},4},grid::Grid{3}) where T<:Real
    Nx,Ny,Nz = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iz ∈ 1:Nz, iy ∈ 1:Ny
		for ix ∈ 1:(Nx÷2), l in 0:0 # 1:((Nx÷2)-1), l in 0:0
			ix1 = ix
			ix2 = (ix > 1 ? Nx - (ix-2) : 1)
			temp1 		= 0.5*( Hi[1+l,ix1,iy,iz] - Hi[1+l,ix2,iy,iz] )
			temp2 		= 0.5*( Hi[2+l,ix1,iy,iz] + Hi[2+l,ix2,iy,iz] )
			Ho[1+l,ix1,iy,iz] 	= temp1
			Ho[2+l,ix1,iy,iz] 	= temp2
			Ho[1+l,ix2,iy,iz] = -temp1
			Ho[2+l,ix2,iy,iz] = temp2
		end
	end
	return Ho
end

function _𝓟x̄!(Ho::AbstractArray{Complex{T},4},Hi::AbstractArray{Complex{T},4},grid::Grid{3}) where T<:Real
    Nx,Ny,Nz = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iz ∈ 1:Nz, iy ∈ 1:Ny
		for ix ∈ 1:(Nx÷2), l in 0:0 # 1:((Nx÷2)-1), l in 0:0
			ix1 = ix
			ix2 = (ix > 1 ? Nx - (ix-2) : 1)
			temp1 		= 0.5*( Hi[1+l,ix1,iy,iz] + Hi[1+l,ix2,iy,iz] )
			temp2 		= 0.5*( Hi[2+l,ix1,iy,iz] - Hi[2+l,ix2,iy,iz] )
			Ho[1+l,ix1,iy,iz] 	= temp1
			Ho[2+l,ix1,iy,iz] 	= temp2
			Ho[1+l,ix2,iy,iz] = temp1
			Ho[2+l,ix2,iy,iz] = -temp2
		end
	end
	return Ho
end

𝓟x!(grid::Grid{2}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x!(reshape(Ho,(2,grid.Nx,grid.Ny)),reshape(Hi,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=true)
𝓟x!(grid::Grid{3}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x!(reshape(Ho,(2,grid.Nx,grid.Ny,grid.Nz)),reshape(Hi,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=true)
𝓟x̄!(grid::Grid{2}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x̄!(reshape(Ho,(2,grid.Nx,grid.Ny)),reshape(Hi,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=true)
𝓟x̄!(grid::Grid{3}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x̄!(reshape(Ho,(2,grid.Nx,grid.Ny,grid.Nz)),reshape(Hi,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=true)

𝓟x(grid::Grid{2}) = LinearMap{ComplexF64}(H -> vec(_𝓟x!(reshape(copy(H),(2,grid.Nx,grid.Ny)),reshape(H,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=false)
𝓟x(grid::Grid{3}) = LinearMap{ComplexF64}(H -> vec(_𝓟x!(reshape(copy(H),(2,grid.Nx,grid.Ny,grid.Nz)),reshape(H,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=false)
𝓟x̄(grid::Grid{2}) = LinearMap{ComplexF64}(H -> vec(_𝓟x̄!(reshape(copy(H),(2,grid.Nx,grid.Ny)),reshape(H,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=false)
𝓟x̄(grid::Grid{3}) = LinearMap{ComplexF64}(H -> vec(_𝓟x̄!(reshape(copy(H),(2,grid.Nx,grid.Ny,grid.Nz)),reshape(H,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=false)

