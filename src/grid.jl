"""
##############################################################################################
#																			   				 #
#		Define a spatial Grid type and utility/convenience functions acting on Grids		 #
#																			   				 #
##############################################################################################
"""

export Grid, δx, δy, δz, δ, x, y, z, x⃗, xc, yc, zc, x⃗c, N, g⃗, _fftaxes, Nranges, ∫

struct Grid{ND,T}
	Δx::T
    Δy::T
    Δz::T
	Nx::Int
    Ny::Int
    Nz::Int
end

Grid(Δx::T, Δy::T, Δz::T, Nx::Int, Ny::Int, Nz::Int) where {T<:Real} = Grid{3,T}(
	Δx,
    Δy,
    Δz,
	Nx,
    Ny,
    Nz,
)

Grid(Δx::T, Δy::T, Nx::Int, Ny::Int) where {T<:Real} = Grid{2,T}(
	Δx,
    Δy,
    1.,
	Nx,
    Ny,
    1,
)

function Base.size(gr::Grid{3})::NTuple{3,Int}
	(gr.Nx, gr.Ny, gr.Nz)
end
function Base.size(gr::Grid{2})::NTuple{2,Int}
	(gr.Nx, gr.Ny)
end
Base.size(gr::Grid,d::Int) = Base.size(gr)[d]


δx(g::Grid) = g.Δx / g.Nx
δy(g::Grid) = g.Δy / g.Ny
δz(g::Grid) = g.Δz / g.Nz

δ(g::Grid{2}) = g.Δx * g.Δy / ( g.Nx * g.Ny )
δ(g::Grid{3}) = g.Δx * g.Δy * g.Δz / ( g.Nx * g.Ny * g.Nz )

∫(intgnd;g::Grid) = sum(intgnd)*δ(g)	# integrate a scalar field over the grid

# voxel/pixel center positions
function x(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δx / g.Nx ) .* (0:(g.Nx-1))) .- g.Δx/2.
	LinRange(-g.Δx/2, g.Δx/2 - g.Δx/g.Nx, g.Nx)
end
function y(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δy / g.Ny ) .* (0:(g.Ny-1))) .- g.Δy/2.
	LinRange(-g.Δy/2, g.Δy/2 - g.Δy/g.Ny, g.Ny)
end
function z(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δz / g.Nz ) .* (0:(g.Nz-1))) .- g.Δz/2.
	LinRange(-g.Δz/2, g.Δz/2 - g.Δz/g.Nz, g.Nz)
end
function x⃗(g::Grid{2,T})::Array{SVector{3,T},2} where T<:Real
	[ SVector{3,T}(xx,yy,zero(T)) for xx in x(g), yy in y(g) ] # (Nx × Ny ) 2D-Array of (x,y,z) vectors at pixel/voxel centers
end
function x⃗(g::Grid{3,T})::Array{SVector{3,T},3} where T<:Real
	[ SVector{3,T}(xx,yy,zz) for xx in x(g), yy in y(g), zz in z(g) ] # (Nx × Ny × Nz) 3D-Array of (x,y,z) vectors at pixel/voxel centers
end

# voxel/pixel corner positions
function xc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	( ( g.Δx / g.Nx ) .* (0:g.Nx) ) .- ( g.Δx/2. * ( 1 + 1. / g.Nx ) )
	# collect(range(-g.Δx/2.0, g.Δx/2.0, length=g.Nx+1))
end
function yc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	( ( g.Δy / g.Ny ) .* (0:g.Ny) ) .- ( g.Δy/2. * ( 1 + 1. / g.Ny ) )
	# collect(range(-g.Δy/2.0, g.Δy/2.0, length=g.Ny+1))
end
function zc(g::Grid{3,T})::Vector{T} where T<:Real
	( ( g.Δz / g.Nz ) .* (0:g.Nz) ) .- ( g.Δz/2. * ( 1 + 1. / g.Nz ) )
	# collect(range(-g.Δz/2.0, g.Δz/2.0, length=g.Nz+1))
end
function x⃗c(g::Grid{2,T})::Array{SVector{3,T},2}  where T<:Real
	# ( (xx,yy) = (xc(g),yc(g)); [SVector{3}(xx[ix],yy[iy],0.) for ix=1:(g.Nx+1),iy=1:(g.Ny+1)] )
	[ SVector{3,T}(xx,yy,zero(T)) for xx in xc(g), yy in yc(g) ]
end
function x⃗c(g::Grid{3,T})::Array{SVector{3,T},3}  where T<:Real
	# ( (xx,yy,zz) = (xc(g),yc(g),zc(g)); [SVector{3}(xx[ix],yy[iy],zz[iz]) for ix=1:(g.Nx+1),iy=1:(g.Ny+1),iz=1:(g.Nz+1)] )
	[ SVector{3,T}(xx,yy,zz) for xx in xc(g), yy in yc(g), zz in zc(g) ]
end

# grid size
@inline N(g::Grid)::Int = *(size(g)...)

import Base: eachindex
@inline eachindex(g::Grid) = CartesianIndices(size(g)) #(1:NN for NN in size(g))

import Base: ndims
@inline function ndims(g::Grid{ND}) where ND
	return ND
end

# reciprocal lattice vectors (from fftfreqs)
function g⃗(gr::Grid{3,T})::Array{SVector{3, T}, 3} where T<:Real
	[	SVector(gx,gy,gz) for   gx in fftfreq(gr.Nx,gr.Nx/gr.Δx),
								gy in fftfreq(gr.Ny,gr.Ny/gr.Δy),
							    gz in fftfreq(gr.Nz,gr.Nz/gr.Δz)		]
end
_fftaxes(gr::Grid{3}) = (2:4)

function g⃗(gr::Grid{2,T})::Array{SVector{3, T}, 2} where T
	 [	SVector(gx,gy,0.) for   gx in fftfreq(gr.Nx,gr.Nx/gr.Δx),
								gy in fftfreq(gr.Ny,gr.Ny/gr.Δy)		]
end
_fftaxes(gr::Grid{2}) = (2:3)


"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""
