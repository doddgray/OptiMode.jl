export Grid, δx, δy, δz, x, y, z, x⃗, xc, yc, zc, x⃗c, N, g⃗, _fftaxes

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

Base.size(gr::Grid{3}) = (gr.Nx, gr.Ny, gr.Nz)
Base.size(gr::Grid{2}) = (gr.Nx, gr.Ny)
Base.size(gr::Grid,d::Int) = Base.size(gr)[d]

δx(g::Grid) = g.Δx / g.Nx
δy(g::Grid) = g.Δy / g.Ny
δz(g::Grid) = g.Δz / g.Nz
# voxel/pixel center positions
function x(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	( ( g.Δx / g.Nx ) .* (0:(g.Nx-1))) .- g.Δx/2.
end
function y(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	( ( g.Δy / g.Ny ) .* (0:(g.Ny-1))) .- g.Δy/2.
end
function z(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	( ( g.Δz / g.Nz ) .* (0:(g.Nz-1))) .- g.Δz/2.
end
function x⃗(g::Grid{2,T})::Array{SVector{3,T},2} where T<:Real
	( (xx,yy) = (x(g),y(g)); [SVector{3,T}(xx[ix],yy[iy],0.) for ix=1:g.Nx,iy=1:g.Ny] ) # (Nx × Ny ) 2D-Array of (x,y,z) vectors at pixel/voxel centers
end
function x⃗(g::Grid{3,T})::Array{SVector{3,T},3} where T<:Real
	( (xx,yy,zz) = (x(g),y(g),z(g)); [SVector{3,T}(xx[ix],yy[iy],zz[iz]) for ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz] ) # (Nx × Ny × Nz) 3D-Array of (x,y,z) vectors at pixel/voxel centers
end

# voxel/pixel corner positions
function xc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	collect( ( ( g.Δx / g.Nx ) .* (0:g.Nx) ) .- ( g.Δx/2. * ( 1 + 1. / g.Nx ) ) )
end
function yc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	collect( ( ( g.Δy / g.Ny ) .* (0:g.Ny) ) .- ( g.Δy/2. * ( 1 + 1. / g.Ny ) ) )
end
function zc(g::Grid{3,T})::Vector{T} where T<:Real
	collect( ( ( g.Δz / g.Nz ) .* (0:g.Nz) ) .- ( g.Δz/2. * ( 1 + 1. / g.Nz ) ) )
end
function x⃗c(g::Grid{2,T})::Array{SVector{3,T},2}  where T<:Real
	( (xx,yy) = (xc(g),yc(g)); [SVector{3}(xx[ix],yy[iy],0.) for ix=1:(g.Nx+1),iy=1:(g.Ny+1)] )
end
function x⃗c(g::Grid{3,T})::Array{SVector{3,T},3}  where T<:Real
	( (xx,yy,zz) = (xc(g),yc(g),zc(g)); [SVector{3}(xx[ix],yy[iy],zz[iz]) for ix=1:(g.Nx+1),iy=1:(g.Ny+1),iz=1:(g.Nz+1)] )
end

# grid size
N(g::Grid)::Int = *(size(g)...)

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
