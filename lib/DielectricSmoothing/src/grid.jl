"""
##############################################################################################
#																			   				 #
#		Define a spatial Grid type and utility/convenience functions acting on Grids		 #
#																			   				 #
##############################################################################################
"""

export Grid, δx, δy, δz, δV, x⃗, xc, yc, zc, x⃗c, N, g⃗, _fftaxes,
	corners, vxlmin, vxlmax, my_fftfreq
export δ, x, y, z # TODO: remove these functions, their names are too short and likely to cause problems

"""
    Grid(Δx, Δy, Nx, Ny)            -> Grid{2}
    Grid(Δx, Δy, Δz, Nx, Ny, Nz)    -> Grid{3}

A uniform finite-difference spatial grid: a rectangular cell of physical size
`Δx × Δy (× Δz)` (μm) centered on the origin, discretized into `Nx × Ny (× Nz)`
pixels (voxels). Pixel *centers* lie at

```math
x_i = -\\frac{Δx}{2} + (i-1)\\,δx, \\qquad δx = \\frac{Δx}{N_x}, \\quad i = 1…N_x
```

(and similarly in y/z), so the cell spans `[-Δx/2, Δx/2)` with periodic boundary
conditions implied by the plane-wave (FFT) basis of the eigensolver:

```text
        ┌──┬──┬──┬──┐  ↑           ●  pixel centers x(g) × y(g)
        ├──●──┼──●──┤  │           ┼  pixel corners (used by `corners` for
        ├──┼──┼──┼──┤  Δy             sub-pixel smoothing)
        ├──●──┼──●──┤  │
        └──┴──┴──┴──┘  ↓
          ←── Δx ──→
```

`Grid` acts like an `AbstractArray` of pixel-center coordinates: `size`, `length`,
`eltype` (`SVector{3}`), iteration and indexing are defined. Key accessors:

- [`x`](@ref), [`y`](@ref), [`z`](@ref): coordinate vectors of pixel centers
- [`δx`](@ref), [`δy`](@ref), [`δz`](@ref), [`δV`](@ref): pixel pitches and volume
- [`x⃗`](@ref): array of `SVector{3}` pixel-center positions
- [`corners`](@ref): pixel-corner tuples for interface detection
- [`g⃗`](@ref): reciprocal-lattice (spatial-frequency) vectors of the FFT basis
"""
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

"""
    δx(g::Grid), δy(g::Grid), δz(g::Grid)

Pixel pitch along each axis, e.g. `δx(g) = g.Δx / g.Nx` (μm).
"""
δx(g::Grid) = g.Δx / g.Nx
δy(g::Grid) = g.Δy / g.Ny
δz(g::Grid) = g.Δz / g.Nz

function δ(g::Grid{2})
	@warn "Deprecation Warning: `δ(::Grid)` needs to be replaced with `δV(::Grid)`"
	g.Δx * g.Δy / ( g.Nx * g.Ny )
end
function δ(g::Grid{3})
	@warn "Deprecation Warning: `δ(::Grid)` needs to be replaced with `δV(::Grid)`"
	g.Δx * g.Δy * g.Δz / ( g.Nx * g.Ny * g.Nz )
end

"""
    δV(g::Grid)

Pixel area (2D, μm²) or voxel volume (3D, μm³): `δx(g)*δy(g)[*δz(g)]`. Integrals of
gridded fields are evaluated as `sum(field) * δV(g)`.
"""
δV(g::Grid{2}) = g.Δx * g.Δy / ( g.Nx * g.Ny )
δV(g::Grid{3}) = g.Δx * g.Δy * g.Δz / ( g.Nx * g.Ny * g.Nz )


# ∫dV(intgnd,g::Grid) = sum(intgnd)*δ(g)	# integrate a scalar field over the grid

"""
`myrange(a,b,N)` remimplements the functionality of the Base method `range(a,b,N)` in a Zygote compatible way
This is just a temporary hack until an rrule for either of the LinRange or StepRangeLen constructors is added to ChainRules.jl 
"""
function myrange(a::Real,b::Real,N::Int)
    step = (b-a)/(N-1.0)
    map(i->(a+step*i),0:(N-1))
end

### Grid Points (Voxel/Pixel Center Positions) ###
"""
    x(g::Grid), y(g::Grid), z(g::Grid)

Coordinate vectors of the pixel/voxel *centers* along each axis: `Nx` (`Ny`, `Nz`)
uniformly spaced values starting at `-Δx/2` with pitch `δx(g)` (the cell is
origin-centered and periodic, so the right edge `+Δx/2` is the image of the left).
"""
function x(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δx / g.Nx ) .* (0:(g.Nx-1))) .- g.Δx/2.
	myrange(-g.Δx/2, g.Δx/2 - g.Δx/g.Nx, g.Nx)
end
function y(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δy / g.Ny ) .* (0:(g.Ny-1))) .- g.Δy/2.
	myrange(-g.Δy/2, g.Δy/2 - g.Δy/g.Ny, g.Ny)
end
function z(g::Grid{ND,T})::Vector{T}  where {ND,T<:Real}
 	# ( ( g.Δz / g.Nz ) .* (0:(g.Nz-1))) .- g.Δz/2.
	myrange(-g.Δz/2, g.Δz/2 - g.Δz/g.Nz, g.Nz)
end
function x⃗(g::Grid{2,T})::Array{SVector{3,T},2} where T<:Real
	# [ SVector{3,T}(xx,yy,zero(T)) for xx in x(g), yy in y(g) ] # (Nx × Ny ) 2D-Array of (x,y,z) vectors at pixel/voxel centers
	map( xy->SVector{3}(first(xy),last(xy),0.0), Iterators.product(x(g),y(g)))
end
function x⃗(g::Grid{3,T})::Array{SVector{3,T},3} where T<:Real
	[ SVector{3,T}(xx,yy,zz) for xx in x(g), yy in y(g), zz in z(g) ] # (Nx × Ny × Nz) 3D-Array of (x,y,z) vectors at pixel/voxel centers
end

import Base: ndims, length, size, firstindex, lastindex, eachindex, getindex, LinearIndices, CartesianIndices, Dims, iterate, eltype
### Size & Dimensionality ###
@inline size(gr::Grid{3})			=	(gr.Nx, gr.Ny, gr.Nz)
@inline size(gr::Grid{2})			=	(gr.Nx, gr.Ny)
@inline size(gr::Grid, d::Int) 		= 	getindex((gr.Nx, gr.Ny, gr.Nz),d)
@inline Dims(gr::Grid)				=	size(gr)
@inline N(g::Grid)::Int 			= 	prod(size(g))	# legacy, TODO: replace all instances of N(g::Grid) with size(g::Grid)
@inline length(g::Grid)::Int 		= 	prod(size(g))
@inline LinearIndices(g::Grid) 		= 	1:length(g)
@inline CartesianIndices(g::Grid) 	= 	CartesianIndices(size(g))
@inline firstindex(g::Grid)			=	1
@inline lastindex(g::Grid)			=	length(g)
@inline eachindex(g::Grid) 			= 	CartesianIndices(g) # 1:length(g) if you want to use LinearIndices by default

@inline function eltype(g::Grid{ND,T}) where {ND,T<:Real}
	return SVector{3,T}
end

@inline function ndims(g::Grid{ND}) where ND
	return ND
end

### Iteration ###
"""
iterate over a (Nx × Ny ) two dimensional grid of (x,y,z) vectors (of type `SVector{3}`) at pixel centers
"""
@inline function iterate(g::Grid{2,T}) where T<:Real
	iterate(( SVector{3,T}(xx,yy,zero(T)) for xx in x(g), yy in y(g) ))
end

@inline function iterate(g::Grid{2,T}, state) where T<:Real
	iterate(( SVector{3,T}(xx,yy,zero(T)) for xx in x(g), yy in y(g) ), state)
end

"""
iterate over a (Nx × Ny × Nz ) three dimensional grid of (x,y,z) vectors (of type `SVector{3}`) at voxel centers
"""
@inline function iterate(g::Grid{3,T}) where T<:Real
	iterate( ( SVector{3,T}(xx,yy,zz) for xx in x(g), yy in y(g), zz in z(g) ) )
end

@inline function iterate(g::Grid{3,T}, state) where T<:Real
	iterate( ( SVector{3,T}(xx,yy,zz) for xx in x(g), yy in y(g), zz in z(g) ), state )
end


### Indexing ###
function getindex(g::Grid{2,T}, I::CartesianIndex) where{T<:Real}
	# @inbounds SVector{3,T}(x(g)[I[1]],y(g)[I[2]],zero(T))
	@inbounds SVector{3,T}(
		g.Δx*((I[1]-1)*inv(g.Nx)-0.5),
		g.Δy*((I[2]-1)*inv(g.Ny)-0.5),
		zero(T),
	)
end

function getindex(g::Grid{2,T}, ix::Int, iy::Int) where{T<:Real}
	# @inbounds SVector{3,T}(x(g)[I[1]],y(g)[I[2]],zero(T))
	@inbounds SVector{3,T}(
		g.Δx*((ix-1)*inv(g.Nx)-0.5),
		g.Δy*((iy-1)*inv(g.Ny)-0.5),
		zero(T),
	)
end

function getindex(g::Grid{3,T}, I::CartesianIndex) where{T<:Real}
	# @inbounds SVector{3,T}(x(g)[I[1]],y(g)[I[2]],zero(T))
	@inbounds SVector{3,T}(
		g.Δx*((I[1]-1)*inv(g.Nx)-0.5),
		g.Δy*((I[2]-1)*inv(g.Ny)-0.5),
		g.Δz*((I[3]-1)*inv(g.Nz)-0.5),
	)
end

function getindex(g::Grid{3,T}, ix::Int, iy::Int, iz::Int) where{T<:Real}
	# @inbounds SVector{3,T}(x(g)[I[1]],y(g)[I[2]],zero(T))
	@inbounds SVector{3,T}(
		g.Δx*((ix-1)*inv(g.Nx)-0.5),
		g.Δy*((iy-1)*inv(g.Ny)-0.5),
		g.Δz*((iz-1)*inv(g.Nz)-0.5),
	)
end

@inline function getindex(g::Grid{ND,T}, idx::Int) where{ND,T<:Real}
	return getindex(g,CartesianIndices(g)[idx])
end

### Reciprocal Lattice Axes and Vectors (from fftfreqs) ###
"""
    _fftaxes(g::Grid)

The array axes along which spatial FFTs act for fields stored as
`(components, Nx, Ny[, Nz])` arrays: `2:3` in 2D, `2:4` in 3D (axis 1 indexes field
components).
"""
@inline _fftaxes(gr::Grid{2}) = (2:3)
@inline _fftaxes(gr::Grid{3}) = (2:4)

"""
fftfreq without special type for AD compatibility
"""
function my_fftfreq(n::Int,fs::Real)
	iseven(n) ? [0:n÷2-1; -n÷2:-1]*fs/n	: [0:(n-1)÷2; -(n-1)÷2:-1]*fs/n
end

"""
    g⃗(g::Grid)

Array of reciprocal-lattice vectors ``\\vec{G}`` of the periodic cell, in FFT
frequency order: `G⃗[i,j] = (gx[i], gy[j], 0)` with `gx = fftfreq(Nx, Nx/Δx)` etc.
All spatial frequencies in this package are in cycles/μm (consistent with `ω = 1/λ`
and `k = n_eff/λ`; no factors of 2π appear). These are the plane-wave basis
frequencies of the Helmholtz eigenproblem, which involves the shifted magnitudes
`|k⃗ - G⃗|` computed by `MaxwellEigenmodes.mag_m_n`.
"""
function g⃗(gr::Grid{2,T})::Array{SVector{3, T}, 2} where T<:Real
	[	SVector{3,T}(gx,gy,0.) for  gx in my_fftfreq(gr.Nx,gr.Nx/gr.Δx),
							   		gy in my_fftfreq(gr.Ny,gr.Ny/gr.Δy)		]
end

function g⃗(gr::Grid{3,T})::Array{SVector{3, T}, 3} where T<:Real
	[	SVector{3,T}(gx,gy,gz) for  gx in my_fftfreq(gr.Nx,gr.Nx/gr.Δx),
									gy in my_fftfreq(gr.Ny,gr.Ny/gr.Δy),
							    	gz in my_fftfreq(gr.Nz,gr.Nz/gr.Δz)		]
end

# @non_differentiable g⃗(::Any)
@non_differentiable _fftaxes(::Any)

### Iterators Over Grid Voxel/Pixel Corner Positions ###

@inline function xc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	myrange(-(g.Δx + δx(g))/2, (g.Δx - δx(g))/2, g.Nx+1)
end

@inline function yc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	myrange(-(g.Δy + δy(g))/2, (g.Δy - δy(g))/2, g.Ny+1)
end

@inline function zc(g::Grid{ND,T})::Vector{T} where {ND,T<:Real}
	myrange(-(g.Δz + δz(g))/2, (g.Δz - δz(g))/2, g.Nz+1)
end

@inline function x⃗c(g::Grid{2,T})  where T<:Real
	( SVector{3,T}(xx,yy,zero(T)) for xx in xc(g), yy in yc(g) )
end

@inline function x⃗c(g::Grid{3,T})  where T<:Real
	( SVector{3,T}(xx,yy,zz) for xx in xc(g), yy in yc(g), zz in zc(g) )
end

"""
    corners(g::Grid)

For every pixel (voxel), the tuple of its 4 (8) corner positions as `SVector{ND}`s,
returned as an array shaped like the grid. Sub-pixel smoothing
([`smooth_ε`](@ref)/[`smooth_scalar`](@ref)) classifies each pixel by evaluating which
shape is foreground at each of its corners.
"""
function corners(g::Grid{2,T}) where T<:Real

	xcs = xc(g)
	ycs = yc(g)
	map(eachindex(g)) do I
		x0 = xcs[I[1]]
		x1 = xcs[I[1]+1]
		y0 = ycs[I[2]]
		y1 = ycs[I[2]+1]
		( 
			SVector{2}( x0, y0 ),
			SVector{2}( x0, y1 ),
			SVector{2}(	x1, y1 ),
			SVector{2}(	x1, y0 ),
		)
	end
end

function corners(g::Grid{3,T}) where T<:Real

	xcs = xc(g)
	ycs = yc(g)
	zcs = yc(g)
	map(eachindex(g)) do I
		x0 = xcs[I[1]]
		x1 = xcs[I[1]+1]
		y0 = ycs[I[2]]
		y1 = ycs[I[2]+1]
		z0 = zcs[I[2]]
		z1 = zcs[I[2]+1]
		( 
			SVector{3}( x0, y0, z0 ),
			SVector{3}( x0, y1, z0 ),
			SVector{3}(	x1, y1, z0 ),
			SVector{3}(	x1, y0, z0 ),
			SVector{3}( x0, y0, z1 ),
			SVector{3}( x0, y1, z1 ),
			SVector{3}(	x1, y1, z1 ),
			SVector{3}(	x1, y0, z1 ),
		)
	end
end

@inline function vxlmin(g::Grid{2,T}) where T<:Real
	(	xx + SVector{3,T}( -δx(g), -δy(g),	zero(T) )	for xx in g		)
end

@inline function vxlmax(g::Grid{2,T}) where T<:Real
	(	xx + SVector{3,T}(  δx(g),  δy(g),	zero(T) )	for xx in g		)
end

@inline function vxlmin(g::Grid{3,T}) where T<:Real
	(	xx + SVector{3,T}( -δx(g), -δy(g), -δz(g) )		for xx in g		)
end

@inline function vxlmax(g::Grid{3,T}) where T<:Real
	(	xx + SVector{3,T}(  δx(g),  δy(g),	δz(g) )		for xx in g		)
end

@inline function vxlmin(crnrs::NTuple{NC,SVector{ND,T}}) where {NC,ND,T<:Real}
	first(crnrs)
end

@inline function vxlmax(crnrs::NTuple{4,SVector{ND,T}}) where {ND,T<:Real}
	@inbounds crnrs[3]
end

@inline function vxlmax(crnrs::NTuple{8,SVector{ND,T}}) where {ND,T<:Real}
	@inbounds crnrs[7]
end