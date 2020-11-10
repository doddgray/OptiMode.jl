################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                 (an attempt)                                 #
#                                                                              #
################################################################################

module OptiMode

## Imports ##
using Plots: plot, heatmap, plot!, heatmap!
using MaxwellFDM: kottke_avg_param
using LinearAlgebra, FFTW, LinearMaps, StaticArrays, GeometryPrimitives, Roots

## Exports ##
export plot_ε, ε_init, εₛ, εₛ⁻¹, ε_tensor, test_εs, test_shapes, ridge_wg, circ_wg, plot, heatmap



## Abbreviations, aliases, etc. ##
SHM3 = SHermitianCompact{3,Float64,6}   # static Hermitian 3×3 matrix Type, renamed for brevity

## Add methods to external packages ##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

## Includes ##
# include("geometry.jl")
# include("smooth.jl")
# include("grads.jl")
# include("solve.jl")
# include("optimize.jl")
include("maxwell.jl")
include("plot.jl")
# include("materials.jl")

## Definitions ##



εᵥ = SHM3( [	1. 	0. 	0. 
                                0. 	1. 	0. 
                                0. 	0. 	1.  ]
)

function ε_tensor(n::Float64)
    n² = n^2 
    ε = SHM3( [	n²      0. 	    0. 
                0. 	    n² 	    0. 
                0. 	    0. 	    n²  ]
    )
end

function test_εs(n₁::Float64,n₂::Float64,n₃::Float64)
    ε₁ = ε_tensor(n₁)
    ε₂ = ε_tensor(n₂)
    ε₃ = ε_tensor(n₃)
    return ε₁, ε₂, ε₃
end

function test_shapes(p::Float64)
    ε₁, ε₂, ε₃ = test_εs(1.42,2.2,3.5)
    ax1b,ax2b = GeometryPrimitives.normalize.(([1.,0.2], [0.,1.]))
    b = Box(					# Instantiate N-D box, here N=2 (rectangle)
        [0,0],					# c: center
        [3.0, 3.0],				# r: "radii" (half span of each axis)
        [ax1b ax2b],			# axes: box axes
        ε₁,						# data: any type, data associated with box shape
        )

    s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
        [0,0],					# c: center
        p,						# r: "radii" (half span of each axis)
        ε₂,						# data: any type, data associated with circle shape
        )
    
    t = regpoly(				# triangle::Polygon using regpoly factory method
        3,						# k: number of vertices
        0.8,					# r: distance from center to vertices
        π/2,					# θ: angle of first vertex
        [0,0],					# c: center
        ε₃,						# data: any type, data associated with triangle
        )

    return [ t, s, b ]
end


function ridge_wg(w::Float64,t_core::Float64,edge_gap::Float64,n_core::Float64,n_subs::Float64,g::MaxwellGrid)::Array{GeometryPrimitives.Shape{2,4,SHM3},1}
    t_subs = (g.Δy -t_core - edge_gap )/2.
    c_subs_y = -g.Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    ax1,ax2 = GeometryPrimitives.normalize.(([1.,0.], [0.,1.]))
    b_core = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0.  ,   0.  ],			# c: center
                    [w  ,   t_core      ],			# r: "radii" (half span of each axis)
                    [ax1 ax2],	    		        # axes: box axes
                    ε_core,					        # data: any type, data associated with box shape
                )
    b_subs = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],            	# c: center
                    [g.Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    [ax1 ax2],	    		        # axes: box axes
                    ε_subs,					        # data: any type, data associated with box shape
                )
    return [b_core,b_subs]
end

function circ_wg(w::Float64,t_core::Float64,edge_gap::Float64,n_core::Float64,n_subs::Float64,g::MaxwellGrid)::Array{GeometryPrimitives.Shape{2,4,SHM3},1}
    t_subs = (g.Δy -t_core - edge_gap )/2.
    c_subs_y = -g.Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    ax1,ax2 = GeometryPrimitives.normalize.(([1.,0.], [0.,1.]))
    b_core = GeometryPrimitives.Sphere(					# Instantiate N-D sphere, here N=2 (circle)
                    SVector(0.,t_core),			# c: center
                    w,						# r: "radii" (half span of each axis)
                    ε_core,					        # data: any type, data associated with box shape
                )
    b_subs = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],            	# c: center
                    [g.Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    [ax1 ax2],	    		        # axes: box axes
                    ε_subs,					        # data: any type, data associated with box shape
                )
    return [b_core,b_subs]
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D;Δx=6.,Δy=4.,Nx=64,Ny=64)::Array{SHM3,3} 
    g = MaxwellGrid(Δx,Δy,Nx,Ny)
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellData)::Array{SHM3,3} 
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D, x::AbstractVector{Float64}, y::AbstractVector{Float64})::Array{SHM3,3} 
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=x,yy=y,i=1:3,j=1:3 ]
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,tree::KDTree,x::Real,y::Real,δx::Real,δy::Real)::SHM3 
    x1,y1 = x+δx/2.,y+δy/2
    x2,y2 = x+δx/2.,y-δy/2
    x3,y3 = x-δx/2.,y-δy/2
    x4,y4 = x-δx/2.,y+δy/2

    s1 = findfirst([x1,y1],tree)
    s2 = findfirst([x2,y2],tree)
    s3 = findfirst([x3,y3],tree)
    s4 = findfirst([x4,y4],tree)

    ε1 = isnothing(s1) ? εᵥ : s1.data
    ε2 = isnothing(s2) ? εᵥ : s2.data
    ε3 = isnothing(s3) ? εᵥ : s3.data
    ε4 = isnothing(s4) ? εᵥ : s4.data
    
    if (ε1==ε2==ε3==ε4)
        return ε1
    else
        sinds = [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]]
        s_fg = shapes[min(sinds...)]
        r₀,nout = surfpt_nearby([x, y], s_fg)
        # bndry_pxl[i,j] = 1
        # nouts[i,j,:] = nout
        vxl = (SVector{2,Float64}(x3,y3), SVector{2,Float64}(x1,y1))
        rvol = volfrac(vxl,nout,r₀)
        sind_bg = max(sinds...)
        ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
        return SHM3(kottke_avg_param(
                SHM3(s_fg.data),
                SHM3(ε_bg),
                SVector{3,Float64}(nout[1],nout[2],0),
                rvol,))
    end
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,Δx=6.,Δy=4.,Nx=64,Ny=64)::Array{SHM3,3}
    g=MaxwellGrid(Δx,Δy,Nx,Ny)
    tree = KDTree(shapes)
    ε_sm = zeros(Float64,g.Nx,g.Ny,3,3)
    for i=1:g.Nx, j=1:g.Ny
        ε_sm[i,j,:,:] = εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)
    end
    return ε_sm
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellGrid)::Array{SHM3,3} 
    tree = KDTree(shapes)
    ε_sm = copy(reshape(  [εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy) for i=1:g.Nx,j=1:g.Ny] , (g.Nx,g.Ny,1)) )
end

# function εₛ⁻¹(shapes::AbstractVector{T},g::MaxwellGrid) where T <: GeometryPrimitives.Shape{2,4,D} where D
function εₛ⁻¹(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellGrid)::Array{SHM3,3} 
    tree = KDTree(shapes)
    ε_sm_inv = copy(reshape( [SHM3(inv(εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy))) for i=1:g.Nx,j=1:g.Ny], (g.Nx,g.Ny,1)) )
end

function εₘₐₓ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D) 
    maximum(vec([shapes[i].data[j,j] for j=1:3,i=1:size(shapes)[1]]))
end


## More includes ##
include("solve.jl")

end
