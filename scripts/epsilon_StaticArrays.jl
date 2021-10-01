export ε_init, εₛ, εₛ⁻¹, ε_tensor, test_εs, εₘₐₓ, ñₘₐₓ, nₘₐₓ

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

ñₘₐₓ(ε⁻¹::AbstractArray{SHM3,3})::Float64 = √(maximum(3 ./ tr.(ε⁻¹)))
nₘₐₓ(ε::AbstractArray{SHM3,3})::Float64 = √(maximum(reinterpret(Float64,ε)))
