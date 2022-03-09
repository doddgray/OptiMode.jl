using LinearAlgebra, StaticArrays, GeometryPrimitives, OptiMode, Test
using ChainRules, Zygote #, FiniteDifferences, ForwardDiff
Δx  =   3.0
Δy  =   4.0
Δz  =   5.0
Nx  =   64
Ny  =   128
Nz  =   256

function demo_shapes2D(p::Vector{T}=rand(17)) where T<:Real
    ε₁  =   diagm([p[1],p[1],p[1]])
    ε₂  =   diagm([p[2],p[3],p[3]])
    ε₃  =   diagm([1.0,1.0,1.0])
    b = Box(					# Instantiate N-D box, here N=2 (rectangle)
        p[4:5],					# c: center
        p[6:7],				# r: "radii" (half span of each axis)
        mapreduce(
            normalize,
            hcat,
            eachcol(reshape(p[8:11],(2,2))),
        ),			# axes: box axes
        ε₁,						# data: any type, data associated with box shape
        )

    s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
        p[12:13],					# c: center
        p[14],						# r: "radii" (half span of each axis)
        ε₂,						# data: any type, data associated with circle shape
        )

    t = regpoly(				# triangle::Polygon using regpoly factory method
        3,						# k: number of vertices
        p[15],					# r: distance from center to vertices
        π/2,					# θ: angle of first vertex
        p[16:17],					# c: center
        ε₃,						# data: any type, data associated with triangle
        )

    # return Geometry([ t, s, b ])
	return ( t, s, b )
end

function smoov1_single(shapes,mat_vals,minds,crnrs::NTuple{NC,SVector{ND,T}}) where{NC,ND,T<:Real} 
    # sinds = corner_sinds(shapes,crnrs)  # indices (of `shapes`) of foreground shapes at corners of pixel/voxel
    # ps = proc_sinds(sinds)
    ps = proc_sinds(corner_sinds(shapes,crnrs))
    @inbounds if iszero(ps[2])
        return mat_vals[minds[first(ps)]]
	elseif iszero(ps[3])
        sidx1   =   ps[1]
        sidx2   =   ps[2]
        xyz     =   sum(crnrs)[1:2]/NC # sum(crnrs)/NC
        r₀_n⃗    =   surfpt_nearby(xyz, shapes[sidx1])
        rvol    =   volfrac((vxlmin(crnrs)[1:2],vxlmax(crnrs)[1:2]),reverse(r₀_n⃗)...)
        # rvol    =   volfrac((vxlmin(crnrs),vxlmax(crnrs)),reverse(r₀_n⃗)...)
        # return εₑ_∂ωεₑ_∂²ωεₑ(
        #     rvol,
        #     normcart(vec3D(n⃗)),
        #     mat_vals[:,minds[sinds[1]]],
        #     mat_vals[:,minds[sinds[2]]],
        # )
        return rvol*mat_vals[minds[sidx1]] + (1.0-rvol)*mat_vals[minds[sidx2]]
    else
        return @inbounds sum(i->mat_vals[minds[i]],ps) / NC  # naive averaging to be used
    end
end

function smoov11(shapes,mat_vals,minds,grid::Grid{ND,TG}) where {ND, TG<:Real} 
	smoothed_vals = map(corners(grid)) do crnrs
		smoov1_single(shapes,mat_vals,minds,crnrs)
	end
	return smoothed_vals
end

function smoov12(shapes,mat_vals,minds,grid::Grid{ND,TG}) where {ND, TG<:Real} 
    smoothed_vals = let shapes=shapes,mat_vals=mat_vals,minds=minds,grid=grid
        map(corners(grid)) do crnrs
		    smoov1_single(shapes,mat_vals,minds,crnrs)
        end
	end
	return smoothed_vals
end

function smoov13(shapes,mat_vals,minds,grid::Grid{ND,TG}) where {ND, TG<:Real} 
    smoothed_vals = map(corners(grid)) do crnrs
        let shapes=shapes,mat_vals=mat_vals,minds=minds
		    smoov1_single(shapes,mat_vals,minds,crnrs)
        end
	end
	return smoothed_vals
end

p2              =   rand(17)
gr2             =   Grid(Δx,Δy,Nx,Ny)
sh2             =   demo_shapes2D(p2)
cs2             =   corner_sinds.((sh2,),corners(gr2))
ps2             =   proc_sinds(corner_sinds(sh2,collect(x⃗c(gr2))))
n_unique_sinds2 =   length.(unique.(cs2))
minds2          =   1:(length(sh2)+1)
mat_vals2       =   (getproperty.(sh2, (:data,))..., diagm([0.9,0.9,0.9]),)
sm1s1           =   smoov1_single(sh2,mat_vals2,minds2,first(corners(gr2)))
sm11            =   smoov11(sh2,mat_vals2,collect(minds2),gr2)
sm12            =   smoov12(sh2,mat_vals2,collect(minds2),gr2)
sm13            =   smoov13(sh2,mat_vals2,collect(minds2),gr2)

function foo2(p)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{2,Float64}(0.1,0.1), SVector{2,Float64}(0.2,0.2), SVector{2,Float64}(0.1,0.25), SVector{2,Float64}(-0.1,0.14) )
    return smoov1_single(shapes2,matvals2,minds2,crnrs)
end

foo2s(p) = sum(foo2(p))

function foo3(p)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{3,Float64}(0.1,0.1,0.1), SVector{3,Float64}(0.2,0.2,0.2), SVector{3,Float64}(0.1,0.3,0.25), SVector{3,Float64}(-0.1,0.13,0.14) )
    return smoov1_single(shapes2,matvals2,minds2,crnrs)
end

foo3s(p) = sum(foo3(p))

p0 = rand(17)
foo2(p0)
foo2s(p0)
Zygote.gradient(foo2s,p0)

grsm1s1          =   Zygote.gradient(p2) do p
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    crnrs = (SVector{3,Float64}(0.1,0.1,0.1), SVector{3,Float64}(0.2,0.2,0.2), SVector{3,Float64}(0.1,0.3,0.25), SVector{3,Float64}(-0.1,0.13,0.14) )
    smval = smoov1_single(shapes2,matvals2,minds2,crnrs)
    return sum(smval)
end

grsm1s2          =   Zygote.gradient(p2) do p
    grid  =   Grid(3.0,4.0,64,128)
    shapes2 = demo_shapes2D(p)
    matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
    smval = smoov1_single(shapes2,matvals2,minds2,first(corners(grid)))
    return sum(smval)
end

##
grsm11          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov11(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end
grsm12          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov12(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end

grsm13          =   Zygote.gradient(p2) do p
                        shapes2 = demo_shapes2D(p)
                        matvals2 = (getproperty.(shapes2, (:data,))..., diagm([0.9,0.9,0.9]),)
                        smvals2 = smoov13(shapes2,matvals2,collect(minds2),gr2)
                        return sum(sum(smvals2))
                    end



function ff1(p)
    shapes = demo_shapes2D(p)
    # data_sum = sum( getproperty.(shapes,(:data,))) 
    data_sum = sum( map(ss->getproperty(ss,:data),shapes) ) 
    return sum(data_sum)
end
ff1(p2)
grff1 = Zygote.gradient(ff1,p2)[1]
typeof(grff1)

function ff2(p)
    shapes = demo_shapes2D(p)
    # data_sum = sum( getproperty.(shapes,(:data,))) 
    xyz = SVector{3,Float64}(0.0,0.1,0.2)
    data_sum = sum( map(ss->first(first(surfpt_nearby(xyz,ss))),shapes) ) 
    return data_sum
end
ff2(p2)
grff2 = Zygote.gradient(ff2,p2)[1]
typeof(grff2)


shapes = demo_shapes2D(rand(17))
ff31(x,y,z) = first(first(surfpt_nearby(SVector{3}(x,y,z),shapes[1])))
ff31(0.3,0.2,1.1)
Zygote.gradient(ff31,0.3,0.2,1.1)

ff312(x,y,z) = first(first(surfpt_nearby(SVector{2}(x,y),shapes[1])))
ff312(0.3,0.2,1.1)
Zygote.gradient(ff312,0.3,0.2,1.1)

ff322(x,y,z) = first(first(surfpt_nearby(SVector{2}(x,y),shapes[2])))
ff322(0.3,0.2,1.1)
Zygote.gradient(ff322,0.3,0.2,1.1)

ff32(x,y,z) = first(first(surfpt_nearby(SVector{3}(x,y,z),shapes[2])))
ff32(0.3,0.2,1.1)
Zygote.gradient(ff32,0.3,0.2,1.1)


##
println("smoov11 benchmark")
@btime smoov11(sh2,mat_vals2,collect(minds2),gr2)
println("smoov12 benchmark")
@btime smoov12(sh2,mat_vals2,collect(minds2),gr2)
println("smoov13 benchmark")
@btime smoov13(sh2,mat_vals2,collect(minds2),gr2)
##

## if plotting
using CairoMakie

let grid=gr2
    fig = Figure(resolution = (600, 400))
    x_grid = OptiMode.x(grid)
    y_grid = OptiMode.y(grid)

    ax11 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax12 = Axis(fig[1, 3]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax21 = Axis(fig[2, 1]; xlabel = "x", ylabel = "y", aspect=DataAspect())
    ax22 = Axis(fig[2, 3]; xlabel = "x", ylabel = "y", aspect=DataAspect())

    # heatmap of lower-left corner shape index ("sind") at each gridpoint
    hm2_sinds   = heatmap!(
        ax11,
        x_grid,
        y_grid,
        reshape(getindex.(cs2,(1,)),size(grid)),
    ) 
    cb11 = Colorbar(fig[1, 2], hm2_sinds; label = "foreground shape index", width = 15, ticksize = 15, tickalign = 1)
    # map of number of shapes at pixel corners, showing edges
    hm2_nunq    = heatmap!(
        ax12,
        x_grid,
        y_grid,
        reshape(n_unique_sinds2,size(grid)),
    )
    cb12 = Colorbar(fig[1, 4], hm2_nunq; label = "number of unique shape indices", width = 15, ticksize = 15, tickalign = 1)
    # heatmap of smoothed (1,1) tensor elements 
    hm2_sm11    = heatmap!(
        ax21,
        x_grid,
        y_grid,
        reshape(getindex.(sm11,(1,)),size(grid)),
    )
    cb21 = Colorbar(fig[2, 2], hm2_sm11; label = "smoothed diagonal tensor value", width = 15, ticksize = 15, tickalign = 1)
    # heatmap of smoothed (1,2) tensor elements 
    # hm2_sm12    = heatmap!(ax22,reshape(getindex.(sm11,(2,)),size(gr2)))
    # cb22 = Colorbar(fig[2, 4], hm2_sm12; label = "values", width = 15, ticksize = 15, tickalign = 1)

    
    
    #colsize!(fig.layout, 1, Aspect(1, 1.0))
    #colgap!(fig.layout, 7)
    # display(fig)
    fig
end


##
@show gr2_inds        =   rand(1:length(gr2)) # rand(eachindex(gr2))
crnrs2          =   collect(corners(gr2))[gr2_inds]
@show smval2          =   smoov_single1(sh2,mat_vals2,minds2,crnrs2)


sm11            =   smoov_single1(sh2,mat_vals2,minds2,first(corners(gr2)))
sm1             =   smoov1(sh2,mat_vals2,collect(minds2),gr2)
##


gr3 =   Grid(Δx,Δy,Δz,Nx,Ny,Nz)

@test xc(gr2) ≈ xc(gr3)
@test yc(gr2) ≈ yc(gr3)
# @test zc(gr2) ≈ zc(gr3)
@test xc(gr2)[1:end-1]  ≈   ( OptiMode.x(gr2) .- (δx(gr2)/2.0,) )
@test xc(gr2)[2:end]    ≈   ( OptiMode.x(gr2) .+ (δx(gr2)/2.0,) )
@test yc(gr2)[1:end-1]  ≈   ( OptiMode.y(gr2) .- (δy(gr2)/2.0,) )
@test yc(gr2)[2:end]    ≈   ( OptiMode.y(gr2) .+ (δy(gr2)/2.0,) )

@test xc(gr3)[1:end-1]  ≈   ( OptiMode.x(gr3) .- (δx(gr3)/2.0,) )
@test xc(gr3)[2:end]    ≈   ( OptiMode.x(gr3) .+ (δx(gr3)/2.0,) )
@test yc(gr3)[1:end-1]  ≈   ( OptiMode.y(gr3) .- (δy(gr3)/2.0,) )
@test yc(gr3)[2:end]    ≈   ( OptiMode.y(gr3) .+ (δy(gr3)/2.0,) )
@test zc(gr3)[1:end-1]  ≈   ( OptiMode.z(gr3) .- (δz(gr3)/2.0,) )
@test zc(gr3)[2:end]    ≈   ( OptiMode.z(gr3) .+ (δz(gr3)/2.0,) )