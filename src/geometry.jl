export Geometry, ridge_wg, ridge_wg_partial_etch, circ_wg, demo_shapes, εs, fεs, fεs!, kguess, fnn̂gs, nn̂gs, fnĝvds, nĝvds, matinds
export εₘₐₓ, nₘₐₓ, materials, plot_shapes #TODO: generalize these methods for materials and ε data


# Geometry(s::Vector{S}) where S<:Shape{N} where N = Geometry{N}(s)
# materials(geom::Geometry) = materials(geom.shapes)#geom.materials
# εs(geom::Geometry) = getfield.(materials(geom),:ε)
# fεs(geom::Geometry) = getfield.(materials(geom),:fε)
# εs(geom::Geometry,lm::Real) = map(f->f(lm),fεs(geom))
# fεs(geom::Geometry) = build_function(εs(geom),λ;expression=Val{false})[1]
# fεs!(geom::Geometry) = build_function(εs(geom),λ;expression=Val{false})[2]


# matinds(geom::Vector{<:Shape}) = vcat((matinds0 = map(s->findfirst(m->isequal(ε(Material(s.data)),ε(m)), materials(geom)),geom); matinds0),maximum(matinds0)+1)
matinds(geom::Vector{<:Shape}) = vcat((matinds0 = map(s->findfirst(m->isequal(get_model(Material(s.data),:ε,:λ),get_model(m,:ε,:λ)), materials(geom)),geom); matinds0),maximum(matinds0)+1)
matinds(shapes,mats) = vcat(map(s->findfirst(m->isequal(get_model(Material(s.data),:ε,:λ),get_model(m,:ε,:λ)),mats),shapes),length(mats)+1)


materials(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = Zygote.@ignore(unique(Material.(getfield.(shapes,:data)))) # # unique!(getfield.(shapes,:data))
# εs(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = getfield.(materials(shapes),:ε)
fεs(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = Zygote.@ignore(ε_fn.(materials(shapes)) )
fεs(mats::AbstractVector{<:AbstractMaterial}) = Zygote.@ignore(ε_fn.(mats) )
εs(shapes::AbstractVector{<:GeometryPrimitives.Shape},lm::Real) = map(f->SMatrix{3,3}(f(lm)),fεs(shapes))

fnn̂gs(shapes::AbstractVector{<:GeometryPrimitives.Shape}) =  Zygote.@ignore( nn̂g_fn.(materials(shapes)) )
fnn̂gs(mats::AbstractVector{<:AbstractMaterial}) =  Zygote.@ignore( nn̂g_fn.(mats) )
nn̂gs(shapes::AbstractVector{<:GeometryPrimitives.Shape},lm::Real) = [SMatrix{3,3}(f(lm)) for f in nn̂g_fn.(materials(shapes))  ] #map(f->SMatrix{3,3}(f(lm)),fnn̂gs(shapes))

fnĝvds(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = Zygote.@ignore( nĝvd_fn.(materials(shapes)) )
fnĝvds(mats::AbstractVector{<:AbstractMaterial}) = Zygote.@ignore( nĝvd_fn.(mats) )
nĝvds(shapes::AbstractVector{<:GeometryPrimitives.Shape},lm::Real) = map(f->SMatrix{3,3}(f(lm)),fnĝvds(shapes))

fχ⁽²⁾s(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = Zygote.@ignore( χ⁽²⁾_fn.(materials(shapes)) )
fχ⁽²⁾s(mats::AbstractVector{<:AbstractMaterial}) = Zygote.@ignore( χ⁽²⁾_fn.(mats) )

struct Geometry
	shapes::Vector #{<:Shape{N}}
	materials::Vector #{AbstractMaterial}
	material_inds::Vector #{Int}
	fεs::Vector #{Function}
	fnn̂gs::Vector #{Function}
	fnĝvds::Vector #{Function}
	fχ⁽²⁾s::Vector #{Function}
	# material_props::Vector{Symbol}
	# material_fns::Vector{Function}
end

function Geometry(shapes)  #where S<:Shape{N} where N
	mats =  materials(shapes)
	fes = fεs(mats)
	fnngs = fnn̂gs(mats)
	fngvds = fnĝvds(mats)
	fchi2s = fχ⁽²⁾s(mats)
	return Geometry(
		shapes,
		mats,
		matinds(shapes,mats),
		fes,
		fnngs,
		fngvds,
		fchi2s,
	)
end

# matinds(geom::Geometry) = vcat((matinds0 = map(s->findfirst(m->isequal(get_model(Material(s.data),:ε,:λ),get_model(m,:ε,:λ)), materials(geom)),geom.shapes); matinds0),maximum(matinds0)+1)
# matinds(geom::Geometry) = vcat(map(s->findfirst(m->isequal(s.data,m), materials(geom.shapes)),geom.shapes),length(geom.shapes)+1)


# fngs(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = getfield.(materials(shapes),:fng)
# ngs(shapes::AbstractVector{<:GeometryPrimitives.Shape},lm::Real) = map(f->f(lm),fngs(shapes))


# εs(shapes::AbstractVector{<:GeometryPrimitives.Shape},lm::Real) = map(fεs(shapes)) do f
# 		ee = f(lm)
# 		eeH = ( ee + ee' ) / 2
# 	end

"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""



function εₘₐₓ(ω::T,shapes::AbstractVector{<:GeometryPrimitives.Shape}) where T<:Real
    maximum(vcat(diag.(εs(shapes,inv(ω)))...))
end
function εₘₐₓ(ω::T,geom::Geometry) where T<:Real
    maximum(vcat(diag.(εs(geom.shapes,inv(ω)))...))
end

function nₘₐₓ(ω::T,shapes::AbstractVector{<:GeometryPrimitives.Shape}) where T<:Real
    sqrt( εₘₐₓ(ω,shapes) )
end
function nₘₐₓ(ω::T,geom::Geometry) where T<:Real
    sqrt( εₘₐₓ(ω,geom.shapes) )
end

kguess(ω,geom) = nₘₐₓ(ω,geom) * ω

# fεs(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = build_function(εs(shapes),λ;expression=Val{false})[1]
# fεs!(shapes::AbstractVector{<:GeometryPrimitives.Shape}) = build_function(εs(shapes),λ;expression=Val{false})[2]


# fes = map(fε,mats)
# feis = map(fε⁻¹,mats)
# εs = [fe(lm) for fe in fes]
# ε⁻¹s = [fei(lm) for fei in feis]

#
"""
################################################################################
#																			   #
#					      Parametric Geometry Methods					  	   #
#																			   #
################################################################################
"""

function ridge_wg(wₜₒₚ::Real,t_core::Real,θ::Real,edge_gap::Real,mat_core,mat_subs,Δx::Real,Δy::Real) #::Geometry{2}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    # ε_core = ε_tensor(n_core)
    # ε_subs = ε_tensor(n_subs)
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
    # verts =     [   wt_half     -wt_half     -wb_half    wb_half
    #                 tc_half     tc_half    -tc_half      -tc_half    ]'
	# verts = [   wt_half     tc_half
	# 			-wt_half    tc_half
	# 			-wb_half    -tc_half
	# 			wb_half     -tc_half    ]
	verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    # SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
					verts,
					mat_core,					                                    # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]
    b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],           	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    mat_subs,					 # data: any type, data associated with box shape
                )
    # return Geometry{2}([core,b_subs])
	return Geometry([core,b_subs])
end

function ridge_wg_partial_etch(wₜₒₚ::Real,t_core::Real,etch_frac::Real,θ::Real,edge_gap::Real,mat_core,mat_subs,Δx::Real,Δy::Real) #::Geometry{2}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    # ε_core = ε_tensor(n_core)
    # ε_subs = ε_tensor(n_subs)
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2

	t_unetch = t_core * ( 1. - etch_frac	)	# unetched thickness remaining of top layer
	c_unetch_y = -Δy/2. + edge_gap/2. + t_subs + t_unetch/2.
    # verts =     [   wt_half     -wt_half     -wb_half    wb_half
    #                 tc_half     tc_half    -tc_half      -tc_half    ]'
	# verts = [   wt_half     tc_half
	# 			-wt_half    tc_half
	# 			-wb_half    -tc_half
	# 			wb_half     -tc_half    ]
	verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    # SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
					verts,
					mat_core,					                                    # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]

	b_unetch = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_unetch_y],           	# c: center
                    [Δx - edge_gap, t_unetch ],	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    mat_core,					 # data: any type, data associated with box shape
                )

	b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],           	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    mat_subs,					 # data: any type, data associated with box shape
                )
	# return [core,b_unetch,b_subs]
	return Geometry([core,b_unetch,b_subs])
end

function demo_shapes(p::T) where T<:Real
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

    # return Geometry([ t, s, b ])
	return Geometry([ t, s, b ])
end

function circ_wg(w::T,t_core::T,edge_gap::T,n_core::T,n_subs::T,Δx::T,Δy::T)::Vector{<:GeometryPrimitives.Shape} where T<:Real
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    ax = [      1.     0.
                0.     1.      ]
    b_core = GeometryPrimitives.Sphere(					# Instantiate N-D sphere, here N=2 (circle)
                    SVector(0.,t_core),			# c: center
                    w,						# r: "radii" (half span of each axis)
                    ε_core,					        # data: any type, data associated with box shape
                )
    b_subs = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],            	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        # axes: box axes
                    ε_subs,					        # data: any type, data associated with box shape
                )
    # return Geometry([b_core,b_subs])
	return Geometry([b_core,b_subs])
end

"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""

################################################################################
#                Plotting conversions for geometry components                  #
################################################################################
import Base: convert
using AbstractPlotting: lines, lines!, scatterlines, scatterlines!, GeometryBasics, Point, PointBased
import AbstractPlotting: convert_arguments
# polygon
convert(::Type{GeometryBasics.Polygon},x::GeometryPrimitives.Polygon) = GeometryBasics.Polygon(vcat([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]], [Point2f0(x.v[1,:]),]))
AbstractPlotting.convert_arguments(P::Type{<:Poly}, x::GeometryPrimitives.Polygon) = (convert(GeometryBasics.Polygon,x),)
AbstractPlotting.convert_arguments(P::PointBased, x::GeometryPrimitives.Polygon) = (decompose(Point, convert(GeometryBasics.Polygon,x)),)
# box
convert(::Type{GeometryBasics.Rect2D},x::GeometryPrimitives.Box) = GeometryBasics.Rect((x.c-x.r)..., 2*x.r...)
convert(::Type{<:GeometryBasics.HyperRectangle},x::GeometryPrimitives.Box) = GeometryBasics.Rect((x.c-x.r)..., 2*x.r...)
convert(::Type{<:GeometryBasics.Polygon},x::GeometryPrimitives.Box) = (pts=decompose(Point,convert(GeometryBasics.Rect2D,x));GeometryBasics.Polygon(vcat(pts,[pts[1],])))
AbstractPlotting.convert_arguments(P::Type{<:Poly}, x::GeometryPrimitives.Box) = (convert(GeometryBasics.Rect2D,x),) #(GeometryBasics.Polygon(Point2f0.(coordinates(GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...)))),)
AbstractPlotting.convert_arguments(P::PointBased, x::GeometryPrimitives.Box) = (decompose(Point,convert(GeometryBasics.Rect2D,x)),)

# function plot_data(geom::Geometry)
# 	n_shapes 	= 	size(geom.shapes,1)
# 	n_mats		=	size(geom.materials,1)
# 	mat_colors 	=	getfield.(geom.materials,(:color,))
# 	shape_colors 	= [mat_colors[geom.material_inds[i]] for i=1:n_shapes]
# 	return Dict()
# end


function plot_shapes(geom::Geometry,ax;bg_color=:black,strokecolor=:white, strokewidth=2,mat_legend=true)
    ax.backgroundcolor=bg_color
	n_shapes 	= 	size(geom.shapes,1)
	n_mats		=	size(geom.materials,1)
	mat_colors 	=	getfield.(geom.materials,(:color,))
	shape_colors = [mat_colors[geom.material_inds[i]] for i=1:n_shapes]
	plys = [ poly!(
				geom.shapes[i],
				color=shape_colors[i],
				axis=ax;
				strokecolor,
				strokewidth,
			) for i=1:n_shapes ]
	if mat_legend
		mat_leg = axislegend(
			ax,
			[PolyElement(color = c, strokecolor = :black) for c in mat_colors],
			String.(nameof.(geom.materials)),
			valign=:top,
			halign=:right,
		)
	end
end

# function plot_model(geom::Geometry)
# 	[lines!(ax_disp[1], 1.2..1.7, fn; label=lbl) for (fn,lbl) in zip(plot_data(geom.materials)...)]
# end

# AbstractPlotting.convert_arguments(x::Polygon) = (GeometryBasics.Polygon([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]]),) #(GeometryBasics.Polygon([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]]),)
# plottype(::Polygon) = Poly
# AbstractPlotting.convert_arguments(x::Box) = (GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...),) #(GeometryBasics.Polygon(Point2f0.(coordinates(GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...)))),)
# plottype(::Box) = Poly

# function shape(b::Box)
#     xc,yc = b.c
#     r1,r2 = b.r
#     A = inv(b.p) .* b.r'
#     e1 = A[:,1] / √sum([a^2 for a in A[:,1]])
#     e2 = A[:,2] / √sum([a^2 for a in A[:,2]])
#     pts = [  r1*e1 + r2*e2,
#              r1*e1 - r2*e2,
#             -r1*e1 - r2*e2,
#             -r1*e1 + r2*e2,
#         ]
#     b_shape = Plots.Shape([Tuple(pt) for pt in pts])
# end
# shape(p::GeometryPrimitives.Polygon) = Plots.Shape([Tuple(p.v[i,:]) for i in range(1,length(p.v[:,1]),step=1)])
# shape(s::GeometryPrimitives.Sphere) = Plots.partialcircle(0, 2π, 100, s.r)
