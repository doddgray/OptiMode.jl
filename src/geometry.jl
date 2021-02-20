export Geometry, ridge_wg, circ_wg, demo_shapes

struct Geometry{N}
	shapes::Vector{Shape{N}}
end
Geometry(s::Vector{S}) where S<:Shape{N} where N = Geometry{N}(s)
materials(geom::Geometry) = materials(geom.shapes)#geom.materials

# Specific Geometry Cases

function ridge_wg(wₜₒₚ::Real,t_core::Real,θ::Real,edge_gap::Real,mat_core,mat_subs,Δx::Real,Δy::Real)::Geometry{2}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    # ε_core = ε_tensor(n_core)
    # ε_subs = ε_tensor(n_subs)
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
    # verts =     [   wt_half     -wt_half     -wb_half    wb_half
    #                 tc_half     tc_half    -tc_half      -tc_half    ]'
	verts = [   wt_half     tc_half
				-wt_half    tc_half
				-wb_half    -tc_half
				wb_half     -tc_half    ]
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
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
    return Geometry([core,b_subs])
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
    return Geometry([b_core,b_subs])
end
