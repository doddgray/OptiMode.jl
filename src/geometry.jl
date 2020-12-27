# export

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

function ridge_wg(wₜₒₚ::Float64,t_core::Float64,θ::Float64,edge_gap::Float64,n_core::Float64,n_subs::Float64,Δx::Float64,Δy::Float64)::Vector{<:Shape}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    tanθ = tan(θ)
    tcore_tanθ = t_core*tanθ
    w_bottom = wₜₒₚ + 2*tcore_tanθ
    verts = 0.5.*   [   wₜₒₚ     -wₜₒₚ     -w_bottom    w_bottom
                        t_core   t_core    -t_core      -t_core    ]'
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    verts,			                                            # v: polygon vertices in counter-clockwise order
                    ε_core,					                                    # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]
    b_subs = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],            	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        # axes: box axes
                    ε_subs,					        # data: any type, data associated with box shape
                )
    return [core,b_subs]
end

function circ_wg(w::Float64,t_core::Float64,edge_gap::Float64,n_core::Float64,n_subs::Float64,Δx::Float64,Δy::Float64)::Vector{<:Shape}
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
    return [b_core,b_subs]
end
