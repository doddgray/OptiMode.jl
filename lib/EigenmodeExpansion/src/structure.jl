# ──────────────────────────────────────────────────────────────────────────────
#  GDS → 3D structure → EME cells.
#
#  Following MEOW, the device is built from a 2D layout (GDS polygons in a plane)
#  plus a vertical "layer stack" giving each GDS layer a thickness and material.
#  This defines a 3D structure which is then sliced perpendicular to the
#  propagation axis into a sequence of z-invariant *cells*; each cell carries a 2D
#  cross-section that is handed to the OptiMode mode solver.
#
#  Coordinate convention (matching MEOW): light propagates along the layout's
#  `prop_axis` in-plane direction. The mode-solver cross-section spans
#    cx — the *other* in-plane (transverse / width) axis, and
#    cy — the vertical (layer-stack / growth) axis.
#  Cross-section coordinates are centred on the simulation window so they line up
#  with the centred `Grid` used by `DielectricSmoothing`.
# ──────────────────────────────────────────────────────────────────────────────

export Layer, LayerStack, Structure, CrossSection, Cell
export cross_section_at, build_cells, simulation_grid

using GeometryPrimitives: Cuboid
using DielectricSmoothing: MaterialShape, Grid

"""
    Layer(; gds_layer, zmin, zmax, material, patterned=true, name="")

One entry of a vertical layer stack. `zmin`/`zmax` are the vertical (growth-axis)
extent in microns, `material` indexes the structure's material list, and
`patterned=true` means the in-plane geometry of this layer comes from the GDS
polygons on `gds_layer`; `patterned=false` is a blanket layer (substrate, buried
oxide, cladding) spanning the full transverse width.
"""
Base.@kwdef struct Layer
    gds_layer::Int = 0
    zmin::Float64
    zmax::Float64
    material::Int
    patterned::Bool = true
    name::String = ""
end

"""
    LayerStack(layers, materials; background, prop_axis=:x)

A vertical layer stack plus the ordered `materials` list (each entry anything
`MaterialDispersion._f_ε_mats` accepts, e.g. a `Material`). `background` is the
material index filling everything not covered by a layer (the cladding).
`prop_axis` (`:x` or `:y`) selects which in-plane GDS axis is the propagation
direction.
"""
Base.@kwdef struct LayerStack
    layers::Vector{Layer}
    materials::Vector{Any}
    background::Int
    prop_axis::Symbol = :x
end

"A GDS layout combined with a layer stack and a simulation window."
struct Structure
    layout::GDSLayout
    stack::LayerStack
    "propagation-axis range (microns): (s_min, s_max)"
    s_range::Tuple{Float64,Float64}
    "transverse-window center & width (microns)"
    transverse::Tuple{Float64,Float64}
    "vertical-window center & height (microns)"
    vertical::Tuple{Float64,Float64}
end

"A sampled 2D cross-section: shapes (foreground→background) and material indices."
struct CrossSection
    shapes::Vector{MaterialShape}
    "material column for each shape; the final entry is the background material"
    minds::Vector{Int}
end

"One EME cell: a z-invariant slice with its cross-section and propagation length."
struct Cell
    index::Int
    s_center::Float64
    length::Float64
    cross_section::CrossSection
end

# ── geometry helpers ─────────────────────────────────────────────────────────

_prop_idx(stack::LayerStack) = stack.prop_axis === :x ? 1 : 2
_trans_idx(stack::LayerStack) = stack.prop_axis === :x ? 2 : 1

"""
    Structure(layout, stack; transverse_pad=1.0, vertical_pad=1.0, s_range=nothing)

Assemble a [`Structure`](@ref) from a parsed GDS `layout` and a `stack`. The
simulation window is sized from the layout's transverse bounds (+ `transverse_pad`
on each side) and the stack's vertical extent (+ `vertical_pad`). `s_range`
defaults to the layout's propagation-axis bounds.
"""
function Structure(layout::GDSLayout, stack::LayerStack;
                   transverse_pad::Real=1.0, vertical_pad::Real=1.0,
                   s_range=nothing)
    pi_, ti_ = _prop_idx(stack), _trans_idx(stack)
    smin, smax = Inf, -Inf
    tmin, tmax = Inf, -Inf
    for p in layout.polygons
        smin = min(smin, minimum(@view p.verts[pi_, :]))
        smax = max(smax, maximum(@view p.verts[pi_, :]))
        tmin = min(tmin, minimum(@view p.verts[ti_, :]))
        tmax = max(tmax, maximum(@view p.verts[ti_, :]))
    end
    isfinite(smin) || error("layout has no polygons")
    sr = s_range === nothing ? (smin, smax) : (Float64(s_range[1]), Float64(s_range[2]))
    tcenter = (tmin + tmax) / 2
    twidth = (tmax - tmin) + 2transverse_pad
    zmin = minimum(l.zmin for l in stack.layers)
    zmax = maximum(l.zmax for l in stack.layers)
    vcenter = (zmin + zmax) / 2
    vheight = (zmax - zmin) + 2vertical_pad
    return Structure(layout, stack, sr, (tcenter, twidth), (vcenter, vheight))
end

"""
    simulation_grid(structure, Nx, Ny) -> Grid

Build the 2D mode-solver `Grid` matching the structure's cross-section window
(`Nx` transverse × `Ny` vertical pixels).
"""
function simulation_grid(s::Structure, Nx::Int, Ny::Int)
    return Grid(s.transverse[2], s.vertical[2], Nx, Ny)
end

"""
    polygon_transverse_intervals(verts, prop_idx, trans_idx, s) -> Vector{Tuple}

Intersect a polygon (vertices `2×N`) with the propagation-axis line `value = s`
and return the transverse intervals `(t0, t1)` covered by the polygon there
(even–odd scanline rule).
"""
function polygon_transverse_intervals(verts::AbstractMatrix, pi_::Int, ti_::Int, s::Real)
    n = size(verts, 2)
    crossings = Float64[]
    for i in 1:n
        j = i == n ? 1 : i + 1
        a_p = verts[pi_, i]; b_p = verts[pi_, j]
        a_t = verts[ti_, i]; b_t = verts[ti_, j]
        lo, hi = min(a_p, b_p), max(a_p, b_p)
        # edge straddles the scan line (half-open to avoid double counting vertices)
        if (s >= lo) && (s < hi) && (a_p != b_p)
            frac = (s - a_p) / (b_p - a_p)
            push!(crossings, a_t + frac * (b_t - a_t))
        end
    end
    sort!(crossings)
    intervals = Tuple{Float64,Float64}[]
    for k in 1:2:(length(crossings) - 1)
        push!(intervals, (crossings[k], crossings[k+1]))
    end
    return intervals
end

"""
    cross_section_at(structure, s) -> CrossSection

Sample the structure's cross-section at propagation position `s` (microns).
Patterned layers contribute one centred [`Cuboid`](@ref) per transverse interval
where their polygons cross `s`; blanket layers contribute a full-width cuboid.
Shapes are ordered patterned-first (foreground) then blanket layers, with the
stack's `background` material as the final `minds` entry.
"""
function cross_section_at(st::Structure, s::Real)
    stack = st.stack
    pi_, ti_ = _prop_idx(stack), _trans_idx(stack)
    tcenter, twidth = st.transverse
    vcenter, _ = st.vertical
    shapes = MaterialShape[]
    minds = Int[]
    ax = [1.0 0.0; 0.0 1.0]
    pos = 0                                                     # 1-based shape position
    # group polygons by gds layer
    for layer in stack.layers
        cy = (layer.zmin + layer.zmax) / 2 - vcenter           # centred vertical
        h = layer.zmax - layer.zmin
        if layer.patterned
            for p in stack_polygons(st, layer.gds_layer)
                for (t0, t1) in polygon_transverse_intervals(p.verts, pi_, ti_, s)
                    cx = (t0 + t1) / 2 - tcenter
                    w = t1 - t0
                    w <= 0 && continue
                    pos += 1
                    push!(shapes, MaterialShape(Cuboid([cx, cy], [w, h], ax), pos))
                    push!(minds, layer.material)               # minds[pos] → material column
                end
            end
        else
            pos += 1
            push!(shapes, MaterialShape(Cuboid([0.0, cy], [twidth, h], ax), pos))
            push!(minds, layer.material)
        end
    end
    push!(minds, stack.background)                              # background column (slot nshapes+1)
    return CrossSection(shapes, minds)
end

"polygons of `st` on a given GDS layer number"
stack_polygons(st::Structure, gds_layer::Int) =
    [p for p in st.layout.polygons if p.layer == gds_layer]

# ── cell division (MEOW `create_cells`) ──────────────────────────────────────

"""
    build_cells(structure; num_cells, s_range=structure.s_range) -> Vector{Cell}

Divide the propagation range into `num_cells` equal-length cells and sample the
cross-section at each cell centre — the EME discretization of MEOW's
`create_cells`. For an adiabatic device, increasing `num_cells` converges the
result.
"""
function build_cells(st::Structure; num_cells::Int, s_range=st.s_range)
    s0, s1 = Float64(s_range[1]), Float64(s_range[2])
    L = (s1 - s0) / num_cells
    cells = Cell[]
    for i in 1:num_cells
        sc = s0 + (i - 0.5) * L
        push!(cells, Cell(i, sc, L, cross_section_at(st, sc)))
    end
    return cells
end

"""
    build_cells(structure, s_boundaries) -> Vector{Cell}

Variant taking explicit cell *boundary* positions (length `num_cells+1`), for
non-uniform meshing (finer cells where the geometry changes fastest).
"""
function build_cells(st::Structure, s_boundaries::AbstractVector)
    cells = Cell[]
    for i in 1:(length(s_boundaries) - 1)
        a, b = s_boundaries[i], s_boundaries[i+1]
        sc = (a + b) / 2
        push!(cells, Cell(i, sc, b - a, cross_section_at(st, sc)))
    end
    return cells
end
