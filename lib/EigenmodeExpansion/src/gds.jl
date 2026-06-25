# ──────────────────────────────────────────────────────────────────────────────
#  Minimal, dependency-free GDSII reader/writer.
#
#  GDSII is the de-facto interchange format for integrated-photonics layouts (it
#  is what GDSFactory, KLayout, etc. export). This file implements just enough of
#  the binary format to (a) read the BOUNDARY (polygon) elements of a flattened
#  layout and (b) write an equivalent layout back out, so that the EME pipeline
#  can be driven end-to-end from a real `.gds` file without a heavy GDS library.
#
#  Records are `[len::UInt16(BE)] [rectype::UInt8] [datatype::UInt8] [payload…]`,
#  `len` counting the 4-byte header. Coordinates are 32-bit integers in *database
#  units*; the UNITS record carries the database-unit size in user units and in
#  metres. We convert everything to microns on read and back on write.
# ──────────────────────────────────────────────────────────────────────────────

export GDSPolygon, GDSLayout, read_gds, write_gds

"A single boundary (polygon) element of a GDS layout, vertices in microns."
struct GDSPolygon
    layer::Int
    datatype::Int
    "vertices as a `2 × N` matrix `[x; y]` (microns); the GDS closing point is dropped"
    verts::Matrix{Float64}
end

"A parsed GDS layout: its boundary polygons and the database-unit size in microns."
struct GDSLayout
    name::String
    "size of one database unit in microns (i.e. `meters_per_dbunit * 1e6`)"
    unit_um::Float64
    polygons::Vector{GDSPolygon}
end

# GDS record types we care about
const _HEADER  = 0x0002
const _BGNLIB  = 0x0102
const _LIBNAME = 0x0206
const _UNITS   = 0x0305
const _ENDLIB  = 0x0400
const _BGNSTR  = 0x0502
const _STRNAME = 0x0606
const _ENDSTR  = 0x0700
const _BOUNDARY = 0x0800
const _LAYER   = 0x0D02
const _DATATYPE = 0x0E02
const _XY      = 0x1003
const _ENDEL   = 0x1100

# ── GDS 8-byte real (excess-64, base-16) ─────────────────────────────────────

"Decode a GDSII 8-byte real from a big-endian `UInt64`."
function _gds_real8_decode(bits::UInt64)::Float64
    sign = (bits >> 63) & 0x1
    expo = Int((bits >> 56) & 0x7f) - 64
    mant = bits & 0x00ffffffffffffff
    val = (mant / 2.0^56) * 16.0^expo
    return sign == 1 ? -val : val
end

"Encode a `Float64` into a GDSII 8-byte real (big-endian `UInt64`)."
function _gds_real8_encode(x::Real)::UInt64
    x == 0 && return UInt64(0)
    sign = x < 0 ? UInt64(1) : UInt64(0)
    v = abs(float(x))
    expo = 0
    while v >= 1.0
        v /= 16.0
        expo += 1
    end
    while v < 1.0 / 16.0
        v *= 16.0
        expo -= 1
    end
    mant = round(UInt64, v * 2.0^56)
    return (sign << 63) | (UInt64(expo + 64) << 56) | (mant & 0x00ffffffffffffff)
end

# ── reader ───────────────────────────────────────────────────────────────────

_read_be(io, ::Type{T}) where {T} = ntoh(read(io, T))

"""
    read_gds(path) -> GDSLayout

Parse the BOUNDARY (polygon) elements of a (flattened) GDSII file. Polygon
vertices are returned in microns. Structure references (SREF/AREF) and PATH
elements are ignored — flatten the layout before export (GDSFactory:
`component.flatten()`) so all geometry is present as boundaries.
"""
function read_gds(path::AbstractString)::GDSLayout
    open(path, "r") do io
        name = ""
        unit_um = 1e-3                       # default: 1 nm db unit, 1 µm user unit
        polys = GDSPolygon[]
        cur_layer = 0
        cur_dtype = 0
        cur_xy = Float64[]
        in_boundary = false
        while !eof(io)
            len = Int(_read_be(io, UInt16))
            rectype = read(io, UInt8)
            datatype = read(io, UInt8)
            rec = (UInt16(rectype) << 8) | UInt16(datatype)
            payload = len > 4 ? read(io, len - 4) : UInt8[]
            if rec == _LIBNAME
                name = String(copy(payload))
            elseif rec == _UNITS
                # two 8-byte reals: [db-unit in user units, db-unit in metres]
                b = IOBuffer(payload)
                _uu = _gds_real8_decode(_read_be(b, UInt64))
                meters = _gds_real8_decode(_read_be(b, UInt64))
                unit_um = meters * 1e6
            elseif rec == _BOUNDARY
                in_boundary = true
                cur_xy = Float64[]
                cur_layer = 0
                cur_dtype = 0
            elseif rec == _LAYER
                cur_layer = Int(ntoh(reinterpret(Int16, payload[1:2])[1]))
            elseif rec == _DATATYPE
                cur_dtype = Int(ntoh(reinterpret(Int16, payload[1:2])[1]))
            elseif rec == _XY
                ints = ntoh.(reinterpret(Int32, payload))
                cur_xy = Float64.(ints) .* unit_um
            elseif rec == _ENDLIB
                break
            elseif rec == _ENDEL
                if in_boundary && !isempty(cur_xy)
                    n = length(cur_xy) ÷ 2
                    pts = reshape(cur_xy, 2, n)          # [x;y] columns
                    # GDS repeats the first vertex as the last; drop it
                    if n > 1 && pts[:, 1] == pts[:, end]
                        pts = pts[:, 1:end-1]
                    end
                    push!(polys, GDSPolygon(cur_layer, cur_dtype, Matrix(pts)))
                end
                in_boundary = false
            end
        end
        return GDSLayout(name, unit_um, polys)
    end
end

# ── writer ─────────────────────────────────────────────────────────────────

_write_be(io, x) = write(io, hton(x))

function _write_record(io, rec::UInt16, payload::Vector{UInt8})
    _write_be(io, UInt16(length(payload) + 4))
    write(io, UInt8((rec >> 8) & 0xff))      # rectype
    write(io, UInt8(rec & 0xff))             # datatype
    write(io, payload)
    return nothing
end

_i16(x) = collect(reinterpret(UInt8, [hton(Int16(x))]))
function _i16vec(xs)
    b = UInt8[]
    for x in xs; append!(b, _i16(x)); end
    return b
end

"""
    write_gds(path, polygons; name="LIB", unit_um=1e-3)

Write a flat GDSII file containing `polygons` (a vector of [`GDSPolygon`](@ref)).
`unit_um` is the database-unit size in microns (default 1 nm). This is the inverse
of [`read_gds`](@ref) and lets tests/examples produce a real `.gds` file without an
external GDS library.
"""
function write_gds(path::AbstractString, polygons::AbstractVector{GDSPolygon};
                   name::AbstractString="LIB", unit_um::Real=1e-3)
    meters = unit_um * 1e-6
    open(path, "w") do io
        _write_record(io, _HEADER, _i16(600))                       # GDS v6
        _write_record(io, _BGNLIB, _i16vec(zeros(Int, 12)))         # mod+acc dates
        nm = collect(codeunits(name)); isodd(length(nm)) && push!(nm, 0x00)
        _write_record(io, _LIBNAME, nm)
        units = UInt8[]
        append!(units, reinterpret(UInt8, [hton(_gds_real8_encode(unit_um))]))  # db unit in user units (µm)
        append!(units, reinterpret(UInt8, [hton(_gds_real8_encode(meters))]))   # db unit in metres
        _write_record(io, _UNITS, units)
        _write_record(io, _BGNSTR, _i16vec(zeros(Int, 12)))
        sn = collect(codeunits("TOP")); isodd(length(sn)) && push!(sn, 0x00)
        _write_record(io, _STRNAME, sn)
        for p in polygons
            _write_record(io, _BOUNDARY, UInt8[])
            _write_record(io, _LAYER, _i16(p.layer))
            _write_record(io, _DATATYPE, _i16(p.datatype))
            n = size(p.verts, 2)
            xy = Int32[]
            for j in 1:n
                push!(xy, round(Int32, p.verts[1, j] / unit_um))
                push!(xy, round(Int32, p.verts[2, j] / unit_um))
            end
            # close the polygon (repeat first vertex)
            push!(xy, round(Int32, p.verts[1, 1] / unit_um))
            push!(xy, round(Int32, p.verts[2, 1] / unit_um))
            xyb = UInt8[]
            for v in xy; append!(xyb, reinterpret(UInt8, [hton(v)])); end
            _write_record(io, _XY, xyb)
            _write_record(io, _ENDEL, UInt8[])
        end
        _write_record(io, _ENDSTR, UInt8[])
        _write_record(io, _ENDLIB, UInt8[])
    end
    return path
end
