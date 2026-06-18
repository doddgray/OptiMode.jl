# Self-contained mode-field image rendering.
#
# Workers save one annotated PNG per band (false-colour |Eₓ|,|E_y|,|E_z| panels plus a
# text header carrying the per-band summary). The renderer is deliberately
# dependency-free — a small pure-Julia PNG encoder (uncompressed DEFLATE blocks, so no
# zlib needed), a viridis-like colormap, and a compact 5×7 bitmap font — so that PNGs
# are produced on any SLURM worker without a plotting stack or display, and the feature
# works identically for `save_fields=true` (full) and summary-only batches.
#
# For publication-quality figures, load the full field data (`load_fields`) into a
# Makie/Plots session locally; these PNGs are for quick at-a-glance inspection of a
# whole sweep alongside the tabular summary.

# ---------------------------------------------------------------------------------
# checksums (CRC-32 / Adler-32) and the minimal PNG container
# ---------------------------------------------------------------------------------

const _CRC32_TABLE = let tbl = Vector{UInt32}(undef, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 0x00000001) != 0 ? (0xedb88320 ⊻ (c >> 1)) : (c >> 1)
        end
        tbl[n+1] = c
    end
    tbl
end

function _crc32(data::AbstractVector{UInt8}, crc::UInt32=0xffffffff)
    @inbounds for b in data
        crc = _CRC32_TABLE[((crc ⊻ b) & 0xff)+1] ⊻ (crc >> 8)
    end
    return crc
end
_crc32_final(data::AbstractVector{UInt8}) = _crc32(data) ⊻ 0xffffffff

function _adler32(data::AbstractVector{UInt8})
    a = UInt32(1)
    b = UInt32(0)
    @inbounds for byte in data
        a = (a + byte) % 0xfff1
        b = (b + a) % 0xfff1
    end
    return (b << 16) | a
end

_be(x::UInt32) = UInt8[(x>>24)&0xff, (x>>16)&0xff, (x>>8)&0xff, x&0xff]
_le16(x::Integer) = UInt8[x&0xff, (x>>8)&0xff]

function _png_chunk(io::IO, typ::String, data::AbstractVector{UInt8})
    write(io, _be(UInt32(length(data))))
    tb = Vector{UInt8}(codeunits(typ))
    write(io, tb)
    write(io, data)
    write(io, _be(_crc32_final(vcat(tb, data))))
    return nothing
end

"zlib stream wrapping `raw` in uncompressed (stored) DEFLATE blocks — no zlib dependency"
function _zlib_stored(raw::AbstractVector{UInt8})
    out = UInt8[0x78, 0x01]                 # zlib header (CM=8, no preset dict, fastest)
    n = length(raw)
    pos = 1
    while pos <= n || n == 0
        chunk = min(n - pos + 1, 65535)
        final = (pos + chunk - 1) >= n
        push!(out, final ? 0x01 : 0x00)     # BFINAL bit + BTYPE=00 (stored)
        append!(out, _le16(chunk))
        append!(out, _le16(0xffff ⊻ chunk)) # NLEN = ~LEN
        chunk > 0 && append!(out, @view raw[pos:pos+chunk-1])
        pos += chunk
        (final || n == 0) && break
    end
    append!(out, _be(_adler32(raw)))
    return out
end

"""
    write_png(path, canvas)

Encode an `(H, W, 3)` `UInt8` RGB array to a PNG file at `path` using a dependency-free
encoder (stored DEFLATE blocks). Returns `path`.
"""
function write_png(path::AbstractString, canvas::AbstractArray{UInt8,3})
    H, W, C = size(canvas)
    C == 3 || throw(ArgumentError("canvas must be (H,W,3) RGB, got $(size(canvas))"))
    # raw scanlines: each row prefixed with filter byte 0 (none)
    raw = Vector{UInt8}(undef, H * (1 + 3W))
    o = 1
    @inbounds for r in 1:H
        raw[o] = 0x00; o += 1
        for c in 1:W, ch in 1:3
            raw[o] = canvas[r, c, ch]; o += 1
        end
    end
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])  # PNG signature
        ihdr = vcat(_be(UInt32(W)), _be(UInt32(H)), UInt8[8, 2, 0, 0, 0]) # 8-bit truecolor
        _png_chunk(io, "IHDR", ihdr)
        _png_chunk(io, "IDAT", _zlib_stored(raw))
        _png_chunk(io, "IEND", UInt8[])
    end
    mv(tmp, path; force=true)
    return path
end

# ---------------------------------------------------------------------------------
# colormap
# ---------------------------------------------------------------------------------

const _VIRIDIS = (
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
    (31, 158, 137), (53, 183, 121), (110, 206, 88), (181, 222, 43), (253, 231, 37),
)

"map t∈[0,1] to an RGB `NTuple{3,UInt8}` on a viridis-like ramp"
function _colormap(t::Real)
    t = isfinite(t) ? clamp(float(t), 0.0, 1.0) : 0.0
    s = t * (length(_VIRIDIS) - 1)
    i = floor(Int, s)
    i >= length(_VIRIDIS) - 1 && return map(UInt8, _VIRIDIS[end])
    f = s - i
    a = _VIRIDIS[i+1]
    b = _VIRIDIS[i+2]
    return ntuple(j -> round(UInt8, a[j] + f * (b[j] - a[j])), 3)
end

# ---------------------------------------------------------------------------------
# 5×7 bitmap font (string-art so glyphs are verifiable by eye); text is upper-cased
# before rendering, so only digits, A–Z, space and a handful of symbols are needed.
# ---------------------------------------------------------------------------------

const _FONT_ART = Dict{Char,Vector{String}}(
    ' ' => ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
    '0' => [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    '1' => ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    '2' => [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    '3' => ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    '4' => ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    '5' => ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    '6' => ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    '7' => ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    '8' => [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    '9' => [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    'A' => [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    'B' => ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    'C' => [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    'D' => ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."],
    'E' => ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    'F' => ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    'G' => [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."],
    'H' => ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    'I' => [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    'J' => ["..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."],
    'K' => ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    'L' => ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    'M' => ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    'N' => ["#...#", "##..#", "#.#.#", "#.#.#", "#..##", "#...#", "#...#"],
    'O' => [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    'P' => ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    'Q' => [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    'R' => ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    'S' => [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    'T' => ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    'U' => ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    'V' => ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    'W' => ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    'X' => ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    'Y' => ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    'Z' => ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    '.' => ["     ", "     ", "     ", "     ", "     ", "..#..", "..#.."],
    ',' => ["     ", "     ", "     ", "     ", "..#..", "..#..", ".#..."],
    ':' => ["     ", "..#..", "..#..", "     ", "..#..", "..#..", "     "],
    '-' => ["     ", "     ", "     ", "#####", "     ", "     ", "     "],
    '+' => ["     ", "..#..", "..#..", "#####", "..#..", "..#..", "     "],
    '=' => ["     ", "     ", "#####", "     ", "#####", "     ", "     "],
    '/' => ["....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."],
    '(' => ["..#..", ".#...", "#....", "#....", "#....", ".#...", "..#.."],
    ')' => ["..#..", "...#.", "....#", "....#", "....#", "...#.", "..#.."],
    '%' => ["##..#", "##.#.", "..#..", ".#.##", "#..##", "     ", "     "],
    '|' => ["..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    '<' => ["...#.", "..#..", ".#...", "#....", ".#...", "..#..", "...#."],
    '>' => [".#...", "..#..", "...#.", "....#", "...#.", "..#..", ".#..."],
    '#' => [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
)

# parsed once into (col-bit) masks: glyph[row] has bit (5-c) set for a lit pixel.
# Byte-indexed and bounds-safe (the art is ASCII) so a short/long row can never error.
_lit(s::AbstractString, c::Int) = c <= ncodeunits(s) && codeunit(s, c) != UInt8(' ')
_row_mask(s::AbstractString) = reduce(|, (_lit(s, c) ? (UInt8(1) << (5 - c)) : UInt8(0) for c in 1:5); init=UInt8(0))
const _FONT = Dict{Char,NTuple{7,UInt8}}(
    ch => ntuple(r -> r <= length(art) ? _row_mask(art[r]) : UInt8(0), 7)
    for (ch, art) in _FONT_ART
)

const _GLYPH_W = 5
const _GLYPH_H = 7

"draw upper-cased ASCII `str` into `canvas` with top-left at (row,col); `scale` enlarges pixels"
function draw_text!(canvas::AbstractArray{UInt8,3}, row::Int, col::Int, str::AbstractString,
    color::NTuple{3,UInt8}=(20, 20, 20); scale::Int=1, spacing::Int=1)
    H, W, _ = size(canvas)
    cx = col
    for ch in uppercase(str)
        glyph = get(_FONT, ch, _FONT[' '])
        for gr in 1:_GLYPH_H, gc in 1:_GLYPH_W
            if (glyph[gr] & (UInt8(1) << (5 - gc))) != 0
                for sr in 0:scale-1, sc in 0:scale-1
                    pr = row + (gr - 1) * scale + sr
                    pc = cx + (gc - 1) * scale + sc
                    if 1 <= pr <= H && 1 <= pc <= W
                        canvas[pr, pc, 1] = color[1]
                        canvas[pr, pc, 2] = color[2]
                        canvas[pr, pc, 3] = color[3]
                    end
                end
            end
        end
        cx += (_GLYPH_W + spacing) * scale
    end
    return canvas
end

text_width(str::AbstractString; scale::Int=1, spacing::Int=1) =
    length(str) * (_GLYPH_W + spacing) * scale

# ---------------------------------------------------------------------------------
# field panels and the composite mode image
# ---------------------------------------------------------------------------------

"colour a real `(Nx,Ny)` field `F` into `canvas` at (row0,col0), upscaled by `scale`,
normalized to `vmax` (its own max if `nothing`); returns the drawn (height,width)."
function blit_field!(canvas::AbstractArray{UInt8,3}, row0::Int, col0::Int,
    F::AbstractMatrix{<:Real}, scale::Int; vmax=nothing)
    Nx, Ny = size(F)
    mx = vmax === nothing ? maximum(F) : vmax
    mx = mx > 0 ? mx : 1.0
    H, W, _ = size(canvas)
    # image rows = +y at top: row r ↔ iy = Ny - r + 1; image cols = +x: col c ↔ ix
    for r in 1:Ny, c in 1:Nx
        iy = Ny - r + 1
        rgb = _colormap(F[c, iy] / mx)
        for sr in 0:scale-1, sc in 0:scale-1
            pr = row0 + (r - 1) * scale + sr
            pc = col0 + (c - 1) * scale + sc
            if 1 <= pr <= H && 1 <= pc <= W
                canvas[pr, pc, 1] = rgb[1]
                canvas[pr, pc, 2] = rgb[2]
                canvas[pr, pc, 3] = rgb[3]
            end
        end
    end
    return (Ny * scale, Nx * scale)
end

"choose an integer upscale so the field panels are at least ~`target` px in their larger dim"
function _panel_scale(Nx::Int, Ny::Int; target::Int=160, maxscale::Int=12)
    d = max(Nx, Ny)
    d <= 0 && return 1
    return clamp(cld(target, d), 1, maxscale)
end

"""
    render_mode_png(path, E, grid, header, titles=("|Ex|","|Ey|","|Ez|"); scale=auto)

Render a composite PNG for one mode: a row of viridis-coloured `|Eₓ|`, `|E_y|`, `|E_z|`
panels (each normalized to the field's global maximum so relative component strengths
are visible) topped by a white header carrying the `header` text lines (the per-band
summary). `E` is the `(3, Nx, Ny)` complex field from `E⃗`. Returns `path`.
"""
function render_mode_png(path::AbstractString, E::AbstractArray{<:Complex,3}, grid,
    header::AbstractVector{<:AbstractString},
    titles=("|EX|", "|EY|", "|EZ|"); scale::Union{Nothing,Int}=nothing)
    _, Nx, Ny = size(E)
    sc = scale === nothing ? _panel_scale(Nx, Ny) : scale
    pw, ph = Nx * sc, Ny * sc

    pad = 8
    title_h = _GLYPH_H + 4
    line_h = _GLYPH_H + 3
    header_h = pad + length(header) * line_h + pad
    panels_w = 3 * pw + 4 * pad
    Wtot = max(panels_w, pad + maximum(text_width.(header); init=0) + pad)
    Htot = header_h + title_h + ph + pad

    canvas = fill(UInt8(255), Htot, Wtot, 3)   # white background

    # header text (dark grey)
    for (i, line) in enumerate(header)
        draw_text!(canvas, pad + (i - 1) * line_h, pad, line, (25, 25, 30); scale=1)
    end
    # separator
    sepr = header_h - 1
    for c in 1:Wtot, ch in 1:3
        canvas[clamp(sepr, 1, Htot), c, ch] = 0xc0
    end

    mags = (abs.(view(E, 1, :, :)), abs.(view(E, 2, :, :)), abs.(view(E, 3, :, :)))
    gmax = maximum(maximum.(mags))
    for (j, F) in enumerate(mags)
        col0 = pad + (j - 1) * (pw + pad)
        draw_text!(canvas, header_h + 2, col0, titles[j], (40, 40, 40); scale=1)
        blit_field!(canvas, header_h + title_h, col0, F, sc; vmax=gmax)
    end
    return write_png(path, canvas)
end
