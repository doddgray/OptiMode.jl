"""
    ColorsExt (MaterialModels)

Colors.jl interoperability extension for MaterialModels.jl.

Provides conversions between the `NTuple{4,Float64}` color representation
used internally by `Material` and the `RGB`/`RGBA` types from Colors.jl.
"""
module ColorsExt

using MaterialModels
using Colors: RGB, RGBA, Color

"""
    to_color(mat::Material) -> RGBA

Convert a material's stored color tuple to a `Colors.RGBA` value.
"""
to_color(mat::MaterialModels.Material) = RGBA(mat.color...)

"""
    to_color(mat::MaterialModels.NumMat) -> RGBA

Convert a NumMat's stored color tuple to a `Colors.RGBA` value.
"""
to_color(mat::MaterialModels.NumMat) = RGBA(mat.color...)

"""
    Material(models, defaults, name, color::Color) -> Material

Construct a `Material` from a `Colors.jl` color value.
"""
function MaterialModels.Material(models::Dict, defaults::Dict, name::Symbol, color::Color)
    r, g, b, a = Float64(color.r), Float64(color.g), Float64(color.b),
                 hasproperty(color, :alpha) ? Float64(color.alpha) : 1.0
    MaterialModels.Material(models, defaults, name, (r, g, b, a))
end

export to_color

end # module ColorsExt
