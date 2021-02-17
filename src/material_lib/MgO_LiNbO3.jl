################################################################################
#                                 MgO:LiNbO₃                                   #
################################################################################
export ε_MgO_LiNbO₃,nₒ²_MgO_LiNbO₃,nₒ_MgO_LiNbO₃,ngₒ_MgO_LiNbO₃,gvdₒ_MgO_LiNbO₃,nₑ²_MgO_LiNbO₃,nₑ_MgO_LiNbO₃,ngₑ_MgO_LiNbO₃,gvdₑ_MgO_LiNbO₃, MgO_LiNbO₃

"""
These functions create a symbolic representation (using ModelingToolkit) of the
Sellmeier Equation model for the temperature and wavelength dependence
of 5% MgO:LiNbO3's (congruent LiNbO3 (CLN) not stoichiometric LiNbO3 (SLN))
ordinary and extraordinary indices of refraction.
Equation form is based on "Temperature and
wavelength dependent refractive index equations for MgO-doped congruent
and stoichiometric LiNbO3" by Gayer et al., Applied Physics B 91,
343?348 (2008)
This model is then exported to other functions that use it and its
derivatives to return index, group index and GVD values as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]
"""
pₑ_MgO_LiNbO₃ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ_MgO_LiNbO₃ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)

function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

nₑ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₑ_MgO_LiNbO₃...)
nₑ_MgO_LiNbO₃_λT_sym = sqrt(nₑ²_MgO_LiNbO₃_λT_sym)
nₑ²_MgO_LiNbO₃_sym = substitute(nₑ²_MgO_LiNbO₃_λT_sym,[T=>pₑ_MgO_LiNbO₃.T₀])
nₑ_MgO_LiNbO₃_sym = sqrt(nₑ²_MgO_LiNbO₃_sym)
_nₑ²_MgO_LiNbO₃_λT = eval(build_function(nₑ²_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₑ_MgO_LiNbO₃_λT = eval(build_function(nₑ_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_ngₑ_MgO_LiNbO₃_λT = eval(build_function(ng(nₑ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvdₑ_MgO_LiNbO₃_λT = eval(build_function(gvd(nₑ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₑ²_MgO_LiNbO₃ = eval(build_function(nₑ²_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_nₑ_MgO_LiNbO₃ = eval(build_function(nₑ_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ngₑ_MgO_LiNbO₃ = eval(build_function(ng(nₑ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvdₑ_MgO_LiNbO₃ = eval(build_function(gvd(nₑ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])

nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ_MgO_LiNbO₃_λT_sym = sqrt(nₒ²_MgO_LiNbO₃_λT_sym)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
nₒ_MgO_LiNbO₃_sym = sqrt(nₒ²_MgO_LiNbO₃_sym)
_nₒ²_MgO_LiNbO₃_λT = eval(build_function(nₒ²_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₒ_MgO_LiNbO₃_λT = eval(build_function(nₒ_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_ngₒ_MgO_LiNbO₃_λT = eval(build_function(ng(nₒ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvdₒ_MgO_LiNbO₃_λT = eval(build_function(gvd(nₒ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₒ²_MgO_LiNbO₃ = eval(build_function(nₒ²_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_nₒ_MgO_LiNbO₃ = eval(build_function(nₒ_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ngₒ_MgO_LiNbO₃ = eval(build_function(ng(nₒ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvdₒ_MgO_LiNbO₃ = eval(build_function(gvd(nₒ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])

ε_MgO_LiNbO₃_λT_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym ] )
ε_MgO_LiNbO₃_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym ] )
# ε_MgO_LiNbO₃_sym = [ nₑ²_MgO_LiNbO₃_sym            0                0
#                         0               nₒ²_MgO_LiNbO₃_sym          0
#                         0                          0         nₒ²_MgO_LiNbO₃_sym ]
_ε_MgO_LiNbO₃_λT, _ε_MgO_LiNbO₃_λT! = eval.(build_function(ε_MgO_LiNbO₃_sym,λ,T))
_ε_MgO_LiNbO₃,_ε_MgO_LiNbO₃! = eval.(build_function(substitute.(ε_MgO_LiNbO₃_sym,[T=>pₒ_MgO_LiNbO₃.T₀]),λ))
function ε_MgO_LiNbO₃(λ::T) where T<:Real
    nₑ² = _nₑ²_MgO_LiNbO₃(λ)
    nₒ² = _nₒ²_MgO_LiNbO₃(λ)
    Diagonal( [ nₑ², nₒ², nₒ² ] )
    # SMatrix{3,3,T,9}( nₑ²,    0.,     0.,
    #                   0.,     nₒ²,    0.,
    #                   0.,     0.,     nₒ², )
end

nₑ²_MgO_LiNbO₃(λ) = _nₑ²_MgO_LiNbO₃(λ)
nₑ_MgO_LiNbO₃(λ) = _nₑ_MgO_LiNbO₃(λ)
ngₑ_MgO_LiNbO₃(λ) = _ngₑ_MgO_LiNbO₃(λ)
nₑ²_MgO_LiNbO₃(λ,T) = _nₑ²_MgO_LiNbO₃_λT(λ,T)
nₑ_MgO_LiNbO₃(λ,T) = _nₑ_MgO_LiNbO₃_λT(λ,T)
ngₑ_MgO_LiNbO₃(λ,T) = _ngₑ_MgO_LiNbO₃_λT(λ,T)
nₑ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = _nₑ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)
nₑ_MgO_LiNbO₃(λ::Unitful.Length) = _nₑ_MgO_LiNbO₃((λ|>u"μm").val)
nₑ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) = _nₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
nₑ_MgO_LiNbO₃(f::Unitful.Frequency) = _nₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)
ngₑ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = _ngₑ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)
ngₑ_MgO_LiNbO₃(λ::Unitful.Length) = _ngₑ_MgO_LiNbO₃((λ|>u"μm").val)
ngₑ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) = _ngₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
ngₑ_MgO_LiNbO₃(f::Unitful.Frequency) = _ngₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)
gvdₑ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = ( _gvdₑ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₑ_MgO_LiNbO₃(λ::Unitful.Length) = ( _gvdₑ_MgO_LiNbO₃((λ|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₑ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) =( _gvdₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₑ_MgO_LiNbO₃(f::Unitful.Frequency) = ( _gvdₑ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"

nₒ²_MgO_LiNbO₃(λ) = _nₒ²_MgO_LiNbO₃(λ)
nₒ_MgO_LiNbO₃(λ) = _nₒ_MgO_LiNbO₃(λ)
ngₒ_MgO_LiNbO₃(λ) = _ngₒ_MgO_LiNbO₃(λ)
nₒ²_MgO_LiNbO₃(λ,T) = _nₒ²_MgO_LiNbO₃_λT(λ,T)
nₒ_MgO_LiNbO₃(λ,T) = _nₒ_MgO_LiNbO₃_λT(λ,T)
ngₒ_MgO_LiNbO₃(λ,T) = _ngₒ_MgO_LiNbO₃_λT(λ,T)
nₒ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = _nₒ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)
nₒ_MgO_LiNbO₃(λ::Unitful.Length) = _nₒ_MgO_LiNbO₃((λ|>u"μm").val)
nₒ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) = _nₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
nₒ_MgO_LiNbO₃(f::Unitful.Frequency) = _nₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)
ngₒ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = _ngₒ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)
ngₒ_MgO_LiNbO₃(λ::Unitful.Length) = _ngₒ_MgO_LiNbO₃((λ|>u"μm").val)
ngₒ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) = _ngₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
ngₒ_MgO_LiNbO₃(f::Unitful.Frequency) = _ngₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)
gvdₒ_MgO_LiNbO₃(λ::Unitful.Length,T::Unitful.Temperature) = ( _gvdₒ_MgO_LiNbO₃((λ|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₒ_MgO_LiNbO₃(λ::Unitful.Length) = ( _gvdₒ_MgO_LiNbO₃((λ|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₒ_MgO_LiNbO₃(f::Unitful.Frequency,T::Unitful.Temperature) =( _gvdₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvdₒ_MgO_LiNbO₃(f::Unitful.Frequency) = ( _gvdₒ_MgO_LiNbO₃(((Unitful.c0/f)|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"


MgO_LiNbO₃ = Material(SMatrix{3,3}(ε_MgO_LiNbO₃_sym))
