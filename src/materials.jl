# Dispersion models for NLO-relevant materials adapted from
# https://github.com/doddgray/optics_modeling/blob/master/nlo/NLO_tools.py

################################################################################
#            Temperature Dependent Index, Group Index and GVD models           #
#                        for phase-matching calculations                       #
################################################################################
# ng(fₙ::Function) = λ -> ( (n,n_pb) = Zygote.pullback(fₙ,λ); ( n - λ * n_pb(1)[1] ) )
using LinearAlgebra, ModelingToolkit, Latexify, StaticArrays, Unitful
c = Unitful.c0      # Unitful.jl speed of light
@parameters λ, T
Dλ = Differential(λ)
DT = Differential(T)
ng_sym(n_sym) = n_sym - λ * expand_derivatives(Dλ(n_sym))
gvd_sym(n_sym) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym))) # gvd = uconvert( ( 1 / ( 2π * c^2) ) * _gvd(lm_um,T_C)u"μm", u"fs^2 / mm" )

################################################################################
#                                 MgO:LiNbO₃                                   #
################################################################################
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
_ngₑ_MgO_LiNbO₃_λT = eval(build_function(ng_sym(nₑ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvdₑ_MgO_LiNbO₃_λT = eval(build_function(gvd_sym(nₑ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₑ²_MgO_LiNbO₃ = eval(build_function(nₑ²_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_nₑ_MgO_LiNbO₃ = eval(build_function(nₑ_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ngₑ_MgO_LiNbO₃ = eval(build_function(ng_sym(nₑ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvdₑ_MgO_LiNbO₃ = eval(build_function(gvd_sym(nₑ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])

nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ_MgO_LiNbO₃_λT_sym = sqrt(nₒ²_MgO_LiNbO₃_λT_sym)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
nₒ_MgO_LiNbO₃_sym = sqrt(nₒ²_MgO_LiNbO₃_sym)
_nₒ²_MgO_LiNbO₃_λT = eval(build_function(nₒ²_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₒ_MgO_LiNbO₃_λT = eval(build_function(nₒ_MgO_LiNbO₃_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_ngₒ_MgO_LiNbO₃_λT = eval(build_function(ng_sym(nₒ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvdₒ_MgO_LiNbO₃_λT = eval(build_function(gvd_sym(nₒ_MgO_LiNbO₃_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_nₒ²_MgO_LiNbO₃ = eval(build_function(nₒ²_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_nₒ_MgO_LiNbO₃ = eval(build_function(nₒ_MgO_LiNbO₃_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ngₒ_MgO_LiNbO₃ = eval(build_function(ng_sym(nₒ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvdₒ_MgO_LiNbO₃ = eval(build_function(gvd_sym(nₒ_MgO_LiNbO₃_sym),λ)) # inputs (λ,T) in ([μm],[°C])

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


################################################################################
#                                   SiO₂                                       #
################################################################################

"""
This code creates a symbolic representation of the Sellmeier Equation model
for the index of refraction of amorphous SiO₂.Equation form is based on:
     Kitamura, et al.
     "Optical constants of silica glass from extreme ultraviolet to far
     infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
which references
    Malitson, “Interspecimen comparison of the refractive index of fused
    silica,” J. Opt. Soc. Am.55,1205–1209 (1965)
and has been validated from 0.21-6.7 μm (free space wavelength).
The thermo-optic coefficient (for 300K) is from the literature, but I forgot
the source.
The symbolic index model and its derivatives in turn are used to generate
numerical functions for the SiO₂ index, group index and GVD as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]
"""

p_SiO₂ = (
    A₀ = 1,
    B₁ = 0.6961663,
    C₁ = (0.0684043)^2,     #                           [μm²]
    B₂ = 0.4079426,
    C₂ = (0.1162414)^2,     #                           [μm²]
    B₃ = 0.8974794,
    C₃ = (9.896161)^2,      #                           [μm²]
    dn_dT = 6.1e-6,         # thermo-optic coefficient  [K⁻¹]
    T₀ = 20,                # reference temperature     [°C]
)

function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    λ² = λ^2
    A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
end

n²_SiO₂_sym = n²_sym_fmt1( λ ; p_SiO₂...)  # does not include temperature dependence, avoiding n² = ( √(n²(T₀)) + dndT*(T-T₀) )² for AD performance when using ε_SiO₂
n_SiO₂_λT_sym = sqrt(n²_SiO₂_sym) + p_SiO₂.dn_dT  *  ( T - p_SiO₂.T₀  )
n_SiO₂_sym = sqrt(n²_SiO₂_sym)

_n²_SiO₂ = eval(build_function(n²_SiO₂_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_n_SiO₂_λT = eval(build_function(n_SiO₂_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_n_SiO₂ = eval(build_function(n_SiO₂_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ng_SiO₂ = eval(build_function(ng_sym(n_SiO₂_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_ng_SiO₂_λT = eval(build_function(ng_sym(n_SiO₂_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvd_SiO₂ = eval(build_function(gvd_sym(n_SiO₂_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvd_SiO₂_λT = eval(build_function(gvd_sym(n_SiO₂_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])

ε_SiO₂_λT_sym = Diagonal(fill(n_SiO₂_λT_sym^2,3)) # n_SiO₂_λT_sym^2 * I
ε_SiO₂_sym = Diagonal(fill(n²_SiO₂_sym,3)) #n²_SiO₂_sym * I
_ε_SiO₂_λT, _ε_SiO₂_λT! = eval.(build_function(ε_SiO₂_λT_sym,λ,T))
_ε_SiO₂,_ε_SiO₂! = eval.(build_function(ε_SiO₂_sym,λ))

function ε_SiO₂(λ::T) where T<:Real
    _n²_SiO₂(λ) * I
    # Diagonal(fill(_n²_SiO₂(λ),3))
    # n² = _n²_SiO₂(λ)
    # SMatrix{3,3,T,9}( n²,    0.,     0.,
    #                   0.,     n²,    0.,
    #                   0.,     0.,     n², )
end

n²_SiO₂(λ) = _n²_SiO₂(λ)
n_SiO₂(λ) = _n_SiO₂(λ)
ng_SiO₂(λ) = _ng_SiO₂(λ)
n²_SiO₂(λ,T) = _n²_SiO₂_λT(λ,T)
n_SiO₂(λ,T) = _n_SiO₂_λT(λ,T)
ng_SiO₂(λ,T) = _ng_SiO₂_λT(λ,T)
n_SiO₂(λ::Unitful.Length,T::Unitful.Temperature) = _n_SiO₂((λ|>u"μm").val,(T|>u"°C").val)
n_SiO₂(λ::Unitful.Length) = _n_SiO₂((λ|>u"μm").val)
n_SiO₂(f::Unitful.Frequency,T::Unitful.Temperature) = _n_SiO₂(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
n_SiO₂(f::Unitful.Frequency) = _n_SiO₂(((Unitful.c0/f)|>u"μm").val)
ng_SiO₂(λ::Unitful.Length,T::Unitful.Temperature) = _ng_SiO₂((λ|>u"μm").val,(T|>u"°C").val)
ng_SiO₂(λ::Unitful.Length) = _ng_SiO₂((λ|>u"μm").val)
ng_SiO₂(f::Unitful.Frequency,T::Unitful.Temperature) = _ng_SiO₂(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
ng_SiO₂(f::Unitful.Frequency) = _ng_SiO₂(((Unitful.c0/f)|>u"μm").val)
gvd_SiO₂(λ::Unitful.Length,T::Unitful.Temperature) = ( _gvd_SiO₂((λ|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_SiO₂(λ::Unitful.Length) = ( _gvd_SiO₂((λ|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_SiO₂(f::Unitful.Frequency,T::Unitful.Temperature) =( _gvd_SiO₂(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_SiO₂(f::Unitful.Frequency) = ( _gvd_SiO₂(((Unitful.c0/f)|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"

################################################################################
#                                   Si₃N₄                                      #
################################################################################
"""
This code first creates a symbolic representation of the
Sellmeier Equation model for the index of refraction.
Equation form is based on Luke, Okawachi, Lamont, Gaeta and Lipson,
"Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
https://doi.org/10.1364/OL.40.004823
valid from 0.31–5.504 um
Thermo-optic coefficients from
Xue, et al.
"Thermal tuning of Kerr frequency combs in silicon nitride microring resonators"
Opt. Express 24.1 (2016) http://doi.org/10.1364/OE.24.000687
The symbolic index model and its derivatives in turn are used to generate
numerical functions for the SiO₂ index, group index and GVD as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]
"""

p_Si₃N₄= (
    A₀ = 1,
    B₁ = 3.0249,
    C₁ = (0.1353406)^2,         #                           [μm²]
    B₂ = 40314,
    C₂ = (1239.842)^2,          #                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
)

n²_Si₃N₄_sym = n²_sym_fmt1( λ ; p_Si₃N₄...)  # does not include temperature dependence, avoiding n² = ( √(n²(T₀)) + dndT*(T-T₀) )² for AD performance when using ε_Si₃N₄
n_Si₃N₄_λT_sym = sqrt(n²_Si₃N₄_sym) + p_Si₃N₄.dn_dT  *  ( T - p_Si₃N₄.T₀  )
n_Si₃N₄_sym = sqrt(n²_Si₃N₄_sym)

_n²_Si₃N₄= eval(build_function(n²_Si₃N₄_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_n_Si₃N₄_λT = eval(build_function(n_Si₃N₄_λT_sym,λ,T)) # inputs (λ,T) in ([μm],[°C])
_n_Si₃N₄= eval(build_function(n_Si₃N₄_sym,λ)) # inputs (λ,T) in ([μm],[°C])
_ng_Si₃N₄= eval(build_function(ng_sym(n_Si₃N₄_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_ng_Si₃N₄_λT = eval(build_function(ng_sym(n_Si₃N₄_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvd_Si₃N₄= eval(build_function(gvd_sym(n_Si₃N₄_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvd_Si₃N₄_λT = eval(build_function(gvd_sym(n_Si₃N₄_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])

ε_Si₃N₄_λT_sym = Diagonal(fill(n_Si₃N₄_λT_sym^2,3)) # n_Si₃N₄_λT_sym^2 * I
ε_Si₃N₄_sym = Diagonal(fill(n²_Si₃N₄_sym,3)) # n²_Si₃N₄_sym * I
_ε_Si₃N₄_λT, _ε_Si₃N₄_λT! = eval.(build_function(ε_Si₃N₄_λT_sym,λ,T))
_ε_Si₃N₄,_ε_Si₃N₄! = eval.(build_function(ε_Si₃N₄_sym,λ))
function ε_Si₃N₄(λ::T) where T<:Real
    _n²_Si₃N₄(λ) * I
    # Diagonal(fill(_n²_Si₃N₄(λ),3))
    # n² = _n²_Si₃N₄(λ)
    # SMatrix{3,3,T,9}( n²,    0.,     0.,
    #                   0.,     n²,    0.,
    #                   0.,     0.,     n², )
end

n²_Si₃N₄(λ) = _n²_Si₃N₄(λ)
n_Si₃N₄(λ) = _n_Si₃N₄(λ)
ng_Si₃N₄(λ) = _ng_Si₃N₄(λ)
n²_Si₃N₄(λ,T) = _n²_Si₃N₄_λT(λ,T)
n_Si₃N₄(λ,T) = _n_Si₃N₄_λT(λ,T)
ng_Si₃N₄(λ,T) = _ng_Si₃N₄_λT(λ,T)
n_Si₃N₄(λ::Unitful.Length,T::Unitful.Temperature) = _n_Si₃N₄((λ|>u"μm").val,(T|>u"°C").val)
n_Si₃N₄(λ::Unitful.Length) = _n_Si₃N₄((λ|>u"μm").val)
n_Si₃N₄(f::Unitful.Frequency,T::Unitful.Temperature) = _n_Si₃N₄(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
n_Si₃N₄(f::Unitful.Frequency) = _n_Si₃N₄(((Unitful.c0/f)|>u"μm").val)
ng_Si₃N₄(λ::Unitful.Length,T::Unitful.Temperature) = _ng_Si₃N₄((λ|>u"μm").val,(T|>u"°C").val)
ng_Si₃N₄(λ::Unitful.Length) = _ng_Si₃N₄((λ|>u"μm").val)
ng_Si₃N₄(f::Unitful.Frequency,T::Unitful.Temperature) = _ng_Si₃N₄(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)
ng_Si₃N₄(f::Unitful.Frequency) = _ng_Si₃N₄(((Unitful.c0/f)|>u"μm").val)
gvd_Si₃N₄(λ::Unitful.Length,T::Unitful.Temperature) = ( _gvd_Si₃N₄((λ|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_Si₃N₄(λ::Unitful.Length) = ( _gvd_Si₃N₄((λ|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_Si₃N₄(f::Unitful.Frequency,T::Unitful.Temperature) =( _gvd_Si₃N₄(((Unitful.c0/f)|>u"μm").val,(T|>u"°C").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"
gvd_Si₃N₄(f::Unitful.Frequency) = ( _gvd_Si₃N₄(((Unitful.c0/f)|>u"μm").val)u"μm" / ( 2π * c^2) ) |> u"fs^2 / mm"

# # ########### Si3N4 Sellmeier model for waveguide phase matching calculations
# #
# def n_Si3N4_sym():
#     """This function creates a symbolic representation (using SymPy) of the
#     Sellmeier Equation model for the index of refraction.
#     Equation form is based on Luke, Okawachi, Lamont, Gaeta and Lipson,
#     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
#     Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     https://doi.org/10.1364/OL.40.004823
#     valid from 0.31–5.504 um
#     Thermo-optic coefficients from
#     Xue, et al.
#     "Thermal tuning of Kerr frequency combs in silicon nitride microring resonators"
#     Opt. Express 24.1 (2016) http://doi.org/10.1364/OE.24.000687
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#     A0 = 1
#     B1 = 3.0249
#     C1 = (0.1353406)**2 # um^2
#     B2 = 40314
#     C2 = (1239.842)**2 # um^2)
#     T0 = 24.5 # reference temperature in [Deg C]
#     n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
#     return lm, T, n_sym
#
# def n_Si3N4(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of Si3N4. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_Si3N4_sym()
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output
#
# def n_g_Si3N4(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of Si3N4. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_Si3N4_sym()
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
#
# def gvd_Si3N4(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of Si3N4. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_Si3N4_sym()
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
#     dn_dT = 2.96e-5 # 1/degK
#     lm, T = sp.symbols('lm T')
#     T0 = 24.5 # reference temperature in [Deg C]
#     n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
#     return lm, T, n_sym
#


    # ################################################################################
    # ##                         LiB3O5, Lithium Triborate (LBO)                    ##
    # ################################################################################
    # def n_LBO_sym(axis='Z'):
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the temperature and wavelength dependent
    #     indices of refraction of LiB3O5 along all three principal axes, here labeled
    #     'X', 'Y' and 'Z'.
    #     Equation form is based on "Temperature dispersion of refractive indices in
    #     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
    #     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
    #     https://doi.org/10.1063/1.360499
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     # coefficient values for each axis
    #     lbo_coeffs = [
    #     [1.4426279, 1.0109932, .011210197, 1.2363218, 91, -127.70167e-6, 122.13435e-6, 53.0e-3],
    #     [1.5014015, 1.0388217, .0121571, 1.7567133, 91, 373.3387e-6, -415.10435e-6, 32.7e-3],
    #     [1.448924, 1.1365228, .011676746, 1.5830069, 91, -446.95031e-6, 419.33410e-6, 43.5e-3],
    #     ]
    #
    #     if axis is 'X':
    #         A,B,C,D,E,G,H,lmig = lbo_coeffs[0]
    #     elif axis is 'Y':
    #         A,B,C,D,E,G,H,lmig = lbo_coeffs[1]
    #     elif axis is 'Z':
    #         A,B,C,D,E,G,H,lmig = lbo_coeffs[2]
    #     else:
    #         raise Exception('unrecognized axis! must be "X","Y" or "Z"')
    #     # lm = sp.symbols('lm',positive=True)
    #     # T_C = T.to(u.degC).m
    #     # T0_C = 20. # reference temperature in [Deg C]
    #     # dT = T_C - T0_C
    #     lm,T = sp.symbols('lm, T',positive=True)
    #     R = lm**2 / ( lm**2 - lmig**2 ) #'normalized dispersive wavelength' for thermo-optic model
    #     dnsq_dT = G * R + H * R**2
    #     n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + dnsq_dT * ( T - 20. ) )
    #     # n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * ( T - 20. ) )
    #     # n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * dT )
    #     # n_sym =  A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * dT
    #     return lm,T, n_sym
    #
    # def n_LBO(lm_in,T_in,axis='Z'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of LiB3O5. Equation form is based on
    #     "Temperature dispersion of refractive indices in
    #     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
    #     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
    #     https://doi.org/10.1063/1.360499
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_LBO_sym(axis=axis)
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     # nsq = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #     # return n(lm_um)
    #
    # def n_g_LBO(lm_in,T_in,axis='Z'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of LiB3O5. Equation form is based on
    #     "Temperature dispersion of refractive indices in
    #     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
    #     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
    #     https://doi.org/10.1063/1.360499
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_LBO_sym(axis=axis)
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_LBO(lm_in,T_in,axis='Z'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of LiB3O5. Equation form is based on
    #     "Temperature dispersion of refractive indices in
    #     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
    #     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
    #     https://doi.org/10.1063/1.360499
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_LBO_sym(axis=axis)
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #
    #
    # ################################################################################
    # ##                          crystalline MgF2                                  ##
    # ################################################################################
    # def n_MgF2_sym(axis='e'):
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the wavelength dependence
    #     of crystalline MgF2's ordinary and extraordinary indices of refraction as
    #     well as the index of thin-film amorphous MgF2.
    #     Sellmeier coefficients for crystalline MgF2 are taken from
    #     "Refractive properties of magnesium fluoride"
    #     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
    #     https://doi.org/10.1364/AO.23.001980
    #     Sellmeier coefficients for amorphous MgF2 are taken from
    #     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
    #     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
    #     https://doi.org/10.1364/OME.7.000989
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     if axis is 'e':
    #         coeffs = [  (0.41344023,0.03684262), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
    #                     (0.50497499,0.09076162),
    #                     (2.4904862,23.771995),
    #                     ]
    #     elif axis is 'o':
    #         coeffs = [  (0.48755108,0.04338408), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
    #                     (0.39875031,0.09461442),
    #                     (2.3120353,23.793604),
    #                     ]
    #     elif axis is 'a':
    #         coeffs = [  (1.73,.0805), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
    #                     ]
    #     else:
    #         raise Exception('unrecognized axis! must be "e", "o" or "a"')
    #     lm, T = sp.symbols('lm T')
    #     A0,λ0 = coeffs[0]
    #     oscillators = A0 * lm**2 / (lm**2 - λ0**2)
    #     if len(coeffs)>1:
    #         for Ai,λi in coeffs[1:]:
    #             oscillators += Ai * lm**2 / (lm**2 - λi**2)
    #     n_sym = sp.sqrt( 1 + oscillators )
    #     return lm, T, n_sym
    #
    # def n_MgF2(lm_in,T_in=300*u.degK,axis='e'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') MgF2.
    #     Sellmeier coefficients for crystalline MgF2 are taken from
    #     "Refractive properties of magnesium fluoride"
    #     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
    #     https://doi.org/10.1364/AO.23.001980
    #     Sellmeier coefficients for amorphous MgF2 are taken from
    #     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
    #     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
    #     https://doi.org/10.1364/OME.7.000989
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_MgF2_sym(axis=axis)
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output.squeeze()
    #
    # def n_g_MgF2(lm_in,T_in=300*u.degK,axis='e'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') MgF2.
    #     Sellmeier coefficients for crystalline MgF2 are taken from
    #     "Refractive properties of magnesium fluoride"
    #     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
    #     https://doi.org/10.1364/AO.23.001980
    #     Sellmeier coefficients for amorphous MgF2 are taken from
    #     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
    #     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
    #     https://doi.org/10.1364/OME.7.000989
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_MgF2_sym(axis=axis)
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    # def gvd_MgF2(lm_in,T_in=300*u.degK,axis='e'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') MgF2.
    #     Sellmeier coefficients for crystalline MgF2 are taken from
    #     "Refractive properties of magnesium fluoride"
    #     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
    #     https://doi.org/10.1364/AO.23.001980
    #     Sellmeier coefficients for amorphous MgF2 are taken from
    #     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
    #     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
    #     https://doi.org/10.1364/OME.7.000989
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_MgF2_sym(axis=axis)
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #
    #
    #
    # ################################################################################
    # ##                         Gallium Arsenide (GaAs)                            ##
    # ################################################################################
    #
    # def n_GaAs_sym():
    #     """This function creates a symbolic representation (using SymPy) of the
    #     a model for the temperature and wavelength dependence of GaAs's refractive
    #     index. The equation form and fit parameters are based on "Improved dispersion
    #     relations for GaAs and applications to nonlinear optics" by Skauli et al.,
    #     JAP 94, 6447 (2003); doi: 10.1063/1.1621740
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #
    #     lm, T = sp.symbols('lm T')
    #     T0 = 22 # reference temperature in [Deg C]
    #     deltaT = T-T0
    #     A = 0.689578
    #     eps2 = 12.99386
    #     G3 = 2.18176e-3
    #     E0 = 1.425 - 3.7164e-4 * deltaT - 7.497e-7* deltaT**2
    #     E1 = 2.400356 - 5.1458e-4 * deltaT
    #     E2 = 7.691979 - 4.6545e-4 * deltaT
    #     E3 = 3.4303e-2 + 1.136e-5 * deltaT
    #     E_phot = (u.h * u.c).to(u.eV*u.um).magnitude / lm  #
    #     n_sym = sp.sqrt( 1  + (A/np.pi)*sp.log((E1**2 - E_phot**2) / (E0**2-E_phot**2)) \
    #                         + (eps2/np.pi)*sp.log((E2**2 - E_phot**2) / (E1**2-E_phot**2)) \
    #                         + G3 / (E3**2-E_phot**2) )
    #
    #     return lm, T, n_sym
    #
    # ###### Temperature Dependent Sellmeier Equation for phase-matching calculations
    # def n_GaAs(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of 5% MgO:LiNbO3. Equation form is based on
    #     "Temperature and wavelength dependent refractive index equations
    #     for MgO-doped congruent and stoichiometric LiNbO3"
    #     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    # #    lm_um = Q_(lm_in).to(u.um).magnitude
    # #    T_C = Q_(T_in).to(u.degC).magnitude
    # #    lm, T, n_sym = n_GaAs_sym()
    # #    n = sp.lambdify([lm,T],n_sym,'numpy')
    # #    return n(lm_um, T_C)
    #
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_GaAs_sym()
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #
    #
    # def n_g_GaAs(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of 5% MgO:LiNbO3. Equation form is based on
    #     "Temperature and wavelength dependent refractive index equations
    #     for MgO-doped congruent and stoichiometric LiNbO3"
    #     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_GaAs_sym()
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_GaAs(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of 5% MgO:LiNbO3. Equation form is based on
    #     "Temperature and wavelength dependent refractive index equations
    #     for MgO-doped congruent and stoichiometric LiNbO3"
    #     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_GaAs_sym()
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #
    #
    # ########### Si3N4 Sellmeier model for waveguide phase matching calculations
    #
    # def n_Si3N4_sym():
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the index of refraction.
    #     Equation form is based on Luke, Okawachi, Lamont, Gaeta and Lipson,
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     https://doi.org/10.1364/OL.40.004823
    #     valid from 0.31–5.504 um
    #     Thermo-optic coefficients from
    #     Xue, et al.
    #     "Thermal tuning of Kerr frequency combs in silicon nitride microring resonators"
    #     Opt. Express 24.1 (2016) http://doi.org/10.1364/OE.24.000687
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     A0 = 1
    #     B1 = 3.0249
    #     C1 = (0.1353406)**2 # um^2
    #     B2 = 40314
    #     C2 = (1239.842)**2 # um^2)
    #     T0 = 24.5 # reference temperature in [Deg C]
    #     n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
    #     return lm, T, n_sym
    #
    # def n_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #
    # def n_g_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #     dn_dT = 2.96e-5 # 1/degK
    #     lm, T = sp.symbols('lm T')
    #     T0 = 24.5 # reference temperature in [Deg C]
    #     n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
    #     return lm, T, n_sym
    #
    # def n_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #
    # def n_g_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_Si3N4(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of Si3N4. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_Si3N4_sym()
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')


    # ########### Silicon Sellmeier model for waveguide phase matching calculations
    #
    # def n_silicon_sym():
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the index of refraction.
    #     Equation form is based on Osgood, Panoiu, Dadap, et al.
    #     "Engineering nonlinearities in nanoscale optical systems: physics and
    # 	applications in dispersion-engineered silicon nanophotonic wires"
    # 	Advances in Optics and Photonics Vol. 1, Issue 1, pp. 162-235 (2009)
    #     https://doi.org/10.1364/AOP.1.000162
    #     valid from 1.2-?? um
    #     Thermo-optic coefficients from
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     eps = 11.6858
    #     A = 0.939816 # um^2
    #     B = 8.10461e-3
    #     lm1 = 1.1071 # um^2
    #     dn_dT = 0.0e-5 # 1/degK
    #     lm, T = sp.symbols('lm T')
    #     T0 = 24.5 # reference temperature in [Deg C]
    #     n_sym = sp.sqrt( eps  +  A / lm**2 + ( B * lm**2 ) / ( lm**2 - lm1 ) ) # + dn_dT * ( T - T0 )
    #     # n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
    #     return lm, T, n_sym
    #
    # def n_silicon(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of silicon. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_silicon_sym()
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #
    # def n_g_silicon(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of silicon. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_silicon_sym()
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_silicon(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of silicon. Equation form is based on
    #     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
    #     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_silicon_sym()
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #
    #
    # ########### SiO2 Sellmeier model for waveguide phase matching calculations
    #
    # def n_SiO₂_sym():
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the index of refraction.
    #     Equation form is based on Kitamura, et al.
    #     "Optical constants of silica glass from extreme ultraviolet to far
    #     infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
    #     which references Malitson, “Interspecimen comparison of the refractive
    #     index of fused silica,” J. Opt. Soc. Am.55,1205–1209 (1965)
    #     and has been validated from 0.21-6.7 μm (free space wavelength)
    #     Thermo-optic coefficients from the literature, forgot source.
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     A0 = 1
    #     B1 = 0.6961663
    #     C1 = (0.0684043)**2 # um^2
    #     B2 = 0.4079426
    #     C2 = (0.1162414)**2 # um^2
    #     B3 = 0.8974794
    #     C3 = (9.896161)**2 # um^2
    #     dn_dT = 6.1e-6 # 1/degK
    #     lm, T = sp.symbols('lm T')
    #     T0 = 20 # reference temperature in [Deg C]
    #     n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) + ( B3 * lm**2 ) / ( lm**2 - C3 ) ) + dn_dT * ( T - T0 )
    #     return lm, T, n_sym
    #
    # def n_SiO₂(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of SiO2. Equation form is based on
    #     Kitamura, et al.
    #     "Optical constants of silica glass from extreme ultraviolet to far
    #     infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_SiO₂_sym()
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output
    #
    # def n_g_SiO₂(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of SiO2. Equation form is based on
    #     Kitamura, et al.
    #     "Optical constants of silica glass from extreme ultraviolet to far
    #     infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_SiO₂_sym()
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    #
    # def gvd_SiO₂(lm_in,T_in):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of SiO2. Equation form is based on
    #     Kitamura, et al.
    #     "Optical constants of silica glass from extreme ultraviolet to far
    #     infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
    #     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_SiO₂_sym()
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
    #
    #
    # ################################################################################
    # ##                                  HfO2                                      ##
    # ################################################################################
    #
    # def n_HfO2_sym(axis='a'):
    #     """This function creates a symbolic representation (using SymPy) of the
    #     Sellmeier Equation model for the wavelength dependence
    #     of crystalline HfO2's ordinary and extraordinary indices of refraction as
    #     well as the index of thin-film amorphous HfO2.
    #     Sellmeier coefficients for crystalline HfO2 are taken from
    #     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
    #     films taken from
    #     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
    #     Surface and Coatings Technology 201.6 (2006)
    #     https://doi.org/10.1016/j.surfcoat.2006.08.074
    #     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
    #     for ALD Hafnia specifically. They also report loss, with a sharp absorption
    #     edge near 5.68 ± 0.09 eV (~218 nm)
    #     Eqn. form:
    #     n = A + B / lam**2 + C / lam**4
    #     ng = A + 3 * B / lam**2 + 5 * C / lam**4
    #     This model is then exported to other functions that use it and its
    #     derivatives to return index, group index and GVD values as a function
    #     of temperature and wavelength.
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     # if axis is 'e':
    #     #     coeffs = [  (0.41344023,0.03684262), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
    #     #                 (0.50497499,0.09076162),
    #     #                 (2.4904862,23.771995),
    #     #                 ]
    #     # elif axis is 'o':
    #     #     coeffs = [  (0.48755108,0.04338408), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
    #     #                 (0.39875031,0.09461442),
    #     #                 (2.3120353,23.793604),
    #     #                 ]
    #     if axis is 'a':
    #         lm, T = sp.symbols('lm T')
    #         # # fit for spectroscopic ellipsometer measurement for a 250nm thick film, good 300-1400nm
    #         # A = 1.85
    #         # B = 1.17e-2 # note values here are for λ in μm, vs nm in the paper
    #         # C = 0.0
    #         # fit for spectroscopic ellipsometer measurement for a 112nm thick film, good 300-1400nm
    #         A = 1.86
    #         B = 7.16e-3 # note values here are for λ in μm, vs nm in the paper
    #         C = 0.0
    #         n_sym =  A + B / lm**2 + C / lm**4
    #         return lm, T, n_sym
    #     if axis is 's':
    #         lm, T = sp.symbols('lm T')
    #         A = 1.59e4 # μm^-2
    #         λ0 = 210.16e-3 # μm, note values here are for λ in μm, vs nm in the paper
    #         n_sym = sp.sqrt(   1  + ( A * lm**2 ) / ( lm**2/λ0**2 - 1 ) )
    #         return lm, T, n_sym
    #     else:
    #         raise Exception('unrecognized axis! must be "e", "o" or "a"')
    #
    #
    # def n_HfO2(lm_in,T_in=300*u.degK,axis='a'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the index of refraction of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') HfO2.
    #     Sellmeier coefficients for crystalline HfO2 are taken from
    #     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
    #     films taken from
    #     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
    #     Surface and Coatings Technology 201.6 (2006)
    #     https://doi.org/10.1016/j.surfcoat.2006.08.074
    #     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
    #     for ALD Hafnia specifically. They also report loss, with a sharp absorption
    #     edge near 5.68 ± 0.09 eV (~218 nm)
    #     Eqn. form:
    #     n = A + B / lam**2 + C / lam**4
    #     ng = A + 3 * B / lam**2 + 5 * C / lam**4
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    #     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    #     lm, T, n_sym = n_HfO2_sym(axis=axis)
    #     n = sp.lambdify([lm,T],n_sym,'numpy')
    #     output = np.zeros((T_C.size,lm_um.size))
    #     for T_idx, TT in enumerate(T_C):
    #         output[T_idx,:] = n(lm_um, T_C[T_idx])
    #     return output.squeeze()
    #
    # def n_g_HfO2(lm_in,T_in=300*u.degK,axis='a'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group index of refraction of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') HfO2.
    #     Sellmeier coefficients for crystalline HfO2 are taken from
    #     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
    #     films taken from
    #     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
    #     Surface and Coatings Technology 201.6 (2006)
    #     https://doi.org/10.1016/j.surfcoat.2006.08.074
    #     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
    #     for ALD Hafnia specifically. They also report loss, with a sharp absorption
    #     edge near 5.68 ± 0.09 eV (~218 nm)
    #     Eqn. form:
    #     n = A + B / lam**2 + C / lam**4
    #     ng = A + 3 * B / lam**2 + 5 * C / lam**4
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_HfO2_sym(axis=axis)
    #     n_sym_prime = sp.diff(n_sym,lm)
    #     n_g_sym = n_sym - lm*n_sym_prime
    #     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    #     return n_g(lm_um, T_C)
    #
    # def gvd_HfO2(lm_in,T_in=300*u.degK,axis='a'):
    #     """Sellmeier Equation model for the temperature and wavelength dependence
    #     of the group velocity dispersion of crystalline (axis='o' or 'e') and
    #     amorphous (axis='a') HfO2.
    #     Sellmeier coefficients for crystalline HfO2 are taken from
    #     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
    #     films taken from
    #     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
    #     Surface and Coatings Technology 201.6 (2006)
    #     https://doi.org/10.1016/j.surfcoat.2006.08.074
    #     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
    #     for ALD Hafnia specifically. They also report loss, with a sharp absorption
    #     edge near 5.68 ± 0.09 eV (~218 nm)
    #     Eqn. form:
    #     n = A + B / lam**2 + C / lam**4
    #     ng = A + 3 * B / lam**2 + 5 * C / lam**4
    #     Variable units are lm in [um] and T in [deg C]
    #     """
    #     lm_um = Q_(lm_in).to(u.um).magnitude
    #     T_C = Q_(T_in).to(u.degC).magnitude
    #     lm, T, n_sym = n_HfO2_sym(axis=axis)
    #     n_sym_double_prime = sp.diff(n_sym,lm,lm)
    #     c = Q_(3e8,'m/s') # unitful definition of speed of light
    #     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    #     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    #     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    #     return gvd.to('fs**2 / mm')
