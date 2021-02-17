################################################################################
#                                   SiO₂                                       #
################################################################################
export ε_SiO₂,n²_SiO₂,n_SiO₂,ng_SiO₂,gvd_SiO₂,SiO₂

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
_ng_SiO₂ = eval(build_function(ng(n_SiO₂_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_ng_SiO₂_λT = eval(build_function(ng(n_SiO₂_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvd_SiO₂ = eval(build_function(gvd(n_SiO₂_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvd_SiO₂_λT = eval(build_function(gvd(n_SiO₂_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])

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

SiO₂ = Material(SMatrix{3,3}(ε_SiO₂_sym))
