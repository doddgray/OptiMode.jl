################################################################################
#                                   Si₃N₄                                      #
################################################################################
export ε_Si₃N₄,n²_Si₃N₄,n_Si₃N₄,ng_Si₃N₄,gvd_Si₃N₄,Si₃N₄

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
_ng_Si₃N₄= eval(build_function(ng(n_Si₃N₄_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_ng_Si₃N₄_λT = eval(build_function(ng(n_Si₃N₄_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])
_gvd_Si₃N₄= eval(build_function(gvd(n_Si₃N₄_sym),λ)) # inputs (λ,T) in ([μm],[°C])
_gvd_Si₃N₄_λT = eval(build_function(gvd(n_Si₃N₄_λT_sym),λ,T)) # inputs (λ,T) in ([μm],[°C])

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

Si₃N₄ = Material(ε_Si₃N₄_sym,ε_Si₃N₄,x->ng_Si₃N₄(x)*I)
