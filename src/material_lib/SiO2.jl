################################################################################
#                                   SiO₂                                       #
################################################################################
export SiO₂

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

p_n²_SiO₂ = (
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

function make_SiO₂(;p_n²=p_n²_SiO₂)
	@variables λ, T
	n = sqrt(n²_sym_fmt1( λ ; p_n²...)) + p_n².dn_dT  *  ( T - p_n².T₀  )
	n² = n^2
	ε 	= diagm([n², n², n²])
	ng = ng_model(n,λ)
	gvd = gvd_model(n,λ)
	models = Dict([
		:n		=>	n,
		:ng		=>	ng,
		:gvd	=>	gvd,
		:ε 		=> 	diagm([n², n², n²]),
	])
	defaults =	Dict([
		:λ		=>		0.8,		# μm
		:T		=>		p_n².T₀,	# °C

	])
	Material(models, defaults, :SiO₂)
end

################################################################

SiO₂ = make_SiO₂()
