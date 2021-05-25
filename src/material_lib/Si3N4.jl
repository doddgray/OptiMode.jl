################################################################################
#                                   Si₃N₄                                      #
################################################################################
export Si₃N₄

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

p_n²_Si₃N₄= (
    A₀ = 1,
    B₁ = 3.0249,
    C₁ = (0.1353406)^2,         #                           [μm²]
    B₂ = 40314,
    C₂ = (1239.842)^2,          #                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
)

function make_Si₃N₄(;p_n²=p_n²_Si₃N₄)
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
	Material(models, defaults, :Si₃N₄, colorant"firebrick1")
end

################################################################

Si₃N₄ = make_Si₃N₄()
