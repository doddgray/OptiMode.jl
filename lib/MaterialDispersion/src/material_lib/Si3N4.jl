################################################################################
#                                   Si₃N₄                                      #
################################################################################
export Si₃N₄, Si₃N₄_GW1, Si₃N₄_GW2

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
    B₂ = 40314.0,
    C₂ = (1239.842)^2,          #                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
    dn²_dT = 2*sqrt(n²_sym_fmt1( 1.55 ; A₀ = 1, B₁ = 3.0249, C₁ = (0.1353406)^2, B₂ = 40314, C₂ = (1239.842)^2,))*2.96e-5,
)   
# # The last term is just 2n₀*dn_dT where n₀=n(λ₀,T₀) is index at the wavelength and temperature where 
# # the thermo-optic coefficient `dn_dT` was measured. `dn²_dT` is the lowest order (linear) thermo-optic
# # coefficient for n²(λ,T) corresponding to `dn_dT`, and avoids square roots which complicate computer algebra.
# # This neglects the wavelength/frequency dependence of thermo-optic coupling, just like `dn_dT`. 

n²_Si₃N₄(λ,T) = n²_sym_fmt1( λ ; p_n²_Si₃N₄...) + p_n²_Si₃N₄.dn²_dT  *  ( T - p_n²_Si₃N₄.T₀ ) 
n²_Si₃N₄_ω(ω,T) = n²_sym_fmt1_ω( ω ; p_n²_Si₃N₄...) + p_n²_Si₃N₄.dn²_dT  *  ( T - p_n²_Si₃N₄.T₀ )

function make_Si₃N₄(;p_n²=p_n²_Si₃N₄)
	@variables ω, λ, T
	n² = n²_Si₃N₄_ω(ω,T)
	n_λ = sqrt(substitute(n²,Dict([(ω=>1/λ),]))) 
	ng = ng_model(n_λ,λ)
	gvd = gvd_model(n_λ,λ)
	models = Dict([
		:n		=>	n_λ,
		:ng		=>	ng,
		:gvd	=>	gvd,
		:ε 		=> 	diagm([n², n², n²]),
		# Kerr (intensity-dependent index) coefficient n₂ ≈ 2.4×10⁻¹⁹ m²/W = 2.4×10⁻⁷ μm²/W
		# for stoichiometric LPCVD Si₃N₄ near 1.55 μm; see Ikeda et al., Opt. Express 16,
		# 12987 (2008) and Krückel et al., Opt. Express 25, 15370 (2017).
		:n₂		=>	2.4e-7,			# μm²/W
	])
	defaults =	Dict([
		:ω		=>		inv(0.8),	# μm⁻¹
		:λ		=>		0.8,		# μm
		:T		=>		p_n².T₀,	# °C

	])
	Material(models, defaults, :Si₃N₄, colorant"firebrick1")
end

################################################################

Si₃N₄ = make_Si₃N₄()


### TODO: Update to Gavin's fit data: (from Slack, Jan 25 2022)
# epsilon = 1 + B*lambda^2/(lambda^2 - lambda0^2)
# Material 1: B = 2.63635 lambda0 = 0.14647
# Material 2: B = 2.49153 lambda0 = 0.13063

# Using the same generic Sellmeier form as above:
#
# function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
#     λ² = λ^2
#     A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
# end
#
# function n²_sym_fmt1_ω( ω ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
#     A₀  + B₁ / ( 1 - C₁*ω^2 ) + B₂ / ( 1 - C₂*ω^2 ) + B₃ / ( 1 - C₃*ω^2 )
# end


p_n²_Si₃N₄_GW1= (
    A₀ = 1,
    B₁  = 2.63635,
    C₁ = (0.14647)^2,         	#                           [μm²]
    B₂ = 0,
    C₂ = 1,          			#                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
    dn²_dT = 2*sqrt(n²_sym_fmt1( 1.55 ; A₀ = 1, B₁ = 2.63635, C₁ = (0.14647)^2, B₂ = 0, C₂ = 1,))*2.96e-5,
)   

p_n²_Si₃N₄_GW2= (
    A₀ = 1,
    B₁  = 2.49153,
    C₁ = (0.13063)^2,         	#                           [μm²]
    B₂ = 0,
    C₂ = 1,          			#                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
    dn²_dT = 2*sqrt(n²_sym_fmt1( 1.55 ; A₀ = 1, B₁ = 2.49153, C₁ = (0.13063)^2, B₂ = 0, C₂ = 1,))*2.96e-5,
)   
# # As above the last term is just 2n₀*dn_dT where n₀=n(λ₀,T₀) is index at the wavelength and temperature where 
# # the thermo-optic coefficient `dn_dT` was measured. `dn²_dT` is the lowest order (linear) thermo-optic
# # coefficient for n²(λ,T) corresponding to `dn_dT`, and avoids square roots which complicate computer algebra.
# # This neglects the wavelength/frequency dependence of thermo-optic coupling, just like `dn_dT`. 

function make_Si₃N₄2(material_symbol;p_n²=p_n²_Si₃N₄_GW1)
	@variables ω, λ, T
	n² = n²_sym_fmt1_ω( ω ; p_n²...) + 	p_n².dn²_dT  *  ( T - p_n².T₀ )
	n_λ = sqrt(substitute(n²,Dict([(ω=>1/λ),]))) 
	ng = ng_model(n_λ,λ)
	gvd = gvd_model(n_λ,λ)
	models = Dict([
		:n		=>	n_λ,
		:ng		=>	ng,
		:gvd	=>	gvd,
		:ε 		=> 	diagm([n², n², n²]),
	])
	defaults =	Dict([
		:ω		=>		inv(0.8),	# μm⁻¹
		:λ		=>		0.8,		# μm
		:T		=>		p_n².T₀,	# °C

	])
	Material(models, defaults, material_symbol, colorant"firebrick1")
end

Si₃N₄_GW1 = make_Si₃N₄2(:Si₃N₄_GW1; p_n²=p_n²_Si₃N₄_GW1)
Si₃N₄_GW2 = make_Si₃N₄2(:Si₃N₄_GW2; p_n²=p_n²_Si₃N₄_GW2)