################################################################################
#                                   silicon                                    #
################################################################################
export silicon, n²_silicon, n²_ω_silicon

"""
This code creates a symbolic representation of the Sellmeier Equation model
for the index of refraction of crystalline silicon. Equation form is based on:
    Frey, Leviton and Madison, "Temperature-dependent refractive index of silicon and germanium"
	https://arxiv.org/pdf/physics/0606168.pdf

in work from NASA Goddard using their Cryogenic High-Accuracy Refraction Measuring System (CHARMS).

The squared index of refraction n² is approximated in a Sellmeier form 

	n² = 1 + ∑ᵢ ( Sᵢ * λ² ) / ( λ² - λᵢ² )

with temperature-dependent coefficients Sᵢ and λᵢ representing the strengths and vacuum 
wavelengths of optical resonances, respectively. Sᵢ and λᵢ are both calcualted as fourth-order
polynomials in absolute temperature `T` (in deg. Kelvin). Model parameters are supplied as
n × 5 matrices Sᵢⱼ and λᵢⱼ, where n is the number of Sellmeier terms. Sᵢ and λᵢ are 
calculated as dot products

	Sᵢ	=	Sᵢⱼ ⋅ [1, T, T^2, T^3, T^4]
	λᵢ	=	λᵢⱼ ⋅ [1, T, T^2, T^3, T^4]

In the referenced paper three-term Sellemeier forms are used, and thus Sᵢⱼ and λᵢⱼ of the form

	Sᵢⱼ	= 	[	S₀₁		S₁₁		S₁₂		S₁₃		S₁₄
				S₀₂		S₂₁		S₂₂		S₂₃		S₂₄
				S₀₃		S₃₁		S₃₂		S₃₃		S₃₄		]

	λᵢⱼ	= 	[	λ₀₁		λ₁₁		λ₁₂		λ₁₃		λ₁₄
				λ₀₂		λ₂₁		λ₂₂		λ₂₃		λ₂₄
				λ₀₃		λ₃₁		λ₃₂		λ₃₃		λ₃₄		]

is provided for germanium Table 10 of the referenced paper. The model matches experiment
    in the vacuum wavelength and temperature ranges:
    
        1.1 μm 	<	λ	<	5.6 μm
        20 K	<	T 	<	300 K
    
and should match decently well slightly outside these ranges.

The symbolic index model and its derivatives in turn are used to generate
numerical functions for the silicon index, group index and GVD as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]
"""

# Sij_lmij_si     =  [
#     10.4907     -1346.61        4.42827E+07     0.299713     -3.51710E+03      1.71400E+06
#     -2.08020E-04        29.1664     -1.76213E+06     -1.14234E-05      42.3892     -1.44984E+05
#     4.21694E-06     -0.278724       -7.61575E+04    1.67134E-07   -0.357957        -6.90744E+03
#     -5.82298E-09        1.05939E-03     678.414      -2.51049E-10      1.17504E-03    -39.3699
#     3.44688E-12     -1.35089E-06        103.243     2.32484E-14      -1.13212E-06      23.5770
# ]' 
# Sᵢⱼ_silicon =   Sij_lmij_si[1:3,:]
# λᵢⱼ_silicon =   Sij_lmij_si[4:6,:]

Sᵢⱼ_silicon =   [   
    10.4907     -0.00020802     4.21694e-6      -5.82298e-9     3.44688e-12
    -1346.61    29.1664         -0.278724       0.00105939      -1.35089e-6
    4.42827e7   -1.76213e6      -76157.5        678.414         103.243      
]

λᵢⱼ_silicon =   [   
    0.299713    -1.14234e-5     1.67134e-7      -2.51049e-10    2.32484e-14
    -3517.1     42.3892         -0.357957       0.00117504      -1.13212e-6
    1.714e6     -144984.0       -6907.44        -39.3699        23.577
]

n²_silicon(λ, T)    =   n²_sym_NASA( λ, T ; Sᵢⱼ=Sᵢⱼ_silicon, λᵢⱼ=λᵢⱼ_silicon,)
n²_ω_silicon(ω, T)  =   n²_sym_NASA_ω( ω, T ; Sᵢⱼ=Sᵢⱼ_silicon, λᵢⱼ=λᵢⱼ_silicon,)


function make_silicon()
	@variables ω, λ, T
	n² = n²_ω_silicon(ω,T-273.15)
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
		:ω		=>		inv(1.55),	# μm⁻¹
		:λ		=>		1.55,		# μm
		:T		=>		295.0,	    # °C

	])
	Material(models, defaults, :silicon, colorant"grey")
end

################################################################

silicon = make_silicon()
