################################################################################
#                                   germanium                                    #
################################################################################
export germanium, n²_germanium, n²_ω_germanium

"""
This code creates a symbolic representation of the Sellmeier Equation model
for the index of refraction of crystalline germanium. Equation form is based on:
    Frey, Leviton and Madison, "Temperature-dependent refractive index of germanium and germanium"
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

	1.9 μm 	<	λ	<	5.5 μm
	20 K	<	T 	<	300 K

and should match decently well slightly outside these ranges.

The symbolic index model and its derivatives in turn are used to generate
numerical functions for the germanium index, group index and GVD as a function
of temperature and wavelength.

Variable units are lm in [um] and T in [deg C]
"""

# Sij_lmij_ge     =  [
# 	13.9723		0.452096	751.447	0.386367	1.08843	-2893.19
# 	2.52809E-03	-3.09197E-03	-14.2843	2.01871E-04	1.16510E-03	-0.967948
# 	-5.02195E-06	2.16895E-05	-0.238093	-5.93448E-07	-4.97284E-06	-0.527016
# 	2.22604E-08	-6.02290E-08	2.96047E-03	-2.27923E-10	1.12357E-08	6.49364E-03
# 	-4.86238E-12	4.12038E-11	-7.73454E-06	5.37423E-12	9.40201E-12	-1.95162E-05
# ]' 
# Sᵢⱼ_germanium =   Sij_lmij_ge[1:3,:]
# λᵢⱼ_germanium =   Sij_lmij_ge[4:6,:]

Sᵢⱼ_germanium =   [   
	13.9723     0.00252809  	-5.02195e-6   	2.22604e-8  	-4.86238e-12
   	0.452096  	-0.00309197   	2.16895e-5  	-6.0229e-8    	4.12038e-11
   	751.447     -14.2843      	-0.238093     	0.00296047  	-7.73454e-6
]

λᵢⱼ_germanium =   [   
    0.386367 	0.000201871 	-5.93448e-7  	-2.27923e-10   	5.37423e-12
    1.08843    	0.0011651    	-4.97284e-6   	1.12357e-8    	9.40201e-12
	-2893.19    -0.967948    	-0.527016     	0.00649364  	-1.95162e-5
]

n²_germanium(λ, T)    =   n²_sym_NASA( λ, T ; Sᵢⱼ=Sᵢⱼ_germanium, λᵢⱼ=λᵢⱼ_germanium,)
n²_ω_germanium(ω, T)  =   n²_sym_NASA_ω( ω, T ; Sᵢⱼ=Sᵢⱼ_germanium, λᵢⱼ=λᵢⱼ_germanium,)

function make_germanium()
	@variables ω, λ, T
	n² = n²_ω_germanium(ω,T-273.15)
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
	Material(models, defaults, :germanium, colorant"brown")
end

################################################################

germanium = make_germanium()
