###########################################################################
#                           Congruent LiNbO₃                              #
###########################################################################
export LiNbO₃

"""
These functions create a symbolic representation (using ModelingToolkit) of the
Sellmeier Equation model for the temperature and wavelength dependence
of Congruent, undoped LiNbO3's ordinary and extraordinary indices of refraction.
The extraordinary refractive index data is from D. Jundt "Temperature-Dependent Sellmeier 
equation for index of refraction, ne, in congruent lithium niobate", Optics Letters (1997). 
The ordinary refractive index is based off  data from "Infrared corrected Sellmeier 
coefficients for congruently grown lithium niobate and 5 mol.% magnesium oxide–doped 
lithium niobate" by Zelmon et al., J. Opt. Soc. Am. B (1997). 
The fit function form and thermo-optic coefficients follow  
"Temperature and wavelength dependent refractive index equations for MgO-doped 
congruent and stoichiometric LiNbO3" by Gayer et al., Applied Physics B 91,
343?348 (2008), under the assumption that the thermo-optic coefficients are
not strongly dependent on the MgO doping level. This assumption is supported by D. Matsuda, 
T. Mizuno, and N. Umemura, "Temperature-dependent phase-matching properties with oo-e and 
oo-o interactions in 5mol% MgO doped congruent LiNbO3", Proc. SPIE (2015).

This model is then exported to other functions that use it and its
derivatives to return index, group index and GVD values as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]
"""
function n²_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

function n²_LiNbO₃_sym_ω(ω, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    a₁ + b₁*f + (a₂ + b₂*f)*ω^2 / (1 - (a₃ + b₃*f)^2*ω^2) + (a₄ + b₄*f)*ω^2 / (1 - a₅^2*ω^2) - a₆ / ω^2
end

pₑ = (
    a₁ = 5.35583,
    a₂ = 0.100473,
    a₃ = 0.20692,
    a₄ = 100,
    a₅ = 11.34927,
    a₆ = 1.5334e-2,
    b₁ = 4.629e-7,
    b₂ = 3.862e-8,
    b₃ = -0.89e-8,
    b₄ = 2.657e-5,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ = (
    a₁ = 5.7731707,
    a₂ = 0.1182989,
    a₃ = 0.2178778,
    a₄ = 123.056818,
    a₅ = 11.9071143,
    a₆ = 2.06701e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 21.0,      # reference temperature in [Deg C]
)

pᵪ₂ = (
	d₃₃ =   20.3,    #   pm/V at 1.313 um
	d₃₁ =   -4.1,    #   pm/V 
	d₂₂ =   2.1,     #   pm/V
	λs  =  [1.313, 1.313, 1.313/2.0]
)

function make_LiNbO₃(;pₒ=pₒ,pₑ=pₑ,pᵪ₂=pᵪ₂)
	@variables λ, ω, T, λs[1:3]

	nₒ² = n²_LiNbO₃_sym_ω(ω, T; pₒ...)
	nₑ² = n²_LiNbO₃_sym_ω(ω, T; pₑ...)
	ε 	= diagm([nₒ², nₒ², nₑ²])
	d₃₃, d₃₁, d₂₂, λᵣs = pᵪ₂
	χ⁽²⁾ᵣ = cat(
		[ 	0.0	 	-d₂₂ 	d₃₁			#	xxx, xxy and xxz
		 	-d₂₂	0.0 	0.0			#	xyx, xyy and xyz
			d₃₁	 	0.0		0.0		],	#	xzx, xzy and xzz
		[ 	-d₂₂	0.0 	0.0			#	yxx, yxy and yxz
			0.0	 	d₂₂ 	d₃₁			#	yyx, yyy and yyz
			0.0	 	d₃₁		0.0		],	#	yzx, yzy and yzz
		[ 	d₃₁	 	0.0 	0.0			#	zxx, zxy and zxz
			0.0	 	d₃₁ 	0.0			#	zyx, zyy and zyz
			0.0	 	0.0 	d₃₃		],	#	zzx, zzy and zzz
		 dims = 3
	)
	

	ε_λ = substitute.(ε,(Dict([(ω=>1/λ),]),))
	nₒ_λ,nₑ_λ = sqrt.(substitute.((nₒ²,nₑ²),(Dict([(ω=>1/λ),]),)))

	models = Dict([
		:nₒ		=>	nₒ_λ,
		:nₑ		=>	nₑ_λ,
		:ε 		=> 	ε,
		:χ⁽²⁾	=>	SArray{Tuple{3,3,3}}(Δₘ(λs,ε_λ, λᵣs, χ⁽²⁾ᵣ)),
	])
	defaults =	Dict([
		:ω		=>		inv(0.8),	# μm⁻¹
		:λ		=>		0.8,		# μm
		:T		=>		24.5,		# °C
		:λs₁	=>		1.064,		# μm
		:λs₂	=>		1.064,		# μm
		:λs₃	=>		0.532,		# μm

	])
	Material(
		models,
		defaults,
		:LiNbO₃,
		colorant"seagreen2", #RGB{N0f8}(0.22,0.596,0.149),	#
		)
end

################################################################

LiNbO₃ = make_LiNbO₃()
