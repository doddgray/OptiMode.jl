###########################################################################
#                           Congruent LiNbO₃                              #
###########################################################################
export LiNbO₃_uz

"""
These functions create a symbolic representation (using ModelingToolkit) of the
Sellmeier Equation model for the temperature and wavelength dependence
of Congruent, undoped LiNbO3's ordinary and extraordinary indices of refraction.
The refractive index is based off  data from "Infrared corrected Sellmeier 
coefficients for congruently grown lithium niobate and 5 mol.% magnesium oxide–doped 
lithium niobate" by Zelmon et al., J. Opt. Soc. Am. B (1997). 

The fit function form and thermo-optic coefficients follow
"Sellmeier and thermo-optic dispersion formulas for the extraordinary ray of
5 mol. % MgO-doped congruent LiNbO3 in the visible, infrared, and terahertz 
regions", Umemura, et al., Applied Optics Vol. 53 No. 25 (2014), under the 
assumption that the thermo-optic coefficients are not strongly dependent on 
the MgO doping level. This assumption is supported by D. Matsuda, T. Mizuno, 
and N. Umemura, "Temperature-dependent phase-matching properties with oo-e and 
oo-o interactions in 5mol% MgO doped congruent LiNbO3", Proc. SPIE (2015).

This model is then exported to other functions that use it and its
derivatives to return index, group index and GVD values as a function
of temperature and wavelength.
Variable units are lm in [um] and T in [deg C]

This model is valid from 0.5 μm to 3.0 μm.

"""
function nₒ²_LiNbO₃_sym(λ, T; a₁, a₂, a₃, b₁, b₂, c₀, c₁, c₂, c₃, c₄, d₁, T₀)
	ΔT = T-T₀
	∫∂n₀ = (c₄/λ^4 + c₃/λ^3 + c₂/λ^2 + c₁/λ + c₀) * ΔT * (2+ d₁*(ΔT + 2*T₀)) / 2.0
    λ² = λ^2
    n₀² = a₁ + a₂ / (λ² - b₁) + a₃ / (λ² - b₂) + ∫∂n₀
	return n₀²
end

function nₒ²_LiNbO₃_sym_ω(ω, T; a₁, a₂, a₃, b₁, b₂, c₀, c₁, c₂, c₃, c₄, d₁, T₀)
    ΔT = T-T₀
	∫∂n₀ = (c₄*ω^4 + c₃*ω^3 + c₂*ω^2 + c₁*ω + c₀) * ΔT * (2+ d₁*(ΔT + 2*T₀)) / 2.0
    ω² = ω^2
    n₀² = a₁ + a₂*ω² / (1 - b₁*ω²) + a₃*ω² / (1 - b₂*ω²) + ∫∂n₀
	return n₀²
end

function nₑ²_LiNbO₃_sym(λ, T; a₁, a₂, a₃, b₁, c₀₀, c₀, c₁, c₂, c₃, d₁, T₀)
	ΔT = T-T₀
	∫∂nₑ = (c₃/λ^3 + c₂/λ^2 + c₁/λ + c₀ + c₀₀*λ) * ΔT * (2+ d₁*(ΔT + 2*T₀)) / 2.0
    λ² = λ^2
    nₑ² = a₁ + a₂ / (λ² - b₁) + a₃*λ² + ∫∂nₑ
	return nₑ²
end

function nₑ²_LiNbO₃_sym_ω(ω, T; a₁, a₂, a₃, b₁, c₀, c₁, c₂, c₃, d₁, T₀)
	ΔT = T-T₀
	∫∂nₑ = (c₃*ω^3 + c₂*ω^2 + c₁*ω + c₀ + c₀₀/ω) * ΔT * (2+ d₁*(ΔT + 2*T₀)) / 2.0
    ω² = ω^2
    nₑ² = a₁ + a₂*ω² / (1 - b₁*ω² ) + a₃/ω² + ∫∂nₑ
	return nₑ²
end

pₑ = (
    a₁ = 4.58045,
    a₂ = 0.11173,
    a₃ = -0.02215,
    b₁ = 0.04393,
	c₀₀ = -0.0744e-5,
	c₀ = 3.5332e-5,
	c₁ = 0.9036e-5,
	c₂ = -0.6643e-5,
	c₃ = 0.4175e-5,
	d₁ = 0.00276, 
    T₀ = 20.0,      # reference temperature in [Deg C]
)
pₒ = (
    a₁ = 20.50733,
    a₂ = 0.11898,
    a₃ = 9109.21880,
    b₁ = 0.04596,
    b₂ = 583.76424,
	c₀ = 1.0908e-5,
	c₁ = -2.9264e-5,
	c₂ = 4.0283e-5,
	c₃ = -2.1143e-5,
	c₄ = 0.4519e-5,
	d₁ = 0.00216,
	T₀ = 20.0,      # reference temperature in [Deg C]
)

pᵪ₂ = (   # NOTE: These are values for MgO:LN, need to be corrected for undoped LN. 
	d₃₃ =   20.3,    #   pm/V
	d₃₁ =   -4.1,    #   pm/V
	d₂₂ =   2.1,     #   pm/V
	λs  =  [1.313, 1.313, 1.313/2.0]
)

function make_LiNbO₃(;pₒ=pₒ,pₑ=pₑ,pᵪ₂=pᵪ₂)
	@variables λ, ω, T, λs[1:3]
	nₒ² = nₒ²_MgO_LiNbO₃_sym_ω(ω, T; pₒ...)
	nₑ² = nₑ²_MgO_LiNbO₃_sym_ω(ω, T; pₑ...)
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

LiNbO₃_uz = make_LiNbO₃()