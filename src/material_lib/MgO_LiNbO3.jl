################################################################################
#                                 MgO:LiNbO₃                                   #
################################################################################
export MgO_LiNbO₃

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
function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

function n²_MgO_LiNbO₃_sym_ω(ω, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    a₁ + b₁*f + (a₂ + b₂*f)*ω^2 / (1 - (a₃ + b₃*f)^2*ω^2) + (a₄ + b₄*f)*ω^2 / (1 - a₅^2*ω^2) - a₆ / ω^2
end

pₑ = (
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
pₒ = (
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

pᵪ₂ = (
	d₃₃ =   20.3,    #   pm/V
	d₃₁ =   -4.1,    #   pm/V
	d₂₂ =   2.1,     #   pm/V
	λs  =  [1.313, 1.313, 1.313/2.0]
)

function make_MgO_LiNbO₃(;pₒ=pₒ,pₑ=pₑ,pᵪ₂=pᵪ₂)
	@variables λ, ω, T, λs[1:3]
	# nₒ² = n²_MgO_LiNbO₃_sym(λ, T; pₒ...)
	# nₑ² = n²_MgO_LiNbO₃_sym(λ, T; pₑ...)
	nₒ² = n²_MgO_LiNbO₃_sym_ω(ω, T; pₒ...)
	nₑ² = n²_MgO_LiNbO₃_sym_ω(ω, T; pₑ...)
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
	
	# ngₒ = ng_model(nₒ,ω)
	# gvdₒ = gvd_model(nₒ,ω)
	
	ε_λ = substitute.(ε,(Dict([(ω=>1/λ),]),))
	nₒ_λ,nₑ_λ = sqrt.(substitute.((nₒ²,nₑ²),(Dict([(ω=>1/λ),]),)))
	# nₑ = sqrt(nₑ²)
	# nₒ = sqrt(nₒ²)
	# ngₑ = ng_model(nₑ,ω)
	# gvdₑ = gvd_model(nₑ,ω)
	models = Dict([
		:nₒ		=>	nₒ_λ,
		# :ngₒ	=>	ngₒ,
		# :gvdₒ	=>	gvdₒ,
		:nₑ		=>	nₑ_λ,
		# :ngₑ	=>	ngₑ,
		# :gvdₑ	=>	gvdₑ,
		# :ng		=>	diagm([ngₒ, ngₒ, ngₑ]),
		# :gvd	=>	diagm([gvdₒ, gvdₒ, gvdₑ]),
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
		:MgO_LiNbO₃,
		colorant"seagreen2", #RGB{N0f8}(0.22,0.596,0.149),	#
		)
end

################################################################

MgO_LiNbO₃ = make_MgO_LiNbO₃()
