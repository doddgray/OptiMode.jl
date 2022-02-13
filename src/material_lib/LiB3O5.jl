################################################################################
#                         LiB₃O₅, Lithium Triborate (LBO)                      #
################################################################################
export LiB₃O₅

"""
These functions create a symbolic representation (using ModelingToolkit) of the
Sellmeier Equation model for the temperature and wavelength dependent
indices of refraction of LiB₃O₅ (Lithium Triborate) along all three principal axes,
here labeled '1', '2' and '3'.
Equation form is based on "Temperature dispersion of refractive indices in
β‐BaB2O4 and LiB₃O₅ crystals for nonlinear optical devices"
by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
https://doi.org/10.1063/1.360499
This model is then exported to other functions that use it and its
derivatives to return index, group index and GVD values as a function
of temperature and wavelength.
Variables (units):
	- `λ` (μm)			vacuum wavelength 
	- `T` (degree C)	material temperature
"""

function n²_LiB₃O₅_sym(λ, T; A, B, C, D, E, G, H, λᵢ, T₀)
    λ² = λ^2
	R = λ² / ( λ² - λᵢ^2 ) #'normalized dispersive wavelength' for thermo-optic model
    dn²dT = G * R + H * R^2
	n² = A + B * λ² / ( λ² - C ) + D * λ² / ( λ² - E ) + dn²dT * ( T - T₀ )
end

function n²_LiB₃O₅_sym_ω(ω, T; A, B, C, D, E, G, H, λᵢ, T₀)
	R = 1 / ( 1 - λᵢ^2*ω^2 ) #'normalized dispersive wavelength' for thermo-optic model
    dn²dT = G * R + H * R^2
	n² = A + B / ( 1 - C*ω^2 ) + D / ( 1 - E*ω^2 ) + dn²dT * ( T - T₀ )
	return n²
end

p_n₁²_LiB₃O₅ = (
    A	=	1.4426279,
	B	=	1.0109932,
	C 	=	0.011210197,
	D 	=	1.2363218,
	E 	=	91.0,
	G 	=	-127.70167e-6,
	H 	=	122.13435e-6,
	λᵢ	=	53.0e-3,
    T₀ 	=	20.0,      # reference temperature in [Deg C]
)

p_n₂²_LiB₃O₅ = (
    A	=	1.5014015, 
	B	=	1.0388217,
	C 	=	0.0121571,
	D 	=	1.7567133,
	E 	=	91.0,
	G 	=	373.3387e-6,
	H 	=	-415.10435e-6,
	λᵢ	=	32.7e-3,
    T₀ 	=	20.0,      # reference temperature in [Deg C]
)

p_n₃²_LiB₃O₅ = (
    A	=	1.448924,
	B	=	1.1365228,
	C 	=	0.011676746,
	D 	=	1.5830069,
	E 	=	91.0,
	G 	=	-446.95031e-6,
	H 	=	419.33410e-6,
	λᵢ	=	43.5e-3,
    T₀ 	=	20.0,      # reference temperature in [Deg C]
)

# ref Table II from:
# https://doi.org/10.1063/1.345765
# https://eksmaoptics.com/nonlinear-and-laser-crystals/nonlinear-crystals/lithium-triborate-lbo-crystals/
pᵪ₂_LiB₃O₅ = (
	d₃₁ =   1.05,			  	#   pm/V
	d₃₂ =   -0.98,			  	#   pm/V
	d₃₃ =   0.05,				#   pm/V
	# d₂₄ =   2.52*d₃₆_KDP,	  	#   pm/V
	# d₁₅	=   -2.30*d₃₆_KDP,	  	#   pm/V
	λs  =  [1.079, 1.079, 1.079/2.0]
)

function make_LiB₃O₅(;p_n₁²=p_n₁²_LiB₃O₅, p_n₂²=p_n₂²_LiB₃O₅, p_n₃²=p_n₃²_LiB₃O₅,pᵪ₂=pᵪ₂_LiB₃O₅)
	@variables ω, λ, T, λs[1:3]
	n₁² = n²_LiB₃O₅_sym_ω(ω, T; p_n₁²...)
	n₂² = n²_LiB₃O₅_sym_ω(ω, T; p_n₂²...)
	n₃² = n²_LiB₃O₅_sym_ω(ω, T; p_n₃²...)
	ε 	= diagm([n₁², n₂², n₃²])
	n₁²_λ = n²_LiB₃O₅_sym(λ, T; p_n₁²...)
	n₂²_λ = n²_LiB₃O₅_sym(λ, T; p_n₂²...)
	n₃²_λ = n²_LiB₃O₅_sym(λ, T; p_n₃²...)
	ε_λ	= diagm([n₁²_λ, n₂²_λ, n₃²_λ])
	d₃₁, d₃₂, d₃₃, λᵣs = pᵪ₂
	χ⁽²⁾ᵣ = 2*cat(
		[ 	0.0	 	0.0 	d₃₁			#	xxx, xxy and xxz
		 	0.0		0.0 	0.0			#	xyx, xyy and xyz
			d₃₁	 	0.0		0.0		],	#	xzx, xzy and xzz
		[ 	0.0		0.0 	0.0			#	yxx, yxy and yxz
			0.0	 	0.0 	d₃₂			#	yyx, yyy and yyz
			0.0	 	d₃₂		0.0		],	#	yzx, yzy and yzz
		[ 	d₃₁	 	0.0 	0.0			#	zxx, zxy and zxz
			0.0	 	d₃₂ 	0.0			#	zyx, zyy and zyz
			0.0	 	0.0 	d₃₃		],	#	zzx, zzy and zzz
		 dims = 3
	)
	n₁ = sqrt(n₁²_λ)
	ng₁ = ng_model(n₁,λ)
	gvd₁ = gvd_model(n₁,λ)
	n₂ = sqrt(n₂²_λ)
	ng₂ = ng_model(n₂,λ)
	gvd₂ = gvd_model(n₂,λ)
	n₃ = sqrt(n₃²_λ)
	ng₃ = ng_model(n₃,λ)
	gvd₃ = gvd_model(n₃,λ)
	models = Dict([
		:n₁		=>	n₁,
		:ng₁	=>	ng₁,
		:gvd₁	=>	gvd₁,
		:n₂		=>	n₂,
		:ng₂	=>	ng₂,
		:gvd₂	=>	gvd₂,
		:n₃		=>	n₃,
		:ng₃	=>	ng₃,
		:gvd₃	=>	gvd₃,
		:ng		=>	diagm([	ng₁,  ng₂,  ng₃,	]),
		:gvd	=>	diagm([	gvd₁, gvd₂, gvd₃,	]),
		:ε 		=> 	ε,
		:χ⁽²⁾	=>	SArray{Tuple{3,3,3}}(Δₘ(λs,ε_λ, λᵣs, χ⁽²⁾ᵣ)),
	])
	defaults =	Dict([
		:λ		=>		0.8,		# μm
		:T		=>		p_n₁².T₀,	# °C
		:λs₁	=>		1.064,		# μm
		:λs₂	=>		1.064,		# μm
		:λs₃	=>		0.532,		# μm
	])
	Material(models, defaults, :LiB₃O₅, colorant"mediumorchid1")
end

################################################################

LiB₃O₅ = make_LiB₃O₅()

# TODO: Adapt python model below to new Julia format
# ################################################################################
# ##                         LiB3O5, Lithium Triborate (LBO)                    ##
# ################################################################################
# def n_LBO_sym(axis='Z'):
#     """This function creates a symbolic representation (using SymPy) of the
#     Sellmeier Equation model for the temperature and wavelength dependent
#     indices of refraction of LiB3O5 along all three principal axes, here labeled
#     'X', 'Y' and 'Z'.
#     Equation form is based on "Temperature dispersion of refractive indices in
#     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
#     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
#     https://doi.org/10.1063/1.360499
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#     # coefficient values for each axis
#     lbo_coeffs = [
#     [1.4426279, 1.0109932, .011210197, 1.2363218, 91, -127.70167e-6, 122.13435e-6, 53.0e-3],
#     [1.5014015, 1.0388217, .0121571, 1.7567133, 91, 373.3387e-6, -415.10435e-6, 32.7e-3],
#     [1.448924, 1.1365228, .011676746, 1.5830069, 91, -446.95031e-6, 419.33410e-6, 43.5e-3],
#     ]
#
#     if axis is 'X':
#         A,B,C,D,E,G,H,lmig = lbo_coeffs[0]
#     elif axis is 'Y':
#         A,B,C,D,E,G,H,lmig = lbo_coeffs[1]
#     elif axis is 'Z':
#         A,B,C,D,E,G,H,lmig = lbo_coeffs[2]
#     else:
#         raise Exception('unrecognized axis! must be "X","Y" or "Z"')
#     # lm = sp.symbols('lm',positive=True)
#     # T_C = T.to(u.degC).m
#     # T0_C = 20. # reference temperature in [Deg C]
#     # dT = T_C - T0_C
#     lm,T = sp.symbols('lm, T',positive=True)
#     R = lm**2 / ( lm**2 - lmig**2 ) #'normalized dispersive wavelength' for thermo-optic model
#     dnsq_dT = G * R + H * R**2
#     n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + dnsq_dT * ( T - 20. ) )
#     # n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * ( T - 20. ) )
#     # n_sym = sp.sqrt( A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * dT )
#     # n_sym =  A + B * lm**2 / ( lm**2 - C ) + D * lm**2 / ( lm**2 - E ) + ( G * ( lm**2 / ( lm**2 - lmig**2 ) ) + H * ( lm**2 / ( lm**2 - lmig**2 ) )**2 ) * dT
#     return lm,T, n_sym
#
# def n_LBO(lm_in,T_in,axis='Z'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of LiB3O5. Equation form is based on
#     "Temperature dispersion of refractive indices in
#     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
#     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
#     https://doi.org/10.1063/1.360499
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_LBO_sym(axis=axis)
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     # nsq = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output
#     # return n(lm_um)
#
# def n_g_LBO(lm_in,T_in,axis='Z'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of LiB3O5. Equation form is based on
#     "Temperature dispersion of refractive indices in
#     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
#     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
#     https://doi.org/10.1063/1.360499
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_LBO_sym(axis=axis)
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
#
# def gvd_LBO(lm_in,T_in,axis='Z'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of LiB3O5. Equation form is based on
#     "Temperature dispersion of refractive indices in
#     β‐BaB2O4 and LiB3O5 crystals for nonlinear optical devices"
#     by Ghosh et al., Journal of Applied Physics 78, 6752 (1995)
#     https://doi.org/10.1063/1.360499
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_LBO_sym(axis=axis)
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
#
#
