################################################################################
#                                    αAl₂O₃                                    #
################################################################################
export αAl₂O₃

"""
This code creates a symbolic representation of a Cauchy Equation model
for the index of refraction of amorphous alumina (αAl₂O₃).
The model fit is due to:
    West, et al. "Low-loss integrated photonics for the blue and ultraviolet regime."
    APL Photonics 4.2 (2019): 026101. [doi:10.1063/1.5052502](https://doi.org/10.1063/1.5052502)
and has been validated from 0.25-.75 μm (free space) wavelength range for films grown by
atomic layer deposition (original work) and chemical vapor deposition.
The symbolic index model and its derivatives in turn are used to generate
numerical functions for the αAl₂O₃ index, group index and GVD as a function
of temperature and wavelength.
Variable units are λ in [um] and T in [deg C]
"""

p_n_αAl₂O₃ = (
    A = 1.602,              #                           [1]
    B = 0.01193,            #                           [μm²]
    C = -0.00036,           #                           [μm⁴]
    dn_dT = 6.1e-6,         # thermo-optic coefficient  [K⁻¹]
    T₀ = 20,                # reference temperature     [°C]
)

# notes & coeffs.from original working python code for reference:
### Cauchy Equation fit coefficients for Gavin's ALD alumina films ###
# Eqn. form:
# n = A + B / lam**2 + C / lam**4
# ng = A + 3 * B / lam**2 + 5 * C / lam**4
# A_alumina = 1.602
# B_alumina = 0.01193
# C_alumina = -0.00036

function make_αAl₂O₃(;p_n=p_n_αAl₂O₃)
	@variables λ, T
	n = n_sym_cauchy( λ ; p_n...) + p_n.dn_dT  *  ( T - p_n.T₀  )
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
		:T		=>		p_n.T₀,		# °C

	])
	Material(models, defaults, :αAl₂O₃)
end

################################################################

αAl₂O₃ = make_αAl₂O₃()
