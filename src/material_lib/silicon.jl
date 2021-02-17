# TODO: Adapt python model below to new Julia format
# ########### Silicon Sellmeier model for waveguide phase matching calculations
#
# def n_silicon_sym():
#     """This function creates a symbolic representation (using SymPy) of the
#     Sellmeier Equation model for the index of refraction.
#     Equation form is based on Osgood, Panoiu, Dadap, et al.
#     "Engineering nonlinearities in nanoscale optical systems: physics and
# 	applications in dispersion-engineered silicon nanophotonic wires"
# 	Advances in Optics and Photonics Vol. 1, Issue 1, pp. 162-235 (2009)
#     https://doi.org/10.1364/AOP.1.000162
#     valid from 1.2-?? um
#     Thermo-optic coefficients from
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#     eps = 11.6858
#     A = 0.939816 # um^2
#     B = 8.10461e-3
#     lm1 = 1.1071 # um^2
#     dn_dT = 0.0e-5 # 1/degK
#     lm, T = sp.symbols('lm T')
#     T0 = 24.5 # reference temperature in [Deg C]
#     n_sym = sp.sqrt( eps  +  A / lm**2 + ( B * lm**2 ) / ( lm**2 - lm1 ) ) # + dn_dT * ( T - T0 )
#     # n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
#     return lm, T, n_sym
#
# def n_silicon(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of silicon. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_silicon_sym()
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output
#
# def n_g_silicon(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of silicon. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_silicon_sym()
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
#
# def gvd_silicon(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of silicon. Equation form is based on
#     "Broadband mid-infrared frequency comb generation in a silicon microresonator"
#     by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
#     Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_silicon_sym()
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
