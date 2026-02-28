# TODO: Adapt python model below to new Julia format
# ################################################################################
# ##                         Gallium Arsenide (GaAs)                            ##
# ################################################################################
#
# def n_GaAs_sym():
#     """This function creates a symbolic representation (using SymPy) of the
#     a model for the temperature and wavelength dependence of GaAs's refractive
#     index. The equation form and fit parameters are based on "Improved dispersion
#     relations for GaAs and applications to nonlinear optics" by Skauli et al.,
#     JAP 94, 6447 (2003); doi: 10.1063/1.1621740
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#
#     lm, T = sp.symbols('lm T')
#     T0 = 22 # reference temperature in [Deg C]
#     deltaT = T-T0
#     A = 0.689578
#     eps2 = 12.99386
#     G3 = 2.18176e-3
#     E0 = 1.425 - 3.7164e-4 * deltaT - 7.497e-7* deltaT**2
#     E1 = 2.400356 - 5.1458e-4 * deltaT
#     E2 = 7.691979 - 4.6545e-4 * deltaT
#     E3 = 3.4303e-2 + 1.136e-5 * deltaT
#     E_phot = (u.h * u.c).to(u.eV*u.um).magnitude / lm  #
#     n_sym = sp.sqrt( 1  + (A/np.pi)*sp.log((E1**2 - E_phot**2) / (E0**2-E_phot**2)) \
#                         + (eps2/np.pi)*sp.log((E2**2 - E_phot**2) / (E1**2-E_phot**2)) \
#                         + G3 / (E3**2-E_phot**2) )
#
#     return lm, T, n_sym
#
# ###### Temperature Dependent Sellmeier Equation for phase-matching calculations
# def n_GaAs(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of 5% MgO:LiNbO3. Equation form is based on
#     "Temperature and wavelength dependent refractive index equations
#     for MgO-doped congruent and stoichiometric LiNbO3"
#     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
#     Variable units are lm in [um] and T in [deg C]
#     """
# #    lm_um = Q_(lm_in).to(u.um).magnitude
# #    T_C = Q_(T_in).to(u.degC).magnitude
# #    lm, T, n_sym = n_GaAs_sym()
# #    n = sp.lambdify([lm,T],n_sym,'numpy')
# #    return n(lm_um, T_C)
#
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_GaAs_sym()
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output
#
#
# def n_g_GaAs(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of 5% MgO:LiNbO3. Equation form is based on
#     "Temperature and wavelength dependent refractive index equations
#     for MgO-doped congruent and stoichiometric LiNbO3"
#     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_GaAs_sym()
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
#
# def gvd_GaAs(lm_in,T_in):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of 5% MgO:LiNbO3. Equation form is based on
#     "Temperature and wavelength dependent refractive index equations
#     for MgO-doped congruent and stoichiometric LiNbO3"
#     by Gayer et al., Applied Physics B 91, p.343-348 (2008)
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_GaAs_sym()
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
