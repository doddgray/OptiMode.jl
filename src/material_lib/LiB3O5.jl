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
