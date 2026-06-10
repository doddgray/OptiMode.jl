# TODO: Adapt python model below to new Julia format
# ################################################################################
# ##                          crystalline MgF2                                  ##
# ################################################################################
# def n_MgF2_sym(axis='e'):
#     """This function creates a symbolic representation (using SymPy) of the
#     Sellmeier Equation model for the wavelength dependence
#     of crystalline MgF2's ordinary and extraordinary indices of refraction as
#     well as the index of thin-film amorphous MgF2.
#     Sellmeier coefficients for crystalline MgF2 are taken from
#     "Refractive properties of magnesium fluoride"
#     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
#     https://doi.org/10.1364/AO.23.001980
#     Sellmeier coefficients for amorphous MgF2 are taken from
#     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
#     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
#     https://doi.org/10.1364/OME.7.000989
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#     if axis is 'e':
#         coeffs = [  (0.41344023,0.03684262), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
#                     (0.50497499,0.09076162),
#                     (2.4904862,23.771995),
#                     ]
#     elif axis is 'o':
#         coeffs = [  (0.48755108,0.04338408), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
#                     (0.39875031,0.09461442),
#                     (2.3120353,23.793604),
#                     ]
#     elif axis is 'a':
#         coeffs = [  (1.73,.0805), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
#                     ]
#     else:
#         raise Exception('unrecognized axis! must be "e", "o" or "a"')
#     lm, T = sp.symbols('lm T')
#     A0,λ0 = coeffs[0]
#     oscillators = A0 * lm**2 / (lm**2 - λ0**2)
#     if len(coeffs)>1:
#         for Ai,λi in coeffs[1:]:
#             oscillators += Ai * lm**2 / (lm**2 - λi**2)
#     n_sym = sp.sqrt( 1 + oscillators )
#     return lm, T, n_sym
#
# def n_MgF2(lm_in,T_in=300*u.degK,axis='e'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') MgF2.
#     Sellmeier coefficients for crystalline MgF2 are taken from
#     "Refractive properties of magnesium fluoride"
#     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
#     https://doi.org/10.1364/AO.23.001980
#     Sellmeier coefficients for amorphous MgF2 are taken from
#     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
#     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
#     https://doi.org/10.1364/OME.7.000989
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_MgF2_sym(axis=axis)
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output.squeeze()
#
# def n_g_MgF2(lm_in,T_in=300*u.degK,axis='e'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') MgF2.
#     Sellmeier coefficients for crystalline MgF2 are taken from
#     "Refractive properties of magnesium fluoride"
#     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
#     https://doi.org/10.1364/AO.23.001980
#     Sellmeier coefficients for amorphous MgF2 are taken from
#     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
#     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
#     https://doi.org/10.1364/OME.7.000989
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_MgF2_sym(axis=axis)
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
# def gvd_MgF2(lm_in,T_in=300*u.degK,axis='e'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') MgF2.
#     Sellmeier coefficients for crystalline MgF2 are taken from
#     "Refractive properties of magnesium fluoride"
#     by Dodge, Applied Optics 23 (12), pp.1980-1985 (1984)
#     https://doi.org/10.1364/AO.23.001980
#     Sellmeier coefficients for amorphous MgF2 are taken from
#     "Self-consistent optical constants of MgF2, LaF3, and CeF3 films"
#     by Rodríguez-de Marcos, et al. Optical Materials Express 7 (3) (2017)
#     https://doi.org/10.1364/OME.7.000989
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_MgF2_sym(axis=axis)
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
#
#
#
