# TODO: Adapt python model below to new Julia format
# ################################################################################
# ##                                  HfO2                                      ##
# ################################################################################
#
# def n_HfO2_sym(axis='a'):
#     """This function creates a symbolic representation (using SymPy) of the
#     Sellmeier Equation model for the wavelength dependence
#     of crystalline HfO2's ordinary and extraordinary indices of refraction as
#     well as the index of thin-film amorphous HfO2.
#     Sellmeier coefficients for crystalline HfO2 are taken from
#     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
#     films taken from
#     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
#     Surface and Coatings Technology 201.6 (2006)
#     https://doi.org/10.1016/j.surfcoat.2006.08.074
#     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
#     for ALD Hafnia specifically. They also report loss, with a sharp absorption
#     edge near 5.68 ± 0.09 eV (~218 nm)
#     Eqn. form:
#     n = A + B / lam**2 + C / lam**4
#     ng = A + 3 * B / lam**2 + 5 * C / lam**4
#     This model is then exported to other functions that use it and its
#     derivatives to return index, group index and GVD values as a function
#     of temperature and wavelength.
#     Variable units are lm in [um] and T in [deg C]
#     """
#     # if axis is 'e':
#     #     coeffs = [  (0.41344023,0.03684262), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
#     #                 (0.50497499,0.09076162),
#     #                 (2.4904862,23.771995),
#     #                 ]
#     # elif axis is 'o':
#     #     coeffs = [  (0.48755108,0.04338408), # A_i,λ_i Lorentz oscillator strength [1] and resonance wavelength [μm]
#     #                 (0.39875031,0.09461442),
#     #                 (2.3120353,23.793604),
#     #                 ]
#     if axis is 'a':
#         lm, T = sp.symbols('lm T')
#         # # fit for spectroscopic ellipsometer measurement for a 250nm thick film, good 300-1400nm
#         # A = 1.85
#         # B = 1.17e-2 # note values here are for λ in μm, vs nm in the paper
#         # C = 0.0
#         # fit for spectroscopic ellipsometer measurement for a 112nm thick film, good 300-1400nm
#         A = 1.86
#         B = 7.16e-3 # note values here are for λ in μm, vs nm in the paper
#         C = 0.0
#         n_sym =  A + B / lm**2 + C / lm**4
#         return lm, T, n_sym
#     if axis is 's':
#         lm, T = sp.symbols('lm T')
#         A = 1.59e4 # μm^-2
#         λ0 = 210.16e-3 # μm, note values here are for λ in μm, vs nm in the paper
#         n_sym = sp.sqrt(   1  + ( A * lm**2 ) / ( lm**2/λ0**2 - 1 ) )
#         return lm, T, n_sym
#     else:
#         raise Exception('unrecognized axis! must be "e", "o" or "a"')
#
#
# def n_HfO2(lm_in,T_in=300*u.degK,axis='a'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the index of refraction of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') HfO2.
#     Sellmeier coefficients for crystalline HfO2 are taken from
#     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
#     films taken from
#     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
#     Surface and Coatings Technology 201.6 (2006)
#     https://doi.org/10.1016/j.surfcoat.2006.08.074
#     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
#     for ALD Hafnia specifically. They also report loss, with a sharp absorption
#     edge near 5.68 ± 0.09 eV (~218 nm)
#     Eqn. form:
#     n = A + B / lam**2 + C / lam**4
#     ng = A + 3 * B / lam**2 + 5 * C / lam**4
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
#     T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
#     lm, T, n_sym = n_HfO2_sym(axis=axis)
#     n = sp.lambdify([lm,T],n_sym,'numpy')
#     output = np.zeros((T_C.size,lm_um.size))
#     for T_idx, TT in enumerate(T_C):
#         output[T_idx,:] = n(lm_um, T_C[T_idx])
#     return output.squeeze()
#
# def n_g_HfO2(lm_in,T_in=300*u.degK,axis='a'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group index of refraction of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') HfO2.
#     Sellmeier coefficients for crystalline HfO2 are taken from
#     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
#     films taken from
#     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
#     Surface and Coatings Technology 201.6 (2006)
#     https://doi.org/10.1016/j.surfcoat.2006.08.074
#     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
#     for ALD Hafnia specifically. They also report loss, with a sharp absorption
#     edge near 5.68 ± 0.09 eV (~218 nm)
#     Eqn. form:
#     n = A + B / lam**2 + C / lam**4
#     ng = A + 3 * B / lam**2 + 5 * C / lam**4
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_HfO2_sym(axis=axis)
#     n_sym_prime = sp.diff(n_sym,lm)
#     n_g_sym = n_sym - lm*n_sym_prime
#     n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
#     return n_g(lm_um, T_C)
#
# def gvd_HfO2(lm_in,T_in=300*u.degK,axis='a'):
#     """Sellmeier Equation model for the temperature and wavelength dependence
#     of the group velocity dispersion of crystalline (axis='o' or 'e') and
#     amorphous (axis='a') HfO2.
#     Sellmeier coefficients for crystalline HfO2 are taken from
#     Cauchy Equation coefficients for amorphous ~100nm sputtered Hafnia (HfO2)
#     films taken from
#     Khoshman and Kordesch. "Optical properties of a-HfO2 thin films."
#     Surface and Coatings Technology 201.6 (2006)
#     https://doi.org/10.1016/j.surfcoat.2006.08.074
#     Fits measured data from sputtered amorphous Hafnia, I haven't found a fit
#     for ALD Hafnia specifically. They also report loss, with a sharp absorption
#     edge near 5.68 ± 0.09 eV (~218 nm)
#     Eqn. form:
#     n = A + B / lam**2 + C / lam**4
#     ng = A + 3 * B / lam**2 + 5 * C / lam**4
#     Variable units are lm in [um] and T in [deg C]
#     """
#     lm_um = Q_(lm_in).to(u.um).magnitude
#     T_C = Q_(T_in).to(u.degC).magnitude
#     lm, T, n_sym = n_HfO2_sym(axis=axis)
#     n_sym_double_prime = sp.diff(n_sym,lm,lm)
#     c = Q_(3e8,'m/s') # unitful definition of speed of light
#     gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
#     gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
#     gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
#     return gvd.to('fs**2 / mm')
