import numpy as np
from scipy.interpolate import interp1d

def calc_supercolor_mags(r, i, z, redshift, rerr = None, ierr = None, zerr = None,  param_dict = {}, calc_error = True):
    # Calculates 'supercolor' magnitudes, which are a parameterization
    # of r, i, and z observer-frame magnitudes, as well as redshift.
    # The parameterization is fit to approximate rest-frame magnitudes
    if np.shape(param_dict.keys()) == (0,): param_dict = set_default_param_dict()
    if 'Acomb' in param_dict.keys():
        F_Acomb = interp1d(param_dict['z'], param_dict['Acomb'], kind = 'linear', bounds_error = False, fill_value = 0)
        A = F_Acomb(redshift)
    else:
        F_ARed = interp1d(param_dict['z'], param_dict['ARed'], kind = 'linear', bounds_error = False, fill_value = 0)
        ARed = F_ARed(redshift)
        F_ABlue = interp1d(param_dict['z'], param_dict['ABlue'], kind = 'linear', bounds_error = False, fill_value = 0)
        ABlue = F_ABlue(redshift)
    try:
        F_BRed = interp1d(param_dict['z'], param_dict['BRed'], kind = 'linear', bounds_error = False, fill_value = 0)
        BRed = F_BRed(redshift)
        F_BBlue = interp1d(param_dict['z'], param_dict['BBlue'], kind = 'linear', bounds_error = False, fill_value = 0)
        BBlue = F_BBlue(redshift)
    except KeyError:
        B = 1 - A
    try:
        m_b = 2.5 * np.log10(ABlue * 10**(0.4 * (r)) + BBlue * 10**(0.4 * (i)))
        m_r = 2.5 * np.log10(ARed * 10**(0.4 * (i)) + BRed * 10**(0.4 * (z)))
    except NameError:
        m_b = -2.5 * np.log10(A * 10**(-0.4 * (r)) + B * 10**(-0.4 * (i)))
        m_r = -2.5 * np.log10(A * 10**(-0.4 * (i)) + B * 10**(-0.4 * (z)))
    if calc_error:
        m_b_err, m_r_err = np.sqrt(rerr**2 + ierr**2), np.sqrt(zerr**2 + ierr**2)
        return m_b, m_r, m_b_err, m_r_err
    else:
        return m_b, m_r

def set_default_param_dict(zarr = None):
    if zarr == None: zarr = np.linspace(0.1, 2, 1000)
    ARed, ABlue = 0.424 * (1 - 1.794 * (zarr - 0.628)), 0.45 * (1 - 1.824 * (zarr - 0.679))
    BRed, BBlue = 0.576 * (1.794 * (zarr - 0.628)), 0.55 * (1.824 * (zarr - 0.679))
    ARed[zarr > 1/1.794 + 0.628] = 0
    BRed[zarr < 0.628] = 0
    ABlue[zarr > 1/1.824 + 0.679] = 0
    BBlue[zarr < 0.679] = 0
    return {'ARed':ARed, 'ABlue':ABlue, 'BRed':BRed, 'BBlue':BBlue, 'z':zarr}
