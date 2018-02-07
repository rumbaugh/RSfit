import numpy as np
from scipy.interpolate import interp1d

def calc_supercolor_mags(r, i, z, redshift, rerr=None, ierr=None, zerr=None,  param_dict={}, calc_error=True):
    # Calculates 'supercolor' magnitudes, which are a parameterization
    # of r, i, and z observer-frame magnitudes, as well as redshift.
    # The parameterization is fit to approximate rest-frame magnitudes
    if np.shape(param_dict.keys()) == (0,): 
        param_dict = set_default_param_dict()
    if 'Acomb' in param_dict.keys():
        F_Acomb = interp1d(param_dict['z'], param_dict['Acomb'], kind='linear', bounds_error=False, fill_value=0)
        A = F_Acomb(redshift)
    else:
        F_Ared = interp1d(param_dict['z'], param_dict['Ared'], kind='linear', bounds_error=False, fill_value=0)
        Ared = F_Ared(redshift)
        F_Ablue = interp1d(param_dict['z'], param_dict['Ablue'], kind='linear', bounds_error=False, fill_value=0)
        Ablue = F_Ablue(redshift)
    try:
        F_Bred = interp1d(param_dict['z'], param_dict['Bred'], kind='linear', bounds_error=False, fill_value=0)
        Bred = F_Bred(redshift)
        F_Bblue = interp1d(param_dict['z'], param_dict['Bblue'], kind='linear', bounds_error=False, fill_value=0)
        Bblue = F_Bblue(redshift)
    except KeyError:
        B = 1 - A
    try:
        m_b = 2.5 * np.log10(Ablue * 10**(0.4 * (r)) + Bblue * 10**(0.4 * (i)))
        m_r = 2.5 * np.log10(Ared * 10**(0.4 * (i)) + Bred * 10**(0.4 * (z)))
    except NameError:
        m_b = -2.5 * np.log10(A * 10**(-0.4 * (r)) + B * 10**(-0.4 * (i)))
        m_r = -2.5 * np.log10(A * 10**(-0.4 * (i)) + B * 10**(-0.4 * (z)))
    if calc_error:
        m_b_err, m_r_err = np.sqrt(rerr**2 + ierr**2), np.sqrt(zerr**2 + ierr**2)
        return m_b, m_r, m_b_err, m_r_err
    else:
        return m_b, m_r

def set_default_param_dict(zarr=None):
    if zarr == None: 
        zarr = np.linspace(0.1, 2, 1000)
    Ared, Ablue = 0.424 * (1 - 1.794 * (zarr - 0.628)), 0.45 * (1 - 1.824 * (zarr - 0.679))
    Bred, Bblue = 0.576 * (1.794 * (zarr - 0.628)), 0.55 * (1.824 * (zarr - 0.679))
    Ared[zarr > 1/1.794 + 0.628] = 0
    Bred[zarr < 0.628] = 0
    Ablue[zarr > 1/1.824 + 0.679] = 0
    Bblue[zarr < 0.679] = 0
    return {'Ared': Ared, 'Ablue': Ablue, 'Bred': Bred, 'Bblue': Bblue, 'z': zarr}
