import numpy as np
import healpy as hp
import camb
import xpol
from .utils import dl2cl, print_warning


args_cosmology = ['H0', 'cosmomc_theta', 'ombh2', 'omch2', 'omk', 
                  'neutrino_hierarchy', 'num_massive_nutrinos',
                  'mnu', 'nnu', 'YHe', 'meffsterile', 'standard_neutrino_neff', 
                  'TCMB', 'tau', 'deltazrei', 'bbnpredictor', 'theta_H0_range'] 

args_InitPower = ['As', 'ns', 'nrun', 'nrunrun', 'r', 'nt', 'ntrun', 'pivot_scalar', 
                  'pivot_tensor', 'parameterization']


def get_spectrum_camb(lmax, 
                      isDl=True, cambres=False, TTonly=False, unlensed=False, CMB_unit=None, 
                      **kwargs):
    """
    """
   
    ## arguments to dictionaries
    kwargs_cosmology={}
    kwargs_InitPower={}
    wantTensor = False

    for key, value in kwargs.items():  # for Python 3, items() instead of iteritems()
        if key in args_cosmology: 
            kwargs_cosmology[key]=value
            if key == 'r':
                wantTensor = True
        elif key in args_InitPower:
            kwargs_InitPower[key]=value
        else:
            print_warning('Wrong keyword: ' + key)

    ## for camb > 1.0
    if not ('H0' in kwargs_cosmology.keys()):
        kwargs_cosmology['H0'] = 67.5

    ## call camb
    pars = camb.CAMBparams()
    pars.set_cosmology(**kwargs_cosmology)
    pars.InitPower.set_params(**kwargs_InitPower)
    pars.WantTensors = True
    results = camb.get_results(pars)

    if (TTonly):
        if unlensed:
            dls = results.get_unlensed_total_cls(lmax=lmax, CMB_unit=CMB_unit).T[0]
        else:
            dls = results.get_total_cls(lmax=lmax, CMB_unit=CMB_unit).T[0]
    else: 
        if unlensed:
            dls = results.get_unlensed_total_cls(lmax=lmax, CMB_unit=CMB_unit).T
        else:
            dls = results.get_total_cls(lmax=lmax, CMB_unit=CMB_unit).T

    #dls = dls * pars.TCMB**2

    if (isDl):
        res = dls
    else:
        cls = dl2cl(dls)
        res = cls

    if (cambres):
        return res, results
    else:
        return res
    

def get_spectrum_const(lmax, isDl=True):
    """
    """
    dls = np.zeros(lmax+1)+1      
    dls[0] = 0
    dls[1] = 0 

    if (isDl):
        res = dls  
    else:
        cls = dl2cl(dls)
        res = cls

    return res


def get_spectrum_map(mapT, lmax=2000, isDL=False): 
    """
    """
    cls = hp.anafast(mapT, lmax=lmax)
    
    if (isDL):
        ell = np.arange(len(cls)) 
        dls = cls * ell * (ell+1) / 2 / np.pi
        return dls
    else:
        return cls


def get_spectrum_noise(lmax, wp, fwhm=None, isDl=True, TTonly=False, CMB_unit=None):
    cls = np.array([(np.pi/10800 * wp * 1e-6) ** 2]*(lmax+1)) # wp is w_p^(-0.5) in uK arcmin
    cls[0] = cls[1] = 0

    if (CMB_unit == 'muK'):
        cls *= 1e12

    if (not TTonly):
        cls = np.array([cls]*3 + [np.zeros(cls.shape)])

    if (isDl):
        res = cl2dl(cls)
    else:
        res = cls

    return res


def get_spectrum_xpol(map_in, lmax, mask):
    blow = np.arange(lmax)
    bupp = blow + 1
    bins = xpol.Bins(blow, bupp)
    xp = xpol.Xpol(mask, bins)

    cl_biased, cl_unbiased_tmp = xp.get_spectra(map_in)

    cl_unbiased = np.zeros(cl_biased.shape)
    cl_unbiased[:, 2:] = cl_unbiased_tmp
    
    return cl_biased, cl_unbiased


