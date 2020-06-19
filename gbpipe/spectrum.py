import numpy as np
import healpy as hp
import camb

from .utils import cl2dl, dl2cl, print_warning

try:
    import xpol
except ModuleNotFoundError:
    print ('xpol is not found. Using camb instead.')
    xpol = camb

args_cosmology = ['H0', 'cosmomc_theta', 'ombh2', 'omch2', 'omk', 
                  'neutrino_hierarchy', 'num_massive_neutrinos',
                  'mnu', 'nnu', 'YHe', 'meffsterile', 'standard_neutrino_neff', 
                  'TCMB', 'tau', 'deltazrei', 'bbnpredictor', 'theta_H0_range'] 

args_InitPower = ['As', 'ns', 'nrun', 'nrunrun', 'r', 'nt', 'ntrun', 'pivot_scalar', 
                  'pivot_tensor', 'parameterization']


def arg2dict(**kwargs):
    ## arguments to dictionaries
    kwargs_cosmology={}
    kwargs_InitPower={}

    for key, value in kwargs.items():  # for Python 3, items() instead of iteritems()
        if key in args_cosmology: 
            kwargs_cosmology[key]=value
        elif key in args_InitPower:
            kwargs_InitPower[key]=value
        else:
            print_warning('Wrong keyword: ' + key)

    return kwargs_cosmology, kwargs_InitPower


def get_spectrum_camb(lmax, 
                      isDl=False, cambres=False, TTonly=False, unlensed=False, CMB_unit=None, 
                      inifile=None,
                      **kwargs):
    """
    """
   
    kwargs_cosmology, kwargs_InitPower = arg2dict(**kwargs)


    ## call camb
    if inifile is None:
        pars = camb.CAMBparams()
    else:
        pars = camb.read_ini(inifile)

    kwargs_cosmology['H0'] = pars.H0
    pars.set_cosmology(**kwargs_cosmology)
    pars.InitPower.set_params(**kwargs_InitPower)
    pars.WantTensors = True

    results = camb.get_results(pars)

    raw_cl = np.logical_not(isDl)
    if (TTonly):
        if unlensed:
            cls = results.get_unlensed_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T[0]
        else:
            cls = results.get_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T[0]
    else: 
        if unlensed:
            cls = results.get_unlensed_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T
        else:
            cls = results.get_total_cls(lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl).T

    if (cambres):
        return cls, results
    else:
        return cls 
    

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


def get_spectrum_noise(lmax, wp, fwhm=None, isDl=False, TTonly=False, CMB_unit=None):
    cls = np.array([(np.pi/10800 * wp * 1e-6) ** 2]*(lmax+1)) # wp is w_p^(-0.5) in uK arcmin
    cls[0] = cls[1] = 0

    if (CMB_unit == 'muK'):
        cls *= 1e12

    if fwhm:
        ell = np.arange(lmax+1)
        cls *= np.exp(ell**2 * fwhm * (np.pi/180)**2 / 8 / np.log(2))

    if (not TTonly):
        cls = np.array(cls, cls*2, cls*2, cls*0) #+ [np.zeros(cls.shape)])

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


