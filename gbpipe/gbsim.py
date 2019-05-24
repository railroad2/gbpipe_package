import os
import sys
import time
import socket
import datetime

import numpy as np
import healpy as hp
import pylab as plt

import paramiko
import multiprocessing as mp
from scipy.interpolate import interp1d

import astropy
from astropy.time import Time, TimeDelta
from astropy.io import fits
from astropy.utils import iers

DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)
sys.path.append(DIRNAME+'../')

from . import gbdir
from .utils import dl2cl, cl2dl
from .gbparam import GBparam
from .utils import set_logger, function_name, today, qu2ippsi


def sim_noise1f(l, wnl, fknee, fsample=1000, alpha=1, rseed=0, return_psd=False):
    """ Generates noise tod which has power spectrum of 
    s(f) = (wnl**2/NFFT)*(1 + (fknee/f)**alpha)

    Parameters
    ----------
    l : int
        Data length 
    wnl : float
        White noise level (NET: Noise Equivalent Temperature)
    fknee : float
        Knee frequency.
    fsample : float
        Sampling frequency.
        Default is 1000 sps(sample per second).
    alpha : float
        Exponent of 1/f noise window.
        Default is 1.
    rseed : int
        Random seed for white noise generation.

    Returns
    -------
    n_1f : float array
        1/f noise. Real part of inverse fft of the spectrum.
    freq : float array
        frequency of psd (optional)
    s_1f : float array
        analytic power spectral density (optional)
         
    """ 
    log = set_logger(function_name())
    
    t = np.arange(l) * 1./fsample
    np.random.seed(rseed)
    n0 = np.random.normal(scale=wnl, size=l)

    s0 = np.fft.fft(n0)/l
    freq = np.fft.fftfreq(l, d=1./fsample)
    freq[0] = freq[1]/100

    s = 1 + (fknee/abs(freq))**alpha
    s[0] = s[1]

    s_1f = s0 * s

    n_1f = np.fft.ifft(s_1f).real

    if return_psd:
        res = [n_1f, (freq, s_1f)] 
    else:
        res = n_1f

    return res


def sim_obs_singlepix(t1, t2, fsample=1000): 
    """ Simulation module for an observation with a single pixel. 
    
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT
    fsample : float
        Sampling frequency in sps.
        Default is 1000.
    
    Returns
    -------
    m_nhit : float array
        N-hit map for the observation
    """
    
    par = GBparam()
    log = set_logger(function_name())

    ##################################
    # define local sidereal time
    ##################################
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    #ut_arr = np.arange(st.unix, et.unix, 1./fsample)
    #ut = Time(ut_arr, format='unix') 
    #lst_ut = ut.sidereal_time('apparent', par.lon).deg # takes ~ 2500 s for 6 hr observation

    ut = st.unix + np.arange(0, int(et.unix-st.unix), 1./fsample)
    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    f = interp1d(ut_1s, lst_1s, fill_value="extrapolate")
    lst = f(ut)

    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * par.omega_gb) % 360


    ##########################################
    # get Rotation matrix & rotate z vector
    ##########################################

    v = (0, 0, 1)
    rmat = gbdir.Rot_matrix(az=az, lst=lst)
    v_obs = gbdir.Rotate(v_arr=v, rmat=rmat) 

    #########################################
    # N-hit map
    #########################################

    nside = 256
    npix = hp.nside2npix(nside)
    m_nhit = np.full(npix, hp.UNSEEN)
    x = v_obs.T[0]
    y = v_obs.T[1]
    z = v_obs.T[2]
    pix_obs = hp.vec2pix(nside, x, y, z)
   
    px, cnt = np.unique(pix_obs, return_counts=True)    
    m_nhit[px] = cnt
    
    return m_nhit


def sim_obs_focalplane(t1, t2, fsample=1000): 
    """ Simulation module for an observation with a focal plane.
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT.
    fample : float
        Sampling frequency in sps.
        Default is 1000.
    
    Returns
    -------
    m_nhit145 : float array
        N-hit map for 145 GHz modules.
    m_nhit220 : float array
        N-hit map for 220 GHz modules.
    """
    par = GBparam()
    log = set_logger(function_name())

    t0 = time.time()
    ##################################
    # define local sidereal time
    ##################################
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    #ut_arr = np.arange(st.unix, et.unix, 1./fsample)
    #ut = Time(ut_arr, format='unix') 
    #lst_ut = ut.sidereal_time('apparent', par.lon).deg # takes ~ 2500 s for 6 hr observation

    ## assuming that the Earth rotation is constant within a second.
    #ut1 = Time(np.arange(st.unix, et.unix+1, 1.), format='unix')
    #ut = [ut1.unix[i] + np.arange(fsample)*(ut1.unix[i+1]-ut1.unix[i])/fsample for i in range(len(ut1)-1)]
    #ut = np.array(ut).flatten()
    #lst1 = ut1.sidereal_time('apparent', par.lon).deg
    #lst = [lst1[i] + np.arange(fsample)*(lst1[i+1]-lst1[i])/fsample for i in range(len(lst1)-1)]
    #lst = np.array(lst).flatten()

    ut = st.unix + np.arange(0, int(et.unix-st.unix), 1./fsample)
    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    f = interp1d(ut_1s, lst_1s, fill_value="extrapolate")
    lst = f(ut)

    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * par.omega_gb) % 360
    log.debug('calculating time for time stamp and lst:', time.time() - t0)


    #######################################
    # get Rotation matrix & rotate focalplane
    #######################################

    t0 = time.time()

    theta = par.pixinfo['theta']
    phi =par.pixinfo['phi']
    v = hp.ang2vec(np.radians(theta), np.radians(phi))
    rmat = gbdir.Rot_matrix(az=az, lst=lst)
    v_obs = gbdir.Rotate(v_arr=v, rmat=rmat) 
    v_obs = np.transpose(v_obs, (2,1,0))

    log.debug('time for rotation:', time.time() - t0)



    #########################################
    # N-hit maps
    #########################################

    nside = 512
    npix = hp.nside2npix(nside)
    nobs = len(v_obs)
    m_nhit145 = np.full(npix, hp.UNSEEN)
    m_nhit220 = np.full(npix, hp.UNSEEN)

    t0 = time.time()
    pix_220 = []
    pix_145 = []
    log.debug (v_obs.shape)
    for vi in v_obs:
        x = vi[0]
        y = vi[1]
        z = vi[2]
        pix_obs = hp.vec2pix(nside, x, y, z)
        pix_220.append(pix_obs[:121])
        pix_145.append(pix_obs[121:])

    pix_145 = np.array(pix_145).flatten()
    pix_220 = np.array(pix_220).flatten()

    px1, cnt1 = np.unique(pix_145, return_counts=True)
    px2, cnt2 = np.unique(pix_220, return_counts=True)

    m_nhit145[px1] = cnt1
    m_nhit220[px2] = cnt2

    log.debug ('time to make N-hit map: ', time.time() - t0)
     
    hp.mollview(m_nhit145)
    plt.savefig('nhit145_1min.png')

    hp.mollview(m_nhit220)
    plt.savefig('nhit220_1min.png')

    return m_nhit145, m_nhit220


def sim_tod_singlepix(t1, t2, fsample=1000, map_in=None, rseed=42):
    """ Simulation module for an observation with a single pixel. 
    
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT.
    fsample : float
        Sampling frequency in sps.
        Default is 1000.
    map_in : float array
        Input map for tod simulation.
        If None, it synthesizes a map by using rseed as a random seed. 
        Default is None.
    rseed : int
        Random seed that is used to synthesize the source map.
    
    Returns
    -------
    tod_I : float array
        Simulated tod I.
    tod_Q : float array
        Simulated tod I.
    tod_U : float array
        Simulated tod I
    """
    log = set_logger(function_name())
    if map_in is None:
        ## synthesize TQU map 
        log.info('Synthesizing a map')
        import camb
        par_camb = camb.CAMBparams()
        par_camb.set_cosmology(H0=67.5)
        res = camb.get_results(par_camb)
        dls = res.get_cmb_power_spectra(par_camb, CMB_unit='K')['total']
        ell = np.arange(len(dls))
        cls = dls.copy() 
        cls[1:,0] = cls[1:,0]*2*np.pi/ell[1:]/(ell[1:]+1)
        cls[1:,1] = cls[1:,1]*2*np.pi/ell[1:]/(ell[1:]+1)
        cls[1:,2] = cls[1:,2]*2*np.pi/ell[1:]/(ell[1:]+1)
        cls[1:,3] = cls[1:,3]*2*np.pi/ell[1:]/(ell[1:]+1)
        np.random.seed(rseed)
        map_in = hp.synfast(cls=cls.T, nside=1024, new=True)

    par = GBparam()

    ##################################
    # define local sidereal time
    ##################################
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    #ut_arr = np.arange(st.unix, et.unix, 1./fsample)
    #ut = Time(ut_arr, format='unix') 
    #lst_ut = ut.sidereal_time('apparent', par.lon).deg # takes ~ 2500 s for 6 hr observation

    # assuming that the Earth rotation is constant within a second.
    """ previous version
    ut1 = Time(np.arange(st.unix, et.unix+1, 1.), format='unix')
    ut = [ut1.unix[i] + np.arange(fsample)*(ut1.unix[i+1]-ut1.unix[i])/fsample for i in range(len(ut1)-1)]
    ut = np.array(ut).flatten()
    lst1 = ut1.sidereal_time('apparent', par.lon).deg
    lst = [lst1[i] + np.arange(fsample)*(lst1[i+1]-lst1[i])/fsample for i in range(len(lst1)-1)]
    lst = np.array(lst).flatten()
    """
    log.info('Making time stamps')
    ut = st.unix + np.arange(0, int(et.unix-st.unix), 1./fsample)
    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    f = interp1d(ut_1s, lst_1s, fill_value='extrapolate')
    lst = f(ut)



    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * par.omega_gb) % 360


    ##########################################
    # get Rotation matrix & rotate
    ##########################################

    v = np.array((0, 0, 1))
    pangle = np.radians(22.5)
    pv = np.array((np.cos(pangle), np.sin(pangle), pangle*0)).T

    log.info('calculating rotation matrix ')
    rmat = gbdir.Rot_matrix(az=az, lst=lst)

    log.info('Rotate vectors')
    v_obs = gbdir.Rotate(v_arr=v, rmat=rmat) 

    log.info('Rotate polarization vectors')
    pv_obs = gbdir.Rotate(v_arr=pv, rmat=rmat)

    log.info('Calculating polarization directions')
    psi_i = gbdir.angle_from_meridian_2D(v_obs, pv_obs)


    #########################################
    # TOD from map_in
    #########################################

    npix = len(map_in[0])
    nside = hp.npix2nside(npix) 
    nobs = len(v_obs)

    log.info('getting npix from vectors ')
    x = v_obs.T[0]
    y = v_obs.T[1]
    z = v_obs.T[2]
    pix_obs = hp.vec2pix(nside, x, y, z)

    I_obs = map_in[0][pix_obs]
    Q_obs = map_in[1][pix_obs]
    U_obs = map_in[2][pix_obs] 

    log.info('getting tods')

    tod_I = I_obs
    tod_Q = Q_obs*np.cos(2*psi_i) - U_obs*np.sin(2*psi_i)
    tod_U = Q_obs*np.sin(2*psi_i) + U_obs*np.cos(2*psi_i)

    log.info('TOD simulation end')
    
    return tod_I, tod_Q, tod_U


def sim_tod_focalplane(t1, t2, fsample=1000, map_in=None, rseed=42):
    """ Simulation module for an observation with a focal plane. 
    
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT.
    fsample : float
        Sampling frequency in sps.
        Default is 1000.
    map_in : float array
        Input map for tod simulation.
        If None, it synthesizes a map by using rseed as a random seed. 
        Default is None.
    rseed : int
        Random seed that is used to synthesize the source map.
    
    Returns
    -------
    ut : float array
        Time stamp in unixtime.
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_I : float array
        Simulated tod I.
    tod_Q : float array
        Simulated tod I.
    tod_U : float array
        Simulated tod I
    """
    param = GBparam()
    log = set_logger(mp.current_process().name)
    if map_in is None:
        ## synthesize TQU map 
        log.info('Synthesizing a map')
        import camb
        par_camb = camb.CAMBparams()
        par_camb.set_cosmology(H0=67.5)
        res = camb.get_results(par_camb)
        dls = res.get_cmb_power_spectra(par_camb, CMB_unit='K')['total']
        ell = np.arange(len(dls))
        cls = dl2cl(dls)

        np.random.seed(rseed)
        map_in = hp.synfast(cls=cls.T, nside=1024, new=True)

    ##################################
    # define local sidereal time
    ##################################
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    #ut_arr = np.arange(st.unix, et.unix, 1./fsample)
    #ut = Time(ut_arr, format='unix') 
    #lst_ut = ut.sidereal_time('apparent', par.lon).deg # takes ~ 2500 s for 6 hr observation

    # assuming that the Earth rotation is constant within a second.
    """ previous version
    ut1 = Time(np.arange(st.unix, et.unix+1, 1.), format='unix')
    ut = [ut1.unix[i] + np.arange(fsample)*(ut1.unix[i+1]-ut1.unix[i])/fsample for i in range(len(ut1)-1)]
    ut = np.array(ut).flatten()
    lst1 = ut1.sidereal_time('apparent', par.lon).deg
    lst = [lst1[i] + np.arange(fsample)*(lst1[i+1]-lst1[i])/fsample for i in range(len(lst1)-1)]
    lst = np.array(lst).flatten()
    """

    log.info ('Making time stamps')
    ut = st.unix + np.arange(0, np.rint(et.unix-st.unix), 1./fsample)
    #ut = np.arange(0, int(et.unix-st.unix), 1./fsample)

    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    f = interp1d(ut_1s, lst_1s, fill_value='extrapolate')
    lst = f(ut)


    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * param.omega_gb) % 360


    ##########################################
    # get Rotation matrix & rotate
    ##########################################

    theta = param.pixinfo['theta']
    phi = param.pixinfo['phi']

    v_arr = hp.ang2vec(np.radians(theta), np.radians(phi))
    log.debug('v_arr.shape: {}'.format(v_arr.shape))
    
    #pangle = np.radians(22.5)
    #pv = np.array((np.cos(pangle), np.sin(pangle), pangle*0)).T

    log.info('calculating rotation matrix ')
    rmat = gbdir.Rot_matrix(az=az, lst=lst)

    log.info('Rotate vectors')

    v_obs = gbdir.Rotate(v_arr=v_arr, rmat=rmat)
    v_zen = gbdir.Rotate(v_arr=(0,0,1), rmat=rmat)
    dec, ra = hp.vec2ang(v_zen, lonlat=True) 

    log.debug('v_obs.shape: {}'.format(v_obs.shape))
    v_obs = np.transpose(v_obs, (2,1,0))
    log.debug('v_obs.shape: {}'.format(v_obs.shape))

    #log.info ('Rotate polarization vectors')
    #pv_obs = gbdir.Rotate(v_arr=pv, rmat=rmat)

    #log.info ('Calculating polarization directions')
    #gbdir.angle_from_meridian_2D(v_obs, pv_obs)
    psi_obs = np.zeros(len(v_arr)) 


    #########################################
    # TOD from map_in
    #########################################

    npix = len(map_in[0])
    nside = hp.npix2nside(npix) 
    nobs = len(v_obs)

    log.info('getting npix from vectors ')

    I_obs = []
    Q_obs = []
    U_obs = []


    for vi in v_obs:
        x = vi[0]
        y = vi[1]
        z = vi[2]
        pix_obs = hp.vec2pix(nside, x, y, z)

        I_obs.append(map_in[0][pix_obs])
        Q_obs.append(map_in[1][pix_obs])
        U_obs.append(map_in[2][pix_obs])

    I_obs = np.array(I_obs).T
    Q_obs = np.array(Q_obs).T
    U_obs = np.array(U_obs).T

    log.info('getting tods')

    tod_I = np.array(I_obs)
    tod_Q = np.array(Q_obs*np.cos(2*psi_obs) - U_obs*np.sin(2*psi_obs))
    tod_U = np.array(Q_obs*np.sin(2*psi_obs) + U_obs*np.cos(2*psi_obs))

    log.info('TOD simulation end')

    return ut, dec, ra, tod_I, tod_Q, tod_U


def sim_tod_focalplane_multi(t1, t2, fsample=1000, map_in=None, rseed=42):
    """ Simulation module for an observation with a focal plane. 
    To test the multiprocessing. 
    
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT.
    fsample : float
        Sampling frequency in sps.
        Default is 1000.
    map_in : float array
        Input map for tod simulation.
        If None, it synthesizes a map by using rseed as a random seed. 
        Default is None.
    rseed : int
        Random seed that is used to synthesize the source map.
    
    Returns
    -------
    ut : float array
        Time stamp in unixtime.
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_I : float array
        Simulated tod I.
    tod_Q : float array
        Simulated tod I.
    tod_U : float array
        Simulated tod I
    """
    param = GBparam()
    log = set_logger(mp.current_process().name)

    if map_in is None:
        ## synthesize TQU map 
        log.info('Synthesizing a map')
        import camb
        par_camb = camb.CAMBparams()
        par_camb.set_cosmology(H0=67.5)
        res = camb.get_results(par_camb)
        dls = res.get_cmb_power_spectra(par_camb, CMB_unit='K')['total']
        ell = np.arange(len(dls))
        cls = dl2cl(dls)

        np.random.seed(rseed)
        map_in = hp.synfast(cls=cls.T, nside=1024, new=True)

    ##################################
    # define local sidereal time
    ##################################
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    #ut_arr = np.arange(st.unix, et.unix, 1./fsample)
    #ut = Time(ut_arr, format='unix') 
    #lst_ut = ut.sidereal_time('apparent', par.lon).deg # takes ~ 2500 s for 6 hr observation

    # assuming that the Earth rotation is constant within a second.
    """ previous version
    ut1 = Time(np.arange(st.unix, et.unix+1, 1.), format='unix')
    ut = [ut1.unix[i] + np.arange(fsample)*(ut1.unix[i+1]-ut1.unix[i])/fsample for i in range(len(ut1)-1)]
    ut = np.array(ut).flatten()
    lst1 = ut1.sidereal_time('apparent', par.lon).deg
    lst = [lst1[i] + np.arange(fsample)*(lst1[i+1]-lst1[i])/fsample for i in range(len(lst1)-1)]
    lst = np.array(lst).flatten()
    """

    log.info('Making time stamps')
    ut = st.unix + np.arange(0, np.rint(et.unix-st.unix), 1./fsample)
    #ut = np.arange(0, int(et.unix-st.unix), 1./fsample)

    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    f = interp1d(ut_1s, lst_1s, fill_value='extrapolate')
    lst = f(ut)


    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * param.omega_gb) % 360


    ##########################################
    # get Rotation matrix & rotate
    ##########################################

    theta = param.pixinfo['theta']
    phi = param.pixinfo['phi']

    v_arr = hp.ang2vec(np.radians(theta), np.radians(phi))
    log.info(v_arr.shape)
    
    del(theta); del(phi)
    #pangle = np.radians(22.5)
    #pv = np.array((np.cos(pangle), np.sin(pangle), pangle*0)).T

    log.info ('calculating rotation matrix ')
    rmat = gbdir.Rot_matrix(az=az, lst=lst)

    log.info ('Rotate vectors')

    v_obs = gbdir.Rotate(v_arr=v_arr, rmat=rmat)
    v_zen = gbdir.Rotate(v_arr=(0,0,1), rmat=rmat)
    dec, ra = hp.vec2ang(v_zen, lonlat=True) 

    log.debug('v_obs.shape: {}'.format(v_obs.shape))
    v_obs = np.transpose(v_obs, (2,1,0))
    log.debug('v_obs.shape: {}'.format(v_obs.shape))

    #log.info ('Rotate polarization vectors')
    #pv_obs = gbdir.Rotate(v_arr=pv, rmat=rmat)

    #log.info ('Calculating polarization directions')
    psi_obs = np.zeros(len(v_arr)) #gbdir.angle_from_meridian_2D(v_obs, pv_obs)


    #########################################
    # TOD from map_in
    #########################################

    npix = len(map_in[0])
    nside = hp.npix2nside(npix) 
    nobs = len(v_obs)

    log.info('getting npix from vectors ')

    I_obs = []
    Q_obs = []
    U_obs = []

    st = time.time()

    pix_obs_arr = []

    for vi in v_obs:
        #x = vi[0]
        #y = vi[1]
        #z = vi[2]
        #pix_obs_arr.append(hp.vec2pix(nside, x, y, z))
        pix_obs_arr.append(fnc(vi))

    log.debug('Time for the for loops: {}'.format(time.time() - st))

    st = time.time()
    p = Pool(4)

    pix_obs_arr2 = p.map(fnc, v_obs)

    log.debug('Time for the for mp map: {}'.format(time.time() - st))

    for pix_obs in pix_obs_arr:
        I_obs.append(map_in[0][pix_obs])
        Q_obs.append(map_in[1][pix_obs])
        U_obs.append(map_in[2][pix_obs])


    I_obs = np.array(I_obs).T
    Q_obs = np.array(Q_obs).T
    U_obs = np.array(U_obs).T

    log.info('getting tods')

    tod_I = np.array(I_obs)
    tod_Q = np.array(Q_obs*np.cos(2*psi_obs) - U_obs*np.sin(2*psi_obs))
    tod_U = np.array(Q_obs*np.sin(2*psi_obs) + U_obs*np.cos(2*psi_obs))

    log.info('TOD simulation end')

    return ut, dec, ra, tod_I, tod_Q, tod_U


def sim_tod_focalplane_module(t1, t2, fsample=1000, map_in=None, rseed=42,
                              module_id=None, nside_hitmap=False, 
                              convention_LT=True):
    """ Simulation module for an observation with a focal plane. 
    
    Parameters
    ----------
    t1 : string
        Starting time of the simulation in ISOT.
    t2 : string
        End time of the simulation in ISOT.
    fsample : float
        Sampling frequency in sps.
        Default is 1000.
    map_in : float array
        Input map for tod simulation.
        If None, it synthesizes a map by using rseed as a random seed. 
        Default is None.
    rseed : int
        Random seed that is used to synthesize the source map.
        Default is 42.
    module_id : int or int array
        Indices of the modules to be used.
    nside_hitmap : int
        Nside of N-hit maps to be generated.
        If False, N-hit map will not be generated.
    convention_LT : bool
        If True, the psi angles are defined as LightTools convention. 
        Otherwise, the psi angles are defined in spherical coordinates.  
    
    Returns
    -------
    ut : float array
        Time stamp in unixtime.
    el : float or float array
        Elevation.
    az : float array
        Azimuth angle
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_Ix_mod : float array
        Simulated tod Ix for given modules.
    tod_Iy_mod : float array
        Simulated tod Iy for given modules.
    tod_psi_mod : float array
        Simulated tod psi for given modules.
    tod_pix_mod : float array
        Observed sky pixel 
    module_id_set : int or int array
        Indices of the used modules.
    hitmap : float array
        N-hit maps for each modules.
    """ 
    param = GBparam()
    log = set_logger(mp.current_process().name)

    if map_in is None:
        ## synthesize TQU map 
        log.info('Synthesizing a map')
        import camb
        par_camb = camb.CAMBparams()
        par_camb.set_cosmology(H0=67.5)
        res = camb.get_results(par_camb)
        dls = res.get_cmb_power_spectra(par_camb, CMB_unit='K')['total']
        ell = np.arange(len(dls))
        cls = dl2cl(dls)

        np.random.seed(rseed)
        map_in = hp.synfast(cls=cls.T, nside=1024, new=True)

    ##################################
    # calculate local sidereal time
    ##################################

    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')

    log.info('Making time stamps')
    ## assuming that the Earth rotation is constant within a second.
    ut = st.unix + np.arange(0, np.rint(et.unix-st.unix), 1./fsample)
    ut_1s = ut[::fsample]
    lst_1s = gbdir.unixtime2lst(ut_1s)
    ut2lst_tmp = interp1d(ut_1s, lst_1s, fill_value='extrapolate')
    lst = ut2lst_tmp(ut)
    del(ut2lst_tmp)

    ######################################
    # define GB rotation (azimuth) angle 
    ######################################

    az0 = 0
    t = ut-ut[0]
    az = (az0 + t * param.omega_gb) % 360

    ##########################################
    # get Rotation matrix & rotate
    ##########################################

    ## get module pixels
    ## modset: all modules in pixel information
    modset = set(map(int, param.pixinfo['mod'])) 
    if (module_id is None):
        module_id = list(modset)

    ## modcnt: number of modules
    module_cnt = len(np.array(module_id).flatten())    

    if module_cnt == 0: 
        module_id = list(modset)
    elif module_cnt == 1:
        module_id = [module_id]
    else:
        module_id = list(module_id)

    ## Is selected modules a subset of modset? 
    if not (set(module_id) <= modset):  
        log.warning('module_id {} should be a subset of {}. Using available modules only.'.format(module_id, modset))
        module_id_set = list(set(module_id).intersection(modset))
        if len(module_id_set) == 0:
            log.critical('No available modules.')
            raise
    else:
        module_id_set = module_id

    ## considering pixels in modules
    modpix_arr = list(map((lambda n: list(np.where(param.pixinfo['mod']==n)[0])), module_id_set))
    modpix = sum(modpix_arr, [])
    modpix_cnt = list(map(len, modpix_arr))

    ## print out the number of pixels in each module.
    for n, npix in zip(module_id_set, modpix_cnt):
        log.debug('Module {} has {} pixels.'.format(int(n), npix))

    ## get rotation matrices
    theta = param.pixinfo['theta'][modpix]
    phi = param.pixinfo['phi'][modpix]

    ## direction in horizontal coordinate, v_arr: (ndetector * 3)
    v_arr = hp.ang2vec(np.radians(theta), np.radians(phi)) 
    log.debug('v_arr.shape: {}'.format(v_arr.shape))
    del(theta)
    del(phi)
    
    log.info('calculating rotation matrix ')
    el = param.el
    rmat = gbdir.Rot_matrix(az=az, lst=lst)
    del(lst)

    log.info('Rotate vectors')
    ## direction on sky, v_obs: (nsample * 3 * ndetector)
    v_obs = gbdir.Rotate(v_arr=v_arr, rmat=rmat) 
    
    ## rotate zenith to get declination and right ascension
    v_zen = gbdir.Rotate(v_arr=(0, 0, 1), rmat=rmat)

    ## longitude and latitude in degrees
    ra, dec = hp.vec2ang(v_zen, lonlat=True) 

    ## polarization angles
    pangle = param.pixinfo['omtffr'][modpix]
    if (convention_LT):
        ## polarisation vectors on focalplane, pv: (ndetector * 3)
        pv = gbdir.psi2vec_xp(v_arr=v_arr, psi=pangle) 
    else:
        ## polarization vectors on focalplane, pv: (ndetector * 3)
        pv = gbdir.psi2vec(v_arr=v_arr, psi=pangle) 
    log.debug('pv.shape: {}'.format(pv.shape))

    log.info ('Rotate polarization vectors')
    ## polarization vectors on sky, pv_obs:(nsample * 3 * ndetector)
    pv_obs = gbdir.Rotate(v_arr=pv, rmat=rmat) 

    log.info('Calculating polarization directions')
    log.debug('v_obs.shape: {}'.format(v_obs.shape))
    log.debug('pv_obs.shape: {}'.format(pv_obs.shape))
    ## polarization angles on sky psi_obs: (nsample * ndetector)
    psi_obs = gbdir.angle_from_meridian(v_obs, pv_obs) 

    del(pv_obs)

    #########################################
    # TOD from map_in
    #########################################

    npix = len(map_in[0])
    nside = hp.npix2nside(npix) 

    log.info('getting npix from vectors ')

    ## QU maps to Intensity & psi maps
    Ip_src, psi_src = qu2ippsi(map_in[1], map_in[2]) 

    ## observed pixels, pix_obs: (nsample * ndetector)
    pix_obs = hp.vec2pix(nside, v_obs[:,0], v_obs[:,1], v_obs[:,2]) 

    ## observed pixels for N-hit map, pix_hit: (nsample * ndetector)
    if nside_hitmap:
        pix_hit = hp.vec2pix(nside_hitmap, v_obs[:,0], v_obs[:,1], v_obs[:,2]) 

    log.info('getting tods')

    del(v_obs); 

    ## I/Q/U_obs: (nsample * ndetector) 
    tod_I = map_in[0][pix_obs] 
    tod_Ip = Ip_src[pix_obs]
    tod_psi = psi_src[pix_obs]

    #tod_Ix = 0.5 * tod_I + tod_Ip * np.cos(tod_psi - psi_obs)**2
    #tod_Iy = 0.5 * tod_I + tod_Ip * np.sin(tod_psi - psi_obs)**2 
    tod_Ix = 0.5 * tod_I + tod_Ip * np.cos(tod_psi - psi_obs)**2
    tod_Iy = 0.5 * tod_I + tod_Ip * np.sin(tod_psi - psi_obs)**2 
    del(tod_I); del(tod_Ip); del(psi_obs)

    tod_Ix_mod = []
    tod_Iy_mod = []
    tod_psi_mod = []
    tod_pix_mod = [] 
    hit_pix_mod = []
    n0 = 0

    for n in np.add.accumulate(modpix_cnt):
        ## tod_Ix/Iy/psi_mod: (module_cnt * nsample * ndetector)
        tod_Ix_mod.append(tod_Ix[:, n0:n])  
        tod_Iy_mod.append(tod_Iy[:, n0:n])
        tod_psi_mod.append(tod_psi[:, n0:n])
        tod_pix_mod.append(pix_obs[:, n0:n])
        if nside_hitmap:
            hit_pix_mod.append(pix_hit[:, n0:n])
        n0 = n

    hitmap = []
    if nside_hitmap:
        for pixs in hit_pix_mod:
            hitmap_tmp = np.full(12*nside_hitmap**2, hp.UNSEEN)
            npix, nhit = np.unique(pixs, return_counts=True) 
            hitmap_tmp[npix] = nhit
            hitmap.append(hitmap_tmp)

    log.info('TOD simulation end')

    #return ut, el, az, dec, ra, \
    #       tod_I_mod, tod_Q_mod, tod_U_mod, \
    #       module_id_set, hitmap

    return ut, el, az, dec, ra, \
           tod_Ix_mod, tod_Iy_mod, tod_psi_mod, tod_pix_mod,\
           module_id_set, hitmap


def wr_tod2fits(fname, ut, az, dec, ra, tod_I, tod_Q, tod_U):
    """ Write tod in fits file.
    
    Parameters
    ----------
    fname : string
        fits file name.
    ut : float array
        Timestamp in unixtime.
    az : float array
        Azimuth angles.
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_I : float array
        TOD I 
    tod_Q : float array
        TOD Q
    tod_U : float array 
        TOD U

    Raise
    -----
    WARNING
        If the fits file is exists, the data will be overwritten.
    """
    log = set_logger(mp.current_process().name)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='UT',    format='D', array=ut,  dim='{}'.format('')),
        fits.Column(name='AZ',    format='E', array=az,  dim='{}'.format('')),
        fits.Column(name='DEC',   format='E', array=dec, dim='{}'.format('')),
        fits.Column(name='RA',    format='E', array=ra,  dim='{}'.format('')),
        fits.Column(name='TOD_I', format='{}E'.format(np.prod(np.shape(tod_I)[1:])),
                    array=tod_I,  dim='{}'.format('')),
        fits.Column(name='TOD_Q', format='{}E'.format(np.prod(np.shape(tod_Q)[1:])), 
                    array=tod_Q,  dim='{}'.format('')),
        fits.Column(name='TOD_U', format='{}E'.format(np.prod(np.shape(tod_U)[1:])), 
                    array=tod_U,  dim='{}'.format(''))
        ])

    try: 
        hdu.writeto(fname)
        log.info('{} has been created.'.format(fname))
    except OSError:
        hdu.writeto(fname, overwrite=True)
        log.warning('{} has been overwritten.'.format(fname))

    return
                

def wr_tod2fits_mod_TQU(fname, ut, az, dec, ra, 
                    tod_I_mod, tod_Q_mod, tod_U_mod, 
                    module_id, **aheaders):
    """ Write tod in fits file. 

    Parameters
    ----------
    fname : string
        fits file name.
    ut : float array
        Timestamp in unixtime.
    az : float array
        Azimuth angles.
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_I_mod : float array
        TOD I for each module.
    tod_Q_mod : float array
        TOD Q for each module.
    tod_U_mod : float array 
        TOD U for each module.
    module_id : int or int array
        Module indices
    **aheaders : dictionary
        Additional headers.
    """
    log = set_logger(mp.current_process().name)
    cols = [fits.Column(name='UT',  format='D', array=ut ),
            fits.Column(name='AZ',  format='E', array=az ),
            fits.Column(name='DEC', format='E', array=dec),
            fits.Column(name='RA',  format='E', array=ra )]

    header = fits.Header()
    for key, value in aheaders.items():
        header[key] = value
        
    for n, tod_I, tod_Q, tod_U in zip(module_id, tod_I_mod, tod_Q_mod, tod_U_mod):
        cols.append(fits.Column(name='TOD_I_mod%d' % (n), 
                        format='{}E'.format(np.prod(np.shape(tod_I)[1:])), 
                        array=tod_I))
        cols.append(fits.Column(name='TOD_Q_mod%d' % (n), 
                        format='{}E'.format(np.prod(np.shape(tod_Q)[1:])), 
                        array=tod_Q))
        cols.append(fits.Column(name='TOD_U_mod%d' % (n), 
                        format='{}E'.format(np.prod(np.shape(tod_U)[1:])), 
                        array=tod_U))

    hdu = fits.BinTableHDU.from_columns(cols, header)

    try: 
        hdu.writeto(fname)
        log.info('{} has been created.'.format(fname))
    except OSError:
        hdu.writeto(fname, overwrite=True)
        log.warning('{} has been overwritten.'.format(fname))

    return


def wr_tod2fits_mod(fname, ut, az, dec, ra, 
                    tod_Ix_mod, tod_Iy_mod, tod_psi_mod, tod_pix_mod, 
                    module_id, **aheaders):
    """ Write tod in fits file. 

    Parameters
    ----------
    fname : string
        fits file name.
    ut : float array
        Timestamp in unixtime.
    az : float array
        Azimuth angles.
    dec : float array
        Declination.
    ra : float array
        Right ascension.
    tod_Ix_mod : float array
        TOD Ix for each module.
    tod_Iy_mod : float array
        TOD Iy for each module.
    tod_psi_mod : float array 
        TOD psi for each module.
    tod_pix_mod : float array   
        Observed sky pixel for each module
    module_id : int or int array
        Module indices
    **aheaders : dictionary
        Additional headers.
    """
    log = set_logger(mp.current_process().name)
    cols = [fits.Column(name='UT',  format='D', array=ut ),
            fits.Column(name='AZ',  format='E', array=az ),
            fits.Column(name='DEC', format='E', array=dec),
            fits.Column(name='RA',  format='E', array=ra )]

    header = fits.Header()
    for key, value in aheaders.items():
        header[key] = value
        
    for n, tod_Ix, tod_Iy, tod_psi, tod_pix in zip(module_id, tod_Ix_mod, tod_Iy_mod, tod_psi_mod, tod_pix_mod):
        cols.append(fits.Column(name='TOD_Ix_mod_%d' % (n), 
                        format='{}D'.format(np.prod(np.shape(tod_Ix)[1:])), 
                        array=tod_Ix))
        cols.append(fits.Column(name='TOD_Iy_mod_%d' % (n), 
                        format='{}D'.format(np.prod(np.shape(tod_Iy)[1:])), 
                        array=tod_Iy))
        cols.append(fits.Column(name='TOD_psi_mod_%d' % (n), 
                        format='{}D'.format(np.prod(np.shape(tod_psi)[1:])), 
                        array=tod_psi))
        cols.append(fits.Column(name='TOD_pix_mod_%d' % (n), 
                        format='{}J'.format(np.prod(np.shape(tod_pix)[1:])), 
                        array=tod_pix))

    hdu = fits.BinTableHDU.from_columns(cols, header)

    try: 
        hdu.writeto(fname)
        log.info('{} has been created.'.format(fname))
    except OSError:
        hdu.writeto(fname, overwrite=True)
        log.warning('{} has been overwritten.'.format(fname))

    return


def wr_tod2fits_noise(fname, ut, noise, module_id, **aheaders):
    """ Write noise tod in fits file. 

    Parameters
    ----------
    fname : string
        fits file name.
    ut : float array
        Timestamp in unixtime.
    noise : float array
        Noise tod.
    module_id : int or int array
        Module indices
    **aheaders : dictionary
        Additional headers.
    """
    log = set_logger(mp.current_process().name)
    header = fits.Header()
    for key, value in aheaders.items():
        header[key] = value

    cols =([fits.Column(name='UT',  format='D', array=ut, dim='{}'.format('')),
            fits.Column(name='N1f', format='E', 
                        array=noise,  dim='{}'.format(''))])

    hdu = fits.BinTableHDU.from_columns(cols, header)

    try: 
        hdu.writeto(fname)
        log.info('{} has been created.'.format(fname))
    except OSError:
        hdu.writeto(fname, overwrite=True)
        log.warning('{} has been overwritten.'.format(fname))

    return


def scp_file(local_path, remote_path, remove=False):
    """ Copy the files to criar. """
    log = set_logger(mp.current_process().name)
    log.info('Copying file {} to criar.'.format(local_path))
    log = set_logger(mp.current_process().name)
    ssh_client=paramiko.client.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.connect('criar')

    with ssh_client.open_sftp() as sftp:
        try:
            sftp.put(local_path, remote_path)
        except Exception as e:
            log.warning('An error occured during sftping the file {}'.format(local_path)) 
            log.warning(str(e))

    log.info('Copying file complete.')

    if (remove):
        os.remove(ofname)
        log.info('File {} has been removed.'.format(local_path))

    return


def func_parallel_tod(t1, t2, fsample, mapname='cmb_rseed42.fits', 
                      module_id=None, fprefix='GBtod', outpath='.', 
                      nside_hitmap=False, transferFile=False):
    """ Function for a parallelization of the tod simulation.
    
    Parameters
    ----------
    t1 : string
        Simulation starting time in ISOT.
    t2 : string
        Simulation end time in ISOT.
    fsample : float
        Sampling frequency in sps.
    mapname : string
        The filename of the input map.
        Default is 'cmb-rseed42.fits'.
    module_id : int or sequence of int
        Module indicies to be used. 
        If None, all the modules are used. 
        Defaule is None.
    fprefix : string
        Prefix for the names of the fits files.
        Default is 'GBtod'.
    outpath : string
        Path to save the data.
        Defalut is '.' (current directory). 
    nside_hitmap : int
        Nside of the N-hit maps. 
        If False, N-hit maps are not going to be calculated. 
        Default is False.
    transferFile : bool
        If True, the data files are transferred to criar automatically.
        Default is False.
    """

    log = set_logger(mp.current_process().name)
    map_in = hp.read_map(mapname, field=(0,1,2), verbose=False)

    res = sim_tod_focalplane_module(t1, t2, map_in=map_in, fsample=fsample, 
                                    module_id=module_id, nside_hitmap=nside_hitmap)

    ut, el, az, dec, ra, tod_Ix, tod_Iy, tod_psi, tod_pix, nmodout, hitmap = res 

    nmodpixs = [np.shape(tod)[1] for tod in tod_Ix]

    fname = '{}_{}_{}.fits'.format(fprefix, t1, t2)
    aheaders = {'FNAME'   : fname, 
                'CTIME'   : (Time.now().isot, 'File created time'),
                'DATATYPE': 'TODSIM',
                'ISOT0'   : (t1, 'Observation start time'), 
                'ISOT1'   : (t2, 'Observation end time'),
                'FSAMPLE' : (fsample, 'Sampling frequency (Hz)'),
                'EL'      : (el, 'Elevation'),
                'NMODULES': (str(list(map(int, nmodout)))[1:-1], 'Used modules'),
                'NMODPIXS': (str(nmodpixs)[1:-1], 'Number of pixels in each module'),
               }

    if (socket.gethostname() == 'criar'):
        opath = outpath
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, t1[:10])
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, fprefix)
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        ofname = os.path.join(opath, fname)
        wr_tod2fits_mod(ofname, ut, az, dec, ra, tod_Ix, tod_Iy, tod_psi, tod_pix, nmodout, **aheaders)
    else:
        opath = outpath 
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, t1[:10])
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, fprefix)
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))

        ofname = os.path.join(opath, fname)
        dfname = os.path.join(opath, fname)
        wr_tod2fits_mod(ofname, ut, az, dec, ra, tod_Ix, tod_Iy, tod_psi, tod_pix, nmodout, **aheaders)

        if (transferFile):
            scp_file(ofname, dfname, remove=True)

    if nside_hitmap:
        hpath = opath + '_hitmap'

        try:
            os.mkdir(hpath)
        except OSError:
            log.warning('The path {} exists.'.format(hpath))

        hfname = os.path.join(hpath, '{}_hitmap_{}_{}.fits'.format(fprefix, t1, t2))
        cnames = ['hitmap_'+str(i) for i in nmodout]

        if (os.path.isfile(hfname)):
            log.warning('{} has been overwritten.'.format(hfname))
            hp.write_map(hfname, hitmap, column_names=cnames, overwrite=True)
        else:
            hp.write_map(hfname, hitmap, column_names=cnames, overwrite=False)

        log.info('N-hit map is written in {}.'.format(hfname))

    return


def func_parallel_noise(t1, t2, dtsec=600, fsample=10, 
                        wnl=1, fknee=1, alpha=1, rseed=0, 
                        module_id=None, fprefix='GBtod_noise', 
                        outpath='.', transferFile=False):
    """ Function for a parallelization of the noise simulation.
    
    Parameters
    ----------
    t1 : string
        Simulation starting time in ISOT.
    t2 : string
        Simulation end time in ISOT.
    dtsec : float
        Time interval between t1 and t2.
    fsample : float
        Sampling frequency in sps.
        Default is 10.
    wnl : float
        White noise level. (The unit is not deterined yet.)
        Default is 1.
    fknee : float
        Knee frequency of the noise in Hz.
        Default is 1.
    alpha : float
        Exponent of the 1/f noise.
        Default is 1.
    rseed : int
        Random seed to be used in white noise generation.
        Default is 0.
    module_id : int or sequence of int
        Module indicies to be used. 
        If None, all the modules are used. 
        Defaule is None.
    fprefix : string
        Prefix for the names of the fits files.
        Default is 'GBtod_noise'.
    outpath : string
        Path to save the data.
        Defalut is '.' (current directory). 
    transferFile : bool
        If True, the data files are transferred to criar automatically.
        Default is False.
    """

    log = set_logger(mp.current_process().name)

    if module_id == None:
        module_id = list(range(6))

    log.info('Making time stamps')
    st = Time(t1, format='isot', scale='utc')
    et = Time(t2, format='isot', scale='utc')
    ut = st.unix + np.arange(0, np.rint(et.unix-st.unix), 1./fsample)

    l = dtsec * fsample

    if (len(ut) != l):
        log.error('Something is wrong. (len(ut)={}) != (l={})'.format(len(ut), l))

    noise = sim_noise1f(l, wnl, fknee, fsample, alpha, rseed=rseed)

    fname = '{}_{}_{}.fits'.format(fprefix, t1, t2)
    aheaders = {'FNAME': fname, 
                'CTIME'   : (Time.now().isot, 'File created time'),
                'DATATYPE': 'NOISE SIM',
                'FSAMPLE' : (fsample, 'Sampling frequency (Hz)'),
                'ISOT0': (t1, 'Observation start time'), 
                'ISOT1': (t2, 'Observation end time'),
                'FSAMPLE': (fsample, 'Sampling frequency (Hz)'),
                'NMODULES': (str(list(map(int, module_id)))[1:-1], 'Used modules'),
                #'NMODPIXS': (str(modpix_cnt)[1:-1], 'Number of pixels in each module'),
               }
    
    if (socket.gethostname() == 'criar'):
        opath = outpath 
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, t1[:10])
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, fprefix)
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        ofname = os.path.join(opath, fname)
        wr_tod2fits_noise(ofname, ut, noise, module_id, **aheaders)
    else:
        opath = outpath
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, t1[:10])
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        opath = os.path.join(opath, fprefix)
        try:
            os.mkdir(opath)
        except OSError:
            log.warning('The path {} exists.'.format(opath))
        ofname = os.path.join(opath, fname)
        dfname = os.path.join(opath, fname)
        wr_tod2fits_noise(ofname, ut, noise, module_id, **aheaders)
        if (transferFile):
            scp_file(ofname, dfname, remove=True)

    return


def GBsim_hpc_parallel_time(
        t1='2019-04-01T00:00:00', t2='2019-04-08T00:00:00', 
        dtsec=600, fsample=10, mapname='cmb_rseed42.fits', 
        module_id=None, fprefix='GBtod_CMB', outpath=None,
        nside_hitmap=False, nproc=8):
    """ GroundBIRD simulation module for TOD. It is parallelized 
    over the time. 

    Parameters
    ----------
    t1 : string
        Simulation starting time in ISOT.
    t2 : string
        Simulation end time in ISOT.
    dtsec : float
        Time interval between t1 and t2 in second.
        Default is 600.
    fsample : float
        Sampling frequency in sps.
    mapname : string
        The filename of the input map.
        Default is 'cmb-rseed42.fits'.
    module_id : int or sequence of int
        Module indicies to be used. 
        If None, all the modules are used. 
        Defaule is None.
    fprefix : string
        Prefix for the names of the fits files.
        Default is 'GBtod_CMB'.
    outpath : string
        Path to save the data.
        Defalut is '.' (current directory). 
    nside_hitmap : int
        Nside of the N-hit maps. 
        If False, N-hit maps are not going to be calculated. 
        Default is False.
    nproc : int
        Maximum number of processes.
        Default is 8.
    """
    log = set_logger(mp.current_process().name)

    st = t1
    et = t2

    if (outpath is None):
        outpath = './{}_GBsim'.format(today())

    st = Time(st, format='isot', scale='utc')
    et = Time(et, format='isot', scale='utc')
    dt = TimeDelta(dtsec, format='sec')

    ## for download the IERS tables in advance
    log.info(st.sidereal_time('mean', 'greenwich'))
    log.info(et.sidereal_time('mean', 'greenwich'))

    Nf = int((et-st)/dt) 

    procs = []

    nmaxproc = min(nproc, mp.cpu_count()-1)
    log.info('Using {} cpus'.format(nmaxproc))

    for i in range(Nf):
        t1_ = (st + i*dt).isot
        t2_ = (st + (i+1)*dt).isot
        log.info('t1={}, t2={}'.format(t1_, t2_))
        proc = mp.Process(target=func_parallel_tod, 
                          args=(t1_, t2_, fsample, mapname, module_id, 
                                fprefix, outpath, nside_hitmap))
        procs.append(proc)

    log.debug(procs)
    
    procs_running = []
    while len(procs):
        if len(procs_running) < nmaxproc:
            procs_running.append(procs[0])
            procs[0].start()
            log.info ('{} has been started'.format(procs[0].name))
            procs.remove(procs[0])
        else:
            for proc in procs_running:
                if not proc.is_alive():
                    proc.join()
                    procs_running.remove(proc)
             
    for proc in procs_running:
        proc.join()

    return
     

def GBsim_noise(
        t1='2019-04-01T00:00:00', t2='2019-04-08T00:00:00', 
        dtsec=600, fsample=10, 
        wnl=1, fknee=1, alpha=1, rseed=0,
        module_id=None, fprefix='GBtod_noise', outpath='.', nproc=8):

    """ GroundBIRD simulation module for noise. It is parallelized 
    over the time.

    Parameters
    ----------
    t1 : string
        Simulation starting time in ISOT.
    t2 : string
        Simulation end time in ISOT.
    dtsec : float
        Time interval between t1 and t2 in second.
        Default is 600.
    fsample : float
        Sampling frequency in sps.
    wnl : float
        White noise level.
        Default is 1.
    fknee : float
        Knee frequency of the noise spectrum in Hz.
        Default is 1.
    alpha : float
        Exponent of the 1/f noise. 
        Default is 1.
    rseed : int
        Random seed to be used for white noise generation.
        Default is 0.
    module_id : int or sequence of int
        Module indicies to be used. 
        If None, all the modules are used. 
        Defaule is None.
    fprefix : string
        Prefix for the names of the fits files.
        Default is 'GBtod_noise'.
    outpath : string
        Path to save the data.
        Defalut is '.' (current directory). 
    nproc : int
        Maximum number of processes.
        Default is 8.
    """
    log = set_logger(mp.current_process().name)

    st = t1
    et = t2

    if (outpath is None):
        outpath = './{}_GBsim'.format(today())

    st = Time(st, format='isot', scale='utc')
    et = Time(et, format='isot', scale='utc')
    dt = TimeDelta(dtsec, format='sec')

    Nf = int((et-st)/dt) 
    procs = []

    nmaxproc = min(nproc, mp.cpu_count()-1)
    log.info('Using {} cpus'.format(nmaxproc))

    nsample = Nf
    np.random.seed(rseed)
    rseeds = np.random.randint(low=0, high=2**32-1, size=nsample)

    for i in range(Nf):
        t1 = (st + i*dt).isot
        t2 = (st + (i+1)*dt).isot
        log.info('t1={}, t2={}'.format(t1, t2))
        proc = mp.Process(target=func_parallel_noise,
                          args=(t1, t2, dtsec, fsample, wnl, fknee, alpha, 
                                rseeds[i], module_id, fprefix, outpath))
        procs.append(proc)

    log.debug(procs)
    
    procs_running = []
    while len(procs):
        if len(procs_running) < nmaxproc:
            procs_running.append(procs[0])
            procs[0].start()
            log.info ('{} has been started'.format(procs[0].name))
            procs.remove(procs[0])
        else:
            for proc in procs_running:
                if not proc.is_alive():
                    proc.join()
                    procs_running.remove(proc)
             
    for proc in procs_running:
        proc.join()

    return


def test_nhit():
    """ Test module for nhit """
    t1 = '2018-12-04T12:00:00.0' 
    t2 = '2018-12-04T12:10:00.0' # +1 min 

    #sim_obs_singlepix(t1, t2, fsample=10)
    sim_obs_focalplane(t1, t2, fsample=10)


def test_tod():
    """ Test module for TOD """
    log = set_logger(function_name())
    t1 = '2018-12-04T12:00:00.0'
    t2 = '2018-12-04T12:01:00.0' # +1 min 
    #t2 = '2018-12-04T12:10:00.0' # +10 min 
    #t2 = '2018-12-04T18:00:00.0' # +6 hr
    #t2 = '2018-12-05T12:00:00.0' # +24 hr
    
    #tod_I, tod_Q, tod_U = sim_tod_singlepix(t1, t2, fsample=1000)
    tod_I, tod_Q, tod_U = sim_tod_focalplane(t1, t2, fsample=1000)
    #n_1f = sim_noise1f(l=len(tod_I), wnl=1e-3, fknee=1)

    log.debug('tod_I.shape: {}'.format(tod_I.shape))
    log.debug('tod_Q.shape: {}'.format(tod_Q.shape))
    log.debug('tod_U.shape: {}'.format(tod_U.shape))

    log.info('test_tod end')
    #plt.plot(tod_I, label='tod_I')
    #plt.plot(tod_Q, label='tod_Q')
    #plt.plot(tod_U, label='tod_U')
    #plt.plot(n_1f, label='n_1f')
    #plt.legend()
    #plt.show()

    return


