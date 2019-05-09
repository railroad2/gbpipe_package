import time
import sys, os
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

from gbpipe import gbdir
from gbpipe.utils import dl2cl, cl2dl
from gbpipe.gbparam import GBparam
from gbpipe.utils import setLogger, funcname, today


def sim_noise1f(l, wnl, fknee, fsample=1000, alpha=1, rseed=0):
    """
    Generates noise tod which has power spectrum of 
    s(f) = (wnl**2/NFFT)*(1 + (fknee/f)**alpha)

    l : int
        data length 
    wnl : float
        white noise level (NET: Noise Equivalent Temperature)
    fknee : float
        knee frequency
    fsample : float
        sampling frequency
    alpha : float
        exponent of 1/f noise window
    """ 
    log = setLogger(funcname())
    
    t = np.arange(l) * 1./fsample
    np.random.seed(rseed)
    n0 = np.random.normal(scale=wnl, size=l)

    s0 = np.fft.fft(n0)/l
    freq = np.fft.fftfreq(l, d=1./fsample)
    freq[0] = freq[1]/100

    s = 1 + (fknee/abs(freq))**alpha
    s[0] = s[1]

    s_1f = s0 * s

    n_1f = np.fft.ifft(s_1f)

    return n_1f.real 


def sim_obs_singlepix(t1, t2, fsample=1000): 
    par = GBparam()
    log = setLogger(funcname())

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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)
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
    par = GBparam()
    log = setLogger(funcname())

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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)
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
    log = setLogger(funcname())
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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)

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
    
    return [tod_I, tod_Q, tod_U] 


def sim_tod_focalplane(t1, t2, fsample=1000, map_in=None, rseed=42):
    param = GBparam()
    log = setLogger(mp.current_process().name)
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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)

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
    param = GBparam()
    log = setLogger(mp.current_process().name)

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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)

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


def sim_tod_focalplane_module(t1, t2, fsample=1000, map_in=None, rseed=42, nmod=None, nside_hitmap=False, xp=True):
    param = GBparam()
    log = setLogger(mp.current_process().name)

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
    lst_1s = gbdir.unixtime2LST(ut_1s)
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

    ## get module pixels
    modset = set(map(int, param.pixinfo['mod'])) # modset: all modules in pixel information
    if (nmod is None):
        nmod = list(modset)

    lmod = len(np.array(nmod).flatten())    # lmod: number of modules

    if lmod == 0: 
        nmod = list(modset)
    elif lmod == 1:
        nmod = [nmod]
    else:
        nmod = list(nmod)

    if not (set(nmod) <= modset):  # Is selected modules a subset of all modules?
        log.warning('nmod {} should be a subset of {}. Using available modules only.'.format(nmod, modset))
        nmod = list(set(nmod).intersection(modset))
        if len(nmod) == 0:
            log.critical('No available modules.')
            raise

    modpixs = []
    nmodpixs = []
    modpixs_arr = list(map((lambda n: list(np.where(param.pixinfo['mod']==n)[0])), nmod))
    modpixs = sum(modpixs_arr, [])
    nmodpixs = list(map(len, modpixs_arr))

    for n, npix in zip(nmod, nmodpixs):
        log.debug('Module {} has {} pixels.'.format(int(n), npix))

    ## get rotation matrices
    theta = param.pixinfo['theta'][modpixs]
    phi = param.pixinfo['phi'][modpixs]

    v_arr = hp.ang2vec(np.radians(theta), np.radians(phi)) # direction in horizontal coordinate, v_arr: (ndetector * 3)
    log.debug('v_arr.shape: {}'.format(v_arr.shape))
    del(theta); del(phi)
    
    log.info('calculating rotation matrix ')
    el = param.EL
    rmat = gbdir.Rot_matrix(AZ=az, LST=lst)
    del(lst)

    log.info('Rotate vectors')
    v_obs = gbdir.Rotate(v_arr=v_arr, rmat=rmat) # direction on sky, v_obs: (nsample * 3 * ndetector)
    
    ## rotate zenith to get declination and right ascension
    v_zen = gbdir.Rotate(v_arr=(0, 0, 1), rmat=rmat)
    ra, dec = hp.vec2ang(v_zen, lonlat=True) # longitude and latitude in degrees

    ## polarization angles
    
    #pangle = np.radians(22.5)
    pangle = param.pixinfo['omtffr'][modpixs]
    if (xp):
        pv = gbdir.psi2vec_xp(v_arr=v_arr, psi=pangle) # pol. vectors on focalplane, pv: (ndetector * 3)
    else:
        pv = gbdir.psi2vec(v_arr=v_arr, psi=pangle) # pol. vectors on focalplane, pv: (ndetector * 3)
    log.debug('pv.shape: {}'.format(pv.shape))

    log.info ('Rotate polarization vectors')
    pv_obs = gbdir.Rotate(v_arr=pv, rmat=rmat) # pol. vectors on sky, pv_obs:(nsample * 3 * ndetector)

    log.info('Calculating polarization directions')
    log.debug('v_obs.shape: {}'.format(v_obs.shape))
    log.debug('pv_obs.shape: {}'.format(pv_obs.shape))
    psi_obs = gbdir.angle_from_meridian(v_obs, pv_obs) #pol. angles on sky psi_obs: (nsample * ndetector)

    del(pv_obs)

    #########################################
    # TOD from map_in
    #########################################

    npix = len(map_in[0])
    nside = hp.npix2nside(npix) 

    log.info('getting npix from vectors ')

    pix_obs = hp.vec2pix(nside, v_obs[:,0], v_obs[:,1], v_obs[:,2]) # observed pixels, pix_obs: (nsample * ndetector)
    if nside_hitmap:
        pix_hit = hp.vec2pix(nside_hitmap, v_obs[:,0], v_obs[:,1], v_obs[:,2]) # observed pixels, pix_obs: (nsample * ndetector)
    log.info('getting tods')

    del(v_obs); 

    I_obs = map_in[0][pix_obs] # I/Q/U_obs: (nsample * ndetector)
    Q_obs = map_in[1][pix_obs]
    U_obs = map_in[2][pix_obs]

    tod_I = np.array(I_obs)
    tod_Q = np.array(Q_obs*np.cos(2*psi_obs) - U_obs*np.sin(2*psi_obs))
    tod_U = np.array(Q_obs*np.sin(2*psi_obs) + U_obs*np.cos(2*psi_obs))

    del(I_obs); del(Q_obs); del(U_obs); del(psi_obs)

    tod_I_mod = []
    tod_Q_mod = []
    tod_U_mod = []
    pix_mod = [] 
    n0 = 0

    for n in np.add.accumulate(nmodpixs):
        tod_I_mod.append(tod_I[:, n0:n])  #tod_I/Q/U_mod: (nmod * nsample * ndetector)
        tod_Q_mod.append(tod_Q[:, n0:n])
        tod_U_mod.append(tod_U[:, n0:n])
        if nside_hitmap:
            pix_mod.append(pix_hit[:, n0:n])
        n0 = n

    log.info('TOD simulation end')
    result = [ut, el, az, dec, ra, tod_I_mod, tod_Q_mod, tod_U_mod, nmod]

    if nside_hitmap:
        hitmap = []
        for pixs in pix_mod:
            hitmap_tmp = np.full(12*nside_hitmap**2, hp.UNSEEN)
            npix, nhit = np.unique(pixs, return_counts=True) 
            hitmap_tmp[npix] = nhit
            hitmap.append(hitmap_tmp)

        result.append(hitmap)
    else:
        result.append([])

    
    #return ut, dec, ra, tod_I_mod, tod_Q_mod, tod_U_mod, nmod
    return result


def wr_tod2fits(fname, ut, az, dec, ra, tod_I, tod_Q, tod_U):
    log = setLogger(mp.current_process().name)
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
                

def wr_tod2fits_mod(fname, ut, az, dec, ra, tod_I_mod, tod_Q_mod, tod_U_mod, nmod, **aheaders):
    log = setLogger(mp.current_process().name)
    cols = [fits.Column(name='UT',  format='D', array=ut ),
            fits.Column(name='AZ',  format='E', array=az ),
            fits.Column(name='DEC', format='E', array=dec),
            fits.Column(name='RA',  format='E', array=ra )]

    header = fits.Header()
    for key, value in aheaders.items():
        header[key] = value
        
    for n, tod_I, tod_Q, tod_U in zip(nmod, tod_I_mod, tod_Q_mod, tod_U_mod):
        cols.append(fits.Column(name='TOD_I_mod%d' % (n), 
                    format='{}E'.format(np.prod(np.shape(tod_I)[1:])), array=tod_I))
        cols.append(fits.Column(name='TOD_Q_mod%d' % (n), 
                    format='{}E'.format(np.prod(np.shape(tod_Q)[1:])), array=tod_Q))
        cols.append(fits.Column(name='TOD_U_mod%d' % (n), 
                    format='{}E'.format(np.prod(np.shape(tod_U)[1:])), array=tod_U))

    hdu = fits.BinTableHDU.from_columns(cols, header)

    try: 
        hdu.writeto(fname)
        log.info('{} has been created.'.format(fname))
    except OSError:
        hdu.writeto(fname, overwrite=True)
        log.warning('{} has been overwritten.'.format(fname))

    return


def wr_tod2fits_noise(fname, ut, noise, nmodin, **aheaders):
    log = setLogger(mp.current_process().name)
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
    log = setLogger(mp.current_process().name)
    log.info('Copying file {} to criar.'.format(local_path))
    log = setLogger(mp.current_process().name)
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


def func_parallel_tod(
        t1, t2, fsample, mapname='cmb_rseed42.fits', 
        nmodin=None, fprefix='GBtod', outpath='.', 
        nside_hitmap=False, transferFile=False
    ):

    log = setLogger(mp.current_process().name)
    map_in = hp.read_map(mapname, field=(0,1,2), verbose=False)

    res = sim_tod_focalplane_module(t1, t2, map_in=map_in, fsample=fsample, 
                                    nmod=nmodin, nside_hitmap=nside_hitmap)

    ut, el, az, dec, ra, tod_I, tod_Q, tod_U, nmodout, hitmap = res 

    nmodpixs = [np.shape(tod)[1] for tod in tod_I]

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
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, t1[:10])
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, fprefix)
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        ofname = os.path.join(opath, fname)
        wr_tod2fits_mod(ofname, ut, az, dec, ra, tod_I, tod_Q, tod_U, nmodout, **aheaders)
    else:
        opath = outpath 
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, t1[:10])
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, fprefix)
        if not(os.path.isdir(opath)):
            os.mkdir(opath)

        ofname = os.path.join(opath, fname)
        dfname = os.path.join(opath, fname)
        wr_tod2fits_mod(ofname, ut, az, dec, ra, tod_I, tod_Q, tod_U, nmodout, **aheaders)

        if (transferFile):
            scp_file(ofname, dfname, remove=True)

    if nside_hitmap:
        hpath = opath + '_hitmap'
        if not(os.path.isdir(hpath)):
            os.mkdir(hpath)
        hfname = os.path.join(hpath, '{}_hitmap_{}_{}.fits'.format(fprefix, t1, t2))
        cnames = ['hitmap_'+str(i) for i in nmodout]
        if (os.path.isfile(hfname)):
            log.warning('{} has been overwritten.'.format(hfname))
            hp.write_map(hfname, hitmap, column_names=cnames, overwrite=True)
        else:
            hp.write_map(hfname, hitmap, column_names=cnames, overwrite=False)
        log.info('N-hit map is written in {}.'.format(hfname))

    return


def func_parallel_noise(
        t1, t2, dtsec=600, fsample=10, 
        wnl=1, fknee=1, alpha=1, rseed=0, 
        nmodin=None, fprefix='GBtod_noise', outpath='.', transferFile=False
    ):

    log = setLogger(mp.current_process().name)

    if nmodin == None:
        nmodin = list(range(6))

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
                'NMODULES': (str(list(map(int, nmodin)))[1:-1], 'Used modules'),
                #'NMODPIXS': (str(nmodpixs)[1:-1], 'Number of pixels in each module'),
               }
    
    if (socket.gethostname() == 'criar'):
        opath = outpath 
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, t1[:10])
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, fprefix)
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        ofname = os.path.join(opath, fname)
        wr_tod2fits_noise(ofname, ut, noise, nmodin, **aheaders)
    else:
        opath = outpath
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, t1[:10])
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        opath = os.path.join(opath, fprefix)
        if not(os.path.isdir(opath)):
            os.mkdir(opath)
        ofname = os.path.join(opath, fname)
        dfname = os.path.join(opath, fname)
        wr_tod2fits_noise(ofname, ut, noise, nmodin, **aheaders)
        if (transferFile):
            scp_file(ofname, dfname, remove=True)

    return


def GBsim_hpc_parallel_time(
        t1='2019-04-01T00:00:00', t2='2019-04-08T00:00:00', 
        dtsec=600, fsample=10, mapname='cmb_rseed42.fits', 
        nmodin=None, fprefix='GBtod_CMB', outpath=None,
        nside_hitmap=False, nproc=8
    ):
    log = setLogger(mp.current_process().name)

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
                          args=(t1_, t2_, fsample, mapname, nmodin, 
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
        nmodin=None, fprefix='GBtod_noise', outpath=None, nproc=8
    ):

    log = setLogger(mp.current_process().name)

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
                                rseeds[i], nmodin, fprefix, outpath))
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
    t1 = '2018-12-04T12:00:00.0' 
    t2 = '2018-12-04T12:10:00.0' # +1 min 
    #t2 = '2018-12-04T12:10:00.0' # +10 min 
    #t2 = '2018-12-04T18:00:00.0' # +6 hr
    #t2 = '2018-12-05T12:00:00.0' # +24 hr
    
    #sim_obs_singlepix(t1, t2, fsample=10)
    sim_obs_focalplane(t1, t2, fsample=10)


def test_tod():
    log = setLogger(funcname())
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


if __name__=='__main__':
    #paratest()
    GBsim_hpc_parallel_time()
    #GBsim_every10min_hpc()

