""" This file is part of gbpipe.

The gbpipe is a package for GroundBIRD data processing.
It provides observation direction calculation functions.
"""

import os
import sys
import time

import numpy as np
import healpy as hp
import astropy
from astropy.time import Time
from scipy.interpolate import interp1d

DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)

from .gbparam import GBparam 
from .utils import set_logger

if sys.version_info < (3,):
    range = xrange


## time conversions
def unixtime2jd(unixtime): 
    """ Convert unixtime to Julian day using astropy.time.

    Parameters
    ----------
    unixtime : float 
        Unixtime.

    Returns 
    -------
    jd : float
        Julian day.
    """
    t = Time(unixtime, format='unix') 
    jd = t.format('jd')
    return jd


def unixtime2lst(unixtime, lon=GBparam.lon, deg=True):
    """ Convert unixtime to local sidereal time using astropy.time.

    Parameters
    ----------
    unixtime : float 
        Unixtime.
    lon : float 
        Longitude in degree.
        Default is GBparam.lon.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns 
    -------
    lst : float
        Local sidereal time in degree or radian.
    """
    t = Time(unixtime, format='unix')
    
    if (deg==True):
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').degree
    else: #returns LST in hourangle
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').value

    return lst


def unixtime2lst_1s(unixtime, lon=GBparam.lon, deg=True):
    """ Convert unixtime to local sidereal time using astropy.time.
    Assuming that the rotation speed of the Earth is constant in 1s, 
    interpolating the local sidereal time.

    Parameters
    ----------
    unixtime : float 
        Unixtime.
    lon : float 
        Longitude in degree.
        Default is GBparam.lon.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns 
    -------
    lst : float
        Local sidereal time in degree or radian.
    """
    if not hasattr(unixtime, '__len__'):
        return unixtime2lst(unixtime, lon=lon, deg=deg)

    ut_min = min(unixtime)
    ut_max = max(unixtime)+1.0

    ut_1s = ut_min + np.arange(0, int(ut_max-ut_min), 1.0)
    lst_1s = unixtime2lst(ut_1s, lon=lon, deg=deg) 
    f = interp1d(ut_1s, lst_1s, fill_value='extrapolate')
    lst = f(unixtime)

    return lst


def unixtime2lst_linear(unixtime, lon=GBparam.lon, deg=True):
    """ Convert unixtime to local sidereal time using astropy.time.
    Assuming that the rotation speed of the Earth is constant in whole 
    time window, interpolating the local sidereal time.

    Parameters
    ----------
    unixtime : float 
        Unixtime.
    lon : float 
        Longitude in degree.
        Default is GBparam.lon.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns 
    -------
    lst : float
        Local sidereal time in degree or radian.
    """
    ut0 = unixtime[0] 
    lst0 = unixtime2lst(ut0, lon=lon, deg=deg) 

    sidereal_day = 86164.091
    if deg:
        rot_earth = 360.0/sidereal_day
    else:
        rot_earth = 2*np.pi/sidereal_day
    #f = interp1d(ut_lin, lst_lin, fill_value='extrapolate')
    f = lambda t: rot_earth * t 
    lst = f(unixtime) + lst0
    if deg:
        lst = lst % 360
    else:
        lst = lst % (2*np.pi)

    return lst


def unixtime2lst_fast(unixtime, Nds=1000, lon=GBparam.lon, deg=True):
    """ Convert unixtime to local sidereal time using astropy.time.
    Down samples the unixtime and assume that the Earth rotation is constant
    between each samples

    Parameters
    ----------
    unixtime : float 
        Unixtime.
    lon : float 
        Longitude in degree.
        Default is GBparam.lon.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns 
    -------
    lst : float
        Local sidereal time in degree or radian.
    """
    ut_ds = unixtime[::Nds]
    lst_ds = unixtime2lst(ut_ds, lon=lon, deg=deg) 
    f = interp1d(ut_ds, lst_ds, fill_value='extrapolate')
    lst = f(unixtime)

    return lst


def jd2lst(jd, lon=GBparam.lon, deg=True):
    """ Convert Julian day to local sidereal time using astropy.time.

    Parameters
    ----------
    jd : float 
        Julian day.
    lon : float 
        Longitude in degree.
        Default is GBparam.lon.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns 
    -------
    lst : float
        Local sidereal time in degree or radian.
    """
    t = Time(jd, format='jd')
    if (deg==True):
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').degree
    else: #returns LST in hourangle
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').value

    return lst


## encoder
def encoder2ang(enc, enc_south=GBparam.encoder_south, deg=True): 
    """ Calculate the azimuth angle of the telescope from the encoder value.

    Parameters
    ----------
    enc : int or int array
        Encoder value. 
    enc_south : int
        Encoder value for the South.
        Default is GBparam.encoder_south.
    deg : bool
        If True, the result will be in degrees, or radians otherwise. 
        Default is True.

    Returns
    -------
    ang : float or float array
        Azimuth angles of the telescope in degree or radian.
    """
    ang = 360.0/8192*(np.array(enc) - enc_south)

    #np.place(ang, ang<0, ang+360)
    ang[ang<0] += 360
    if not deg:
        ang = np.radians(ang) 

    return ang


## coordinate ang angle tools
def theta_coord(angles):
    """ Get theta axis given a direction for parallactic angle calculation.

    Parameters
    ----------
    angles : float array 
        (theta, phi) angles or a sequence of (theta, phi).

    Returns
    -------
    theta_axis : vector or vector array
        theta axis or axes given a direction or directions.
    """
    theta = angles[0]
    phi = angles[1]
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    theta_axis = np.array([cost * cosp, cost * sinp, -sint])
    phi_axis = np.array([-sinp, cosp, 0.0])
    return theta_axis #, phi_axis


def angle_from_meridian_2D(r, v):
    """ Calculate the angle from the meridian.

    Parameters
    ----------
    r : vector or vector array
        Directional vectors
    v : vector or vector array
        vectors to be tested. 

    Returns
    -------
    psi : float or floar array
        angles from the meridians.
    """

    r=np.array(r)
    v=np.array(v)

    if v.shape[0] == 3:
        r = r.T
        v = v.T

    theta, phi = hp.vec2ang(r) 
    e_theta = np.array((np.cos(theta)*np.cos(phi), 
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta))).T

    if (len(e_theta) == 1):
        e_theta = e_theta.flatten()

    ecv = np.cross(e_theta, v)

    r = np.array(r)
    v = np.array(v)
    if hasattr(r[0], '__iter__') and hasattr(v, '__iter__'):
        edv = np.einsum('ij,ij->i', e_theta, v)
        edv[edv > 1] = 1.0
        edv[edv < -1] = -1.0
        sign = np.sign(np.einsum('ij,ij->i', r, ecv))
    elif hasattr(r[0], '__iter__'):
        edv = np.einsum('ij,j->i', e_theta, v)
        edv[edv > 1] = 1.0
        edv[edv < -1] = -1.0
        sign = np.sign(np.einsum('ij,j->i', r, ecv))
    elif hasattr(v[0], '__iter__'):
        edv = np.einsum('j,ij->i', e_theta, v)
        edv[edv > 1] = 1.0
        edv[edv < -1] = -1.0
        sign = np.sign(np.einsum('j,ij->i', r, ecv))
    else:
        edv = np.dot(e_theta, v)
        if edv > 1:
            edv = 1.0
        elif edv < -1:
            edv = -1.0
        sign = np.sign(np.dot(r, ecv))

    psi = np.arccos(edv) * sign

    return psi
 

def angle_from_meridian(r, v):
    """ Calculate the angle from the meridian.

    Parameters
    ----------
    r : vector array
        Directional vectors
    v : vector array
        vectors to be tested. 

    Returns
    -------
    psi : float or floar array
        angles from the meridians.
    """

    log = set_logger()
    ## r: (nsample, 3, ndetector)
    r=np.array(r) 
    ## v: (nsample, 3, ndetector)
    v=np.array(v) 

    if len(r.shape) < 3:
        return angle_from_meridian_2D(r, v)

    nsample, _, ndetector = r.shape
    shape = (nsample, ndetector)
    ## r: (nsample, ndetector, 3)
    r=r.transpose(0, 2, 1) 
    ## v: (nsample, ndetector, 3)
    v=v.transpose(0, 2, 1) 

    theta, phi = hp.vec2ang(r)
    theta = theta.reshape(shape)
    phi = phi.reshape(shape)

    ## e_theta: (3, nsample, ndetector) = meridian 
    e_theta = np.array((np.cos(theta)*np.cos(phi), 
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta)))        
    ## e_theta: (nsample, ndetector, 3) 
    e_theta = e_theta.transpose(1, 2, 0) 

    log.debug('r.shape: {}'.format(r.shape))
    log.debug('theta.shape: {}'.format(theta.shape))
    log.debug('phi.shape: {}'.format(phi.shape))
    log.debug('e_theta.shape: {}'.format(e_theta.shape))
    log.debug('v.shape: {}'.format(v.shape))

    ## ecv: (nsample, ndetector, 3)
    ecv = np.cross(e_theta, v, axisa=2, axisb=2, axisc=2)

    log.debug('ecv.shape = {}'.format(ecv.shape)) 

    ## edv: (nsample, ndetector)
    edv = np.einsum('ijk,ijk->ij', e_theta, v) 
    edv[edv > 1.0] = 1.0
    edv[edv < -1.0] = -1.0
    ## sign: (nsample, ndetector)
    sign = np.sign(np.einsum('ijk,ijk->ij', r, ecv)) 

    # psi: (nsample, ndetector)
    psi = np.arccos(edv) * sign 

    return psi 


def psi2vec(v_arr, psi):
    """ Convert the psi angles to polarization vectors. 
    The psi angle is defined w.r.t. the local theta vector 
    in spherical coordinate. 

    Parameters
    ----------
    v_arr : vector array
        Directional vectors

    psi : float array
        psi angles defined in spherical coordinate. 

    Returns
    -------
    p : vector array
        polarization vectors.
    """
    r = np.array(v_arr)
    psi = np.array(psi)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    theta, phi = hp.vec2ang(r) 

    e_theta = np.array([np.cos(theta)*np.cos(phi), 
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta)])        # meridian 
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0*phi])


    p = np.transpose(cos_psi*e_theta + sin_psi*e_phi)

    return p
      

def psi2vec_xp(v_arr, psi):
    """
    Convert the psi angle to a vector. 
    The psi angle is defined w.r.t. the x_p vector 
    in LightTools convention.

    Parameters
    ----------
    v_arr : vector array
        Directional vectors

    psi : float array
        psi angles defined w.r.t. the x_p vector.

    Returns
    -------
    p : vector array
        polarization vectors.
    """
    r = np.array(v_arr) # r: (ndetector * 3)
    psi = np.array(psi) # psi : (ndetector)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    theta, phi = hp.vec2ang(r) 

    y = np.array([0, 1, 0])
    xp = np.cross(r, y) # xp: (ndetector * 3)
    xp = (xp.T / np.linalg.norm(xp, axis=1)).T
    yp = np.cross(r, xp)

    p = np.transpose(xp.T*sin_psi + yp.T*cos_psi)
    
    return p

     
def parallactic_angle(ze, deg=True, coord=['C', 'G'], healpy=True): 
    """ 
    Calculates angle between theta axes in two coordinate systems 
    given a direction.
    
    Parameters
    ----------
    ze  : vector or an array of vectors
        Direction vector in equatorial coordinate.
    deg : bool
        If it is set True, the output angle is in degree. 
        Otherwise, the output angle is in radian 
        Default is True.
    coord : sequence of characters
        Coordinate systems. The first is source and the other is 
        destination. 
        Default is ['C', 'G'].
    healpy : bool
        If True, *healpy.Rotator.angle_ref* module will be used 
        for this calculation. Otherwise, direct calculation.
        Default is True.
    
    Returns
    --------
    psi_par : float or float array
        Parallactic angles for the given direction
    """
    R_coord = hp.Rotator(coord=coord)
    ze = np.array(ze)
    if healpy:
        if ze.shape[0] != 3:
            ze = ze.T
        psi_par = R_coord.angle_ref(ze)
    else:
        # old version
        if (hasattr(ze[0], '__iter__')):
            # direction in Gal.
            zg = np.tensordot(R_coord.mat, ze, axes=(1,1)).T 
            # theta direction in Equ. theta_p
            xp = theta_coord(hp.vec2ang(ze)).T 
            # theta_p in Gal. theta_e
            xe = np.tensordot(R_coord.mat, xp, axes=(1,1)).T 
            # theta_g
            xg = theta_coord(hp.vec2ang(zg)).T 
            # angle theta_e to theta_g
            xcx = np.cross(xe, xg)
            xdx = np.einsum('ij,ij->i', xe, xg)
            xdx[xdx > 1] = 1.0
            xdx[xdx < -1] = -1.0
            sign = -np.sign(np.einsum('ij,ij->i',zg, xcx))
            psi_par = np.arccos(xdx) * sign
        else:
            zg = R_coord(ze) 
            xp = theta_coord(hp.vec2ang(ze)).flatten()
            xe = R_coord(xp)
            xg = theta_coord(hp.vec2ang(zg)).flatten()
            xcx = np.cross(xe, xg) 
            xdx = np.dot(xe, xg) 
            if xdx > 1: xdx = 1 
            elif xdx < -1: xdx = -1 
            sign = -np.sign(np.dot(zg, xcx))
            psi_par = np.arccos(xdx) * sign

    return psi_par


## coordinate transformation between equ and gal
def coord_transform_map(m, coord, pixel=False):
    """ Coordinate transformation of a healpix map given coordinate systems
    using *healpy.Rotator.rotate_map_pixel* and 
    *healpy.Rotator.rotate_map_alms*. In general, rotation using alms is 
    more precise. For the partial maps, *healpy.Rotator.rotate_map_pixel*
    should be used.

    Parameters
    ----------
    m : array
        An healpix map to be transformed. 
    coord : sequence of characters
        Coordinate systems. The first is source and the other is 
        destination. 
        Default is ['C', 'G'].
    pixel : bool
        If True, it use *healpy.Rotator.rotate_map_pixel* module. 
        Otherwise, *healpy.Rotator.rotate_map_alms*. 
        Default is False.

    Returns
    -------
    mp : array
        Transformed healpix map.
    """
    R = hp.Rotator(coord=coord)
    if pixel:
        mp = R.rotate_map_pixel(m)
    else:
        mp = R.rotate_map_alms(m)

    return mp


def equ2gal(m_equ, healpy=True, pixel=False):
    """ Transform the coordinate of a healpix map in Equatorial 
    coordinate to Galactic coordinate. 

    Parameters
    ----------
    m_equ : array 
        A healpix map or TQU maps in Equatorial coordinate. 
    healpy : bool
        If True, healpy routines are used. 
        Otherwise, direct calculation.
        Default is True.
    pixel : bool
        If True, it use *healpy.Rotator.rotate_map_pixel* module. 
        Otherwise, *healpy.Rotator.rotate_map_alms*. 
        This is ignored when healpy is False. 
        Default is False.

    Returns
    -------
    m_gal : array
        A healpix map or TQU maps in Galactic coordinate. 
    """
    if healpy:
        m_gal = coord_transform_map(m_equ, coord=['C', 'G'], pixel=pixel)
    else:
        ## old version
        npix = len(m_equ)

        if npix == 3:
            m_gal = equ2gal_pol(m_equ)

        nside = hp.npix2nside(npix) 

        m_gal = np.full(npix, hp.UNSEEN)
        r = hp.Rotator(coord=['G', 'C'])
        n_gal = np.arange(npix)

        v_gal = hp.pix2vec(nside, n_gal)
        v_equ = r(v_gal)
        n_equ = hp.vec2pix(nside, v_equ[0], v_equ[1], v_equ[2])
        m_gal[n_gal] = m_equ[n_equ]
        
    return m_gal


def gal2equ(m_gal, healpy=True, pixel=False):
    """ Transform the coordinate of a healpix map in Galactic
    coordinate to Equatorial coordinate. 

    Parameters
    ----------
    m_gal : array
        A healpix map or TQU maps in Galactic coordinate. 
    healpy : bool
        If True, healpy routines are used. 
        Otherwise, direct calculation.
        Default is True.
    pixel : bool
        If True, it use *healpy.Rotator.rotate_map_pixel* module. 
        Otherwise, *healpy.Rotator.rotate_map_alms*. 
        This is ignored when healpy is False. 
        Default is False.

    Returns
    -------
    m_equ : array 
        A healpix map or TQU maps in Equatorial coordinate. 
    """
    if healpy:
        m_equ = coord_transform_map(m_gal, coord=['G', 'C'], pixel=pixel)
    else:
        ## old version
        npix = len(m_gal)

        if npix == 3:
            m_equ = gal2equ_pol(m_gal)

        nside = hp.npix2nside(npix) 

        m_equ = np.full(npix, hp.UNSEEN)
        r = hp.Rotator(coord=['C', 'G'])
        n_equ = np.arange(npix)

        v_equ = hp.pix2vec(nside, n_equ)
        v_gal = r(v_equ)
        n_gal = hp.vec2pix(nside, v_gal[0], v_gal[1], v_gal[2])
        m_equ[n_equ] = m_gal[n_gal]
            
    return m_equ


def equ2gal_pol(m_equ):
    """ Transform the coordinate of healpix map in Equatorial 
    coordinate to Galactic coordinate by direct calculations.
    Using *equ2gal* with *healpy=True* is recommended. 

    Parameters
    ----------
    m_equ : array 
        Healpix TQU maps in Equatorial coordinate. 

    Returns
    -------
    m_gal : array
        Healpix TQU maps in Galactic coordinate. 
    """
    T_equ = m_equ[0]
    Q_equ = m_equ[1]
    U_equ = m_equ[2]

    T_gal = equ2gal(T_equ)
    Q_gal = equ2gal(Q_equ)
    U_gal = equ2gal(U_equ)

    npix = len(T_gal)
    nside = hp.npix2nside(npix)

    pix = np.arange(npix)
    v_gal = hp.pix2vec(nside, pix)
    psi_p = parallactic_angle(v_gal.T)

    m_gal = [T_gal, Q_gal, U_gal]

    return m_gal


def gal2equ_pol(m_gal):
    """ Transform the coordinate of healpix map in Galactic 
    coordinate to Equatorial coordinate by direct calculations.
    Using *gal2equ* with *healpy=True* is recommended. 

    Parameters
    ----------
    m_gal : array
        Healpix TQU maps in Galactic coordinate. 

    Returns
    -------
    m_equ : array 
        Healpix TQU maps in Equatorial coordinate. 
    """
    T_gal = m_gal[0]
    Q_gal = m_gal[1]
    U_gal = m_gal[2]

    T_equ = gal2equ(T_gal)
    Q_equ = gal2equ(Q_gal)
    U_equ = gal2equ(U_gal)
    
    npix = len(T_equ)
    nside = hp.npix2nside(npix)

    pix = np.arange(npix)
    v_equ = np.array(hp.pix2vec(nside, pix)).T
    psi_p = parallactic_angle(v_equ, coord=['C','G'])
        
    Q = Q_equ * np.cos(2*psi_p) - U_equ * np.sin(2*psi_p)
    U = Q_equ * np.sin(2*psi_p) + U_equ * np.cos(2*psi_p)

    m_equ = [T_equ, Q, U]

    return m_equ


## Rotational matrices
def euler_ZYZ(angles, deg=True, new=False): 
    """ Calculates rotation matrix according to the wikipedia convention 
    (extrinsic z-y-z).

    Parameters
    ----------
    angles : array of (3 * float)
        Euler angles (alpha, beta, gamma). 
    deg : bool
        If True, the inputs are in degrees, or radians otherwise. 
        Default is True.
    new : bool
        If True, the angles are in convention of euler_matrix_new 
        of healpix, (gamma, beta, alpha), for a consistency with 
        healpy Rotator class. Otherwise, (alpha, beta, gamma).

    Returns
    -------
    R : array of 3x3 matrices
        Rotation matrices given the Euler angles. 
    """

    if new:
        psi, theta, phi = angles # gamma beta alpha
    else:
        phi, theta, psi = angles # alpha beta gamma

    if deg:
        phi = np.radians(phi)
        theta = np.radians(theta)
        psi = np.radians(psi)

    len_phi   = phi.size
    len_theta = theta.size
    len_psi   = psi.size
    len_arr   = max((len_phi, len_theta, len_psi))
    
    if (len_arr > 1):
        if (len_phi == 1):
            phi = np.array([phi] * len_arr)
        if (len_theta == 1):
            theta = np.array([theta] * len_arr)
        if (len_psi == 1): 
            psi = np.array([psi] * len_arr)

    c1 = np.cos(phi)
    c2 = np.cos(theta)
    c3 = np.cos(psi)
    s1 = np.sin(phi)
    s2 = np.sin(theta)
    s3 = np.sin(psi)

    Rtmp = np.array([[ c1*c2*c3-s1*s3,  -c3*s1-c1*c2*s3, c1*s2], 
                     [ c1*s3+c2*c3*s1,   c1*c3-c2*s1*s3, s1*s2], 
                     [-c3*s2,            s2*s3,          c2   ]])

    if (len(Rtmp.shape)==3):
        R = np.transpose(Rtmp, (2, 0, 1))
    else:
        R = Rtmp

    return R


def Rot_matrix_healpix(el=GBparam.el, az=0, 
                       lat=GBparam.lat, lst=0, 
                       psi=0, coord='C'): 
    """Calculates the rotation matrix of telescope by healpy routines.
    All the angles are in degree. The input angles cannot be arrays.

    Parameters
    ----------
    el : float or float array
        Elevation of the telescope in Horizontal coordinate, in degree. 
        el = 90 - (tilt angle).
        Default is GBparam.El.
    az : float or float array
        Azimuth angle in Horizontal coordinate.
        Default is 0.
    lat : float or float array
        Latitude of observation site.
        Default is GBparam.lat.
    lst : float or float array
        Right ascension (or hour angle) in Local Sidereal Time in degree.
        Default is 0.
    psi : float or float array
        Roll angles of the telescope.
        Default is 0.
    coord : 'G', 'C' or 'E'
        Result coordinate system. 'G' for Galactic coordinate, 
        'C' for Equatorial coordinate, or 'H' for Horizontal coordinate. 
        Default is 'C'.

    Returns
    -------
    rmat : a 3x3 matrix or an array of 3x3 matrices
        Rotation matrices.     
    """

    ## rotation of GB w.r.t. ground
    r1 = hp.Rotator((psi, 90.-el, 180.-az), eulertype='Y', deg=True) 
    if (coord == 'H'):
        rmat = r1.mat
    else: 
        ## Horizontal to Equatorial coordinate. 
        r2 = hp.Rotator((0, 90.-lat, lst), eulertype='Y', deg=True) 
        rmat = np.matmul(r2.mat, r1.mat)
        if (coord == 'G'):
            R_E2G = hp.Rotator(coord=['C', 'G'])
            rmat = np.matmul(R_E2G.mat, rmat)

    return rmat


def Rot_matrix(el=GBparam.el, az=0, 
               lat=GBparam.lat, lst=0, 
               psi=0, coord='C'): 
    """Computes rotation matrix with euler_ZYZ routine.
    All the angles are in degree and can be arrays. 

    Parameters
    ----------
    el : float or float array
        Elevation of the telescope in Horizontal coordinate, in degree. 
        el = 90 - (tilt angle).
        Default is GBparam.El.
    az : float or float array
        Azimuth angle in Horizontal coordinate.
        Default is 0.
    lat : float or float array
        Latitude of observation site.
        Default is GBparam.lat.
    lst : float or float array
        Right ascension (or hour angle) in Local Sidereal Time in degree.
        Default is 0.
    psi : float or float array
        Roll angles of the telescope.
        Default is 0.
    coord : 'G', 'C' or 'H'
        Result coordinate system. 'G' for Galactic coordinate, 
        'C' for Equatorial coordinate, or 'H' for Horizontal coordinate. 
        Default is 'C'.

    Returns
    -------
    rmat : a 3x3 matrix or an array of 3x3 matrices
        Rotation matrices.     
    """

    el = np.array(el)
    az = np.array(az)
    lat = np.array(lat)
    lst = np.array(lst)
    psi = np.array(psi)

    r1 = euler_ZYZ((psi, 90.-el, 180.-az), deg=True, new=True) # horizontal coordinate - azimuth from the north.
    if (coord =='H'):
        rmat = r1
    else:
        r2 = euler_ZYZ((0, 90.-lat, lst), deg=True, new=True)
        rmat = np.matmul(r2, r1)
        if (coord == 'G'):
            R_E2G = hp.Rotator(coord=['C', 'G'])
            r3mat = [R_E2G.mat]
            rmat = np.matmul(r3mat, rmat)

    return rmat


def Rot_matrix_equatorial(ra, dec, psi=0, coord='C', deg=True): 
    """Computes rotation matrix with from ra, dec, and psi. 
    All the angles can be arrays. 

    Parameters
    ----------
    ra : float or float array
        Right ascension.
        Default is 0.
    dec : float or float array
        Declination.
        Default is 0.
    psi : float or float array
        Roll angles of the telescope.
        Default is 0.
    coord : 'G', 'C' or 'H'
        Result coordinate system. 'G' for Galactic coordinate, 
        'C' for Equatorial coordinate.
        Default is 'C'.
    deg : bool.
        If True, angles are in degrees.

    Returns
    -------
    rmat : a 3x3 matrix or an array of 3x3 matrices
        Rotation matrices.     
    """

    phi = np.array(ra)
    psi = np.array(psi)
    if deg:
        theta = 90.0 - np.array(dec)
    else:
        theta = np.pi/2 - np.array(dec)

    r1 = euler_ZYZ((psi, theta, phi), deg=deg, new=True) # horizontal coordinate - azimuth from the north.
    if (coord =='C'):
        rmat = r1
    elif (coord == 'G'):
        R_E2G = hp.Rotator(coord=['C', 'G'])
        r2mat = [R_E2G.mat]
        rmat = np.matmul(r2mat, rmat)

    return rmat


def Rotate(v_arr, rmat=None):
    """ Rotate vectors with rotation matrices.

    Parameters
    ----------
    v_arr : vector or vector array
        Vectors to be rotated
    rmat : 3x3 matrices or array of them. 
        Rotation matrices. If None, it is calculated with 
        Rot_matrix() with default values.
        Default is None.

    Return
    ------
    vp_arr : vector or vector array
        Rotated vectors. 
    """
    log = set_logger()
    if (rmat is None):
        rmat = Rot_matrix()

    log.debug('v_arr.shape: {}'.format(np.array(v_arr).shape))
    log.debug('v_arr.T.shape: {}'.format(np.array(v_arr).T.shape))
    log.debug('rmat.shape: {}'.format(rmat.shape))
    vp_arr = np.matmul(rmat, np.array(v_arr).T)
    parangle = 0

    #if (len(vp_arr[0])!=1):
    #    vp_arr = vp_arr.T
    #else:
    #    vp_arr = np.ravel(vp_arr.T)

    return vp_arr 


## angles from rotational matrices
def rmat2euler(rmat, deg=True):
    """ returns euler angles (ZYZ convention) for the rotation matrices.
    http://www.gregslabaugh.net/publications/euler.pdf.
    
    Parameters
    ----------
    rmat : 3x3 matrix or array of them
        Rotational matrices.

    Returns
    -------
    phi : float or float array
        phi (alpha or yaw) angles from the rotation matrices.    
    theta : float or float array
        theta (beta or pitch) angles from the rotation matrices.    
    psi : float or float array
        psi (gamma or roll) angles from the rotation matrices.    
    """

    if len(np.array(rmat).shape) == 2:
        theta = np.arctan2(np.sqrt(rmat[0,2]**2+rmat[1,2]**2), rmat[2,2])
        phi = np.arctan2(rmat[1,2], rmat[0,2])
        psi = np.arctan2(rmat[2,1], -rmat[2,0])
    elif len(np.array(rmat).shape) == 3:
        theta = np.arctan2(np.sqrt(rmat[:,0,2]**2+rmat[:,1,2]**2), rmat[:,2,2])
        phi = np.arctan2(rmat[:,1,2], rmat[:,0,2])
        psi = np.arctan2(rmat[:,2,1], -rmat[:,2,0])
    else:
        print('Invalid input matrix. Input should be 3x3 matrix or array of 3x3 matrices.')

    if deg:
        phi, theta, psi = np.degrees((phi, theta, psi))

    return phi, theta, psi


def rmat2equatorial(rmat, deg=True):
    """ returns angles in equatorial coordinate for the rotation matrices (ZYZ).
    http://www.gregslabaugh.net/publications/euler.pdf.
    
    Parameters
    ----------
    rmat : 3x3 matrix or array of them
        Rotational matrices.

    Returns
    -------
    ra : float or float array
        right ascension in equatorial coordinate.  
    dec : float or float array
        declination in equatorial coordinate. 
    psi : float or float array
        psi (gamma or roll) angles.
    """

    if len(np.array(rmat).shape) == 2:
        dec = np.arctan2(np.sqrt(rmat[0,2]**2+rmat[1,2]**2), rmat[2,2])
        ra = np.arctan2(rmat[1,2], rmat[0,2])
        psi = np.arctan2(rmat[2,1], -rmat[2,0])
    elif len(np.array(rmat).shape) == 3:
        dec = np.arctan2(np.sqrt(rmat[:,0,2]**2+rmat[:,1,2]**2), rmat[:,2,2])
        ra = np.arctan2(rmat[:,1,2], rmat[:,0,2])
        psi = np.arctan2(rmat[:,2,1], -rmat[:,2,0])
    else:
        print('Invalid input matrix. Input should be 3x3 matrix or array of 3x3 matrices.')

    dec = np.pi/2 - dec
        
    if deg:
        ra = np.degrees(ra)
        dec = np.degrees(dec)
        psi = np.degrees(psi)


    return ra, dec, psi



