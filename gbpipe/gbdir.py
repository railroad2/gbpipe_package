#
# This file is part of GBpipe.
#
# GBpipe is a package for GroundBIRD data processing.
#
# It provides observation direction calculation functions.

import healpy as hp
import numpy  as np
import os, sys
import time
import astropy
from astropy.time import Time

DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)

from gbparam import GBparam 
from utils import setLogger, funcname, processname

if sys.version_info < (3,):
    range = xrange


def unixtime2JD(unixtime): # unixtime (float)
    t = Time(unixtime, format='unix') 
    t.format('jd')
    return t.value


def unixtime2LST(unixtime, lon=GBparam.lon, deg=True):
    t = Time(unixtime, format='unix')
    
    if (deg==True):
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').degree
    else: #returns LST in hourangle
        lst = t.sidereal_time('apparent', longitude=str(lon)+'d').value

    return lst


def JD2LST(jd, lon=GBparam.lon, deg=True):
    t = Time(jd, format='jd')
    if (deg==True):
        return t.sidereal_time('apparent', longitude=str(lon)+'d').degree
    else: #returns LST in hourangle
        return t.sidereal_time('apparent', longitude=str(lon)+'d').value


def encoder2ang(enc, enc_south=GBparam.encoder_south): # encoder value to angle
    deg = 360.0/8192*(np.array(enc) - enc_south)
    if (not hasattr(deg, '__iter__')):
        deg = [deg]

    np.place(deg, deg<0, deg+360)

    return deg


def euler_ZYZ(angles, deg=True): 
    """ Calculates rotation matrix according to the wikipedia convention (extrinsic z-y-z) """
    if (deg):
        phi   = np.array(np.radians(angles[2])) # alpha
        theta = np.array(np.radians(angles[1])) # beta
        psi   = np.array(np.radians(angles[0])) # gamma 
    else:
        phi   = np.array(angles[2]) # alpha
        theta = np.array(angles[1]) # beta
        psi   = np.array(angles[0]) # gamma 

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


def x_coord(angles):
    """ Get x (theta) axis given a direction for parallactic angle calculation """
    theta= angles[0]
    phi  = angles[1]
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    x_ = np.array([cost * cosp, cost * sinp, -sint])
    #y_ = np.array([-sinp, cosp, 0.0])
    return x_ #, y_


def angle_from_meridian_2D(r, v):
    """
    """
    r=np.array(r)
    v=np.array(v)
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
        edv[np.where(edv > 1)] = 1.0
        sign = np.sign(np.einsum('ij,ij->i', r, ecv))
    elif hasattr(r[0], '__iter__'):
        edv = np.einsum('ij,j->i', e_theta, v)
        edv[np.where(edv > 1)] = 1.0
        sign = np.sign(np.einsum('ij,j->i', r, ecv))
    elif hasattr(v[0], '__iter__'):
        edv = np.einsum('j,ij->i', e_theta, v)
        edv[np.where(edv > 1)] = 1.0
        sign = np.sign(np.einsum('j,ij->i', r, ecv))
    else:
        edv = np.dot(e_theta, v)
        if edv > 1:
            edv = 1.0
        sign = np.sign(np.dot(r, ecv))

    psi = np.arccos(edv) * sign

    return psi
 

def angle_from_meridian(r, v):
    """
    """
    log = setLogger()
    r=np.array(r) # r: (nsample, 3, ndetector)
    v=np.array(v) # v: (nsample, 3, ndetector)
    nsample, _, ndetector = r.shape
    shape = (nsample, ndetector)
    r=r.transpose(0, 2, 1) # r: (nsample, ndetector, 3)
    v=v.transpose(0, 2, 1) # v: (nsample, ndetector, 3)

    theta, phi = hp.vec2ang(r)

    theta = theta.reshape(shape)
    phi = phi.reshape(shape)

    # e_theta: (3, nsample, ndetector)
    e_theta = np.array((np.cos(theta)*np.cos(phi), 
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta)))        # meridian 
    e_theta = e_theta.transpose(1, 2, 0) # e_theta: (nsample, ndetector, 3) 

    log.debug('r.shape: {}'.format(r.shape))
    log.debug('theta.shape: {}'.format(theta.shape))
    log.debug('phi.shape: {}'.format(phi.shape))
    log.debug('e_theta.shape: {}'.format(e_theta.shape))
    log.debug('v.shape: {}'.format(v.shape))

    ecv = np.cross(e_theta, v, axisa=2, axisb=2, axisc=2)

    log.debug('ecv.shape = {}'.format(ecv.shape)) # ecv: (nsample, ndetector, 3)
    
    edv = np.einsum('ijk,ijk->ij', e_theta, v) # edv: (nsample, ndetector)
    edv[np.where(edv > 1.0)] = 1.0
    sign = np.sign(np.einsum('ijk,ijk->ij', r, ecv)) # sign: (nsample, ndetector)

    psi = np.arccos(edv) * sign # psi: (nsample, ndetector)

    return psi 


def psi2vec(v_arr, psi):
    """
    Convert the psi angle to a vector. 
    The psi angle is defined w.r.t. the local theta vector in spherical coordinate. 
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
    The psi angle is defined w.r.t. the x_p vector in LightTools convention.
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
    Calculates angle between theta axes given a direction in Equatorial and Galactic coordinates.
    *numpy.einsum()* functions in some products are replaced with numpy.tensordot() to improve speed. 

    Parameters
    ----------
    ze  : float or float array
        Direction vector in equatorial coordinate.
    deg : bool
        If it is set 'True', the output angle is in degree. 
        Otherwise, the output angle is in radian (default: True).
    
    Returns
    --------
    psi_par : float or float array
        parallactic angle for the given direction
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
            zg = np.tensordot(R_coord.mat, ze, axes=(1,1)).T # direction in Gal.
            xp = xp_coord(hp.vec2ang(ze)).T # theta direction in Equ. theta_p
            xe = np.tensordot(R_coord.mat, xp, axes=(1,1)).T # theta_p in Gal. theta_e
            xg = xp_coord(hp.vec2ang(zg)).T # theta_g
            xcx = np.cross(xe, xg) # angle theta_e to theta_g
            xdx = np.einsum('ij,ij->i', xe, xg)
            xdx[xdx > 1] = 1.0
            xdx[xdx < -1] = -1.0
            sign = -np.sign(np.einsum('ij,ij->i',zg, xcx))
            psi_par = np.arccos(xdx) * sign
        else:
            zg = R_coord(ze) 
            xp = xp_coord(hp.vec2ang(ze)).flatten()
            xe = R_coord(xp)
            xg = xp_coord(hp.vec2ang(zg)).flatten()
            xcx = np.cross(xe, xg) 
            xdx = np.dot(xe, xg) 
            if xdx > 1: xdx = 1 
            elif xdx < -1: xdx = -1 
            sign = -np.sign(np.dot(zg, xcx))
            psi_par = np.arccos(xdx) * sign

    return psi_par


def rotate_map(m, coord, pixel=False):
    R = hp.Rotator(coord=coord)
    if pixel:
        mp = R.rotate_map_pixel(m)
    else:
        mp = R.rotate_map_alms(m)

    return mp


def equ2gal(m_equ, healpy=True, pixel=False):
    if healpy:
        R = hp.Rotator(coord=['C', 'G'])
        if pixel:
            m_gal = R.rotate_map_pixel(m_equ)
        else:
            m_gal = R.rotate_map_alms(m_gal)
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


def equ2gal_pol(m_equ):
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


def gal2equ(m_gal, healpy=True, pixel=False):
    if healpy:
        R = hp.Rotator(coord=['G', 'C'])
        if pixel:
            m_equ = R.rotate_map_pixel(m_gal)
        else:
            m_equ = R.rotate_map_alms(m_gal)
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


def gal2equ_pol(m_gal):
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

    m_gal = [T_equ, Q, U]

    return m_gal


def Rot_matrix_healpix(EL=GBparam.EL, AZ=0, LAT=GBparam.lat, LST=0, PSI=0, coord='C'): # angles in degree by default
    """Calculates the rotation matrix of GroundBIRD with healpix routine.
    Parameters
    ----------
    Returns
    -------
    """
    
    # rotation matrix calculation (default : Equatorial coordinate 'C')
    r1 = hp.Rotator( (PSI, 90.-EL, 180.-AZ), eulertype='Y', deg=True) # rotation of GB w.r.t. ground
    if (coord == 'H'):
        rmat = r1.mat
    else: 
        r2 = hp.Rotator( (0, 90.-LAT, LST), eulertype='Y', deg=True) # horizontal to equatorial coordinate. 
        rmat = np.matmul(r2.mat, r1.mat)
        if (coord == 'G'):
            R_E2G = hp.Rotator(coord=['C', 'G'])
            rmat = np.matmul(R_E2G.mat, rmat)

    return rmat


def Rot_matrix(EL=GBparam.EL, AZ=0, LAT=GBparam.lat, LST=0, PSI=0, coord='C'): # angles in degree by default
    """Computes rotation matrix with euler_ZYZ routine.

    Parameters
    ----------
    EL : float or float array
        Elevation in Horizontal coordinate. 
        Default = GBparam.El
    AZ : float or float array
        Azimuth angle in Horizontal coordinate.
        Default = 0
    LAT : float or float array
        Latitude of observation site.
        Default = GBparam.lat
    LST : float or float array
        Right ascension (or hour angle) in Local Sidereal Time.
        Default = 0
    PSI : float or float array
        Default = 0
    coord : 'G', 'C' or 'E'
        Result coordinate system. 'G' for Galactic coordinate, 
        'C' for Equatorial coordinate, or 'E' for Celestial coordinate. 
        Default = 'C'

    Returns
    -------
    rmat : a 3x3 matrix or an array of 3x3 matrices
        Rotation matrices.     
    """

    EL = np.array(EL)
    AZ = np.array(AZ)
    LAT = np.array(LAT)
    LST = np.array(LST)

    r1 = euler_ZYZ((PSI, 90.-EL, 180.-AZ), deg=True)
    if (coord =='H'):
        rmat = r1
    else:
        r2 = euler_ZYZ((0, 90.-LAT, LST), deg=True)
        rmat = np.matmul(r2, r1)
        if (coord == 'G'):
            R_E2G = hp.Rotator(coord=['C', 'G'])
            r3mat = [R_E2G.mat]
            rmat = np.matmul(r3mat, rmat)

    return rmat


def Rotate(v_arr, rmat=None):
    """ Rotate vectors with rotation matrices.

    Parameters
    ----------
    v_arr : a vector or vector array
        Vectors to be rotated
    rmat : 
        Rotation matrices. If None, it is calculated with 
        Rot_matrix() with default values.
        Default = None

    Return
    ------

    """
    log = setLogger()
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


def dir_sky(v_arr):
    rmat = Rot_matrix()
    return Rotate(v_arr, rmat)
    

def dir_pol(v_arr):
    rmat = Rot_matrix()
    return Rotate(v_arr, rmat)


def rmat2euler(rmat):
    """ returns euler angles (ZYZ convention) for the rotation matrix 
    http://www.gregslabaugh.net/publications/euler.pdf"""

    theta = np.arctan2(np.sqrt(rmat[0,2]**2+rmat[1,2]**2), rmat[2,2])
    phi = -np.arctan2(rmat[1,2], rmat[0,2])
    psi = np.arctan2(rmat[2,1], rmat[2,0])

    return (phi, theta, psi)

