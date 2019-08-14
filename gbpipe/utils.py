""" This a part of gbpipe package. 
Tools for gbpipe.
"""
from __future__ import print_function
import os, sys
import socket
import logging
import datetime

import multiprocessing as mp

import numpy as np
import healpy as hp

try:
    import colorlog
except ImportError:
    pass

CRED = '\033[1;31m'
CGRE = '\033[1;32m'
CYEL = '\033[1;33m'
CBLU = '\033[1;34m'
CEND = '\033[0m'


def set_logger(name=mp.current_process().name, display=True, 
               filename=None, level='INFO'):
    """ Setting up logger. 

    Parameters
    ----------
    name : string
        The name of the logger.
        Default : current process name
    display : bool
        If True, the log is displayed on the console. 
        Default : True
    filename : string
        The name of log file. 
        If None, the log will not be written on a file. 
        Default : None
    level : string
        Log level. 'INFO', 'DEBUG' or 'WARNING'
        Default: 'INFO'

    Returns
    -------
    logger : logging.RootLogger
        Logger
    """
    #if name is None:
    #    name = mp.current_process().name

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        #logger.warning('The logger {} already exists.'.format(logger.name)) 
        pass
    else:
        if level=='INFO':
            logger.setLevel(logging.INFO)
        elif level=='DEBUG':
            logger.setLevel(logging.DEBUG)
        elif level=='WARNING':
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        logfmt = ' [%(levelname)-8s] %(asctime)s {0} [%(name)s][%(funcName)s] %(message)s'.format(hostname())
        datefmt = '%b %d %H:%M:%S'
        f = logging.Formatter(logfmt, datefmt)

        ch = logging.StreamHandler()
        if 'colorlog' in sys.modules and os.isatty(2):
            cfmt = '%(log_color)s' + logfmt
            log_colors = {'DEBUG' : 'bold_blue', 
                          'INFO' : 'white',
                          'WARNING' : 'bold_yellow',
                          'ERROR': 'bold_red',
                          'CRITICAL': 'bold_red'}

            fc = colorlog.ColoredFormatter(cfmt, datefmt, 
                                           log_colors=log_colors)
            ch.setFormatter(fc)
        else:
            ch.setFormatter(f)

        if (filename is None):
            t = datetime.datetime.now().date().isoformat()
            filename = './log_name_{}.log'.format(t)

        fh = logging.FileHandler(filename)
        fh.setFormatter(f)
        logger.addHandler(fh)

        if display:
            logger.addHandler(ch)

    return logger


def print_warning(*msg):
    """ print warning in yellow color.

    Parameters
    ----------
    msg : string or sequence of strings. 
        Message.
    """
    print (CYEL+'[WARNING]', *msg, end='')
    print (CEND)


def print_error(*msg):
    """ print error in red color.

    Parameters
    ----------
    msg : string or sequence of strings. 
        Message.
    """
    print (CRED+'[ERROR  ]', *msg, end='')
    print (CEND)


def print_msg(*msg):
    """ print message in green color.

    Parameters
    ----------
    msg : string or sequence of strings. 
        Message.
    """
    print (CGRE+'[MESSAGE]', *msg, end='')
    print (CEND)

print_message = print_msg


def print_debug(*msg):
    """ print message in blue color.

    Parameters
    ----------
    msg : string or sequence of strings. 
        Message.
    """
    print (CBLU+'[DEBUG]', *msg, end='')
    print (CEND)


def arr_rank(arr):
    """ Rank of the array

    Parameter
    ---------
    arr : array
        Array.

    Returns
    -------
    rank : int
        Rank of the array.
    """
    rank = len(np.shape(arr))

    return rank


def dl2cl(dls):
    """ Convert the angular spectrum D_l to C_l.
    C_l = D_l * 2 * np.pi / l / (l+1)
    
    Parameters
    ----------
    dls : array
        Angular spectrum, D_l, to be converted. 

    Returns
    -------
    cls : array
        Converted array.
    """
    if (arr_rank(dls)==1):
        cls = dls.copy()
        ell = np.arange(len(cls))
        cls[1:] = cls[1:] * (2. * np.pi) / (ell[1:] * (ell[1:] + 1))
    elif (arr_rank(dls)==2):
        if (len(dls) < 10):
            cls = dls.copy()
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) \
                             / (ell[1:] * (ell[1:]+1))
        else:
            cls = np.transpose(dls.copy())
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) \
                             / (ell[1:] * (ell[1:]+1))
            cls = np.transpose(cls) 
    return cls


def cl2dl(cls):
    """ Convert the angular spectrum C_l to D_l.
    D_l = C_l * l * (l+1) / 2 / pi
    
    Parameters
    ----------
    cls : array
        Angular spectrum, C_l, to be converted. 

    Returns
    -------
    dls : array
        Converted array.
    """
    if (arr_rank(cls)==1):
        dls = cls.copy()
        ell = np.arange(len(dls))
        dls[1:] = dls[1:] / (2. * np.pi) * (ell[1:] * (ell[1:] + 1))
    elif (arr_rank(cls)==2):
        if (len(cls) < 10):
            dls = cls.copy()
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) \
                             * (ell[1:] * (ell[1:]+1))
        else:
            dls = np.transpose(cls.copy())
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) \
                             * (ell[1:] * (ell[1:]+1))
            dls = np.transpose(dls) 

    return dls


def variation_cl(cls):
    """ Variation of angular spectrum C_l

    Parameters
    ----------
    cls : array
        Angular power spectrum.

    Returns
    -------
    var : float
        variation of the angular power spectrum.
    """
    if (arr_rank(cls)==1):
        ell = np.arange(len(cls))
        var = np.sum((2*ell+1)*cls/4./np.pi) 
    else:
        var = []
        ell = np.arange(len(cls[0]))
        for i in range(len(cls)):
            var.append(np.sum((2*ell+1)*cls[i]/4./np.pi))

    return var


def today():
    """ Returns the date of today in string."""
    return datetime.datetime.now().strftime('%Y-%m-%d')


def function_name():
    """ Returns the name of the current function."""
    return sys._getframe(1).f_code.co_name


def process_name():
    """ Returns the name of the current process."""
    return mp.current_process().name


def hostname():
    """ Returns the hostname."""
    return socket.gethostname()


def qu2ippsi(Q, U):
    """ Convert Q/U maps to intensity and psi maps.
    
    Parameters
    ----------
    Q : float array
        Map of stokes parameter Q
    U : float array
        Map of stokes parameter Q
    
    Returns
    -------
    Ip : float array
        Polarization intensity map.
    psi : float array
        Map of psi angles. 
    """
    Ip = np.sqrt(Q**2 + U**2)
    psi = 0.5 * np.arctan2(U, Q)

    return Ip, psi


def ippsi2qu(Ip, psi):
    """ Convert intensity and psi maps to QU maps

    Parameters
    ----------
    Ip : float array
        Polarization intensity map.
    psi : float array
        Map of psi angles. 
    
    Returns
    -------
    Q : float array
        Map of stokes parameter Q
    U : float array
        Map of stokes parameter Q
    """
    Q = Ip * np.cos(2*psi)
    U = Ip * np.sin(2*psi)

    return Q, U


