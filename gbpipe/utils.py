from __future__ import print_function
import os, sys
import socket
import logging
import datetime
import numpy as np
import healpy as hp
import multiprocessing as mp

try:
    import colorlog
except ImportError:
    pass

CRED = '\033[1;31m'
CGRE = '\033[1;32m'
CYEL = '\033[1;33m'
CBLU = '\033[1;34m'
CEND = '\033[0m'


def setLogger(name=None, logconsole=True, logfname=None):
    if name is None:
        name = mp.current_process().name

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        #logger.warning('The logger {} already exists.'.format(logger.name)) 
        pass
    else:
        logger.setLevel(logging.DEBUG)
        #logger.setLevel(logging.INFO)
        logfmt = ' [%(levelname)-8s] %(asctime)s {0} [%(name)s][%(funcName)s] %(message)s'.format(socket.gethostname())
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

            fc = colorlog.ColoredFormatter(cfmt, datefmt, log_colors=log_colors)
            ch.setFormatter(fc)
        else:
            ch.setFormatter(f)

        if (logfname is None):
            logfname = './log/tmp.log'

        fh = logging.FileHandler(logfname)
        fh.setFormatter(f)
        logger.addHandler(fh)

        if logconsole:
            logger.addHandler(ch)

    return logger


def print_warning(*msg):
    print (CYEL+'[WARNING]', *msg, end='')
    print (CEND)


def print_error(*msg):
    print (CRED+'[ERROR  ]', *msg, end='')
    print (CEND)


def print_msg(*msg):
    print (CGRE+'[MESSAGE]', *msg, end='')
    print (CEND)

print_message = print_msg


def print_debug(*msg):
    print (CBLU+'[DEBUG]', *msg, end='')
    print (CEND)


def arrdim(arr):
    return len(np.shape(arr))


def dl2cl(dls):
    if (arrdim(dls)==1):
        cls = dls.copy()
        ell = np.arange(len(cls))
        cls[1:] = cls[1:] * (2. * np.pi) / (ell[1:] * (ell[1:] + 1))
        return cls
    elif (arrdim(dls)==2):
        if (len(dls) < 6):
            cls = dls.copy()
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) / (ell[1:] * (ell[1:]+1))
            return cls
        else:
            cls = np.transpose(dls.copy())
            ell = np.arange(len(cls[0]))
            for i in range(len(cls)):
                cls[i][1:] = cls[i][1:] * (2. * np.pi) / (ell[1:] * (ell[1:]+1))
            return np.transpose(cls)
        

def cl2dl(cls):
    if (arrdim(cls)==1):
        dls = cls.copy()
        ell = np.arange(len(dls))
        dls[1:] = dls[1:] / (2. * np.pi) * (ell[1:] * (ell[1:] + 1))
        return dls
    elif (arrdim(cls)==2):
        if (len(cls) < 7):
            dls = cls.copy()
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) * (ell[1:] * (ell[1:]+1))
            return dls
        else:
            dls = np.transpose(cls.copy())
            ell = np.arange(len(dls[0]))
            for i in range(len(dls)):
                dls[i][1:] = dls[i][1:] / (2. * np.pi) * (ell[1:] * (ell[1:]+1))
            return np.transpose(dls)


def variation_cl(cls):
    if (arrdim(cls)==1):
        ell = np.arange(len(cls))
        var = np.sum((2*ell+1)*cls/4./np.pi) 
    else:
        var = []
        ell = np.arange(len(cls[0]))
        for i in range(len(cls)):
            var.append(np.sum((2*ell+1)*cls[i]/4./np.pi))

    return var


def today():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def funcname():
    return sys._getframe(1).f_code.co_name


def processname():
    return mp.current_process().name

