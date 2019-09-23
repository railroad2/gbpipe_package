""" This is a part of gbpipe.
GBparam is a class storing parameters of GroundBIRD analysis.
"""

from __future__ import print_function
import os.path
import sys

if (sys.version_info > (3,)):
    import configparser as cparser
else:
    import ConfigParser as cparser

import numpy as np
import pylab as plt
import matplotlib.lines as mline

from astropy.utils.console import color_print

DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)

from .utils import set_logger

class GBparam:
    """ Stores parameters of GroundBIRD analysis.
    All angles are in degree.

    Attributes
    -------
    tilt : float
        Tilt angle of the telescope.
        Default is 30.
    el : float
        Elevation of apperture center.
        90 - tilt (in degree).
        Default is 60.
    lat : float
        Latitude of the telescope.
        Default is lat_canary =  28d16m7s N =  28.268611.
    lon : float
        Longitude of the telescope
        Default is lon_canary = 16d36m20s W = -16.605555.
    fsample : float
        Sampling frequency.
    omega_gb : float
        Angular speed of the telescope.
        Default is 120 degree/s (20 rpm).
    omega_earth : float
        Angular speed of the Earth' rotation
        1 rotation in 1 sidereal day (~ 86164s)
        Default is 360 / 86164 degree/s.
    encoder_south : integer
        Encoder value of the South.
        Default is 3185 (at Kita Counter Hall, KEK).
        This value will be changed after installing at Teide Observatory.
    fname_pixel : string
        Name of the pixel information file.
        If it is not defined or the file does not exist, 
        dummy data with one pixel is used. 
    pixinfo : pixel info
    """
    tilt          = 30.
    el            = 90. - 30.
    lat           = (28. + 16./60 +  7./3600)
    lon           = -1 * (16. + 36./60 + 20./3600)
    fsample       = 1200         # sample/s
    omega_gb      = 360. / 3     # degree/s
    omega_earth   = 360. / 86164 # degree/s
    encoder_south = 3185         # at KCH 
    fname_pixel   = os.path.join(DIRNAME, './data/pixelinfo.dat')
    pixinfo = []

    log = set_logger('GBparam')
    def __init__(self, fname=os.path.join(DIRNAME,'data/default.ini')):
        """ __init__ method. The parameters and pixel information are loaded.  

        Parameters
        ----------
        fname : string
            The name and path of the name of the parameter file.
            Default is default.ini.
        """
        if (not os.path.isfile(fname)):
            self.log.warning('File "%s" does not exist. Using internal values.' % fname)
        else:
            self.load_parameters(fname) 

        self.load_pixelInfo(self.fname_pixel)

    def get_option(self, cp, sect, opt):
        """ Get option from given a config parser. 

        Parameters
        ----------
        cp : cparser.RawConfigParser
            Config parser from cparser. 
        sect : string
            Section for the option.
        opt : string
            Option name.

        Returns
        -------
        option : string
            Option values.
        """
        try:
            option = cp.get(sect, opt)
        except cparser.NoOptionError:
            self.log.warning('No option {0} in {1}'.format(opt, sect))
            option = getattr(self, opt)

        return option
                 
    def load_parameters(self, fname):
        """ Load parameters from a parameter file.

        Parameters
        ----------
        fname : string
            File name for the parameters.
        """ 
        cp = cparser.RawConfigParser()
        cpath   = fname
        cp.read(cpath)

        self.tilt           = float(self.get_option(cp, 'GB', 'tilt'))
        self.lat            = float(self.get_option(cp, 'GB', 'lat'))
        self.lon            = float(self.get_option(cp, 'GB', 'lon'))
        self.rot_speed      = float(self.get_option(cp, 'GB', 'rot_speed'))
        self.fsample        = int(self.get_option(cp, 'GB', 'fsample'))
        self.encoder_south  = int(self.get_option(cp, 'GB', 'encoder_south'))
        self.fname_pixel    = str(self.get_option(cp, 'GB', 'fname_pixel')).replace('MODULE_PATH', DIRNAME)
        self.omega_earth    = float(eval(self.get_option(cp, 'others', 'omega_earth')))

        self.EL             = 90. - self.tilt
        self.omega_gb       = 360. * self.rot_speed/60.

    def load_pixelInfo(self, fname=fname_pixel):
        """ Load pixel information from a pixel information file.

        Parameters
        ----------
        fname : string
            File name for the pixel information.
        """ 
        try:
            if (fname[-3:]=='csv'):
                self.pixinfo = np.genfromtxt(fname, names=True, delimiter=',')
            else:
                self.pixinfo = np.genfromtxt(fname, names=True)
        #except IOError: # for python2
        except OSError:
            self.log.warning('Pixel information file "%s" does not exist. Using a dummy focalplane.' % fname)
            self.fname_pixel = os.path.join(DIRNAME, './data/pixelinfo.dat')
            self.pixinfo = np.genfromtxt(self.fname_pixel, names=True)

    def show_parameters(self):
        """ Print parameters. """
        print('-'*50)
        print('GroundBIRD parameters')
        print('-'*50)
        print('Tilt      = ', self.tilt, '(deg)')
        print('Elevation = ', self.EL, '(deg)')
        print('Latitude  = ', self.lat, '(deg)')
        print('Longitude = ', self.lon, '(deg)')
        print('sampling frequency = ', self.fsample, '(sample/s)')
        print('rotation speed = ', self.omega_gb, '(deg/s)')
        print('earth rotation speed = ', self.omega_earth, '(deg/s)')
        print('encoder of South direction = ', self.encoder_south)
        print('-'*50)
        print()

    def show_pixelInfo(self):
        """ Print pixel information. """ 
        print('-'*50)
        print('GroundBIRD pixel information')
        print('-'*50)
        print(self.pixinfo.dtype.names)
        print(self.pixinfo)
        print('-'*50)
        print()

    def plot_focalplane2(self):
        """ Plot focal plane. """
        import pylab as plt
        import matplotlib.lines as mline
        ax=plt.subplot(111)
        ax.axis('equal')
        x = self.pixinfo['Yfoc']  # due to the axis definition in LT, x and y are exchanged.
        y = -1.0*self.pixinfo['Xfoc']
        #sc=ax.scatter(x, y, s=80, marker='o', 
        #              #edgecolors=self.pixinfo['mod'], 
        #              edgecolors='k',
        #              facecolors='none',
        #              linewidth=0.5)

        ## antenna direction
        ll = 3.2
        for p in self.pixinfo:
            px = p['Yfoc']
            py = p['Xfoc']
            detector = plt.Circle((px, py), 0.1, color='k', fill=False)
            ax.add_artist(detector)

            ang = p['omtfoc'] 
            sin0 = np.sin(np.radians(ang))
            cos0 = np.cos(np.radians(ang))
            sin90 = np.sin(np.radians(ang+90))
            cos90 = np.cos(np.radians(ang+90))

            xl1 = (-ll*cos90, ll*cos90) + p['Yfoc']
            yl1 = (-ll*sin90, ll*sin90) + p['Xfoc']

            xl2 = (-ll*cos0, ll*cos0) + p['Yfoc']
            yl2 = (-ll*sin0, ll*sin0) + p['Xfoc']

            l1 = mline.Line2D(xl1, yl1, C='b', linewidth=0.5)
            l2 = mline.Line2D(xl2, yl2, C='r', linewidth=0.5)
            ax.add_line(l1)
            ax.add_line(l2)

        ax.ylim=(-100, 100)
        ax.xlim=(-100, 100)
        #plt.colorbar(sc) 
        plt.show()

    def plot_focalplane(self):
        """ Plot focal plane. """
        import pylab as plt
        x = -1*self.pixinfo['Yfoc']  # due to the axis definition in LT, x and y are exchanged.
        y = self.pixinfo['Xfoc']

        plt.scatter(x, y)
        plt.axis('equal')

        #plt.colorbar(sc) 
        plt.show()

    def plot_beam(self):
        """ Plot beam on the sky. """
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        ax = fig.add_axes(rect)#plt.subplot(111)
        ax.axis('equal')
        theta = np.radians(self.pixinfo['theta'])  # due to the axis definition in LT, x and y are exchanged.
        phi = np.radians(self.pixinfo['phi'])
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)

        sc=ax.scatter(x, y, s=80, marker='o', 
                      #edgecolors=self.pixinfo['mod'], 
                      edgecolors='k',
                      facecolors='none',
                      linewidth=0.5)

        #ax_polar = fig.add_axes(rect, polar=True, frameon=False)
        #ax_polar.plot(phi, theta, 'o')
        #ax_polar.grid(True)

        ## antenna direction
        ll = 0.005
        for p in self.pixinfo:
            theta = np.radians(p['theta'])
            phi = np.radians(p['phi'])
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            ang = p['omtffr']
            sin0 = np.sin(np.radians(ang))
            cos0 = np.cos(np.radians(ang))
            sin90 = np.sin(np.radians(ang+90))
            cos90 = np.cos(np.radians(ang+90))

            xl1 = (-ll*cos90, ll*cos90) + x
            yl1 = (-ll*sin90, ll*sin90) + y 

            xl2 = (-ll*cos0, ll*cos0) + x 
            yl2 = (-ll*sin0, ll*sin0) + y

            l1 = mline.Line2D(xl1, yl1, C='b', linewidth=0.5)
            l2 = mline.Line2D(xl2, yl2, C='r', linewidth=0.5)
            ax.add_line(l1)
            ax.add_line(l2)

        #plt.colorbar(sc) 
        plt.show()

    def plot_beam_polar(self):
        """ Plot beam on the sky on polar coordinate. """ 
        ax=plt.subplot(111, projection='polar'); 
        ax.set_theta_zero_location("N")
        phi = np.radians(self.pixinfo['phi']-90)
        theta = (self.pixinfo['theta'])
        sc=ax.scatter(phi, theta,  
                      c=self.pixinfo['mod'], 
                      s=70)
                      #c=(pix['omtfoc']-pix['omtffr']) )
                      #s=abs(pix['omtfoc']-pix['omtffr'])*100)
        #ax.set_rmax(15)

        ## antenna direction (not used)
        #lt = 0.3
        #lp = 3
        #for p in self.pixinfo:
        #    theta_l1 = (p['theta']-lt, p['theta']+lt)
        #    phi_l1 = (np.radians(p['phi']), np.radians(p['phi']))
        #    theta_l2 = (p['theta'], p['theta'])
        #    phi_l2 = (np.radians(p['phi'])-np.radians(lp), np.radians(p['phi'])+np.radians(lp))
        #    ang = p['omtffr'] 
        #    l1 = mline.Line2D(phi_l1, theta_l1)
        #    l2 = mline.Line2D(phi_l2, theta_l2, C='r')
        #    ax.add_line(l1)
        #    ax.add_line(l2)

        plt.colorbar(sc) 

        plt.show()
        
