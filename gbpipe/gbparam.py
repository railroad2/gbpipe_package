from __future__ import print_function
import numpy as np
import os.path
import sys
if (sys.version_info > (3,)):
    import configparser as cparser
else:
    import ConfigParser as cparser
from astropy.utils.console import color_print

DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)

from utils import setLogger

class GBparam:
    """ Stores parameters of GroundBIRD.
    All angles are in degree.

    Members
    -------
    tilt : float
        Tilt angle of the telescope
        default = 30
    EL : float
        Elevation of apperture center
        90 - tilt (in degree)
        default = 60
    lat : float
        Lattitude of the telescope
        default = lat_canary =  28d16m7s N =  28.268611
    lon : float
        Longitude of the telescope
        default = lon_canary = 16d36m20s W = -16.605555
    omega_gb : float
        Angular speed of the telescope
        default = 120 degree/s (20 rpm)
    omega_earth : float
        Angular speed of the Earth' rotation
        1 rotation in 1 sidereal day (~ 86164s)
        default = 360 / 86164
    encoder_south : integer
        Encoder value of the South
    fname_pixel : string
        Name of the pixel information file.
        If it is not defined or the file does not exist, 
        dummy data with one pixel is used. 
    """
    tilt          = 30.
    EL            = 90. - 30.
    lat           = (28. + 16./60 +  7./3600)
    lon           = -1 * (16. + 36./60 + 20./3600)
    fsample       = 1200         # sample/s
    omega_gb      = 360. / 3     # degree/s
    omega_earth   = 360. / 86164 # degree/s
    encoder_south = 3185         # at KCH
    fname_pixel   = os.path.join(DIRNAME, './tmp/pixelinfo.dat')
    log = setLogger('GBparam')

    """
    pixinfo       = np.array([(0, 0, 0, 0., 0., 0., 0., 0., 0.)], 
                            dtype=[('Npix','int32'), ('Nmodule', 'int32'), ('Npix_mod', 'int32'), 
                                   ('X_fc', 'float64'), ('Y_fc', 'float64'), 
                                   ('theta', 'float64'), ('phi', 'float64'), 
                                   ('psi_fc', 'float64'), ('psi_far', 'float64')])

    """
    pixinfo = []
    def __init__(self, fname=os.path.join(DIRNAME,'default.ini')):
        if (not os.path.isfile(fname)):
            self.log.warning('File "%s" does not exist. Using internal values.' % fname)
        else:
            self.load_settings(fname) 

        self.load_pixelInfo(self.fname_pixel)

    def get_option(self, cp, sect, opt):
        try:
            return cp.get(sect, opt)
        except cparser.NoOptionError:
            self.log.warning('No option {0} in {1}'.format(opt, sect))
            return getattr(self, opt)
                 
    def load_settings(self, fname):
        cp = cparser.RawConfigParser()
        cpath   = fname
        cp.read(cpath)

        self.tilt           = float(self.get_option(cp, 'GB', 'tilt'))
        self.lat            = float(self.get_option(cp, 'GB', 'lat'))
        self.lon            = float(self.get_option(cp, 'GB', 'lon'))
        self.rot_speed      = float(self.get_option(cp, 'GB', 'rot_speed'))
        self.fsample        = int(self.get_option(cp, 'GB', 'fsample'))
        self.encoder_south  = int(self.get_option(cp, 'GB', 'encoder_south'))
        self.fname_pixel    = os.path.abspath(self.get_option(cp, 'GB', 'fname_pixel'))
        self.omega_earth    = float(eval(self.get_option(cp, 'others', 'omega_earth')))

        self.EL             = 90. - self.tilt
        self.omega_gb       = 360. * self.rot_speed/60.

    def load_pixelInfo(self, fname=fname_pixel):
        try:
            if (fname[-3:]=='csv'):
                self.pixinfo = np.genfromtxt(fname, names=True, delimiter=',')
            else:
                self.pixinfo = np.genfromtxt(fname, names=True)
        #except IOError:
        except OSError:
            self.log.warning('Pixel information file "%s" does not exist. Using dummy focalplane.' % fname)
            self.fname_pixel = os.path.join(DIRNAME, './tmp/pixelinfo.dat')
            self.pixinfo = np.genfromtxt(self.fname_pixel, names=True)

    def show_parameters(self):
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
        print('-'*50)
        print('GroundBIRD pixel information')
        print('-'*50)
        print(self.pixinfo.dtype.names)
        print(self.pixinfo)
        print('-'*50)
        print()

    def logtest(self):
        self.log.info('it is debug')
        self.log.debug('it is debug')
        self.log.error('it is debug')
        self.log.warning('it is debug')
    
    def plot_focalplane(self):
        import pylab as plt
        import matplotlib.lines as mline
        ax=plt.subplot(111)
        ax.axis('equal')
        x = self.pixinfo['Yfoc']  # due to the axis definition in LT, x and y are exchanged.
        y = self.pixinfo['Xfoc']
        sc=ax.scatter(x, y, s=80, marker='o', 
                      #edgecolors=self.pixinfo['mod'], 
                      edgecolors='k',
                      facecolors='none',
                      linewidth=0.5)

        ## antenna direction
        ll = 3
        for p in self.pixinfo:
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

        #plt.colorbar(sc) 
        plt.show()

    def plot_beam(self):
        import pylab as plt
        import matplotlib.lines as mline
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
        import pylab as plt
        import matplotlib.lines as mline
        ax=plt.subplot(111, projection='polar'); 
        ax.set_theta_zero_location("N")
        phi = np.radians(self.pixinfo['phi']-90)
        theta = (self.pixinfo['theta'])
        sc=ax.scatter(phi, theta,  
                      c=self.pixinfo['mod'], 
                      s=70)
                      #c=(pix['omtfoc']-pix['omtffr']) )
                      #s=abs(pix['omtfoc']-pix['omtffr'])*100)
        ax.set_rmax(15)

        """
        ## antenna direction
        lt = 0.3
        lp = 3
        for p in self.pixinfo:
            theta_l1 = (p['theta']-lt, p['theta']+lt)
            phi_l1 = (np.radians(p['phi']), np.radians(p['phi']))

            theta_l2 = (p['theta'], p['theta'])
            phi_l2 = (np.radians(p['phi'])-np.radians(lp), np.radians(p['phi'])+np.radians(lp))
            
            ang = p['omtffr'] 

            l1 = mline.Line2D(phi_l1, theta_l1)
            l2 = mline.Line2D(phi_l2, theta_l2, C='r')
            ax.add_line(l1)
            ax.add_line(l2)

        plt.colorbar(sc) 
        """
        plt.show()
        
def test_parser():
    par = GBparam()
    par.show_pixelInfo()
    par.show_parameters()
    par.logtest()

if __name__=='__main__':
    test_parser()
