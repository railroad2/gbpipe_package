import sys
import time
import unittest

import numpy as np
import pylab as plt

sys.path.insert(0, '..')
from gbpipe import gbsim
from gbpipe import destripe # module under test

class destripe_test(unittest.TestCase):
    nsample = 600*1000 # 10 min data with 1 ksps sampling
    wnl = 1 # white noise level
    fknee = 1 # knee frequency
    data = gbsim.sim_noise1f(nsample, wnl, fknee, alpha=2)
    t0 = 0
    
    def setUp(self):
        self.t0 = time.time()

    def test1_tod_fft(self):
        freq, dataf = destripe.tod_fft(self.data)
        plt.loglog(freq, np.abs(dataf))

    def test2_tod_ifft(self):
        freq, dataf = destripe.tod_fft(self.data)
        data = destripe.tod_ifft(dataf)
        assert self.data.all() == data.all(), "Inverse FFT does not work correctly"

    def test3_polyfit_for_rej(self):
        deg = 4
        Nsample = 100 
        coeff = np.random.uniform(-10, 10, deg)
        x = np.linspace(-10, 10, Nsample)
        fnc = np.poly1d(coeff)
        y = fnc(x) 
        
        yfits = []
        for i in range(deg):
            yfit = destripe.polyfit_for_rej(x, y, deg=i)
            yfits.append(yfit)

        plt.figure()
        plt.plot(x, y)
        plt.plot(x, np.array(yfits).T)
    
    def test4_destripe_poly(self):
        dd, bl = destripe.destripe_poly(self.data, 1000, deg=0, return_baseline=True)
        plt.figure()
        plt.plot(self.data)
        plt.plot(dd)
        plt.plot(bl)
        

    def test5_destripe_fft(self):
        dd, bl = destripe.destripe_fft(self.data, self.fknee, return_baseline=True)

        plt.figure()
        plt.plot(self.data)
        plt.plot(dd)
        plt.plot(bl)
        
        plt.show()


    def tearDown(self):
        print("   Elapsed time =", time.time() - self.t0)
    

if __name__=='__main__':
    unittest.main()    

