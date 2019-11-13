import sys
import time
import unittest

import numpy as np
import pylab as plt

sys.path.insert(0, '..')
from gbpipe import gbsim
from gbpipe import destripe # module under test

class destripe_test(unittest.TestCase):

    def setUp(self):
        self.nsample = 3600*1000 # 1 hr data with 1 ksps sampling
        self.wnl = 1 # white noise level
        self.fknee = 1 # knee frequency
        self.data = gbsim.sim_noise1f(self.nsample, self.wnl, self.fknee)
    
    def test_tod_fft(self):
        t0 = time.time()

        freq, dataf = destripe.tod_fft(self.data)
        plt.loglog(freq, np.abs(dataf))
        plt.show()

if __name__=='__main__':
    unittest.main()    
