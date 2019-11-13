import time

import numpy as np
import pylab as plt

from scipy.fftpack import fft, ifft

def tod_fft(data, Nds=2e5):
    clk = 200e6  # clock speed of FPGA = 200 MHz
    Fs = clk / Nds  # default = 1000
    ts = 1/Fs

    N = data.size   
    tt = np.arange(N)
    t = tt * ts # time stamp
    T = N / Fs
    freq = np.fft.fftfreq(N) 
    dataf = np.fft.fft(data)

    return freq, dataf

