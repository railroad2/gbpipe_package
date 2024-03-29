import time

import numpy as np

from scipy.fftpack import fft, ifft

def tod_fft(data, Nds=2e5):
    clk = 200e6  # clock speed of FPGA = 200 MHz
    Fs = clk / Nds  # default = 1000
    ts = 1/Fs

    N = data.size   
    tt = np.arange(N)
    t = tt * ts # time stamp
    T = N / Fs
    freq = np.fft.fftfreq(N, 1/Nds) 
    dataf = np.fft.fft(data)

    return freq, dataf


def tod_ifft(data):
    datat = np.fft.ifft(data)

    return datat


def polyfit_for_rej(x, y, deg=0):
    par1d = np.polyfit(x, y, deg)
    fnc = np.poly1d(par1d)
    y_fit = fnc(x)
    
    return y_fit

def destripe_average(data, fitlength, deg=None, return_baseline=False):
    # without for loop
    try:
        y = np.reshape(data, (len(data)//fitlength, fitlength))
    except:
        l = len(data)//fitlength * (fitlength+1)
        y = np.zeros(l)
        y[:len(data)] = data
        y = np.reshape(y, (len(y)//fitlength, fitlength))

    baselinetmp = np.average(y, axis=1)
    baseline = np.zeros(np.shape(y))
    baseline = baseline.T
    baseline[:,:] = baselinetmp
    baseline = baseline.T.flatten()[:len(data)]
    data_destriped = data - baseline

    if return_baseline:
        return data_destriped, baseline
    else:
        return data_destriped
    

def destripe_poly(data, fitlength, deg=0, return_baseline=False):
    Nslice = np.int(np.ceil(len(data) / fitlength))
    baseline = []

    for i in range(Nslice):
        y = data[i*fitlength:(i+1)*fitlength]
        x = np.arange(len(y))
        y_fit = polyfit_for_rej(x, y, deg=deg)
        baseline += list(y_fit)

    baseline = np.array(baseline)

    data_destriped = data - baseline 

    if return_baseline:
        return data_destriped, baseline
    else:
        return data_destriped


def destripe_fft(data, cutfreq, filter_type='RECT', return_baseline=False):
    freq, dataf = tod_fft(data, 2e5)
    
    lpf = np.ones(len(freq)) 
    hpf = np.ones(len(freq))

    if filter_type == 'RECT':
        lpf[np.abs(freq)>=cutfreq] = 0
        hpf[np.abs(freq)<cutfreq] = 0
     
    baseline = tod_ifft(dataf * lpf)
    data_destriped = tod_ifft(dataf * hpf)

    if return_baseline:
        return data_destriped.real, baseline.real
    else:
        return data_destriped.real


    
    
