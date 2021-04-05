import sys

sys.path.insert(0, '..')
import numpy as np
from gbpipe import gbsim


def test_sim_noise_focalplane_module():
    t1 = '2021-04-05T00:00:00'
    t2 = '2021-04-05T01:00:00'

    noise = gbsim.sim_noise_focalplane_module(t1, t2, fsample=10) 
    print (noise)


def test_GBsim_noise_fullmod():
    t1 = '2021-04-05T00:00:00'
    t2 = '2021-04-05T01:00:00'

    wnl = 1
    fknee = 0.1
    alpha = 1 
    rseed = 0
    fsample = 1000

    noise = gbsim.GBsim_noise_fullmod(t1, t2, fsample=fsample, 
                wnl=wnl, fknee=fknee, alpha=alpha, rseed=rseed,
                module_id=[0, 1, 2]) 
    
    return


if __name__=='__main__':

    #test_sim_noise_focalplane_module()
    test_GBsim_noise_fullmod()
