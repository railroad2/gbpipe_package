import os
import numpy as np
import healpy as hp
import datetime

from gbpipe import gbsim

def doit():
    t1 = "2019-09-01T00:00:00"
    #t2 = "2019-09-01T00:20:00" # 20 min for test
    t2 = "2019-09-02T00:00:00" # 1 day
    dtsec = 600
    fsample = 1000

    #cmbname   = "/home/klee_ext/kmlee/maps/cmb_rseed42_0_5deg.fits"
    cmbname   = "/home/klee_ext/kmlee/maps/cmb_toy.fits"
    fg145name = "/home/klee_ext/kmlee/maps/fg145comb_Ns1024_fwhm0_5.fits"
    fg220name = "/home/klee_ext/kmlee/maps/fg220comb_Ns1024_fwhm0_5.fits"
    dt = datetime.datetime.now()
    nproc = 16
    nside = 1024
    nside_hitmap = 1024 

    outpath = "/home/klee_ext/kmlee/hpc_data/{}_GBsim_1day_mod".format(dt.strftime("%Y-%m-%d"))

    """
    gbsim.GBsim_noise_long(t1, t2, dtsec=dtsec,
        fsample=fsample, 
        wnl=310e-6, fknee=0.1, alpha=1, rseed=0,
        module_id=None, fprefix="GBtod_noise",
        outpath=outpath, nproc=nproc)

    gbsim.GBsim_noise_long(t1, t2, dtsec=dtsec,
        fsample=fsample, 
        wnl=310e-6, fknee=0.0, alpha=1, rseed=0,
        module_id=None, fprefix="GBtod_wnoise",
        outpath=outpath, nproc=nproc)
    """

    """
    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=cmbname, module_id=(0, 1, 2, 3, 4, 5, 6), 
        fprefix="GBtod_cmb", outpath=outpath, 
        nside_hitmap=nside_hitmap, nproc=nproc)
    """

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=fg145name, module_id=(1, 2, 3, 4, 5, 6),
        fprefix="GBtod_fg145comb", outpath=outpath,
        nside=nside, nside_hitmap=False, nproc=nproc)

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=fg220name, module_id=0,
        fprefix="GBtod_fg220comb", outpath=outpath,
        nside=nside, nside_hitmap=False, nproc=nproc)

    return

if __name__ == "__main__":
    doit()

