import os
import numpy as np
import healpy as hp
import datetime

from gbpipe import gbsim

def doit():
    t1 = "2019-05-01T00:00:00"
    t2 = "2019-05-01T01:00:00"
    dtsec = 600
    fsample = 13

    cmbname = "/home/klee_ext/kmlee/maps/cmb_rseed42.fits"
    fg145name = "/home/klee_ext/kmlee/maps/fg145_equ.fits"
    fg220name = "/home/klee_ext/kmlee/maps/fg225_equ.fits"
    dt = datetime.datetime.now()
    nproc = 8
    nside_hitmap = None

    outpath = "/home/klee_ext/kmlee/hpc_data/{}_GBsim_test".format(dt.strftime("%Y-%m-%d"))

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=cmbname, module_id=0,
        fprefix="GBtod_cmb220", outpath=outpath,
        nside_hitmap=nside_hitmap, nproc=nproc)

    return 

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=cmbname, module_id=(1, 2, 3, 4, 5, 6), 
        fprefix="GBtod_cmb145", outpath=outpath, 
        nside_hitmap=nside_hitmap, nproc=nproc)

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=fg220name, module_id=0,
        fprefix="GBtod_fg220", outpath=outpath,
        nside_hitmap=False, nproc=nproc)

    gbsim.GBsim_hpc_parallel_time(t1, t2, dtsec=dtsec,
        fsample=fsample, mapname=fg145name, module_id=(1, 2, 3, 4, 5, 6),
        fprefix="GBtod_fg145", outpath=outpath,
        nside_hitmap=False, nproc=nproc)

    gbsim.GBsim_noise(t1, t2, dtsec=dtsec,
        fsample=fsample, 
        wnl=1e-10, fknee=0.1, alpha=1, rseed=0,
        module_id=None, fprefix="GBtod_noise",
        outpath=outpath, nproc=nproc)

if __name__ == "__main__":
    doit()
