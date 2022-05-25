import os
import sys

import numpy as np

import toast
import toast.todmap
import toast.pipeline_tools
from toast.mpi import MPI


mpiworld, procs, rank = toast.mpi.get_world()
comm = toast.mpi.Comm(mpiworld)


def define_data(detectors, nsamples):
    data = toast.Data(comm)  

    ## create empty
    tod = toast.tod.TODCache(comm.comm_group, detectors, nsamples)

    obs = {}
    obs["name"] = "default"
    obs["tod"] = tod

    data.obs.append(obs)

    return data


def read_pnt_tod(module, pixids=None, nsamples=None, startidx=0, endidx=None):
    pntpath = '/raid01/kmlee/tod_sim/GBsim_4days/2021-09-01/'
    pntfname = f'pointing_module_{module}.npz'

    if endidx is None:
        endidx = nsamples + startidx if nsamples is None else None

    pnt = np.load(os.path.join(pntpath, pntfname))
    pixs = pnt['pixs'][pixids, startidx:endidx]
    psis = pnt['psis'][pixids, startidx:endidx]

    return pixs, psis 


def run_madam(data, params, name_out=None):
    madam = toast.todmap.OpMadam(params=params, name_out=name_out)
    madam.exec(data)

    return 

