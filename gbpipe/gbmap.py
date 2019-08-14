import numpy as np
import healpy as hp

def gbmask_cut_edge(map_in, angle, nside_out=None, deg=True):
    nside_in = hp.npix2nside(len(map_in))
    if nside_out is None:
        nside_out = nside_in

    npix_out = hp.nside2npix(nside_out)

    map_tmp = map_in.copy()
    map_tmp[map_in == 0] = hp.UNSEEN
    idx = np.arange(len(map_tmp))
    idx_obs = idx[map_tmp != hp.UNSEEN]
    theta0, phi0 = hp.pix2ang(nside_in, idx_obs[0])
    theta1, phi1 = hp.pix2ang(nside_in, idx_obs[-1])

    if deg:
        angle = np.radians(angle)

    idx_strip = hp.query_strip(nside_out, theta0+angle, theta1-angle)

    map_out = np.full(npix_out, 0) #hp.UNSEEN) 
    map_out[idx_strip] = 1 

    return map_out
                            
