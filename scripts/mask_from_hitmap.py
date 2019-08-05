import sys
import numpy as np
import healpy as hp
import pylab as plt

def gbmap_cut_edge(map_in, angle, deg=True):
    nside = hp.npix2nside(len(map_in))
    map_tmp = map_in.copy()
    map_tmp[map_in == 0] = hp.UNSEEN
    idx = np.arange(len(map_tmp))
    idx_obs = idx[map_in != hp.UNSEEN]
    theta0, phi0 = hp.pix2ang(nside, idx_obs[0])
    theta1, phi1 = hp.pix2ang(nside, idx_obs[-1])

    if deg:
        angle = np.radians(angle)

    idx_strip = hp.query_strip(nside, theta0+angle, theta1-angle) 
    
    map_out = np.full(len(map_in), 0) #hp.UNSEEN) 
    map_out[idx_strip] = 1 

    return map_out

    
if __name__=='__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    hitmap = hp.read_map(infile)

    mask = gbmap_cut_edge(hitmap, 0) 
    mask_5deg = gbmap_cut_edge(hitmap, 5)
    mask_20deg = gbmap_cut_edge(hitmap, 20)

    hp.mollview(mask)
    hp.mollview(mask_5deg)
    hp.mollview(mask_20deg)
    plt.show()

