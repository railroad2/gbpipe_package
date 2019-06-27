import numpy as np
import healpy as hp
import pylab as plt

def syn_toymap():
    lmax = 4096
    cl = np.zeros((4, lmax))
    rseed = 42

    cl[0, 5] = 1.0   # TT
    cl[1, 10] = 0.1  # EE
    cl[2, 10] = 0.01 # BB

    np.random.seed(rseed)
    maps = hp.synfast(cl, nside=1024, lmax=lmax, new=True)

    extra_header = (['clTT5', 1.0],
                    ['clEE10', 0.1],
                    ['clBB10', 0.01],
                    ['rseed', 42],)
    
    hp.write_map('cmb_toy.fits', maps, extra_header=extra_header, overwrite=True)

    return maps

if __name__ == '__main__':
    maps = syn_toymap()

    hp.mollview(maps[0], title='T')
    hp.mollview(maps[1], title='Q')
    hp.mollview(maps[2], title='U')
    plt.show()

