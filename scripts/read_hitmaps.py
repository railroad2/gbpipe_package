import os
import sys
import numpy as np
import healpy as hp
import pylab as plt

def read_hitmaps(filelist, field=None):
    m_arr = []

    for fn in filelist:
        print (fn)
        m, h = hp.read_map(fn, field=field, verbose=False, h=True) 
        m[m < -1e15] = 0
        m_arr.append(m)
        h = dict(h)

    return np.array(m_arr)
    

if __name__=='__main__':
    path = sys.argv[1]
    #path = '/scratch/kmlee_hpc/2019-04-22_GBsim'
    ls = os.listdir(path)
    hit_ls = [s for s in ls if 'hitmap' in s]
    hit_ls.sort()
    hit_ls = [os.path.join(path, l) for l in hit_ls]
    m = read_hitmaps(hit_ls)
    while len(m.shape) > 1:
        m = np.sum(m, axis=0)

    m[m<1] = hp.UNSEEN

    if (len(sys.argv)==3):
        hp.write_map(sys.argv[2], m)

    #hp.mollview(m, xsize=3000)

    #plt.show()



