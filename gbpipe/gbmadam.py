import os
import numpy as np
import healpy as hp
import pylab as plt

from scipy.interpolate import interp1d
from mpi4py import MPI
from astropy.io import fits

import libmadam_wrapper as madam
from gbpipe import gbparam, gbdir
from gbpipe.gbsim import sim_noise1f, sim_noise1f_old, noise_psd
from gbpipe.utils import set_logger, hostname


def set_parameters(nside, fsample, nsample, outpath):
    pars = {}

    pars['info'] = 2
    #pars['nthreads'] = 1
    pars['nsubchunk'] = 0
    #pars['isubchunk'] = 1
    #pars['time_unit'] = 
    pars['base_first'] = 1
    #pars['nshort'] = 10
    pars['nside_map'] = nside
    pars['nside_cross'] = nside // 2
    pars['nside_submap'] = nside // 4
    #pars['good_baseline_fraction'] = 
    #pars['concatenate_messages'] = 
    pars['allreduce'] = True
    #pars['reassign_submaps'] = 
    #pars['pixmode_map'] = 
    #pars['pixmode_cross'] = 
    #pars['pixlim_map'] = 
    #pars['pixlim_cross'] = 
    #pars['incomplete_matrices'] = 
    #pars['allow_decoupling'] = 
    #pars['kfirst'] = False
    pars['basis_func'] = 'polynomial'
    pars['basis_order'] = 0
    #pars['iter_min'] = 
    pars['iter_max'] = 10
    #pars['cglimit'] = 
    pars['fsample'] = fsample
    #pars['mode_detweight'] = 
    #pars['flag_by_horn'] = 
    #pars['write_cut'] = 
    #pars['checknan'] = 
    #pars['sync_output'] = 
    #pars['skip_existing'] = 
    pars['temperature_only'] = False
    #pars['force_pol'] = False
    pars['noise_weights_from_psd'] = True
    #pars['radiometers'] = 
    #pars['psdlen'] = 
    #pars['psd_down'] = 
    #pars['kfilter'] = False
    #pars['diagfilter'] = 0.0
    #pars['precond_width_min'] = 
    #pars['precond_width_max'] = 
    #pars['use_fprecond'] = 
    #pars['use_cgprecond'] = 
    #pars['rm_monopole'] = True
    #pars['path_output'] = '/home/klee_ext/kmlee/hpc_data/madam_test/'
    pars['path_output'] = outpath 
    pars['file_root'] = 'madam_test'

    pars['write_map'] = True
    pars['write_binmap'] = True
    pars['write_hits'] = True
    pars['write_matrix'] = False#True
    pars['write_wcov'] = False#True
    pars['write_base'] = True
    pars['write_mask'] = False#True
    pars['write_leakmatrix'] = False

    #pars['unit_tod'] = 
    #pars['file_gap_out'] = 
    #pars['file_mc'] = 
    #pars['write_tod'] = True
    #pars['file_inmask'] = 
    #pars['file_spectrum'] = 
    #pars['file_gap'] = 
    #pars['binary_output'] = 
    #pars['nwrite_binary'] = 
    #pars['file_covmat'] = 
    #pars['detset'] = 
    #pars['detset_nopol'] = 
    #pars['survey'] = ['hm1:{} - {}'.format(0, nsample / 2),]
    #pars['bin_subsets'] = True
    #pars['mcmode'] = 

    return pars


def pixels_for_detectors(module, pix_mod, ra, dec, psi, nside=1024):
    npix = nside * nside * 12

    param = gbparam.GBparam()
    gbpix = param.pixinfo
    module_idx = module

    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]

    v_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    p_det = gbdir.psi2vec_xp(v_det, psi_det)

    v_det = v_det[pix_mod]
    p_det = p_det[pix_mod]

    rmat = gbdir.Rot_matrix_equatorial(ra, dec, psi, deg=True)
    v_obs = gbdir.Rotate(v_det, rmat)
    p_obs = gbdir.Rotate(p_det, rmat)
    pixs = hp.vec2pix(nside, v_obs[:,0], v_obs[:,1], v_obs[:,2])
    psis = gbdir.angle_from_meridian(v_obs, p_obs)

    return pixs, psis


def madam_map_QU(
        inpath,
        outpath,
        module = 1,
        pix_mod=np.arange(23), 
        nside=1024, 
        observationtime=86400):

    # some lines to play with multiple files efficienly.
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger(level='INFO')

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))

    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = npix // fsample #600
    nnz = 3
    nfile = 3
    length = observationtime * fsample 
    #length = 600 * nfile * fsample

    log.info('Using module {}, detector {}'.format(module, pix_mod))
    log.info('Loading tod')

    
    cmb145_fnames = os.listdir(os.path.join(inpath,'GBtod_cmb145'))
    cmb220_fnames = os.listdir(os.path.join(inpath,'GBtod_cmb220'))
    fg145_fnames = os.listdir(os.path.join(inpath,'GBtod_fg145'))
    fg220_fnames = os.listdir(os.path.join(inpath,'GBtod_fg220'))
    noise_fnames = os.listdir(os.path.join(inpath,'GBtod_noise'))

    cmb145_fnames.sort()
    cmb220_fnames.sort()
    fg145_fnames.sort()
    fg220_fnames.sort()
    noise_fnames.sort()

    dets = []

    for pix_idx in pix_mod:
        dets.append('mod{}det{}'.format(module, pix_idx))

    ra = np.zeros(length)
    dec = np.zeros(length) 
    psi_arr = np.zeros((length, 23))
    #pix_arr = []
    ix_arr = np.zeros((length, 23)) 
    #iy_arr = []
    noi_arr = []

    idx_last = 0
    #for i, _ in enumerate(noise_fnames[:nfile]):

    for i, _ in enumerate(noise_fnames):
        c1f = os.path.join(inpath, 'GBtod_cmb145', cmb145_fnames[i])
        c2f = os.path.join(inpath, 'GBtod_cmb220', cmb220_fnames[i]) 
        f1f = os.path.join(inpath, 'GBtod_fg145', fg145_fnames[i])
        f2f = os.path.join(inpath, 'GBtod_fg220', fg220_fnames[i])
        nf  = os.path.join(inpath, 'GBtod_noise', noise_fnames[i])

        if module==0:
            print (c2f)  
            hdu = fits.open(c2f) 
            len_cur = len(hdu[1].data)
            ra[idx_last:idx_last+len_cur] = hdu[1].data['ra']
            dec[idx_last:idx_last+len_cur] = hdu[1].data['dec']
            ix_arr[idx_last:idx_last+len_cur] = hdu[1].data['tod_ix_mod_'+str(module)]
            psi_arr[idx_last:idx_last+len_cur] = hdu[1].data['tod_psi_mod_'+str(module)]
            hdu.close()

            hdu = fits.open(f2f)
            ix_arr[idx_last:idx_last+len_cur] += hdu[1].data['tod_ix_mod_'+str(module)]
            hdu.close()

        else:
            print (c1f)  
            hdu = fits.open(c1f) 
            len_cur = len(hdu[1].data)
            ra[idx_last:idx_last+len_cur] = hdu[1].data['ra']
            dec[idx_last:idx_last+len_cur] = hdu[1].data['dec']
            ix_arr[idx_last:idx_last+len_cur] = hdu[1].data['tod_ix_mod_'+str(module)]
            psi_arr[idx_last:idx_last+len_cur] = hdu[1].data['tod_psi_mod_'+str(module)]
            hdu.close()

            hdu = fits.open(f1f)
            ix_arr[idx_last:idx_last+len_cur] += hdu[1].data['tod_ix_mod_'+str(module)]
            hdu.close()

        hdu = fits.open(nf)
        noi = hdu[1].data['n1f']
        noi /= np.sqrt(365 * 3 * 0.8)
        noi = np.array([noi] * len(pix_mod)).T
        ix_arr[idx_last:idx_last+len_cur] += noi

        idx_last += len_cur

        hdu.close()

    ## end of for

    print(ra.shape)
    print(dec.shape)
    psi_arr = psi_arr.T
    print(psi_arr.shape)

    log.info('Calculating pointings')

    nsample = len(ra) 

    log.info('number of samples = {}'.format(nsample))

    pix_obs_arr = []
    psi_obs_arr = []

    for det, psi in zip(pix_mod, psi_arr):
        res = pixels_for_detectors(module, det, ra, dec, psi, nside)
        pix_obs_arr.append(res[0]) 
        psi_obs_arr.append(res[1])

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
        log.info('Directory {} have been made.'.format(outpath))

    log.info('Setting parameter')
    pars = set_parameters(nside, fsample, nsample, outpath)

    np.random.seed(1) 

    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    log.info('Generating time stamp')
    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    ## concatenate the arrays
    signal_arr = ix_arr #- iy

    signal  = np.concatenate(signal_arr, axis=None)
    pix_obs = np.concatenate(pix_obs_arr, axis=None)
    psi_obs = np.concatenate(psi_obs_arr, axis=None)

    del(signal_arr)
    del(pix_obs_arr)
    del(psi_obs_arr)

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = pix_obs 
    #del(pix_obs)

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    pixweights[::3] = 1
    pixweights[1::3] = np.cos(2*psi_obs)
    pixweights[2::3] = np.sin(2*psi_obs) 
    #del(psi_obs)

    ## noise psd
    psdf, psdv = noise_psd(0.1)
    pars['noise_weights_from_psd'] = True

    signal_in = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    signal_in[:] = signal 
    del(signal)

    log.info('Setting periods')
    nperiod = 100
    periods = np.zeros(nperiod, dtype=int)

    for i in range(nperiod):
        periods[i] = int(nsample//nperiod*i)

    log.debug ('periods={}'.format(periods))

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf) 
    log.debug('npsdbin={}'.format(npsdbin))
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)

    for i in range(npsdtot):
        psdvals[i*npsdbin:(i+1)*npsdbin] = np.abs(psdv[:npsdbin])
    
    log.info('Calling Madam')
    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    return


def main():
    import trackback
    import smtplib
    from email.mime.text import MIMEText

    try:
        inpath = '/home/klee_ext/kmlee/hpc_data/GBsim_1day_beam/2019-09-01/pixel_tod/'
        outpath = '/home/klee_ext/kmlee/test_madam/2019-08-08_gb_all_mod'
        for i in range(7):
            madam_map_QU(module=i, outpath=outpath+str(i))

    except Exception as err:
        print ('There is some error. Sending the information message...')
        print (str(err))
        print (traceback.format_exc())

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('phynbook@gmail.com', 'womostlgdkfuekmz')

        text = f'An error occured during running "{__file__}" on "{hostname()}".\n'
        text += f'{err}\n{traceback.format_exc()}\n'
        msg = MIMEText(text)
        msg['Subject'] = f'ERROR: {__file__} on {hostname()}'
        s.sendmail("kmlee@hep.korea.ac.kr", "kmlee@hep.korea.ac.kr", msg.as_string())
        s.quit()


if __name__=='__main__':
    main()


