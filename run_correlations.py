from astropy.io import fits
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import os
import sys
import common as cmm

if len(sys.argv)!=6 :
    print "Usage : run_correlations.py nbins nside_cmbl use_wiener[0,1] nsims randomize_points[0,1]"
    exit(1)

#Maximum scale (deg) in the computation of the 2PCF
thmax=1.
#Number of bins in theta
nth=int(sys.argv[1])
#Resolution of the CMB lensing map
nside_cmbl=int(sys.argv[2])
#Do we implement a wiener filter?
do_wiener=int(sys.argv[3])
#How many simulations do we run for the errors?
nsims=int(sys.argv[4])
#Do we randomize points as well as fields?
do_randomize_points=int(sys.argv[5])

randomize_points=False
if do_randomize_points>0 :
    randomize_points=True
use_wiener=False
if do_wiener>0 :
    use_wiener=True

#Create output directory
outdir="outputs_ns%d_nb%d"%(nside_cmbl,nth)
if use_wiener :
    outdir+="_wiener"
if randomize_points :
    outdir+="_randp"
outdir+="/"
os.system("mkdir -p "+outdir)

#Generate data in right format
fname_cmbl,fname_mask_cmbl=cmm.reform_data(nside_cmbl,use_wiener=use_wiener)

fname_alldata=outdir+"wth_qxk_all"

if os.path.isfile(fname_alldata+".npz") :
    print " File already exists"
    exit(0)

print "Reading kappa map and mask"
mask =hp.read_map(fname_mask_cmbl,verbose=False)
field=hp.read_map(fname_cmbl,verbose=False)

print "Reading QSOs and DLAs"
data_dla=(fits.open(cmm.fname_dla))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data
n_dla=len(data_dla)
n_qso=len(data_qso)
n_qsu=len(np.where(data_qso['UNIHI'])[0])

print "Reading QSO mask"
mask_qso=hp.read_map(cmm.fname_mask_qso,verbose=False)
ndens_qso=n_qso/(4*np.pi*np.mean(mask_qso))
ndens_dla=n_dla/(4*np.pi*np.mean(mask_qso))

print "Computing the DLA 2PCF"
th_dla,wth_dla,hf_dla,hm_dla=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_dla,
                                                 thmax,nth,fname_out=outdir+"corr_c_dla.txt")

print "Computing the QSO 2PCF"
th_qso,wth_qso,hf_qso,hm_qso=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,
                                                 thmax,nth,fname_out=outdir+"corr_c_qso.txt")
wth_dlo=wth_dla-wth_qso

print "Computing the QSO-UNIFORM 2PCF"
th_qsu,wth_qsu,hf_qsu,hm_qsu=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,
                                                 thmax,nth,fname_out=outdir+"corr_c_qsu.txt",
                                                 cut_name='UNIHI',weight_name='NO_WEIGHT')

def get_random_corr(isim) :
    cleanup=False
    fname_dla=outdir+'corr_c_dla_random%d.txt'%isim
    fname_qso=outdir+'corr_c_qso_random%d.txt'%isim
    fname_qsu=outdir+'corr_c_qsu_random%d.txt'%isim
    if (not ((os.path.isfile(fname_dla)) and (os.path.isfile(fname_qso)) and (os.path.isfile(fname_qsu)))) :
        print isim
        cleanup=True
        if randomize_points :
            cmm.random_points(1000+isim,mask_qso,n_dla,fname_out=outdir+'dla_random_%d.fits'%isim,
                              weights=data_dla['W'])
            cmm.random_points(1000+isim,mask_qso,n_qso,fname_out=outdir+'qso_random_%d.fits'%isim,
                              weights=data_qso['W'])
            cmm.random_points(1000+isim,mask_qso,n_qsu,fname_out=outdir+'qsu_random_%d.fits'%isim,
                              weights=None)
        cmm.random_map(1000+isim,mask,cmm.fname_kappa_cl,fname_out=outdir+'map_random_%d.fits'%isim,
                       use_wiener=use_wiener)

    if randomize_points :
        th_dla,w_dla,hf_dla,hm_dla=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       outdir+'dla_random_%d.fits'%isim,thmax,nth,
                                                       fname_out=fname_dla)
        th_qso,w_qso,hf_qso,hm_qso=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       outdir+'qso_random_%d.fits'%isim,thmax,nth,
                                                       fname_out=fname_qso)
        th_qsu,w_qsu,hf_qsu,hm_qsu=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       outdir+'qsu_random_%d.fits'%isim,thmax,nth,
                                                       weight_name='NO_WEIGHT',fname_out=fname_qsu)
    else :
        th_dla,w_dla,hf_dla,hm_dla=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       cmm.fname_dla,thmax,nth,fname_out=fname_dla)
        th_qso,w_qso,hf_qso,hm_qso=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       cmm.fname_qso,thmax,nth,fname_out=fname_qso)
        th_qsu,w_qsu,hf_qsu,hm_qsu=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                       cmm.fname_qso,thmax,nth,cut_name='UNIHI',
                                                       weight_name='NO_WEIGHT',fname_out=fname_qsu)
        
    if cleanup :
        if randomize_points :
            os.system('rm '+outdir+'dla_random_%d.fits '%isim+outdir+'qso_random_%d.fits '%isim+
                      outdir+'qsu_random_%d.fits '%isim)
        os.system('rm '+outdir+'map_random_%d.fits '%isim)

    return th_dla,w_dla,hf_dla,hm_dla,th_qso,w_qso,hf_qso,hm_qso,th_qsu,w_qsu,hf_qsu,hm_qsu

print "Generating %d random measurements"%nsims
data_randoms=np.zeros([nsims,4,4,nth])
data_randoms_2=np.zeros([nsims,4,2*nth])
for i in np.arange(nsims) :
    td,wd,fd,md,tq,wq,fq,mq,tu,wu,fu,mu=get_random_corr(i)
    data_randoms[i,0,:,:]=np.array([td,wd,fd,md])
    data_randoms[i,1,:,:]=np.array([tq,wq,fq,mq])
    data_randoms[i,2,:,:]=np.array([tu,wu,fu,mu])
    data_randoms[i,3,:,:]=data_randoms[i,0,:,:]-data_randoms[i,1,:,:]
    data_randoms_2[i,:,:nth]=np.array([td,wd,fd,md])
    data_randoms_2[i,:,nth:]=np.array([tq,wq,fq,mq])
tharr=np.mean(data_randoms[:,0,0,:],axis=0)

np.savez(fname_alldata,th=tharr,
         wth_dla=wth_dla,hf_dla=hf_dla,hm_dla=hm_dla,
         wth_qso=wth_qso,hf_qso=hf_qso,hm_qso=hm_qso,
         wth_qsu=wth_qsu,hf_qsu=hf_qsu,hm_qsu=hm_qsu,
         randoms=data_randoms,randoms_2=data_randoms_2)
