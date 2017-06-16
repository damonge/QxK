from astropy.io import fits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import interp1d
import os
import sys
import common as cmm
import pyccl as ccl
import cl2wth as c2w
from scipy.integrate import quad
import scipy.stats as st

#Number of DLAs in the catalog
n_dla=34050
#Number of QSOs in the catalog
n_qso=297301
#Maximum scale (deg) in the computation of the 2PCF
thmax=1.
#Number of bins in theta
nth=16
#Maximum angle to use in the fit
th_thr=0.7
#Do we compute the 2PCF using the slow python implementation?
compute_with_python=False

#Create output directory
outdir="outputs_nb%d/"%nth
os.system("mkdir -p "+outdir)

#Generate data in right format
cmm.reform_data()

print "Reading kappa map and mask"
mask=hp.read_map(cmm.fname_mask_cmbl,verbose=False)
field=hp.read_map(cmm.fname_cmbl,verbose=False)

print "Reading QSO mask"
mask_qso=hp.read_map(cmm.fname_mask_qso,verbose=False)

print "Computing the DLA 2PCF"
data_dla=(fits.open(cmm.fname_dla))[1].data
th_dla,wth_dla,hf_dla,hm_dla=cmm.compute_xcorr_c(cmm.fname_cmbl,cmm.fname_mask_cmbl,cmm.fname_dla,
                                                 thmax,nth,fname_out=outdir+"corr_c_dla.txt")

print "Computing the QSO 2PCF"
th_qso,wth_qso,hf_qso,hm_qso=cmm.compute_xcorr_c(cmm.fname_cmbl,cmm.fname_mask_cmbl,cmm.fname_qso,
                                                 thmax,nth,fname_out=outdir+"corr_c_qso.txt")
wth_dlao=wth_dla-wth_qso

def get_random_corr(isim) :
    cleanup=False
    fname_dla=outdir+'corr_c_dla_random%d.txt'%isim
    fname_qso=outdir+'corr_c_qso_random%d.txt'%isim
    if (not ((os.path.isfile(fname_dla)) and (os.path.isfile(fname_qso)))) :
        print isim
        cleanup=True
        cmm.random_points(mask_qso,34050,fname_out='data/dla_random_%d.fits'%isim,weights=np.ones(n_dla))
        cmm.random_points(mask_qso,297301,fname_out='data/qso_random_%d.fits'%isim,weights=np.ones(n_qso))
        cmm.random_map(mask,cmm.fname_kappa_cl,fname_out='data/map_random_%d.fits'%isim)

    th_dla,w_dla,hf_dla,hm_dla=cmm.compute_xcorr_c('data/map_random_%d.fits'%isim,cmm.fname_mask_cmbl,
                                                   'data/dla_random_%d.fits'%isim,thmax,nth,
                                                   fname_out=fname_dla)
    th_qso,w_qso,hf_qso,hm_qso=cmm.compute_xcorr_c('data/map_random_%d.fits'%isim,cmm.fname_mask_cmbl,
                                                   'data/qso_random_%d.fits'%isim,thmax,nth,
                                                   fname_out=fname_qso)

    if cleanup :
        os.system('rm data/dla_random_%d.fits data/qso_random_%d.fits data/map_random_%d.fits'%(isim,isim,isim))
    
    return th_dla,w_dla,hf_dla,hm_dla,th_qso,w_qso,hf_qso,hm_qso

nsims=1000
print "Generating %d random measurements"%nsims
data_randoms=np.zeros([nsims,3,4,nth])
for i in np.arange(nsims) :
    td,wd,fd,md,tq,wq,fq,mq=get_random_corr(i)
    data_randoms[i,0,:,:]=np.array([td,wd,fd,md])
    data_randoms[i,1,:,:]=np.array([tq,wq,fq,mq])
    data_randoms[i,2,:,:]=data_randoms[i,0,:,:]-data_randoms[i,1,:,:]
tharr=np.mean(data_randoms[:,0,0,:],axis=0)

print  "Computing covariance matrices"
mean_dla=np.mean(data_randoms[:,0,1,:],axis=0)
covar_dla=np.mean(data_randoms[:,0,1,:,None]*data_randoms[:,0,1,None,:],axis=0)-mean_dla[:,None]*mean_dla[None,:]
corr_dla=covar_dla/np.sqrt(np.diag(covar_dla)[None,:]*np.diag(covar_dla)[:,None])

mean_qso=np.mean(data_randoms[:,1,1,:],axis=0)
covar_qso=np.mean(data_randoms[:,1,1,:,None]*data_randoms[:,1,1,None,:],axis=0)-mean_qso[:,None]*mean_qso[None,:]
corr_qso=covar_qso/np.sqrt(np.diag(covar_qso)[None,:]*np.diag(covar_qso)[:,None])

mean_dlao=np.mean(data_randoms[:,2,1,:],axis=0)
covar_dlao=np.mean(data_randoms[:,2,1,:,None]*data_randoms[:,2,1,None,:],axis=0)-mean_dlao[:,None]*mean_dlao[None,:]
corr_dlao=covar_dlao/np.sqrt(np.diag(covar_dlao)[None,:]*np.diag(covar_dlao)[:,None])

plt.figure()
plt.imshow(corr_dla,origin='lower',interpolation='nearest')

plt.figure()
plt.imshow(corr_qso,origin='lower',interpolation='nearest')

plt.figure()
plt.imshow(corr_dlao,origin='lower',interpolation='nearest')

print "Computing theory prediction"
def get_nz_oversample(bins,nzar,nbins) :
    zsub=(bins[1:]+bins[:-1])*0.5
    nsub=(nzar+0.0)/((np.sum(nzar)+0.)*(bins[1]-bins[0]))
    nf=interp1d(zsub,nsub,bounds_error=False,fill_value=0)
    zarr=bins[0]+(bins[-1]-bins[0])*np.arange(nbins)/(nbins-1.)
    pzarr=nf(zarr)
    return zarr,pzarr
nz,bn=np.histogram(data_dla['z_abs'],range=[0,7],bins=50)
zarr_dlao,nzarr_dlao=get_nz_oversample(bn,nz,256)
bzarr_dlao=cmm.bias_dla(zarr_dlao)
nz,bins=np.histogram(data_dla['zqso'],range=[0,7],bins=50)
zarr_qso,nzarr_qso=get_nz_oversample(bn,nz,256)
bzarr_qso=cmm.bias_qso(zarr_qso)

print "  Cls"
if not os.path.isfile(outdir+"cls_th.txt") :
    cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)#,transfer_function='eisenstein_hu')
    clt_dlao=ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlao,nzarr_dlao),(zarr_dlao,bzarr_dlao))
    clt_qso =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso,nzarr_qso),(zarr_qso,bzarr_qso))
    clt_cl  =ccl.ClTracerCMBLensing(cosmo)
    larr    =np.concatenate((1.*np.arange(500),500+10.*np.arange(950)))
    cl_dlao =ccl.angular_cl(cosmo,clt_dlao,clt_cl,larr,l_limber=-1)
    cl_qso  =ccl.angular_cl(cosmo,clt_qso ,clt_cl,larr,l_limber=-1)
    np.savetxt(outdir+"cls_th.txt",np.transpose([larr,cl_dlao,cl_qso]))
larr,cl_dlao,cl_qso=np.loadtxt(outdir+"cls_th.txt",unpack=True)
cl_dla=cl_dlao+cl_qso

print "  w(theta)"
if not os.path.isfile(outdir+"wth_th.txt") :
    tharr_th=2.*np.arange(256)/255.
    wth_th_dlao=c2w.compute_wth(larr,cl_dlao,tharr_th)
    wth_th_qso =c2w.compute_wth(larr,cl_qso ,tharr_th)
    np.savetxt(outdir+"wth_th.txt",np.transpose([tharr_th,wth_th_dlao,wth_th_qso]))
tharr_th,wth_th_dlao,wth_th_qso=np.loadtxt(outdir+"wth_th.txt",unpack=True)
wth_th_dla=wth_th_dlao+wth_th_qso

plt.figure()
plt.plot(zarr_dlao,nzarr_dlao)
plt.plot(zarr_qso,nzarr_qso)

plt.figure()
plt.plot(larr,cl_dla ,'r-')
plt.plot(larr,cl_dlao,'b-')
plt.plot(larr,cl_qso ,'g-')

plt.figure()
plt.plot(tharr_th,wth_th_dla ,'r-',lw=2)
plt.plot(tharr_th,wth_th_qso ,'g-',lw=2)
plt.plot(tharr_th,wth_th_dlao,'b-',lw=2)
plt.errorbar(tharr,wth_dla ,yerr=np.sqrt(np.diag(covar_dla)) ,fmt='ro',label='DLA+QSO')
plt.errorbar(tharr,wth_qso ,yerr=np.sqrt(np.diag(covar_qso)) ,fmt='go',label='QSO')
plt.errorbar(tharr,wth_dlao,yerr=np.sqrt(np.diag(covar_dlao)),fmt='bo',label='DLA')
plt.plot([-1,-1],[-1,-1],'k-',lw=2,label='Theory, $b_{\\rm DLA}=2$')
plt.xlim([0,1])
plt.ylim([-0.005,0.025])
plt.xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=18)
plt.ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle$',fontsize=18)
plt.legend(loc='upper right',frameon=False,fontsize=16)

#Binning theory
dth=tharr[1]-tharr[0]
wth_f_dlao=interp1d(tharr_th,tharr_th*wth_th_dlao,bounds_error=False,fill_value=0)
wth_pr_dlao=np.array([quad(wth_f_dlao,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

#Data vectors and covariances
i_good=np.where(tharr<th_thr)[0]; ndof=len(i_good)
dv=wth_dlao[i_good]; tv=wth_pr_dlao[i_good]; cv=(covar_dlao[i_good,:])[:,i_good]; icv=np.linalg.inv(cv)
#chi^2
chi2_null=np.dot(dv,np.dot(icv,dv))
chi2_pred=np.dot(dv-tv,np.dot(icv,dv-tv))
#Analytic Best-fit and errors
sigma_b=1./np.sqrt(np.dot(tv,np.dot(icv,tv)))
b_bf=np.dot(tv,np.dot(icv,dv))/np.dot(tv,np.dot(icv,tv))
print chi2_null/ndof,chi2_pred/ndof,chi2_null-chi2_pred
print 1-st.chi2.cdf(chi2_null,ndof),1-st.chi2.cdf(chi2_pred,ndof)
print "b_DLA = %.3lf +- %.3lf"%(2*b_bf,2*sigma_b)

plt.figure()
plt.plot(tharr_th,b_bf*wth_th_dlao,'b-',lw=2,label='Best fit')
plt.errorbar(tharr[i_good],wth_dlao[i_good],yerr=np.sqrt(np.diag(covar_dlao)[i_good]),fmt='bo',label='Data')
plt.xlim([0,th_thr])
plt.ylim([-0.005,0.025])
plt.xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=18)
plt.ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle$',fontsize=18)
plt.legend(loc='upper right',frameon=False,fontsize=16)

plt.show()
