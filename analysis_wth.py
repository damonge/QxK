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

if len(sys.argv)!=4 :
    print "Usage : analysis_wth.py nbins nside_cmbl use_wiener"
    exit(1)

plot_stuff=True
#Number of DLAs in the catalog
n_dla=34050
#Number of QSOs in the catalog
n_qso=297301
#Number of QSOs in the UNIFORM catalog
n_qsu=107513
#Maximum scale (deg) in the computation of the 2PCF
thmax=1.
#Number of bins in theta
nth=int(sys.argv[1])
#Resolution of the CMB lensing map
nside_cmbl=int(sys.argv[2])
do_wiener=int(sys.argv[3])
#Maximum angle to use in the fit
th_thr=1.0
#Do we compute the 2PCF using the slow python implementation?
compute_with_python=False

use_wiener=False
if do_wiener>0 :
    use_wiener=True

#Create output directory
if use_wiener :
    outdir="outputs_ns%d_nb%d_wiener/"%(nside_cmbl,nth)
else :
    outdir="outputs_ns%d_nb%d/"%(nside_cmbl,nth)
os.system("mkdir -p "+outdir)

#Generate data in right format
fname_cmbl,fname_mask_cmbl=cmm.reform_data(nside_cmbl,use_wiener=use_wiener)

print "Reading kappa map and mask"
mask =hp.read_map(fname_mask_cmbl,verbose=False)
field=hp.read_map(fname_cmbl,verbose=False)

print "Reading QSO mask"
mask_qso=hp.read_map(cmm.fname_mask_qso,verbose=False)
ndens_qso=n_qso/(4*np.pi*np.mean(mask_qso))
ndens_dla=n_dla/(4*np.pi*np.mean(mask_qso))

print "Computing the DLA 2PCF"
data_dla=(fits.open(cmm.fname_dla))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data
data_dla=(fits.open(cmm.fname_dla))[1].data
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
        cmm.random_points(1000+isim,mask_qso,n_dla,fname_out=outdir+'dla_random_%d.fits'%isim,
                          weights=data_dla['W'])
        cmm.random_points(1000+isim,mask_qso,n_qso,fname_out=outdir+'qso_random_%d.fits'%isim,
                          weights=data_qso['W'])
        cmm.random_points(1000+isim,mask_qso,n_qsu,fname_out=outdir+'qsu_random_%d.fits'%isim,
                          weights=None)
        cmm.random_map(1000+isim,mask,cmm.fname_kappa_cl,fname_out=outdir+'map_random_%d.fits'%isim,
                       use_wiener=use_wiener)

    th_dla,w_dla,hf_dla,hm_dla=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                   outdir+'dla_random_%d.fits'%isim,thmax,nth,
                                                   fname_out=fname_dla)
    th_qso,w_qso,hf_qso,hm_qso=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                   outdir+'qso_random_%d.fits'%isim,thmax,nth,
                                                   fname_out=fname_qso)
    th_qsu,w_qsu,hf_qsu,hm_qsu=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                   outdir+'qsu_random_%d.fits'%isim,thmax,nth,
                                                   weight_name='NO_WEIGHT',fname_out=fname_qsu)

    if cleanup :
        os.system('rm '+
                  outdir+'dla_random_%d.fits '%isim+outdir+'qso_random_%d.fits '%isim+
                  outdir+'qsu_random_%d.fits '%isim+outdir+'map_random_%d.fits '%isim)

    return th_dla,w_dla,hf_dla,hm_dla,th_qso,w_qso,hf_qso,hm_qso,th_qsu,w_qsu,hf_qsu,hm_qsu

nsims=1000
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

print  "Computing covariance matrices"
mean_all=np.mean(data_randoms_2[:,1,:],axis=0)
covar_all=np.mean(data_randoms_2[:,1,:,None]*data_randoms_2[:,1,None,:],axis=0)-mean_all[:,None]*mean_all[None,:]
corr_all=covar_all/np.sqrt(np.diag(covar_all)[None,:]*np.diag(covar_all)[:,None])

mean_dla=np.mean(data_randoms[:,0,1,:],axis=0)
covar_dla=np.mean(data_randoms[:,0,1,:,None]*data_randoms[:,0,1,None,:],axis=0)-mean_dla[:,None]*mean_dla[None,:]
corr_dla=covar_dla/np.sqrt(np.diag(covar_dla)[None,:]*np.diag(covar_dla)[:,None])

mean_qso=np.mean(data_randoms[:,1,1,:],axis=0)
covar_qso=np.mean(data_randoms[:,1,1,:,None]*data_randoms[:,1,1,None,:],axis=0)-mean_qso[:,None]*mean_qso[None,:]
corr_qso=covar_qso/np.sqrt(np.diag(covar_qso)[None,:]*np.diag(covar_qso)[:,None])

mean_qsu=np.mean(data_randoms[:,2,1,:],axis=0)
covar_qsu=np.mean(data_randoms[:,2,1,:,None]*data_randoms[:,2,1,None,:],axis=0)-mean_qsu[:,None]*mean_qsu[None,:]
corr_qsu=covar_qsu/np.sqrt(np.diag(covar_qsu)[None,:]*np.diag(covar_qsu)[:,None])

mean_dlo=np.mean(data_randoms[:,3,1,:],axis=0)
covar_dlo=np.mean(data_randoms[:,3,1,:,None]*data_randoms[:,3,1,None,:],axis=0)-mean_dlo[:,None]*mean_dlo[None,:]
corr_dlo=covar_dlo/np.sqrt(np.diag(covar_dlo)[None,:]*np.diag(covar_dlo)[:,None])

if plot_stuff :
    plt.figure()
    plt.imshow(corr_dla,origin='lower',interpolation='nearest')
    
    plt.figure()
    plt.imshow(corr_qso,origin='lower',interpolation='nearest')
    
    plt.figure()
    plt.imshow(corr_qsu,origin='lower',interpolation='nearest')
    
    plt.figure()
    plt.imshow(corr_dlo,origin='lower',interpolation='nearest')

    plt.figure()
    plt.imshow(corr_all,origin='lower',interpolation='nearest')

print "Computing theory prediction"
def get_nz_oversample(bins,nzar,nbins) :
    zsub=(bins[1:]+bins[:-1])*0.5
    nsub=(nzar+0.0)/((np.sum(nzar)+0.)*(bins[1]-bins[0]))
    nf=interp1d(zsub,nsub,bounds_error=False,fill_value=0)
    zarr=bins[0]+(bins[-1]-bins[0])*np.arange(nbins)/(nbins-1.)
    pzarr=nf(zarr)
    return zarr,pzarr
nz,bn=np.histogram(data_dla['z_abs'],range=[0,7],bins=50)
zarr_dlo,nzarr_dlo=get_nz_oversample(bn,nz,256)
bzarr_dlo=cmm.bias_dla(zarr_dlo)
nz,bins=np.histogram(data_dla['zqso'],range=[0,7],bins=50)
zarr_qso,nzarr_qso=get_nz_oversample(bn,nz,256)
bzarr_qso=cmm.bias_qso(zarr_qso)

print "  Cls"
ll,nll,cll=np.loadtxt(cmm.fname_kappa_cl,unpack=True)
cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
wl=(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10)
lmx_wl=ll[-1]
def get_wiener(ell) :
    ret=np.zeros(len(ell))
    ids_good=np.where(ell<=lmx_wl)[0]
    ret[ids_good]=wl[(ell.astype(int))[ids_good]]
    return ret

if not os.path.isfile(outdir+"cls_th.txt") :
    cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
    clt_dlo =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo,nzarr_dlo),(zarr_dlo,bzarr_dlo))
    clt_qso =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso,nzarr_qso),(zarr_qso,bzarr_qso))
    clt_cmbl=ccl.ClTracerCMBLensing(cosmo)
    larr     =np.concatenate((1.*np.arange(500),500+10*np.arange(950)))
    if use_wiener :
        wil=get_wiener(larr)
    else :
        wil=np.ones(len(larr))
    cl_dd=ccl.angular_cl(cosmo,clt_dlo ,clt_dlo ,larr,l_limber=-1)
    cl_dc=ccl.angular_cl(cosmo,clt_dlo ,clt_cmbl,larr,l_limber=-1)*wil
    cl_qq=ccl.angular_cl(cosmo,clt_qso ,clt_qso ,larr,l_limber=-1)
    cl_qc=ccl.angular_cl(cosmo,clt_qso ,clt_cmbl,larr,l_limber=-1)*wil
    cl_cc=ccl.angular_cl(cosmo,clt_cmbl,clt_cmbl,larr,l_limber=-1)*wil**2
    np.savetxt(outdir+"cls_th.txt",np.transpose([larr,cl_dc,cl_qc,cl_dd,cl_qq,cl_cc]))
larr,cl_dc,cl_qc,cl_dd,cl_qq,cl_cc=np.loadtxt(outdir+"cls_th.txt",unpack=True)
#nl_qq=np.ones_like(cl_qq)/ndens_qso
#nl_dd=np.ones_like(cl_dd)/ndens_dla
#if plot_stuff :
#    plt.figure()
#    plt.plot(larr,cl_dd,'m-')
#    plt.plot(larr,nl_dd,'m--')
#    plt.plot(larr,cl_qq,'c-')
#    plt.plot(larr,nl_qq,'c--')
#    plt.plot(larr,cl_qq+nl_qq,'c-.')
#    plt.plot(larr,cl_cc,'y-')
#    plt.plot(larr,cl_dc,'b-')
#    plt.plot(larr,cl_qc,'g-')
#    plt.plot(larr,cl_qc+cl_dc,'r-')
#    plt.loglog()
#    plt.show();
#cl_dla=cl_dc+cl_qc

print "  w(theta)"
if not os.path.isfile(outdir+"wth_th.txt") :
    tharr_th=2.*np.arange(256)/255.
    wth_th_dlo=c2w.compute_wth(larr,cl_dc,tharr_th)
    wth_th_qso =c2w.compute_wth(larr,cl_qc ,tharr_th)
    np.savetxt(outdir+"wth_th.txt",np.transpose([tharr_th,wth_th_dlo,wth_th_qso]))
tharr_th,wth_th_dlo,wth_th_qso=np.loadtxt(outdir+"wth_th.txt",unpack=True)
wth_th_dla=wth_th_dlo+wth_th_qso

if plot_stuff :
    plt.figure()
    plt.plot(zarr_dlo,nzarr_dlo)
    plt.plot(zarr_qso,nzarr_qso)
    
    plt.figure()
    plt.plot(tharr_th,wth_th_dla ,'r-',lw=2)
    plt.plot(tharr_th,wth_th_qso ,'g-',lw=2)
    plt.plot(tharr_th,wth_th_dlo,'b-',lw=2)
    plt.errorbar(tharr,wth_dla,yerr=np.sqrt(np.diag(covar_dla)),fmt='ro',label='DLA+QSO')
    plt.errorbar(tharr,wth_qso,yerr=np.sqrt(np.diag(covar_qso)),fmt='go',label='QSO')
    plt.errorbar(tharr,wth_dlo,yerr=np.sqrt(np.diag(covar_dlo)),fmt='bo',label='DLA')
    plt.plot([-1,-1],[-1,-1],'k-',lw=2,label='Theory, $b_{\\rm DLA}=2$')
    plt.xlim([0,1])
    plt.ylim([-0.005,0.025])
    plt.xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=18)
    plt.ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle$',fontsize=18)
    plt.legend(loc='upper right',frameon=False,fontsize=16)
    
#Binning theory
dth=tharr[1]-tharr[0]
wth_f_dlo=interp1d(tharr_th,tharr_th*wth_th_dlo,bounds_error=False,fill_value=0)
wth_pr_dlo=np.array([quad(wth_f_dlo,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

wth_f_qso=interp1d(tharr_th,tharr_th*wth_th_qso,bounds_error=False,fill_value=0)
wth_pr_qso=np.array([quad(wth_f_qso,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

wth_f_dla=interp1d(tharr_th,tharr_th*wth_th_dla,bounds_error=False,fill_value=0)
wth_pr_dla=np.array([quad(wth_f_dla,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

i_good=np.where(tharr<th_thr)[0]; ndof=len(i_good)


#Fitting the 2PCF difference
#Data vectors and covariances
dv=wth_dlo[i_good]; tv=wth_pr_dlo[i_good]; cv=(covar_dlo[i_good,:])[:,i_good]; icv=np.linalg.inv(cv)
#chi^2
#Analytic Best-fit and errors
sigma_b=1./np.sqrt(np.dot(tv,np.dot(icv,tv)))
b_bf=np.dot(tv,np.dot(icv,dv))/np.dot(tv,np.dot(icv,tv))
print "DLA-QSO"
print b_bf,sigma_b
pv=b_bf*tv
chi2=np.dot(dv-pv,np.dot(icv,dv-pv))
pte=1-st.chi2.cdf(chi2,ndof-1)
print chi2,ndof-1,pte
if plot_stuff :
    plt.figure()
    plt.plot(tharr_th,b_bf*wth_th_dlo,'b-',lw=2,label='Best fit')
    plt.errorbar(tharr[i_good],dv,yerr=np.sqrt(np.diag(covar_dlo)[i_good]),fmt='bo',label='Data')
    plt.xlim([0,th_thr])
    plt.ylim([-0.005,0.025])
    plt.xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=18)
    plt.ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle$',fontsize=18)
    plt.legend(loc='upper right',frameon=False,fontsize=16)

#Fitting both 2PCFs with b_DLA and b_QSO
dv=np.concatenate((wth_dla[i_good],wth_qso[i_good]));
tv1=np.concatenate((wth_pr_dlo[i_good],np.zeros(ndof)));
tv2=np.concatenate((wth_pr_qso[i_good],wth_pr_qso[i_good]))
tv=np.array([tv1,tv2])
cv=covar_all
#cv[:ndof,:ndof]=(covar_dla[i_good,:])[:,i_good];
#cv[ndof:,ndof:]=(covar_qso[i_good,:])[:,i_good];
icv=np.linalg.inv(cv)

cov_b=np.linalg.inv(np.dot(tv,np.dot(icv,np.transpose(tv))))
b_bf=np.dot(cov_b,np.dot(tv,np.dot(icv,np.transpose(dv))))

print "DLA+QSO"
print b_bf,np.sqrt(np.diag(cov_b))
print cov_b/np.sqrt(np.diag(cov_b)[None,:]*np.diag(cov_b)[:,None])
pv=np.dot(b_bf,tv);
chi2=np.dot((dv-pv),np.dot(icv,(dv-pv)));
pte=1-st.chi2.cdf(chi2,2*ndof-2)
print chi2,2*ndof-2,pte

if plot_stuff :
    plt.show()
