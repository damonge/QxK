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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


if len(sys.argv)!=7 :
    print "Usage : analysis_wth.py nbins nside_cmbl use_wiener[0,1] nsims randomize_points[0,1] plot_stuff[0,1]"
    exit(1)

verbose=False
#Maximum scale (deg) in the computation of the 2PCF
thmax=1.
#Number of bins in theta
nth=int(sys.argv[1])
#Resolution of the CMB lensing map
nside_cmbl=int(sys.argv[2])
do_wiener=int(sys.argv[3])
nsims=int(sys.argv[4])
do_randomize_points=int(sys.argv[5])
do_plot_stuff=int(sys.argv[6])
#Maximum angle to use in the fit
th_thr=1.0

randomize_points=False
if do_randomize_points>0 :
    randomize_points=True
use_wiener=False
if do_wiener>0 :
    use_wiener=True
plot_stuff=False
if do_plot_stuff>0 :
    plot_stuff=True

#Create output directory
outdir="outputs_ns%d_nb%d"%(nside_cmbl,nth)
if use_wiener :
    outdir+="_wiener"
if randomize_points :
    outdir+="_randp"
outdir+="/"

fname_alldata=outdir+"wth_qxk_all"

data_dla=(fits.open(cmm.fname_dla))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data

d=np.load(fname_alldata+'.npz')

tharr=np.mean(d['randoms'][:,0,0,:],axis=0)
wth_dla=d['wth_dla']
wth_qso=d['wth_qso']
wth_dlo=wth_dla-wth_qso

if verbose :
    print " Computing covariance matrices"
mean_all=np.mean(d['randoms_2'][:,1,:],axis=0)
covar_all=np.mean(d['randoms_2'][:,1,:,None]*d['randoms_2'][:,1,None,:],axis=0)-mean_all[:,None]*mean_all[None,:]
corr_all=covar_all/np.sqrt(np.diag(covar_all)[None,:]*np.diag(covar_all)[:,None])

mean_dla=np.mean(d['randoms'][:,0,1,:],axis=0)
covar_dla=np.mean(d['randoms'][:,0,1,:,None]*d['randoms'][:,0,1,None,:],axis=0)-mean_dla[:,None]*mean_dla[None,:]
corr_dla=covar_dla/np.sqrt(np.diag(covar_dla)[None,:]*np.diag(covar_dla)[:,None])

mean_qso=np.mean(d['randoms'][:,1,1,:],axis=0)
covar_qso=np.mean(d['randoms'][:,1,1,:,None]*d['randoms'][:,1,1,None,:],axis=0)-mean_qso[:,None]*mean_qso[None,:]
corr_qso=covar_qso/np.sqrt(np.diag(covar_qso)[None,:]*np.diag(covar_qso)[:,None])

mean_qsu=np.mean(d['randoms'][:,2,1,:],axis=0)
covar_qsu=np.mean(d['randoms'][:,2,1,:,None]*d['randoms'][:,2,1,None,:],axis=0)-mean_qsu[:,None]*mean_qsu[None,:]
corr_qsu=covar_qsu/np.sqrt(np.diag(covar_qsu)[None,:]*np.diag(covar_qsu)[:,None])

mean_dlo=np.mean(d['randoms'][:,3,1,:],axis=0)
covar_dlo=np.mean(d['randoms'][:,3,1,:,None]*d['randoms'][:,3,1,None,:],axis=0)-mean_dlo[:,None]*mean_dlo[None,:]
corr_dlo=covar_dlo/np.sqrt(np.diag(covar_dlo)[None,:]*np.diag(covar_dlo)[:,None])

if plot_stuff :
#    plt.figure()
#    plt.imshow(corr_dla,origin='lower',interpolation='nearest')
#    
#    plt.figure()
#    plt.imshow(corr_qso,origin='lower',interpolation='nearest')
#    
#    plt.figure()
#    plt.imshow(corr_qsu,origin='lower',interpolation='nearest')
#    
#    plt.figure()
#    plt.imshow(corr_dlo,origin='lower',interpolation='nearest')
#
    plt.figure()
    ax=plt.gca()
    im=ax.imshow(corr_all,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.text(0.27,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes)
    ax.text(0.27,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes)
    ax.text(0.77,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes)
    ax.text(0.77,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes)
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=14)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig("doc/corrmat_wth.pdf",bbox_inches='tight')

if verbose :
    print " Computing theory prediction"
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

if verbose :
    print "   Cls"
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

if verbose :
    print "   w(theta)"
if not os.path.isfile(outdir+"wth_th.txt") :
    tharr_th=2.*np.arange(256)/255.
    wth_th_dlo=c2w.compute_wth(larr,cl_dc,tharr_th)
    wth_th_qso =c2w.compute_wth(larr,cl_qc ,tharr_th)
    np.savetxt(outdir+"wth_th.txt",np.transpose([tharr_th,wth_th_dlo,wth_th_qso]))
tharr_th,wth_th_dlo,wth_th_qso=np.loadtxt(outdir+"wth_th.txt",unpack=True)
wth_th_dla=wth_th_dlo+wth_th_qso

'''
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
'''
    
#Binning theory
dth=tharr[1]-tharr[0]
wth_f_dlo=interp1d(tharr_th,tharr_th*wth_th_dlo,bounds_error=False,fill_value=0)
wth_pr_dlo=np.array([quad(wth_f_dlo,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

wth_f_qso=interp1d(tharr_th,tharr_th*wth_th_qso,bounds_error=False,fill_value=0)
wth_pr_qso=np.array([quad(wth_f_qso,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

wth_f_dla=interp1d(tharr_th,tharr_th*wth_th_dla,bounds_error=False,fill_value=0)
wth_pr_dla=np.array([quad(wth_f_dla,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

i_good=np.where(tharr<th_thr)[0]; ndof=len(i_good);


#Fitting the 2PCF difference
#Data vectors and covariances
dv=wth_dlo[i_good]; tv=wth_pr_dlo[i_good]; cv=(covar_dlo[i_good,:])[:,i_good]; icv=np.linalg.inv(cv)
#chi^2
#Analytic Best-fit and errors
sigma_b=1./np.sqrt(np.dot(tv,np.dot(icv,tv)))
b_bf=np.dot(tv,np.dot(icv,dv))/np.dot(tv,np.dot(icv,tv))
pv=b_bf*tv
chi2=np.dot(dv-pv,np.dot(icv,dv-pv))
pte=1-st.chi2.cdf(chi2,ndof-1)
print " Bias from QSO-DLA difference"
print "  b_DLA = %.3lf +- %.3lf"%(b_bf,sigma_b)
print "  chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2,ndof-1,pte)
print " ------"
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    ax.errorbar(tharr,1E3*wth_dlo,yerr=1E3*np.sqrt(np.diag(covar_dlo)),fmt='ko',
                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
    ax.plot(tharr_th,b_bf*1E3*wth_th_dlo,'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,th_thr])
    ax.set_ylim([-0.1,0.35])
    ax.set_xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=14)
    ax.set_ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle\\times10^3$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig("doc/wth_x1.pdf",bbox_inches='tight')

#Fitting both 2PCFs with b_DLA and b_QSO
dv=np.concatenate((wth_dla[i_good],wth_qso[i_good]));
tv1=np.concatenate((wth_pr_dlo[i_good],np.zeros(ndof)));
tv2=np.concatenate((wth_pr_qso[i_good],wth_pr_qso[i_good]))
tv=np.array([tv1,tv2])
cv=np.zeros([2*ndof,2*ndof])
cv[:ndof,:ndof]=covar_all[:ndof,:ndof]
cv[:ndof,ndof:]=covar_all[:ndof,nth:nth+ndof]
cv[ndof:,:ndof]=covar_all[nth:nth+ndof,:ndof]
cv[ndof:,ndof:]=covar_all[nth:nth+ndof,nth:nth+ndof]
#cv=covar_all
icv=np.linalg.inv(cv)

cov_b=np.linalg.inv(np.dot(tv,np.dot(icv,np.transpose(tv))))
b_bf=np.dot(cov_b,np.dot(tv,np.dot(icv,np.transpose(dv))))
sigma_b=np.sqrt(np.diag(cov_b))
pv=np.dot(b_bf,tv);
chi2=np.dot((dv-pv),np.dot(icv,(dv-pv)));
pte=1-st.chi2.cdf(chi2,2*ndof-2)

print " Bias from simultaneous fit to QSOs and DLAs"
print "  b_DLA = %.3lf +- %.3lf"%(b_bf[0],sigma_b[0])
print "  b_QSO = %.3lf +- %.3lf"%(b_bf[1],sigma_b[1])
print "  r = %.3lE"%(cov_b[0,1]/(sigma_b[0]*sigma_b[1]))
print "  chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2,2*ndof-2,pte)
print " "
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    ax.plot(tharr_th,b_bf[0]*1E3*wth_th_dlo+b_bf[1]*1E3*wth_th_qso,'b-',lw=2)
    ax.plot(tharr_th,b_bf[1]*1E3*wth_th_qso,'r-',lw=2)
    ax.errorbar(tharr,1E3*wth_dla,yerr=1E3*np.sqrt(np.diag(covar_all[:nth,:nth])),fmt='bo',label='$\\kappa\\times{\\rm DLA}$')
    ax.errorbar(tharr,1E3*wth_qso,yerr=1E3*np.sqrt(np.diag(covar_all[nth:,nth:])),fmt='ro',label='$\\kappa\\times{\\rm QSO}$')
    ax.plot([-1,-1],[-1,-1],'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,th_thr])
    ax.set_ylim([-0.1,0.5])
    ax.set_xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=14)
    ax.set_ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle\\times10^3$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig("doc/wth_x2.pdf",bbox_inches='tight')

if plot_stuff :
    plt.show()
