import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import common as cmm
from scipy.interpolate import interp1d
from astropy.io import fits

bqso=2.5
bdla=2.

class Exper(object) :
    def __init__(self,ndla,nqso,fname_noise_cmbl,fsky) :
        self.ndla=ndla
        self.nqso=nqso
        data=np.loadtxt(fname_noise_cmbl,unpack=True)
        self.nlcmb=interp1d(data[0],data[1],bounds_error=False,fill_value=0)#1E100)
        self.clcmb=interp1d(data[0],data[2],bounds_error=False,fill_value=0)#1E100)
        self.fsky=fsky

    def cl_noise_dla(self,larr) :
        return 4*np.pi*self.fsky*np.ones_like(larr)/(self.ndla+0.)

    def cl_noise_qso(self,larr) :
        return 4*np.pi*self.fsky*np.ones_like(larr)/(self.nqso+0.)

    def cl_noise_cmb(self,larr) :
        return self.nlcmb(larr)

    def cl_snoise_cmb(self,larr) :
        return self.clcmb(larr)

xp_pl_dr12=Exper(3.4E4,1.5E5,cmm.fname_kappa_cl,0.25)
data_dla=(fits.open(cmm.fname_dla_n12))[1].data
def get_nz_oversample(bins,nzar,nbins) :
    zsub=(bins[1:]+bins[:-1])*0.5
    nsub=(nzar+0.0)/((np.sum(nzar)+0.)*(bins[1]-bins[0]))
    nf=interp1d(zsub,nsub,bounds_error=False,fill_value=0)
    zarr=bins[0]+(bins[-1]-bins[0])*np.arange(nbins)/(nbins-1.)
    pzarr=nf(zarr)
    return zarr,pzarr
nz,bn=np.histogram(data_dla['z_abs'],range=[0,7],bins=50); zarr_dlo,nzarr_dlo=get_nz_oversample(bn,nz,256)
bzarr_dlo=cmm.bias_dla(zarr_dlo)
nz,bins=np.histogram(data_dla['zqso'],range=[0,7],bins=50); zarr_qso,nzarr_qso=get_nz_oversample(bn,nz,256)
bzarr_qso=cmm.bias_qso(zarr_qso)

#cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)#,transfer_function='eisenstein_hu')
clt_dlo=ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo,nzarr_dlo),(zarr_dlo,bzarr_dlo))
clt_qso=ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso,nzarr_qso),(zarr_qso,bzarr_qso))
clt_cmbl=ccl.ClTracerCMBLensing(cosmo)

def get_fisher(xp,ll) :
    cl_dd=ccl.angular_cl(cosmo,clt_dlo ,clt_dlo ,ll,l_limber=-1)*bdla**2
    cl_dq=ccl.angular_cl(cosmo,clt_dlo ,clt_qso ,ll,l_limber=-1)*bdla*bqso
    cl_dc=ccl.angular_cl(cosmo,clt_dlo ,clt_cmbl,ll,l_limber=-1)*bdla
    cl_qq=ccl.angular_cl(cosmo,clt_qso ,clt_qso ,ll,l_limber=-1)*bqso**2
    cl_qc=ccl.angular_cl(cosmo,clt_qso ,clt_cmbl,ll,l_limber=-1)*bqso
    cl_cc=ccl.angular_cl(cosmo,clt_cmbl,clt_cmbl,ll,l_limber=-1)
    cl_aa=cl_dd+2*cl_dq+cl_qq
    cl_aq=cl_dq+cl_qq
    cl_ac=cl_dc+cl_qc

    nl_dd=0*cl_dd+xp.cl_noise_dla(ll) 
    nl_dc=1*cl_dc
    nl_aa=0*cl_aa+xp.cl_noise_dla(ll)
    nl_aq=0*cl_aq+0./(1./xp.cl_noise_dla(ll)+1./xp.cl_noise_qso(ll))
    nl_ac=1*cl_ac
    nl_qq=0*cl_qq+xp.cl_noise_qso(ll)
    nl_qc=1*cl_qc
    nl_cc=1*cl_cc+xp.cl_noise_cmb(ll)

    dvecdd=np.array([cl_dc/bdla,np.zeros_like(cl_dc)])
    dvecdq=np.array([cl_qc/bqso,cl_qc/bqso])
    dvec  =np.array([dvecdd,dvecdq])
    covar =np.array([[nl_aa*nl_cc+nl_ac*nl_ac,nl_aq*nl_cc+nl_ac*nl_qc],
                     [nl_aq*nl_cc+nl_ac*nl_qc,nl_qq*nl_cc+nl_qc*nl_qc]])
    covar*=(1./(xp.fsky*(2*ll+1.)))[None,None,:]
    
    fisher_l=np.zeros([len(ll),2,2])
    for i in np.arange(len(ll)) : 
        icovar=np.linalg.inv(covar[:,:,i])
        for i1 in np.arange(2) :
            dv1=dvec[i1,:,:]
            for i2 in np.arange(2) :
                dv2=dvec[i2,:,:]
                fisher_l[i,i1,i2]=np.dot(dv1[:,i],np.dot(icovar,dv2[:,i]))
    fisher=np.sum(fisher_l,axis=0)

    print np.mean(covar[0,1,:]/np.sqrt(covar[0,0,:]*covar[1,1,:]))
    print 1./np.sqrt(np.sum((cl_dc/bdla)**2*(2*ll+1)*0.25/(nl_dd*nl_cc+nl_dc**2)))
    print 1./np.sqrt(np.sum((cl_qc/bqso)**2*(2*ll+1)*0.25/(nl_qq*nl_cc+nl_qc**2)))
    print np.sqrt(np.diag(np.linalg.inv(fisher)))
    exit(1)
    
    plt.plot(ll,cl_dd,'r-')
    plt.plot(ll,np.sqrt(2*(cl_dd+nl_dd)/(xp.fsky*(2*ll+1.))),'r--')
    plt.plot(ll,cl_qq,'g-')
    plt.plot(ll,np.sqrt(2*(cl_qq+nl_qq)/(xp.fsky*(2*ll+1.))),'g--')
    plt.plot(ll,cl_cc,'b-')
    plt.plot(ll,np.sqrt(2*(cl_cc+nl_cc)**2/(xp.fsky*(2*ll+1.))),'b--')
    plt.loglog()
    plt.show()

get_fisher(xp_pl_dr12,np.arange(1000)+8.)
