import numpy as np
import healpy as hp
import pymaster as nmt
import os
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.integrate import quad

def compute_xcorr_c(fname_field,fname_mask,fname_catalog,thmax_deg,nbins,logbin=False,
                    thmin_deg=0,fname_out='none',weight_name='W',cut_name='NO_CUT') :
    """Computes 2PCF between a pixelized field and a set of points. Uses fast C implementation.
    fname_field : path to healpix map containing the field
    fname_mask : path to healpix map containing the field's mask
    fname_catalog : path to FITS file containing the points
    thmax_deg : maximum angular separation in degrees
    nbins : number of bins in angular separation
    logbin : should I use logarithmic bins?
    thmin_deg : minimum angular separation in degrees
    fname_out : path to output filename
    cut_name : column name corresponding to a cuts flag. Only objects with that flag!=0 will be used.
    weight_name : column name corresponding to the weights.
    """

    if (fname_out!='none') and (os.path.isfile(fname_out)) :
        th,wth,hf,hm=np.loadtxt(fname_out,unpack=True)
    else :
        do_log=0
        if logbin : do_log=1
        command ="./FieldXCorr "
        command+=fname_field+" "
        command+=fname_mask+" "
        command+=fname_catalog+" "
        command+=fname_out+" "
        command+="%lf %d %d %lf "%(thmax_deg,nbins,do_log,thmin_deg)
        command+=weight_name+" "
        command+=cut_name+" "
        command+=" > "+fname_out+"_log"
        os.system(command)
        th,wth,hf,hm=np.loadtxt(fname_out,unpack=True)

    return th,wth,hf,hm

def random_map(seed,mask,fname_cl,fname_out='none',use_wiener=False) :
    """Generates gaussian random field with correct power spectrum
    mask : healpix map containing a binary mask
    fname_cl : path to ascii file containing the field's power spectrum
    fname_out : path to output FITS file
    """
    np.random.seed(seed)
    nside=hp.npix2nside(len(mask))
    ll,nll,cll=np.loadtxt(fname_cl,unpack=True)
    cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
    nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
    if use_wiener :
        alm=hp.synalm(cl,lmax=2048,new=True,verbose=False) 
        alm=hp.almxfl(alm,(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10))
        mp=hp.alm2map(alm,nside,lmax=2048)
    else :
        mp=hp.synfast(cl,nside,lmax=2048,new=True,verbose=False)
    mp*=mask

    if fname_out!='none' :
        hp.write_map(fname_out,mp)

    return mp

def delta_n_map(lat,lon,nside,completeness_map=None,completeness_thr=0.1,weights=None,rot=None) :
    """
    Computes overdensity map from a catalog.

    lat : array of latitude values for each object
    lon : array of longitude values for each object
    nside : resolution of output map
    completeness_map : map (with resolution nside) defining the completeness in each pixel.
                       If None, 100% completeness will be assumed across the full map
    completeness_thr : completeness threshold defining which pixels to include. All pixels
                       below this threshold will be masked.
    weights : array of weights for each object. If None, unit weights will be used
    rot : rotation to apply to (lat,lon) before generating map. If None, no rotation will
          be applied.

    returns : map of delta_n
    """
    npix=hp.nside2npix(nside)
    
    #Generate spherical coordinates
    th0=np.pi*(90-lat)/180; ph0=np.pi*lon/180
    if rot is not None :
        th0,ph0=rot(th0,ph0)
    #Transform coordinates into pixel ids
    ipix=hp.ang2pix(nside,th0,ph0)

    #Bin particles into map
    if weights is None :
        weights=np.ones_like(lat)
    mpn=np.bincount(ipix,minlength=npix,weights=weights)+0.

    #Generate mask
    mask=np.ones(npix)
    if completeness_map is not None :
        if len(completeness_map)!=npix :
            raise ValueError("Completeness map has wrong length")
        mask[completeness_map<completeness_thr]=0.
    else :
        completeness_map=np.ones(npix)
    ipix_unmasked=np.where(mask>0.01)[0]
        
    #Generate mean number map
    nmean=np.sum(mpn*mask)/np.sum(completeness_map*mask)
    mpm=nmean*completeness_map*mask

    #Generate delta map
    mpd=np.zeros(npix)
    mpd[ipix_unmasked]=mpn[ipix_unmasked]/mpm[ipix_unmasked]-1

    return mpd

def compute_cell(mp1,mp2,bpws,mask=None,are_fields=False,workspace=None) :
    """
    Computes power spectrum of two scalar maps.

    mp1 : first map
    mp2 : second map
    bpws : a NmtBin object describing the bandpowers of the output power spectrum.
    mask : weights map to use in the computation of the PCL. This parameter must be passed unless are_fields=True.
    are_fields : if True, mp1 and mp2 are already NmtField objects. Otherwise, they are maps and their NmtFields will be computed.
    workspace : if None, a new NmtWorkspace object will be computed to estimate the PCL. Otherwise, pass a valid NmtWorkspace (with which you have already computed the mode-coupling matrix) to speed up the calculation.
    
    returns: array containing the power spectrum
    """
    
    if are_fields :
        fld1=mp1; fld2=mp2
    else :
        fld1=nmt.NmtField(mask,[mp1])
        fld2=nmt.NmtField(mask,[mp2])

    if workspace is None :
        workspace=nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(fld1,fld2,bpws)

    cl_decoupled=workspace.decouple_cell(nmt.compute_coupled_cell(fld1,fld2))

    return cl_decoupled[0]

def compute_theory(z,nz,bz,cosmo,x_out,dx=None,return_correlation=False,filter_function=None) :
    """
    Computes theoretical 2-point statistic for kappa x delta

    z : array of redshifts
    nz : redshift distribution (arbitrary normalization) sampled at z
    bz : bias as a function of redshift sampled at z
    cosmo : a ccl Cosmology object
    x_out : array of scales (either ell or theta) at which you want to sample the 2-point statistic
    dx : width of the bins in x (should be a scalar)
    return_correlation : if True, the correlation function will be computed (with x_out giving the values of the angular separation in degrees). Otherwise, the power spectrum will be computed (with x_out giving the multipoles at which it's estimated.
    filter_function : if not None, this should be a function that takes ell and returns a filter w(ell). The power spectra will then be applied this filter before computing correlation functions if return_correlation is True.
    """

    #Create tracers
    clt_d=ccl.ClTracerNumberCounts(cosmo,False,False,(z,nz),(z,bz))
    clt_k=ccl.ClTracerCMBLensing(cosmo,z_source=1100.)

    #Compute power spectrum for a large range of ells
    larr=np.concatenate((1.*np.arange(500),500+10.*np.arange(950)))
    cell=ccl.angular_cl(cosmo,clt_d,clt_k,larr)

    #Either sample C_ell at the relevant scales or compute correlation function
    if return_correlation :
        if filter_function is None :
            wf=np.ones_like(cell)
        else :
            wf=filter_function(larr)
            
        if dx is None :
            thmax=x_out[-1]
        else :
            thmax=x_out[-1]+0.5*dx
        tharr=thmax*np.arange(256)/255.
        wtharr=ccl.correlation(cosmo,larr,cell*wf,tharr,method='Bessel')
        func2p=interp1d(tharr,wtharr,bounds_error=False,fill_value=0)
    else :
        func2p=interp1d(larr,cell,bounds_error=False,fill_value=0)
        
    if dx is None :
        ret2p=func2p(x_out)
    else :
        ret2p=np.array([quad(func2p,x-0.5*dx,x+0.5*dx)[0]/dx for x in x_out])

    return ret2p
