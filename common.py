import numpy as np
import healpy as hp
import os
import pyfits as pf
import healpy as hp
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

nside_qso=64

fname_mask_cmbl_orig='data/mask.fits'
fname_mask_cmbl_prefix='data/mask'
fname_kappa_cmbl_prefix='data/kappa'
fname_alm_cmbl='data/dat_klm.fits'
fname_dla_n12_orig='data/DLA_DR12_v2.dat'
fname_dla_g16_orig='data/table3.dat'
fname_qso_orig='data/DR12Q.fits'
fname_dla_n12='data/DLA_DR12_n12.fits'
fname_dla_n12b='data/DLA_DR12_n12b.fits'
fname_dla_g16='data/DLA_DR12_g16.fits'
fname_qso='data/QSO_DR12.fits'
fname_mask_qso='data/mask_QSO_ns%d.fits'%nside_qso
fname_mask_cmass='data/msk_cmass.fits'
fname_kappa_cl='data/nlkk.dat'

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

def compute_xcorr_py(pos1,w1,map2,mask2,thmax_deg,nbins,nside,logbin=False,thmin_deg=0,fname_out='none') :
    """Computes 2PCF between a pixelized field and a set of points. Uses slower python implementation.
    pos1 : 3D positions of the points (unit vectors)
    w1 : weights for each point
    map2 : healpix map of the field
    mask2 : healpix map of the field's mask
    nside : healpix resolution parameter
    thmax_deg : maximum angular separation in degrees
    nbins : number of bins in angular separation
    logbin : should I use logarithmic bins?
    thmin_deg : minimum angular separation in degrees
    fname_out : path to output filename
    """

    if (fname_out!='none') and (os.path.isfile(fname_out)) :
        th,wth,hf_sum,hm_sum=np.loadtxt(fname_out,unpack=True)
    else :
        hf_sum=np.zeros(nbins)
        hn_sum=np.zeros(nbins)
        thmax=thmax_deg*np.pi/180
        thmin=thmin_deg*np.pi/180
        thmax_extend=thmax*1.2
        
        ipix=np.arange(hp.nside2npix(nside))
        pospix=np.array(hp.pix2vec(nside,ipix))

        if logbin :
            logthmax=np.log10(thmax)
            logthmin=np.log10(thmin)
            for i in np.arange(len(pos1)) :
                pos=pos1[i,:]
                w=w1[i]
                disc=hp.query_disc(nside,pos,thmax_extend)
                o_m_cth=1-np.minimum(np.dot(pos,pospix[:,disc]),1)
                lth=0.5*np.log10(2*o_m_cth+o_m_cth*o_m_cth*0.33333+o_m_cth*o_m_cth*o_m_cth*0.0888888)
                
                n=mask2[disc]
                f=map2[disc]
                hn,bins=np.histogram(lth,range=[logthmin,logthmax],bins=nbins,weights=n)
                hf,bins=np.histogram(lth,range=[logthmin,logthmax],bins=nbins,weights=f)
                hn_sum+=hn*w
                hf_sum+=hf*w
                
                th=10.**(0.5*(bins[1:]+bins[:-1]))*180/np.pi
        else :
            for i in np.arange(len(pos1)) :
                pos=pos1[i,:]
                w=w1[i]
                disc=hp.query_disc(nside,pos,thmax_extend)
                o_m_cth=1-np.minimum(np.dot(pos,pospix[:,disc]),1)
                th=np.sqrt(2*o_m_cth+o_m_cth*o_m_cth*0.33333+o_m_cth*o_m_cth*o_m_cth*0.0888888)
                
                n=mask2[disc]
                f=map2[disc]
                hn,bins=np.histogram(th,range=[0,thmax],bins=nbins,weights=n)
                hf,bins=np.histogram(th,range=[0,thmax],bins=nbins,weights=f)
                hn_sum+=hn*w
                hf_sum+=hf*w
        
            th=0.5*(bins[1:]+bins[:-1])*180/np.pi

        igood=np.where(hn_sum>0)
        wth=np.zeros(nbins)
        wth[igood]=hf_sum[igood]/hn_sum[igood]

        if fname_out!='none' :
            np.savetxt(fname_out,np.transpose([th,wth,hf_sum,hn_sum]))

    return th,wth,hf_sum,hn_sum

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

def random_points(seed,mask,npart,fname_out='none',weights=None) :
    """Generates set of points randomly sampled on the sphere within a given mask.
    mask : healpix map containing the mask
    npart : number of points
    fname_out : path to output file
    weights : array to draw random weights from
    """

    data_dla=(fits.open(fname_dla))[1].data
    data_qso=(fits.open(fname_qso))[1].data
    pz_qso,bins=np.histogram(data_qso['Z_PIPE'],range=[-0.1,7.5],bins=50,normed=True)
    pz_par,bins=np.histogram(data_dla['zqso'  ],range=[-0.1,7.5],bins=50,normed=True)
    pz_dla,bins=np.histogram(data_dla['z_abs' ],range=[-0.1,7.5],bins=50,normed=True)
    zarr=0.5*(bins[1:]+bins[:-1])
    w_weight=np.zeros_like(pz_qso);
    igood=np.where(pz_qso>0)[0];
    w_weight[igood]=pz_par[igood]/pz_qso[igood]; w_weight[zarr>5.5]=0
    wfunc=interp1d(zarr,w_weight,bounds_error=False,fill_value=False,kind='nearest')

    fsky=np.mean(mask)
    nside=hp.npix2nside(len(mask))
    nfull=int(npart/fsky)
    np.random.seed(seed)
    rand_nums=np.random.rand(2*nfull)
    th=np.arccos(-1+2*rand_nums[::2])
    phi=2*np.pi*rand_nums[1::2]
    ipx=hp.ang2pix(nside,th,phi)
    isgood=np.where(mask[ipx]>0)[0]
    print len(isgood)
    
    b=90-180*th[isgood]/np.pi
    l=180*phi[isgood]/np.pi
    
    if weights!=None :
        w=np.random.choice(weights,size=len(isgood))
    else :
        w=np.ones_like(b)

    if fname_out!='none' :
        tbhdu=pf.new_table([pf.Column(name='B',format='D',array=b),
                            pf.Column(name='L',format='D',array=l),
                            pf.Column(name='W',format='D',array=w)])
        tbhdu.writeto(fname_out)

def bias_qso(z) :
    """QSO bias according to 1705.04718
    """
    return 1.*np.ones_like(z)

def bias_dla(z) :
    """Default DLA bias
    """
    return 1.*np.ones_like(z)

#def random_points(seed,mask,npart,fname_out='none',weights=None) :
#    """Generates set of points randomly sampled on the sphere within a given mask.
#    mask : healpix map containing the mask
#    npart : number of points
#    fname_out : path to output file
#    weights : array to draw random weights from
#    """
#    fsky=np.mean(mask)
#    nside=hp.npix2nside(len(mask))
#    nfull=int(npart/fsky)
#    np.random.seed(seed)
#    rand_nums=np.random.rand(2*nfull)
#    th=np.arccos(-1+2*rand_nums[::2])
#    phi=2*np.pi*rand_nums[1::2]
#    ipx=hp.ang2pix(nside,th,phi)
#    isgood=np.where(mask[ipx]>0)[0]
#    print len(isgood)
#    
#    b=90-180*th[isgood]/np.pi
#    l=180*phi[isgood]/np.pi
#    
#    if weights!=None :
#        w=np.random.choice(weights,size=len(isgood))
#    else :
#        w=np.ones_like(b)
#
#    if fname_out!='none' :
#        tbhdu=pf.new_table([pf.Column(name='B',format='D',array=b),
#                            pf.Column(name='L',format='D',array=l),
#                            pf.Column(name='W',format='D',array=w)])
#        tbhdu.writeto(fname_out)

'''
def reform_data(nside,use_wiener=False) :
    """Generates data in the right format
    """

    print "Generating dataset"
    if use_wiener :
        fname_mask_cmbl='data/mask_%d_wiener.fits'%nside
        fname_cmbl='data/kappa_%d_wiener.fits'%nside
    else :
        fname_mask_cmbl='data/mask_%d.fits'%nside
        fname_cmbl='data/kappa_%d.fits'%nside

    r=hp.Rotator(coord=['C','G'])

    #CMB lensing mask
    print " Generating lensing mask"
    if not os.path.isfile(fname_mask_cmbl) :
        mask=hp.ud_grade(hp.read_map(fname_mask_cmbl_orig,verbose=False),nside_out=nside)
        mask[mask<1.0]=0
        hp.write_map(fname_mask_cmbl,mask)

    #Kappa_CMB
    print " Generating kappa map"
    if not os.path.isfile(fname_cmbl) :
        almk=hp.read_alm(fname_alm_cmbl)
        if use_wiener :
            data=np.loadtxt(fname_kappa_cl);
            ll,nll,cll=np.loadtxt(fname_kappa_cl,unpack=True)
            cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
            nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
            wl=(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10)
            almk=hp.almxfl(almk,wl)
        k=hp.alm2map(almk,nside,verbose=False)
        k*=hp.read_map(fname_mask_cmbl,verbose=False)
        hp.write_map(fname_cmbl,k)

    #DLA and QSO data
    print " Generating DLA and QSO fits files"
    if not (os.path.isfile(fname_qso) and os.path.isfile(fname_dla) and os.path.isfile(fname_dla_hisn) and os.path.isfile(fname_dla_g16) and os.path.isfile(fname_mask_qso)) :
        #Read DLA data
        print "  DLA"
        data_dla16=np.genfromtxt(fname_dla_g16_orig,
                                 dtype='i8,S18,i8,i8,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                                 names=['ThingID','SDSS','Plate','MJD','Fiber','RA','DEC','zqso',
                                        'SNRSpec','b_z_DLA','B_z_DLA','logpn','logpy','logpn_y','logpy_y',
                                        'pn','py','z_abs','log_NHI'])
        is_good=np.where(data_dla16['py']>0.9)[0]
        data_dla16=data_dla16[is_good]
        th_dla16_c =(90-data_dla16['DEC'])*np.pi/180.
        phi_dla16_c=data_dla16['RA']*np.pi/180.
        th_dla16_g,phi_dla16_g=r(th_dla16_c,phi_dla16_c)
        b_dla16=90-180*th_dla16_g/np.pi
        l_dla16=180*phi_dla16_g/np.pi
        ndla16=len(th_dla16_g)        

        data_dla=np.genfromtxt(fname_dla_orig,skip_header=2,
                               dtype='i8,S16,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                               names=['ThingID','MJD-plate-fiber','RA','DEC','zqso','fDLA','fBAL',
                                      'BI','CNR','z_abs','NHI','cc','log(Pcc)','saap','Dcore','Pcore',
                                      'Fcore','Ecore','Ccore','fl','CII_1334','SiII_1526','FeII_1608',
                                      'AlII_1670','FeII_2344','FeII_2374','FeII_2382','FeII_2586',
                                      'FeII_2600','MgII_2796','MgII_2803','MgI_2852'])
        th_dla_c =(90-data_dla['DEC'])*np.pi/180.
        phi_dla_c=data_dla['RA']*np.pi/180.
        th_dla_g,phi_dla_g=r(th_dla_c,phi_dla_c)
        b_dla=90-180*th_dla_g/np.pi
        l_dla=180*phi_dla_g/np.pi
        ndla=len(th_dla_g)

        hisn=np.where(data_dla['CNR']>2)[0]
        data_dla_hisn=data_dla[hisn]
        th_dla_hisn_c =(90-data_dla_hisn['DEC'])*np.pi/180.
        phi_dla_hisn_c=data_dla_hisn['RA']*np.pi/180.
        th_dla_hisn_g,phi_dla_hisn_g=r(th_dla_hisn_c,phi_dla_hisn_c)
        b_dla_hisn=90-180*th_dla_hisn_g/np.pi
        l_dla_hisn=180*phi_dla_hisn_g/np.pi
        ndla_hisn=len(data_dla_hisn)
        
        #Read QSO data
        print "  QSO"
        data_qso=(fits.open(fname_qso_orig))[1].data
        th_qso_c =(90-data_qso['DEC'])*np.pi/180.
        phi_qso_c=data_qso['RA']*np.pi/180.
        th_qso_g,phi_qso_g=r(th_qso_c,phi_qso_c)
        b_qso=90-180*th_qso_g/np.pi
        l_qso=180*phi_qso_g/np.pi
        nqso=len(th_qso_g)

        #Create QSO mask
        print "  Mask"
        ipix_qso=hp.ang2pix(nside_qso,th_qso_g,phi_qso_g)
        mp_qso,bins=np.histogram(ipix_qso,range=[0,hp.nside2npix(nside_qso)],bins=hp.nside2npix(nside_qso))
        mask_qso=np.zeros(hp.nside2npix(nside_qso)); mask_qso[mp_qso>0]=1;

        #Compute QSO weights
        pz_qso,bins=np.histogram(data_qso['Z_PIPE'],range=[-0.1,7.5],bins=50,normed=False)
        pz_par,bins=np.histogram(data_dla['zqso'  ],range=[-0.1,7.5],bins=50,normed=False)
        pz_dla,bins=np.histogram(data_dla['z_abs' ],range=[-0.1,7.5],bins=50,normed=False)
        pz_par16,bins=np.histogram(data_dla16['zqso' ],range=[-0.1,7.5],bins=50,normed=False)
        pz_parhi,bins=np.histogram(data_dla_hisn['zqso' ],range=[-0.1,7.5],bins=50,normed=False)
        zarr=0.5*(bins[1:]+bins[:-1])

        w_weight=np.zeros_like(pz_qso);
        igood=np.where(pz_qso>0)[0];
        w_weight[igood]=pz_par[igood]/pz_qso[igood]; w_weight[zarr>5.5]=0
        wfunc=interp1d(zarr,w_weight,bounds_error=False,fill_value=False,kind='nearest')
        w_qso=wfunc(data_qso['Z_PIPE'])
        plt.plot(zarr,w_weight,'r-')


        w_weight=np.zeros_like(pz_qso);
        igood=np.where(pz_qso>0)[0];
        w_weight[igood]=pz_par16[igood]/pz_qso16[igood]; w_weight[zarr>5.5]=0
        wfunc=interp1d(zarr,w_weight,bounds_error=False,fill_value=False,kind='nearest')
        w_qso_g16=wfunc(data_qso['Z_PIPE'])
        plt.plot(zarr,w_weight,'g-')

        w_dla=np.ones(ndla)
        w_dla_hisn=np.ones(ndla_hisn)
        w_dla_g16=np.ones(ndla_g16)
        pz_wei,bins=np.histogram(data_qso['Z_PIPE'],range=[-0.1,7.5],bins=50,normed=True,weights=w_qso)

        #Write DLA data
        tbhdu=pf.new_table([pf.Column(name='ThingID',format='K',array=data_dla['ThingID']),
                            pf.Column(name='RA'     ,format='D',array=data_dla['RA']),
                            pf.Column(name='DEC'    ,format='D',array=data_dla['DEC']),
                            pf.Column(name='zqso'   ,format='D',array=data_dla['zqso']),
                            pf.Column(name='z_abs'  ,format='D',array=data_dla['z_abs']),
                            pf.Column(name='NHI'    ,format='D',array=data_dla['NHI']),
                            pf.Column(name='fDLA'   ,format='D',array=data_dla['fDLA']),
                            pf.Column(name='B'      ,format='D',array=b_dla),
                            pf.Column(name='L'      ,format='D',array=l_dla),
                            pf.Column(name='W'      ,format='D',array=w_dla)])
        tbhdu.writeto(fname_dla,clobber=True)

        #Write DLA_hisn data
        tbhdu=pf.new_table([pf.Column(name='ThingID',format='K',array=data_dla_hisn['ThingID']),
                            pf.Column(name='RA'     ,format='D',array=data_dla_hisn['RA']),
                            pf.Column(name='DEC'    ,format='D',array=data_dla_hisn['DEC']),
                            pf.Column(name='zqso'   ,format='D',array=data_dla_hisn['zqso']),
                            pf.Column(name='z_abs'  ,format='D',array=data_dla_hisn['z_abs']),
                            pf.Column(name='NHI'    ,format='D',array=data_dla_hisn['NHI']),
                            pf.Column(name='fDLA'   ,format='D',array=data_dla_hisn['fDLA']),
                            pf.Column(name='B'      ,format='D',array=b_dla_hisn),
                            pf.Column(name='L'      ,format='D',array=l_dla_hisn),
                            pf.Column(name='W'      ,format='D',array=w_dla_hisn)])
        tbhdu.writeto(fname_dla_hisn,clobber=True)

        #Write QSO data
        unihi=np.zeros_like(data_qso['UNIFORM']); unihi[:]=data_qso['UNIFORM']; unihi[data_qso['Z_PIPE']<2]=0
        tbhdu=pf.new_table([pf.Column(name='SDSS_NAME',format='18A',array=data_qso['SDSS_NAME']),
                            pf.Column(name='UNIFORM'  ,format='I',array=data_qso['UNIFORM']),
                            pf.Column(name='UNIHI'  ,format='I',array=unihi),
                            pf.Column(name='RA'       ,format='D',array=data_qso['RA']),
                            pf.Column(name='DEC'      ,format='D',array=data_qso['DEC']),
                            pf.Column(name='Z_PIPE'   ,format='D',array=data_qso['Z_PIPE']),
                            pf.Column(name='B'        ,format='D',array=b_qso),
                            pf.Column(name='L'        ,format='D',array=l_qso),
                            pf.Column(name='W'        ,format='D',array=w_qso)])
        tbhdu.writeto(fname_qso,clobber=True)

        #Write QSO mask
        hp.write_map(fname_mask_qso,mask_qso)

    return fname_cmbl,fname_mask_cmbl
'''
