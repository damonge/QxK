import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyfits as pf
import common as cmm
import sys
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

if len(sys.argv)!=2 :
    print "Usage: plotting.py which_fig"
    exit(1)

col_N12="#0180EF"
col_N12B="#0180EF"
col_G16="#EF9001"

if sys.argv[1]=='fig1a' or sys.argv[1]=='all' :
    data_QSO=(fits.open(cmm.fname_qso))[1].data
    data_DLA_N12=(fits.open(cmm.fname_dla_n12))[1].data
    data_DLA_N12B=(fits.open(cmm.fname_dla_n12b))[1].data
    data_DLA_G16=(fits.open(cmm.fname_dla_g16))[1].data
    
    plt.figure(); ax=plt.gca()
    hn12,b=np.histogram(data_DLA_N12['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dn12=hn12; zn12=(b[1:]+b[:-1])*0.5
    ax.plot(zn12,dn12,'-',color=col_N12,lw=2,label='DLAs N12')

    hn12b,b=np.histogram(data_DLA_N12B['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dn12b=hn12b; zn12b=(b[1:]+b[:-1])*0.5
    ax.plot(zn12b,dn12b,'--',color=col_N12B,lw=2,label='DLAs N12B')

    hg16,b=np.histogram(data_DLA_G16['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dg16=hg16; zg16=(b[1:]+b[:-1])*0.5
    ax.plot(zg16,dg16,'-',color=col_G16,lw=2,label='DLAs G16')

    ax.set_xlabel('$z$',fontsize=15)
    ax.set_ylabel('$N(z),\\,\\,({\\rm normalized})$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.set_xlim([1.8,5])    
    ax.set_ylim([0,1.3])
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig('doc/nz_dla.pdf',bbox_inches='tight')


if sys.argv[1]=='fig1b' or sys.argv[1]=='all' :
    data_QSO=(fits.open(cmm.fname_qso))[1].data
    data_DLA_N12=(fits.open(cmm.fname_dla_n12))[1].data
    data_DLA_N12B=(fits.open(cmm.fname_dla_n12b))[1].data
    data_DLA_G16=(fits.open(cmm.fname_dla_g16))[1].data

    plt.figure(); ax=plt.gca()
    hqso,b=np.histogram(data_QSO['Z_PIPE'],bins=60,range=[1.6,5.5],normed=True)
    dqso=hqso; zqso=(b[1:]+b[:-1])*0.5
    ax.plot(zqso,dqso,'k-',lw=2,label='All QSOs')

    hn12,b=np.histogram(data_DLA_N12['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dn12=hn12; zn12=(b[1:]+b[:-1])*0.5
    ax.plot(zn12,dn12,'-',color=col_N12,lw=2,label='QSOs w. DLAs N12')

    hn12b,b=np.histogram(data_DLA_N12B['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dn12b=hn12b; zn12b=(b[1:]+b[:-1])*0.5
    ax.plot(zn12b,dn12b,'--',color=col_N12B,lw=2,label='QSOs w. DLAs N12B')

    hg16,b=np.histogram(data_DLA_G16['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dg16=hg16; zg16=(b[1:]+b[:-1])*0.5
    ax.plot(zg16,dg16,'-',color=col_G16,lw=2,label='QSOs w. DLAs G16')

    ax.set_xlabel('$z$',fontsize=15)
    ax.set_ylabel('$N(z),\\,\\,({\\rm normalized})$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.set_xlim([1.8,5])    
    ax.set_ylim([0,1.35])
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig('doc/nz_qso.pdf',bbox_inches='tight')

if sys.argv[1]=='fig2a' or sys.argv[1]=='all' :
    d=np.load("outputs_ell2_2002_ns2048_nlb50_apo0.000/cl_qxk_all.npz")
    nsims=len(d['randoms'])
    
    igood=np.where(d['ll']<1002)[0]; nl=len(igood); nlt=len(d['ll']);
    mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
    covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
    corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])
    corr_all_n12p=np.zeros([2*nl,2*nl])
    corr_all_n12p[:nl,:nl]=corr_all_n12[:nlt,:nlt][igood,:][:,igood]
    corr_all_n12p[:nl,nl:]=corr_all_n12[:nlt,nlt:][igood,:][:,igood]
    corr_all_n12p[nl:,:nl]=corr_all_n12[nlt:,:nlt][igood,:][:,igood]
    corr_all_n12p[nl:,nl:]=corr_all_n12[nlt:,nlt:][igood,:][:,igood]

    plt.figure(); ax=plt.gca()
    ax.set_title("$C_\ell\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_all_n12p,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.text(0.05,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes,color='w')
    ax.text(0.05,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='w')
    ax.text(0.55,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes,color='w')
    ax.text(0.55,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='w')
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_cls.pdf',bbox_inches='tight')


if sys.argv[1]=='fig2b' or sys.argv[1]=='all' :
    d=np.load("outputs_thm3.0_ns2048_nb16_wiener/wth_qxk_all.npz")
    nsims=len(d['randoms'])
    
    mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
    covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
    corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])

    plt.figure(); ax=plt.gca()
    ax.set_title("$\\xi(\\theta)\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_all_n12,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.text(0.27,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes,color='k')
    ax.text(0.27,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='k')
    ax.text(0.77,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes,color='k')
    ax.text(0.77,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='k')
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_wth.pdf',bbox_inches='tight')
    
    

plt.show()
