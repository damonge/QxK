import numpy as np
import healpy
import scipy
import os
import pickle
import sys
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pymaster as nmt
import pyccl as ccl

# read command line arguments
parser = argparse.ArgumentParser(description='Calculate bias of QSOs and DLAs.')
parser.add_argument('-M', type=int, help='maximum ell value, default is 1000')
parser.add_argument('-m', help='minimum bin value')
parser.add_argument('-b', type=int, help='bandpower width (25, 50, or 75); default is 50')
parser.add_argument('--sim', help='use random simulation (instead of data)',action="store_true")
parser.add_argument('--nsim', help='simulation number (default is random)')
parser.add_argument('--plot', help='plot resulting 2d gaussian',action="store_true")
parser.add_argument('-A', help='use analytical errors instead of simulated error',action="store_true")

args = parser.parse_args()

if args.M == None:
    max_ell = 1000
else:
    max_ell = args.M

if args.b == None:
    bsize = 50
else:
    bsize = args.b
    
#import precalculated power spectra


theory_dla = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/theory/theory_dla_'+str(bsize)+'.pkl','rb'))
theory_qso = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/theory/theory_qso_'+str(bsize)+'.pkl','rb'))

nsim = int(len(os.listdir('/Users/Joseph/Brookhaven/Research/pickled/sim_data'+str(bsize)+'/'))/2)

b = nmt.NmtBin(2048,nlb= bsize)
ell_arr=b.get_effective_ells()

min_ell_bin = 0
max_ell_bin = (np.abs(ell_arr-max_ell)).argmin()
ells = ell_arr[min_ell_bin:max_ell_bin]

if args.sim:
    if args.nsim == None:
        N = np.random.randint(nsim)
    else:
        N = args.nsim
    print('simulation number:',N)
    data_qso = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/sim_data'+str(bsize)+'/cps_qso_'+str(N).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]
    data_dla = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/sim_data'+str(bsize)+'/cps_dla_'+str(N).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]   
else:    
    data_qso = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps/cps_qso_'+str(bsize)+'.pkl','rb'))[0]
    data_dla = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps/cps_dla_'+str(bsize)+'.pkl','rb'))[0]

#Calculate error of Cls using simulations    
sim_qso = np.zeros((len(ell_arr),nsim))
sim_dla = np.zeros((len(ell_arr),nsim))
for i in range(nsim):
    sim_qso[:,i] = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/sim_data'+str(bsize)+'/cps_qso_'+str(i).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0]
    sim_dla[:,i] = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/sim_data'+str(bsize)+'/cps_dla_'+str(i).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0]

#error_qso = np.std(sim_qso,axis=1)
#error_dla = np.std(sim_dla,axis=1)

#calculate full inverse covariance matrix for all ell bins

if args.A: 
    #Approximation using errors calculated analytically (still uses l+1 correlation from sims)
    #load masks and calculate fraction of the sky covered by them
    qso_mask = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/masks/mask_qso_apo.pkl','rb'))
    dla_mask = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/masks/mask_dla_apo.pkl','rb'))
    f_qso = sum(qso_mask)/len(qso_mask)
    f_dla = sum(dla_mask)/len(dla_mask)
    #load power spectra
    cps_qso = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps/cps_qso_'+str(bsize)+'.pkl','rb'))[0]
    cps_dla = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps/cps_dla_'+str(bsize)+'.pkl','rb'))[0]
    ps_dd = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/power_spectra/ps_dd_'+str(bsize)+'.pkl','rb'))[0]
    ps_qq = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/power_spectra/ps_qq_'+str(bsize)+'.pkl','rb'))[0]
    ps_kk = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/power_spectra/ps_kk_'+str(bsize)+'.pkl','rb'))[0]
    cps_dq = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps/cps_dla_qso_'+str(bsize)+'.pkl','rb'))[0]
    #Calculate Error
    average_qsoXqso = (np.array(abs(ps_kk*ps_qq+cps_qso**2)/(bsize*(2*ell_arr+1)*f_qso)))
    average_dlaXdla = (np.array(abs(ps_kk*ps_dd+cps_dla**2)/(bsize*(2*ell_arr+1)*f_dla)))
    average_qsoXdla = (np.array(abs(ps_kk*cps_dq+cps_qso*cps_dla)/(bsize*(2*ell_arr+1)*f_dla)))
    diagP2 = np.zeros(2*len(ell_arr)-2)

else:
    average_qsoXqso = np.average(sim_qso**2,axis=1)
    average_dlaXdla = np.average(sim_dla**2,axis=1)
    average_qsoXdla = abs(np.average((sim_qso)*(sim_dla),axis=1))
    qsoP0 = sim_qso[:len(sim_qso)-1,:]
    qsoP1 = sim_qso[1:,:]
    dlaP0 = sim_dla[:len(sim_qso)-1,:]
    dlaP1 = sim_dla[1:,:]
    average_qsoXqsoP1 = np.average(qsoP0*qsoP1,axis=1)
    average_dlaXdlaP1 = np.average(dlaP0*dlaP1,axis=1)
    diagP2 = np.zeros(2*len(ell_arr)-2)
    diagP2[::2] = average_qsoXqsoP1
    diagP2[1::2] = average_dlaXdlaP1

diag = np.zeros(2*len(ell_arr))
diag[::2] = average_qsoXqso
diag[1::2] = average_dlaXdla

diagP1 = np.zeros(2*len(ell_arr)-1)
diagP1[::2] = average_qsoXdla

Cov = np.zeros((2*len(ell_arr),2*len(ell_arr)))
np.fill_diagonal(Cov,diag)

CovP1 = np.zeros((2*len(ell_arr)-1,2*len(ell_arr)-1))
np.fill_diagonal(CovP1,diagP1)
Cov[1:,:len(Cov)-1] += CovP1
Cov[:len(Cov)-1,1:] += CovP1

CovP2 = np.zeros((2*len(ell_arr)-2,2*len(ell_arr)-2))
np.fill_diagonal(CovP2,diagP2)
Cov[2:,:len(Cov)-2] += CovP2
Cov[:len(Cov)-2,2:] += CovP2

iCov = scipy.linalg.inv(Cov)
    
if args.m == None:
    min_ell_bin = 0
else:
    min_ell_bin = int(args.m)
    
max_ell_bin = (np.abs(ell_arr-max_ell)).argmin()
ells = ell_arr[min_ell_bin:max_ell_bin]

iCovariance = iCov[2*min_ell_bin:2*max_ell_bin,2*min_ell_bin:2*max_ell_bin]

theory_q = np.zeros(2*len(ells))
theory_q[::2] = theory_qso[min_ell_bin:max_ell_bin]
theory_q[1::2] = theory_qso[min_ell_bin:max_ell_bin]

theory_d = np.zeros(2*len(ells))
theory_d[1::2] = theory_dla[min_ell_bin:max_ell_bin]

data = np.zeros(2*len(ells))
data[::2] = data_qso[min_ell_bin:max_ell_bin]
data[1::2] = data_dla[min_ell_bin:max_ell_bin]                

def loglike(x):
    delta = x[0]*theory_q + x[1]*theory_d - data
    chi2 = np.dot(np.transpose(delta),np.dot(iCovariance,delta))
    return -chi2/2.0

def negloglike(x):
    return -loglike(x)

[b_qso, b_dla] = scipy.optimize.minimize(negloglike,np.array([0,0])).x
dof = 2*len(ells)
chi2 = -2*loglike([b_qso, b_dla])
prob = scipy.stats.chi2.cdf(chi2,dof)
    

iCovB = np.zeros((2,2))
iCovB[0,0] = np.dot(theory_q,np.dot(iCovariance,theory_q))
iCovB[1,0] = np.dot(theory_q,np.dot(iCovariance,theory_d))
iCovB[0,1] = np.dot(theory_q,np.dot(iCovariance,theory_d))
iCovB[1,1] = np.dot(theory_d,np.dot(iCovariance,theory_d))

CovB = scipy.linalg.inv(iCovB)
qso_error = np.sqrt(CovB[0,0])
dla_error = np.sqrt(CovB[1,1])

        

print('\n','b_qso:',str(np.around(b_qso,2))+' ± '+str(np.around(qso_error,2)),'\n','b_dla:',str(np.around(b_dla,2))+' ± '+str(np.around(dla_error,2)),'\n','dof:  ',dof,'\n','chi2: ',np.around(chi2,2),'\n','prob: ',np.around(prob,2),'\n')

if args.plot:
    b_y = np.arange(np.round(b_qso)-3,np.round(b_qso)+3,0.01)
    b_x = np.arange(np.round(b_dla)-3,np.round(b_dla)+3,0.01)
    l = len(b_x)
    B = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            B[i,j] = np.exp(loglike([b_y[i],b_x[j]]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(B,interpolation='nearest')  
    ax.set_yticklabels(np.append('',np.arange(np.round(b_qso)-3,np.round(b_qso)+3)))
    ax.set_xticklabels(np.append('',np.arange(np.round(b_dla)-3,np.round(b_dla)+3)))
    plt.colorbar()
    plt.xlabel('b_dla')
    plt.ylabel('b_qso')
    plt.show()
    plt.close()
    
"""if args.plot:
    
    def gauss(x):
        b = np.matrix([b_qso,b_dla])
        logl = np.dot(np.square(b-x),np.dot(iCovB,np.square(b.transpose()-x.transpose())))
        return np.exp(-logl)

    b_x = np.arange(np.round(b_qso*2)/2-3,np.round(b_qso*2)/2+3,0.01)
    b_y = np.arange(np.round(b_dla*2)/2-3,np.round(b_dla*2)/2+3,0.01)
    l = len(b_x)
    B = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            x = np.matrix([b_x[i],b_y[j]])
            B[i,j] = gauss(x)
    
    B = np.flipud(B)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(B,interpolation='nearest')  
    ax.set_yticklabels(np.append('',np.arange(np.round(b_qso*2)/2+3,np.round(b_qso*2)/2-3,-1)))
    ax.set_xticklabels(np.append('',np.arange(np.round(b_dla*2)/2-3,np.round(b_dla*2)/2+3,1)))
    plt.colorbar()
    plt.xlabel('b_dla')
    plt.ylabel('b_qso')
    plt.show()
    plt.close()"""